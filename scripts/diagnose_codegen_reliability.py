"""Fast diagnostic tool for CAD-code-generation reliability, NOT a full statistical harness.

Runs all 13 held-out seeds' RAW prompts (never the refined ones — this measures the
CAD-code-writing model's own reliability, independent of Stage 2 entirely) through a single
CAD-generation call each, scores via evaluate_leg1 against the seed bank's hand-authored ground
truth, and reports a stage-by-stage breakdown. Built to A/B one variable at a time (model,
prompt rules, or a repair/precheck wrapper) across repeated runs — see CLI args below — while
iterating on fixes for the failure modes diagnosed in docs/TRAINED_OPTIMIZERS.md's Model 1
end-to-end result (11 of 12 held-out pairs tied at failure, mostly hallucinated CadQuery API
calls, not genuine geometry/dimension problems).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cadyfiner.generators.local_ollama import generate as ollama_generate
from cadyfiner.oracle.checks import evaluate_leg1
from cadyfiner.oracle.execute import CADQUERY_PROMPT_RULES, extract_code, run_cadquery
from cadyfiner.spec import DesignBrief

FAMILIES_DIR = Path(__file__).resolve().parents[1] / "prompts" / "seed_bank" / "families"
MANIFEST_PATH = Path(__file__).resolve().parents[1] / "prompts" / "seed_bank" / "manifest.json"


def load_heldout_seeds() -> list[dict]:
    manifest = json.loads(MANIFEST_PATH.read_text())
    all_items: dict[str, dict] = {}
    for path in FAMILIES_DIR.glob("*.json"):
        for item in json.loads(path.read_text()):
            all_items[item["id"]] = item
    seeds = []
    for role in ("heldout_same_family", "heldout_family"):
        for seed_id in manifest["split"][role]["seed_ids"]:
            seeds.append(all_items[seed_id])
    return seeds


def main() -> None:
    model = sys.argv[1] if len(sys.argv) > 1 else "gemma4:e4b"
    base_url = sys.argv[2] if len(sys.argv) > 2 else "http://localhost:11434"
    label = sys.argv[3] if len(sys.argv) > 3 else model
    # Optional 4th arg: a Python module path exposing PROMPT_RULES (str) and/or a
    # generate_and_execute(prompt, out_dir, generate_kwargs) -> ExecutionResult override, for
    # testing a candidate fix without editing this script per candidate.
    override_module = sys.argv[4] if len(sys.argv) > 4 else None

    prompt_rules = CADQUERY_PROMPT_RULES
    custom_call = None
    if override_module:
        import importlib

        mod = importlib.import_module(override_module)
        prompt_rules = getattr(mod, "PROMPT_RULES", CADQUERY_PROMPT_RULES)
        custom_call = getattr(mod, "generate_and_execute", None)

    seeds = load_heldout_seeds()
    out_root = Path("workspace/diagnose") / label.replace(":", "_").replace("/", "_")
    stage_counts: dict[str, int] = {}
    detail_lines = []

    for i, seed in enumerate(seeds):
        raw_prompt = seed["raw_prompt"]
        gt = DesignBrief.model_validate(seed["ground_truth"])
        out_dir = out_root / f"s{i}_{seed['id']}"
        kwargs = {"model": model, "base_url": base_url, "temperature": 0.5, "max_tokens": 2500, "timeout": 240}

        try:
            if custom_call is not None:
                result = custom_call(prompt_rules + f"\nDesign request:\n{raw_prompt}\n", out_dir, kwargs)
            else:
                code = extract_code(ollama_generate(prompt_rules + f"\nDesign request:\n{raw_prompt}\n", **kwargs))
                result = run_cadquery(code, out_dir, timeout_s=90)
        except Exception as exc:  # noqa: BLE001 -- an infra hiccup (Ollama timeout/connection
            # drop) on one seed must not kill the whole comparison run; every other seed's
            # result is still valid signal. Recorded as its own stage so it's never silently
            # conflated with a real generation failure in the stage-breakdown counts.
            print(f"[{i+1}/{len(seeds)}] {seed['id']}: INFRA_ERROR {type(exc).__name__}: {exc}", flush=True)
            stage_counts["infra_error"] = stage_counts.get("infra_error", 0) + 1
            detail_lines.append({"seed_id": seed["id"], "stopped_at": "infra_error", "overall_pass": False,
                                  "detail": f"{type(exc).__name__}: {exc}"})
            continue

        leg1 = evaluate_leg1(result, gt)
        stage_counts[leg1.stopped_at] = stage_counts.get(leg1.stopped_at, 0) + 1
        line = f"[{i+1}/{len(seeds)}] {seed['id']}: stopped_at={leg1.stopped_at} pass={leg1.overall_pass}"
        print(line, flush=True)
        fail_stage = next((s for s in leg1.stages if not s.passed), None)
        if fail_stage:
            print(f"    {fail_stage.name}: {fail_stage.detail[:200]}", flush=True)
        detail_lines.append({"seed_id": seed["id"], "stopped_at": leg1.stopped_at, "overall_pass": leg1.overall_pass,
                              "detail": (fail_stage.detail if fail_stage else "full pass")})

    print(f"\n=== {label}: stage breakdown over {len(seeds)} seeds ===")
    for stage in ("infra_error", "execute", "mesh_validity", "spec_conformance", "manufacturability"):
        print(f"  stopped_at={stage}: {stage_counts.get(stage, 0)}")
    n_pass = sum(1 for d in detail_lines if d["overall_pass"])
    print(f"  full pass: {n_pass}/{len(seeds)}")

    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "summary.json").write_text(json.dumps({"label": label, "model": model, "stage_counts": stage_counts,
                                                          "n_pass": n_pass, "n_total": len(seeds), "details": detail_lines}, indent=2))
    print(f"\nWrote {out_root / 'summary.json'}")


if __name__ == "__main__":
    main()
