"""Re-run the exit-criteria harness with the specialized trained Model 1 as the
Stage-2 backend, instead of a general-purpose model — the real end-to-end
comparison docs/TRAINED_OPTIMIZERS.md section 5 calls for. CAD-code generation
itself still uses a general-purpose model (local_trained was never trained to
write CadQuery, only to fill Stage 2's JSON spec — see cli.py's docstring on
why those two roles are kept separate).
"""

from __future__ import annotations

import json
import sys
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cadyfiner.generators.local_ollama import generate as ollama_generate
from cadyfiner.generators.local_trained import generate as trained_generate
from cadyfiner.harness import run_paired_evaluation, summarize, SeedCase
from cadyfiner.spec import DesignBrief
import cadyfiner.refine_stage2 as stage2_module

FAMILIES_DIR = Path(__file__).resolve().parents[1] / "prompts" / "seed_bank" / "families"
MANIFEST_PATH = Path(__file__).resolve().parents[1] / "prompts" / "seed_bank" / "manifest.json"

CADQUERY_MODEL = sys.argv[1] if len(sys.argv) > 1 else "gemma4:e4b"
STAGE2_BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
STAGE2_ADAPTER_PATH = str(Path(__file__).resolve().parents[1] / "training" / "adapters" / "stage2")
ROLES = sys.argv[2].split(",") if len(sys.argv) > 2 else ["heldout_same_family", "heldout_family"]


def load_seeds() -> list[SeedCase]:
    manifest = json.loads(MANIFEST_PATH.read_text())
    all_items = {}
    for path in FAMILIES_DIR.glob("*.json"):
        for item in json.loads(path.read_text()):
            all_items[item["id"]] = item
    seeds = []
    for role, block in manifest["split"].items():
        if role not in ROLES:
            continue
        for seed_id in block["seed_ids"]:
            item = all_items[seed_id]
            seeds.append(SeedCase(
                id=item["id"], family=item["family"], raw_prompt=item["raw_prompt"],
                ground_truth=DesignBrief.model_validate(item["ground_truth"]), role=role,
            ))
    return seeds


def main() -> None:
    seeds = load_seeds()
    print(f"Loaded {len(seeds)} seeds. CAD-gen model={CADQUERY_MODEL}, Stage-2 model=local_trained")

    # run_paired_evaluation's fill_gaps() call uses whatever `generate` it's given for
    # Stage 2 — but harness.py's _run_one() always uses that SAME callable for CAD-code
    # generation too. Since local_trained can't write CadQuery, we can't pass it as THE
    # generator; instead monkeypatch fill_gaps's underlying call site is not an option
    # either (harness.py calls fill_gaps(extraction, generate, ...) with the one generate
    # it was given). Simplest correct fix: bind a small dispatcher that routes to the
    # trained model only when called with the Stage-2 prompt shape, and to the general
    # model otherwise. Stage-2 prompts are recognizable: they always start with
    # refine_stage2._SYSTEM_PROMPT's fixed preamble.
    stage2_marker = stage2_module._SYSTEM_PROMPT.split("\n")[0]

    def dispatch(prompt: str, **kwargs):
        if prompt.startswith(stage2_marker):
            return trained_generate(prompt, base_model=STAGE2_BASE_MODEL, adapter_path=STAGE2_ADAPTER_PATH)
        return ollama_generate(prompt, model=CADQUERY_MODEL, **{k: v for k, v in kwargs.items() if k != "model"})

    results = run_paired_evaluation(
        seeds, dispatch, reps=1,
        out_root=Path("workspace/harness_trained") / CADQUERY_MODEL.replace(":", "_"),
        generate_kwargs={"temperature": 0.5, "max_tokens": 2500, "timeout": 240},
    )
    for r in results:
        outcome = "REFINED WINS" if r.refined_pass and not r.raw_pass else \
                  "RAW WINS" if r.raw_pass and not r.refined_pass else "TIE"
        print(f"  {r.seed_id:<32} raw={r.raw_pass!s:6} refined={r.refined_pass!s:6}  {outcome}  fallback={r.used_refined_fallback}")

    report = summarize(results)
    print(f"\n=== TRAINED MODEL 1 EXIT CRITERIA REPORT ===")
    print(f"n_pairs={report.n_pairs}  n_distinct_seeds={report.n_distinct_seeds}  excluded_fallback={report.n_excluded_fallback}")
    print(f"refined_wins={report.refined_wins}  raw_wins={report.raw_wins}  ties={report.ties}")
    print(f"win_rate={report.win_rate_excluding_ties:.1%}  ci95={report.win_rate_ci95}  p={report.sign_test_p:.4f}")
    print(f"\nVERDICT: {report.verdict}")

    out_path = Path("workspace/harness_trained") / CADQUERY_MODEL.replace(":", "_") / "report.json"
    out_path.write_text(json.dumps({
        "cadquery_model": CADQUERY_MODEL, "stage2_model": "local_trained",
        "n_pairs": report.n_pairs, "n_distinct_seeds": report.n_distinct_seeds,
        "n_excluded_fallback": report.n_excluded_fallback,
        "refined_wins": report.refined_wins, "raw_wins": report.raw_wins, "ties": report.ties,
        "win_rate": report.win_rate_excluding_ties, "ci95": list(report.win_rate_ci95),
        "sign_test_p": report.sign_test_p, "by_family": report.by_family, "by_role": report.by_role,
        "verdict": report.verdict,
    }, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
