"""Run the mechanical-domain pilot: raw (unrefined) prompts through the local
generator, scored against hand-authored ground truth.

Purpose (see the plan's phase-2 checkpoint): determine whether the
wall_planter-derived "bounded specificity depth" policy transfers to
mechanical/functional parts before committing to a per-object-class policy
in the refiner. No refiner is involved here — these are the raw seed
prompts as authored, at three specificity levels per family.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cadyfiner.generators.local_ollama import generate, OllamaGenerationError
from cadyfiner.oracle.execute import CADQUERY_PROMPT_RULES, extract_code, run_cadquery
from cadyfiner.oracle.checks import evaluate_leg1
from cadyfiner.spec import DesignBrief

SEEDS_PATH = Path(__file__).resolve().parents[1] / "prompts" / "seed_bank" / "pilot_mechanical" / "seeds.json"
OUT_DIR = Path(__file__).resolve().parents[1] / "workspace" / "pilot_mechanical"
RESULTS_PATH = OUT_DIR / "results.jsonl"

MODEL = sys.argv[1] if len(sys.argv) > 1 else "dolphincoder:7b"
REPS = int(sys.argv[2]) if len(sys.argv) > 2 else 1


def main() -> None:
    seeds = json.loads(SEEDS_PATH.read_text())
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    for seed in seeds:
        ground_truth = DesignBrief.model_validate(seed["ground_truth"])
        for rep in range(REPS):
            print(f"[{seed['id']} rep{rep}] generating with {MODEL}...", flush=True)
            prompt = CADQUERY_PROMPT_RULES + f"\nDesign request:\n{seed['raw_prompt']}\n"
            t0 = time.time()
            try:
                raw_output = generate(prompt, model=MODEL, temperature=0.5, max_tokens=1500, timeout=180)
            except OllamaGenerationError as exc:
                print(f"  generation failed: {exc}")
                results.append({
                    "id": seed["id"], "family": seed["family"], "level": seed["level"], "rep": rep,
                    "model": MODEL, "gen_error": str(exc), "leg1": None,
                })
                continue
            gen_time = time.time() - t0
            code = extract_code(raw_output)

            run_out_dir = OUT_DIR / "artifacts" / f"{seed['id']}_rep{rep}"
            execution = run_cadquery(code, run_out_dir, timeout_s=60)
            leg1 = evaluate_leg1(execution, ground_truth)

            print(f"  gen={gen_time:.0f}s  exec_ok={execution.ok}  overall_pass={leg1.overall_pass}  stopped_at={leg1.stopped_at}")

            (run_out_dir).mkdir(parents=True, exist_ok=True)
            (run_out_dir / "code.py").write_text(code)
            (run_out_dir / "raw_output.txt").write_text(raw_output)

            results.append({
                "id": seed["id"],
                "family": seed["family"],
                "level": seed["level"],
                "rep": rep,
                "model": MODEL,
                "gen_time_s": gen_time,
                "execution_ok": execution.ok,
                "execution_error_type": execution.error_type,
                "leg1_overall_pass": leg1.overall_pass,
                "leg1_stopped_at": leg1.stopped_at,
                "leg1_stages": [{"name": s.name, "passed": s.passed, "detail": s.detail} for s in leg1.stages],
                "measured": leg1.measured,
            })
            RESULTS_PATH.write_text("\n".join(json.dumps(r) for r in results) + "\n")

    print(f"\nWrote {len(results)} results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
