"""Build oracle-filtered supervised fine-tuning data for the specialized Stage-2 model.

Methodology (see docs/TRAINED_OPTIMIZERS.md for the full write-up):
for each synthetic raw prompt, run the CURRENT teacher pipeline (Stage 1
extract -> Stage 2 fill_gaps via a general local LLM) to get a candidate
refined prompt + spec. Generate CadQuery from BOTH the raw prompt and the
refined prompt, using the SAME downstream generator. Score both against
the REFINED spec's own stated target_dims/features (holding the target
constant isolates whether refining the PROMPT text helps the generator hit
that target, the same logic the seed-bank harness uses with a real ground
truth — here the "ground truth" is teacher-synthesized, not independently
authored, which is the honest limitation of doing this at training-data
scale rather than by hand).

Keep the (input, output) pair as a training example ONLY if the refined
arm reaches at least as far through the Leg-1 cascade as the raw arm
(stage ordinality: execute < mesh_validity < spec_conformance <
manufacturability < pass) — this is rejection sampling against the
objective oracle already built for this project, not blind imitation of
whatever the teacher model said. A case where refining made things
measurably worse is discarded, not learned from.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cadyfiner.generators.local_ollama import generate, OllamaGenerationError
from cadyfiner.oracle.checks import evaluate_leg1
from cadyfiner.oracle.execute import CADQUERY_PROMPT_RULES, extract_code, run_cadquery
from cadyfiner.refine import extract
from cadyfiner.refine_stage2 import _build_prompt, fill_gaps
from cadyfiner.spec import DesignBrief

TEACHER_MODEL = sys.argv[1] if len(sys.argv) > 1 else "gemma4:e4b"
PROMPTS_PATH = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("workspace/synthetic_prompts.json")
OUT_PATH = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("workspace/distillation_data.jsonl")
MAX_PROMPTS = int(sys.argv[4]) if len(sys.argv) > 4 else 999

_STAGE_ORDER = ["execute", "mesh_validity", "spec_conformance", "manufacturability"]


def _stage_rank(leg1_result) -> int:
    """How far into the cascade this result got, plus 1 if it fully passed."""

    rank = _STAGE_ORDER.index(leg1_result.stopped_at) if leg1_result.stopped_at in _STAGE_ORDER else 0
    return rank + (1 if leg1_result.overall_pass else 0)


def main() -> None:
    prompts = json.loads(PROMPTS_PATH.read_text())["prompts"][:MAX_PROMPTS]
    kwargs = {"model": TEACHER_MODEL, "temperature": 0.5, "max_tokens": 2000, "timeout": 90, "max_retries": 1}

    accepted, rejected, errors = 0, 0, 0
    out_f = open(OUT_PATH, "a")
    log_path = OUT_PATH.with_suffix(".log.jsonl")
    log_f = open(log_path, "a")

    for i, item in enumerate(prompts):
        raw_prompt = item["raw_prompt"]
        print(f"[{i+1}/{len(prompts)}] {raw_prompt[:60]!r}", flush=True)
        t0 = time.time()
        try:
            extraction = extract(raw_prompt)
            stage2_input_prompt = _build_prompt(extraction)  # exact training INPUT text
            filled = fill_gaps(extraction, generate, **kwargs)
            if not filled.parse_ok:
                rejected += 1
                log_f.write(json.dumps({"raw_prompt": raw_prompt, "outcome": "stage2_parse_failed"}) + "\n")
                continue

            raw_out_dir = Path("workspace/distill_artifacts") / f"p{i}_raw"
            refined_out_dir = Path("workspace/distill_artifacts") / f"p{i}_refined"

            raw_code = extract_code(generate(CADQUERY_PROMPT_RULES + f"\nDesign request:\n{raw_prompt}\n", **kwargs))
            raw_exec = run_cadquery(raw_code, raw_out_dir, timeout_s=60)

            refined_text = filled.spec.refined_prompt or raw_prompt
            refined_code = extract_code(generate(CADQUERY_PROMPT_RULES + f"\nDesign request:\n{refined_text}\n", **kwargs))
            refined_exec = run_cadquery(refined_code, refined_out_dir, timeout_s=60)

            # Score both arms against the teacher's own synthesized target (see module
            # docstring for why this is a self-consistency signal, not independent ground truth).
            target_spec = DesignBrief(
                prompt=raw_prompt, target_dims=filled.spec.target_dims, features=filled.spec.features,
            )
            raw_result = evaluate_leg1(raw_exec, target_spec)
            refined_result = evaluate_leg1(refined_exec, target_spec)

            raw_rank, refined_rank = _stage_rank(raw_result), _stage_rank(refined_result)
            elapsed = time.time() - t0

            if refined_rank >= raw_rank:
                accepted += 1
                training_output = {
                    "target_dims": filled.spec.target_dims.model_dump(exclude_none=True),
                    "features": [f.model_dump(exclude_none=True) for f in filled.spec.features],
                    "process_notes": filled.spec.process_notes,
                    "style_notes": filled.spec.style_notes,
                    "assumptions_made": filled.spec.assumptions_made,
                    "refined_prompt": filled.spec.refined_prompt,
                }
                out_f.write(json.dumps({"input": stage2_input_prompt, "output": training_output}) + "\n")
                out_f.flush()
            else:
                rejected += 1

            log_f.write(json.dumps({
                "raw_prompt": raw_prompt, "raw_rank": raw_rank, "refined_rank": refined_rank,
                "accepted": refined_rank >= raw_rank, "elapsed_s": elapsed,
            }) + "\n")
            log_f.flush()

        except (OllamaGenerationError, Exception) as exc:  # noqa: BLE001 — keep the pipeline moving past any single failure
            errors += 1
            log_f.write(json.dumps({"raw_prompt": raw_prompt, "outcome": "error", "error": f"{type(exc).__name__}: {exc}"}) + "\n")
            log_f.flush()
            continue

        print(f"  accepted={accepted} rejected={rejected} errors={errors}", flush=True)

    out_f.close()
    log_f.close()
    print(f"\nDone. accepted={accepted} rejected={rejected} errors={errors} -> {OUT_PATH}")


if __name__ == "__main__":
    main()
