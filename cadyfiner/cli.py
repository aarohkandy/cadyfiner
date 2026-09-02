"""cadyfiner CLI: the drop-in usable interface.

    cadyfiner refine "make me a wall planter"
    cadyfiner generate "make me a wall planter" --out model.stl
    cadyfiner check model.stl --spec spec.json

Local Ollama by default (``--backend local``, default model
``gemma4:e4b`` — chosen over qwen2.5-coder and dolphincoder this session
after live comparison found it noticeably more reliable at both CadQuery
codegen and the structured-JSON gap-filling task; see README for the
comparison). ``--backend frontier`` uses Claude and requires
``ANTHROPIC_API_KEY``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from cadyfiner.oracle.checks import evaluate_leg1
from cadyfiner.oracle.execute import CADQUERY_PROMPT_RULES, extract_code, run_cadquery
from cadyfiner.refine import extract
from cadyfiner.refine_stage2 import fill_gaps
from cadyfiner.spec import DesignBrief


# Where the specialized LoRA adapters land once trained — see training/train_lora.py
# and docs/TRAINED_OPTIMIZERS.md. Base models are matched to task complexity: Stage 2
# needs open-ended generation competence, the (separate, optimize.py-only) policy
# mutation-proposer's output space is a handful of fixed tags.
_TRAINED_STAGE2_CONFIG = {"base_model": "Qwen/Qwen2.5-1.5B-Instruct", "adapter_path": "training/adapters/stage2"}


def _get_generator(backend: str):
    """The CAD-code-writing generator — always general-purpose. The specialized Stage-2
    model (``local_trained``) is intentionally NOT selectable here: it was trained only to
    produce Stage 2's structured JSON, not CadQuery source, and would not write usable code."""

    if backend == "local":
        from cadyfiner.generators.local_ollama import generate
        return generate
    if backend == "frontier":
        from cadyfiner.generators.frontier import generate
        return generate
    raise ValueError(f"unknown backend: {backend!r} (expected 'local' or 'frontier')")


def _get_stage2_generator(backend: str):
    """The gap-filling generator for Stage 2 specifically — this one DOES accept
    ``local_trained``, since that's exactly the task it was fine-tuned for."""

    if backend == "local_trained":
        from cadyfiner.generators.local_trained import generate
        return generate
    return _get_generator(backend)


def _add_backend_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--backend", choices=["local", "frontier"], default="local", help="Generator for CadQuery code.")
    p.add_argument(
        "--stage2-backend", choices=["local", "frontier", "local_trained"], default=None,
        help="Generator for the prompt-refining Stage 2 step (default: same as --backend). "
             "'local_trained' uses the specialized fine-tuned model instead of a general-purpose one.",
    )
    p.add_argument("--model", default=None, help="Override the default model for the chosen backend.")
    p.add_argument("--temperature", type=float, default=0.5)
    p.add_argument("--max-tokens", type=int, default=2500)
    p.add_argument("--timeout", type=float, default=240.0)


def _generate_kwargs(args, backend: str | None = None) -> dict:
    backend = backend or args.backend
    if backend == "local_trained":
        return {**_TRAINED_STAGE2_CONFIG, "temperature": args.temperature, "max_tokens": args.max_tokens}
    kwargs = {"temperature": args.temperature, "max_tokens": args.max_tokens, "timeout": args.timeout}
    if args.model:
        kwargs["model"] = args.model
    elif backend == "local":
        kwargs["model"] = "gemma4:e4b"
    return kwargs


def cmd_refine(args) -> None:
    stage2_backend = args.stage2_backend or args.backend
    generate = _get_stage2_generator(stage2_backend)
    extraction = extract(args.prompt)
    filled = fill_gaps(extraction, generate, **_generate_kwargs(args, stage2_backend))

    if args.json:
        print(json.dumps(filled.spec.model_dump(exclude_none=True), indent=2))
        return

    print(f"Object class: {filled.spec.object_class}")
    print(f"\nRefined prompt:\n{filled.spec.refined_prompt}\n")
    if filled.spec.assumptions_made:
        print("Assumptions made (not stated by you):")
        for a in filled.spec.assumptions_made:
            print(f"  - {a}")
    if not filled.parse_ok:
        print("\nWARNING: the LLM's gap-filling response could not be parsed; falling back to Stage 1 extraction only.", file=sys.stderr)


def cmd_generate(args) -> None:
    generate = _get_generator(args.backend)
    prompt_text = args.prompt
    if args.refine:
        stage2_backend = args.stage2_backend or args.backend
        stage2_generate = _get_stage2_generator(stage2_backend)
        extraction = extract(args.prompt)
        filled = fill_gaps(extraction, stage2_generate, **_generate_kwargs(args, stage2_backend))
        prompt_text = filled.spec.refined_prompt or args.prompt
        print(f"Refined prompt used for generation:\n{prompt_text}\n", file=sys.stderr)

    full_prompt = CADQUERY_PROMPT_RULES + f"\nDesign request:\n{prompt_text}\n"
    raw_output = generate(full_prompt, **_generate_kwargs(args))
    code = extract_code(raw_output)

    out_dir = Path(args.out).parent if args.out else Path("workspace/cli_out")
    execution = run_cadquery(code, out_dir, timeout_s=args.timeout)
    if not execution.ok:
        print(f"Generation failed: {execution.error_type}: {execution.error_message}", file=sys.stderr)
        sys.exit(1)

    dest = Path(args.out) if args.out else Path(execution.stl_path)
    if args.out and execution.stl_path != str(dest):
        dest.parent.mkdir(parents=True, exist_ok=True)
        Path(execution.stl_path).rename(dest)
    print(f"Wrote {dest}")

    if args.spec:
        spec = DesignBrief.model_validate(json.loads(Path(args.spec).read_text()))
        result = evaluate_leg1(execution, spec)
        print(f"\nQuality check: {'PASS' if result.overall_pass else 'FAIL'} (stopped at {result.stopped_at})")
        print(result.feedback_text())


def cmd_check(args) -> None:
    from cadyfiner.oracle.execute import ExecutionResult

    spec = DesignBrief.model_validate(json.loads(Path(args.spec).read_text())) if args.spec else DesignBrief(prompt="")
    execution = ExecutionResult(ok=True, stl_path=args.stl_path)
    result = evaluate_leg1(execution, spec)
    print(f"{'PASS' if result.overall_pass else 'FAIL'} (stopped at {result.stopped_at})")
    print(result.feedback_text())
    if not args.spec:
        print("\n(no --spec given: mesh-validity/manufacturability only, no BRep provenance since this STL wasn't generated by cadyfiner itself)")


def main() -> None:
    parser = argparse.ArgumentParser(prog="cadyfiner", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_refine = sub.add_parser("refine", help="Refine a raw CAD prompt into an expanded one, without generating geometry.")
    p_refine.add_argument("prompt")
    p_refine.add_argument("--json", action="store_true", help="Print the full DesignBrief JSON instead of a summary.")
    _add_backend_args(p_refine)
    p_refine.set_defaults(func=cmd_refine)

    p_gen = sub.add_parser("generate", help="Generate a CadQuery model from a prompt (optionally refined first) and export STL.")
    p_gen.add_argument("prompt")
    p_gen.add_argument("--refine", action="store_true", help="Refine the prompt before generating (recommended).")
    p_gen.add_argument("--out", default=None, help="Output STL path (default: workspace/cli_out/model-<id>.stl).")
    p_gen.add_argument("--spec", default=None, help="Path to a ground-truth DesignBrief JSON to check the result against.")
    _add_backend_args(p_gen)
    p_gen.set_defaults(func=cmd_generate)

    p_check = sub.add_parser("check", help="Run the Leg-1 quality check against an existing STL.")
    p_check.add_argument("stl_path")
    p_check.add_argument("--spec", default=None, help="Path to a ground-truth DesignBrief JSON to check against.")
    p_check.set_defaults(func=cmd_check)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
