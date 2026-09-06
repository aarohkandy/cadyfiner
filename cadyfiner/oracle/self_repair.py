"""Self-repair retry: feed the exact execution error back to the same generator for
one corrected attempt.

WHY this exists: real diagnosis across 13 held-out seeds run through gemma4:e4b found
~8/26 arm-attempts failed on hallucinated/nonexistent CadQuery API calls (methods that
don't exist, or called on the wrong object type), 4/26 on stateful Workplane-chaining
mistakes ("no pending wires"), 2/26 on plain SyntaxErrors, and 2/26 on hard-rule
violations (prefilter rejections) — almost none ever reached a genuine geometry/dimension
problem. ``CADQUERY_PROMPT_RULES`` already states the contract once, up front; a concrete
stack trace from the model's OWN code is a much stronger correction signal than restating
the same static rules a second time (the "self-debugging" prompting pattern).

Deliberately a separate, opt-in function rather than a change to ``run_cadquery`` or its
existing call sites — see the calling task for why adoption is a later step.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from cadyfiner.oracle.execute import ExecutionResult, extract_code, run_cadquery

# Scoped to exactly the failure classes the diagnosis found are actually about the
# CODE (a fixable bug or a broken hard rule), not the environment or the clock.
# `timeout`/`no_result_file` are resource/wall-clock kills a corrected generation can't
# reliably fix and cost another full 60-250s Ollama call to find out; `cadquery_unavailable`
# is an install problem, not a prompt problem; `export_failed` never showed up in the 26
# diagnosed failures, so it's left out rather than guessed at.
#
# `api_check_rejected` (cadyfiner.oracle.api_check, wired into run_cadquery alongside this
# module in the combined variant) is included deliberately: it's the SAME class of bug this
# module already repairs (a hallucinated CadQuery call), just caught earlier with a more
# precise message ("line N: `.union` is not a valid Shape method, did you mean...") than a
# generic subprocess traceback — if anything, an easier repair target, not a harder one.
# Found live: without this, combining the two modules silently broke repair for exactly the
# hallucination cases both were built to fix, since api_check now wins the race to reject
# the code before the subprocess (and its `exception`-type error) is ever reached.
_REPAIRABLE_ERROR_TYPES = frozenset({"prefilter_rejected", "exception", "no_result_variable", "api_check_rejected"})


class RepairedExecutionResult(ExecutionResult):
    repair_attempts: int = 0


def _build_repair_prompt(original_prompt: str, code: str, result: ExecutionResult) -> str:
    if result.error_type == "prefilter_rejected":
        # A prefilter rejection means a HARD RULE from the instructions above was
        # violated, not a runtime accident — say so explicitly rather than handing back
        # something that reads like a generic bug report the model might route around
        # while keeping the same violation in spirit.
        failure_section = (
            "This was REJECTED BEFORE EXECUTION for violating a hard rule stated above: "
            f"{result.error_message}\n"
            "That is a hard requirement, not a bug to work around — remove or replace "
            "whatever triggered it. Do not try to satisfy the letter of the check while "
            "keeping the same underlying approach."
        )
    else:
        failure_section = (
            f"Running this code failed with this exact error:\n{result.error_type}: "
            f"{result.error_message}\n\n"
            "Common real causes for this: a method or attribute that doesn't exist on the "
            "CadQuery object you called it on (methods live on Workplane/Shape objects, "
            "never as bare `cq.*` functions), a wrong argument count or order, or calling "
            "an operation like extrude/cut before a sketch or wire is actually pending on "
            "the Workplane chain. Check the exact API you used against real CadQuery and "
            "fix the root cause, don't just suppress the error."
        )
    return (
        f"{original_prompt}\n\n"
        "Your previous attempt below failed and needs a corrected version.\n\n"
        f"Previous code:\n```python\n{code}```\n\n"
        f"{failure_section}\n\n"
        "Return the corrected code the same way as before: Python only, a `result` "
        "variable, no prose."
    )


def generate_and_execute_with_repair(
    prompt: str,
    generate_fn: Callable[..., str],
    generate_kwargs: dict[str, Any],
    out_dir: Path,
    *,
    max_repairs: int = 1,
    run_cadquery_kwargs: dict[str, Any] | None = None,
) -> RepairedExecutionResult:
    """Drop-in replacement for ``extract_code(generate_fn(prompt, **kwargs))`` +
    ``run_cadquery(code, out_dir)``, with up to ``max_repairs`` corrective retries.

    ``prompt`` is the FULL prompt normally passed to ``generate_fn`` at existing call
    sites — i.e. already ``CADQUERY_PROMPT_RULES + "<design request>"`` — so the repair
    round can carry the same hard-rule context forward without the caller needing to
    pass rules and request separately.
    """

    run_kwargs = run_cadquery_kwargs or {}
    code = extract_code(generate_fn(prompt, **generate_kwargs))
    result = run_cadquery(code, out_dir, **run_kwargs)

    attempts = 0
    while not result.ok and attempts < max_repairs and result.error_type in _REPAIRABLE_ERROR_TYPES:
        attempts += 1
        repair_prompt = _build_repair_prompt(prompt, code, result)
        code = extract_code(generate_fn(repair_prompt, **generate_kwargs))
        result = run_cadquery(code, out_dir, **run_kwargs)

    return RepairedExecutionResult(**result.model_dump(), repair_attempts=attempts)
