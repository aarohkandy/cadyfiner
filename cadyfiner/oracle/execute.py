"""Sandboxed CadQuery execution: subprocess isolation + timeout + JSON IPC.

Public entry point is :func:`run_cadquery`. Two-layer defense, ported from
the two existing local repos rather than depending on either as a package
(``ai-cad`` drags in FastAPI/planner/compiler machinery cadyfiner doesn't
need; ``cadybara-kitchen``'s in-process ``exec()`` has no timeout, which is
fine for a human-watched CLI sweep and unsafe for an unattended optimizer
loop):

1. A cheap substring/regex prefilter (ported from ``cadybara-kitchen``'s
   ``DISALLOWED_PATTERNS``) rejects obviously-bad code before paying for a
   subprocess + CadQuery import cycle. This is a fast-fail heuristic, not a
   security boundary.
2. Actual isolation is the OS process itself (:mod:`cadyfiner.oracle.
   _subprocess_entry`, invoked out-of-process in its own session with a
   wall-clock timeout and RLIMIT_CPU/RLIMIT_AS/RLIMIT_FSIZE/RLIMIT_NPROC
   resource caps) — the same subprocess pattern
   ``ai-cad/backend/app/services/executors/cadquery_executor.py`` uses,
   extended with process-group cleanup and stricter resource limits.

Honesty note: this isolates against *hangs and resource exhaustion* from
LLM-written code that is buggy, not against a determined malicious
adversary (no network namespace isolation, no filesystem jail). That's the
right threat model for code written by Ollama/Claude under a constrained
prompt, not for untrusted internet input.

This module went through one adversarial-review pass that found 18
confirmed issues (13 in this file / its subprocess entry point, 5 in
checks.py) before any of it had run outside hand-written smoke tests.
Several fixes below exist specifically because of that pass, noted inline;
see the git history / review transcript for the full findings if the
reasoning behind a particular guard looks non-obvious.
"""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import uuid
from pathlib import Path

from pydantic import BaseModel, ConfigDict

CADQUERY_PROMPT_RULES = """You are generating parametric CAD source code.

Return only Python CadQuery code. Do not explain the design in prose.

Hard requirements:
- Use millimeters.
- Use `import cadquery as cq`.
- Do not use `from cadquery import ...`; all CadQuery calls go through `cq`.
- Build all geometry starting from `cq.Workplane("XY")` as the base plane. The
  +Z direction is "up" / height. Do not build the model starting from a "YZ",
  "XZ", or other non-XY base plane — a checker downstream measures height
  against the Z axis and treats X/Y as the horizontal footprint, so building
  on a different base plane will make a correct model measure as wrong.
- Define the final model in a variable named `result` (a CadQuery Workplane or Shape).
- Do not call `cq.exporters.export`, `.exportStl(`, `.exportStep(`, or any other
  export method; the harness exports STL after your code runs.
- Do not call `print`, `show_object`, `display`, or any viewer function.
- Do not read files, write files, use networking, shell commands, subprocesses,
  or the `multiprocessing`/`threading` modules.
- Prefer simple, printable, watertight solids using boxes, cylinders, holes, cuts,
  unions, fillets, and chamfers.
- If fillets fail on complex geometry, omit them instead of producing invalid code.
"""

_BLOCK_RE = re.compile(r"```(?:python|py)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)

# Plain substrings for patterns that are never legitimate. Regexes (compiled
# below) for ones that would otherwise false-positive on realistic code —
# an adversarial review confirmed "socket" bans "socket head cap screw"
# (a completely ordinary phrase in the mechanical-parts domain this project
# targets), "os." bans any variable named e.g. `pos` followed by attribute
# access, and "print(" bans identifiers like `make_blueprint(`. Verified by
# executing all three examples as real, valid CadQuery through this exact
# prefilter before and after the fix.
_DISALLOWED_SUBSTRINGS = (
    "subprocess",
    "requests",
    "httpx",
    "urllib",
    "shutil",
    "pathlib",
    "from os",
    "from sys",
    "from cadquery",
    "exporters.export",
    "show_object",
    "multiprocessing",
    "threading",
    "concurrent.futures",
    "__",
)

_DISALLOWED_REGEXES = (
    # \bsocket\b alone still matches the ordinary English phrase "socket head
    # cap screw" (confirmed: that was the review's own false-positive repro,
    # and its own suggested \bsocket\b fix does not actually avoid it) — the
    # real intent is banning the networking module, so anchor to the import.
    re.compile(r"\b(import|from)\s+socket\b"),
    re.compile(r"(?<![a-z0-9_])os\."),
    re.compile(r"(?<![a-z0-9_])sys\."),
    re.compile(r"(?<![a-z0-9_])open\s*\("),
    re.compile(r"(?<![a-z0-9_])exec\s*\("),
    re.compile(r"(?<![a-z0-9_])eval\s*\("),
    re.compile(r"(?<![a-z0-9_])print\s*\("),
    re.compile(r"(?<![a-z0-9_])display\s*\("),
    re.compile(r"import\s+os\b"),
    re.compile(r"import\s+sys\b"),
    # Matched against `lowered`, so the alternatives must be lowercase too —
    # an earlier version of this regex used mixed case and silently matched
    # nothing at all against the pre-lowered string. Caught by testing the
    # exact adversarial-review repro, not by reading the code.
    re.compile(r"\.export(stl|step|brep|3mf|dxf)?\s*\("),
)


class ExecutionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool
    stl_path: str | None = None
    cq_volume: float | None = None
    cq_bbox: dict[str, float] | None = None
    cq_is_valid_brep: bool | None = None
    cq_n_solids: int | None = None
    cq_area: float | None = None
    error_type: str | None = None
    error_message: str | None = None
    wall_time_s: float | None = None
    resource_limits_applied: dict[str, bool] | None = None
    code: str = ""


def extract_code(llm_output: str) -> str:
    """Pull a fenced ```python``` block out of raw LLM output, or fall back to the whole text."""

    match = _BLOCK_RE.search(llm_output)
    if match:
        return match.group(1).strip() + "\n"
    return llm_output.strip() + "\n"


def prefilter(code: str) -> str | None:
    """Return a rejection reason, or None if the code passes the fast-fail heuristic."""

    lowered = code.lower()
    for pattern in _DISALLOWED_SUBSTRINGS:
        if pattern in lowered:
            return f"disallowed pattern: {pattern!r}"
    for regex in _DISALLOWED_REGEXES:
        if regex.search(lowered):
            return f"disallowed pattern: {regex.pattern!r}"
    if "import cadquery" not in lowered:
        return "missing `import cadquery`"
    if "result" not in code:
        return "no `result` assignment found"
    return None


def run_cadquery(
    code: str,
    out_dir: Path,
    *,
    timeout_s: float = 45.0,
    cpu_seconds: int = 40,
    address_space_bytes: int = 4 * 1024 ** 3,
    file_size_bytes: int = 300 * 1024 ** 2,
    max_processes: int = 32,
    stdio_capture_bytes: int = 64 * 1024,
) -> ExecutionResult:
    """Execute CadQuery source in an isolated subprocess and return its result.

    ``code`` should already be extracted (see :func:`extract_code`) —
    fenced-block stripping happens once, at the caller, not here.
    """

    rejection = prefilter(code)
    if rejection is not None:
        return ExecutionResult(ok=False, error_type="prefilter_rejected", error_message=rejection, code=code)

    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex[:12]
    payload_path = out_dir / f"payload-{run_id}.json"
    result_path = out_dir / f"result-{run_id}.json"
    # stdout/stderr redirected to real files (not pipes): this both bounds
    # what the parent ever holds in memory (a pipe read via capture_output
    # is unbounded until the child exits; a file is capped by RLIMIT_FSIZE
    # in the child) and lets us read back a tail deterministically. Found
    # by adversarial review: capture_output=True on a child whose output
    # volume is influenced by generated code has no size cap at all.
    stdout_path = out_dir / f"stdout-{run_id}.log"
    stderr_path = out_dir / f"stderr-{run_id}.log"

    payload_path.write_text(
        json.dumps(
            {
                "code": code,
                "out_dir": str(out_dir),
                "result_path": str(result_path),
                "run_id": run_id,
                "cpu_seconds": cpu_seconds,
                "address_space_bytes": address_space_bytes,
                "file_size_bytes": file_size_bytes,
                "max_processes": max_processes,
            }
        ),
        encoding="utf-8",
    )

    def _cleanup(*paths: Path) -> None:
        for p in paths:
            p.unlink(missing_ok=True)

    def _tail(path: Path, n_bytes: int) -> str:
        if not path.exists():
            return ""
        data = path.read_bytes()
        return data[-n_bytes:].decode("utf-8", errors="replace")

    proc = None
    try:
        with open(stdout_path, "wb") as out_f, open(stderr_path, "wb") as err_f:
            proc = subprocess.Popen(
                [sys.executable, "-m", "cadyfiner.oracle._subprocess_entry", str(payload_path)],
                stdout=out_f,
                stderr=err_f,
                start_new_session=True,  # own process group, so a timeout can kill the whole tree
            )
            proc.wait(timeout=timeout_s)
        returncode = proc.returncode
    except subprocess.TimeoutExpired:
        # Kill the whole process group, not just the direct child — a
        # multiprocessing-spawning child would otherwise orphan workers that
        # survive past this wall-clock backstop. Confirmed by adversarial
        # review as a real gap in the previous single-PID `.kill()` behavior.
        if proc is not None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            proc.wait(timeout=5)
        # A slow-to-report run may have finished and written a valid result
        # just as (or after) the wall-clock timeout fired. Prefer the real
        # outcome over a misleading "timeout" label when that happened —
        # confirmed reachable by adversarial review (an atexit handler that
        # sleeps after a fully successful build).
        if result_path.exists():
            try:
                raw = json.loads(result_path.read_text(encoding="utf-8"))
                _cleanup(payload_path, result_path, stdout_path, stderr_path)
                return ExecutionResult(**raw, code=code)
            except (json.JSONDecodeError, OSError):
                pass
        _cleanup(payload_path, result_path, stdout_path, stderr_path)
        return ExecutionResult(
            ok=False,
            error_type="timeout",
            error_message=f"execution exceeded {timeout_s}s wall-clock timeout",
            code=code,
        )

    if not result_path.exists():
        stderr_tail = _tail(stderr_path, stdio_capture_bytes)
        _cleanup(payload_path, stdout_path, stderr_path)
        return ExecutionResult(
            ok=False,
            error_type="no_result_file",
            error_message=(
                f"subprocess exited {returncode} without writing a result "
                f"(likely killed by a resource limit). stderr tail: {stderr_tail[-500:]}"
            ),
            code=code,
        )

    raw = json.loads(result_path.read_text(encoding="utf-8"))
    _cleanup(payload_path, result_path, stdout_path, stderr_path)
    return ExecutionResult(**raw, code=code)
