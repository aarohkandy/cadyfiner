"""Static hallucinated-API pre-check for generated CadQuery code.

WHY this exists: real diagnosis across 13 held-out seeds run through gemma4:e4b found
~8/26 arm-attempts failing on CadQuery API calls that simply don't exist (a module
function that's really a method, an attribute never present on any real CadQuery class),
plus 2/26 outright SyntaxErrors — all of which currently only surface after paying for a
full sandboxed subprocess dispatch and its wall-clock timeout, then reading a generic
Python traceback out of ``_subprocess_entry.py``'s catch-all ``exception`` type. This
module catches the ones it safely can, cheaply, before any of that.

A separate module rather than folded into ``prefilter()``: ``prefilter()`` is explicitly
documented (see its own docstring and ``execute.py``'s module docstring) as a cheap
textual/regex heuristic with no imports. This check parses an AST and introspects the
real installed ``cadquery`` package — a different cost model (seconds on first import,
cached; microseconds per call after) and a different kind of check, so it gets its own
module, its own tests, and doesn't blur that distinction for ``prefilter()``'s callers.

**Catches** (confirmed against the real installed cadquery 2.8.0, not guessed):
- Bare nonexistent module attributes: ``cq.Angle``, ``cq.union`` (real diagnosis examples;
  ``union`` specifically gets a "that's a method, not a module function" message, since
  that confusion is exactly what the diagnosis caught).
- Nonexistent methods/attributes chained off an expression whose CadQuery type
  (Workplane / Sketch / the OCC Shape family) is statically determinable through a
  straight-line sequence of assignments and fluent-chained calls: ``wp.xAxis``,
  ``cq.Workplane("XY").workplaneAt(...)``, ``shape.getTranslation()`` (all real diagnosis
  examples).
- Code that doesn't parse as Python at all (free — it comes from the same ``ast.parse``
  call this needs anyway).

**Cannot catch, by construction — stated plainly, not silently**:
- Dynamic/computed attribute access (``getattr(wp, name)``) — invisible to static AST
  inspection.
- Attribute access on anything whose type this checker's simple model can't pin down:
  results of user-defined helper functions, comprehensions, subscripts, ternaries,
  variables only ever assigned inside a loop/if/def body (type tracking here only runs
  over module-top-level statements — see below), or chained through any type-changing
  method not in the small known-transitions table. All of these are treated as unknown
  and *skipped* — this trades recall for zero added false-positive risk, deliberately.
- Wrong-arity/wrong-argument calls to a method that DOES exist — ``Workplane.add()`` /
  ``Workplane.rotate()`` from the real diagnosis are both this: a real method, called with
  the wrong argument count. An existence check cannot and does not attempt to catch this
  class; it would need signature/arity checking, a different, unimplemented feature.
- "No pending wires" / "cannot find a solid on stack" chaining-sequence errors — these are
  runtime *state* errors (calling extrude before any sketch is pending on the Workplane
  chain), not a misspelled name; nothing about this feature addresses them.
- Two-levels-deep submodule attribute access (``cq.selectors.Foo``, ``cq.exporters.Bar``)
  is not verified — only direct ``cq.<attr>`` and Workplane/Sketch/Shape-family method
  chains are. (``cq.exporters.export`` specifically is already separately banned by
  ``prefilter()``'s existing substring rule, so this gap has no practical effect there.)
"""

from __future__ import annotations

import ast
import difflib
from functools import lru_cache

_SHAPE_FAMILY = ("Shape", "Edge", "Face", "Solid", "Wire", "Vertex", "Compound", "Shell")

# Verified against a real running snippet this session:
#   cq.Workplane("XY").box(...).faces(">Z").workplane().sketch().rect(...).finalize().extrude(5)
# Without this table, .finalize() would be checked against Workplane's own attribute set
# (it isn't one -- only Sketch has it) and would false-reject a genuinely valid, executed
# pattern. Any OTHER type-changing method not listed here just stops the type chase
# (unknown from that point on) rather than guessing -- under-catches, never false-positives.
_TRANSITIONS = {
    ("workplane", "sketch"): "sketch",   # Workplane.sketch() enters 2D sketch mode
    ("workplane", "val"): "shape",
    ("sketch", "val"): "shape",
    ("sketch", "finalize"): "workplane",  # Sketch.finalize() returns to the parent Workplane
}


@lru_cache(maxsize=1)
def _introspect_api() -> dict[str, frozenset]:
    import cadquery as cq  # ~5.9s cold in this venv; cached so it's paid once per process,
    # not once per candidate -- the whole point is to be cheaper than a subprocess+timeout
    # after the first call, not before it.

    def pub(cls) -> frozenset:
        return frozenset(a for a in dir(cls) if not a.startswith("_"))

    shape = frozenset().union(*(pub(getattr(cq, n)) for n in _SHAPE_FAMILY))
    return {
        "module": pub(cq),
        "workplane": pub(cq.Workplane),
        "sketch": pub(cq.Sketch),
        "shape": shape,
    }


def _infer(expr: ast.expr, env: dict[str, str], api: dict[str, frozenset], cq_aliases: set[str]) -> str | None:
    if isinstance(expr, ast.Name):
        if expr.id in cq_aliases:
            return "module"
        return env.get(expr.id)
    if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Attribute):
        base = _infer(expr.func.value, env, api, cq_aliases)
        attr = expr.func.attr
        if base == "module":
            return {"Workplane": "workplane", "Sketch": "sketch"}.get(attr)
        if base in ("workplane", "sketch", "shape") and attr in api[base]:
            return _TRANSITIONS.get((base, attr), base)
        return None  # invalid, or a type-changing method we don't model -- stop the chase
    return None  # bare Attribute, subscripts, comprehensions, other calls, results of
    # user-defined functions, etc. -- all deliberately "unknown"


def _check_one(node: ast.Attribute, env: dict[str, str], api: dict[str, frozenset], cq_aliases: set[str], cq_version: str) -> str | None:
    base = _infer(node.value, env, api, cq_aliases)
    if base is None or node.attr.startswith("_"):
        return None
    if base == "module":
        if node.attr in api["module"]:
            return None
        if node.attr in api["workplane"] or node.attr in api["shape"]:
            return (
                f"line {node.lineno}: `cq.{node.attr}` does not exist in cadquery "
                f"{cq_version} -- `{node.attr}` is a method on Workplane/Shape, not a "
                f"module-level function (e.g. `shape.{node.attr}(other)`, not "
                f"`cq.{node.attr}(...)`)"
            )
        close = difflib.get_close_matches(node.attr, sorted(api["module"]), n=1, cutoff=0.6)
        suggestion = f" (did you mean `cq.{close[0]}`?)" if close else ""
        return f"line {node.lineno}: `cq.{node.attr}` does not exist in cadquery {cq_version}{suggestion}"
    if node.attr in api[base]:
        return None
    close = difflib.get_close_matches(node.attr, sorted(api[base]), n=1, cutoff=0.6)
    suggestion = f" (did you mean `.{close[0]}(...)`?)" if close else ""
    kind = {"workplane": "Workplane", "sketch": "Sketch", "shape": "Shape"}[base]
    return f"line {node.lineno}: `.{node.attr}` is not a valid {kind} attribute/method in cadquery {cq_version}{suggestion}"


def check_api_usage(code: str) -> str | None:
    """Return a rejection reason, or None if no statically-detectable API misuse is found.

    Same contract shape as :func:`prefilter` — callers chain them identically.
    """

    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError) as exc:
        return f"code does not parse: {exc}"

    cq_aliases = {
        alias.asname or alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
        if alias.name == "cadquery"
    }
    if not cq_aliases:
        return None  # nothing to check against; prefilter() separately requires the import

    try:
        api = _introspect_api()
        import cadquery as cq

        cq_version = getattr(cq, "__version__", getattr(cq, "version", "?"))
    except Exception:
        return None  # fail open: never block a run because introspection itself broke

    env: dict[str, str] = {}
    findings: list[tuple[int, int, str]] = []
    # Type tracking only over TOP-LEVEL statements (tree.body), deliberately -- see module
    # docstring. Nested-block statements are still walked for Attribute-node checks using
    # whatever `env` accumulated from preceding top-level statements.
    for stmt in tree.body:
        for node in ast.walk(stmt):
            if isinstance(node, ast.Attribute):
                msg = _check_one(node, env, api, cq_aliases, cq_version)
                if msg:
                    findings.append((node.lineno, node.col_offset, msg))
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
            t = _infer(stmt.value, env, api, cq_aliases)
            name = stmt.targets[0].id
            if t:
                env[name] = t
            else:
                env.pop(name, None)
        elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name) and stmt.value is not None:
            t = _infer(stmt.value, env, api, cq_aliases)
            name = stmt.target.id
            if t:
                env[name] = t
            else:
                env.pop(name, None)

    if not findings:
        return None
    findings.sort(key=lambda f: (f[0], f[1]))
    extra = f" (+{len(findings) - 1} more similar issue(s))" if len(findings) > 1 else ""
    return findings[0][2] + extra
