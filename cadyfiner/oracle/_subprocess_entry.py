"""Out-of-process CadQuery execution entry point.

Invoked as ``python -m cadyfiner.oracle._subprocess_entry <payload.json>`` by
:mod:`cadyfiner.oracle.execute`. Ports ai-cad's out-of-process pattern
(``ai-cad/backend/app/services/executors/runtime.py``: subprocess + JSON
payload in, JSON result out), with additions neither ai-cad's runtime nor
cadybara-kitchen's ``cadquery_runner.py`` has: OS resource limits (CPU time,
address space, file size, process count). cadyfiner runs hundreds of
unattended, LLM-written executions rather than a single human-watched
sweep, so a runaway fillet loop, a multiprocessing fork bomb, or an
unbounded file write needs a backstop that doesn't depend on the parent
process's wall-clock timeout alone.

Deliberately does not restrict builtins inside the exec namespace (unlike
cadybara-kitchen's runner). The real sandbox boundary here is the OS
process itself — short-lived, resource-limited, and killed by the parent's
timeout — not an in-process restricted-builtins dict, which is well known
to be escapable and was giving cadybara-kitchen's runner a false sense of
security anyway (it left ``__import__`` in its allowed builtins).

STL output filename is scoped by ``run_id`` (matching the payload/result
convention), not a fixed ``model.stl`` — an earlier version used a fixed
name, which an adversarial review caught as a real correctness bug: two
overlapping ``run_cadquery()`` calls sharing an ``out_dir`` (routine once
the optimizer runs generations in parallel) would silently overwrite each
other's STL, corrupting whichever result was scored second.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any


def _write_result(result_path: Path, payload: dict[str, Any]) -> None:
    result_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _set_resource_limits(
    cpu_seconds: int, address_space_bytes: int, file_size_bytes: int, max_processes: int
) -> dict[str, bool]:
    """Apply POSIX resource limits, returning which ones actually took effect.

    Every limit is wrapped independently: on macOS, ``RLIMIT_AS`` cannot be
    lowered at all (raises ``ValueError: current limit exceeds maximum
    limit``) — verified directly while building this module. Swallowing
    that silently would make the cap a no-op with no way for a caller to
    know; the returned dict makes that visible instead.

    ``max_processes`` was originally 32, added by an adversarial review as
    a fork-bomb guard, verified only on macOS — where ``RLIMIT_NPROC``
    doesn't exist at all, so the guard was silently a total no-op there and
    the value was never really exercised. First real Linux run (homebase)
    hit it immediately: importing numpy inside the sandboxed subprocess
    failed with a bare ``KeyboardInterrupt`` from deep inside
    ``numpy._core.multiarray`` — RLIMIT_NPROC counts threads too on Linux,
    and NumPy's BLAS backend spins up a thread pool on import, well past 32.
    Every prior "verified" test of this limit only ran on the platform
    where it does nothing; raised to 512, comfortably above what NumPy/OCC
    need while still bounding an actual fork bomb.
    """

    applied = {"cpu": False, "address_space": False, "file_size": False, "nproc": False}
    try:
        import resource
    except ImportError:
        return applied  # resource module is POSIX-only; skip silently on Windows.

    try:
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
        applied["cpu"] = True
    except (ValueError, OSError):
        pass
    try:
        resource.setrlimit(resource.RLIMIT_AS, (address_space_bytes, address_space_bytes))
        applied["address_space"] = True
    except (ValueError, OSError):
        pass
    try:
        resource.setrlimit(resource.RLIMIT_FSIZE, (file_size_bytes, file_size_bytes))
        applied["file_size"] = True
    except (ValueError, OSError):
        pass
    try:
        resource.setrlimit(resource.RLIMIT_NPROC, (max_processes, max_processes))
        applied["nproc"] = True
    except (ValueError, OSError, AttributeError):
        pass  # RLIMIT_NPROC doesn't exist on macOS at all.
    return applied


def main() -> None:
    payload_path = Path(sys.argv[1])
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    result_path = Path(payload["result_path"])
    started = time.perf_counter()

    limits_applied = _set_resource_limits(
        cpu_seconds=int(payload.get("cpu_seconds", 40)),
        address_space_bytes=int(payload.get("address_space_bytes", 4 * 1024 ** 3)),
        file_size_bytes=int(payload.get("file_size_bytes", 300 * 1024 ** 2)),
        max_processes=int(payload.get("max_processes", 512)),
    )

    try:
        import cadquery as cq
    except Exception as exc:
        _write_result(
            result_path,
            {
                "ok": False,
                "error_type": "cadquery_unavailable",
                "error_message": str(exc),
                "wall_time_s": time.perf_counter() - started,
                "resource_limits_applied": limits_applied,
            },
        )
        return

    code = payload["code"]
    namespace: dict[str, Any] = {"cq": cq}

    try:
        exec(compile(code, "<cadyfiner-generated>", "exec"), namespace)
    except Exception as exc:
        _write_result(
            result_path,
            {
                "ok": False,
                "error_type": "exception",
                "error_message": f"{type(exc).__name__}: {exc}",
                "wall_time_s": time.perf_counter() - started,
                "resource_limits_applied": limits_applied,
            },
        )
        return

    result = namespace.get("result")
    if result is None:
        _write_result(
            result_path,
            {
                "ok": False,
                "error_type": "no_result_variable",
                "error_message": "Generated code did not define a `result` variable.",
                "wall_time_s": time.perf_counter() - started,
                "resource_limits_applied": limits_applied,
            },
        )
        return

    stl_path = Path(payload["out_dir"]) / f"model-{payload['run_id']}.stl"
    stl_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        shape = result.val() if hasattr(result, "val") else result
        # Exact BRep-level facts from the CAD kernel itself, not the
        # tessellation. These are authoritative in a way a mesh-derived
        # approximation cannot be: OCC's own boolean-union output can be a
        # Compound of >1 Solid (e.g. two parts that only touch tangentially
        # never actually fuse into one solid) even when a mesh-connectivity
        # check on the exported STL would miss it because triangles at the
        # seam happen to share vertices. Verified directly against a real
        # tangent-union test case while building this module.
        is_valid_brep = bool(shape.isValid())
        n_solids = len(shape.Solids())
        occ_volume = float(shape.Volume())
        occ_area = float(shape.Area())
        bbox = shape.BoundingBox()
        # tolerance/angularTolerance tuned for a reasonably faithful but not
        # excessively heavy mesh for the secondary (trimesh-based) checks.
        shape.exportStl(str(stl_path), tolerance=0.01, angularTolerance=0.1)
    except Exception as exc:
        _write_result(
            result_path,
            {
                "ok": False,
                "error_type": "export_failed",
                "error_message": f"{type(exc).__name__}: {exc}",
                "wall_time_s": time.perf_counter() - started,
                "resource_limits_applied": limits_applied,
            },
        )
        return

    _write_result(
        result_path,
        {
            "ok": True,
            "stl_path": str(stl_path),
            "cq_volume": occ_volume,
            "cq_bbox": {"x": float(bbox.xlen), "y": float(bbox.ylen), "z": float(bbox.zlen)},
            "cq_is_valid_brep": is_valid_brep,
            "cq_n_solids": n_solids,
            "cq_area": occ_area,
            "wall_time_s": time.perf_counter() - started,
            "resource_limits_applied": limits_applied,
        },
    )


if __name__ == "__main__":
    main()
