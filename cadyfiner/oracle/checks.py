"""Leg-1: the single gating quality composite, used identically by the
optimizer's accept/reject decision and the final exit-criteria harness.

Short-circuits cheapest-first (mirrors MUSE's reported failure cascade —
arXiv 2605.28579 — where most failures occur at the earliest, cheapest
stages): execute -> mesh validity -> spec conformance -> manufacturability.

Mesh validity is gated on exact BRep-level facts from the CAD kernel
(``ExecutionResult.cq_is_valid_brep``, ``cq_n_solids``), not on trimesh's
tessellation-derived ``is_watertight``/``body_count``. This was a deliberate
correction made while building this module: a hand-written test case (a
cylinder union'd with a back panel that only touches it tangentially,
never actually overlapping) produced a CadQuery ``Compound`` of 2 separate
``Solid``s — a genuine modeling defect — but trimesh's mesh-connectivity
check reported ``body_count == 1`` anyway, because triangles at the tangent
seam happened to share vertices in the tessellation. OCC's own
``Solids()``/``isValid()`` caught it; the mesh-based check missed it. STL
tessellation artifacts run the other way too: a solid OCC considers fully
valid can still export with tiny non-watertight seams at curved-face
boundaries (verified directly: identical face count and non-watertight
result across three STL export tolerances on a known-OCC-valid solid) —
that's a mesh print-readiness question, not a design-validity one, so it's
checked in the manufacturability stage instead of gating here.

This module never claims to check what it cannot. A feature with no count
or size in the spec is reported as unverifiable, not silently passed — see
:func:`evaluate_leg1`'s docstring for why that distinction matters here.

Axis convention: ``CADQUERY_PROMPT_RULES`` asks generated code to build on
``cq.Workplane("XY")`` (height <-> Z), but an adversarial review confirmed a
prompt rule alone doesn't guarantee it — a model can pass every hard
requirement and still build on ``cq.Workplane("YZ")``, and the same review
found that even under the XY convention, a single stated width/depth/length
was always compared against the *smaller* of the two horizontal extents
regardless of which one it actually was, false-failing correct models. Both
are fixed the same way: :func:`_match_linear_dimensions` brute-forces the
assignment of stated height/width/depth/length onto the 3 measured axes
that minimizes total relative error, rather than assuming any fixed
mapping. This is deliberately more forgiving of legitimate axis ordering
than of an actually-wrong envelope, and explicitly flags a stated
dimension it cannot assign an axis to (more than 3 given at once) instead
of silently dropping it — also caught by that review.

IMPORTANT — this applies to BOTH callers of this function: ``spec`` here
is the design brief to check the mesh AGAINST, and both
``cadyfiner.optimize._score_candidate`` and
``cadyfiner.harness.run_paired_evaluation`` deliberately pass the seed
bank's hand-authored ground-truth spec, never the refiner's own emitted
spec, for the same reason in both places — a refiner that emitted a
mostly-empty spec (or simply declined to state anything) would otherwise
pass spec_conformance almost for free by grading its own homework, which
is exactly the kind of reward-hacking this project's own empirical
findings warn against. (An earlier version of this docstring claimed the
optimizer scored the refiner's own spec; that was never actually true of
the code and has been corrected here, not there.)
"""

from __future__ import annotations

import math
from itertools import permutations
from typing import Any

from pydantic import BaseModel, ConfigDict

from cadyfiner.oracle.execute import ExecutionResult
from cadyfiner.spec import DesignBrief

# Generic FDM-printability sanity floors/ceilings, independent of any stated
# spec. Derived from this session's analysis of the 94 real cad_grade STLs
# (see research/baseline/conform.py): 0 of 94 real items fell outside
# [10mm, 500mm] max-dimension, so these are deliberately loose bounds meant
# to catch generation pathologies (a 2mm sliver, a 3-meter runaway), not to
# second-guess legitimate design choices.
_MIN_SCALE_MM = 5.0
_MAX_SCALE_MM = 600.0
_MIN_WALL_PROXY_MM = 0.6
_MAX_DEGENERATE_FACE_FRACTION = 0.02


class CheckStage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    passed: bool
    detail: str


class Leg1Result(BaseModel):
    model_config = ConfigDict(extra="forbid")

    overall_pass: bool
    stopped_at: str
    stages: list[CheckStage]
    measured: dict[str, Any]

    def feedback_text(self) -> str:
        """Render failing/unverifiable stages as text for the optimizer's reflective loop."""

        lines = [f"[{s.name}] {'PASS' if s.passed else 'FAIL'}: {s.detail}" for s in self.stages]
        return "\n".join(lines)


def _stop(stages: list[CheckStage], stopped_at: str, measured: dict[str, Any]) -> Leg1Result:
    return Leg1Result(overall_pass=False, stopped_at=stopped_at, stages=stages, measured=measured)


def _estimate_through_holes(mesh: Any) -> int:
    """Approximate through-hole count from Euler characteristic.

    For a single closed orientable watertight surface, euler = 2 - 2*genus,
    and each independent through-hole/handle contributes one to genus. This
    is a real but approximate signal (a hole tunneling through two adjacent
    faces still counts as one handle; a blind hole that doesn't pass through
    contributes nothing) — used only as a best-effort check, never as
    ground truth.
    """

    genus = (2 - int(mesh.euler_number)) / 2
    return max(0, round(genus))


_TOOTH_MIN_RELATIVE_AMPLITUDE = 0.005  # 0.5% of mean radius


def _estimate_tooth_count(mesh: Any) -> int | None:
    """Approximate radial-feature count (gear teeth, splines) via a mid-height cross-section.

    Sections the mesh at its centroid height, resamples the boundary radius
    onto a uniform angular grid, and finds the dominant angular frequency
    via FFT. Two approaches were tried during development: peak-counting
    with a fixed minimum-distance parameter looked exact on a first
    synthetic 3-gear test (8/12/20 teeth) but that test happened to hand it
    the right distance implicitly; retested honestly (distance not tuned to
    the answer) it badly overcounted low-tooth-count gears on tessellation
    noise (31 detected on an 8-tooth gear). Dominant-frequency FFT is exact
    on the same three cases (8, 12, 20) with no count-dependent tuning.

    The remaining failure mode — a plain smooth cylinder's polygon
    tessellation has its own dominant angular frequency (its facet count)
    and would otherwise be reported as having that many "teeth" — is
    rejected by an amplitude gate: real teeth measured 2.4-4.1% of mean
    radius peak-to-peak on the test gears; tessellation faceting on a plain
    cylinder measured 0.01%, two orders of magnitude smaller. Returns None
    (no significant radial feature detected) below that gate, rather than a
    misleading count.
    """

    import numpy as np

    try:
        section = mesh.section(plane_origin=mesh.centroid, plane_normal=(0, 0, 1))
        if section is None:
            return None
        planar, _ = section.to_2D()
        pts = planar.vertices
    except Exception:
        return None
    if len(pts) < 8:
        return None

    centroid = pts.mean(axis=0)
    rel = pts - centroid
    theta = np.arctan2(rel[:, 1], rel[:, 0])
    radius = np.linalg.norm(rel, axis=1)
    order = np.argsort(theta)
    grid = np.linspace(-np.pi, np.pi, 720, endpoint=False)
    radius_interp = np.interp(grid, theta[order], radius[order], period=2 * np.pi)
    mean_r = radius_interp.mean()
    if mean_r <= 0:
        return None

    spectrum = np.abs(np.fft.rfft(radius_interp - mean_r)) / len(radius_interp)
    spectrum[:3] = 0  # DC and near-DC: overall shape, not a repeating feature
    spectrum[100:] = 0  # implausible tooth counts for a printable part
    dominant_freq = int(np.argmax(spectrum))
    dominant_amplitude = float(spectrum[dominant_freq])

    if dominant_amplitude / mean_r < _TOOTH_MIN_RELATIVE_AMPLITUDE:
        return None
    return dominant_freq


def _relative_error(measured: float, target: float) -> float:
    """Relative error, correctly handling a target of exactly 0.

    An earlier version used ``abs(measured - target) / target if target
    else 0.0`` — an adversarial review confirmed a spec stating an explicit
    0.0 dimension made that fall back to the truthiness branch and report
    zero error regardless of the actual measurement. A 0.0 target is a
    degenerate spec value (no real CAD dimension is legitimately zero), so
    any nonzero measurement against it is treated as maximally wrong.
    """

    if target > 1e-9:
        return abs(measured - target) / target
    return 0.0 if abs(measured) < 1e-9 else float("inf")


def _match_linear_dimensions(
    ext: list[float], stated: dict[str, float], tolerance_pct: float
) -> tuple[list[str], bool, int | None]:
    """Best-fit assignment of stated height/width/depth/length onto the 3 measured axes.

    Brute-forces every way to assign up to 3 stated linear dimensions to the
    3 measured bounding-box axes (at most 3! = 6 permutations to consider
    per combination, trivially cheap) and reports whichever assignment
    minimizes total relative error, rather than assuming a fixed axis
    mapping. See the module docstring for why: LLM-generated code isn't
    guaranteed to honor the "build on XY" prompt rule, and even when it
    does, a single stated in-plane dimension has no inherent reason to be
    the *smaller* of the two horizontal extents.

    Returns (detail_lines, all_assigned_pass, axis_index_used_for_height).
    A 4th (or later) stated linear dimension beyond the 3 measurable axes
    is reported explicitly as unverifiable rather than silently dropped.
    """

    # `is not None`, not truthiness -- `stated.get(k)` alone would repeat the
    # exact 0.0-as-absent bug _relative_error was written to fix, just moved
    # here. Caught by testing this function's own regression case, not by
    # reading the code a second time.
    names = [k for k in ("height", "width", "depth", "length") if stated.get(k) is not None]
    if not names:
        return [], True, None

    n = min(len(names), 3)
    best: tuple[float, tuple[int, ...], list[float]] | None = None
    for axis_combo in permutations(range(3), n):
        errs = [_relative_error(ext[axis_combo[i]], stated[names[i]]) for i in range(n)]
        total = sum(min(e, 10.0) for e in errs)  # cap so one inf doesn't NaN the comparison
        if best is None or total < best[0]:
            best = (total, axis_combo, errs)
    _, axis_combo, errs = best  # type: ignore[misc]

    lines = []
    height_axis = None
    for i in range(n):
        name = names[i]
        target = stated[name]
        m = ext[axis_combo[i]]
        err = errs[i]
        ok = err <= tolerance_pct
        lines.append(f"{name}: measured {m:.1f}mm vs stated {target:.1f}mm ({err*100:+.0f}%) {'OK' if ok else 'FAIL'} [best-fit axis]")
        if name == "height":
            height_axis = axis_combo[i]
    all_pass = all(errs[i] <= tolerance_pct for i in range(n))

    for name in names[3:]:
        lines.append(
            f"{name}: {stated[name]:.1f}mm stated but only 3 measurable axes exist and all are already "
            f"assigned to other stated dimensions — not independently verifiable"
        )
        all_pass = False

    return lines, all_pass, height_axis


def evaluate_leg1(
    execution: ExecutionResult,
    spec: DesignBrief,
    *,
    dim_tolerance_pct: float = 0.15,
) -> Leg1Result:
    stages: list[CheckStage] = []
    measured: dict[str, Any] = {}

    # --- Stage 1: execute -------------------------------------------------
    if not execution.ok:
        stages.append(
            CheckStage(
                name="execute",
                passed=False,
                detail=f"{execution.error_type}: {execution.error_message}",
            )
        )
        return _stop(stages, "execute", measured)
    stages.append(CheckStage(name="execute", passed=True, detail="executed and exported STL"))

    import trimesh  # local import: keeps this dependency out of the sandboxed subprocess

    mesh = trimesh.load(execution.stl_path, force="mesh")
    # Prefer the OCC-exact bounding box when we have it; fall back to the
    # mesh's own extents otherwise (e.g. no BRep provenance). Silently
    # defaulting to [0, 0, 0] here — an earlier version of this function did
    # exactly that — corrupts every downstream dimension and scale check for
    # any STL without OCC provenance, which is every externally-sourced
    # model this module is ever asked to score. Caught by testing against
    # the cad_grade corpus, not by reading the code.
    ext = (
        [execution.cq_bbox["x"], execution.cq_bbox["y"], execution.cq_bbox["z"]]
        if execution.cq_bbox
        else [float(v) for v in mesh.extents]
    )
    measured.update(bbox_x=float(ext[0]), bbox_y=float(ext[1]), bbox_z=float(ext[2]))

    # --- Stage 2: mesh validity ---------------------------------------------
    # Gated on exact BRep facts when we have them (every model cadyfiner
    # generates itself does, via execute.py). Some callers score STLs with
    # no such provenance — e.g. externally-sourced models, or the cad_grade
    # corpus used to regression-test this module, which ships compiled STLs
    # with no source code to re-derive BRep facts from. `None` must never be
    # silently coerced to "verified invalid" (bool(None) is False, but "we
    # don't know" and "the CAD kernel said no" are different claims and the
    # diagnostic text must say which one actually happened) — it falls back
    # to a clearly-labeled, lower-confidence trimesh approximation instead.
    if execution.cq_is_valid_brep is not None and execution.cq_n_solids is not None:
        brep_valid = bool(execution.cq_is_valid_brep)
        n_solids = execution.cq_n_solids
        single_solid = n_solids == 1
        measured.update(is_valid_brep=brep_valid, n_solids=n_solids, validity_source="occ_exact")
        mesh_valid = brep_valid and single_solid
        problems = []
        if not brep_valid:
            problems.append("CAD kernel reports this shape as an invalid solid (self-intersecting or non-manifold)")
        if not single_solid:
            problems.append(f"{n_solids} disconnected solid(s) in the result (expected exactly 1 — check for tangent, non-overlapping unions)")
        detail = "single valid solid (OCC-verified)" if mesh_valid else "; ".join(problems)
    else:
        single_solid = bool(mesh.body_count == 1)
        watertight_proxy = bool(mesh.is_watertight)
        measured.update(
            is_valid_brep=None,
            n_solids=int(mesh.body_count),
            validity_source="trimesh_approximation_no_occ_provenance",
        )
        mesh_valid = single_solid and watertight_proxy
        problems = []
        if not single_solid:
            problems.append(f"{mesh.body_count} disconnected mesh component(s) (expected 1)")
        if not watertight_proxy:
            problems.append("mesh not watertight")
        detail = (
            ("single connected watertight mesh (no BRep provenance — trimesh approximation, lower confidence)" if mesh_valid else "; ".join(problems) + " (no BRep provenance — trimesh approximation, lower confidence)")
        )

    stages.append(CheckStage(name="mesh_validity", passed=mesh_valid, detail=detail))
    if not mesh_valid:
        return _stop(stages, "mesh_validity", measured)
    # Prefer OCC's exact analytic volume/area (from the BRep) over trimesh's
    # tessellation-approximated ones when available; they're not derived
    # from a mesh at all, so they aren't sensitive to export tolerance.
    volume = execution.cq_volume if execution.cq_volume is not None else float(mesh.volume)
    area = execution.cq_area if execution.cq_area is not None else float(mesh.area)
    wall_proxy = 2 * volume / area if area > 0 else float("nan")
    n_holes_est = _estimate_through_holes(mesh)
    stl_watertight = bool(mesh.is_watertight)
    winding_ok = bool(mesh.is_winding_consistent)
    measured.update(
        volume=volume,
        area=area,
        wall_thickness_proxy_mm=wall_proxy,
        through_holes_est=n_holes_est,
        stl_watertight=stl_watertight,
        winding_consistent=winding_ok,
    )

    # --- Stage 3: spec conformance -----------------------------------------
    stated = spec.stated_dimensions()
    any_real_comparison = False  # tracks whether any comparison actually ran,
    # for both callers the module docstring warns about: the optimizer's own
    # accept/reject scoring of a refiner's emitted spec, and the exit-criteria
    # harness's raw-vs-refined comparison — an adversarial review found the
    # original "vacuously passed" label only fired when dim_checks/
    # feature_checks were BOTH entirely empty, so a spec whose every stated
    # field/feature happened to be unverifiable still silently passed without
    # even that label making the pass visible.

    linear_lines, linear_pass, height_axis = _match_linear_dimensions(ext, stated, dim_tolerance_pct)
    dim_checks = list(linear_lines)
    dim_pass = linear_pass
    if linear_lines:
        any_real_comparison = True

    if "diameter" in stated:
        target = stated["diameter"]
        # Diameter is a footprint measurement, not a height one. If a height
        # was assigned an axis above, exclude it from the candidate axes;
        # otherwise default to X/Y (index 0,1) — preserves the tested
        # behavior for the common diameter+height / diameter-alone cases
        # while still respecting an explicit non-Z height assignment.
        candidate_axes = [i for i in range(3) if i != height_axis] if height_axis is not None else [0, 1]
        m = max(ext[i] for i in candidate_axes)
        err = _relative_error(m, target)
        ok = err <= dim_tolerance_pct
        dim_pass &= ok
        any_real_comparison = True
        dim_checks.append(f"diameter: measured {m:.1f}mm vs stated {target:.1f}mm ({err*100:+.0f}%) {'OK' if ok else 'FAIL'}")

    if "thickness" in stated:
        target = stated["thickness"]
        err = _relative_error(wall_proxy, target)
        # wall_thickness_proxy is a whole-body average (2*Volume/Area), which
        # an adversarial review showed can be badly wrong for any part with
        # non-uniform wall thickness (e.g. a uniform-2mm-wall ring plus a
        # solid base measured 3.6mm average and false-FAILed; a shell with a
        # genuinely dangerous 0.3mm region next to a thick boss measured a
        # safe-looking 2.86mm average and never got flagged). Checked here
        # at a much wider tolerance and explicitly labeled low-confidence
        # rather than gated at the same precision as a direct measurement —
        # a real fix would need a localized thickness map (e.g. ray-casting
        # each mesh sample against its opposite face), not implemented here.
        proxy_tolerance = max(dim_tolerance_pct * 3, 0.5)
        ok = err <= proxy_tolerance
        dim_pass &= ok
        any_real_comparison = True
        dim_checks.append(
            f"wall thickness (proxy 2V/A, whole-body average, LOW CONFIDENCE): measured {wall_proxy:.1f}mm "
            f"vs stated {target:.1f}mm ({err*100:+.0f}%, checked at {proxy_tolerance*100:.0f}% tolerance) {'OK' if ok else 'FAIL'}"
        )

    feature_checks: list[str] = []
    feature_pass = True
    tooth_est: int | None = None
    tooth_est_computed = False
    # Every hole-kind feature is compared against the SAME whole-mesh
    # through-hole estimate, because the Euler-characteristic estimate
    # cannot attribute individual holes to individual feature kinds. An
    # adversarial review found the original code compared each hole-kind
    # feature's count against that same total independently — so a spec
    # stating e.g. one mounting_hole and one drainage_hole (2 total) could
    # never pass even on a perfect model, since neither individual count
    # (1) equals the true total (2). Fixed by summing all through-hole-kind
    # counts and checking the sum once. Blind (non-through) holes are
    # excluded from this branch entirely — the through-hole estimator's own
    # docstring says a blind hole contributes nothing to genus, so routing
    # them here would make any blind_hole spec unconditionally fail;
    # they're routed to the "not independently verifiable" branch instead.
    through_hole_feats = [
        f for f in spec.features if f.count is not None and "hole" in f.kind.lower() and "blind" not in f.kind.lower()
    ]
    if through_hole_feats:
        total_stated_holes = sum(f.count for f in through_hole_feats)
        ok = n_holes_est == total_stated_holes
        feature_pass &= ok
        any_real_comparison = True
        kinds = ", ".join(f"{f.kind}={f.count}" for f in through_hole_feats)
        feature_checks.append(
            f"through-hole features ({kinds}): estimated {n_holes_est} total through-hole(s) on the mesh vs "
            f"stated total {total_stated_holes} (approximate genus-based estimate, cannot attribute per-kind) "
            f"{'OK' if ok else 'FAIL'}"
        )

    for feat in spec.features:
        kind_lower = feat.kind.lower()
        if feat.count is not None and "hole" in kind_lower and "blind" not in kind_lower:
            continue  # already covered by the aggregate through-hole check above
        if feat.count is not None and ("tooth" in kind_lower or "teeth" in kind_lower or "spline" in kind_lower):
            if not tooth_est_computed:
                tooth_est = _estimate_tooth_count(mesh)
                tooth_est_computed = True
            if tooth_est is None:
                feature_checks.append(
                    f"feature '{feat.kind}': no significant radial feature detected (below noise-vs-signal amplitude gate), not evaluated"
                )
            else:
                # +/-1 tolerance: the FFT estimator was exact on 8/12/20-tooth
                # synthetic test gears but real generated geometry is noisier.
                ok = abs(tooth_est - feat.count) <= 1
                feature_pass &= ok
                any_real_comparison = True
                feature_checks.append(
                    f"feature '{feat.kind}': estimated {tooth_est} radial feature(s) vs stated count {feat.count} "
                    f"(approximate FFT-based estimate, +/-1 tolerance) {'OK' if ok else 'FAIL'}"
                )
        else:
            feature_checks.append(f"feature '{feat.kind}': not independently verifiable from mesh, not evaluated")

    conformance_pass = dim_pass and feature_pass
    # Always keep the itemized listing (including "not independently
    # verifiable" entries for individual features) rather than replacing it
    # wholesale when nothing was verified — an earlier version of this fix
    # discarded that itemization exactly when it mattered most (a spec with
    # several stated-but-unverifiable features), regressing on this
    # module's own "report as unverifiable, never silently drop" principle.
    if dim_checks or feature_checks:
        conformance_detail = "; ".join(dim_checks + feature_checks)
    else:
        conformance_detail = "no stated dimensions or features in spec at all"
    if not any_real_comparison:
        conformance_detail += " — NOTE: nothing above was independently verified; passed without real evaluation"
    measured["dim_checks"] = dim_checks
    measured["feature_checks"] = feature_checks
    measured["any_real_comparison"] = any_real_comparison
    stages.append(CheckStage(name="spec_conformance", passed=conformance_pass, detail=conformance_detail))
    if not conformance_pass:
        return _stop(stages, "spec_conformance", measured)

    # --- Stage 4: manufacturability (generic, always runs last) ------------
    manuf_problems = []
    max_dim = max(ext)
    if not (_MIN_SCALE_MM <= max_dim <= _MAX_SCALE_MM):
        manuf_problems.append(f"largest dimension {max_dim:.1f}mm outside sane range [{_MIN_SCALE_MM}, {_MAX_SCALE_MM}]mm")
    if not math.isnan(wall_proxy) and wall_proxy < _MIN_WALL_PROXY_MM:
        # This is a whole-body average (2*Volume/Area): an adversarial review
        # showed it can hide a genuinely thin, fragile region entirely when
        # unrelated thick geometry elsewhere in the same body pulls the
        # average up. It catches an obviously-too-thin BODY; it cannot catch
        # an obviously-too-thin LOCAL FEATURE. Not fixed here — a real fix
        # needs a localized thickness map, e.g. ray-casting each mesh sample
        # against its opposite face (trimesh.proximity can do this).
        manuf_problems.append(
            f"whole-body average wall-thickness proxy {wall_proxy:.2f}mm below printable floor {_MIN_WALL_PROXY_MM}mm "
            f"(coarse average — a thin local feature next to thick geometry elsewhere would not be caught by this check)"
        )
    degenerate = int((mesh.area_faces < 1e-9).sum())
    degenerate_frac = degenerate / max(len(mesh.faces), 1)
    if degenerate_frac > _MAX_DEGENERATE_FACE_FRACTION:
        manuf_problems.append(f"{degenerate}/{len(mesh.faces)} degenerate faces ({degenerate_frac*100:.1f}%)")
    if not stl_watertight or not winding_ok:
        # A design OCC considers a valid single solid can still export with
        # tessellation-seam cracks at curved-face boundaries — not a design
        # defect, but a real print-prep concern (a slicer sees the STL, not
        # the BRep), so it's flagged here rather than in mesh_validity.
        manuf_problems.append(
            "exported STL has tessellation-seam gaps (not watertight or inconsistent winding) — "
            "likely needs mesh repair before printing even though the underlying solid is valid"
        )
    measured["degenerate_faces"] = degenerate

    manuf_pass = not manuf_problems
    stages.append(
        CheckStage(
            name="manufacturability",
            passed=manuf_pass,
            detail="within sane printability bounds" if manuf_pass else "; ".join(manuf_problems),
        )
    )

    return Leg1Result(
        overall_pass=manuf_pass,
        stopped_at="manufacturability",
        stages=stages,
        measured=measured,
    )
