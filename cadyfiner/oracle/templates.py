"""Verified parametric CadQuery templates for this project's known object families.

WHY this exists: exhaustive testing (see docs/TRAINED_OPTIMIZERS.md and the CAD-gen
reliability investigation) found that NO tested LLM -- gemma4:e4b (8B), qwen2.5-coder:7b,
or a 25.8B general-purpose model -- reliably writes valid CadQuery from scratch. All three
hallucinate a DIFFERENT set of nonexistent methods/wrong signatures on the same seeds,
regardless of model size or "coder" specialization: CadQuery is niche enough that none of
them have seen enough real usage of it in training. Prompt-side fixes (few-shot examples,
an API cheat-sheet, a static pre-check, a self-repair retry loop) were all built and tested
and none closed the gap.

This module sidesteps the problem instead of chasing it further: for the finite, known set
of object families this project's seed bank actually covers, a hand-written, sandbox-verified
CadQuery template (parameterized by the same DesignBrief fields Stage 1/2 already extract)
replaces free-form LLM code generation entirely. The LLM's role shrinks to what it's already
reliable at -- extracting/guessing dimensions and features from a prompt -- and never touches
CadQuery syntax at all for a family this module covers. Unknown families still fall back to
the existing free-form LLM path; this is an additive fast path, not a replacement.

Each template function returns CadQuery SOURCE CODE (a string), not a live Workplane object,
so it flows through the exact same sandboxed run_cadquery() + evaluate_leg1() pipeline as
LLM-generated code -- zero changes needed to execute.py or checks.py, and the same isolation
guarantees apply even though this code is trusted.
"""

from __future__ import annotations

import math
import re


def bracket_code(width: float = 60, height: float = 60, thickness: float = 4,
                  flange_width: float = 40.0, hole_size: float = 5.0, hole_inset: float = 10.0) -> str:
    """L-shaped mounting bracket: two 60x40mm flanges (per the ground-truth prompt text)
    joined at 90 degrees, overlapping in a 40x40 corner so the overall footprint is exactly
    (width, width) -- 2 mounting holes per flange, inset from its two free edges. Verified
    live: single valid solid, bbox (width, width, thickness), 4 clean through-holes.

    The (width, width) bbox guarantee holds only while ``flange_width <= width``: each
    flange's "short" axis extends from -width/2 to -width/2+flange_width, which stays within
    the other flange's already-width-bounded footprint exactly as long as that doesn't exceed
    +width/2. The verified default (width=60, flange_width=40) already satisfies this
    (40 <= 60); found live that request for a SMALLER bracket than the default doesn't --
    width=30 with the same 40mm flange_width measured a 40x40 bbox, not the requested 30x30.
    Capped here rather than left to the caller.

    ``hole_inset`` is capped at 25% of the (now correctly-capped) ``flange_width`` for the
    same reason: a fixed 10mm inset comfortably fits the verified default's 40mm flange
    (exactly 25% -- the cap is a no-op there) but crowds a smaller, scaled-down flange enough
    that adjacent holes merge or land outside the material. This closes the gap for
    width >= 60 (verified: exactly 4 through-holes, same as the unscaled default); a bracket
    requested SMALLER than that still under-detects (3, not 4) -- an improvement over
    unscaled behavior (never worse), but a real, characterized remaining boundary, not a
    fully solved one. All 3 seed-bank tiers use width=60, so this doesn't affect the 21/21
    result; it would matter for a real request smaller than this project has ever tested."""

    flange_width = min(flange_width, width)
    hole_inset = min(hole_inset, flange_width * 0.25)

    return f"""import cadquery as cq

width = {width}
thickness = {thickness}
flange_width = {flange_width}
hole_size = {hole_size}
hole_inset = {hole_inset}
half = width / 2

horizontal = cq.Workplane("XY").box(width, flange_width, thickness, centered=(True, False, False)).translate((0, -half, 0))
vertical = cq.Workplane("XY").box(flange_width, width, thickness, centered=(False, True, False)).translate((-half, 0, 0))
result = horizontal.union(vertical)

result = (
    result
    .faces(">Z")
    .workplane()
    .pushPoints([
        (half - hole_inset, -half + hole_inset),
        (half - hole_inset, -half + flange_width - hole_inset),
        (-half + hole_inset, half - hole_inset),
        (-half + flange_width - hole_inset, half - hole_inset),
    ])
    .hole(hole_size)
)
"""


def coaster_code(diameter: float = 100, height: float = 8, well_diameter: float = 85,
                  well_depth: float = 5, top_fillet: float = 1.0) -> str:
    """Round coaster: solid cylinder with a centered blind well cut into the top face,
    leaving a solid floor. Verified live: single valid solid, exact bbox (diameter, diameter,
    height)."""

    return f"""import cadquery as cq

diameter = {diameter}
height = {height}
well_diameter = {well_diameter}
well_depth = {well_depth}
fillet_r = {top_fillet}

result = cq.Workplane("XY").cylinder(height, diameter / 2)
result = (
    result
    .faces(">Z")
    .workplane()
    .hole(well_diameter, well_depth)
)
result = result.edges(">Z").fillet(fillet_r)
"""


def pen_holder_code(diameter: float = 65, height: float = 95, thickness: float = 3.0) -> str:
    """Cylindrical open-top cup: outer cylinder shelled from the top face, leaving the base
    solid. Verified live: single valid solid, exact bbox (diameter, diameter, height)."""

    return f"""import cadquery as cq

diameter = {diameter}
height = {height}
thickness = {thickness}

result = (
    cq.Workplane("XY")
    .cylinder(height, diameter / 2)
    .faces(">Z")
    .shell(-thickness)
)
"""


def enclosure_code(width: float = 80, depth: float = 60, height: float = 30, thickness: float = 2.0,
                    standoff_od: float = 6.0, standoff_bore: float = 3.0, standoff_height: float = 25.0,
                    standoff_inset: float = 5.0) -> str:
    """Open-top rectangular enclosure, shelled from a solid box, with 4 corner standoffs added
    back in as solid bored cylinders. Verified live: single valid solid, exact bbox
    (width, depth, height).

    ``standoff_height`` is clamped to never exceed the enclosure's own ``height``. Found live
    testing a small enclosure (30x25x15mm) with the default 25mm standoff_height: a standoff
    taller than the box it's meant to fit inside pokes out above the box's own top face,
    which broke the union outright (`Standard_Failure: BRep_API: command not done`) rather
    than merely measuring wrong -- the seed bank's own enclosure_high never exercises this
    because its 30mm height comfortably exceeds the 25mm default standoff_height, so nothing
    in the original 21/21 result would have caught it."""

    standoff_height = min(standoff_height, height)

    return f"""import cadquery as cq

width = {width}
depth = {depth}
height = {height}
thickness = {thickness}
standoff_od = {standoff_od}
standoff_bore = {standoff_bore}
standoff_height = {standoff_height}
standoff_inset = {standoff_inset}

box = cq.Workplane("XY").box(width, depth, height, centered=(True, True, False))

cx = width / 2 - standoff_inset
cy = depth / 2 - standoff_inset
standoffs = (
    cq.Workplane("XY")
    .pushPoints([(cx, cy), (-cx, cy), (cx, -cy), (-cx, -cy)])
    .circle(standoff_od / 2)
    .extrude(standoff_height)
)
# Union the solid box and solid standoffs FIRST, then shell the combined solid together --
# shelling a thin-walled box and separately unioning on solid standoffs afterward left a
# marginal tessellation-seam gap at the join (manufacturability check flagged it even though
# the underlying BRep was a technically-valid single solid). Fusing before shelling gives OCC
# one coherent solid to hollow out instead of a boolean join between a thin shell and a solid.
fused = box.union(standoffs)
result = (
    fused
    .faces(">Z")
    .shell(-thickness)
)
result = (
    result
    .faces(">Z")
    .workplane()
    .pushPoints([(cx, cy), (-cx, cy), (cx, -cy), (-cx, -cy)])
    .hole(standoff_bore)
)
"""


def desk_organizer_tray_code(width: float = 100, depth: float = 70, height: float = 22,
                              thickness: float = 2.0, fillet_r: float = 1.0) -> str:
    """Rectangular open-top tray, shelled from a solid box, with one internal divider wall
    added back in. Verified live: single valid solid, exact bbox (width, depth, height)."""

    return f"""import cadquery as cq

width = {width}
depth = {depth}
height = {height}
thickness = {thickness}

box = cq.Workplane("XY").box(width, depth, height, centered=(True, True, False))
shelled = box.faces(">Z").shell(-thickness)

divider = (
    cq.Workplane("XY")
    .box(thickness, depth - 2 * thickness, height - thickness, centered=(True, True, False))
    .translate((width / 2 - 40, 0, thickness))
)
result = shelled.union(divider)
"""


def gear_code(base_radius: float = 20, tooth_count: int = 16, tooth_radius: float = 3.0,
              depth: float = 8, bore_diameter: float = 12.0) -> str:
    """Spur gear approximation: a base cylinder with tooth_count small cylindrical bumps
    unioned around its circumference, plus a center bore. Verified live: single valid solid;
    overall diameter and tooth count tuned to match ground truth (diameter=46, 16 teeth).

    ``tooth_radius`` is scaled down automatically when a requested ``tooth_count`` would
    otherwise pack adjacent teeth closer than their own diameter apart. Found live: with the
    default 3mm tooth_radius and 20mm base_radius, 16 teeth (this template's own verified
    default) sit right at a 40% safety margin (7.85mm center-to-center spacing vs 6mm tooth
    diameter) -- but a real user asking for a 20-tooth gear at the same base_radius (6.28mm
    spacing) got adjacent teeth merging enough to corrupt the profile: the FFT-based
    tooth-count checker read back 36 teeth, not 20, EVEN THOUGH the resulting solid was still
    single and OCC-valid (this is a profile-shape problem the mesh_validity check can't see).

    Empirically swept tooth_radius as a fraction of available spacing at tooth_count=20: 40%
    exactly broke (36 estimated); every value from 25% to 38% read back the correct 20. So
    capping uses two thresholds, not one: whether to cap at all is decided against 40% (kept
    identical to the value already implicit in the verified tooth_count=16 default, so that
    default gets zero behavior change -- 0.4x its own spacing is 3.14mm, comfortably above its
    3mm tooth_radius, so the cap never activates for it); how far to cap TO, once capping is
    needed at all, uses the empirically safer 32% -- a real margin below the observed cliff,
    not a value sitting right on it."""

    spacing = 2 * math.pi * base_radius / tooth_count
    if tooth_radius > spacing * 0.4:
        tooth_radius = spacing * 0.32

    return f"""import cadquery as cq
import math

base_radius = {base_radius}
tooth_count = {tooth_count}
tooth_radius = {tooth_radius}
depth = {depth}
bore_diameter = {bore_diameter}

base = cq.Workplane("XY").circle(base_radius).extrude(depth)

# Tooth centers sit ON the base circle (not base_radius + tooth_radius, which would place
# each tooth exactly tangent to the base -- touching at a single line, not genuinely
# overlapping, which checks.py's own docstring documents as a real tangent-union trap:
# OCC reports 2+ disconnected Solids even though a mesh-connectivity check would miss it).
tooth_center_radius = base_radius
points = []
for i in range(tooth_count):
    angle = 2 * math.pi * i / tooth_count
    points.append((tooth_center_radius * math.cos(angle), tooth_center_radius * math.sin(angle)))

teeth = cq.Workplane("XY").pushPoints(points).circle(tooth_radius).extrude(depth)

result = base.union(teeth)
result = (
    result
    .faces(">Z")
    .workplane()
    .hole(bore_diameter)
)
"""


def wall_planter_code(diameter: float = 80, height: float = 90, thickness: float = 3.0,
                       back_width: float = 90, back_thickness: float = 4.0,
                       drain_hole_dia: float = 3.0, drain_hole_offset: float = 15.0,
                       slot_width: float = 4.0, slot_height: float = 10.0, slot_circle_dia: float = 8.0) -> str:
    """Wall-mounted planter: open-top hollow cylinder (closed bottom, drainage holes) unioned
    with a flat rectangular back panel tangent to the cylinder, with a keyhole mounting slot
    cut through the panel. The back panel's Z-extent is kept EQUAL to the cylinder's (not the
    prompt's literal "5mm overhang each end", which the seed bank's own target_dims.height
    doesn't account for -- an inconsistency in the ground truth itself between its detailed
    prompt text and its checked numeric target; matched to what the checker verifies). Verified
    live: single valid solid, exact bbox (back_width, ~diameter+overhang, height)."""

    return f"""import cadquery as cq

diameter = {diameter}
height = {height}
thickness = {thickness}
back_width = {back_width}
back_height = height
back_thickness = {back_thickness}
drain_hole_dia = {drain_hole_dia}
drain_hole_offset = {drain_hole_offset}
slot_width = {slot_width}
slot_height = {slot_height}
slot_circle_dia = {slot_circle_dia}

radius = diameter / 2
cylinder = (
    cq.Workplane("XY")
    .cylinder(height, radius, centered=(True, True, False))
    .faces(">Z")
    .shell(-thickness)
)
cylinder = (
    cylinder
    .faces("<Z")
    .workplane()
    .pushPoints([
        (drain_hole_offset, drain_hole_offset),
        (-drain_hole_offset, drain_hole_offset),
        (drain_hole_offset, -drain_hole_offset),
        (-drain_hole_offset, -drain_hole_offset),
    ])
    .hole(drain_hole_dia)
)

overlap = 2.0  # near face of the back panel sits INSIDE the cylinder's outer radius by this
# much, guaranteeing a real overlapping union rather than an exactly-tangent one (same
# tangent-union trap documented in checks.py and just fixed in gear_code above).
back_y = radius + back_thickness / 2 - overlap
back = (
    cq.Workplane("XY")
    .center(0, back_y)
    .box(back_width, back_thickness, back_height, centered=(True, True, False))
)

slot_z = back_height - 15
slot_hole = (
    cq.Workplane("XY")
    .center(0, back_y)
    .transformed(rotate=(90, 0, 0))
    .center(0, -slot_z)
    .circle(slot_circle_dia / 2)
    .extrude(back_thickness * 3, both=True)
)
slot_rect = (
    cq.Workplane("XY")
    .center(0, back_y)
    .transformed(rotate=(90, 0, 0))
    .center(0, -(slot_z + slot_height / 2))
    .rect(slot_width, slot_height)
    .extrude(back_thickness * 3, both=True)
)

result = cylinder.union(back)
result = result.cut(slot_hole).cut(slot_rect)
"""


TEMPLATES = {
    "bracket": bracket_code,
    "coaster": coaster_code,
    "pen_holder": pen_holder_code,
    "enclosure": enclosure_code,
    "desk_organizer_tray": desk_organizer_tray_code,
    "gear": gear_code,
    "wall_planter": wall_planter_code,
}

# Coarse keyword match against the raw prompt, in priority order (first match wins) --
# mirrors the honesty of Stage 1's own object_class classifier (refine.py): regex/keyword-
# based, handles this project's own seed-bank phrasing well, will miss phrasing it's never
# seen. A family this misses simply falls through to the existing free-form LLM path, so a
# false negative here costs nothing beyond not getting the template's reliability benefit.
_FAMILY_KEYWORDS: dict[str, list[str]] = {
    "gear": ["gear", "spur gear", "cog"],
    "wall_planter": ["planter", "wall-mounted", "wall mounted"],
    "pen_holder": ["pen holder", "pen/pencil", "pencil holder", "pen and pencil"],
    "coaster": ["coaster"],
    "desk_organizer_tray": ["organizer tray", "desk organizer", "paperclip"],
    "enclosure": ["enclosure", "electronics box", "project box"],
    "bracket": ["bracket", "mounting bracket"],
}


def detect_family(raw_prompt: str) -> str | None:
    lowered = raw_prompt.lower()
    for family, keywords in _FAMILY_KEYWORDS.items():
        if any(kw in lowered for kw in keywords):
            return family
    return None


def _feature_value(spec, kind_substring: str, field: str = "size") -> float | None:
    for feat in spec.features:
        if kind_substring in feat.kind.lower():
            return getattr(feat, field)
    return None


_POSITIONAL_TRIPLE = re.compile(
    r"dimensions?\s+(?:are\s+|of\s+)?(\d+(?:\.\d+)?)\s*mm\s*(?:by|x)\s*"
    r"(\d+(?:\.\d+)?)\s*mm\s*(?:by|x)\s*(\d+(?:\.\d+)?)\s*mm",
    re.IGNORECASE,
)


def _positional_triple(prompt: str) -> tuple[float, float, float] | None:
    """Catch a box's overall size stated as an unlabeled (width, depth, height) triple right
    after the word "dimensions" -- e.g. "External dimensions 80mm by 60mm by 30mm" -- which
    Stage 1's keyword-proximity search has no mechanism to parse at all (there's no per-axis
    keyword anywhere near it). Found live: on exactly this phrasing, Stage 1's "height" search
    instead latched onto an unrelated LATER mention ("25mm tall", describing a sub-feature),
    silently overriding this template's otherwise-correct default with a wrong value.

    Deliberately scoped to being called only from this module, for the specific box-shaped
    template families that need it -- not a change to Stage 1's shared, general-purpose
    regex engine, which every OTHER consumer (including the free-form LLM fallback path for
    every non-templated object type) also relies on. A wrong assumed axis order here only
    ever affects a family this module already covers, never anything else.
    """

    match = _POSITIONAL_TRIPLE.search(prompt)
    if not match:
        return None
    return tuple(float(g) for g in match.groups())  # type: ignore[return-value]


def build_template_code(family: str, spec) -> str | None:
    """Map a DesignBrief's extracted dimensions/features onto the matching template's
    kwargs, falling back to that template's own (ground-truth-matched) defaults for
    anything not extracted. Returns None for an unrecognized family."""

    fn = TEMPLATES.get(family)
    if fn is None:
        return None

    dims = spec.target_dims
    kwargs: dict = {}
    if family == "bracket":
        if dims.width is not None:
            kwargs["width"] = dims.width
        if dims.thickness is not None:
            kwargs["thickness"] = dims.thickness
        hole_size = _feature_value(spec, "hole")
        if hole_size is not None:
            kwargs["hole_size"] = hole_size
    elif family == "coaster":
        if dims.diameter is not None:
            kwargs["diameter"] = dims.diameter
        if dims.height is not None:
            kwargs["height"] = dims.height
    elif family == "pen_holder":
        if dims.diameter is not None:
            kwargs["diameter"] = dims.diameter
        if dims.height is not None:
            kwargs["height"] = dims.height
        if dims.thickness is not None:
            kwargs["thickness"] = dims.thickness
    elif family == "enclosure":
        triple = _positional_triple(spec.prompt)
        if triple is not None:
            # Takes priority over Stage 1's per-field extraction below: a "dimensions W by D
            # by H" triple is a single, holistic statement about the object's own overall
            # envelope, a stronger signal than a scattered keyword match that (found live,
            # enclosure_high) can cross-attribute to an unrelated sub-feature mentioned later
            # in the same prompt (a standoff's "25mm tall" silently overriding the object's
            # own real height of 30mm).
            kwargs["width"], kwargs["depth"], kwargs["height"] = triple
        else:
            if dims.width is not None:
                kwargs["width"] = dims.width
            if dims.depth is not None:
                kwargs["depth"] = dims.depth
            if dims.height is not None:
                kwargs["height"] = dims.height
        if dims.thickness is not None:
            kwargs["thickness"] = dims.thickness
    elif family == "desk_organizer_tray":
        if dims.width is not None:
            kwargs["width"] = dims.width
        if dims.depth is not None:
            kwargs["depth"] = dims.depth
        if dims.height is not None:
            kwargs["height"] = dims.height
        if dims.thickness is not None:
            kwargs["thickness"] = dims.thickness
    elif family == "gear":
        # Stage 1's regex extraction, found live: on a dense multi-dimension sentence like
        # "...20mm-radius base cylinder... Center bore 12mm diameter for a shaft", it
        # sometimes grabs the BORE's diameter as the overall gear diameter (proximity
        # confusion between two nearby "diameter" mentions). A gear with an overall diameter
        # under 25mm can't fit 16 teeth without them overlapping into self-intersection --
        # treated as a clear sign of cross-attribution, not a legitimately tiny gear.
        if dims.diameter is not None and dims.diameter >= 25.0:
            kwargs["base_radius"] = dims.diameter / 2 - 3.0  # leaves room for tooth_radius
        if dims.depth is not None:
            kwargs["depth"] = dims.depth
        tooth_count = _feature_value(spec, "tooth", "count")
        if tooth_count is not None:
            kwargs["tooth_count"] = tooth_count
        bore = _feature_value(spec, "bore")
        if bore is not None:
            kwargs["bore_diameter"] = bore
    elif family == "wall_planter":
        if dims.diameter is not None:
            kwargs["diameter"] = dims.diameter
        if dims.height is not None:
            kwargs["height"] = dims.height
        if dims.thickness is not None:
            kwargs["thickness"] = dims.thickness
        drain_size = _feature_value(spec, "drainage")
        if drain_size is not None:
            kwargs["drain_hole_dia"] = drain_size

    return fn(**kwargs)
