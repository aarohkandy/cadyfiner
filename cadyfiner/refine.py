"""The prompt refiner: raw user prompt in, expanded prompt + structured spec out.

Two stages, per the design's core architectural decision:

Stage 1 (this module, ``extract``): deterministic, regex-based slot
extraction — units, dimensions, and feature counts found near their naming
keywords in the raw prompt, plus a coarse decorative-vs-mechanical
classification. Pattern (keyword-proximate number search) generalized from
``ai-cad/backend/app/services/planners/rule_based_planner.py``'s narrower
version, which only handled 3 keywords for 5 hardcoded object kinds — the
underlying technique is reusable, the code wasn't.

Stage 2 (``fill_gaps``): a single constrained LLM call fills missing slots.
*How much* it fills is a policy parameter keyed by ``object_class``, not a
universal constant — deliberately, per the architecture's core finding:
"stop before exact-placement detail" is derived entirely from the
wall_planter ladder (a decorative object) and the mechanical pilot
(``prompts/seed_bank/pilot_mechanical/``) exists specifically to check
whether that same policy helps or hurts brackets/gears/enclosures before
it gets hard-coded here.

Every value Stage 2 invents rather than reads from the prompt is logged to
``DesignBrief.assumptions_made`` — the refiner should never be silently
opinionated about a dimension or feature the user never mentioned.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from cadyfiner.spec import DesignBrief, Feature, TargetDimensions

_UNIT_CONVERSION_TO_MM = {
    "mm": 1.0, "millimeter": 1.0, "millimeters": 1.0,
    "cm": 10.0, "centimeter": 10.0, "centimeters": 10.0,
    "m": 1000.0, "meter": 1000.0, "meters": 1000.0,
    "in": 25.4, "inch": 25.4, "inches": 25.4,
}
# Longest-first order so alternation tries "millimeters" before "millimeter" before... down to the
# bare abbreviations — matters because regex alternation picks the first alternative that matches
# at a position, not the longest overall, and a trailing \b alone isn't quite enough insurance once
# spelled-out forms share prefixes with each other (though it IS enough against the abbreviations,
# since e.g. "in" can never satisfy \b while still inside "inch").
_UNIT_NAMES = (
    "millimeters", "millimeter", "centimeters", "centimeter", "meters", "meter",
    "inches", "inch", "mm", "cm", "in", "m",
)
_UNIT_ALTERNATION = "|".join(_UNIT_NAMES)

# Adversarial review found two real bugs sharing this one root cause: (1) with no trailing \b, "10
# millimeters" prefix-matched unit="m" (meters) out of "millimeters", a 1000x error — fixed by
# adding spelled-out forms (above) AND a trailing \b so a real prefix match still can't stick if a
# letter continues past it; (2) _UNIT_PATTERN below required \b on BOTH sides, but a digit
# immediately before a unit letter ("30cm") is not a \b transition (both count as \w to Python's
# regex engine), so the compact style — the most common way units are actually written — was
# invisible to default-unit detection. Fixed by allowing a digit (or nothing) before the unit,
# excluding only a preceding letter, since \b already blocks matching inside a longer word from the
# right side (e.g. "in" inside "inside" or "inch").
_UNIT_PATTERN = re.compile(rf"(?<![a-zA-Z])({_UNIT_ALTERNATION})\b", re.IGNORECASE)
# Value group order matters: a fraction ("1/2") must be tried before a bare decimal, or the bare
# decimal alternative matches just the numerator and stops there.
_NUMBER_NEAR_UNIT = re.compile(
    rf"(?P<value>\d+\s*/\s*\d+|\d+(?:[.,]\d+)?)\s*(?P<unit>{_UNIT_ALTERNATION})?\b",
    re.IGNORECASE,
)


def _parse_value(raw: str) -> float:
    """Parse a matched numeric string: plain decimal, or a simple fraction like '1/2'.

    European decimal-comma ('80,5') is normalized to a decimal point; a
    genuine thousands-separator comma essentially never appears in a CAD
    dimension (nobody writes "1,000mm" meaning one meter in a design
    prompt), so no attempt is made to distinguish the two — this was an
    explicit, documented tradeoff, not an oversight.
    """

    raw = raw.strip()
    if "/" in raw:
        num, _, denom = raw.partition("/")
        return float(num.strip()) / float(denom.strip())
    return float(raw.replace(",", "."))

# Keywords searched near a dimension name, tagged with where the number
# normally falls relative to the word. Noun forms ("diameter", "thickness")
# can go either way in English ("diameter 80mm" / "80mm diameter") but
# lean forward; adjective forms ("tall", "wide", "thick") are used almost
# exclusively as "<number> <unit> <adjective>" ("90mm tall") — the number
# precedes. Getting this backwards was a real, confirmed bug: "80 mm
# outer diameter" (noun, forward) worked, but "90 mm tall" (adjective,
# should be backward) matched a stray forward number from later in the
# same sentence instead of the correct value immediately before it.
# Order matters within a field's tuple: "diameter" is checked before
# "width"/"wide" so a round object's diameter isn't misread as its width.
_DIMENSION_KEYWORDS: dict[str, tuple[tuple[str, str], ...]] = {
    "diameter": (("diameter", "forward"), ("across", "backward"), ("dia.", "forward")),
    "height": (("height", "forward"), ("tall", "backward"), ("high", "backward")),
    "width": (("width", "forward"), ("wide", "backward")),
    "depth": (("depth", "forward"), ("deep", "backward")),
    "length": (("length", "forward"), ("long", "backward")),
    "thickness": (("thickness", "forward"), ("thick", "backward"), ("wall", "forward")),
}

# "radius" isn't a DesignBrief slot (diameter is), so it's handled separately
# with a x2 multiplier applied at the call site — common in gear/round-part
# phrasing ("20mm-radius base cylinder") and otherwise invisible to a
# diameter-only keyword search.
_RADIUS_KEYWORDS = (("radius", "forward"),)

_MECHANICAL_KEYWORDS = (
    "bracket", "gear", "tooth", "teeth", "enclosure", "bolt", "screw", "shaft",
    "mount", "mounting", "fitting", "pipe", "flange", "bearing", "spline",
    "standoff", "housing", "chassis", "fastener", "keyway", "bore",
)
_DECORATIVE_KEYWORDS = (
    "planter", "vase", "ornament", "figurine", "snowman", "decoration",
    "sculpture", "hook", "holder", "toy",
)

# Regression: the generic filler group ((?:[\w-]+\s+)?) used to appear BEFORE the kind
# alternation and, being tried first, greedily consumed the kind word itself (e.g. "mounting" in "4
# mounting holes") — so kind was None for essentially every real instance of the phrasing this
# pattern exists to distinguish, collapsing every hole to the generic "hole" kind. Fixed by trying
# the kind alternation first; the filler group still exists after it for phrasing like "4 small
# mounting holes" (though it will lose the kind word specifically when a qualifier separates it
# from "holes" — a narrower, rarer case than the original bug, not chased further here).
_HOLE_COUNT_PATTERN = re.compile(
    r"(?P<count>\d+)\s+(?:(?P<kind>mounting|drainage|screw|bolt|pilot)\s+)?(?:[\w-]+\s+)?holes?",
    re.IGNORECASE,
)
_TOOTH_COUNT_PATTERN = re.compile(r"(?P<count>\d+)\s*(?:teeth|-tooth)", re.IGNORECASE)


def _nearest_number_mm(
    text: str, keyword_index: int, keyword_len: int, default_unit: str, prefer: str
) -> float | None:
    """Find the number belonging to a keyword occurrence: whichever direction is closer wins.

    Tried a fixed "preferred direction first, other direction as fallback"
    order first; it still failed on real text like "80 mm outer diameter,
    with a 3 mm wall thickness" — forward-from-"diameter" finds *something*
    (the wrong "3", from the unrelated following clause) before ever
    considering the backward match, so the fallback never triggers even
    though the correct "80" is closer, just on the other side. Switched to
    picking whichever direction's match is fewer characters from the
    keyword, with ``prefer`` only used to break exact ties (rare, and
    harmless either way). A symmetric window without this distance
    comparison was the original bug this replaced: it grabbed unrelated
    numbers that merely happened to be nearby (verified against real
    prompt text while building this module).
    """

    forward_text = text[keyword_index + keyword_len : keyword_index + keyword_len + 20]
    backward_text = text[max(0, keyword_index - 20) : keyword_index]

    forward_match = _NUMBER_NEAR_UNIT.search(forward_text)
    backward_matches = list(_NUMBER_NEAR_UNIT.finditer(backward_text))
    backward_match = backward_matches[-1] if backward_matches else None  # closest = last one before the keyword

    forward_dist = forward_match.start() if forward_match else None
    backward_dist = (len(backward_text) - backward_match.end()) if backward_match else None

    if forward_dist is None and backward_dist is None:
        return None
    if backward_dist is None or (forward_dist is not None and forward_dist < backward_dist):
        chosen = forward_match
    elif forward_dist is None or backward_dist < forward_dist:
        chosen = backward_match
    else:
        chosen = forward_match if prefer == "forward" else backward_match

    value = _parse_value(chosen.group("value"))
    unit = (chosen.group("unit") or default_unit).lower()
    return value * _UNIT_CONVERSION_TO_MM.get(unit, 1.0)


def _detect_default_unit(prompt: str) -> str:
    match = _UNIT_PATTERN.search(prompt)
    return match.group(1).lower() if match else "mm"


def _keyword_regex(kw: str) -> re.Pattern:
    """A word-boundary-safe search pattern for one keyword.

    Regression: plain substring search (``kw in lowered`` / ``lowered.find(kw)``)
    matched a keyword inside an unrelated longer word — "height" inside
    "heightened", "mount" inside a word that happens to contain it — fabricating
    dimensions or misclassifying the object from text that never actually said
    what was matched. A trailing ``\\b`` is skipped for keywords ending in
    punctuation (e.g. "dia.") since a boundary can never hold between two
    non-word characters (the period and whatever follows it).
    """

    trailing = r"" if kw.endswith(".") else r"\b"
    return re.compile(r"\b" + re.escape(kw) + trailing, re.IGNORECASE)


_MECHANICAL_PATTERN = re.compile(r"\b(?:" + "|".join(re.escape(k) for k in _MECHANICAL_KEYWORDS) + r")\b", re.IGNORECASE)
_DECORATIVE_PATTERN = re.compile(r"\b(?:" + "|".join(re.escape(k) for k in _DECORATIVE_KEYWORDS) + r")\b", re.IGNORECASE)


def _classify_object(prompt: str) -> str:
    """Ties (including zero hits either way) resolve to mechanical_functional.

    Deliberate: DEPTH_POLICY's mechanical_functional category list is a
    strict superset of decorative's (it adds topology/feature_placement).
    On genuine ambiguity, erring toward the fuller policy risks a bit of
    over-specification at worst; erring toward the bounded one risks
    under-specifying the literal functional requirement of a part that
    actually needed exact dimensions/placement — the worse failure mode
    per this project's own architecture (see refine_stage2.py's
    DEPTH_POLICY docstring).
    """

    mech_hits = len(_MECHANICAL_PATTERN.findall(prompt))
    decor_hits = len(_DECORATIVE_PATTERN.findall(prompt))
    return "decorative" if decor_hits > mech_hits else "mechanical_functional"


@dataclass
class ExtractionResult:
    spec: DesignBrief
    coverage: dict[str, bool]  # which dimension slots were actually found in the text


def extract(raw_prompt: str) -> ExtractionResult:
    """Stage 1: deterministic slot extraction from raw prompt text.

    Approximate by nature (regex proximity search, not real NLP) — reports
    what it found via ``coverage`` so Stage 2 knows what's genuinely missing
    versus what it's about to invent.
    """

    default_unit = _detect_default_unit(raw_prompt)

    dims: dict[str, float] = {}
    coverage: dict[str, bool] = {}
    for field, keywords in _DIMENSION_KEYWORDS.items():
        if field == "width" and "diameter" in dims:
            continue  # already claimed by diameter; don't double-read the same number as width
        found = False
        for kw, prefer in keywords:
            match = _keyword_regex(kw).search(raw_prompt)
            if match is None:
                continue
            value = _nearest_number_mm(raw_prompt, match.start(), len(match.group(0)), default_unit, prefer)
            if value is not None:
                dims[field] = value
                found = True
                break
        coverage[field] = found

    if "diameter" not in dims:
        for kw, prefer in _RADIUS_KEYWORDS:
            match = _keyword_regex(kw).search(raw_prompt)
            if match is None:
                continue
            value = _nearest_number_mm(raw_prompt, match.start(), len(match.group(0)), default_unit, prefer)
            if value is not None:
                dims["diameter"] = value * 2
                coverage["diameter"] = True
                break

    features: list[Feature] = []
    for match in _HOLE_COUNT_PATTERN.finditer(raw_prompt):
        kind = match.group("kind")
        kind_name = f"{kind.lower()}_hole" if kind else "hole"
        features.append(Feature(kind=kind_name, count=int(match.group("count"))))
    # Regression: "24-tooth" and "24 teeth" appearing in the same prompt (a single fact stated
    # twice, common when a spec repeats itself in prose) each matched separately, producing two
    # Feature entries for what is normally one real tooth count. A gear has one tooth count, so
    # collapse to unique counts rather than one Feature per regex match.
    tooth_counts = {int(match.group("count")) for match in _TOOTH_COUNT_PATTERN.finditer(raw_prompt)}
    for count in sorted(tooth_counts):
        features.append(Feature(kind="gear_tooth", count=count))

    spec = DesignBrief(
        prompt=raw_prompt,
        units="mm",  # DesignBrief.units is the internal working unit; dims already converted to mm above
        object_class=_classify_object(raw_prompt),
        target_dims=TargetDimensions(**dims),
        features=features,
    )
    return ExtractionResult(spec=spec, coverage=coverage)
