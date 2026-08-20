"""Stage 2 of the refiner: a single constrained LLM call fills gaps left by
Stage 1's deterministic extraction (:mod:`cadyfiner.refine`).

Kept as a separate module from ``refine.py`` because this stage's core
design lever — *how much* detail to add per object class — is set by
``DEPTH_POLICY``, which is deliberately data (not hard-coded logic) so it
can be revised from empirical pilot results without touching the call
mechanics around it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable

from cadyfiner.refine import ExtractionResult
from cadyfiner.spec import DesignBrief, TargetDimensions

_VALID_DIM_KEYS = set(TargetDimensions.model_fields)
# The literal placeholder the system prompt's own JSON example uses for a feature's "kind" — an
# adversarial review found a real (if weak) LLM sometimes echoes the schema example back as if it
# were a real answer, which would otherwise be accepted as a genuine feature named "string".
_SCHEMA_PLACEHOLDER_FEATURE_KIND = "string"

# What kind of information Stage 2 is allowed to add, per object class.
# Categories follow the preprint's own recommended taxonomy (identity,
# function, interface, dimensions, process, topology, feature placement)
# rather than a single "specificity level" number, because the evidence
# this project is built on found that number alone doesn't transfer across
# object types: the wall_planter ladder's own data showed level 7
# ("process" depth: dimensions + wall thickness + a named interface) beat
# level 10 ("topology"/"feature placement" depth: exact keyhole slot
# geometry, drainage hole layout, fillets) on both geometric reliability
# and (per the human-vote reanalysis) overall preference — but that result
# is single-family evidence. See prompts/seed_bank/pilot_mechanical/ for
# the mechanical-domain check this policy split is actually based on.
DEPTH_POLICY: dict[str, list[str]] = {
    "decorative": ["identity", "function", "interface", "dimensions", "process"],
    "mechanical_functional": [
        "identity", "function", "interface", "dimensions", "process", "topology", "feature_placement"
    ],
}

_SYSTEM_PROMPT = """You are a CAD design-spec assistant. Given a raw user prompt and dimensions/features
already extracted from it, produce a JSON object with exactly these fields:

{{
  "target_dims": {{"width": null, "depth": null, "height": null, "diameter": null, "length": null, "thickness": null}},
  "features": [{{"kind": "string", "count": null, "size": null, "size_unit": "mm"}}],
  "process_notes": ["string"],
  "style_notes": ["string"],
  "assumptions_made": ["string"],
  "refined_prompt": "string"
}}

Rules:
- Fill target_dims values already given below EXACTLY as given; do not change them.
- You may fill in MISSING target_dims and features, but ONLY within these categories: {categories}.
  Do not invent exact feature placement, fillet radii, or topology details unless "topology" or
  "feature_placement" is in that list.
- Every value you invent (not given by the user) MUST be logged as a plain-English sentence in
  assumptions_made, e.g. "Assumed 80mm outer diameter since none was stated."
- refined_prompt must be a single natural-language paragraph describing the object with all the
  dimensions and features above included, suitable for a CAD code generator to read directly.
- Output ONLY the JSON object. No prose before or after it.

Raw user prompt:
{raw_prompt}

Already extracted (do not change these):
target_dims: {extracted_dims}
features: {extracted_features}
object_class: {object_class}
"""

_EXPECTED_TOP_LEVEL_KEYS = {"target_dims", "features", "process_notes", "style_notes", "assumptions_made", "refined_prompt"}


def _extract_first_json_object(text: str) -> dict | None:
    """Parse the JSON object in ``text`` that actually looks like our expected schema.

    Regression, two rounds: (1) a greedy ``\\{.*\\}`` (DOTALL) regex matched
    from the FIRST ``{`` to the LAST ``}`` in the whole response — if the
    model echoed an unrelated brace pair before its real answer, the
    "JSON" captured spanned both and was invalid. (2) The first fix
    (``json.JSONDecoder.raw_decode`` at the first ``{``) solved that but
    introduced a new failure: when the text contains TWO separate,
    individually-valid JSON objects (e.g. the model echoes a small
    unrelated example like ``{"foo": "bar"}`` before its real, differently
    shaped answer), the first one parses successfully and gets returned —
    it's valid JSON, just the wrong object. Fixed by requiring the parsed
    object to actually contain at least one of our expected top-level
    keys before accepting it, and continuing to search past any complete
    JSON object that doesn't.
    """

    decoder = json.JSONDecoder()
    start = text.find("{")
    while start != -1:
        try:
            obj, end = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            start = text.find("{", start + 1)
            continue
        if isinstance(obj, dict) and _EXPECTED_TOP_LEVEL_KEYS & obj.keys():
            return obj
        start = text.find("{", end)
    return None


@dataclass
class FillGapsResult:
    spec: DesignBrief
    raw_llm_output: str
    parse_ok: bool


def _build_prompt(extraction: ExtractionResult) -> str:
    categories = DEPTH_POLICY.get(extraction.spec.object_class or "decorative", DEPTH_POLICY["decorative"])
    return _SYSTEM_PROMPT.format(
        categories=", ".join(categories),
        raw_prompt=extraction.spec.prompt,
        extracted_dims=json.dumps(extraction.spec.stated_dimensions()),
        extracted_features=json.dumps([f.model_dump(exclude_none=True) for f in extraction.spec.features]),
        object_class=extraction.spec.object_class,
    )


def fill_gaps(
    extraction: ExtractionResult,
    generate: Callable[..., str],
    *,
    max_retries: int = 2,
    **generate_kwargs,
) -> FillGapsResult:
    """Stage 2: one constrained LLM call, retried on invalid JSON.

    ``generate`` is any of the ``cadyfiner.generators.*`` adapter
    functions (or a compatible callable) — this stage is generator-agnostic
    by construction, matching the requirement to work against both local
    Ollama and a frontier backend.
    """

    prompt = _build_prompt(extraction)
    last_output = ""
    for attempt in range(max_retries + 1):
        last_output = generate(prompt, **generate_kwargs)
        data = _extract_first_json_object(last_output)
        if data is None:
            continue

        # Regression: the merge loops below used to sit OUTSIDE this try/except, so a
        # plausible-but-wrong-typed LLM response (target_dims as a string like "not applicable"
        # instead of a dict, features as a list of strings instead of dicts) raised an uncaught
        # AttributeError instead of falling through to a retry like a JSON parse failure does.
        # isinstance guards make each malformed shape a silent skip rather than a crash.
        try:
            merged_dims = dict(extraction.spec.stated_dimensions())
            target_dims_data = data.get("target_dims")
            if isinstance(target_dims_data, dict):
                for k, v in target_dims_data.items():
                    # Regression: an unfiltered key (e.g. "radius", which Stage 1 treats as a
                    # first-class diameter synonym but this schema has no equivalent for) made
                    # DesignBrief(...) raise a pydantic extra="forbid" ValidationError below,
                    # discarding the ENTIRE otherwise-good response, not just the bad key.
                    if k in _VALID_DIM_KEYS and k not in merged_dims and v is not None:
                        merged_dims[k] = v

            existing_feature_kinds = {f.kind for f in extraction.spec.features}
            new_features = list(extraction.spec.features)
            features_data = data.get("features")
            if isinstance(features_data, list):
                for f in features_data:
                    if (
                        isinstance(f, dict)
                        and f.get("kind")
                        and f["kind"] != _SCHEMA_PLACEHOLDER_FEATURE_KIND
                        and f["kind"] not in existing_feature_kinds
                    ):
                        new_features.append(f)

            refined_prompt = data.get("refined_prompt")
            if not refined_prompt or not str(refined_prompt).strip():
                # A missing/empty refined_prompt is a failed attempt, not a successful one with
                # nothing to show — an earlier version accepted this as parse_ok=True, and the
                # harness would then silently send the identical raw prompt to both the "raw" and
                # "refined" arms of a paired comparison while still counting it as a decisive result.
                continue

            spec = DesignBrief(
                prompt=extraction.spec.prompt,
                refined_prompt=refined_prompt,
                object_class=extraction.spec.object_class,
                target_dims=merged_dims,
                features=new_features,
                process_notes=data.get("process_notes") or [],
                style_notes=data.get("style_notes") or [],
                assumptions_made=data.get("assumptions_made") or [],
            )
        except Exception:
            continue
        return FillGapsResult(spec=spec, raw_llm_output=last_output, parse_ok=True)

    # Every attempt failed to parse: fall back to the Stage 1 extraction
    # verbatim rather than crash — a refiner that silently loses the user's
    # already-extracted dimensions on a bad LLM response would be worse
    # than one that just doesn't add anything this round.
    fallback = extraction.spec.model_copy(update={"refined_prompt": extraction.spec.prompt})
    return FillGapsResult(spec=fallback, raw_llm_output=last_output, parse_ok=False)
