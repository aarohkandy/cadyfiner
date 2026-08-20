from __future__ import annotations

import json

from cadyfiner.refine import ExtractionResult
from cadyfiner.refine_stage2 import fill_gaps
from cadyfiner.spec import DesignBrief


def _extraction():
    spec = DesignBrief(prompt="a small decorative planter", object_class="decorative", target_dims={"width": 80.0}, features=[])
    return ExtractionResult(spec=spec, coverage={"width": True})


class TestFillGapsRobustness:
    def test_wrong_typed_json_does_not_crash(self):
        """Regression: merge loops sat outside the try/except, so a wrong-typed LLM response
        (target_dims as a string, features as a list of strings) raised an uncaught AttributeError."""

        def mock(prompt, **kw):
            return json.dumps(
                {"target_dims": "not applicable", "features": ["a mounting hole"], "process_notes": [],
                 "style_notes": [], "assumptions_made": [], "refined_prompt": "x"}
            )

        result = fill_gaps(_extraction(), mock, max_retries=0)  # must not raise
        assert result.parse_ok

    def test_json_extraction_skips_unrelated_leading_brace_pair(self):
        """Regression: a greedy first-to-last-brace regex matched across two separate JSON
        objects in the response; the fix after that (first-valid-JSON) still picked the WRONG
        one when an earlier, unrelated object happened to also be individually valid JSON."""
        valid = {
            "target_dims": {}, "features": [], "process_notes": [], "style_notes": [],
            "assumptions_made": [], "refined_prompt": "a small round planter, 80mm diameter",
        }
        output = 'For example {"foo": "bar"}... here is my answer: ' + json.dumps(valid)

        def mock(prompt, **kw):
            return output

        result = fill_gaps(_extraction(), mock, max_retries=2)
        assert result.parse_ok
        assert result.spec.refined_prompt == valid["refined_prompt"]

    def test_invalid_target_dims_key_does_not_discard_whole_response(self):
        """Regression: an LLM-returned key not in TargetDimensions (e.g. 'radius') made the
        DesignBrief construction raise pydantic's extra='forbid' error, discarding an otherwise
        good response via the blanket except."""

        def mock(prompt, **kw):
            return json.dumps(
                {"target_dims": {"radius": 40.0, "height": 90.0}, "features": [], "process_notes": [],
                 "style_notes": [], "assumptions_made": [], "refined_prompt": "a round object"}
            )

        result = fill_gaps(_extraction(), mock, max_retries=0)
        assert result.parse_ok
        dims = result.spec.stated_dimensions()
        assert "radius" not in dims
        assert dims.get("height") == 90.0

    def test_schema_placeholder_feature_kind_rejected(self):
        """Regression: the LLM sometimes echoes the system prompt's own JSON example
        ({"kind": "string", ...}) back as if it were a real feature."""

        def mock(prompt, **kw):
            return json.dumps(
                {"target_dims": {}, "features": [{"kind": "string", "count": None, "size": None, "size_unit": "mm"}],
                 "process_notes": [], "style_notes": [], "assumptions_made": [], "refined_prompt": "x"}
            )

        result = fill_gaps(_extraction(), mock, max_retries=0)
        assert not any(f.kind == "string" for f in result.spec.features)

    def test_missing_refined_prompt_is_a_failed_attempt_not_a_success(self):
        """Regression: valid JSON with a null/missing refined_prompt was accepted as parse_ok=True,
        so the harness would silently send byte-identical raw prompt text to both arms of a paired
        comparison while still counting the pair as decisive."""
        calls = []

        def mock(prompt, **kw):
            calls.append(1)
            return json.dumps(
                {"target_dims": {}, "features": [], "process_notes": [], "style_notes": [],
                 "assumptions_made": [], "refined_prompt": None}
            )

        result = fill_gaps(_extraction(), mock, max_retries=2)
        assert not result.parse_ok
        assert len(calls) == 3  # exhausted all retries

    def test_good_response_still_works(self):
        def mock(prompt, **kw):
            return json.dumps(
                {"target_dims": {"height": 90.0}, "features": [{"kind": "drainage_hole", "count": 4}],
                 "process_notes": ["3mm walls"], "style_notes": [], "assumptions_made": ["Assumed 90mm height"],
                 "refined_prompt": "A cylindrical planter, 90mm tall, with 4 drainage holes."}
            )

        result = fill_gaps(_extraction(), mock, max_retries=0)
        assert result.parse_ok
        assert result.spec.stated_dimensions()["height"] == 90.0
        assert any(f.kind == "drainage_hole" for f in result.spec.features)
