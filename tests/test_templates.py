from __future__ import annotations

import json
from pathlib import Path

import pytest

from cadyfiner.oracle.checks import evaluate_leg1
from cadyfiner.oracle.execute import run_cadquery
from cadyfiner.oracle.templates import TEMPLATES, build_template_code, detect_family
from cadyfiner.refine import extract
from cadyfiner.spec import DesignBrief

FAMILIES_DIR = Path(__file__).resolve().parents[1] / "prompts" / "seed_bank" / "families"

# All 21 seed-bank items pass. Two Stage 1 proximity-matching bugs (coaster_high,
# pen_holder_high) were fixed by clipping the keyword-search window at a clause boundary
# (comma/period/standalone "x") -- see _nearest_number_mm in refine.py. The third
# (enclosure_high, an unlabeled "80mm by 60mm by 30mm" positional triple with no per-axis
# keyword at all) needed a different fix: cadyfiner.oracle.templates._positional_triple,
# scoped ONLY to this module's own box-shaped template families (not a change to Stage 1's
# shared regex engine used by every other consumer, including the free-form LLM fallback
# path for non-templated object types). See docs/CADGEN_RELIABILITY.md for the full story.
_KNOWN_STAGE1_EXTRACTION_MISATTRIBUTIONS: set[str] = set()


class TestTemplatesPassTheirOwnGroundTruth:
    """Each template, called with NO overrides (its own defaults), must independently
    satisfy its family's real seed-bank ground truth -- the actual verification these
    templates were built and tuned against."""

    @pytest.mark.parametrize("family", sorted(TEMPLATES))
    def test_default_template_passes_ground_truth(self, family, tmp_path):
        items = json.loads((FAMILIES_DIR / f"{family}.json").read_text())
        gt = DesignBrief.model_validate(items[0]["ground_truth"])  # all tiers share one ground truth
        code = TEMPLATES[family]()
        result = run_cadquery(code, tmp_path / family, timeout_s=30)
        leg1 = evaluate_leg1(result, gt)
        assert leg1.overall_pass, leg1.feedback_text()


class TestFamilyDetection:
    @pytest.mark.parametrize(
        "prompt,expected",
        [
            ("Create a mounting bracket with holes.", "bracket"),
            ("Make me a coaster for my drink.", "coaster"),
            ("Make me a pen holder for my desk.", "pen_holder"),
            ("Create an open-top electronics enclosure.", "enclosure"),
            ("Create a gear.", "gear"),
            ("Make me a small tray for holding paperclips and coins on my desk.", "desk_organizer_tray"),
            ("Make me a planter I can put on my wall.", "wall_planter"),
        ],
    )
    def test_known_phrasing_detected(self, prompt, expected):
        assert detect_family(prompt) == expected

    def test_unrecognized_prompt_returns_none(self):
        assert detect_family("Design a novelty desk trophy shaped like a rocket.") is None


class TestBuildTemplateCode:
    def test_unrecognized_family_returns_none(self):
        spec = DesignBrief(prompt="anything")
        assert build_template_code("not_a_real_family", spec) is None

    def test_extracted_dimensions_override_defaults(self, tmp_path):
        """A genuinely well-extracted dimension should flow through to the generated code,
        not just silently use the template's baked-in default."""

        spec = DesignBrief(prompt="Make me a pen holder for my desk.")
        spec.target_dims.diameter = 50
        spec.target_dims.height = 120
        code = build_template_code("pen_holder", spec)
        assert "diameter = 50" in code
        assert "height = 120" in code

    def test_enclosure_prefers_positional_triple_over_misattributed_keyword_match(self):
        """Real bug found live (enclosure_high): Stage 1's keyword-proximity search for
        'height' has no candidate near the object's own unlabeled '80mm by 60mm by 30mm'
        dimensions statement, so it instead latches onto an unrelated LATER mention (a
        standoff's '25mm tall'), silently overriding this template's correct default (30)
        with a wrong one (25). The positional triple, when present, must win."""

        prompt = (
            "Create a rectangular electronics enclosure. External dimensions 80mm by 60mm "
            "by 30mm, uniform wall thickness 2mm, open top. Four cylindrical mounting "
            "standoffs in the corners, 6mm outer diameter, 3mm bore, 25mm tall, inset 5mm "
            "from each corner wall."
        )
        spec = DesignBrief(prompt=prompt)
        spec.target_dims.height = 25.0  # what Stage 1 actually (wrongly) extracts today
        code = build_template_code("enclosure", spec)
        assert "width = 80" in code
        assert "depth = 60" in code
        assert "height = 30" in code

    def test_enclosure_falls_back_to_per_field_extraction_without_a_triple(self):
        """No 'dimensions A by B by C' phrasing present -> ordinary per-field extraction
        (or the template's own defaults) still applies, unaffected by the triple check."""

        spec = DesignBrief(prompt="Create an open-top electronics enclosure.")
        spec.target_dims.height = 42.0
        code = build_template_code("enclosure", spec)
        assert "height = 42" in code

    def test_gear_rejects_implausibly_small_extracted_diameter(self):
        """Real bug found live: Stage 1 sometimes extracts a gear's BORE diameter (e.g. 12mm)
        as the overall gear diameter on dense text like '...12mm diameter for a shaft'. A
        16-tooth gear can't fit in a <25mm overall diameter without self-intersecting teeth --
        treated as a clear cross-attribution error, not a legitimately tiny gear."""

        spec = DesignBrief(prompt="Create a gear.")
        spec.target_dims.diameter = 12.0
        code = build_template_code("gear", spec)
        assert "base_radius = 20" in code  # falls back to the default, not (12/2 - 3) = 3


class TestFullSeedBankIntegration:
    """The real end-to-end check: family detection + Stage 1 extraction + template dispatch
    + the actual sandboxed execution/scoring pipeline, against every seed in the bank (not
    just the canonical high-detail ground-truth prompt)."""

    def test_pass_rate_across_all_seeds(self, tmp_path):
        n_pass, n_total, failures = 0, 0, []
        for path in sorted(FAMILIES_DIR.glob("*.json")):
            for item in json.loads(path.read_text()):
                n_total += 1
                family = detect_family(item["raw_prompt"])
                assert family == item["family"], f"{item['id']}: detected {family!r}"

                extraction = extract(item["raw_prompt"])
                code = build_template_code(family, extraction.spec)
                result = run_cadquery(code, tmp_path / item["id"], timeout_s=30)
                gt = DesignBrief.model_validate(item["ground_truth"])
                leg1 = evaluate_leg1(result, gt)
                if leg1.overall_pass:
                    n_pass += 1
                elif item["id"] not in _KNOWN_STAGE1_EXTRACTION_MISATTRIBUTIONS:
                    failures.append(f"{item['id']}: {leg1.feedback_text()}")

        assert not failures, "\n".join(failures)
        # All 21 seed-bank items pass. _KNOWN_STAGE1_EXTRACTION_MISATTRIBUTIONS is kept (now
        # empty) as the mechanism for the next real extraction gap, rather than deleted --
        # every prior one found here was fixed by adding to this set first, then fixing it.
        assert n_pass == n_total == 21
