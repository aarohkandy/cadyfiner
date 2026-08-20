from __future__ import annotations

from cadyfiner.refine import extract


class TestDimensionExtraction:
    def test_noun_form_forward_phrasing(self):
        r = extract("material thickness 4mm")
        assert r.spec.stated_dimensions().get("thickness") == 4.0

    def test_adjective_form_backward_phrasing(self):
        """Regression: '90mm tall' has the number BEFORE the adjective; a forward-only search
        used to grab an unrelated later number instead."""
        r = extract("a cylinder 90 mm tall and 80 mm outer diameter")
        dims = r.spec.stated_dimensions()
        assert dims.get("height") == 90.0
        assert dims.get("diameter") == 80.0

    def test_outer_diameter_phrasing_with_trailing_clause(self):
        """Regression: '80 mm outer diameter, with a 3 mm wall thickness' used to grab the
        thickness's '3' for diameter, since forward search found *something* before ever
        trying backward."""
        r = extract("90 mm tall and 80 mm outer diameter, with a 3 mm wall thickness")
        dims = r.spec.stated_dimensions()
        assert dims.get("diameter") == 80.0
        assert dims.get("thickness") == 3.0

    def test_full_wall_planter_ladder_level_7(self):
        text = (
            "Design a cylindrical 3D-printable wall-mounted planter, 90 mm tall and 80 mm outer "
            "diameter, with a 3 mm wall thickness, an open top, and a flat back panel containing "
            "one mounting hole sized for a #8 screw."
        )
        r = extract(text)
        dims = r.spec.stated_dimensions()
        assert dims.get("height") == 90.0
        assert dims.get("diameter") == 80.0
        assert dims.get("thickness") == 3.0

    def test_no_dimensions_in_minimal_prompt(self):
        r = extract("Make me a planter I can put on my wall.")
        assert r.spec.stated_dimensions() == {}

    def test_radius_converted_to_diameter(self):
        r = extract("a 20mm-radius base cylinder")
        assert r.spec.stated_dimensions().get("diameter") == 40.0

    def test_unit_conversion_cm_to_mm(self):
        r = extract("a plate 5cm thick")
        assert r.spec.stated_dimensions().get("thickness") == 50.0

    def test_spelled_out_millimeters_not_misread_as_meters(self):
        """Regression: no trailing \\b let 'millimeters' prefix-match unit='m' (meters), a 1000x error."""
        r = extract("height 10 millimeters, width 40mm")
        dims = r.spec.stated_dimensions()
        assert dims.get("height") == 10.0
        assert dims.get("width") == 40.0

    def test_compact_unit_detected_as_default(self):
        """Regression: _UNIT_PATTERN required \\b on both sides, invisible to digit+unit with no space."""
        from cadyfiner.refine import _detect_default_unit

        assert _detect_default_unit("A gear 50 wide and 30cm tall.") == "cm"

    def test_keyword_substring_inside_unrelated_word_not_matched(self):
        """Regression: plain substring search matched 'height' inside 'heightened'."""
        r = extract("The heightened box is 80mm wide.")
        dims = r.spec.stated_dimensions()
        assert dims.get("height") is None
        assert dims.get("width") == 80.0

    def test_fraction_inch_parsed_correctly(self):
        r = extract("Wall thickness of 1/2 inch, diameter 4 inches.")
        dims = r.spec.stated_dimensions()
        assert abs(dims["thickness"] - 12.7) < 0.01

    def test_european_decimal_comma(self):
        r = extract("The plate is 80,5mm wide and 3mm thick.")
        assert abs(r.spec.stated_dimensions()["width"] - 80.5) < 0.01

    def test_outer_diameter_still_correct_after_all_fixes(self):
        """Full regression check of the audited wall_planter level-7 ladder text."""
        text = (
            "Design a cylindrical 3D-printable wall-mounted planter, 90 mm tall and 80 mm outer "
            "diameter, with a 3 mm wall thickness, an open top, and a flat back panel containing "
            "one mounting hole sized for a #8 screw."
        )
        dims = extract(text).spec.stated_dimensions()
        assert dims == {"height": 90.0, "diameter": 80.0, "thickness": 3.0}


class TestObjectClassification:
    def test_bracket_is_mechanical(self):
        assert extract("Create a mounting bracket with holes.").spec.object_class == "mechanical_functional"

    def test_gear_is_mechanical(self):
        assert extract("Create a gear.").spec.object_class == "mechanical_functional"

    def test_planter_is_decorative(self):
        assert extract("Make me a planter I can put on my wall.").spec.object_class == "decorative"

    def test_pen_holder_is_decorative(self):
        assert extract("Make me a pen holder for my desk.").spec.object_class == "decorative"

    def test_no_keyword_substring_false_positive(self):
        """Regression: 'mount' substring inside a longer word used to flip classification."""
        oc = extract("A bracket that hooks onto the shelf, 80mm wide, holds a load.").spec.object_class
        assert oc == "mechanical_functional"

    def test_ties_default_to_mechanical_functional(self):
        from cadyfiner.refine import _classify_object

        assert _classify_object("A thing with no keywords at all.") == "mechanical_functional"


class TestFeatureExtraction:
    def test_hole_count_extraction(self):
        # Digits only, not spelled-out numbers ("four") — a known, documented Stage 1
        # limitation; Stage 2's LLM call is the intended backstop for this gap.
        r = extract("with 4 drainage holes, each 3mm")
        holes = [f for f in r.spec.features if "hole" in f.kind]
        assert holes and holes[0].count == 4

    def test_tooth_count_extraction(self):
        r = extract("a gear with exactly 16 teeth")
        teeth = [f for f in r.spec.features if f.kind == "gear_tooth"]
        assert teeth and teeth[0].count == 16

    def test_hole_kind_attribution(self):
        """Regression: the filler group used to greedily consume the kind word, collapsing
        every hole to the generic 'hole' kind regardless of stated kind."""
        r = extract("The bracket needs 4 mounting holes and 2 drainage holes.")
        kinds = {f.kind for f in r.spec.features}
        assert "mounting_hole" in kinds
        assert "drainage_hole" in kinds

    def test_tooth_count_deduped_across_phrasings(self):
        """Regression: '24-tooth' and '24 teeth' in the same prompt produced two Features
        for one fact."""
        r = extract("A gear that is 24-tooth. It has 24 teeth for meshing with the drive gear.")
        assert len(r.spec.features) == 1
        assert r.spec.features[0].count == 24
