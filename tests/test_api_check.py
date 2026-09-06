from __future__ import annotations

from cadyfiner.oracle.api_check import check_api_usage


class TestCheckApiUsage:
    def test_module_level_hallucination_cq_angle(self):
        code = 'import cadquery as cq\na = cq.Angle(45)\nresult = cq.Workplane("XY").box(1, 1, 1)\n'
        msg = check_api_usage(code)
        assert msg is not None
        assert "Angle" in msg

    def test_module_function_confusion_cq_union(self):
        code = (
            'import cadquery as cq\n'
            'a = cq.Workplane("XY").box(10, 10, 10)\n'
            'b = cq.Workplane("XY").box(5, 5, 5)\n'
            'result = cq.union(a, b)\n'
        )
        msg = check_api_usage(code)
        assert msg is not None
        assert "method on Workplane/Shape, not a module-level function" in msg

    def test_chained_hallucination_workplaneat(self):
        code = 'import cadquery as cq\nresult = cq.Workplane("XY").workplaneAt((0, 0, 10))\n'
        msg = check_api_usage(code)
        assert msg is not None
        assert "workplaneAt" in msg

    def test_shape_level_hallucination_gettranslation(self):
        code = (
            'import cadquery as cq\n'
            'result = cq.Workplane("XY").box(1, 1, 1)\n'
            's = result.val()\n'
            't = s.getTranslation()\n'
        )
        msg = check_api_usage(code)
        assert msg is not None
        assert "getTranslation" in msg

    def test_valid_box_code_not_rejected(self):
        code = 'import cadquery as cq\nresult = cq.Workplane("XY").box(10, 10, 10).faces(">Z").workplane().hole(3)\n'
        assert check_api_usage(code) is None

    def test_valid_sketch_finalize_chain_not_rejected(self):
        """Load-bearing false-positive guard: without the _TRANSITIONS table this exact,
        real, executed CadQuery pattern would false-reject on .finalize() (Sketch-only,
        not Workplane) and .extrude() (back on Workplane after finalize())."""
        code = (
            'import cadquery as cq\n'
            'result = (\n'
            '    cq.Workplane("XY")\n'
            '    .box(40, 40, 10)\n'
            '    .faces(">Z")\n'
            '    .workplane()\n'
            '    .sketch()\n'
            '    .rect(20, 20)\n'
            '    .finalize()\n'
            '    .extrude(5)\n'
            ')\n'
        )
        assert check_api_usage(code) is None

    def test_valid_method_union_not_confused_with_module_function(self):
        code = (
            'import cadquery as cq\n'
            'a = cq.Workplane("XY").box(10, 10, 10)\n'
            'b = cq.Workplane("XY").box(5, 5, 5)\n'
            'result = a.union(b)\n'
        )
        assert check_api_usage(code) is None

    def test_unknown_type_chain_not_flagged(self):
        """Documents/verifies the stated 'can't trace through user functions' limitation
        fails safe (skips), not falsely (flags valid downstream usage as invalid)."""
        code = (
            'import cadquery as cq\n'
            'def make_box(w, d, h):\n'
            '    return cq.Workplane("XY").box(w, d, h)\n'
            'result = make_box(1, 2, 3).faces(">Z").hole(1)\n'
        )
        assert check_api_usage(code) is None

    def test_syntax_error_detected(self):
        code = 'import cadquery as cq\nresult = cq.Workplane("XY").box(1, 1, 1\n'
        msg = check_api_usage(code)
        assert msg is not None
        assert "does not parse" in msg

    def test_no_cadquery_import_returns_none(self):
        assert check_api_usage("result = 1\n") is None

    def test_submodule_chain_not_checked(self):
        """cq.exporters.export is 2-levels-deep submodule access, out of scope for this
        checker (documented limitation) -- separately banned by prefilter()'s existing
        substring rule, so this is a documented gap with no practical effect there."""
        code = 'import cadquery as cq\nresult = cq.Workplane("XY").box(1, 1, 1)\ncq.exporters.export(result, "out.stl")\n'
        assert check_api_usage(code) is None
