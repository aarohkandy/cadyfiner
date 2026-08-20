from __future__ import annotations

from pathlib import Path

from cadyfiner.oracle.execute import extract_code, prefilter, run_cadquery


class TestExtractCode:
    def test_fenced_block(self):
        out = "Here is code:\n```python\nimport cadquery as cq\nresult = 1\n```\nDone."
        assert extract_code(out) == "import cadquery as cq\nresult = 1\n"

    def test_no_fence_falls_back_to_whole_text(self):
        assert extract_code("result = 1") == "result = 1\n"


class TestPrefilter:
    def test_good_code_passes(self):
        code = 'import cadquery as cq\nresult = cq.Workplane("XY").box(10, 10, 10)\n'
        assert prefilter(code) is None

    def test_missing_result_rejected(self):
        code = 'import cadquery as cq\ncq.Workplane("XY").box(10,10,10)\n'
        assert prefilter(code) is not None

    def test_missing_import_rejected(self):
        code = "result = 1\n"
        assert prefilter(code) is not None

    def test_disallowed_subprocess_rejected(self):
        code = 'import cadquery as cq\nimport subprocess\nresult = cq.Workplane("XY").box(1,1,1)\n'
        assert prefilter(code) is not None

    def test_disallowed_direct_export_rejected(self):
        code = 'import cadquery as cq\nresult = cq.Workplane("XY").box(1,1,1)\nresult.val().exportStl("/tmp/x.stl")\n'
        assert prefilter(code) is not None

    def test_socket_head_cap_screw_not_false_positive(self):
        """Regression: 'socket' alone used to ban the ordinary phrase 'socket head cap screw'."""
        code = (
            'import cadquery as cq\n# clearance hole for an M3 socket head cap screw\n'
            'result = cq.Workplane("XY").box(10,10,10).faces(">Z").workplane().hole(3.2)\n'
        )
        assert prefilter(code) is None

    def test_import_socket_module_still_rejected(self):
        code = 'import cadquery as cq\nimport socket\nresult = cq.Workplane("XY").box(1,1,1)\n'
        assert prefilter(code) is not None

    def test_pos_attribute_not_false_positive(self):
        """Regression: 'os.' substring check used to ban any variable named e.g. 'pos' with attribute access."""
        code = (
            'import cadquery as cq\npos = cq.Vector(10,0,0)\n'
            'result = cq.Workplane("XY").workplane(offset=pos.z).box(10,10,10)\n'
        )
        assert prefilter(code) is None

    def test_make_blueprint_not_false_positive(self):
        """Regression: 'print(' substring check used to ban identifiers merely containing that substring."""
        code = (
            'import cadquery as cq\n'
            'def make_blueprint(width, depth):\n    return cq.Workplane("XY").box(width, depth, 5)\n'
            'result = make_blueprint(40, 30)\n'
        )
        assert prefilter(code) is None


class TestRunCadquery:
    def test_good_code_produces_valid_result(self, tmp_path):
        code = 'import cadquery as cq\nresult = cq.Workplane("XY").box(10, 10, 10).faces(">Z").workplane().hole(3)\n'
        result = run_cadquery(code, tmp_path, timeout_s=30)
        assert result.ok
        assert result.cq_is_valid_brep is True
        assert result.cq_n_solids == 1
        assert Path(result.stl_path).exists()

    def test_missing_result_variable(self, tmp_path):
        code = 'import cadquery as cq\ncq.Workplane("XY").box(1,1,1)\n'
        result = run_cadquery(code, tmp_path, timeout_s=30)
        assert not result.ok
        assert result.error_type == "prefilter_rejected"

    def test_infinite_loop_is_killed_by_resource_limit(self, tmp_path):
        code = (
            'import cadquery as cq\nx = 0\nwhile True:\n    x += 1\n'
            'result = cq.Workplane("XY").box(10,10,10)\n'
        )
        result = run_cadquery(code, tmp_path, timeout_s=10, cpu_seconds=3)
        assert not result.ok
        assert result.error_type in ("no_result_file", "timeout")

    def test_two_calls_do_not_collide_on_stl_filename(self, tmp_path):
        """Regression: STL used to write to a fixed 'model.stl', corrupting concurrent/overlapping runs."""
        code_a = 'import cadquery as cq\nresult = cq.Workplane("XY").box(10, 10, 10)\n'
        code_b = 'import cadquery as cq\nresult = cq.Workplane("XY").box(20, 20, 20)\n'
        result_a = run_cadquery(code_a, tmp_path, timeout_s=30)
        result_b = run_cadquery(code_b, tmp_path, timeout_s=30)
        assert result_a.stl_path != result_b.stl_path
        assert Path(result_a.stl_path).exists()
        assert Path(result_b.stl_path).exists()
