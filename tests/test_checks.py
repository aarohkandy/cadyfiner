from __future__ import annotations

import tempfile

import trimesh

from cadyfiner.oracle.checks import evaluate_leg1
from cadyfiner.oracle.execute import ExecutionResult, run_cadquery
from cadyfiner.spec import DesignBrief, Feature, TargetDimensions


def _exec_from_mesh(mesh, **overrides):
    tmp = tempfile.NamedTemporaryFile(suffix=".stl", delete=False)
    mesh.export(tmp.name)
    defaults = dict(
        ok=True, stl_path=tmp.name, cq_volume=float(mesh.volume), cq_area=float(mesh.area),
        cq_bbox={"x": float(mesh.extents[0]), "y": float(mesh.extents[1]), "z": float(mesh.extents[2])},
        cq_is_valid_brep=True, cq_n_solids=1,
    )
    defaults.update(overrides)
    return ExecutionResult(**defaults)


class TestExecuteStage:
    def test_failed_execution_stops_at_execute(self):
        execution = ExecutionResult(ok=False, error_type="exception", error_message="boom")
        result = evaluate_leg1(execution, DesignBrief(prompt="x"))
        assert not result.overall_pass
        assert result.stopped_at == "execute"


class TestMeshValidityStage:
    def test_occ_verified_single_solid_passes(self, box_execution_result):
        result = evaluate_leg1(box_execution_result, DesignBrief(prompt="x"))
        assert result.stopped_at != "mesh_validity" or result.overall_pass

    def test_occ_invalid_brep_fails(self, box_execution_result):
        box_execution_result.cq_is_valid_brep = False
        result = evaluate_leg1(box_execution_result, DesignBrief(prompt="x"))
        assert not result.overall_pass
        assert result.stopped_at == "mesh_validity"

    def test_multi_solid_fails(self, box_execution_result):
        box_execution_result.cq_n_solids = 2
        result = evaluate_leg1(box_execution_result, DesignBrief(prompt="x"))
        assert not result.overall_pass
        assert result.stopped_at == "mesh_validity"

    def test_missing_occ_provenance_falls_back_to_trimesh_not_false_invalid(self, box_execution_result):
        """Regression: bool(None) used to silently read as 'CAD kernel says invalid', which is a
        different (and wrong) claim from 'we have no provenance to check'."""
        box_execution_result.cq_is_valid_brep = None
        box_execution_result.cq_n_solids = None
        result = evaluate_leg1(box_execution_result, DesignBrief(prompt="x"))
        stage = next(s for s in result.stages if s.name == "mesh_validity")
        assert "no BRep provenance" in stage.detail
        assert "CAD kernel reports this shape as an invalid solid" not in stage.detail

    def test_missing_occ_bbox_falls_back_to_mesh_extents_not_zero(self, box_execution_result):
        """Regression: ext used to silently default to [0,0,0] when cq_bbox was None, corrupting
        every downstream dimension/scale check."""
        box_execution_result.cq_bbox = None
        box_execution_result.cq_is_valid_brep = None
        box_execution_result.cq_n_solids = None
        result = evaluate_leg1(box_execution_result, DesignBrief(prompt="x"))
        assert result.measured["bbox_x"] > 0


class TestSpecConformanceStage:
    def test_vacuous_pass_on_empty_spec(self, box_execution_result):
        result = evaluate_leg1(box_execution_result, DesignBrief(prompt="x"))
        assert result.overall_pass
        stage = next(s for s in result.stages if s.name == "spec_conformance")
        assert "passed without real evaluation" in stage.detail or "no stated dimensions" in stage.detail

    def test_correct_dimensions_pass_regardless_of_build_axis_orientation(self, tmp_path):
        """Regression: spec_conformance assumed height<->Z with no enforcement; a model built on a
        different base plane would fail even though it's geometrically correct."""
        code = 'import cadquery as cq\nresult = cq.Workplane("XY").box(30, 20, 60)\n'
        execution = run_cadquery(code, tmp_path, timeout_s=30)
        spec = DesignBrief(prompt="x", target_dims=TargetDimensions(height=60, width=30, depth=20))
        result = evaluate_leg1(execution, spec)
        assert result.overall_pass

    def test_single_stated_dimension_matches_larger_axis_when_thats_correct(self, box_execution_result):
        """Regression: a single stated width/depth/length was always compared against the SMALLER
        of the two horizontal extents, false-failing a model with it on the larger axis."""
        box_execution_result.cq_bbox = {"x": 100.0, "y": 40.0, "z": 20.0}
        mesh = trimesh.creation.box(extents=[100.0, 40.0, 20.0])
        box_execution_result.cq_volume = float(mesh.volume)
        box_execution_result.cq_area = float(mesh.area)
        spec = DesignBrief(prompt="x", target_dims=TargetDimensions(width=100.0))
        result = evaluate_leg1(box_execution_result, spec)
        assert result.overall_pass

    def test_zero_stated_dimension_is_not_vacuous_pass(self, box_execution_result):
        """Regression: `x / target if target else 0.0` treated a stated 0.0 as 'no error'."""
        spec = DesignBrief(prompt="x", target_dims=TargetDimensions(height=0.0))
        result = evaluate_leg1(box_execution_result, spec)
        assert not result.overall_pass

    def test_excess_linear_dimensions_flagged_not_silently_dropped(self, box_execution_result):
        """Regression: a 4th stated linear dimension (more than 3 measurable axes) used to be
        silently dropped from the check entirely rather than flagged."""
        spec = DesignBrief(prompt="x", target_dims=TargetDimensions(height=20, width=50, depth=30, length=1000))
        result = evaluate_leg1(box_execution_result, spec)
        stage = next(s for s in result.stages if s.name == "spec_conformance")
        assert "length" in stage.detail
        assert "not independently verifiable" in stage.detail

    def test_multiple_hole_kind_features_are_aggregated_not_compared_independently(self, tmp_path):
        """Regression: two hole-kind features (e.g. mounting_hole=1 + drainage_hole=1) were each
        compared against the SAME total count independently, so a perfect 2-hole model always failed."""
        code = (
            'import cadquery as cq\n'
            'result = cq.Workplane("XY").box(50,40,10).faces(">Z").workplane()'
            '.pushPoints([(-15,0),(15,0)]).hole(4)\n'
        )
        execution = run_cadquery(code, tmp_path, timeout_s=30)
        spec = DesignBrief(prompt="x", features=[Feature(kind="mounting_hole", count=1), Feature(kind="drainage_hole", count=1)])
        result = evaluate_leg1(execution, spec)
        assert result.overall_pass

    def test_blind_hole_routed_to_unverifiable_not_through_hole_check(self, tmp_path):
        """Regression: a 'blind_hole' feature matched the 'hole' substring check and was compared
        against the through-hole genus estimate, which can never detect a non-through pocket."""
        code = (
            'import cadquery as cq\n'
            'result = cq.Workplane("XY").box(50,40,10).faces(">Z").workplane()'
            '.pushPoints([(-15,0),(15,0)]).hole(4)\n'
        )
        execution = run_cadquery(code, tmp_path, timeout_s=30)
        spec = DesignBrief(prompt="x", features=[Feature(kind="blind_hole", count=2)])
        result = evaluate_leg1(execution, spec)
        stage = next(s for s in result.stages if s.name == "spec_conformance")
        assert "not independently verifiable" in stage.detail


class TestManufacturabilityStage:
    def test_reasonable_box_passes(self, box_execution_result):
        result = evaluate_leg1(box_execution_result, DesignBrief(prompt="x"))
        assert result.overall_pass
        assert result.stopped_at == "manufacturability"

    def test_tiny_sliver_fails_scale_floor(self):
        mesh = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
        tmp = tempfile.NamedTemporaryFile(suffix=".stl", delete=False)
        mesh.export(tmp.name)
        execution = ExecutionResult(
            ok=True, stl_path=tmp.name, cq_volume=float(mesh.volume), cq_area=float(mesh.area),
            cq_bbox={"x": 1.0, "y": 1.0, "z": 1.0}, cq_is_valid_brep=True, cq_n_solids=1,
        )
        result = evaluate_leg1(execution, DesignBrief(prompt="x"))
        assert not result.overall_pass
        assert result.stopped_at == "manufacturability"


class TestGearToothCounting:
    def test_gear_tooth_count_matches_within_tolerance(self, tmp_path):
        code = """import cadquery as cq
import math
n_teeth = 16
base_r = 20
result = cq.Workplane("XY").circle(base_r).extrude(8)
for i in range(n_teeth):
    angle = 360 * i / n_teeth
    tooth = cq.Workplane("XY").center(base_r * math.cos(math.radians(angle)), base_r * math.sin(math.radians(angle))).circle(3).extrude(8)
    result = result.union(tooth)
"""
        execution = run_cadquery(code, tmp_path, timeout_s=45)
        assert execution.ok
        spec = DesignBrief(prompt="x", features=[Feature(kind="gear_tooth", count=16)])
        result = evaluate_leg1(execution, spec)
        assert result.overall_pass

    def test_wrong_gear_tooth_count_fails(self, tmp_path):
        code = """import cadquery as cq
import math
n_teeth = 16
base_r = 20
result = cq.Workplane("XY").circle(base_r).extrude(8)
for i in range(n_teeth):
    angle = 360 * i / n_teeth
    tooth = cq.Workplane("XY").center(base_r * math.cos(math.radians(angle)), base_r * math.sin(math.radians(angle))).circle(3).extrude(8)
    result = result.union(tooth)
"""
        execution = run_cadquery(code, tmp_path, timeout_s=45)
        spec = DesignBrief(prompt="x", features=[Feature(kind="gear_tooth", count=30)])
        result = evaluate_leg1(execution, spec)
        assert not result.overall_pass
