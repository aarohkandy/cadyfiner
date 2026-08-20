from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import trimesh


@pytest.fixture
def tmp_stl_path(tmp_path):
    """Give tests a scratch path to export an STL to."""

    return tmp_path / "model.stl"


@pytest.fixture
def box_execution_result():
    """A hand-built ExecutionResult (no real CadQuery run) for a known 50x30x20mm box.

    Used to test checks.py logic in isolation without paying for a real
    subprocess + CadQuery execution on every test.
    """

    from cadyfiner.oracle.execute import ExecutionResult

    mesh = trimesh.creation.box(extents=[50.0, 30.0, 20.0])
    tmp = tempfile.NamedTemporaryFile(suffix=".stl", delete=False)
    mesh.export(tmp.name)
    return ExecutionResult(
        ok=True,
        stl_path=tmp.name,
        cq_volume=float(mesh.volume),
        cq_area=float(mesh.area),
        cq_bbox={"x": 50.0, "y": 30.0, "z": 20.0},
        cq_is_valid_brep=True,
        cq_n_solids=1,
    )
