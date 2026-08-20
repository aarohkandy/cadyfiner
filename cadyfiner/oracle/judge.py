"""Leg 2: a VLM judge on rendered views, reporting-only, never gating.

Per the architecture's core empirical finding (this session's join of the
cad_grade human-vote archive against its own geometry: near-zero
correlation, r=-0.004 to -0.095, between mesh validity/dimensional
accuracy and human preference — raters judge appearance, not topology) and
per the module docstring in ``cadyfiner.oracle.checks``: this leg is
reported as a secondary, explicitly-caveated number, never a gate. It runs
only at final exit-criteria evaluation time, not in the optimizer's hot
loop, so it never needed to be fast.

Rendering uses matplotlib's software 3D rasterizer, not trimesh's built-in
pyglet-backed offscreen renderer — the latter's macOS windowing backend
was found broken for headless use while building this module
(``AttributeError: 'CocoaAlternateEventLoop' object has no
'platform_event_loop'``, both with pyglet 1.5.x and 2.x). matplotlib needs
no windowing system at all, which is the right property for unattended
automation regardless of that specific bug.

Scoring is graded against the ORIGINAL raw prompt, never the refined one —
otherwise a refiner could inflate its own score by rewriting the prompt to
describe whatever it actually built, rather than what the user asked for.
"""

from __future__ import annotations

import base64
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import httpx

_RUBRIC = """You are judging whether a 3D-printed object matches what a user asked for, looking only at
its rendered appearance from two angles — you cannot check exact dimensions or internal structure.

User's request: "{raw_prompt}"

Rate the rendered object on a 1-10 scale:
1-2: unrecognizable or clearly wrong object
3-4: some resemblance but misses the core intent
5-6: recognizable as the right kind of object but with obvious visual problems (broken shape, missing an
     obviously-implied feature, wildly wrong proportions)
7-8: looks like a correct, coherent version of what was asked, minor issues at most
9-10: excellent — exactly what was asked for, clean and well-proportioned

Respond with ONLY a single line: "SCORE: <number>" followed by one sentence of reasoning.
"""

_SCORE_RE = re.compile(r"SCORE:\s*(\d+(?:\.\d+)?)", re.IGNORECASE)


@dataclass
class JudgeResult:
    score: float | None  # None if the response couldn't be parsed
    raw_response: str
    caveat: str = (
        "Single VLM judge, uncalibrated against a broad human panel — this project's own data shows "
        "geometric correctness and human visual preference are near-uncorrelated axes, so treat this as "
        "a directional signal about apparent coherence, not a validated quality score."
    )


def render_views(stl_path: str, out_dir: Path, *, n_views: int = 2) -> list[Path]:
    """Render a mesh from a few fixed angles to PNG files, no windowing system required."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import trimesh
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    mesh = trimesh.load(stl_path, force="mesh")
    bounds = mesh.bounds
    extents = [bounds[1][i] - bounds[0][i] for i in range(3)]
    out_dir.mkdir(parents=True, exist_ok=True)

    angles = [(25, 45), (25, 135)][:n_views]
    paths = []
    for i, (elev, azim) in enumerate(angles):
        fig = plt.figure(figsize=(4, 4), dpi=100)
        ax = fig.add_subplot(111, projection="3d")
        collection = Poly3DCollection(
            mesh.vertices[mesh.faces], facecolor="lightsteelblue", edgecolor="dimgray", linewidths=0.05
        )
        ax.add_collection3d(collection)
        ax.set_xlim(bounds[0][0], bounds[1][0])
        ax.set_ylim(bounds[0][1], bounds[1][1])
        ax.set_zlim(bounds[0][2], bounds[1][2])
        ax.set_box_aspect(extents if all(e > 0 for e in extents) else None)
        ax.view_init(elev=elev, azim=azim)
        ax.set_axis_off()
        plt.tight_layout(pad=0)
        path = out_dir / f"view_{i}.png"
        fig.savefig(path, dpi=100)
        plt.close(fig)
        paths.append(path)
    return paths


def judge(
    stl_path: str,
    raw_prompt: str,
    out_dir: Path,
    *,
    model: str = "gemma4:e4b",
    base_url: str = "http://localhost:11434",
    timeout: float = 300.0,
) -> JudgeResult:
    """Render + score. Requires a vision-capable Ollama model (checked via /api/show's capabilities list)."""

    views = render_views(stl_path, out_dir)
    images_b64 = [base64.b64encode(p.read_bytes()).decode() for p in views]

    payload = {
        "model": model,
        "prompt": _RUBRIC.format(raw_prompt=raw_prompt),
        "images": images_b64,
        "stream": False,
    }
    response = httpx.post(f"{base_url.rstrip('/')}/api/generate", json=payload, timeout=timeout)
    response.raise_for_status()
    text = response.json().get("response", "")

    match = _SCORE_RE.search(text)
    score = float(match.group(1)) if match else None
    return JudgeResult(score=score, raw_response=text)
