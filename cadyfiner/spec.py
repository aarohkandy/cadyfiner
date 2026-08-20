"""The structured spec object cadyfiner extracts from and renders prompts into.

Shaped after ai-cad's ``DesignBrief``/``TargetDimensions``
(``ai-cad/backend/app/models/schemas.py``), which is a clean, minimal slot
model worth keeping compatible with. Two changes from that source:

- ``required_features: list[str]`` (free text) is replaced with
  ``features: list[Feature]`` (structured count/size), because Leg-1
  spec-conformance checking needs a number to compare against a measured
  mesh, not a sentence.
- ``assumptions_made`` is new. Every dimension or feature the refiner
  invents rather than reads from the user prompt is logged here, so the
  refiner is auditable rather than silently opinionated.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TargetDimensions(StrictModel):
    """Overall envelope dimensions, in ``DesignBrief.units``.

    ``length``/``thickness`` extend ai-cad's original width/depth/height/
    diameter set: wall thickness is a first-class, frequently-stated slot
    in this domain (it's the rung where cad_grade's wall_planter data shows
    the sharpest reliability/fidelity tradeoff), not a derived quantity.
    """

    width: float | None = None
    depth: float | None = None
    height: float | None = None
    diameter: float | None = None
    length: float | None = None
    thickness: float | None = None


class Feature(StrictModel):
    """A single machine-checkable geometric requirement.

    ``kind`` is a free string (e.g. "hole", "drainage_hole", "mounting_hole",
    "tooth", "fillet", "slot", "rib") rather than an enum, because the
    mechanical/functional scope (gears, brackets, enclosures) has feature
    vocabulary that a fixed enum would constantly fall short of. What makes
    a feature *checkable* is having a ``count`` and/or ``size`` — a feature
    with neither is still recorded (for prompt fidelity) but Leg-1 cannot
    verify it and will say so rather than silently passing.
    """

    kind: str
    count: int | None = None
    size: float | None = None
    size_unit: Literal["mm", "cm", "in"] | None = None
    pattern: str | None = None
    notes: list[str] = Field(default_factory=list)


class DesignBrief(StrictModel):
    prompt: str
    refined_prompt: str | None = None
    units: Literal["mm", "cm", "in"] = "mm"
    object_class: Literal["decorative", "mechanical_functional"] | None = None
    target_dims: TargetDimensions = Field(default_factory=TargetDimensions)
    features: list[Feature] = Field(default_factory=list)
    tolerances: dict[str, float] | None = None
    process_notes: list[str] = Field(default_factory=list)
    style_notes: list[str] = Field(default_factory=list)
    assumptions_made: list[str] = Field(default_factory=list)

    def stated_dimensions(self) -> dict[str, float]:
        """Dimension slots that carry a value, keyed by field name."""

        return {
            name: value
            for name, value in self.target_dims.model_dump().items()
            if value is not None
        }

    def checkable_features(self) -> list[Feature]:
        """Features with enough structure that a check *could* be built for them.

        Note this is broader than what ``cadyfiner.oracle.checks`` actually
        verifies today: it checks ``count`` for hole-kind and tooth/gear-kind
        features via mesh-derived estimates (genus-based hole counting,
        FFT-based radial-feature counting), and nothing else — ``size`` is
        recorded but not yet independently measured against any geometry.
        An adversarial review found this docstring previously claimed
        "count and/or size" as the checkable contract while ``checks.py``
        used a narrower, undocumented inline filter, letting the two drift.
        This property is a coarse "has structure worth eventually checking"
        filter for spec authors, not a promise of what's verified now —
        see ``evaluate_leg1`` in ``cadyfiner/oracle/checks.py`` for the
        actual, current verification behavior.
        """

        return [f for f in self.features if f.count is not None or f.size is not None]
