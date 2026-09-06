"""Verify each hand-written template against its family's real seed-bank ground truth,
via the exact same sandboxed run_cadquery + evaluate_leg1 pipeline LLM-generated code uses."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cadyfiner.oracle.checks import evaluate_leg1
from cadyfiner.oracle.execute import run_cadquery
from cadyfiner.oracle.templates import TEMPLATES
from cadyfiner.spec import DesignBrief

FAMILIES_DIR = Path(__file__).resolve().parents[1] / "prompts" / "seed_bank" / "families"

only = sys.argv[1] if len(sys.argv) > 1 else None

for path in sorted(FAMILIES_DIR.glob("*.json")):
    family = path.stem
    if only and family != only:
        continue
    if family not in TEMPLATES:
        continue
    items = json.loads(path.read_text())
    gt = DesignBrief.model_validate(items[0]["ground_truth"])  # all tiers share the same ground truth
    code = TEMPLATES[family]()
    result = run_cadquery(code, Path("workspace/verify_templates") / family, timeout_s=30)
    leg1 = evaluate_leg1(result, gt)
    print(f"=== {family} ===")
    print(leg1.feedback_text())
    print(f"OVERALL PASS: {leg1.overall_pass}\n")
