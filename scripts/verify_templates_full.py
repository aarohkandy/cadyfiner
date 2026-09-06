"""Verify the full template pipeline (family detection + parameter mapping) against ALL 21
seed-bank items (all tiers, all families), not just the canonical high-detail prompt --
this is the real test of whether family detection and defaults generalize across phrasing."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cadyfiner.oracle.checks import evaluate_leg1
from cadyfiner.oracle.execute import run_cadquery
from cadyfiner.oracle.templates import build_template_code, detect_family
from cadyfiner.refine import extract
from cadyfiner.spec import DesignBrief

FAMILIES_DIR = Path(__file__).resolve().parents[1] / "prompts" / "seed_bank" / "families"

n_pass, n_total, n_family_miss = 0, 0, 0
for path in sorted(FAMILIES_DIR.glob("*.json")):
    for item in json.loads(path.read_text()):
        n_total += 1
        raw_prompt = item["raw_prompt"]
        gt = DesignBrief.model_validate(item["ground_truth"])

        detected = detect_family(raw_prompt)
        if detected != item["family"]:
            n_family_miss += 1
            print(f"[{item['id']}] FAMILY DETECTION MISMATCH: detected={detected!r} actual={item['family']!r}")
            continue

        extraction = extract(raw_prompt)
        code = build_template_code(detected, extraction.spec)
        result = run_cadquery(code, Path("workspace/verify_templates_full") / item["id"], timeout_s=30)
        leg1 = evaluate_leg1(result, gt)
        status = "PASS" if leg1.overall_pass else f"FAIL ({leg1.stopped_at})"
        print(f"[{item['id']}] {status}")
        if leg1.overall_pass:
            n_pass += 1
        else:
            print(f"    {leg1.feedback_text()}")

print(f"\n{n_pass}/{n_total} full pass, {n_family_miss} family-detection misses")
