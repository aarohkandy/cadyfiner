"""Quick, focused test of the _propose_mutation prompt fix (see git history on optimize.py).

Two corrections vs. the first version of this script, both discovered by inspecting the real
scenario data against the REAL current DEPTH_POLICY:

1. "did the model propose a fixable edit" was measured by comparing child.label != parent.label.
   That's wrong: _propose_mutation() builds a new label string whenever object_class/add/remove
   are non-null, EVEN IF the add/remove is a no-op against the current policy (e.g. adding
   "dimensions" to mechanical_functional, which already has it). Corrected to compare
   child.depth_policy != parent.depth_policy -- the only thing that actually matters for
   run_optimizer's beam search.

2. The hand-labeled "policy_fixable" ground truth itself assumes an add/remove would change
   something, without checking against DEPTH_POLICY's CURRENT contents. Checked directly: 19 of
   24 "policy_fixable=True" labels propose adding a category ALREADY present in that object
   class's current list -- a guaranteed no-op. Only 5/50 scenarios (all "decorative" + add
   topology/feature_placement) would actually change anything today. This script computes and
   reports BOTH the original label-based agreement (for comparison to prior results) and the
   corrected, mechanically-grounded one.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cadyfiner.generators.local_ollama import generate as ollama_generate
from cadyfiner.optimize import Candidate, _propose_mutation
from cadyfiner.refine_stage2 import DEPTH_POLICY

CADQUERY_MODEL = sys.argv[1] if len(sys.argv) > 1 else "gemma4:e4b"
BASE_URL = sys.argv[2] if len(sys.argv) > 2 else "http://localhost:11434"


def real_effect(scenario: dict) -> bool:
    oc = scenario["object_class"]
    policy = DEPTH_POLICY.get(oc, [])
    add, remove = scenario.get("add") or "", scenario.get("remove") or ""
    would_add = bool(add) and add not in policy
    would_remove = bool(remove) and remove in policy
    return would_add or would_remove


def main() -> None:
    scenarios = json.loads(Path("workspace/policy_scenarios.json").read_text())["scenarios"]
    parent = Candidate(depth_policy=DEPTH_POLICY, label="baseline")

    correct_label, correct_real, n = 0, 0, 0
    tp = fp = tn = fn = 0  # against the CORRECTED (real_effect) ground truth
    for s in scenarios:
        diagnostics = [f"[seed_x] {s['diagnostic_text']}"]
        expected_label = bool(s["policy_fixable"])
        expected_real = real_effect(s)

        child = _propose_mutation(parent, diagnostics, ollama_generate, {"model": CADQUERY_MODEL, "base_url": BASE_URL, "temperature": 0.3, "max_tokens": 300, "timeout": 120})
        got_label_diff = child.label != parent.label  # the OLD, flawed proxy
        got_real = child.depth_policy != parent.depth_policy  # the CORRECT check

        n += 1
        correct_label += int(got_label_diff == expected_label)
        correct_real += int(got_real == expected_real)
        if expected_real and got_real:
            tp += 1
        elif expected_real and not got_real:
            fn += 1
        elif not expected_real and got_real:
            fp += 1
        else:
            tn += 1
        print(f"[{n}/{len(scenarios)}] expected_label={expected_label} expected_real={expected_real} got_real={got_real}  {'OK' if got_real==expected_real else 'WRONG'}")

    print(f"\nagreement vs ORIGINAL hand labels (label-diff proxy, comparable to the first test run): {correct_label}/{n} = {correct_label/n:.1%}")
    print(f"agreement vs CORRECTED real-effect ground truth (real depth_policy diff): {correct_real}/{n} = {correct_real/n:.1%}")
    print(f"confusion vs corrected ground truth: tp={tp} fp={fp} tn={tn} fn={fn}  (positive=has a real effect)")
    n_real_true = sum(1 for s in scenarios if real_effect(s))
    print(f"corrected ground truth: {n_real_true}/{n} scenarios have a real effect; trivial always-no-effect baseline scores {n - n_real_true}/{n}")


if __name__ == "__main__":
    main()
