"""Real end-to-end comparison for Model 2: the general-purpose model's
mutation proposals vs the trained model's, on the SAME real diagnostics,
scored against a MECHANICALLY-CORRECTED ground truth (real_effect below) —
not the raw "policy_fixable" hand label, which was found (see
docs/TRAINED_OPTIMIZERS.md) to assume an add/remove would change something
without checking it against DEPTH_POLICY's actual current contents: 19 of
24 "policy_fixable=True" labels propose adding a category already present
for that object class, a guaranteed no-op under the real _propose_mutation
mechanics. Also scores "got" by the real depth_policy diff, not by
child.label != parent.label (that label always changes on any non-null
proposal, even a no-op one — see the same finding).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cadyfiner.generators.local_ollama import generate as ollama_generate
from cadyfiner.generators.local_trained import generate as trained_generate
from cadyfiner.optimize import Candidate, _propose_mutation
from cadyfiner.refine_stage2 import DEPTH_POLICY

CADQUERY_MODEL = sys.argv[1] if len(sys.argv) > 1 else "gemma4:e4b"
BASE_URL = sys.argv[2] if len(sys.argv) > 2 else "http://localhost:11434"
POLICY_ADAPTER = str(Path(__file__).resolve().parents[1] / "training" / "adapters" / "policy")


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

    general_correct, trained_correct, n = 0, 0, 0
    g_tp = g_fp = g_tn = g_fn = 0
    t_tp = t_fp = t_tn = t_fn = 0
    for s in scenarios:
        diagnostics = [f"[seed_x] {s['diagnostic_text']}"]
        expected = real_effect(s)

        general_child = _propose_mutation(parent, diagnostics, ollama_generate, {"model": CADQUERY_MODEL, "base_url": BASE_URL, "temperature": 0.3, "max_tokens": 300, "timeout": 120})
        general_real = general_child.depth_policy != parent.depth_policy

        trained_child = _propose_mutation(
            parent, diagnostics, trained_generate,
            {"base_model": "Qwen/Qwen2.5-0.5B-Instruct", "adapter_path": POLICY_ADAPTER, "temperature": 0.1, "max_tokens": 200},
        )
        trained_real = trained_child.depth_policy != parent.depth_policy

        n += 1
        general_correct += int(general_real == expected)
        trained_correct += int(trained_real == expected)
        if expected and general_real:
            g_tp += 1
        elif expected and not general_real:
            g_fn += 1
        elif not expected and general_real:
            g_fp += 1
        else:
            g_tn += 1
        if expected and trained_real:
            t_tp += 1
        elif expected and not trained_real:
            t_fn += 1
        elif not expected and trained_real:
            t_fp += 1
        else:
            t_tn += 1
        print(f"[{n}/{len(scenarios)}] expected_real_effect={expected}  general={general_real}  trained={trained_real}")

    n_real_true = sum(1 for s in scenarios if real_effect(s))
    print(f"\ncorrected ground truth: {n_real_true}/{n} scenarios have a real effect; trivial always-no-effect baseline scores {n - n_real_true}/{n}")
    print(f"general-purpose model agreement: {general_correct}/{n} = {general_correct/n:.1%}  (tp={g_tp} fp={g_fp} tn={g_tn} fn={g_fn})")
    print(f"trained model agreement:         {trained_correct}/{n} = {trained_correct/n:.1%}  (tp={t_tp} fp={t_fp} tn={t_tn} fn={t_fn})")


if __name__ == "__main__":
    main()
