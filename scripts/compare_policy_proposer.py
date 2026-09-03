"""Real end-to-end comparison for Model 2: the general-purpose model's
mutation proposals vs the trained model's, on the SAME real diagnostics,
scored by whether the proposed edit is policy_fixable-correct per this
project's own expert-labeled data (workspace/policy_scenarios.json) —
not the training data itself, to avoid circularity.
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
POLICY_ADAPTER = str(Path(__file__).resolve().parents[1] / "training" / "adapters" / "policy")


def main() -> None:
    scenarios = json.loads(Path("workspace/policy_scenarios.json").read_text())["scenarios"]
    parent = Candidate(depth_policy=DEPTH_POLICY, label="baseline")

    general_correct, trained_correct, n = 0, 0, 0
    for s in scenarios:
        diagnostics = [f"[seed_x] {s['diagnostic_text']}"]
        expected_fixable = bool(s["policy_fixable"])

        general_child = _propose_mutation(parent, diagnostics, ollama_generate, {"model": CADQUERY_MODEL, "temperature": 0.3, "max_tokens": 300, "timeout": 120})
        general_fixable = general_child.label != parent.label  # _propose_mutation returns `parent` unchanged on no-op/failure

        trained_child = _propose_mutation(
            parent, diagnostics, trained_generate,
            {"base_model": "Qwen/Qwen2.5-0.5B-Instruct", "adapter_path": POLICY_ADAPTER, "temperature": 0.1, "max_tokens": 200},
        )
        trained_fixable = trained_child.label != parent.label

        n += 1
        general_correct += int(general_fixable == expected_fixable)
        trained_correct += int(trained_fixable == expected_fixable)
        print(f"[{n}/{len(scenarios)}] expected_fixable={expected_fixable}  general={general_fixable}  trained={trained_fixable}")

    print(f"\ngeneral-purpose model agreement: {general_correct}/{n} = {general_correct/n:.1%}")
    print(f"trained model agreement:         {trained_correct}/{n} = {trained_correct/n:.1%}")


if __name__ == "__main__":
    main()
