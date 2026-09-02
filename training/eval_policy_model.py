"""Quick evaluation of the trained policy-mutation-proposer against its own held-out split.

Not a substitute for the real end-to-end comparison (running it inside
optimize.py against the general-purpose model, per docs/TRAINED_OPTIMIZERS.md
section 5) — this just checks the model learned SOMETHING sane before
spending time on that more expensive comparison: does it produce valid
JSON, does it pick one of the 7 real tags, does its policy_fixable-style
judgment (add/remove both null vs not) roughly track the expert labels.
"""

from __future__ import annotations

import json
import sys

sys.path.insert(0, ".")
from local_trained import generate  # noqa: E402

VALID_TAGS = {"identity", "function", "interface", "dimensions", "process", "topology", "feature_placement"}


def main() -> None:
    data_path, base_model, adapter_path = sys.argv[1], sys.argv[2], sys.argv[3]
    rows = [json.loads(line) for line in open(data_path)]
    eval_rows = rows[: max(1, int(len(rows) * 0.15))]  # matches train_lora.py's holdout split

    n_valid_json, n_valid_tag, n_agree_fixable = 0, 0, 0
    for i, row in enumerate(eval_rows):
        raw = generate(row["input"], base_model=base_model, adapter_path=adapter_path, temperature=0.1, max_tokens=200)
        expected = row["output"]
        try:
            start, end = raw.index("{"), raw.rindex("}") + 1
            parsed = json.loads(raw[start:end])
            n_valid_json += 1
        except (ValueError, json.JSONDecodeError):
            parsed = None

        tag = (parsed.get("add") or parsed.get("remove")) if parsed else None
        if tag is None or (isinstance(tag, str) and tag in VALID_TAGS):
            n_valid_tag += 1

        expected_fixable = bool(expected.get("add") or expected.get("remove"))
        got_fixable = bool(parsed and (parsed.get("add") or parsed.get("remove"))) if parsed else False
        if expected_fixable == got_fixable:
            n_agree_fixable += 1

        print(f"[{i+1}/{len(eval_rows)}] expected={{'add':{expected.get('add')!r},'remove':{expected.get('remove')!r}}} got={raw[:150]!r}")

    n = len(eval_rows)
    print(f"\nvalid_json={n_valid_json}/{n}  valid_tag_or_null={n_valid_tag}/{n}  agrees_on_fixable={n_agree_fixable}/{n}")


if __name__ == "__main__":
    main()
