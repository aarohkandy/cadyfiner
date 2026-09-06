from __future__ import annotations

import cadyfiner.oracle.self_repair as self_repair_module
from cadyfiner.oracle.execute import ExecutionResult
from cadyfiner.oracle.self_repair import generate_and_execute_with_repair

GOOD_CODE = '```python\nimport cadquery as cq\nresult = cq.Workplane("XY").box(10, 10, 10)\n```'
BAD_CODE = '```python\nimport cadquery as cq\nresult = cq.Workplane("XY").box(10, 10, 10).nonexistent_method()\n```'
BANNED_CODE = '```python\nimport cadquery as cq\nimport subprocess\nresult = cq.Workplane("XY").box(1, 1, 1)\n```'


class TestGenerateAndExecuteWithRepair:
    def test_succeeds_first_try_without_repair(self, tmp_path):
        calls = []

        def fake_generate(prompt, **kwargs):
            calls.append(prompt)
            return GOOD_CODE

        result = generate_and_execute_with_repair("design a box", fake_generate, {}, tmp_path)
        assert result.ok
        assert result.repair_attempts == 0
        assert len(calls) == 1

    def test_repairs_after_runtime_exception(self, tmp_path):
        calls = []

        def fake_generate(prompt, **kwargs):
            calls.append(prompt)
            return BAD_CODE if len(calls) == 1 else GOOD_CODE

        result = generate_and_execute_with_repair("design a box", fake_generate, {}, tmp_path)
        assert result.ok
        assert result.repair_attempts == 1
        assert len(calls) == 2
        repair_prompt = calls[1]
        # In the combined variant, api_check catches this hallucinated call before the
        # subprocess does -- error_type is "api_check_rejected" with a more specific message,
        # not the generic "exception: AttributeError" a bare subprocess run would produce.
        # Exact error text and previous code must both be present either way.
        assert "nonexistent_method" in repair_prompt
        assert "design a box" in repair_prompt  # original prompt carried forward

    def test_gives_up_after_max_repairs(self, tmp_path):
        calls = []

        def fake_generate(prompt, **kwargs):
            calls.append(prompt)
            return BAD_CODE

        result = generate_and_execute_with_repair("design a box", fake_generate, {}, tmp_path, max_repairs=1)
        assert not result.ok
        assert result.repair_attempts == 1
        assert len(calls) == 2  # original + exactly 1 repair, then stop

    def test_prefilter_rejection_gets_hard_rule_framing(self, tmp_path):
        calls = []

        def fake_generate(prompt, **kwargs):
            calls.append(prompt)
            return BANNED_CODE if len(calls) == 1 else GOOD_CODE

        result = generate_and_execute_with_repair("design a box", fake_generate, {}, tmp_path)
        assert result.ok
        assert result.repair_attempts == 1
        assert "hard rule" in calls[1].lower()
        assert "subprocess" in calls[1]

    def test_non_repairable_error_type_is_not_retried(self, tmp_path, monkeypatch):
        """timeout/no_result_file/etc. are resource or environment failures, not code bugs
        a regenerated attempt can fix -- confirm the loop doesn't burn a repair on one."""

        def fake_run_cadquery(code, out_dir, **kwargs):
            return ExecutionResult(ok=False, error_type="timeout", error_message="exceeded 45s", code=code)

        monkeypatch.setattr(self_repair_module, "run_cadquery", fake_run_cadquery)

        calls = []

        def fake_generate(prompt, **kwargs):
            calls.append(prompt)
            return GOOD_CODE

        result = generate_and_execute_with_repair("design a box", fake_generate, {}, tmp_path)
        assert not result.ok
        assert result.repair_attempts == 0
        assert len(calls) == 1
