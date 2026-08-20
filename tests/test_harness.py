from __future__ import annotations

from cadyfiner.harness import PairedResult, _wilson_ci, sign_test_p_value, summarize


def _pair(seed_id, family, role, raw_pass, refined_pass, rep=0, used_refined_fallback=False):
    return PairedResult(
        seed_id=f"{seed_id}_rep{rep}", base_seed_id=seed_id, family=family, role=role,
        raw_pass=raw_pass, refined_pass=refined_pass, raw_detail="", refined_detail="",
        used_refined_fallback=used_refined_fallback,
    )


class TestSignTestPValue:
    """Cross-checked against scipy.stats.binomtest during development; scipy is not a
    dependency of this package so the reference values are pinned here instead."""

    def test_matches_known_scipy_values(self):
        cases = [
            (5, 0, 0.062500),
            (8, 2, 0.109375),
            (6, 4, 0.753906),
            (3, 2, 1.000000),
            (10, 10, 1.000000),
            (15, 5, 0.041389),
        ]
        for wins, losses, expected in cases:
            assert abs(sign_test_p_value(wins, losses) - expected) < 1e-5

    def test_zero_pairs_returns_one(self):
        assert sign_test_p_value(0, 0) == 1.0


class TestWilsonCI:
    """Cross-checked against known textbook Wilson score interval values."""

    def test_matches_known_reference_values(self):
        cases = [
            (5, 10, 0.237, 0.763),
            (8, 10, 0.492, 0.943),
            (0, 10, 0.0, 0.278),
        ]
        for k, n, exp_lo, exp_hi in cases:
            lo, hi = _wilson_ci(k, n)
            assert abs(lo - exp_lo) < 0.01
            assert abs(hi - exp_hi) < 0.01


class TestSummarize:
    def test_ground_truth_used_for_both_arms_means_win_requires_decisive_pass(self):
        results = [
            _pair("s1", "fam", "heldout", raw_pass=False, refined_pass=True),
            _pair("s2", "fam", "heldout", raw_pass=True, refined_pass=False),
            _pair("s3", "fam", "heldout", raw_pass=True, refined_pass=True),
        ]
        report = summarize(results)
        assert report.refined_wins == 1
        assert report.raw_wins == 1
        assert report.ties == 1
        assert report.n_pairs == 3

    def test_all_ties_is_inconclusive(self):
        results = [_pair("s1", "fam", "heldout", raw_pass=True, refined_pass=True)]
        report = summarize(results)
        assert "INCONCLUSIVE" in report.verdict

    def test_by_family_breakdown_sums_correctly(self):
        results = [
            _pair("s1", "fam_a", "heldout", raw_pass=False, refined_pass=True),
            _pair("s2", "fam_b", "heldout", raw_pass=True, refined_pass=False),
        ]
        report = summarize(results)
        assert report.by_family["fam_a"]["refined_wins"] == 1
        assert report.by_family["fam_b"]["raw_wins"] == 1

    def test_verdict_agrees_with_printed_p_value_at_small_n(self):
        """Regression: PASS/NOT-SUPPORTED used to be decided by the Wilson CI lower bound, which
        disagrees with the exact sign-test p-value it also prints in the same sentence — a 4/4
        sweep gives CI=[51%,100%] (would say PASS) but exact p=0.125 (not significant at alpha=0.05)."""
        results = [_pair(f"s{i}", "fam", "heldout", raw_pass=False, refined_pass=True) for i in range(4)]
        report = summarize(results)
        assert report.sign_test_p > 0.05
        assert "PASS" not in report.verdict.split(":")[0]  # must not claim PASS when p is not significant
        assert "NOT SUPPORTED" in report.verdict

    def test_pass_verdict_when_actually_significant(self):
        results = [_pair(f"s{i}", "fam", "heldout", raw_pass=False, refined_pass=True) for i in range(8)]
        report = summarize(results)
        assert report.sign_test_p < 0.05
        assert report.verdict.startswith("PASS")

    def test_significant_regression_is_flagged_not_called_underpowered(self):
        """Regression: a significant regression (raw beats refined) fell into the same
        'treat as underpowered' text as a genuinely inconclusive result."""
        results = [_pair(f"s{i}", "fam", "heldout", raw_pass=True, refined_pass=False) for i in range(8)]
        report = summarize(results)
        assert report.sign_test_p < 0.05
        assert report.verdict.startswith("FAILS")
        assert "not noise" in report.verdict

    def test_fallback_pairs_excluded_from_decisive_stats(self):
        """Regression: a pair where Stage 2 couldn't produce a usable refined prompt (both arms
        got byte-identical raw text) was still counted as a legitimate decisive win/loss."""
        results = [
            _pair("s1", "fam", "heldout", raw_pass=False, refined_pass=True),
            _pair("s2", "fam", "heldout", raw_pass=False, refined_pass=True, used_refined_fallback=True),
        ]
        report = summarize(results)
        assert report.n_excluded_fallback == 1
        assert report.n_distinct_seeds == 1
        assert report.refined_wins == 1

    def test_reps_are_clustered_by_seed_not_treated_as_independent_trials(self):
        """Regression: replaying the same small seed pool more times (reps>1) turned a
        non-significant result into an arbitrarily 'significant' one, since every rep was scored
        as an independent trial with no seed-clustering correction."""
        # 3 distinct seeds, each replayed 20 times with an identical 60%/40% split per seed —
        # zero new design challenges tested, only noise-averaging within each seed.
        results = []
        for seed_idx in range(3):
            for rep in range(20):
                refined_pass = rep < 12  # 12/20 = 60% within this seed
                results.append(_pair(f"s{seed_idx}", "fam", "heldout", raw_pass=not refined_pass, refined_pass=refined_pass, rep=rep))
        report = summarize(results)
        assert report.n_distinct_seeds == 3  # not 60
        assert report.n_pairs == 60

    def test_majority_vote_within_seed(self):
        results = [
            _pair("s1", "fam", "heldout", raw_pass=False, refined_pass=True, rep=0),
            _pair("s1", "fam", "heldout", raw_pass=False, refined_pass=True, rep=1),
            _pair("s1", "fam", "heldout", raw_pass=True, refined_pass=False, rep=2),
        ]
        report = summarize(results)
        assert report.n_distinct_seeds == 1
        assert report.refined_wins == 1  # 2 of 3 reps were refined wins -> seed counts as one refined win
