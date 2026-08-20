"""The exit-criteria harness: paired raw-vs-refined comparison with statistics.

Matches the DOE lesson the "7/10 Stars Please" preprint this project is
built on had to learn the hard way — a pooled/unpaired comparison was
initially misleading until Bradley-Terry schedule-adjustment corrected for
uneven opponent strength. Same-seed, same-generator pairing sidesteps that
class of confound entirely: every comparison is refined-prompt-for-seed-X
vs raw-prompt-for-seed-X, on the same backend, so any difference is
attributable to the refiner rather than to which seeds happened to be
easier.

Uses the exact same Leg-1 composite (:func:`cadyfiner.oracle.checks.
evaluate_leg1`) the optimizer trains against — one definition of "pass,"
per that module's own design note, so training signal and the final
reported result can't silently drift apart.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from cadyfiner.oracle.checks import evaluate_leg1
from cadyfiner.oracle.execute import CADQUERY_PROMPT_RULES, extract_code, run_cadquery
from cadyfiner.refine import extract
from cadyfiner.refine_stage2 import fill_gaps
from cadyfiner.spec import DesignBrief


@dataclass
class SeedCase:
    id: str
    family: str
    raw_prompt: str
    ground_truth: DesignBrief
    role: str  # "train" | "heldout_same_family" | "heldout_family"


@dataclass
class PairedResult:
    seed_id: str
    base_seed_id: str  # seed_id without the "_repN" suffix — the unit statistics should cluster on
    family: str
    role: str
    raw_pass: bool
    refined_pass: bool
    raw_detail: str
    refined_detail: str
    used_refined_fallback: bool = False  # True if Stage 2 couldn't produce a usable refined prompt
    # and both arms received byte-identical prompt text — see run_paired_evaluation's docstring.


def _run_one(prompt_text: str, ground_truth: DesignBrief, generate: Callable, out_dir: Path, generate_kwargs: dict):
    full_prompt = CADQUERY_PROMPT_RULES + f"\nDesign request:\n{prompt_text}\n"
    raw_output = generate(full_prompt, **generate_kwargs)
    code = extract_code(raw_output)
    execution = run_cadquery(code, out_dir, timeout_s=90)
    return evaluate_leg1(execution, ground_truth)


def run_paired_evaluation(
    seeds: list[SeedCase],
    generate: Callable[..., str],
    *,
    reps: int = 1,
    out_root: Path = Path("workspace/harness"),
    generate_kwargs: dict | None = None,
) -> list[PairedResult]:
    """Run every seed both raw and refined, same generator, same ground truth for both arms.

    Ground truth is used for BOTH arms deliberately (see checks.py's module
    docstring) — a refiner cannot win by simply declining to state
    anything, since the raw arm is scored against the real target too, not
    against "whatever the raw prompt happened to say."

    Persists after every seed, and skips (rather than aborts on) a seed
    whose generator call fails outright — found live: a run crashed on an
    unhandled Ollama timeout on the very first seed, losing all results
    including from work already done, because the file was previously only
    written once at the very end. Local generation calls on CPU-only
    inference routinely take 60-250s each and Ollama serializes requests to
    one loaded model, so a timeout under any concurrent load is a real,
    expected occasional outcome, not an exceptional one.

    If Stage 2 cannot produce a usable refined prompt, the "refined" arm
    falls back to sending the same raw prompt text as the "raw" arm — an
    adversarial review found this degenerate pair was still counted as a
    legitimate decisive win/loss by ``summarize()``. Each such pair is
    flagged via ``used_refined_fallback`` so ``summarize()`` can exclude it
    from the significance test while still reporting how often it happened
    (a high fallback rate is itself important diagnostic signal about
    Stage 2's reliability).
    """

    generate_kwargs = generate_kwargs or {}
    out_root.mkdir(parents=True, exist_ok=True)
    results_path = out_root / "paired_results.json"
    results: list[PairedResult] = []
    skipped: list[str] = []

    for seed in seeds:
        for rep in range(reps):
            seed_id = f"{seed.id}_rep{rep}"
            try:
                raw_leg1 = _run_one(
                    seed.raw_prompt, seed.ground_truth, generate,
                    out_root / f"{seed_id}_raw", generate_kwargs,
                )

                extraction = extract(seed.raw_prompt)
                filled = fill_gaps(extraction, generate, **generate_kwargs)
                # fill_gaps() already treats a missing/empty refined_prompt as a failed attempt
                # internally (see refine_stage2.py), but after every retry is exhausted its
                # fallback spec still carries prompt==raw_prompt — this equality check catches
                # that fallback regardless of exactly how it was produced.
                used_fallback = (not filled.spec.refined_prompt) or (filled.spec.refined_prompt == seed.raw_prompt)
                refined_prompt_text = filled.spec.refined_prompt or seed.raw_prompt
                refined_leg1 = _run_one(
                    refined_prompt_text, seed.ground_truth, generate,
                    out_root / f"{seed_id}_refined", generate_kwargs,
                )
            except Exception as exc:  # noqa: BLE001 — generator/network failures are expected, not exceptional, here
                skipped.append(f"{seed_id}: {type(exc).__name__}: {exc}")
                (out_root / "skipped.json").write_text(json.dumps(skipped, indent=2))
                continue

            results.append(
                PairedResult(
                    seed_id=seed_id,
                    base_seed_id=seed.id,
                    family=seed.family,
                    role=seed.role,
                    raw_pass=raw_leg1.overall_pass,
                    refined_pass=refined_leg1.overall_pass,
                    raw_detail=raw_leg1.feedback_text(),
                    refined_detail=refined_leg1.feedback_text(),
                    used_refined_fallback=used_fallback,
                )
            )
            results_path.write_text(json.dumps([r.__dict__ for r in results], indent=2))

    if skipped:
        print(f"WARNING: {len(skipped)} seed(s) skipped due to generator/execution failures — see skipped.json")
    return results


def _wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def sign_test_p_value(wins: int, losses: int) -> float:
    """Two-sided exact sign-test p-value (binomial, p=0.5), no scipy dependency needed."""

    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)

    def binom_coeff(n: int, k: int) -> int:
        return math.comb(n, k)

    tail = sum(binom_coeff(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


@dataclass
class ExitCriteriaReport:
    n_pairs: int
    n_distinct_seeds: int
    n_excluded_fallback: int
    refined_wins: int
    raw_wins: int
    ties: int
    win_rate_excluding_ties: float
    win_rate_ci95: tuple[float, float]
    sign_test_p: float
    by_family: dict = field(default_factory=dict)
    by_role: dict = field(default_factory=dict)
    verdict: str = ""


def _outcome(r: PairedResult) -> str:
    if r.refined_pass and not r.raw_pass:
        return "refined_win"
    if r.raw_pass and not r.refined_pass:
        return "raw_win"
    return "tie"


def _cluster_by_seed(results: list[PairedResult]) -> list[str]:
    """Collapse every rep of one seed into a single outcome via majority vote.

    Regression: treating every (seed, rep) pair as an independent Bernoulli
    trial let ``reps`` inflate apparent statistical significance without
    testing a single new design challenge — replaying the same seed pool
    50x at a fixed win ratio turned a non-significant p=0.245 (n=60) into
    p<1e-18 (n=3000), verified live while fixing this. Grouping by
    ``base_seed_id`` first makes the unit of the significance test "one
    distinct design challenge," which is what the paired-comparison design
    is actually meant to test; reps still reduce per-seed noise via the
    majority vote, they just can't manufacture sample size.
    """

    from collections import Counter, defaultdict

    by_seed: dict[str, list[PairedResult]] = defaultdict(list)
    for r in results:
        by_seed[r.base_seed_id].append(r)

    outcomes = []
    for seed_id, reps in by_seed.items():
        counts = Counter(_outcome(r) for r in reps)
        top = counts.most_common()
        if len(top) > 1 and top[0][1] == top[1][1]:
            outcomes.append("tie")  # no majority: e.g. 1 win, 1 loss across 2 reps
        else:
            outcomes.append(top[0][0])
    return outcomes


def summarize(results: list[PairedResult]) -> ExitCriteriaReport:
    fallback_results = [r for r in results if r.used_refined_fallback]
    usable_results = [r for r in results if not r.used_refined_fallback]

    outcomes = _cluster_by_seed(usable_results)
    refined_wins = outcomes.count("refined_win")
    raw_wins = outcomes.count("raw_win")
    ties = outcomes.count("tie")
    n_decisive = refined_wins + raw_wins

    win_rate = refined_wins / n_decisive if n_decisive else float("nan")
    ci = _wilson_ci(refined_wins, n_decisive) if n_decisive else (float("nan"), float("nan"))
    p = sign_test_p_value(refined_wins, raw_wins) if n_decisive else float("nan")

    def _breakdown(key_fn) -> dict[str, dict]:
        # Built from usable_results (not the seed-clustered outcomes) since family/role are
        # per-result metadata, not per-seed — a seed only ever has one family/role anyway, so
        # this still reports one bucket per distinct seed's clustered-equivalent behavior in
        # aggregate, just without re-deriving a separate clustering pass per breakdown.
        out: dict[str, dict] = {}
        for r in usable_results:
            b = out.setdefault(key_fn(r), {"refined_wins": 0, "raw_wins": 0, "ties": 0, "n": 0})
            b["n"] += 1
            outcome = _outcome(r)
            b["refined_wins" if outcome == "refined_win" else "raw_wins" if outcome == "raw_win" else "ties"] += 1
        return out

    by_family = _breakdown(lambda r: r.family)
    by_role = _breakdown(lambda r: r.role)

    # Regression: the verdict used to be decided by the Wilson CI lower bound (a normal
    # approximation) while a DIFFERENT criterion — the exact sign-test p-value — was computed and
    # printed in the very same sentence. The two disagree at small n, including exactly the sample
    # sizes this project's own seed bank produces (n=4 all-wins: CI=[51%,100%] says PASS, exact
    # p=0.125 says not significant). The exact test is now the actual decision criterion; Wilson CI
    # is reported for context only.
    SIGNIFICANCE_ALPHA = 0.05
    if n_decisive == 0:
        verdict = "INCONCLUSIVE: no decisive pairs (every pair tied pass/fail)"
    elif refined_wins > raw_wins and p < SIGNIFICANCE_ALPHA:
        verdict = f"PASS: refined wins {win_rate:.0%} of decisive pairs (n={n_decisive} distinct seeds), 95% CI [{ci[0]:.0%}, {ci[1]:.0%}] (p={p:.3f})"
    elif raw_wins > refined_wins and p < SIGNIFICANCE_ALPHA:
        # Regression: this branch didn't exist — a significant regression fell into the same
        # "NOT SUPPORTED... treat as underpowered" text as a genuinely inconclusive result,
        # actively telling the reader to disregard real detected evidence of harm as noise.
        verdict = (
            f"FAILS: refined performs significantly WORSE than raw — win rate {win_rate:.0%} "
            f"(n={n_decisive} distinct seeds), 95% CI [{ci[0]:.0%}, {ci[1]:.0%}] (p={p:.3f}) — "
            f"this is a detected regression, not noise"
        )
    else:
        verdict = (
            f"NOT SUPPORTED at this sample size: win rate {win_rate:.0%}, 95% CI [{ci[0]:.0%}, {ci[1]:.0%}] "
            f"(p={p:.3f}) — n={n_decisive} distinct-seed decisive pairs is small; do not treat as a "
            f"negative result, treat as underpowered"
        )

    if fallback_results:
        verdict += f" [{len(fallback_results)} pair(s) excluded: Stage 2 could not produce a usable refined prompt]"

    return ExitCriteriaReport(
        n_pairs=len(results),
        n_distinct_seeds=len(outcomes),
        n_excluded_fallback=len(fallback_results),
        refined_wins=refined_wins,
        raw_wins=raw_wins,
        ties=ties,
        win_rate_excluding_ties=win_rate,
        win_rate_ci95=ci,
        sign_test_p=p,
        by_family=by_family,
        by_role=by_role,
        verdict=verdict,
    )
