"""Native reflective optimizer for the refiner.

Deliberately NOT a GEPA/DSPy dependency — an adversarial Plan-agent review
of the original architecture did the throughput math and found GEPA's
headline efficiency numbers are measured against GPU-cluster multi-module
DSPy programs, a different regime than optimizing one prompt/rule-set over
tens of evaluations on a CPU-only local box (~30-90s per CadQuery
generation attempt here). What's actually valuable from that lineage —
feeding rich textual failure diagnostics to an LLM that proposes a
mutation, instead of collapsing everything to a scalar reward — is
implemented natively below in a few hundred lines: a small beam of
candidate depth-policy variants, scored via the SAME Leg-1 composite
function used by the exit-criteria harness (never a separate metric — see
``cadyfiner.oracle.checks``'s module docstring for why that consistency
matters), with failures fed back as text to a single "propose a mutation"
LLM call each round.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from cadyfiner.oracle.checks import evaluate_leg1
from cadyfiner.oracle.execute import CADQUERY_PROMPT_RULES, extract_code, run_cadquery
from cadyfiner.refine import extract
from cadyfiner.refine_stage2 import DEPTH_POLICY, fill_gaps
from cadyfiner.spec import DesignBrief


@dataclass
class Candidate:
    """One point in the search space: a depth policy per object class."""

    depth_policy: dict[str, list[str]]
    label: str
    score: float | None = None
    replay_score: float | None = None
    diagnostics: list[str] = field(default_factory=list)
    history: list[str] = field(default_factory=list)


@dataclass
class TrainSeed:
    id: str
    raw_prompt: str
    ground_truth: DesignBrief


def _score_candidate(
    candidate: Candidate,
    seeds: list[TrainSeed],
    generate: Callable[..., str],
    out_root: Path,
    generate_kwargs: dict,
) -> tuple[float, list[str]]:
    """Run every training seed through this candidate's policy, return (pass_rate, diagnostics)."""

    import cadyfiner.refine_stage2 as stage2_module

    original_policy = stage2_module.DEPTH_POLICY
    stage2_module.DEPTH_POLICY = candidate.depth_policy
    diagnostics: list[str] = []
    passes = 0
    try:
        for seed in seeds:
            extraction = extract(seed.raw_prompt)
            filled = fill_gaps(extraction, generate, **generate_kwargs)
            prompt = CADQUERY_PROMPT_RULES + f"\nDesign request:\n{filled.spec.refined_prompt or seed.raw_prompt}\n"
            raw_output = generate(prompt, **generate_kwargs)
            code = extract_code(raw_output)
            execution = run_cadquery(code, out_root / candidate.label / seed.id, timeout_s=60)
            result = evaluate_leg1(execution, seed.ground_truth)
            if result.overall_pass:
                passes += 1
            else:
                diagnostics.append(f"[{seed.id}] {result.feedback_text()}")
    finally:
        stage2_module.DEPTH_POLICY = original_policy

    return passes / len(seeds) if seeds else 0.0, diagnostics


def _propose_mutation(
    parent: Candidate, diagnostics: list[str], generate: Callable[..., str], generate_kwargs: dict
) -> Candidate:
    """One LLM call: given failure diagnostics, propose a depth-policy mutation."""

    prompt = f"""You are tuning a CAD prompt-refiner's depth policy: a list of information
categories (from: identity, function, interface, dimensions, process, topology, feature_placement)
the refiner is allowed to add per object class.

Current policy:
{json.dumps(parent.depth_policy, indent=2)}

Recent failures when generating from this policy (each line: [seed_id] check stage that failed and why):
{chr(10).join(diagnostics[:15])}

Propose ONE change to the policy that might fix some of these failures — add or remove ONE category
from ONE object class's list. Output ONLY a JSON object: {{"object_class": "...", "add": "..." or null, "remove": "..." or null}}
"""
    raw = generate(prompt, **generate_kwargs)
    try:
        start, end = raw.index("{"), raw.rindex("}") + 1
        data = json.loads(raw[start:end])
    except (ValueError, json.JSONDecodeError):
        return parent  # malformed proposal: no-op mutation, parent survives unchanged

    child_policy = copy.deepcopy(parent.depth_policy)
    oc = data.get("object_class")
    if oc not in child_policy:
        return parent
    if data.get("add") and data["add"] not in child_policy[oc]:
        child_policy[oc].append(data["add"])
    if data.get("remove") and data["remove"] in child_policy[oc]:
        child_policy[oc].remove(data["remove"])

    return Candidate(
        depth_policy=child_policy,
        label=f"{parent.label}->{oc}:{data.get('add') or ''}{'-' if data.get('remove') else ''}{data.get('remove') or ''}",
        history=[*parent.history, f"mutated from {parent.label}"],
    )


def run_optimizer(
    train_seeds: list[TrainSeed],
    replay_seeds: list[TrainSeed],
    generate: Callable[..., str],
    *,
    rounds: int = 5,
    beam_size: int = 3,
    out_root: Path = Path("workspace/optimize"),
    generate_kwargs: dict | None = None,
) -> Candidate:
    """Beam search over depth-policy variants, gated by no-regression on a held-out replay set.

    Returns the best-scoring candidate found. ``replay_seeds`` must be
    disjoint from ``train_seeds`` — scoring a mutation against the same
    seeds it was diagnosed on would let it overfit the exact failures in
    front of it rather than generalize.
    """

    generate_kwargs = generate_kwargs or {}
    out_root.mkdir(parents=True, exist_ok=True)
    seed = Candidate(depth_policy=copy.deepcopy(DEPTH_POLICY), label="seed")
    seed.score, seed.diagnostics = _score_candidate(seed, train_seeds, generate, out_root, generate_kwargs)
    seed.replay_score, _ = _score_candidate(seed, replay_seeds, generate, out_root, generate_kwargs)
    beam = [seed]

    for round_idx in range(rounds):
        # Regression: this used to freeze replay_baseline at the ORIGINAL seed candidate's replay
        # score, before round 0, and never recompute it as the beam evolved — so once a mutation
        # outperformed the seed and replaced it in the beam, every future round's regression gate
        # kept comparing against a value that no longer reflected what was actually in the beam.
        # Recomputed each round from the current beam's own (already-cached) replay scores instead.
        replay_baseline = max(c.replay_score for c in beam if c.replay_score is not None)

        children = []
        for parent in beam:
            # Regression: this used to rescore `parent` on train_seeds YET AGAIN here, purely to
            # get fresh diagnostics, then throw the refreshed score away (`_, diagnostics = ...`)
            # — so a candidate's .score (including on the FINAL candidate this function returns)
            # could permanently reflect only its very first-ever evaluation, while the mutation
            # pressure driving the next child came from a separately-run, potentially different
            # score. Now reuses parent.diagnostics (cached when the candidate was scored, whether
            # at seed init or when it was created as a child below) instead of re-running it.
            if not parent.diagnostics:
                continue  # nothing failing under this candidate; no mutation pressure
            child = _propose_mutation(parent, parent.diagnostics, generate, generate_kwargs)
            child.score, child.diagnostics = _score_candidate(child, train_seeds, generate, out_root, generate_kwargs)
            child.replay_score, _ = _score_candidate(child, replay_seeds, generate, out_root, generate_kwargs)
            if child.replay_score < replay_baseline - 0.05:  # regression gate, small tolerance for noise
                continue
            children.append(child)

        beam = sorted(beam + children, key=lambda c: c.score or 0.0, reverse=True)[:beam_size]
        (out_root / f"round_{round_idx}.json").write_text(
            json.dumps(
                [{"label": c.label, "score": c.score, "replay_score": c.replay_score, "policy": c.depth_policy} for c in beam],
                indent=2,
            )
        )

    return beam[0]
