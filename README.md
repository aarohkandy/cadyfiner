# cadyfiner

A CAD prompt refiner: rewrites underspecified text-to-CAD prompts into
better-specified ones, with an automated quality oracle and a statistical
exit-criteria harness to prove (or disprove) that the refinement actually
helps — on prompts and object types it has never seen.

Built in one session on top of evidence gathered from three existing repos
(`cadybara-kitchen`, `ai-cad`, `cad_grade`) plus a real preprint reanalyzing
human preference votes over 94 generated CAD models. See
[`research/`](research/) for the raw analysis. Every non-obvious design
decision below is tied to something measured, not assumed — and every
foundational module went through an independent adversarial code review
(31 confirmed bugs found and fixed across two passes; see git history).

## What it actually does

```
raw prompt ──▶ Stage 1: regex extraction ──▶ Stage 2: LLM gap-fill ──▶ refined prompt + spec
                (units, dims, features,          (fills ONLY the info
                 decorative/mechanical             categories the object
                 classification)                   class's policy allows)
```

The refined prompt goes to a CAD-code-writing LLM (local Ollama or a
frontier model); the resulting CadQuery source runs in a sandboxed
subprocess; the resulting STL is scored by **Leg 1** (execute → mesh
validity → spec conformance → manufacturability, all objective, all fast)
and optionally **Leg 2** (a VLM judge on rendered views, reporting-only,
never gating — see "Why two legs" below).

## Quick start

```bash
pip install -e .
ollama pull gemma4:e4b   # or any vision-capable-if-you-want-Leg-2 local model

cadyfiner refine "make me a bracket with mounting holes"
cadyfiner generate "make me a wall planter" --refine --out planter.stl
cadyfiner check planter.stl --spec my_ground_truth.json
```

`--backend frontier` uses Claude instead of local Ollama; needs
`ANTHROPIC_API_KEY` (not exercised end-to-end in this environment — no key
was available while building this, so treat that path as implemented but
unverified until run once for real).

## The evidence this is built on

1. **Geometric analysis of 94 real generated STLs** (`cad_grade`'s public
   dataset): for the one family with an audited prompt ladder
   (`wall_planter`), specificity level 7 hit 100% watertight geometry *and*
   the best dimensional fidelity (~2% error), beating both less-detailed
   levels (up to 55% dimensional error) and level 10 (66% reliability
   collapse — adding a keyhole slot + drainage holes + fillets
   simultaneously overloads the generator).
2. **A real preprint** ("7/10 Stars Please," in this project's `research/`
   directory) independently reanalyzed the same arena's human pairwise
   votes with Bradley–Terry schedule-adjustment and found the same level-7
   lead — but with real caveats: it loses its direct matchup to level 3,
   the three object families disagree on the best level, and it's one
   rater's session, not population evidence.
3. **A join of that vote data to the STL geometry** (done this session):
   near-zero correlation (r=-0.004 to -0.095) between mesh validity and
   human preference. **Raters judge rendered appearance, not topology** —
   which is why this project has two separate, never-conflated quality
   legs instead of one.
4. **A live pilot on 3 mechanical families** (bracket/gear/enclosure) using
   a weak local model (`dolphincoder:7b`) found the model too unreliable
   at basic instruction-following to isolate a specificity effect — most
   failures were wrong export calls and syntax errors, not
   specificity-driven geometry problems. This is itself the finding: **the
   depth policy for mechanical/functional parts is a reasoned default
   (full depth, including exact placement — because that IS the functional
   requirement for a bracket or gear), not an empirically validated one**
   the way the decorative policy is. Say so if you use this on
   mechanical parts and it doesn't work well; that's expected until a
   better local model re-runs the pilot cleanly (see
   `scripts/run_pilot.py`).

## Architecture, and why each piece looks the way it does

- **`cadyfiner/spec.py`** — the `DesignBrief` spec object (dimensions,
  features, process notes, `assumptions_made`). Shaped after `ai-cad`'s
  `DesignBrief`/`TargetDimensions` schema, which is genuinely reusable;
  `ai-cad`'s `DesignValidator` heuristics are not (verified: they're
  mug-specific keyword rules), so they weren't reused despite the
  superficial similarity.
- **`cadyfiner/refine.py`** (Stage 1) + **`refine_stage2.py`** (Stage 2) —
  deterministic extraction, then one constrained LLM call. *How much*
  Stage 2 is allowed to add is `DEPTH_POLICY`, keyed by object class, not
  a universal constant — this is the direct implementation of finding #1.
- **`cadyfiner/oracle/execute.py`** + **`_subprocess_entry.py`** — sandboxed
  CadQuery execution: subprocess isolation, wall-clock timeout,
  RLIMIT_CPU/AS/FSIZE/NPROC, a substring/regex prefilter as defense in
  depth. Captures exact BRep-level facts (`isValid()`, `Solids()`,
  `Volume()`, `Area()`) from the CAD kernel directly, not from tessellated
  mesh approximations — found live that OCC and trimesh can disagree (a
  tangent, non-overlapping union reads as 1 body via mesh connectivity but
  is genuinely 2 separate `Solid`s per OCC).
- **`cadyfiner/oracle/checks.py`** — `evaluate_leg1`, the single scoring
  function used identically by the optimizer and the exit-criteria
  harness. Short-circuits execute → mesh_validity → spec_conformance →
  manufacturability, cheapest-first (mirrors MUSE's reported failure
  cascade). Every "can't verify this" case is reported, never silently
  passed.
- **`cadyfiner/oracle/judge.py`** — Leg 2, the VLM judge. Matplotlib-based
  rendering (trimesh's pyglet-backed offscreen renderer was found broken
  for headless use on macOS while building this — no windowing system
  needed instead). Scores against the *original* prompt, never the
  refined one, so a refiner can't inflate its own score by rewriting the
  prompt to match whatever it built.
- **`cadyfiner/optimize.py`** — a native (~250-line) reflective beam
  search over `DEPTH_POLICY` variants, not a GEPA/DSPy dependency. An
  adversarial Plan-agent review did the throughput math: GEPA's headline
  efficiency numbers are measured against GPU-cluster multi-module DSPy
  programs, a different regime than tuning one prompt/rule-set over tens
  of evaluations on a CPU-only box. What's actually valuable from that
  lineage — feed rich failure diagnostics to an LLM proposing a mutation,
  instead of a scalar reward — is what's implemented here.
- **`cadyfiner/harness.py`** — paired (same-seed, same-generator)
  raw-vs-refined statistical evaluation. Exact sign-test p-value is the
  actual PASS/FAIL criterion (not a normal-approximation CI bound — these
  disagree at small n, confirmed live). Reps are clustered by seed before
  computing significance, so replaying a small seed pool can't manufacture
  apparent significance.
- **`cadyfiner/generators/local_trained.py`** — an alternative, optional
  backend: two small LoRA-fine-tuned models (Qwen2.5-1.5B for Stage 2's
  gap-filling, Qwen2.5-0.5B for the optimizer's mutation proposals),
  specialized for exactly one job each, kept alongside — not instead of —
  the general-purpose Ollama/frontier path. Full methodology, data
  provenance, and honest results: [`docs/TRAINED_OPTIMIZERS.md`](docs/TRAINED_OPTIMIZERS.md).

## Why two legs, never combined into one score

The evidence above (finding #3) makes this the load-bearing design
decision. Leg 1 (execute/geometry/spec-conformance/manufacturability) and
Leg 2 (VLM appearance judge) measure genuinely different things. Combining
them into one scalar would hide which axis actually moved. Leg 1 gates;
Leg 2 only reports, and only at final evaluation time — never in the
optimizer's hot loop, so a renderer never needs to be fast.

## The seed bank

`prompts/seed_bank/families/*.json` — 7 families, 21 prompts (3
specificity tiers each), split into train / heldout-same-family /
heldout-family per `prompts/seed_bank/manifest.json`. Scoped down from the
original plan's aspirational "≥40 held-out seeds" target given real
session time constraints — documented honestly rather than padded; expect
wide confidence intervals at this n, and treat `enclosure` (the one family
never touched during any tuning) as the primary generalization check.

Run it:

```bash
python scripts/run_pilot.py <model>              # raw prompts only, no refiner — the checkpoint step
python scripts/run_harness.py <model> heldout_same_family,heldout_family
```

Results land in `workspace/harness/<model>/report.json` with a paired win
rate, 95% CI, exact sign-test p, and per-family/per-role breakdowns.

## Known limitations (say these out loud, don't bury them)

- **Stage 1 extraction is regex-based, not real NLP.** It handles the
  phrasing patterns in this project's own seed bank well (tested); it will
  get confused by phrasing it's never seen. Stage 2's LLM call is the
  intended backstop, not Stage 1 being perfect.
- **The mechanical-parts depth policy is unvalidated** (see evidence #4
  above) — a reasoned default, not a proven-optimal one.
- **The frontier (`ANTHROPIC_API_KEY`) backend has never been run
  end-to-end** in this environment. The code path exists and mirrors the
  validated local path; verify it once before trusting it in production.
- **The wall-thickness proxy (`2×Volume/Area`) is a whole-body average.**
  It can miss a genuinely thin local feature next to unrelated thick
  geometry, and can false-fail a correct part whose average is pulled up
  by an unrelated thick section. Checked at a wide (50%) tolerance and
  explicitly labeled low-confidence for exactly this reason — a real fix
  would need a localized thickness map (e.g. `trimesh.proximity`
  ray-casting), not implemented here.
- **Leg 2's VLM judge is one uncalibrated model**, not validated against a
  broad human panel (this project's own vote-archive evidence shows why
  that would be hard to fully trust anyway — see finding #3).
- **The exit-criteria seed bank is small.** Treat any single harness run's
  verdict as a first read, not a final one — rerun with more reps/seeds
  before making a real decision based on it.

## Testing

```bash
pytest tests/ -v
```

~2500 lines of implementation, matched by a comparably-sized adversarial
review + regression-test pass — every fixed bug above has a permanent
test named after the failure it prevents, not just what it checks.
