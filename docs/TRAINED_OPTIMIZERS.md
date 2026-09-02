# Two Small Trained Models for CAD Prompt Refinement

**Status: data generation and training in progress as of this writing. Sections marked
`[PENDING]` will be filled in with real numbers once both runs complete — this document is
being written alongside the work, not after it, so you can see the plan before the result.**

## Abstract

`cadyfiner`'s refiner has two places where a general-purpose LLM currently makes a judgment
call: (1) *Stage 2*, which reads a raw CAD prompt and decides what missing detail to add, and
(2) the *optimizer*, which reads a batch of generation failures and decides how to adjust the
refiner's policy. Both currently call out to whatever large model is configured (`gemma4:e4b`
locally, or Claude). This document describes replacing each with a small, specialized model —
fine-tuned for exactly its one job — while keeping the general-purpose path fully intact as an
alternative. We explain why two *different-sized* models are the right call rather than one,
how their training data was built (and why the two datasets are built by genuinely different
methods), what "training" means here precisely (LoRA fine-tuning, not training from scratch),
and how we know whether either model actually helps before trusting it.

## 1. Why two separate models, not one

The user's original framing was "the optimizer" as a single thing to train. It isn't — it's
two components with different jobs, different call frequencies, and different-sized problems:

| | Stage 2 (gap-filler) | Optimizer (mutation-proposer) |
|---|---|---|
| Runs | every single `refine` call | only during an `optimize.py` tuning run |
| Input | a raw prompt + what Stage 1 extracted | a policy state + recent failure diagnostics |
| Output space | open-ended (any dimension value, any feature list, a full paragraph) | ~14 possible actions (add or remove 1 of 7 tags) + a "no edit" option |
| Why it matters | determines every user's actual output quality | determines how fast/well the policy self-improves during tuning |

Because the output spaces are so different in size, we use two differently-sized base models
rather than forcing both onto the same one:

- **Model 1 (Stage-2 gap-filler): Qwen2.5-1.5B-Instruct.** Needs real language competence —
  producing a coherent refined paragraph and correctly-typed JSON from an open-ended input.
- **Model 2 (mutation-proposer): Qwen2.5-0.5B-Instruct.** The task is closer to classification
  over a small fixed vocabulary than open-ended generation; a much smaller model is plausibly
  enough capacity, and it's cheaper to run inside the optimizer's inner loop where it may be
  called many times per tuning run.

Both are [Apache 2.0](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct), fine-tunable on CPU,
and continue this project's existing precedent of using the Qwen2.5 family (`cadybara-kitchen`'s
own experiment configs already used `qwen2.5`/`qwen2.5-coder` variants for the same reason: it's
a well-supported, small-model-friendly, instruction-tuned family).

**Both existing mechanisms are kept, unconditionally.** `refine_stage2.fill_gaps` and
`optimize.py`'s `_propose_mutation` still work exactly as before, calling whatever generic model
is configured. The trained models are a new, alternative `generate`-compatible backend
(`cadyfiner/generators/local_trained.py`) a caller can swap in — nothing about the existing,
already-tested code paths changes.

## 2. What "training" actually means here

Neither model is trained from scratch — that would need far more data and compute than a CPU
box can reasonably provide. Both are **LoRA fine-tuned**: the base model's ~0.5-1.5 billion
parameters stay frozen, and a small set of low-rank adapter matrices (rank 16, applied to the
attention projection layers) gets trained on top — about **0.28-0.44% of the total parameter
count** (verified directly: 4.36M trainable out of 1.55B for the 1.5B model, 2.16M out of 496M
for the 0.5B model). This is standard practice for adapting a small pretrained model to a
narrow task without the cost of full fine-tuning, and it's specifically what makes CPU-only
training on homebase (8 vCPU, no GPU — see the project's own `CLAUDE.md`) tractable: full
fine-tuning of even the 0.5B model's complete parameter set would be far slower.

We measured this directly before committing to a real run (see `training/train_lora.py`):
on homebase, two LoRA optimizer steps (batch size 1, gradient accumulation 8, so 16
micro-batches) took **~6s for the 0.5B model and ~11s for the 1.5B model** on short synthetic
examples. Extrapolated to a realistic dataset size (~100-300 examples, 3 epochs), full training
for either model is expected to take **single-digit minutes, not hours** — the bottleneck in
this whole project is data *generation* (each example needs 1-3 real LLM calls plus, for Model
1, real CAD generation and execution), not training compute.

## 3. Training data: two different methods, and why

Neither model has a large pre-existing labeled dataset to learn from — this project's hand-
authored seed bank has 21 examples, nowhere near enough to fine-tune anything without severe
overfitting. Both datasets are therefore **synthesized**, but by different methods matched to
what each task actually needs.

### 3.1 Model 1: oracle-filtered rejection sampling

Model 1's job (suggest what to add to an underspecified prompt) has real geometric consequences
that can be objectively checked — this project already built exactly the tool for that (Leg 1,
`cadyfiner/oracle/checks.py`). So the data isn't just "whatever a bigger model said," it's
filtered by whether the bigger model's suggestion actually produced better geometry:

1. **176 synthetic raw prompts** were generated by 15 parallel agents, each assigned a distinct
   object category (holders, planters, hooks, brackets, gears, enclosures, clips, fittings,
   etc.), explicitly instructed to write REALISTIC, underspecified prompts the way an actual
   user types them — not padded with exact dimensions, since a fully-specified prompt gives the
   gap-filler nothing to do. Deduplicated to 176 distinct prompts (105 mechanical / 71
   decorative — see `workspace/synthetic_prompts.json`).
2. For each prompt, the **current teacher pipeline** runs: Stage 1 extraction, then Stage 2's
   existing LLM call (`gemma4:e4b`) produces a candidate refined prompt + spec.
3. **Both** the raw prompt and the refined prompt are sent through CAD generation and execution
   (`cadyfiner.oracle.execute`), and **both** are scored by `evaluate_leg1` against the *same*
   target — the refined spec's own stated dimensions/features. Holding the target constant
   isolates one question: does refining the prompt TEXT help the generator hit that target,
   independent of what the target is?
4. **Rejection sampling**: the (input, output) pair is kept as a training example only if the
   refined arm reaches at least as far through the Leg-1 cascade (execute → mesh validity →
   spec conformance → manufacturability → full pass) as the raw arm did. A case where refining
   made the result *worse* is discarded, not learned from.

**Honest limitation**, stated plainly: the "target" here is teacher-synthesized, not
independently authored the way the 21-example seed bank's ground truth is. This is a
self-consistency filter (does refining help hit the refiner's own stated target), not proof
that the target itself is correct. It's an appropriate tradeoff at training-data scale — hand-
authoring verified ground truth for 100+ examples the way the seed bank's 21 were built was not
feasible in this timeframe — but it's a real, different (weaker) form of evidence than the
seed-bank harness's results, and should be read that way.

Script: `scripts/build_distillation_data.py`. Run: `python scripts/build_distillation_data.py
gemma4:e4b workspace/synthetic_prompts.json workspace/distillation_data.jsonl 90` — scoped to
90 of the 176 prompts given the wall-clock cost (3 LLM/CAD-generation calls per prompt at
45-150s each on CPU-only local inference).

**[PENDING]** Final accepted/rejected/error counts, once the run completes.

### 3.2 Model 2: expert-labeled scenarios, not exhaustive rejection sampling

Model 2's task is much narrower — pick one of ~14 possible edits, or none, given a diagnostic.
Generating its training data via the same rejection-sampling method as Model 1 would mean
running `optimize.py`'s full beam search many times over, which itself runs multiple seeds
through multiple full CAD-generation-and-execution cycles per round — expensive and, for a task
this constrained, disproportionate. Instead:

1. **Real diagnostic text from this project's own actual runs** (the mechanical pilot, and both
   harness runs) was collected verbatim — genuine `TypeError`s, hallucinated CadQuery
   attributes, prefilter rejections, and dimension mismatches this project actually produced,
   not hypothetical ones.
2. A panel of parallel agents labeled each scenario: which object class it applies to, and
   critically, **whether it's policy-fixable at all**.
3. Three more agent batches generated additional *synthetic* scenarios in the same style,
   covering patterns underrepresented in the (small) real sample.
4. Every label was then adversarially re-checked by a separate agent for internal consistency
   (does the proposed tag actually address the stated failure, is the policy-fixable/not call
   correct) and corrected where wrong.

**The central finding driving this dataset's design**: most of this project's own real failures
are **not** something a depth-policy edit can fix at all. Looking at the actual diagnostic text
collected — `AttributeError: module 'cadquery' has no attribute 'Angle'`, `TypeError:
Workplane.add() takes 2 positional arguments but 17 were given`, `SyntaxError: invalid syntax`,
a prefilter rejection for calling a forbidden method — these are the downstream CAD-code-writing
model producing bad code, which no amount of added prompt detail changes. Only failures that are
actually about *missing information* (a dimension came out wrong because nothing disambiguated
it, a feature was never mentioned) are within the optimizer's reach. Training Model 2 to
recognize this distinction and output "no edit" rather than confidently guessing an irrelevant
one is as important as training it to propose good edits when one would actually help.

**[PENDING]** Final scenario count, policy-fixable/not-fixable split, and correction rate from
verification, once the workflow completes.

## 4. Training configuration

Both models: LoRA rank 16, alpha 32, applied to `q_proj`/`k_proj`/`v_proj`/`o_proj`; 3 epochs;
batch size 1 with gradient accumulation 8 (effective batch size 8); learning rate 2e-4; CPU-only
(`use_cpu=True`, `torch==2.14.0+cpu`); trained via TRL's `SFTTrainer` on chat-formatted
`{role, content}` message pairs, so the input format matches each call site's actual prompt
text exactly — Model 1 trains on `refine_stage2._build_prompt()`'s literal output, Model 2 on
`optimize.py`'s `_propose_mutation()` prompt format, so either can be swapped in as a drop-in
`generate()` replacement with zero change to the calling code. Full script: `training/train_lora.py`.

Run on homebase (`~/work/cadyfiner_train/`), resource-capped via
`systemd-run --user --scope -p MemoryMax=<20-24G>` — homebase is a shared box running the
user's other active services (13 Docker containers, a long-running tmux session observed at
setup time), not a dedicated training machine, so training never claims unbounded RAM/CPU.

**[PENDING]** Final training loss curves, wall-clock time, and adapter sizes.

## 5. Evaluation: how we'll know if either model actually helps

Per this project's own established standard (see the main README's evaluation section and
`cadyfiner/harness.py`), a trained model is not trusted until it beats the alternative on the
same paired, statistically-tested methodology already built:

- **Model 1**: re-run `scripts/run_harness.py` with `local_trained` as the Stage-2 backend
  instead of `gemma4:e4b`, on the same held-out seed bank, and compare win rates.
- **Model 2**: re-run `optimize.py` with `local_trained` as the mutation-proposer, and compare
  how many rounds it takes to reach a given train-set pass rate, and whether the final policy
  it converges to differs from what the general-purpose model converges to.

**[PENDING]** Results once both models are trained and both re-evaluations complete.

## 6. Known limitations (stated up front, not discovered later)

- Both training sets are **synthetic, teacher-generated, and modest in size** (dozens to low
  hundreds of examples) — this is fine-tuning a narrow skill onto an already-competent small
  instruction-tuned model, not teaching CAD knowledge from nothing, but it means neither model
  should be expected to generalize as broadly as the general-purpose backend it's meant to
  specialize past.
- Model 1's training target (Section 3.1) is self-consistency against a teacher-synthesized
  spec, not independently authored ground truth.
- Model 2's "no edit" scenarios are a genuinely new idea for this project (the original
  `optimize.py` mutation-proposer always proposes *something*) — if evaluation shows the trained
  model outputs "no edit" too often or too rarely relative to what's actually useful, that's a
  labeling-calibration problem to revisit, not a training-mechanics one.
- Neither model has been evaluated end-to-end as of this writing (Section 5 is pending) — until
  it is, treat both as "built and plumbed in," not "proven better."
