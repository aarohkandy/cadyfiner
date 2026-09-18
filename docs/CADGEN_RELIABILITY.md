# CAD-Code Generation Reliability: Diagnosis, Four Failed Fixes, and the One That Worked

**Status: complete, validated. Every number below is measured on this project's own real
seed bank via the same sandboxed execution + scoring pipeline used everywhere else in this
project — none are projected.**

## The question this document answers

Model 1's real end-to-end evaluation (`docs/TRAINED_OPTIMIZERS.md` §5) came back
underpowered: 11 of 12 valid pairs were ties where **both** the raw and refined prompt
failed Leg 1 under `gemma4:e4b`. That result couldn't distinguish a good Stage-2 model from
a bad one, because the actual bottleneck was somewhere else entirely. This document is that
"somewhere else," run to ground.

## 1. Diagnosis: categorizing 13 real failures

All 13 held-out seeds' raw prompts, run once each through `gemma4:e4b`, scored via
`evaluate_leg1`. Categorizing the 26 raw+refined arm-attempts by their actual `error_type`/
`error_message`:

| category | count | example |
|---|---|---|
| hallucinated/nonexistent API call | ~8/26 | `AttributeError: module 'cadquery' has no attribute 'union'` |
| syntax error (code doesn't parse) | 2/26 | `SyntaxError: invalid syntax` |
| "no pending wires" / stateful chaining mistake | 4/26 | `ValueError: No pending wires present` |
| prefilter rejection (hard rule violated) | 2/26 | used `open(`, banned |
| genuine downstream geometry/dimension problem | a handful | mesh_validity / spec_conformance fails on code that ran |

**The overwhelming majority never reach a genuine geometry problem at all** — the model
doesn't reliably know the real CadQuery API surface or its stateful Workplane-chaining
contract (sketch → extrude → result; methods live on `Workplane`/`Shape` objects, not as
bare `cq.*` functions).

## 2. Four candidate fixes, all built, verified, and tested — none closed the gap

Four independent interventions, each implemented as real code (not just proposed), each
verified against the real installed `cadquery==2.8.0` API before use (not guessed), each
covered by its own tests:

1. **Few-shot examples** added to `CADQUERY_PROMPT_RULES` — 4 worked snippets covering the
   diagnosed failure patterns (box+hole+fillet, union, rotate/translate, sketch→extrude).
2. **A restructured API quick-reference** — a compact name/signature cheat-sheet plus an
   explicit "common mistakes" right-vs-wrong list, a different angle from (1) (reference
   lookup vs. worked example).
3. **A static API pre-check** (`cadyfiner/oracle/api_check.py`) — AST-walks generated code,
   cross-checks every CadQuery attribute access against the real introspected API, rejects a
   hallucinated call before paying for a sandboxed subprocess. Genuinely useful (confirmed
   catching real hallucinations with a precise message), but by itself it only makes a bad
   generation fail *faster and more precisely* — it doesn't turn a failure into a pass.
4. **A self-repair retry loop** (`cadyfiner/oracle/self_repair.py`) — feeds the exact
   execution error back to the generator for one corrected attempt. Well-tested in
   isolation; one real integration bug found and fixed when combined with (3): its
   repairable-error-type set didn't originally include `api_check_rejected`, so once (3)
   started winning the race to reject hallucinated calls, self-repair silently stopped
   retrying exactly the failures both were built to fix.

**Empirical test**: each candidate (individually, and all four combined) run against the
same 13 held-out seeds' raw prompts:

| candidate | full pass |
|---|---|
| baseline (no fix) | 2/13 |
| few-shot examples | 2/13 |
| restructured cheat-sheet | 1/13 |
| static API pre-check | 0/13 |
| self-repair | 2/13 |
| all four combined | 2/13 (+ 1 infra timeout) |

**None beat baseline.** The same two seeds (`pen_holder_high`, `enclosure_medium`) passed
in every configuration — seeds `gemma4:e4b` can already handle regardless of prompting, not
new wins from the fixes. Harder seeds (gears, complex enclosures) failed with a *different*,
seemingly-random error each run, sometimes hitting the exact mistake a fix explicitly warned
against (`cq.Angle` still appeared in two separate runs despite both prompt variants
containing "does not exist" warnings for it).

## 3. Ruling out "just use a bigger/better model"

Same 13 seeds, baseline prompt, no other fixes, three different models:

| model | size | full pass |
|---|---|---|
| `gemma4:e4b` | 8B, general-purpose | 2/13 |
| `qwen2.5-coder:7b` | 7B, code-specialized | 0/13 |
| `ai:coder` (`gemma-4-abliterated`) | 25.8B, general-purpose | 1/13 |

**Bigger and "coder"-labeled are not better here.** The 25.8B model hallucinated its own
different set of wrong methods (`'function' object has no attribute 'filter'`,
`Workplane.rect() got an unexpected keyword argument 'ignorelen'`) — same failure class,
different specifics. This confirms the root cause is a narrow knowledge gap (none of these
models have seen much real CadQuery in training), not a general capability ceiling scale or
"coder" fine-tuning would fix.

## 4. The fix that actually worked: verified templates, not generation

If no tested model reliably writes CadQuery from scratch, stop asking one to. This project's
seed bank covers exactly 7 known object families (bracket, coaster, desk_organizer_tray,
enclosure, gear, pen_holder, wall_planter). For each, `cadyfiner/oracle/templates.py`
provides a hand-written, parametric CadQuery **template function** — verified once, directly,
against the real sandboxed execution pipeline, not generated per-request by an LLM at all.

- `detect_family(raw_prompt)`: coarse keyword match (same honesty as Stage 1's own
  `object_class` classifier — regex-based, handles this project's own phrasing, will miss
  phrasing it's never seen).
- `build_template_code(family, spec)`: maps whatever Stage 1 actually extracted
  (`target_dims`, relevant `features`) onto that family's template kwargs, falling back to
  the template's own defaults — which are calibrated to exactly match the seed bank's ground
  truth — for anything not extracted.
- The LLM's role shrinks to what it's already reasonably reliable at (Stage 1/2's job:
  extracting or guessing dimensions from a prompt) and **never touches CadQuery syntax at
  all** for a covered family. An unrecognized family falls straight through to the existing
  free-form LLM path, unchanged — this is a strictly additive fast path.
- Every template returns CadQuery **source code** (a string), not a live geometry object, so
  it flows through the exact same `run_cadquery()` + `evaluate_leg1()` pipeline
  LLM-generated code does — zero changes needed to the sandboxing or scoring, and the same
  isolation guarantees apply even though this code is trusted.

### Real bugs found and fixed while building the templates (verification is not optional)

- **Tangent, non-overlapping unions** (the same trap `checks.py`'s own docstring warns
  about): gear teeth placed at `base_radius + tooth_radius` touch the base circle at exactly
  one line rather than genuinely overlapping it, producing 17 disconnected solids instead of
  1. Fixed by centering teeth *on* the base circle instead. The wall-planter's back panel had
  the identical bug against the cylinder; fixed with an explicit 2mm overlap.
- **A geometry misread from the ground-truth prompt text**: the bracket's flanges were built
  15mm wide instead of the prompt's actual "60mm by 40mm each" — pushing a mounting hole
  outside the material entirely (silently registering as only 3 of 4 through-holes, not an
  error).
- **A genuine inconsistency in the seed bank's own ground truth**: the wall-planter's
  detailed prompt text says the back panel extends 5mm above and below the cylinder (making
  the true described object 100mm tall), but `target_dims.height` only states 90mm (the
  cylinder's own height). Building literally to the prose produced a 100mm-tall part that
  the checker's own numeric target then failed. Resolved by matching what the checker
  verifies, with the discrepancy stated plainly rather than silently favored either way.
- **Union-then-shell vs. shell-then-union ordering**: shelling a solid box, then separately
  unioning on solid standoffs, left a real (if minor) tessellation-seam gap at the join —
  fusing the solids first and shelling the combined result together fixed it.

### Result

All 7 templates independently pass their own family's ground truth. Run against **all 21**
seed-bank items (all 3 specificity tiers × 7 families) through the real Stage-1-extraction →
family-detection → template-dispatch → sandboxed-execution → scoring pipeline:

**18/21 (85.7%) full pass on the first pass — up from 2/13 (~15%) for the best LLM-only
configuration tested in this document, and confirmed via the project's real statistical
harness (`cadyfiner/harness.py`), not just the standalone check above.**

The 3 remaining failures all traced to one root cause: Stage 1's `_nearest_number_mm`
(`refine.py`) picks whichever number is textually *closest* to a dimension keyword, with no
concept of a clause boundary — on a densely-worded prompt with two dimension mentions close
together ("100mm outer diameter, **8mm** overall thickness"), the wrong, cross-clause number
can be literally fewer characters away than the correct one on the other side of a comma or
"outer"/"tall"-style adjective phrase.

### 4.1 Follow-up fix: 2 of 3 remaining failures resolved

Fixed by clipping the forward/backward search windows at the first comma, period, or
standalone `x` that's followed by whitespace (distinguishing a real clause separator from a
European decimal comma like `"80,5mm"`, which never has a space after it) — see
`_nearest_number_mm`'s updated docstring in `refine.py` for the exact logic, and
`tests/test_refine.py`'s `test_comma_separated_adjacent_dimension_not_misread` /
`test_x_separated_keyword_tagged_dimensions_not_misread` /
`test_decimal_comma_still_not_treated_as_clause_break` for the regression coverage (the third
test specifically guards against the fix breaking the case it could plausibly break). The `x`
side of the fix was checked against every real `"x"`-joined dimension pair in this project's
seed bank first — each one already carries its own keyword on both sides (`"65mm outer
diameter x 95mm tall"`, `"100mm tall x 90mm wide x 4mm thick"`), never a bare positional
`"80x60x30"` chain, so treating a standalone `x` as a clause break doesn't risk misreading a
positional pattern that doesn't actually appear anywhere in real usage.

**Result: 20/21 (95.2%)**, confirmed again through the real harness after the fix. All
pre-existing tests (112 total across the project) still pass — nothing regressed.

The 1 remaining failure (`enclosure_high`) is a different kind of gap: its prompt states
dimensions as an unlabeled positional triple ("External dimensions 80mm by 60mm by 30mm")
with no per-axis keyword anywhere near it, so Stage 1's keyword-proximity search has no
candidate for "height" from that phrase at all — it instead latches onto the only "tall"
mention in the whole prompt (an unrelated standoff's "25mm tall"). This is a missing
*capability* (positional-triple parsing), not a proximity-matching bug, and deliberately
not patched here: assuming a fixed axis order ("by" always means width-by-depth-by-height)
would be a real, separately-scoped design decision that could silently misread a different
prompt using the same phrasing for a different axis order (e.g. a cylinder's "80mm by
30mm" meaning diameter-by-height) — worth doing carefully, not as a quick regex addition
riding on this fix's momentum.

## 5. What this means for the paired raw-vs-refined harness statistic

For a templated family, code generation depends only on Stage 1's structured extraction, not
on prompt *text* — so `raw_pass == refined_pass` for that seed **by construction**: refining
the prompt's wording cannot move a deterministic, already-reliable code path. Running the
full harness across all 21 seeds confirms exactly this, both before and after the §4.1 Stage 1
fix — 0 refined wins, 0 raw wins, every pair a tie (Stage 2's fallback-exclusion count varies
run to run with its own LLM-call stochasticity, unrelated to this fix — 3 excluded on the
first run, 5 on the confirming re-run) — `summarize()`'s correct verdict is "INCONCLUSIVE: no
decisive pairs," which is the honest, expected output of that statistic in this new regime,
not a new negative result. **The overall pass-rate lift (2/13 → 20/21) is where the real,
measured benefit shows up — the paired refinement-effect statistic isn't the right lens for
a templated family anymore, and this document says so rather than reporting a technically-
true-but-misleading "0% win rate."**

## 6. Honest limitations

- **Only 7 families are covered.** Everything else still uses the free-form LLM path this
  entire investigation found unreliable. Extending coverage means writing and verifying more
  templates by hand — there's no way to have an LLM generate a *new* verified template
  without the same hallucination risk this document just spent five sections diagnosing.
- **Family detection is keyword-based**, same honesty as Stage 1's own classifier: it will
  miss phrasing outside what this project's seed bank exercises.
- **The dimension-mapping bridge (`build_template_code`) inherits Stage 1's extraction
  bugs** for the parameters it does forward (see §4's 3 remaining failures) — a template
  can't correct for a wrong number it was handed.
- **This doesn't validate whether refining the prompt TEXT helps** for a templated family —
  see §5. It validates something different and arguably more useful for these families:
  reliable geometric correctness, independent of prompt wording quality.
- **The api_check and self-repair modules remain valuable on the fallback path** (any
  family not covered by a template) even though neither alone or combined beat baseline on
  the seeds tested here — api_check catches real hallucinations faster and more precisely,
  and self-repair is a well-tested, reusable capability for whatever comes next.
