# Biotools SAT Parity Notes

## Scope

These notes summarize the current Biotools horizon-2 parity gap between:

- APE SAT
- snakeAPE `multi-shot-lazy`

This document is based on the current validated baseline as of 2026-03-31 and incorporates the earlier investigation captured in `/Volumes/ZGMF-X20A/GARYU/findings.md`.

## Current Validated Baseline

Exact length 2 on Biotools:

- SAT total: `763`
- snakeAPE total: `695` raw, `693` unique
- Exact normalized workflow matches: `693`
- SAT-only tool-sequence mismatches: `70`
- snakeAPE-only tool-sequence mismatches: `0`
- Remaining strict-signature mismatch: `2` snakeAPE-only variants on already-matched tool sequences

This is the current stable baseline. It is materially better than the older `901 / 897` state because the large snakeAPE-only overgeneration has already been removed.

## What Is Wrong

### 1. Exact-horizon early-goal blocking is real

The lazy encoding currently contains the generic rule:

```asp
:- occurs(t, run(_)), holds(t-1, goal).
```

This is correct for "stop once goal is already reached" behavior, but it is too strong for exact-length synthesis.

For Biotools length 2, some first-step tools can already produce an artifact compatible with the final goal. When that happens, `holds(1, goal)` becomes true and step 2 is forbidden entirely, even though APE SAT still allows a length-2 workflow.

Confirmed first-step tools with at least one goal-compatible output under the taxonomy:

- `IDBAc`
- `clc_find_variations`
- `clc_unmapped_reads`
- `clc_mapping_info`
- `clc_extract_consensus`

This explains a real subset of the missing SAT-only length-2 workflows.

### 2. Early-goal blocking does not explain all missing cases

The remaining 70 SAT-only tool sequences are not all from the early-goal case.

Notably, several missing families such as:

- `cytofkit -> ...`
- `msmsTests -> filtercontrol`

were not explained by the quick goal-compatibility inspection alone. That means there is at least one additional issue beyond the `holds(t-1, goal)` blocker.

The most likely remaining area is exact-horizon final-step output selection and query-time output filtering, not generic lazy reachability.

### 3. There was an older translation-side output compression bug

The earlier investigation in `findings.md` identified an older translation artifact where `clc_extract_consensus` port 1 lost broad format descendants during output compression. That caused workflows such as:

- `clc_extract_consensus -> gmap`

to disappear because the consumer-side format checks could never pass.

Important: this appears to be an older code/result mismatch, not the current primary blocker.

Fresh translation from the current code showed many format values for that port, which suggests that specific translation bug has already been fixed in the translator. The current parity gap is therefore no longer dominated by that issue.

## What Already Worked

### 1. Query-horizon goal-output canonicalization was a good change

The `chosen_goal_output` work in the lazy encoding was the main successful correction.

It moved the Biotools horizon-2 baseline from roughly:

- `901` raw / `897` unique

to:

- `695` raw / `693` unique

and removed the large `snakeAPE-only` tool-sequence overgeneration. The current comparison result of:

- `70` SAT-only
- `0` snakeAPE-only

is much cleaner and gives a usable target for parity work.

### 2. Multi-shot min/max handling was fixed

The multi-shot harness now always grounds from horizon 1 upward and only uses `min-length` to control when solving/output starts.

That matters for exact-length parity work because `--max-length 2` should still build the same incremental state as a normal multi-shot run, only truncated at horizon 2.

This was a correctness fix for the solver harness, not the cause of the remaining SAT gap.

## What Was Tried And Rejected

### 1. Reverse query-time `compatible/2` direction

Change tried:

- adjust the query-horizon producer admissibility check in `step.lp`

Observed result:

- no improvement in solution counts
- horizon-2 grounding became about 50% slower
- horizon-2 solving became about 30% slower

Conclusion:

- not the root cause
- reverted

### 2. Broad exact-horizon relaxation

Change tried:

- allow continuing after an earlier goal in exact-horizon mode

Observed result:

- search reopened too broadly
- horizon-2 counts jumped back toward the old overgeneration regime
- one run hit the global solution cap (`1003` seen, `999` stored at horizon 2)

Conclusion:

- too permissive
- reverted

### 3. Narrow "continue after goal only if reused" rule

Change tried:

- allow continuation only when the later step consumed an earlier goal-satisfying artifact

Observed result:

- still caused large solving blow-ups
- did not converge to a safe, count-stable fix

Conclusion:

- not shippable in its current form
- reverted

### 4. Chosen-witness-only output narrowing / widened query-step output domains

Change tried:

- keep only the witness output canonicalized
- widen other final-step output choice domains

Observed result:

- no count improvement
- horizon-2 grounding regressed to roughly `75s`
- solving also regressed

Conclusion:

- did not improve parity
- reverted

### 5. Full witness-chain rewrite

Change tried:

- replace global propagated goal-hold semantics with explicit goal-witness dependency semantics

Observed result:

- horizon-2 grounding regressed heavily
- horizon-2 solving did not finish in a reasonable time

Conclusion:

- too expensive to land directly
- reverted

### 6. Prototype hybrid fallback inside `multi-shot-lazy`

Change tried:

- keep the baseline `multi-shot-lazy` run
- add a second constrained `multi-shot-lazy` recovery pass for the goal-reuse corner
- merge deduplicated results

Observed result:

- implementation was started but not validated
- not kept in the tree

Conclusion:

- still a possible direction
- needs a tighter design before another implementation attempt

## Current Conclusion

The remaining Biotools horizon-2 parity gap is now much narrower and much better isolated than before:

- the large snakeAPE-only overgeneration problem is fixed
- a real subset of missing SAT workflows is caused by exact-horizon early-goal blocking
- another subset remains unexplained by that mechanism alone

The hard part is that every direct encoding relaxation tried so far either:

- brought back major overgeneration, or
- made grounding/solving much slower without improving parity

## Recommended Next Step

Stay within `multi-shot-lazy`, but avoid another global encoding rewrite.

The most practical next step is one of:

1. A constrained hybrid recovery pass inside `multi-shot-lazy` for the early-goal reuse corner only.
2. Stronger diagnostics for the remaining 70 SAT-only sequences so the next fix targets one missing family at a time.

What should not be repeated without a narrower hypothesis:

- broad exact-horizon relaxation
- generic producer admissibility changes
- full witness-chain rewrites

