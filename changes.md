# snakeAPE Change Notes

This file records the recent solver/debugging changes made during the SAT-parity investigation, why each change was needed, and what issue it addressed.

## 1. `multi-shot-lazy` regression that produced zero answer sets

### Issue

`multi-shot-lazy` stopped producing any solutions, even on `defect_concentration`, where workflows were known to exist. Every horizon grounded and solved immediately as unsatisfiable.

### Root Cause

The lazy step encoding had lost the rule that chooses `use_lazy_candidate(Tool, Candidate)`. Without that choice, later steps could not bind inputs or emit outputs, so the solver could never construct workflows.

### Change

The missing candidate-choice rule was restored in the lazy step encoding.

### Why

This was a hard solver regression, not a comparison-only issue. It made all later analysis meaningless because the solver was structurally unable to build workflows.

## 2. `python snakeAPE ...` entrypoint fix

### Issue

Running `python snakeAPE ...` from the `snakeAPE` root failed with a relative-import error in `snakeAPE/__main__.py`.

### Root Cause

The package entrypoint assumed module-style invocation only.

### Change

`snakeAPE/__main__.py` was adjusted so both invocation styles work:

- `python -m snakeAPE ...`
- `python snakeAPE ...`

### Why

This restored the historical local workflow and removed a distracting CLI failure while debugging solver behavior.

## 3. SAT-vs-snakeAPE comparison tooling

### Issue

We needed a precise way to compare APE SAT `solutions.txt` against snakeAPE `answer_sets.txt` instead of relying on aggregate counts alone.

### Change

A normalization/comparison utility was built to:

- parse raw SAT complete solutions
- parse snakeAPE answer sets
- normalize tool sequences, bindings, and produced outputs
- compare both at tool-sequence and strict workflow-signature levels

### Why

This made the mismatch concrete. It showed that some discrepancies were true solver-semantic differences, while others were only provenance or binding-order differences.

## 4. Shared-solver overgeneration fix

### Issue

At `defect_concentration` length 8, snakeAPE produced too many workflows compared with SAT. The main confirmed class was overgeneration from alternative binding/provenance choices on shared tool sequences.

### Root Cause

The shared multi-shot encodings allowed extra symmetric provenance combinations that the SAT reference did not use.

### Change

The shared `multi-shot` and `multi-shot-lazy` encodings were tightened with same-signature/provenance restrictions so the confirmed extra binding variants were removed without collapsing valid workflows.

### Why

This brought the solver down from the inflated count to the intended cumulative count behavior through length 8.

## 5. Missing tool-sequence parity fix

### Issue

After overgeneration was reduced, SAT still had `252` tool sequences that snakeAPE did not produce.

### Root Cause

The ASP encodings still contained clingo-side restrictions that SAT mode does not enforce, especially around repeated tools and same-tool reuse patterns.

### Change

The repeated-tool and same-tool self-loop bans were removed from the shared multi-shot encodings/checks.

### Why

This restored tool-sequence parity with SAT. After this change, the cumulative count through length 8 matched SAT at `1332`.

## 6. Output-side canonicalization of shown bindings

### Issue

Even after count parity and tool-sequence parity were fixed, strict SAT equality still did not hold. snakeAPE and SAT often agreed on the same workflow skeleton but serialized different `ape_bind` provenance choices.

### Root Cause

Some differences were below the solver-count level and lived in how equivalent binding structures were emitted and reconstructed.

### Change

Python-side canonicalization was added before workflow reconstruction so shown symbols are normalized in a SAT-style order where possible.

### Why

This reduced the remaining strict-signature mismatch class significantly without disturbing the now-correct solver counts.

## 7. Current accepted state

For `defect_concentration`, the current accepted validation point is:

- length 7: `72`
- cumulative through length 8: `1332`
- tool-sequence parity with SAT: fixed

There is still a remaining strict-signature mismatch class where SAT and snakeAPE can disagree on exact provenance/binding order while still agreeing on workflow coverage and counts. That class is currently treated as non-blocking because it does not change the workflow skeletons being enumerated.

## 8. Comparator move into `snakeAPE` root

### Issue

The comparison utility lived under `ironAPE/APE_Example`, which made it harder to treat it as part of the snakeAPE debugging toolchain.

### Change

The comparator was moved into the `snakeAPE` root as a standalone script.

### Why

This puts the debugging utility next to the solver it is used to validate and makes the local workflow clearer.

## 9. Comparator output improvements

### Issue

The original comparison output focused on raw counts and mismatch samples, but it did not explain the practical meaning of each mismatch class.

### Change

The comparator output was expanded to include:

- comparison scope
- interpretation of exact matches vs strict-signature mismatches vs tool-sequence mismatches
- explanations for representative mismatch samples
- explicit indication when tool-sequence coverage already matches

### Why

This makes the tool usable as an engineering diagnosis aid rather than only a raw diff utility.

## 10. Biotools length-1 SAT parity fix in `multi-shot-lazy`

### Issue

For the Biotools `config.json` problem, APE SAT produced exactly one length-1 workflow (`IDBac`), while `snakeAPE` `multi-shot-lazy` produced `5000` length-1 answer sets. The mismatch was dominated by thousands of spurious `clc_mapping_info` models plus seven one-step `IDBAc` variants.

### Root Cause

There were three separate horizon-1 lazy-path problems:

- the exact-horizon step-1 goal filter was disabled whenever the original workflow input broadly matched the goal taxonomy, even though SAT does not allow workflow inputs to count as final outputs
- the intended `use_all_generated_data=ALL` output-count guard was not reliably excluding multi-output candidates in the lazy horizon-1 specialization
- for one-step exact-goal solving, lazy output-choice facts still allowed every compatible terminal descendant of the goal class, which created multiple `IDBAc` variants

### Change

The lazy horizon-1 path was tightened to match SAT/APE semantics:

- `initial_goal_filter_active(1)` now stays active in exact-horizon mode instead of turning off when `initial_goal_holds`
- `multi-shot-lazy/check.lp` now forbids reusing workflow inputs as final goal outputs, mirroring APE's `inputsAreNotOutputs` rule
- `multi-shot-lazy/step_initial.lp` now applies a direct hard constraint that rejects candidates whose output-port count does not match the goal count under `use_all_generated_data=ALL`
- `multi-shot-lazy/base.lp` now collapses goal-matching output-choice values in the horizon-1 goal filter to one deterministic representative per category, removing the one-step terminal-descendant explosion

### Why

This fixes a true solver-semantics mismatch, not just a formatting or extraction issue. It restores exact Biotools length-1 parity with SAT without changing the later-horizon `defect_concentration` reference counts.

### Validation

- Biotools exact length 1:
  - SAT: `1`
  - `snakeAPE multi-shot-lazy`: `1`
  - normalized workflow comparison: exact parity, no tool-sequence mismatch, no strict-signature mismatch
- `defect_concentration` regression:
  - length 7: `72`
  - cumulative through length 8: `1332`

## 11. Later-horizon `multi-shot-lazy` grounding reduction for Biotools

### Issue

After the horizon-1 SAT mismatch was fixed, Biotools could still stall at later lazy horizons, especially horizon 2. The main known causes were the generic later-horizon `eligible/4` reconstruction and the repeated carry-forward of all artifact dimension facts.

### Change

The lazy later-horizon path was restructured in three exact ways:

- artifact dimensions are now time-independent via `artifact_dim(...)` instead of being copied forward as `holds(t, dim(...))` at every horizon
- later-step bindings no longer rebuild eligibility from the full artifact-state join; they use direct workflow-input bindability plus previous-output port compatibility
- the lazy translator now emits `lazy_candidate_port_bindable(...)` facts and conservatively prunes lazy candidates that cannot reach the workflow goal through the precomputed compatibility graph

### Why

This moves repeated later-horizon work into one-time translation/indexing and shrinks the lazy grounding domain without changing accepted workflows.

### Validation

- `defect_concentration` still matches the accepted lazy reference:
  - length 7: `72`
  - cumulative through length 8: `1332`
- Biotools horizon-2 grounding is now isolated to the remaining slow stage instead of the earlier raw eligibility/dimension-carry-forward shape, but it is not fully eliminated yet in this session.
