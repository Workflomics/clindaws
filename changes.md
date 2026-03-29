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
