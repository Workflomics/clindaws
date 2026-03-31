# Translation And Stepwise Grounding Improvement Plan

## Goal

Reduce the end-to-end cost of `multi-shot-lazy` in two places:

1. Translation/build time in the Python pipeline.
2. Per-horizon grounding time in the clingo multi-shot solver.

The current pipeline already grounds `base` only once per multi-shot run, but it still pays a large translation cost up front and still pays a substantial horizon-2/horizon-3 grounding cost in the lazy encoding.

## Current Pipeline

### Translation

The runner builds a `FactBundle` in Python through:

- `build_fact_bundle_grounding_opt()` in [`snakeAPE/translator.py`](/Volumes/ZGMF-X20A/GARYU/snakeAPE/snakeAPE/translator.py)
- `build_fact_bundle_grounding_opt_lazy()` in [`snakeAPE/translator.py`](/Volumes/ZGMF-X20A/GARYU/snakeAPE/snakeAPE/translator.py)

`FactBundle` currently stores the translated facts as one large LP string:

- [`FactBundle.facts`](/Volumes/ZGMF-X20A/GARYU/snakeAPE/snakeAPE/models.py)

### Solver installation

The solver then injects those translated facts with:

```python
control.add("base", [], facts.facts)
control.ground([("base", [])])
```

in [`snakeAPE/solver.py`](/Volumes/ZGMF-X20A/GARYU/snakeAPE/snakeAPE/solver.py).

This means the Python translator does the semantic expansion once, but clingo still has to:

- parse the whole fact string,
- intern all symbols again,
- ground those facts into the control again.

For `single-shot-*`, that cost is paid once per horizon.
For `multi-shot-*`, that cost is paid once per run.

### Multi-shot grounding

The current multi-shot loop grounds:

- `base` once,
- then `step_initial(1)` or `step(h)` and `check(h)` once per horizon

through `_solve_multi_shot_with_programs()` in [`snakeAPE/solver.py`](/Volumes/ZGMF-X20A/GARYU/snakeAPE/snakeAPE/solver.py).

So the remaining stepwise grounding cost is mostly in:

- `step.lp`
- `step_initial.lp`
- `check.lp`

and in the size/shape of the translated fact base that those programs join against.

## Main Bottlenecks

### 1. Large string-based fact materialization

The translator currently builds a large in-memory text representation and then clingo reparses it.

This has three costs:

- Python string building and copying
- clingo parsing/interning cost
- duplicated representation of the same translated knowledge

### 2. Translation does all expensive lazy preprocessing eagerly

`build_fact_bundle_grounding_opt_lazy()` currently computes:

- lazy candidate port expansions
- bindable-pair discovery
- relevance pruning
- output-choice compression

before the solver even starts grounding.

That is good for solver-side pruning, but it makes Step 1 expensive for large inputs such as Biotools.

### 3. Horizon grounding still has too much query-time work

Even after a one-time `base` grounding, the later `step(h)` and `check(h)` parts still join against a large lazy candidate universe.

That means horizon-2 and horizon-3 grounding are still expensive even in the multi-shot path.

## Improvement Options

## Option A: Structured Fact IR + Direct Injection

This is the highest-value architectural cleanup.

### Idea

Stop treating the translated fact base as a giant LP string only. Instead, make translation produce a structured fact IR and install that IR directly into the solver.

Example IR shape:

- `FactAtom(predicate: str, args: tuple[str | int, ...])`
- or grouped arrays per predicate

### Why it helps

It removes repeated LP text handling and creates a reusable intermediate representation that can support:

- direct control injection,
- optional text rendering for debugging,
- later incremental/delta installation if needed.

### Concrete plan

1. Extend [`FactBundle`](/Volumes/ZGMF-X20A/GARYU/snakeAPE/snakeAPE/models.py) so it can carry both:
   - `facts_text` for debugging and artifact output
   - `fact_atoms` or another structured representation for solver installation
2. Refactor `_FactWriter` in [`snakeAPE/translator.py`](/Volumes/ZGMF-X20A/GARYU/snakeAPE/snakeAPE/translator.py) into a sink abstraction:
   - text sink
   - structured sink
   - optional dual sink
3. Add a solver-side installer such as:
   - `_install_fact_bundle(control, facts)`
4. Prototype two installation modes:
   - direct fact injection through clingo backend or symbolic API
   - if backend-added facts are not visible enough to later grounding, emit a pre-grounded or precompiled facts module instead of reparsing LP text

### Important caveat

This must be validated carefully against clingo’s grounding semantics.

If backend-injected facts do not participate in later grounding the same way as `control.add("base", ..., text)`, then the fallback should be:

- keep the structured IR as the source of truth,
- render LP text only as a compatibility layer,
- or serialize to a more direct load format once and reuse that.

### Expected win

- lower translation-to-grounding handoff cost
- less memory churn
- easier future caching and incremental installation

## Option B: Split Translation Into Static Facts And Candidate Facts

### Idea

Today the fact bundle is one flat block. Split it into:

- common static ontology/config facts
- tool schema facts
- lazy candidate facts
- optional derived helper facts

### Why it helps

This makes it possible to:

- cache stable parts independently,
- inject only the expensive candidate part differently,
- reuse static parts across repeated benchmark runs.

### Concrete plan

1. Add sections to `FactBundle`, for example:
   - `common_facts`
   - `candidate_facts`
   - `derived_facts`
2. Keep the final combined debug output for `translation.lp`, but do not require the solver to consume a single monolithic string.
3. Measure each section separately in the translation summary.

### Expected win

- better observability
- easier caching
- groundwork for direct injection and selective reuse

## Option C: Translation Cache On Disk

### Idea

Persist the translated lazy fact bundle and its metadata so repeated runs on the same config/tool/ontology inputs do not rebuild Step 1 from scratch.

### Suggested cache key

Hash over:

- config file contents relevant to synthesis
- ontology file contents
- tool annotation file contents
- selected translation builder
- code version or git commit

### Cached payload

- structured fact IR
- optional rendered `translation.lp`
- predicate counts / tool stats / cache stats

### Expected win

- big speedup for repeated Biotools parity loops
- makes solver/encoding iteration cheaper even if Step 1 itself is not optimized immediately

### Risk

Low risk if the cache is clearly versioned and opt-in.

## Option D: Stream Translation Instead Of Building Large Intermediate Lists

### Idea

The lazy builder currently accumulates large Python structures such as `candidate_records`, then performs later passes over them.

Some of that is necessary, but some can be streamed or compacted.

### Concrete plan

1. Replace large dict-heavy records with smaller typed tuples or dataclasses.
2. Store compact indexes instead of repeated nested dicts where possible:
   - per-port dimension arrays
   - signature-id indexes
   - producer-to-consumer compatibility maps
3. Emit fact sections incrementally once a phase is complete rather than holding everything as text until the end.
4. Add phase-level progress reporting inside the builder for:
   - candidate expansion
   - bindable-pair construction
   - relevance pruning
   - output compression

### Expected win

- lower peak RSS during translation
- more visible progress on large benchmarks
- cleaner path to later selective caching

## Option E: Precompute More Grounding-Time Helpers In Translation

### Idea

Move more horizon-independent helper work from ASP grounding into the Python translation phase.

### Candidate helpers

- canonicalized signature-sharing relations
- bindability indexes
- output-profile compatibility summaries
- goal-relevant output summaries

### Why it helps

If a relation does not depend on workflow time, it is usually cheaper to emit it once as a fact than to force clingo to rediscover it inside `step(h)` or `check(h)` at every horizon.

### Expected win

- smaller per-horizon grounding joins
- more predictable horizon-2 and horizon-3 grounding times

### Caveat

This should be limited to truly time-independent relations. The earlier failed true incremental lazy attempt showed that time-sensitive availability must not be approximated incorrectly.

## Stepwise Grounding Improvements

## Option F: Split `step.lp` Into Smaller Horizon Parts

### Idea

The current multi-shot loop grounds a fairly broad `step(h)` and `check(h)` block each horizon.

Split the encoding into narrower parts such as:

- `step_seed(h-1)` for carry-forward state only
- `step_run(h)` for tool choice
- `step_bind(h)` for binding constraints
- `check_query(h)` for query-horizon-only checks

### Why it helps

It becomes easier to:

- ground only the pieces needed at each phase,
- isolate the expensive part in timing data,
- move query-only logic out of generic step grounding.

### Expected win

- smaller horizon deltas
- better profiling
- cleaner future tuning

## Option G: Move More Query-Horizon Logic Behind Externals

### Idea

If some query-time constraints do not need new grounding per horizon, activate them through externals instead of grounding a large new `check(h)` block.

### Concrete plan

1. Keep the generic rule skeleton grounded once.
2. Use horizon-specific external atoms such as `query(h)` only to turn checks on and off.
3. Avoid parameterized re-grounding for logic that can be stated over an already-grounded `time/1` domain.

### Why it helps

Re-grounding is often more expensive than flipping an external, especially once the candidate universe is large.

### Caveat

This is only useful when the rule structure itself is reusable. If a rule truly depends on a new horizon parameter to generate fresh ground instances, it still needs a delta grounding step.

## Option H: Ground The Full Time Domain Once For The Multi-Shot Lazy Path

### Idea

For runs where `solution_length_max` is known and moderate, predeclare:

- `time(1..max_horizon)`

and ground more of the step schema once, while still solving incrementally with `query(h)`.

### Why it helps

This trades some upfront base grounding for less repeated per-horizon grounding work.

### Best use

- benchmarking mode
- exact-length parity runs
- cases where horizon-2/horizon-3 grounding dominates and `max_horizon` is small

### Risk

This can backfire if it makes base grounding too large. It should be tested as an optional mode, not as a default change.

## Option I: Separate Query-Only Output Filtering From Generic Step Rules

### Idea

Several parity experiments showed that exact-horizon and query-horizon logic is fragile and expensive.

Make query-only output filtering a dedicated program part instead of mixing it into generic step rules.

### Why it helps

- reduces generic `step(h)` size
- makes exact-horizon behavior easier to debug
- lowers the chance that a parity fix accidentally slows all horizons

## Option J: Revisit Incremental Delta Injection Only After Semantic Safety Is Fixed

### Idea

There is still a valid long-term path where lazy candidate deltas are installed over one persistent control.

### Important constraint

The previous true incremental lazy prototype was not sound with the current `candidate_layers` semantics, because candidates discovered in a later backward relevance layer may still need to be usable at an earlier workflow time.

So:

- do not revive horizon-based candidate activation as a near-term performance fix
- do revive the direct-fact-installation pieces that are independent of that semantic issue

### Safe subset

The following is still worth reusing later:

- structured fact IR
- direct fact installation
- split fact sections
- horizon-independent helper emission

## Recommended Order

## Phase 1: Low-risk wins

1. Add phase timing inside the lazy translator.
2. Split `FactBundle` into sections.
3. Add a disk cache for translated fact bundles.
4. Add a structured fact IR alongside the existing LP text.

## Phase 2: Installation-path cleanup

1. Prototype direct fact installation from the structured IR.
2. Validate whether backend-installed facts behave identically during later grounding.
3. If needed, fall back to a precompiled/pre-rendered facts module rather than raw text reparsing.

## Phase 3: Stepwise grounding reduction

1. Split `step.lp` / `check.lp` into narrower program parts.
2. Move reusable query-time logic behind externals where possible.
3. Benchmark pre-grounding the full time domain once for small `max_length` runs.

## Phase 4: Optional deeper changes

1. Push more horizon-independent helpers from ASP into translation.
2. Revisit safe incremental installation only after the semantic constraints are redesigned.

## Measurement Plan

Use existing timing outputs:

- `translation_sec`
- `base_grounding_sec`
- per-horizon `grounding_sec`
- per-horizon `solving_sec`

Primary benchmarks:

- Biotools length 2
- Biotools length 3
- `defect_concentration` parity reference

Success criteria:

- lower Step 1 translation time on repeated runs
- lower horizon-2 grounding time without changing solution counts
- no regression on accepted parity baselines

## Practical Recommendation

The best next move is not another semantic rewrite.

The best next move is:

1. introduce a structured fact IR,
2. cache translated lazy bundles,
3. prototype direct fact installation from translation,
4. split step/query grounding so horizon-2 work becomes measurable and smaller.

That path improves performance without reopening the correctness problems that came from changing exact-horizon semantics.

