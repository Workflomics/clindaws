# clindaws

## Running

Run from the workspace root:

```bash
./clindaws/clindaws-cli <path-to-config.json> [flags]
```


## Modes

The CLI supports 3 runtime modes:

- `single-shot`
- `multi-shot`

Meaning:

- `single-shot`: one-shot solve over a full grounding for `time(1..max_length)`
  to `solution_length.max`, stopping once the configured workflow limit is met
- `multi-shot`: APE-style incremental grounding and solving

Backend note:

- `--optimized` is currently supported only for `multi-shot`
- `multi-shot --optimized` switches to the optimized-candidate backend under

All runtime ASP encodings are vendored under `clindaws/encodings`.

## Common Commands

Normal run:

```bash
./clindaws/clindaws-cli ironAPE/APE_Example/defect_concentration/config.json --mode multi-shot --output-dir /tmp/clindaws-run
```

Single-shot:

```bash
./clindaws/clindaws-cli ironAPE/APE_Example/defect_concentration/config.json --mode single-shot --solutions 1 --no-graphs --output-dir /tmp/clindaws-single
```


Plain multi-shot:

```bash
./clindaws/clindaws-cli ironAPE/APE_Example/defect_concentration/config.json --mode multi-shot --solutions 1 --no-graphs --output-dir /tmp/clindaws-multi
```

Grounding only:

```bash
./clindaws/clindaws-cli ironAPE/APE_Example/biotools/config.json --mode multi-shot --ground-only --ground-only-stage full --output-dir /tmp/clindaws-ground
```

Translation only:

```bash
./clindaws/clindaws-cli ironAPE/APE_Example/defect_concentration/config.json --mode single-shot --translate-only --output-dir /tmp/clindaws-translate
```

Translation only with optimized-candidate translation:

```bash
./clindaws/clindaws-cli ironAPE/APE_Example/defect_concentration/config.json --mode multi-shot --optimized --translate-only --output-dir /tmp/clindaws-translate-opt
```

Optimized multi-shot:

```bash
./clindaws/clindaws-cli ironAPE/APE_Example/biotools/config.json --mode multi-shot --optimized --output-dir /tmp/clindaws-multi-opt
```


## Output Artifacts

Each run writes artifacts into `--output-dir` or the directory derived from the config.

Important files:

- `translation.lp`
- `translation_summary.json`
- `grounding_summary.json` for `--ground-only`
- `workflow_signatures__<config>__<mode>__<opt|noopt>[__parallel_<mode>].json` for normal solve runs
- `answer_sets__...txt` only when `--write-raw-answer-sets` is enabled
- `asp_run_log.csv`
- `asp_run_summary.csv`

The CSV logs are written to the output directory for each run (same as `--output-dir`).

`asp_run_log.csv` is append-only and records runtime information per completed stage or horizon, including:

- `mode`
- `solver_family`
- `solver_approach`
- `translation_builder`
- `translation_schema`
- timing columns
- peak RSS memory columns
- `workflow_candidates_found`
- satisfiability / stored-workflow counts

`asp_run_summary.csv` is append-only and contains one row per completed invocation with total:

- translation time
- base grounding time
- total grounding time
- total solving time
- total rendering time
- total runtime
- final solution count

If a run is interrupted, `asp_run_log.csv` still contains all stages that were completed before the interruption.

Current count basis:

- `workflows` in CLI summaries are canonical workflow candidates stored by the solver
- `raw_models` are optional diagnostic counts over pre-canonical clingo answer sets
- `solutions` caps stored canonical workflows; for `single-shot` this cap is enforced after workflow canonicalization rather than as a raw clingo answer-set limit
- the `workflow_signatures__...json` artifact is the primary machine-readable result artifact for
  parity and benchmarking

## Useful Flags

- `--mode single-shot|single-shot-sliding-window|multi-shot`
- `--grounding python|hybrid|clingo` — grounding strategy (default `hybrid`)
- `--output-dir ...`
- `--solutions N`
- `--min-length N`
- `--max-length N`
- `--parallel-mode ...` — clingo solve parallel mode, e.g. `8,compete`
- `--project` / `--no-project` — enable/disable clingo model projection during solving
- `--no-graphs`
- `--graph-format png|dot|svg`
- `--optimized` — enable the optimized backend; for `multi-shot` this selects the optimized-candidate translation/encoding path
- `--translation-workers N` — worker processes for optimized-candidate translation (default 1, sequential)
- `--ground-only`
- `--ground-only-stage base|full`
- `--translate-only`
- `--write-raw-answer-sets` — emit raw witness-level answer sets for debugging
- `--debug` — print diagnostic raw-model and workflow-candidate counters during solving
- `--benchmark-repetitions N` — repeat the grounding benchmark N times
- `--summary-top-tools N` — include top N expanded tools in translation/grounding summaries

## Benchmarks

Test cases and parity references against APE 2.6.1 are maintained in a separate
repository: <https://github.com/Workflomics/APEvASP_test_case_bench>.
