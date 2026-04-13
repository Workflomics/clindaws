# snakeAPE

## Running

Run from the workspace root (`/Volumes/ZGMF-X20A/GARYU`):

```bash
PYTHONPATH=snakeAPE python -m clindaws <path-to-config.json> [flags]
```

Example:

```bash
PYTHONPATH=snakeAPE python -m clindaws ironAPE/APE_Example/defect_concentration/config.json --mode multi-shot --output-dir /tmp/snakeape-run
```

## Modes

The CLI supports 2 runtime modes:

- `single-shot`
- `multi-shot`

Meaning:

- `single-shot`: one-shot solve over a full grounding for `time(1..max_length)`
- `multi-shot`: APE-style incremental grounding and solving

Backend note:

- `--optimized` is currently supported only for `multi-shot`
- `multi-shot --optimized` switches to the compressed-candidate backend under
  `encodings/multi_shot_compressed_candidate`
- `single-shot --optimized` is not implemented yet

All runtime ASP encodings are vendored under `snakeAPE/encodings`.

## Common Commands

Normal run:

```bash
PYTHONPATH=snakeAPE python -m clindaws ironAPE/APE_Example/defect_concentration/config.json --mode multi-shot --output-dir /tmp/snakeape-run
```

Single-shot:

```bash
PYTHONPATH=snakeAPE python -m clindaws ironAPE/APE_Example/defect_concentration/config.json --mode single-shot --solutions 1 --no-graphs --output-dir /tmp/snakeape-single
```

Plain multi-shot:

```bash
PYTHONPATH=snakeAPE python -m clindaws ironAPE/APE_Example/defect_concentration/config.json --mode multi-shot --solutions 1 --no-graphs --output-dir /tmp/snakeape-multi
```

Grounding only:

```bash
PYTHONPATH=snakeAPE python -m clindaws ironAPE/APE_Example/biotools/config.json --mode multi-shot --ground-only --ground-only-stage full --output-dir /tmp/snakeape-ground
```

Translation only:

```bash
PYTHONPATH=snakeAPE python -m clindaws ironAPE/APE_Example/defect_concentration/config.json --mode single-shot --translate-only --output-dir /tmp/snakeape-translate
```

Translation only with compressed-candidate optimization:

```bash
PYTHONPATH=snakeAPE python -m clindaws ironAPE/APE_Example/defect_concentration/config.json --mode multi-shot --optimized --translate-only --output-dir /tmp/snakeape-translate-opt
```

Optimized multi-shot:

```bash
PYTHONPATH=snakeAPE python -m clindaws ironAPE/APE_Example/biotools/config.json --mode multi-shot --optimized --output-dir /tmp/snakeape-multi-opt
```

Parallel translation expansion (8 workers):

```bash
PYTHONPATH=snakeAPE python -m clindaws ironAPE/APE_Example/biotools/config.json --mode multi-shot --optimized --translation-workers 8 --output-dir /tmp/snakeape-multi-opt-par
```

## Output Artifacts

Each run writes artifacts into `--output-dir` or the directory derived from the config.

Important files:

- `translation.lp`
- `translation_summary.json`
- `grounding_summary.json` for `--ground-only`
- `workflow_signatures.json` for normal solve runs
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
- satisfiability / stored-solution counts

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
- `workflow_signatures.json` is the primary machine-readable result artifact for
  parity and benchmarking

## Useful Flags

- `--mode single-shot|multi-shot`
- `--grounding python|hybrid|clingo` — grounding strategy (default `hybrid`)
- `--output-dir ...`
- `--solutions N`
- `--min-length N`
- `--max-length N`
- `--parallel-mode ...` — clingo solve parallel mode, e.g. `8,compete`
- `--project` / `--no-project` — enable/disable clingo model projection during solving
- `--no-graphs`
- `--graph-format png|dot|svg`
- `--optimized` — enable the optimized backend; for `multi-shot` this selects the compressed-candidate translation/encoding path
- `--translation-workers N` — worker processes for optimized compressed-candidate translation (default 1, sequential)
- `--ground-only`
- `--ground-only-stage base|full`
- `--translate-only`
- `--write-raw-answer-sets` — emit raw witness-level answer sets for debugging
- `--debug` — print diagnostic raw-model and workflow-candidate counters during solving
- `--benchmark-repetitions N` — repeat the grounding benchmark N times
- `--summary-top-tools N` — include top N expanded tools in translation/grounding summaries
