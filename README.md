# snakeAPE

## Running

Run from the `snakeAPE` root with either of these forms:

```bash
python snakeAPE <path-to-config.json> [flags]
```

or:

```bash
python -m snakeAPE <path-to-config.json> [flags]
```

Example from `/Volumes/ZGMF-X20A/GARYU/snakeAPE`:

```bash
python snakeAPE ../ironAPE/APE_Example/defect_concentration/config.json --mode multi-shot-lazy --output-dir /tmp/snakeape-run
```

Run from the workspace root with:

```bash
PYTHONPATH=snakeAPE python -m snakeAPE <path-to-config.json> [flags]
```

Example from `/Volumes/ZGMF-X20A/GARYU`:

```bash
PYTHONPATH=snakeAPE python -m snakeAPE ironAPE/APE_Example/defect_concentration/config.json --mode multi-shot-lazy --output-dir /tmp/snakeape-run
```

## Modes

The CLI supports 3 runtime modes:

- `single-shot`
- `multi-shot`
- `multi-shot-lazy`

Meaning:

- `single-shot`: legacy single-shot runtime schema
- `multi-shot`: APE-style incremental runtime schema
- `multi-shot-lazy`: lazy candidate incremental runtime schema

All runtime ASP encodings are vendored under `snakeAPE/encodings`.

## Common Commands

Normal run:

```bash
PYTHONPATH=snakeAPE python -m snakeAPE ironAPE/APE_Example/defect_concentration/config.json --mode multi-shot-lazy --output-dir /tmp/snakeape-run
```

Normal run from the `snakeAPE` root:

```bash
python snakeAPE ../ironAPE/APE_Example/defect_concentration/config.json --mode multi-shot-lazy --output-dir /tmp/snakeape-run
```

Legacy single-shot:

```bash
PYTHONPATH=snakeAPE python -m snakeAPE ironAPE/APE_Example/defect_concentration/config.json --mode single-shot --solutions 1 --no-graphs --output-dir /tmp/snakeape-single
```

Legacy multi-shot:

```bash
PYTHONPATH=snakeAPE python -m snakeAPE ironAPE/APE_Example/defect_concentration/config.json --mode multi-shot --solutions 1 --no-graphs --output-dir /tmp/snakeape-multi
```

Grounding only:

```bash
PYTHONPATH=snakeAPE python -m snakeAPE ironAPE/APE_Example/biotools/config.json --mode multi-shot-lazy --ground-only --ground-only-stage full --output-dir /tmp/snakeape-ground
```

Translation only:

```bash
PYTHONPATH=snakeAPE python -m snakeAPE ironAPE/APE_Example/defect_concentration/config.json --mode single-shot --translate-only --output-dir /tmp/snakeape-translate
```

Translation only with lazy candidates:

```bash
PYTHONPATH=snakeAPE python -m snakeAPE ironAPE/APE_Example/defect_concentration/config.json --mode multi-shot-lazy --translate-only --output-dir /tmp/snakeape-lazy-translate
```

Multi-shot lazy:

```bash
PYTHONPATH=snakeAPE python -m snakeAPE ironAPE/APE_Example/biotools/config.json --mode multi-shot-lazy --output-dir /tmp/snakeape-multi-lazy
```

## Comparing SAT and snakeAPE Outputs

The comparison utility now lives in the `snakeAPE` root:

```bash
cd snakeAPE
python3 compare_solutions.py <left-solutions-file> <right-solutions-file> [flags]
```

Example comparing APE SAT output against snakeAPE answer sets for exact length 8:

```bash
cd snakeAPE
python3 compare_solutions.py \
  /tmp/ape_sat_compare/sat_output/solutions.txt \
  /tmp/ape_sat_compare/snake_output/answer_sets.txt \
  --left-config /tmp/ape_sat_compare/config_sat_compare.json \
  --left-name SAT \
  --right-name snakeAPE \
  --length 8 \
  --sample-limit 5
```

The comparator explains three levels of agreement:

- exact normalized workflow matches
- same tool sequence but different strict signature
- workflows present on only one side at the tool-sequence level

## Output Artifacts

Each run writes artifacts into `--output-dir` or the directory from the config.

Important files:

- `translation.lp`
- `translation_summary.json`
- `grounding_summary.json` for `--ground-only`
- `solutions.txt` and `workflow_signatures.json` for normal runs
- `answer_sets.txt` only when `--write-raw-answer-sets` is enabled
- `snakeAPE/run_log.csv`
- `snakeAPE/run_summary.csv`

The CSV logs are written to the root of the `snakeAPE` project directory, not to `--output-dir`.

`snakeAPE/run_log.csv` is append-only and records runtime information per completed stage or horizon, including:

- `mode`
- `solver_family`
- `solver_approach`
- `translation_builder`
- `translation_schema`
- timing columns
- peak RSS memory columns
- satisfiability / stored-solution counts

`snakeAPE/run_summary.csv` is append-only and contains one row per completed invocation with total:

- translation time
- base grounding time
- total grounding time
- total solving time
- total rendering time
- total runtime
- final solution count

If a run is interrupted, `snakeAPE/run_log.csv` still contains all stages that were completed before the interruption.

## Useful Flags

- `--mode ...`
- `--output-dir ...`
- `--solutions N`
- `--min-length N`
- `--max-length N`
- `--no-graphs`
- `--ground-only`
- `--ground-only-stage base|full`
- `--translate-only`
- `--write-raw-answer-sets`
