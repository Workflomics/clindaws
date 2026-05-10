# clindaws

## Installation

Use a Python virtual environment from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m clindaws --help
```

For full installation notes, including optional Java and Graphviz requirements,
see `INSTALL.md`.

## Running

Run from the repository root:

```bash
python -m clindaws <path-to-config.json> [flags]
```

Example:

```bash
python -m clindaws test_cases/GeoGMT/E0/config_ASP_GeoGMT_E0_NC.json --mode multi-shot --output-dir /tmp/clindaws-run
```

## Modes

The CLI supports 3 runtime modes:

- `single-shot`
- `multi-shot`
- `optimized`

Meaning:

- `single-shot`: one-shot solve over a full grounding for `time(1..max_length)`
- `multi-shot`: APE-style incremental grounding and solving
- `optimized`: optimized-candidate multi-shot backend

Backend note:

- `--mode optimized` switches to the optimized-candidate backend under
  `encodings/multi_shot_optimized_candidate`
- `--mode optimized --decomp kcluster` enables the K-Cluster decompression path
- `--mode optimized --decomp one-to-n` enables the 1:N decompression path
- `--decomp 1n` and `--decomp 1:n` are accepted aliases for `one-to-n`
- `--optimized` remains as a compatibility alias for the current optimized
  backend; it does not enable a decompression mode
- `single-shot --optimized` is not implemented yet

All runtime ASP encodings are vendored under `clindaws/encodings`.

## Common Commands

Normal run:

```bash
python -m clindaws test_cases/GeoGMT/E0/config_ASP_GeoGMT_E0_NC.json --mode multi-shot --output-dir /tmp/clindaws-run
```

Single-shot:

```bash
python -m clindaws test_cases/GeoGMT/E0/config_ASP_GeoGMT_E0_NC.json --mode single-shot --solutions 1 --no-graphs --output-dir /tmp/clindaws-single
```

Plain multi-shot:

```bash
python -m clindaws test_cases/GeoGMT/E0/config_ASP_GeoGMT_E0_NC.json --mode multi-shot --solutions 1 --no-graphs --output-dir /tmp/clindaws-multi
```

Grounding only:

```bash
python -m clindaws test_cases/GeoGMT/E0/config_ASP_GeoGMT_E0_NC.json --mode multi-shot --ground-only --ground-only-stage full --output-dir /tmp/clindaws-ground
```

Translation only:

```bash
python -m clindaws test_cases/GeoGMT/E0/config_ASP_GeoGMT_E0_NC.json --mode single-shot --translate-only --output-dir /tmp/clindaws-translate
```

Translation only with optimized-candidate translation:

```bash
python -m clindaws test_cases/GeoGMT/E0/config_ASP_GeoGMT_E0_NC.json --mode optimized --translate-only --output-dir /tmp/clindaws-translate-opt
```

Optimized multi-shot:

```bash
python -m clindaws test_cases/GeoGMT/E0/config_ASP_GeoGMT_E0_NC.json --mode optimized --output-dir /tmp/clindaws-multi-opt
```

Optimized multi-shot with K-Cluster decompression:

```bash
python -m clindaws test_cases/GeoGMT/E0/config_ASP_GeoGMT_E0_NC.json --mode optimized --decomp kcluster --output-dir /tmp/clindaws-multi-kcluster
```

Optimized multi-shot with 1:N decompression:

```bash
python -m clindaws test_cases/GeoGMT/E0/config_ASP_GeoGMT_E0_NC.json --mode optimized --decomp one-to-n --output-dir /tmp/clindaws-multi-1n
```

Parallel translation expansion (8 workers):

```bash
python -m clindaws test_cases/GeoGMT/E0/config_ASP_GeoGMT_E0_NC.json --mode optimized --translation-workers 8 --output-dir /tmp/clindaws-multi-opt-par
```

## Output Artifacts

Each run writes artifacts into `--output-dir` or the directory derived from the config.
No result path flag is required for the provided test cases because the result
location is already defined in each config file. Use `--output-dir` only when
you want to store results somewhere else.

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

## APE Comparison Note

APE and clindaws report per-horizon workflow candidates on different bases.

APE uses a sliding-window style horizon run: the candidates reported at horizon
`H` are the candidates found for that `1..H` window. clindaws multi-shot runs
are cumulative: candidates found at each step are added to the stored workflow
candidate set as the horizon advances.

When comparing benchmark counts, do not compare one APE horizon row directly to
the cumulative clindaws total unless the earlier clindaws horizons are included
in the interpretation. For example, in the constrained defect concentration
variant, clindaws finds 126 workflow candidates at time step 9. APE reports 954
workflow candidates at step 10, while clindaws reports 1080 at step 10 because
the clindaws count is cumulative: `954 + 126 = 1080`.

## Useful Flags

- `--mode single-shot|multi-shot|optimized`
- `--decomp kcluster|one-to-n` — optimized decompression mode; requires `--mode optimized`
- `--grounding python|hybrid|clingo` — grounding strategy (default `hybrid`)
- `--output-dir ...`
- `--solutions N`
- `--min-length N`
- `--max-length N`
- `--parallel-mode ...` — clingo solve parallel mode, e.g. `8,compete`
- `--project` / `--no-project` — enable/disable clingo model projection during solving
- `--no-graphs`
- `--graph-format png|dot|svg`
- `--optimized` — compatibility alias for the optimized backend without decompression
- `--translation-workers N` — worker processes for optimized-candidate translation (default 1, sequential)
- `--ground-only`
- `--ground-only-stage base|full`
- `--translate-only`
- `--write-raw-answer-sets` — emit raw witness-level answer sets for debugging
- `--debug` — print diagnostic raw-model and workflow-candidate counters during solving
- `--benchmark-repetitions N` — repeat the grounding benchmark N times
- `--summary-top-tools N` — include top N expanded tools in translation/grounding summaries
