# Installation

These instructions assume you are working from the repository root after
extracting the Zenodo archive.

## Python Environment

Use Python 3.10 or newer.

Create and activate a local virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the Python runtime dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Check that the CLI starts:

```bash
python -m clindaws --help
```

## Optional System Dependencies

APE benchmark scripts require Java and use the vendored APE jar:

```bash
java -version
bash test_scripts/run_test_cases_ape.sh
```

Graph rendering to PNG or SVG requires Graphviz's `dot` executable:

```bash
dot -V
```

Graphviz is not needed when using `--no-graphs` or `--graph-format dot`.

## Smoke Test

Run a small optimized solve without graph rendering:

```bash
python -m clindaws test_cases/GeoGMT/E0/config_ASP_GeoGMT_E0_NC.json --mode optimized --solutions 1 --no-graphs
```
