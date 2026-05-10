#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

bash "$SCRIPT_DIR/run_test_cases_ape.sh"
bash "$SCRIPT_DIR/run_test_cases_multi-shot.sh"
bash "$SCRIPT_DIR/run_test_cases_optimized.sh"
bash "$SCRIPT_DIR/run_full_bio.sh"
