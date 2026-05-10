#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

FULL_BIO_CASES=(
  "test_cases/MassSpectrometry/No1/config_ASP_Masspectrometry_No1_full_bio_tools_WC.json"
  "test_cases/MassSpectrometry/No2/config_ASP_Masspectrometry_No2_full_bio_tools_WC.json"
  "test_cases/MassSpectrometry/No3/config_ASP_Masspectrometry_No3_full_bio_tools_WC.json"
  "test_cases/MassSpectrometry/No4/config_ASP_Masspectrometry_No4_full_bio_tools_WC.json"
)

for config in "${FULL_BIO_CASES[@]}"; do
  echo "==> MassSpectrometry full-bio optimized: $config"
  python -m clindaws "$config" --mode optimized
done
