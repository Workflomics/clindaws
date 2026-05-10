#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

APE_JAR="APE/APE-2.6.1-executable.jar"
JAVA_OPTS=(-Xms4g -Xmx24g)

run_ape() {
  local config="$1"
  echo "==> APE SAT: $config"
  java "${JAVA_OPTS[@]}" -jar "$APE_JAR" synthesis "$config" --sat --benchmark
}

while IFS= read -r config; do
  run_ape "$config"
done < <(
  find test_cases -type f \( -name 'config_SAT*.json' -o -name 'config_ASP_SAT*.json' \) | sort
)
