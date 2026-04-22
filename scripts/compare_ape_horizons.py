#!/usr/bin/env python3
"""Compare local horizon counts against APE logs and classify divergences.

This script is intentionally small and repo-local so benchmark parity work can
be driven from the APE run logs instead of ad hoc inspection of individual
artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable


def _load_ape_counts(path: Path) -> dict[int, int]:
    counts: dict[int, int] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            horizon = int(row["horizon"])
            counts[horizon] = int(row["solutions_found"])
    return counts


def _load_local_counts(path: Path) -> dict[int, int]:
    if path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        horizons = payload["horizons"] if isinstance(payload, dict) else payload
        return {
            int(record["horizon"]): int(record.get("unique_workflows_stored", 0))
            for record in horizons
        }

    counts: dict[int, int] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            horizon_text = row.get("horizon") or row.get("length")
            count_text = row.get("solutions_found") or row.get("unique_workflows_stored")
            if not horizon_text or not count_text:
                continue
            counts[int(horizon_text)] = int(count_text)
    return counts


def _format_counts(label: str, counts: dict[int, int], horizons: Iterable[int]) -> list[str]:
    return [f"{label:10} " + " ".join(f"h{h}={counts.get(h, 0)}" for h in horizons)]


def _classify(
    ape: dict[int, int],
    plain: dict[int, int] | None,
    optimized: dict[int, int] | None,
) -> str:
    if plain is None or optimized is None:
        return "insufficient-data"
    if plain == ape and optimized == ape:
        return "all-match"
    if plain == ape and optimized != ape:
        return "optimized-only"
    if plain != ape and optimized == ape:
        return "plain-only"
    if plain == optimized and plain != ape:
        return "shared-key-likely"
    return "mixed"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ape-log", required=True, type=Path, help="APE benchmark ape_run_log.csv")
    parser.add_argument("--plain", type=Path, help="Local plain multi-shot horizon_summary.json or csv")
    parser.add_argument("--optimized", type=Path, help="Local optimized multi-shot horizon_summary.json or csv")
    parser.add_argument("--fail-on-mismatch", action="store_true", help="Return exit code 1 unless all compared counts match APE")
    args = parser.parse_args()

    ape_counts = _load_ape_counts(args.ape_log)
    plain_counts = _load_local_counts(args.plain) if args.plain else None
    optimized_counts = _load_local_counts(args.optimized) if args.optimized else None

    all_horizons = sorted(
        set(ape_counts)
        | (set(plain_counts) if plain_counts is not None else set())
        | (set(optimized_counts) if optimized_counts is not None else set())
    )

    print("\n".join(_format_counts("APE", ape_counts, all_horizons)))
    if plain_counts is not None:
        print("\n".join(_format_counts("plain", plain_counts, all_horizons)))
    if optimized_counts is not None:
        print("\n".join(_format_counts("optimized", optimized_counts, all_horizons)))

    classification = _classify(ape_counts, plain_counts, optimized_counts)
    print(f"classification: {classification}")

    if not args.fail_on_mismatch:
        return 0

    compared = [counts for counts in (plain_counts, optimized_counts) if counts is not None]
    if all(counts == ape_counts for counts in compared):
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
