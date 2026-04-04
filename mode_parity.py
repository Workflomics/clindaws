#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

from compare_solutions import build_comparison_report, parse_any
from snakeAPE.runner import run_once


DEFAULT_MODES = ("multi-shot", "single-shot", "multi-shot-lazy")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run snakeAPE mode parity checks against an oracle mode.")
    parser.add_argument("config", help="Path to the config.json file.")
    parser.add_argument(
        "--oracle-mode",
        default="multi-shot-lazy",
        help="Reference mode to compare against.",
    )
    parser.add_argument(
        "--mode",
        dest="modes",
        action="append",
        help="Mode to compare against the oracle. Repeat to add multiple modes.",
    )
    parser.add_argument(
        "--output-root",
        default="/tmp/snakeape-mode-parity",
        help="Directory for per-mode outputs.",
    )
    parser.add_argument(
        "--grounding",
        default="hybrid",
        choices=("python", "hybrid", "clingo"),
        help="Grounding strategy passed to run_once.",
    )
    parser.add_argument("--solutions", type=int, help="Override solution cap.")
    parser.add_argument("--min-length", type=int, help="Override minimum length.")
    parser.add_argument("--max-length", type=int, help="Override maximum length.")
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=10,
        help="Number of mismatch samples to keep in reports.",
    )
    parser.add_argument(
        "--json-out",
        help="Optional path for a JSON summary report.",
    )
    return parser


def _progress(message: str) -> None:
    print(message, flush=True)


def _run_mode(
    *,
    config_path: Path,
    mode: str,
    output_root: Path,
    grounding: str,
    solutions: int | None,
    min_length: int | None,
    max_length: int | None,
) -> Path:
    mode_output_dir = output_root / mode
    result = run_once(
        config_path,
        mode=mode,
        grounding_strategy=grounding,
        output_dir=mode_output_dir,
        solutions=solutions,
        min_length=min_length,
        max_length=max_length,
        render_graphs=False,
        write_raw_answer_sets=True,
        progress_callback=_progress,
    )
    return result.answer_set_path


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    modes = tuple(dict.fromkeys(args.modes or DEFAULT_MODES))
    oracle_mode = args.oracle_mode

    print(f"Running oracle mode: {oracle_mode}")
    oracle_answer_set_path = _run_mode(
        config_path=config_path,
        mode=oracle_mode,
        output_root=output_root,
        grounding=args.grounding,
        solutions=args.solutions,
        min_length=args.min_length,
        max_length=args.max_length,
    )
    oracle_parsed = parse_any(oracle_answer_set_path, config_path)

    reports: dict[str, object] = {
        "config": str(config_path),
        "oracle_mode": oracle_mode,
        "oracle_answer_set_path": str(oracle_answer_set_path),
        "modes": {},
    }

    for mode in modes:
        print(f"Running comparison mode: {mode}")
        answer_set_path = _run_mode(
            config_path=config_path,
            mode=mode,
            output_root=output_root,
            grounding=args.grounding,
            solutions=args.solutions,
            min_length=args.min_length,
            max_length=args.max_length,
        )
        parsed = parse_any(answer_set_path, config_path)
        report = build_comparison_report(
            oracle_parsed,
            parsed,
            left_name=oracle_mode,
            right_name=mode,
            sample_limit=args.sample_limit,
            length=None,
            tool_sequence=None,
        )
        reports["modes"][mode] = report

        counts = report["counts"]
        workflow_level = report["workflow_level"]
        print(
            f"{mode}: "
            f"oracle={counts[f'{oracle_mode}_total_solutions']} "
            f"mode={counts[f'{mode}_total_solutions']} "
            f"exact={workflow_level['exact_workflow_matches']} "
            f"strict={workflow_level['strict_signature_mismatch'][mode]} "
            f"tool_seq={workflow_level['tool_sequence_only_mismatch'][mode]}"
        )

    if args.json_out:
        Path(args.json_out).resolve().write_text(
            json.dumps(reports, indent=2) + "\n",
            encoding="utf-8",
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
