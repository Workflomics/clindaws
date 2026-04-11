"""Command-line interface for snakeAPE."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from clindaws.execution.runner import (
    run_ground_only,
    run_once,
    run_translate_only,
)


def _progress(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _format_count_summary(*, workflows: int, raw_models: int) -> str:
    """Format canonical workflow and raw model counts consistently."""

    return f"workflows={workflows} raw_models={raw_models}"


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""

    parser = argparse.ArgumentParser(description="snakeAPE solver and benchmark CLI")
    parser.add_argument("config", help="Path to an APE-style config.json file.")
    parser.add_argument(
        "--mode",
        choices=(
            "single-shot",
            "multi-shot",
        ),
        default="multi-shot",
        help="Solver mode to execute.",
    )
    parser.add_argument(
        "--grounding",
        choices=("python", "hybrid", "clingo"),
        default="hybrid",
        help="Grounding strategy for a normal run.",
    )
    parser.add_argument(
        "--output-dir",
        help="Override the output directory. Defaults to a sibling snakeAPE_results directory next to the config file.",
    )
    parser.add_argument(
        "--solutions",
        type=int,
        help="Override the solution limit used both for canonical workflow storage and the native clingo model cap.",
    )
    parser.add_argument(
        "--min-length",
        type=int,
        help="Override the minimum solution length.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        help="Override the maximum solution length.",
    )
    parser.add_argument(
        "--parallel-mode",
        help="Override clingo solve parallel mode, for example 8,compete. Applies to solving only.",
    )
    parser.add_argument(
        "--project",
        dest="project",
        action="store_true",
        default=None,
        help="Enable clingo model projection on the encoding's declared projection predicates during solving.",
    )
    parser.add_argument(
        "--no-project",
        dest="project",
        action="store_false",
        help="Disable clingo model projection during solving. Multi-shot enables projection by default.",
    )
    parser.add_argument(
        "--graph-format",
        choices=("png", "dot", "svg"),
        default="png",
        help="Rendered graph format.",
    )
    parser.add_argument(
        "--no-graphs",
        action="store_true",
        help="Skip graph rendering.",
    )
    parser.add_argument(
        "--write-raw-answer-sets",
        action="store_true",
        help="Write raw witness-level answer sets for debugging.",
    )
    parser.add_argument(
        "--optimized",
        action="store_true",
        help="For direct modes, precompute selected static helper relations and bindability facts in Python before grounding.",
    )
    parser.add_argument(
        "--ground-only",
        action="store_true",
        help="Run translation plus grounding only, without solving.",
    )
    parser.add_argument(
        "--translate-only",
        action="store_true",
        help="Run translation only and write a translation summary JSON.",
    )
    parser.add_argument(
        "--ground-only-stage",
        choices=("base", "full"),
        default="base",
        help="Grounding stop point for --ground-only runs.",
    )
    parser.add_argument(
        "--benchmark-repetitions",
        type=int,
        default=1,
        help="Repeat the grounding benchmark this many times.",
    )
    parser.add_argument(
        "--summary-top-tools",
        type=int,
        default=20,
        help="Include this many top expanded tools in translation/grounding summaries.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""

    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.ground_only and args.translate_only:
            parser.error("--ground-only cannot be combined with --translate-only.")
        if args.ground_only_stage != "base" and not args.ground_only:
            parser.error("--ground-only-stage requires --ground-only.")

        if args.ground_only:
            grounding_result = run_ground_only(
                args.config,
                mode=args.mode,
                grounding_strategy=args.grounding,
                stage=args.ground_only_stage,
                output_dir=args.output_dir,
                solutions=args.solutions,
                min_length=args.min_length,
                max_length=args.max_length,
                progress_callback=_progress,
                optimized=args.optimized,
            )
            if grounding_result.translation_path is not None:
                print(f"Translation written to: {grounding_result.translation_path}")
            print(f"Translation summary written to: {grounding_result.translation_summary_path}")
            print(f"Grounding summary written to: {grounding_result.grounding_summary_path}")
            if grounding_result.run_log_path is not None:
                print(f"Run log written to: {grounding_result.run_log_path}")
            if grounding_result.run_summary_path is not None:
                print(f"Run summary written to: {grounding_result.run_summary_path}")
            if grounding_result.grounded_horizons:
                horizons = ", ".join(str(h) for h in grounding_result.grounded_horizons)
                print(f"Grounded horizons: {horizons}")
            else:
                print("Grounded horizons: base only")
            print(
                f"Ground-only run complete: mode={grounding_result.mode} "
                f"strategy={grounding_result.grounding_strategy} stage={grounding_result.stage} "
                f"translation={grounding_result.timings.translation_sec:.3f}s "
                f"grounding={grounding_result.timings.grounding_sec:.3f}s "
                f"total={grounding_result.timings.total_sec:.3f}s"
            )
            return 0

        if args.translate_only:
            translation_result = run_translate_only(
                args.config,
                mode=args.mode,
                grounding_strategy=args.grounding,
                output_dir=args.output_dir,
                solutions=args.solutions,
                min_length=args.min_length,
                max_length=args.max_length,
                progress_callback=_progress,
                optimized=args.optimized,
            )
            if translation_result.translation_path is not None:
                print(f"Translation written to: {translation_result.translation_path}")
            print(f"Translation summary written to: {translation_result.translation_summary_path}")
            if translation_result.run_log_path is not None:
                print(f"Run log written to: {translation_result.run_log_path}")
            if translation_result.run_summary_path is not None:
                print(f"Run summary written to: {translation_result.run_summary_path}")
            print(
                f"Translate-only run complete: mode={translation_result.mode} "
                f"strategy={translation_result.grounding_strategy} "
                f"translation={translation_result.timings.translation_sec:.3f}s "
                f"total={translation_result.timings.total_sec:.3f}s"
            )
            return 0

        run_result = run_once(
            args.config,
            mode=args.mode,
            grounding_strategy=args.grounding,
            output_dir=args.output_dir,
            solutions=args.solutions,
            min_length=args.min_length,
            max_length=args.max_length,
            parallel_mode=args.parallel_mode,
            project_models=args.project,
            graph_format=args.graph_format,
            render_graphs=not args.no_graphs,
            write_raw_answer_sets=args.write_raw_answer_sets,
            progress_callback=_progress,
            optimized=args.optimized,
        )
        if run_result.translation_path is not None:
            print(f"Translation written to: {run_result.translation_path}")
        if run_result.run_log_path is not None:
            print(f"Run log written to: {run_result.run_log_path}")
        if run_result.run_summary_path is not None:
            print(f"Run summary written to: {run_result.run_summary_path}")
        if run_result.timed_out:
            print(
                f"Run timed out after configured limit during {run_result.completed_stage}.",
                file=sys.stderr,
                flush=True,
            )
            return 124
        if run_result.answer_set_path is not None:
            print(f"Answer sets written to: {run_result.answer_set_path}")
        if run_result.solution_summary_path is not None:
            print(f"Solutions written to: {run_result.solution_summary_path}")
        if run_result.workflow_signature_path is not None:
            print(f"Workflow signatures written to: {run_result.workflow_signature_path}")
        print(
            f"Run complete: mode={run_result.mode} strategy={run_result.grounding_strategy} "
            f"{_format_count_summary(workflows=run_result.unique_solutions_found, raw_models=run_result.raw_models_seen)} "
            f"translation={run_result.timings.translation_sec:.3f}s "
            f"grounding={run_result.timings.grounding_sec:.3f}s "
            f"solving={run_result.timings.solving_sec:.3f}s "
            f"total={run_result.timings.total_sec:.3f}s"
        )
        if run_result.raw_models_seen != run_result.unique_solutions_found:
            print(
                "Count basis: workflows are unique tool sequences; "
                "raw_models are pre-canonical answer sets seen by clingo."
            )
        for path in run_result.graph_paths:
            print(f"Graph artifact: {Path(path)}")
        return 0
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr, flush=True)
        return 130
