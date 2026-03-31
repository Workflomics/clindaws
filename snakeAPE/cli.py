"""Command-line interface for snakeAPE."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .runner import (
    benchmark_grounding,
    run_ground_only,
    run_once,
    run_translate_only,
    run_translate_only_full_variants,
    run_translate_only_lazy,
)


def _progress(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""

    parser = argparse.ArgumentParser(description="snakeAPE solver and benchmark CLI")
    parser.add_argument("config", help="Path to an APE-style config.json file.")
    parser.add_argument(
        "--mode",
        choices=(
            "single-shot",
            "single-shot-opt",
            "single-shot-lazy",
            "multi-shot",
            "multi-shot-opt",
            "multi-shot-lazy",
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
        help="Override the number of solutions to search for.",
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
        "--benchmark-grounding",
        action="store_true",
        help="Run python, hybrid, and clingo grounding strategies and write a benchmark CSV.",
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
        "--translate-only-full-variants",
        action="store_true",
        help="Run translate-only using the eager full-variant candidate expansion.",
    )
    parser.add_argument(
        "--translate-only-lazy",
        action="store_true",
        help="Run translate-only using the lazy candidate expansion.",
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
        if args.benchmark_grounding and args.ground_only:
            parser.error("--benchmark-grounding cannot be combined with --ground-only.")
        if args.benchmark_grounding and args.translate_only:
            parser.error("--benchmark-grounding cannot be combined with --translate-only.")
        if args.benchmark_grounding and args.translate_only_full_variants:
            parser.error("--benchmark-grounding cannot be combined with --translate-only-full-variants.")
        if args.benchmark_grounding and args.translate_only_lazy:
            parser.error("--benchmark-grounding cannot be combined with --translate-only-lazy.")
        if args.ground_only and args.translate_only:
            parser.error("--ground-only cannot be combined with --translate-only.")
        if args.ground_only and args.translate_only_full_variants:
            parser.error("--ground-only cannot be combined with --translate-only-full-variants.")
        if args.ground_only and args.translate_only_lazy:
            parser.error("--ground-only cannot be combined with --translate-only-lazy.")
        if args.translate_only and args.translate_only_full_variants:
            parser.error("--translate-only cannot be combined with --translate-only-full-variants.")
        if args.translate_only and args.translate_only_lazy:
            parser.error("--translate-only cannot be combined with --translate-only-lazy.")
        if args.translate_only_full_variants and args.translate_only_lazy:
            parser.error("--translate-only-full-variants cannot be combined with --translate-only-lazy.")
        if args.ground_only_stage != "base" and not args.ground_only:
            parser.error("--ground-only-stage requires --ground-only.")
        if args.summary_top_tools < 1:
            parser.error("--summary-top-tools must be at least 1.")
        if args.translate_only_full_variants and args.mode not in {"single-shot-opt", "multi-shot-opt"}:
            parser.error("--translate-only-full-variants requires --mode single-shot-opt or --mode multi-shot-opt.")
        if args.translate_only_lazy and args.mode not in {
            "single-shot-opt",
            "single-shot-lazy",
            "multi-shot-opt",
            "multi-shot-lazy",
        }:
            parser.error(
                "--translate-only-lazy requires --mode single-shot-opt, single-shot-lazy, multi-shot-opt, or multi-shot-lazy."
            )

        if args.benchmark_grounding:
            result = benchmark_grounding(
                args.config,
                mode=args.mode,
                output_dir=args.output_dir,
                solutions=args.solutions,
                min_length=args.min_length,
                max_length=args.max_length,
                graph_format=args.graph_format,
                render_graphs=not args.no_graphs,
                repetitions=args.benchmark_repetitions,
                progress_callback=_progress,
            )
            print(f"Benchmark written to: {result.output_path}")
            for record in result.records:
                print(
                    f"[rep {record.repetition}] {record.strategy}: "
                    f"translation={record.translation_sec:.3f}s "
                    f"grounding={record.grounding_sec:.3f}s "
                    f"solving={record.solving_sec:.3f}s "
                    f"total={record.total_sec:.3f}s "
                    f"raw_solutions={record.raw_solutions_found} "
                    f"unique_solutions={record.solutions_found}"
                )
            return 0

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
                summary_top_tools=args.summary_top_tools,
                progress_callback=_progress,
            )
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
                f"grounding={grounding_result.timings.grounding_sec:.3f}s"
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
                summary_top_tools=args.summary_top_tools,
                progress_callback=_progress,
            )
            print(f"Translation written to: {translation_result.translation_path}")
            print(f"Translation summary written to: {translation_result.translation_summary_path}")
            if translation_result.run_log_path is not None:
                print(f"Run log written to: {translation_result.run_log_path}")
            if translation_result.run_summary_path is not None:
                print(f"Run summary written to: {translation_result.run_summary_path}")
            print(
                f"Translate-only run complete: mode={translation_result.mode} "
                f"strategy={translation_result.grounding_strategy} "
                f"translation={translation_result.timings.translation_sec:.3f}s"
            )
            return 0

        if args.translate_only_full_variants:
            translation_result = run_translate_only_full_variants(
                args.config,
                mode=args.mode,
                grounding_strategy=args.grounding,
                output_dir=args.output_dir,
                solutions=args.solutions,
                min_length=args.min_length,
                max_length=args.max_length,
                summary_top_tools=args.summary_top_tools,
                progress_callback=_progress,
            )
            print(f"Translation written to: {translation_result.translation_path}")
            print(f"Translation summary written to: {translation_result.translation_summary_path}")
            if translation_result.run_log_path is not None:
                print(f"Run log written to: {translation_result.run_log_path}")
            if translation_result.run_summary_path is not None:
                print(f"Run summary written to: {translation_result.run_summary_path}")
            print(
                f"Translate-only full-variants run complete: mode={translation_result.mode} "
                f"strategy={translation_result.grounding_strategy} "
                f"translation={translation_result.timings.translation_sec:.3f}s"
            )
            return 0

        if args.translate_only_lazy:
            translation_result = run_translate_only_lazy(
                args.config,
                mode=args.mode,
                grounding_strategy=args.grounding,
                output_dir=args.output_dir,
                solutions=args.solutions,
                min_length=args.min_length,
                max_length=args.max_length,
                summary_top_tools=args.summary_top_tools,
                progress_callback=_progress,
            )
            print(f"Translation written to: {translation_result.translation_path}")
            print(f"Translation summary written to: {translation_result.translation_summary_path}")
            if translation_result.run_log_path is not None:
                print(f"Run log written to: {translation_result.run_log_path}")
            if translation_result.run_summary_path is not None:
                print(f"Run summary written to: {translation_result.run_summary_path}")
            print(
                f"Translate-only lazy run complete: mode={translation_result.mode} "
                f"strategy={translation_result.grounding_strategy} "
                f"translation={translation_result.timings.translation_sec:.3f}s"
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
            graph_format=args.graph_format,
            render_graphs=not args.no_graphs,
            progress_callback=_progress,
        )
        print(f"Translation written to: {run_result.translation_path}")
        print(f"Answer sets written to: {run_result.answer_set_path}")
        print(f"Solutions written to: {run_result.solution_summary_path}")
        if run_result.run_log_path is not None:
            print(f"Run log written to: {run_result.run_log_path}")
        if run_result.run_summary_path is not None:
            print(f"Run summary written to: {run_result.run_summary_path}")
        print(
            f"Run complete: mode={run_result.mode} strategy={run_result.grounding_strategy} "
            f"raw_solutions={run_result.raw_answer_sets_found} "
            f"unique_solutions={run_result.unique_solutions_found} "
            f"translation={run_result.timings.translation_sec:.3f}s "
            f"grounding={run_result.timings.grounding_sec:.3f}s solving={run_result.timings.solving_sec:.3f}s"
        )
        for path in run_result.graph_paths:
            print(f"Graph artifact: {Path(path)}")
        return 0
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr, flush=True)
        return 130
