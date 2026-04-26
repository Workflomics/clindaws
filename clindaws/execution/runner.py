"""High-level execution entrypoints.

The runner owns the end-to-end lifecycle around the lower-level clingo solver:

- load and normalize configuration,
- choose the effective translation/backend family,
- build the fact bundle and translation summaries,
- execute grounding/solving in a worker process with timeout control,
- reconstruct canonical workflow candidates and write run artifacts.

Public modes stay small (`single-shot`, `single-shot-sliding-window`, `multi-shot`), while this module maps
them onto the internal backends used by translation and solving.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from time import perf_counter

from clindaws.core.config import load_config, SnakeConfig
from clindaws.core.models import (
    FactBundle,
    GroundingRunResult,
    HorizonRecord,
    RunResult,
    TimingBreakdown,
    TranslationRunResult,
)
from clindaws.core.ontology import Ontology
from clindaws.rendering.rendering import (
    render_solution_graphs,
    write_workflow_signatures,
)
from clindaws.core.runtime_stats import (
    ProcessTreePeakMonitor,
    current_peak_rss_mb,
)
from clindaws.execution.runner_modes import (
    ProgressCallback,
    _GROUNDER_DISPATCH,
    _effective_translation_strategy,
    _mode_config,
)
from clindaws.execution.runner_utils import (
    _answer_sets_filename,
    _combined_peak_mb,
    _compressed_candidate_engaged,
    _default_solution_dir,
    _effective_internal_solver_mode,
    _effective_parallel_mode,
    _effective_project_models,
    _effective_solve_start_horizon,
    _is_clasp_id_overflow,
    _records_with_combined_peak_rss,
    _report,
    _sanitize_filename_token,
    _validate_run_config,
    _workflow_signatures_filename,
)
from clindaws.execution.runner_worker import (
    _run_solve_in_worker,
    _timed_out_run_result,
)
from clindaws.execution.runner_bundle import (
    _load_tools_for_mode,
    _select_fact_bundle,
    _tool_output_dims_lookup,
    _workflow_input_dims_from_config,
)
from clindaws.execution.runner_translation import (
    _run_metadata_payload,
    _solve_callback_profile_payload,
    _translation_schema,
    _workflow_input_compression_payload,
    _write_translation_summary,
)
from clindaws.execution.runner_output import (
    _RunCsvWriters,
    _horizon_record_payload,
)
from clindaws.core.workflow import reconstruct_solution


@dataclass(frozen=True)
class RunContext:
    """All translation-phase results, passed from _prepare_run_context to callers."""

    config: SnakeConfig
    ontology: Ontology
    solution_dir: Path
    fact_bundle: FactBundle
    workflow_input_dims: dict[str, dict[str, tuple[str, ...]]]
    tool_output_dims: dict[tuple[str, int], dict[str, tuple[str, ...]]]
    translation_sec: float
    translation_peak_rss_mb: float
    translation_peak_combined_rss_mb: float
    effective_translation_strategy: str
    resolved_translation_builder: str
    run_metadata: dict[str, object]


def _prepare_run_context(
    config_path: str | Path,
    *,
    mode: str,
    grounding_strategy: str,
    output_dir: str | Path | None = None,
    solutions: int | None = None,
    min_length: int | None = None,
    max_length: int | None = None,
    progress_callback: ProgressCallback = None,
    optimized: bool = False,
    max_workers: int = 1,
    memory_monitor: ProcessTreePeakMonitor | None = None,
) -> RunContext:
    """Load config and build the translation-phase context for one run.

    The returned context is the hand-off point between translation and the later
    solve/render phases. It packages the resolved config, output directory,
    backend-specific fact bundle, and translation diagnostics in one named
    object so callers do not need to thread anonymous tuples around.
    """

    mode_config = _mode_config(mode)
    if optimized and mode_config.solver_family == "single-shot":
        raise ValueError("--optimized is not yet supported for single-shot modes.")
    if optimized and mode != "multi-shot":
        raise ValueError("--optimized supports only multi-shot.")
    config = load_config(config_path)
    config = config.model_copy(update={
        "solutions": solutions if solutions is not None else config.solutions,
        "solution_length_min": min_length if min_length is not None else config.solution_length_min,
        "solution_length_max": max_length if max_length is not None else config.solution_length_max,
    })
    _validate_run_config(config)
    solution_dir = Path(output_dir).resolve() if output_dir else _default_solution_dir(config)
    solution_dir.mkdir(parents=True, exist_ok=True)

    effective_translation_strategy = _effective_translation_strategy(mode, grounding_strategy)

    _report(progress_callback, "Step 1: translation started.")
    start = perf_counter()
    ontology = Ontology.from_file(config.ontology_path, config.ontology_prefix)
    tools = _load_tools_for_mode(config, mode_config.translation_pathway)
    run_metadata = _run_metadata_payload(config=config, ontology=ontology, tools=tools)
    fact_bundle, resolved_translation_builder = _select_fact_bundle(
        mode_config=mode_config,
        mode=mode,
        config=config,
        ontology=ontology,
        tools=tools,
        optimized=optimized,
        progress_callback=progress_callback,
        max_workers=max_workers,
    )
    translation_sec = perf_counter() - start
    translation_peak_rss_mb = current_peak_rss_mb()
    if memory_monitor is not None:
        memory_monitor.sample_now()
        translation_peak_combined_rss_mb = memory_monitor.current_peak_mb()
    else:
        translation_peak_combined_rss_mb = translation_peak_rss_mb
    _report(progress_callback, f"Step 1 complete: translation finished after {translation_sec:.3f}s.")

    return RunContext(
        config=config,
        ontology=ontology,
        solution_dir=solution_dir,
        fact_bundle=fact_bundle,
        workflow_input_dims=_workflow_input_dims_from_config(config),
        tool_output_dims=_tool_output_dims_lookup(tools),
        translation_sec=translation_sec,
        translation_peak_rss_mb=translation_peak_rss_mb,
        translation_peak_combined_rss_mb=translation_peak_combined_rss_mb,
        effective_translation_strategy=effective_translation_strategy,
        resolved_translation_builder=resolved_translation_builder,
        run_metadata=run_metadata,
    )


def run_translate_only(
    config_path: str | Path,
    *,
    mode: str,
    grounding_strategy: str,
    output_dir: str | Path | None = None,
    solutions: int | None = None,
    min_length: int | None = None,
    max_length: int | None = None,
    progress_callback: ProgressCallback = None,
    optimized: bool = False,
    max_workers: int = 1,
) -> TranslationRunResult:
    """Run translation only and write translation diagnostics."""

    with ProcessTreePeakMonitor() as memory_monitor:
        ctx = _prepare_run_context(
            config_path,
            mode=mode,
            grounding_strategy=grounding_strategy,
            output_dir=output_dir,
            solutions=solutions,
            min_length=min_length,
            max_length=max_length,
            progress_callback=progress_callback,
            optimized=optimized,
            max_workers=max_workers,
            memory_monitor=memory_monitor,
        )
        csv_writers = _RunCsvWriters(
            csv_dir=ctx.config.solutions_dir_path,
            mode=mode,
            grounding_strategy=grounding_strategy,
            fact_count=ctx.fact_bundle.fact_count,
            run_metadata=ctx.run_metadata,
            translation_builder=ctx.resolved_translation_builder,
            translation_schema=_translation_schema(ctx.fact_bundle),
            optimized_enabled=optimized,
            effective_parallel_mode=None,
            compressed_candidate_engaged=_compressed_candidate_engaged(ctx.fact_bundle),
        )
        csv_writers.step_writer.log_translation(translation_sec=ctx.translation_sec)

        translation_summary_path, _ = _write_translation_summary(
            config=ctx.config,
            solution_dir=ctx.solution_dir,
            mode=mode,
            grounding_strategy=grounding_strategy,
            translation_builder=ctx.resolved_translation_builder,
            effective_translation_strategy=ctx.effective_translation_strategy,
            fact_bundle=ctx.fact_bundle,
            translation_path=None,
            translation_sec=ctx.translation_sec,
        )
        combined_peak_rss_mb = _combined_peak_mb(memory_monitor)
        csv_writers.summary_writer.log_summary(
            completed_stage="translate_only",
            timings=TimingBreakdown(
                translation_sec=ctx.translation_sec,
                grounding_sec=0.0,
                solving_sec=0.0,
                rendering_sec=0.0,
            ),
            translation_peak_rss_mb=ctx.translation_peak_combined_rss_mb,
            combined_peak_rss_mb=combined_peak_rss_mb,
            base_grounding_sec=0.0,
            base_grounding_peak_rss_mb=0.0,
            horizon_records=(),
            raw_solutions_found=0,
            raw_models_seen=0,
            solutions_found=0,
            effective_parallel_mode=None,
        )
        run_log_path = csv_writers.run_log_path
        run_summary_path = csv_writers.run_summary_path

        return TranslationRunResult(
            config=ctx.config,
            mode=mode,
            grounding_strategy=grounding_strategy,
            translation_builder=ctx.resolved_translation_builder,
            effective_translation_strategy=ctx.effective_translation_strategy,
            fact_bundle=ctx.fact_bundle,
            timings=TimingBreakdown(
                translation_sec=ctx.translation_sec,
                grounding_sec=0.0,
                solving_sec=0.0,
                rendering_sec=0.0,
            ),
            translation_peak_rss_mb=ctx.translation_peak_combined_rss_mb,
            combined_peak_rss_mb=combined_peak_rss_mb,
            translation_path=None,
            translation_summary_path=translation_summary_path,
            run_log_path=run_log_path,
            run_summary_path=run_summary_path,
        )


def run_once(
    config_path: str | Path,
    *,
    mode: str,
    grounding_strategy: str,
    output_dir: str | Path | None = None,
    solutions: int | None = None,
    min_length: int | None = None,
    max_length: int | None = None,
    parallel_mode: str | None = None,
    project_models: bool | None = None,
    graph_format: str = "png",
    render_graphs: bool = True,
    write_raw_answer_sets: bool = False,
    debug: bool = False,
    progress_callback: ProgressCallback = None,
    optimized: bool = False,
    max_workers: int = 1,
) -> RunResult:
    """Run one snakeAPE execution."""

    with ProcessTreePeakMonitor() as memory_monitor:
        ctx = _prepare_run_context(
            config_path,
            mode=mode,
            grounding_strategy=grounding_strategy,
            output_dir=output_dir,
            solutions=solutions,
            min_length=min_length,
            max_length=max_length,
            progress_callback=progress_callback,
            optimized=optimized,
            max_workers=max_workers,
            memory_monitor=memory_monitor,
        )
        config = ctx.config
        solution_dir = ctx.solution_dir
        fact_bundle = ctx.fact_bundle
        translation_sec = ctx.translation_sec
        translation_peak_rss_mb = ctx.translation_peak_combined_rss_mb
        run_metadata = ctx.run_metadata
        _translation_builder = ctx.resolved_translation_builder
        effective_parallel_mode = _effective_parallel_mode(mode, parallel_mode, optimized)
        internal_solver_mode = _effective_internal_solver_mode(mode, fact_bundle)
        csv_writers = _RunCsvWriters(
            csv_dir=config.solutions_dir_path,
            mode=mode,
            grounding_strategy=grounding_strategy,
            fact_count=fact_bundle.fact_count,
            run_metadata=run_metadata,
            translation_builder=_translation_builder,
            translation_schema=_translation_schema(fact_bundle),
            optimized_enabled=optimized,
            effective_parallel_mode=effective_parallel_mode,
            compressed_candidate_engaged=_compressed_candidate_engaged(fact_bundle),
        )
        csv_writers.step_writer.log_translation(translation_sec=translation_sec)

        effective_project_models = _effective_project_models(mode, project_models)
        diagnostic_counts_enabled = bool(debug or write_raw_answer_sets or mode == "multi-shot")
        capture_raw_models = bool(write_raw_answer_sets)
        if effective_parallel_mode:
            _report(progress_callback, f"Step 3a: effective solve parallel mode is {effective_parallel_mode}.")
        remaining_timeout = config.timeout_sec - translation_sec
        _timed_out = False
        _solve_start = perf_counter()
        if remaining_timeout <= 0:
            _timed_out = True
            _report(
                progress_callback,
                f"Translation already exceeded timeout ({config.timeout_sec}s); skipping solve.",
            )
            combined_peak_rss_mb = _combined_peak_mb(memory_monitor)
            csv_writers.step_writer.log_timeout(elapsed_ms=0, memory_used_mb=combined_peak_rss_mb)
            csv_writers.summary_writer.log_summary(
                completed_stage="translation_timeout",
                timings=TimingBreakdown(
                    translation_sec=translation_sec,
                    grounding_sec=0.0,
                    solving_sec=0.0,
                    rendering_sec=0.0,
                ),
                translation_peak_rss_mb=translation_peak_rss_mb,
                combined_peak_rss_mb=combined_peak_rss_mb,
                base_grounding_sec=0.0,
                base_grounding_peak_rss_mb=0.0,
                horizon_records=(),
                raw_solutions_found=0,
                raw_models_seen=0,
                solutions_found=0,
                effective_parallel_mode=effective_parallel_mode,
            )
            return _timed_out_run_result(
                config=config,
                mode=mode,
                grounding_strategy=grounding_strategy,
                fact_bundle=fact_bundle,
                translation_sec=translation_sec,
                translation_peak_rss_mb=translation_peak_rss_mb,
                combined_peak_rss_mb=combined_peak_rss_mb,
                solve_start=_solve_start,
                completed_stage="translation_timeout",
                run_log_path=csv_writers.run_log_path,
                run_summary_path=csv_writers.run_summary_path,
            )

        try:
            solve_output, _timed_out, horizon_peak_rss_by_horizon = _run_solve_in_worker(
                mode=internal_solver_mode,
                config=config,
                ontology=ctx.ontology,
                fact_bundle=fact_bundle,
                workflow_input_dims=ctx.workflow_input_dims,
                tool_output_dims=ctx.tool_output_dims,
                capture_raw_models=capture_raw_models,
                diagnostic_counts_enabled=diagnostic_counts_enabled,
                parallel_mode=effective_parallel_mode,
                project_models=effective_project_models,
                remaining_timeout=remaining_timeout,
                progress_callback=progress_callback,
                memory_monitor=memory_monitor,
            )
        except RuntimeError as exc:
            if (
                mode == "multi-shot"
                and not optimized
                and _is_clasp_id_overflow(exc)
            ):
                _report(
                    progress_callback,
                    "Legacy multi-shot grounding exceeded clasp's internal id limit; "
                    "retrying with optimized direct precompute.",
                )
                return run_once(
                    config_path,
                    mode=mode,
                    grounding_strategy=grounding_strategy,
                    output_dir=output_dir,
                    solutions=solutions,
                    min_length=min_length,
                    max_length=max_length,
                    parallel_mode=parallel_mode,
                    project_models=project_models,
                    graph_format=graph_format,
                    render_graphs=render_graphs,
                    write_raw_answer_sets=write_raw_answer_sets,
                    debug=debug,
                    progress_callback=progress_callback,
                    optimized=True,
                    max_workers=max_workers,
                )
            raise

        if _timed_out:
            _report(progress_callback, "Configured timeout reached; stopping run immediately.")
            combined_peak_rss_mb = _combined_peak_mb(memory_monitor)
            csv_writers.step_writer.log_timeout(
                elapsed_ms=round(max(0.0, perf_counter() - _solve_start) * 1000),
                memory_used_mb=combined_peak_rss_mb,
            )
            csv_writers.summary_writer.log_summary(
                completed_stage="run_timeout",
                timings=TimingBreakdown(
                    translation_sec=translation_sec,
                    grounding_sec=0.0,
                    solving_sec=max(0.0, perf_counter() - _solve_start),
                    rendering_sec=0.0,
                ),
                translation_peak_rss_mb=translation_peak_rss_mb,
                combined_peak_rss_mb=combined_peak_rss_mb,
                base_grounding_sec=0.0,
                base_grounding_peak_rss_mb=0.0,
                horizon_records=(),
                raw_solutions_found=0,
                raw_models_seen=0,
                solutions_found=0,
                effective_parallel_mode=effective_parallel_mode,
            )
            return _timed_out_run_result(
                config=config,
                mode=mode,
                grounding_strategy=grounding_strategy,
                fact_bundle=fact_bundle,
                translation_sec=translation_sec,
                translation_peak_rss_mb=translation_peak_rss_mb,
                combined_peak_rss_mb=combined_peak_rss_mb,
                solve_start=_solve_start,
                completed_stage="run_timeout",
                run_log_path=csv_writers.run_log_path,
                run_summary_path=csv_writers.run_summary_path,
            )

        horizon_records = _records_with_combined_peak_rss(
            solve_output.horizon_records,
            memory_monitor=memory_monitor,
            peaks_by_horizon=horizon_peak_rss_by_horizon,
        )
        if solve_output.base_grounding_sec or solve_output.base_grounding_peak_rss_mb:
            csv_writers.step_writer.log_base_grounding(
                base_grounding_sec=solve_output.base_grounding_sec,
                base_grounding_peak_rss_mb=solve_output.base_grounding_peak_rss_mb,
            )
        for record in horizon_records:
            csv_writers.step_writer.log_horizon(record)

        candidate_solutions = tuple(
            reconstruct_solution(
                index + 1,
                symbols,
                dict(fact_bundle.tool_labels),
                workflow_input_dims=ctx.workflow_input_dims,
                tool_output_dims=ctx.tool_output_dims,
            )
            for index, symbols in enumerate(solve_output.raw_solutions)
        )
        solutions = tuple(
            reconstruct_solution(
                index + 1,
                symbols,
                dict(fact_bundle.tool_labels),
                workflow_input_dims=ctx.workflow_input_dims,
                tool_output_dims=ctx.tool_output_dims,
            )
            for index, symbols in enumerate(solve_output.solutions)
        )
        # ``solutions`` is the canonical stored result surface. ``candidate_solutions``
        # mirrors raw callback order and is only useful when raw diagnostics are
        # requested.

        answer_set_path: Path | None = None
        workflow_signature_path: Path | None = None
        graph_paths: tuple[Path, ...] = ()
        rendering_sec = 0.0

        _report(progress_callback, "Step 4: writing outputs and rendering artifacts...")
        render_start = perf_counter()
        if write_raw_answer_sets:
            # Raw answer sets are a debug artifact. The default machine-readable
            # result surface is the config/mode-specific workflow-signature file below.
            answer_set_path = solution_dir / _answer_sets_filename(
                config=config,
                mode=mode,
                optimized_enabled=optimized,
                effective_parallel_mode=effective_parallel_mode,
            )
            if solve_output.raw_solutions:
                _answer_set_content = "".join(
                    f"Answer Set {index}\n"
                    + " ".join(sorted(str(symbol) for symbol in symbols))
                    + "\n\n"
                    for index, symbols in enumerate(solve_output.raw_solutions, start=1)
                )
                answer_set_path.write_text(_answer_set_content, encoding="utf-8")
            else:
                _answer_set_content = "No answer sets found.\n"
                answer_set_path.write_text(_answer_set_content, encoding="utf-8")
        workflow_signature_path = write_workflow_signatures(
            solution_dir
            / _workflow_signatures_filename(
                config=config,
                mode=mode,
                optimized_enabled=optimized,
                effective_parallel_mode=effective_parallel_mode,
            ),
            solutions,
        )
        graph_path_list: list[Path] = []
        if render_graphs:
            figures_dir = solution_dir / "Figures"
            max_graphs = config.number_of_generated_graphs
            for solution in solutions[:max_graphs]:
                graph_path_list.extend(render_solution_graphs(figures_dir, solution, graph_format))
        graph_paths = tuple(graph_path_list)
        rendering_sec = perf_counter() - render_start
        _report(progress_callback, f"Step 4 complete: output writing/rendering finished after {rendering_sec:.3f}s.")

        combined_peak_rss_mb = _combined_peak_mb(memory_monitor)
        horizon_summary_path = solution_dir / "horizon_summary.json"
        horizon_summary_path.write_text(
            json.dumps(
                {
                    "mode": mode,
                    "grounding_strategy": grounding_strategy,
                    "internal_schema": fact_bundle.internal_schema,
                    "internal_solver_mode": internal_solver_mode,
                    "earliest_solution_step": fact_bundle.earliest_solution_step,
                    "effective_solve_start_horizon": _effective_solve_start_horizon(
                        config=config,
                        fact_bundle=fact_bundle,
                    ),
                    "python_precompute_enabled": fact_bundle.python_precompute_enabled,
                    "python_precompute_fact_count": fact_bundle.python_precompute_fact_count,
                    "python_precompute_stats": dict(sorted(fact_bundle.python_precompute_stats.items())),
                    "backend_stats": fact_bundle.backend_stats,
                    "workflow_input_compression": _workflow_input_compression_payload(
                        config=config,
                        mode=mode,
                        internal_solver_mode=internal_solver_mode,
                        compression_active=effective_project_models,
                    ),
                    "solve_callback_profile": _solve_callback_profile_payload(
                        horizon_records,
                        solving_sec=solve_output.solving_sec,
                    ),
                    "effective_parallel_mode": effective_parallel_mode,
                    "translation_peak_rss_mb": translation_peak_rss_mb,
                    "combined_peak_rss_mb": combined_peak_rss_mb,
                    "base_grounding_peak_rss_mb": solve_output.base_grounding_peak_rss_mb,
                    "base_grounding_sec": solve_output.base_grounding_sec,
                    "timed_out": _timed_out,
                    "horizons": _horizon_record_payload(horizon_records),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        csv_writers.summary_writer.log_summary(
            completed_stage="run",
            timings=TimingBreakdown(
                translation_sec=translation_sec,
                grounding_sec=solve_output.grounding_sec,
                solving_sec=solve_output.solving_sec,
                rendering_sec=rendering_sec,
            ),
            translation_peak_rss_mb=translation_peak_rss_mb,
            combined_peak_rss_mb=combined_peak_rss_mb,
            base_grounding_sec=solve_output.base_grounding_sec,
            base_grounding_peak_rss_mb=solve_output.base_grounding_peak_rss_mb,
            horizon_records=horizon_records,
            raw_solutions_found=len(solve_output.raw_solutions) if diagnostic_counts_enabled else 0,
            raw_models_seen=(
                sum(record.models_seen for record in horizon_records)
                if diagnostic_counts_enabled
                else 0
            ),
            solutions_found=len(solutions),
            grounded_horizons=tuple(record.horizon for record in horizon_records),
            effective_parallel_mode=effective_parallel_mode,
        )
        run_log_path = csv_writers.run_log_path
        run_summary_path = csv_writers.run_summary_path

        return RunResult(
            config=config,
            mode=mode,
            grounding_strategy=grounding_strategy,
            fact_bundle=fact_bundle,
            solutions=solutions,
            timings=TimingBreakdown(
                translation_sec=translation_sec,
                grounding_sec=solve_output.grounding_sec,
                solving_sec=solve_output.solving_sec,
                rendering_sec=rendering_sec,
            ),
            translation_peak_rss_mb=translation_peak_rss_mb,
            combined_peak_rss_mb=combined_peak_rss_mb,
            base_grounding_peak_rss_mb=solve_output.base_grounding_peak_rss_mb,
            base_grounding_sec=solve_output.base_grounding_sec,
            horizon_records=horizon_records,
            translation_path=None,
            answer_set_path=answer_set_path,
            solution_summary_path=None,
            workflow_signature_path=workflow_signature_path,
            graph_paths=graph_paths,
            raw_models_seen=(
                sum(record.models_seen for record in horizon_records)
                if diagnostic_counts_enabled
                else 0
            ),
            raw_answer_sets_found=len(solve_output.raw_solutions) if diagnostic_counts_enabled else 0,
            unique_solutions_found=len(solutions),
            diagnostic_counts_enabled=diagnostic_counts_enabled,
            timed_out=False,
            completed_stage="run",
            run_log_path=run_log_path,
            run_summary_path=run_summary_path,
        )


def run_ground_only(
    config_path: str | Path,
    *,
    mode: str,
    grounding_strategy: str,
    stage: str = "base",
    output_dir: str | Path | None = None,
    solutions: int | None = None,
    min_length: int | None = None,
    max_length: int | None = None,
    progress_callback: ProgressCallback = None,
    optimized: bool = False,
    max_workers: int = 1,
) -> GroundingRunResult:
    """Run translation plus grounding without solving."""

    with ProcessTreePeakMonitor() as memory_monitor:
        ctx = _prepare_run_context(
            config_path,
            mode=mode,
            grounding_strategy=grounding_strategy,
            output_dir=output_dir,
            solutions=solutions,
            min_length=min_length,
            max_length=max_length,
            progress_callback=progress_callback,
            optimized=optimized,
            max_workers=max_workers,
            memory_monitor=memory_monitor,
        )
        config = ctx.config
        solution_dir = ctx.solution_dir
        fact_bundle = ctx.fact_bundle
        translation_sec = ctx.translation_sec
        translation_peak_rss_mb = ctx.translation_peak_combined_rss_mb
        translation_builder = ctx.resolved_translation_builder
        effective_translation_strategy = ctx.effective_translation_strategy
        run_metadata = ctx.run_metadata
        csv_writers = _RunCsvWriters(
            csv_dir=config.solutions_dir_path,
            mode=mode,
            grounding_strategy=grounding_strategy,
            fact_count=fact_bundle.fact_count,
            run_metadata=run_metadata,
            translation_builder=translation_builder,
            translation_schema=_translation_schema(fact_bundle),
            optimized_enabled=optimized,
            effective_parallel_mode=None,
            compressed_candidate_engaged=_compressed_candidate_engaged(fact_bundle),
        )
        csv_writers.step_writer.log_translation(translation_sec=translation_sec)

        translation_summary_path, translation_summary = _write_translation_summary(
            config=config,
            solution_dir=solution_dir,
            mode=mode,
            grounding_strategy=grounding_strategy,
            translation_builder=translation_builder,
            effective_translation_strategy=effective_translation_strategy,
            fact_bundle=fact_bundle,
            translation_path=None,
            translation_sec=translation_sec,
        )

        mode_config = _mode_config(mode)
        if not mode_config.supports_ground_only:
            raise ValueError(f"Ground-only runs do not support mode {mode}.")

        internal_solver_mode = _effective_internal_solver_mode(mode, fact_bundle)
        grounded_horizon_peaks: dict[int, float] = {}

        def _log_ground_horizon(record: HorizonRecord) -> None:
            combined_peak_mb = _combined_peak_mb(memory_monitor)
            grounded_horizon_peaks[record.horizon] = combined_peak_mb
            csv_writers.step_writer.log_horizon(replace(record, peak_rss_mb=combined_peak_mb))

        grounding_output = _GROUNDER_DISPATCH[internal_solver_mode](
            config,
            fact_bundle,
            stage=stage,
            progress_callback=progress_callback,
            base_grounding_callback=lambda sec, peak: csv_writers.step_writer.log_base_grounding(
                base_grounding_sec=sec,
                base_grounding_peak_rss_mb=peak,
            ),
            horizon_record_callback=_log_ground_horizon,
        )

        _report(
            progress_callback,
            f"Step 2 complete: grounding finished after {grounding_output.grounding_sec:.3f}s.",
        )

        horizon_records = _records_with_combined_peak_rss(
            grounding_output.horizon_records,
            memory_monitor=memory_monitor,
            peaks_by_horizon=grounded_horizon_peaks,
        )
        combined_peak_rss_mb = _combined_peak_mb(memory_monitor)
        grounding_summary_path = solution_dir / "grounding_summary.json"
        grounding_summary_path.write_text(
            json.dumps(
                {
                    "mode": mode,
                    "grounding_strategy": grounding_strategy,
                    "stage": stage,
                    "internal_schema": fact_bundle.internal_schema,
                    "internal_solver_mode": internal_solver_mode,
                    "fact_count": fact_bundle.fact_count,
                    "translation_path": None,
                    "translation_summary_path": str(translation_summary_path),
                    "earliest_solution_step": fact_bundle.earliest_solution_step,
                    "python_precompute_enabled": fact_bundle.python_precompute_enabled,
                    "python_precompute_fact_count": fact_bundle.python_precompute_fact_count,
                    "python_precompute_stats": dict(sorted(fact_bundle.python_precompute_stats.items())),
                    "backend_stats": fact_bundle.backend_stats,
                    "workflow_input_compression": _workflow_input_compression_payload(
                        config=config,
                        mode=mode,
                        internal_solver_mode=internal_solver_mode,
                    ),
                    "grounded_horizons": list(grounding_output.grounded_horizons),
                    "translation_peak_rss_mb": translation_peak_rss_mb,
                    "combined_peak_rss_mb": combined_peak_rss_mb,
                    "base_grounding_peak_rss_mb": grounding_output.base_grounding_peak_rss_mb,
                    "base_grounding_sec": grounding_output.base_grounding_sec,
                    "horizon_records": _horizon_record_payload(horizon_records),
                    "translation_sec": translation_sec,
                    "grounding_sec": grounding_output.grounding_sec,
                    "total_sec": translation_sec + grounding_output.grounding_sec,
                    "translation_summary": translation_summary,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        csv_writers.summary_writer.log_summary(
            completed_stage="ground_only_full" if stage == "full" else "ground_only_base",
            timings=TimingBreakdown(
                translation_sec=translation_sec,
                grounding_sec=grounding_output.grounding_sec,
                solving_sec=0.0,
                rendering_sec=0.0,
            ),
            translation_peak_rss_mb=translation_peak_rss_mb,
            combined_peak_rss_mb=combined_peak_rss_mb,
            base_grounding_sec=grounding_output.base_grounding_sec,
            base_grounding_peak_rss_mb=grounding_output.base_grounding_peak_rss_mb,
            horizon_records=horizon_records,
            raw_models_seen=0,
            raw_solutions_found=0,
            solutions_found=0,
            grounded_horizons=grounding_output.grounded_horizons,
            effective_parallel_mode=None,
        )
        run_log_path = csv_writers.run_log_path
        run_summary_path = csv_writers.run_summary_path

        return GroundingRunResult(
            config=config,
            mode=mode,
            grounding_strategy=grounding_strategy,
            stage=stage,
            fact_bundle=fact_bundle,
            timings=TimingBreakdown(
                translation_sec=translation_sec,
                grounding_sec=grounding_output.grounding_sec,
                solving_sec=0.0,
                rendering_sec=0.0,
            ),
            translation_peak_rss_mb=translation_peak_rss_mb,
            combined_peak_rss_mb=combined_peak_rss_mb,
            base_grounding_peak_rss_mb=grounding_output.base_grounding_peak_rss_mb,
            base_grounding_sec=grounding_output.base_grounding_sec,
            horizon_records=horizon_records,
            translation_path=None,
            translation_summary_path=translation_summary_path,
            grounding_summary_path=grounding_summary_path,
            grounded_horizons=grounding_output.grounded_horizons,
            run_log_path=run_log_path,
            run_summary_path=run_summary_path,
        )
