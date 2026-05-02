"""Clingo solver orchestration.

This module maps the public CLI modes onto concrete ASP program families and
then drives grounding/solving in clingo.

- ``multi-shot`` uses the direct incremental encoding family.
- ``multi-shot --optimized`` uses the optimized-candidate incremental family.
- ``single-shot`` currently reuses the direct ``multi_shot`` encoding files but
  grounds them once over ``time(1..max_length)`` and solves on that single
  control object.

Canonical workflow candidates are the default stored result surface. Raw answer
sets remain optional diagnostics for debugging multiplicity and callback cost.
"""

from __future__ import annotations

from time import perf_counter

from clindaws.core.models import FactBundle, HorizonRecord, SnakeConfig
from clindaws.core.ontology import Ontology
from clindaws.core.runtime_stats import current_peak_rss_mb
from clindaws.execution.solver_control import (
    BaseGroundingCallback,
    HorizonRecordCallback,
    ProgressCallback,
    _make_grounding_control,
    _multi_shot_optimized_candidate_program_paths,
    _multi_shot_program_paths,
    _projection_runtime_facts,
    _single_shot_program_paths,
)
from clindaws.execution.solver_multi_shot import (
    _dynamic_grounding_horizon_parts,
    _dynamic_horizon_parts,
    _ground_multi_shot_control,
    _multi_shot_grounding_horizon_parts,
    _multi_shot_horizon_parts,
    _solve_multi_shot_with_programs,
)
from clindaws.execution.solver_single_shot import (
    _single_shot_full_ground_parts,
    _single_shot_overlay,
    _solve_single_shot_once,
    _solve_single_shot_with_programs,
)
from clindaws.execution.solver_solutions import GroundingOutput, SolveOutput
from clindaws.execution.solver_utils import (
    _collect_direct_multishot_metrics,
    _interrupt_guard,
    _report,
    _run_interruptible,
)


def solve_single_shot(
    config: SnakeConfig,
    facts: FactBundle,
    *,
    ontology: Ontology | None = None,
    workflow_input_dims: dict[str, dict[str, tuple[str, ...]]] | None = None,
    tool_output_dims: dict[tuple[str, int], dict[str, tuple[str, ...]]] | None = None,
    progress_callback: ProgressCallback = None,
    base_grounding_callback: BaseGroundingCallback = None,
    horizon_record_callback: HorizonRecordCallback = None,
    capture_raw_models: bool = False,
    diagnostic_counts_enabled: bool = True,
    parallel_mode: str | None = None,
    project_models: bool = False,
) -> SolveOutput:
    """Solve single-shot mode as one grounding over time(1..max_length)."""

    return _solve_single_shot_once(
        config,
        facts,
        _single_shot_program_paths(optimized=facts.python_precompute_enabled),
        ontology=ontology,
        workflow_input_dims=workflow_input_dims,
        tool_output_dims=tool_output_dims,
        progress_callback=progress_callback,
        base_grounding_callback=base_grounding_callback,
        horizon_record_callback=horizon_record_callback,
        capture_raw_models=capture_raw_models,
        diagnostic_counts_enabled=diagnostic_counts_enabled,
        parallel_mode=parallel_mode,
        project_models=project_models,
    )


def solve_single_shot_sliding_window(
    config: SnakeConfig,
    facts: FactBundle,
    *,
    ontology: Ontology | None = None,
    workflow_input_dims: dict[str, dict[str, tuple[str, ...]]] | None = None,
    tool_output_dims: dict[tuple[str, int], dict[str, tuple[str, ...]]] | None = None,
    progress_callback: ProgressCallback = None,
    base_grounding_callback: BaseGroundingCallback = None,
    horizon_record_callback: HorizonRecordCallback = None,
    capture_raw_models: bool = False,
    diagnostic_counts_enabled: bool = True,
    parallel_mode: str | None = None,
    project_models: bool = False,
) -> SolveOutput:
    """Solve single-shot mode by traversing the configured horizon range."""

    return _solve_single_shot_with_programs(
        config,
        facts,
        _single_shot_program_paths(optimized=facts.python_precompute_enabled),
        ontology=ontology,
        workflow_input_dims=workflow_input_dims,
        tool_output_dims=tool_output_dims,
        progress_callback=progress_callback,
        base_grounding_callback=base_grounding_callback,
        horizon_record_callback=horizon_record_callback,
        solve_all_horizons=False,
        capture_raw_models=capture_raw_models,
        diagnostic_counts_enabled=diagnostic_counts_enabled,
        parallel_mode=parallel_mode,
        project_models=project_models,
    )


def ground_single_shot(
    config: SnakeConfig,
    facts: FactBundle,
    *,
    stage: str = "base",
    progress_callback: ProgressCallback = None,
    base_grounding_callback: BaseGroundingCallback = None,
    horizon_record_callback: HorizonRecordCallback = None,
) -> GroundingOutput:
    """Ground the one-shot single-shot backend without solving."""

    control = _make_grounding_control()
    for program_path in _single_shot_program_paths(optimized=facts.python_precompute_enabled):
        control.load(str(program_path))
    control.add("base", [], facts.facts)
    runtime_facts = _projection_runtime_facts(mode="single-shot", project_models=False)
    if runtime_facts:
        control.add("base", [], runtime_facts)
    if facts.python_precomputed_facts:
        control.add("base", [], facts.python_precomputed_facts)
    if stage == "full":
        control.add(
            "single_shot",
            [],
            _single_shot_overlay(config.solution_length_min, config.solution_length_max),
        )

    total_grounding = 0.0
    base_grounding_peak_rss_mb = 0.0
    base_grounding_sec = 0.0
    grounded_horizons: list[int] = []
    horizon_records: list[HorizonRecord] = []

    try:
        with _interrupt_guard(control) as is_interrupted:
            _report(progress_callback, "Step 2: grounding started.")
            _report(progress_callback, "Grounding: base program...")
            start = perf_counter()
            _run_interruptible(lambda: control.ground([("base", [])]), is_interrupted)
            elapsed = perf_counter() - start
            total_grounding += elapsed
            base_grounding_sec = elapsed
            base_grounding_peak_rss_mb = current_peak_rss_mb()
            if base_grounding_callback is not None:
                base_grounding_callback(base_grounding_sec, base_grounding_peak_rss_mb)
            _report(progress_callback, f"Grounding progress: base program finished after {elapsed:.3f}s.")

            if stage == "full":
                horizon = config.solution_length_max
                grounding_parts: list[tuple[str, float]] = []
                full_ground_parts = _single_shot_full_ground_parts(horizon)
                _report(progress_callback, f"Grounding: single-shot full program (1..{horizon})...")
                start = perf_counter()
                _run_interruptible(
                    lambda: control.ground(
                        [(name, list(args)) for name, args in full_ground_parts]
                    ),
                    is_interrupted,
                )
                full_ground_elapsed = perf_counter() - start
                total_grounding += full_ground_elapsed
                grounding_parts.append(("single_shot_full", full_ground_elapsed))
                _report(
                    progress_callback,
                    f"Grounding progress: single-shot full program finished after {full_ground_elapsed:.3f}s.",
                )
                grounded_horizons.append(horizon)
                record = HorizonRecord(
                    horizon=horizon,
                    grounding_sec=total_grounding - base_grounding_sec,
                    solving_sec=0.0,
                    peak_rss_mb=current_peak_rss_mb(),
                    satisfiable=False,
                    models_seen=0,
                    models_stored=0,
                    unique_workflows_seen=0,
                    unique_workflows_stored=0,
                    grounding_parts=tuple(grounding_parts),
                )
                horizon_records.append(record)
                if horizon_record_callback is not None:
                    horizon_record_callback(record)
                _report(
                    progress_callback,
                    f"Grounding progress: single-shot finished after {total_grounding:.3f}s.",
                )
            elif stage != "base":
                raise ValueError(f"Unsupported ground-only stage: {stage}")
    finally:
        control.cleanup()

    return GroundingOutput(
        base_grounding_peak_rss_mb=base_grounding_peak_rss_mb,
        base_grounding_sec=base_grounding_sec,
        grounding_sec=total_grounding,
        grounded_horizons=tuple(grounded_horizons),
        horizon_records=tuple(horizon_records),
    )


def solve_multi_shot(
    config: SnakeConfig,
    facts: FactBundle,
    *,
    ontology: Ontology | None = None,
    workflow_input_dims: dict[str, dict[str, tuple[str, ...]]] | None = None,
    tool_output_dims: dict[tuple[str, int], dict[str, tuple[str, ...]]] | None = None,
    progress_callback: ProgressCallback = None,
    base_grounding_callback: BaseGroundingCallback = None,
    horizon_record_callback: HorizonRecordCallback = None,
    capture_raw_models: bool = False,
    diagnostic_counts_enabled: bool = True,
    parallel_mode: str | None = None,
    project_models: bool = False,
) -> SolveOutput:
    """Solve using the multi-shot encoding."""
    return _solve_multi_shot_with_programs(
        config,
        facts,
        _multi_shot_program_paths(),
        mode="multi-shot",
        ontology=ontology,
        workflow_input_dims=workflow_input_dims,
        tool_output_dims=tool_output_dims,
        progress_callback=progress_callback,
        base_grounding_callback=base_grounding_callback,
        horizon_record_callback=horizon_record_callback,
        solve_all_horizons=False,
        stop_on_solution=False,
        horizon_parts_builder=_multi_shot_horizon_parts,
        capture_raw_models=capture_raw_models,
        diagnostic_counts_enabled=diagnostic_counts_enabled,
        parallel_mode=parallel_mode,
        project_models=project_models,
    )


def ground_multi_shot(
    config: SnakeConfig,
    facts: FactBundle,
    *,
    stage: str = "base",
    progress_callback: ProgressCallback = None,
    base_grounding_callback: BaseGroundingCallback = None,
    horizon_record_callback: HorizonRecordCallback = None,
) -> GroundingOutput:
    """Ground the multi-shot encoding without solving."""

    control = _make_grounding_control()
    for program_path in _multi_shot_program_paths():
        control.load(str(program_path))
    control.add("base", [], facts.facts)
    if facts.python_precomputed_facts:
        control.add("base", [], facts.python_precomputed_facts)
    return _ground_multi_shot_control(
        control,
        config,
        stage=stage,
        collect_horizon_metrics=_collect_direct_multishot_metrics(facts),
        progress_callback=progress_callback,
        base_grounding_callback=base_grounding_callback,
        horizon_record_callback=horizon_record_callback,
        horizon_parts_builder=_multi_shot_grounding_horizon_parts,
    )


def solve_multi_shot_optimized_candidate(
    config: SnakeConfig,
    facts: FactBundle,
    *,
    ontology: Ontology | None = None,
    workflow_input_dims: dict[str, dict[str, tuple[str, ...]]] | None = None,
    tool_output_dims: dict[tuple[str, int], dict[str, tuple[str, ...]]] | None = None,
    progress_callback: ProgressCallback = None,
    base_grounding_callback: BaseGroundingCallback = None,
    horizon_record_callback: HorizonRecordCallback = None,
    capture_raw_models: bool = False,
    diagnostic_counts_enabled: bool = True,
    parallel_mode: str | None = None,
    project_models: bool = False,
) -> SolveOutput:
    """Solve using the optimized-candidate multi-shot encoding."""

    return _solve_multi_shot_with_programs(
        config,
        facts,
        _multi_shot_optimized_candidate_program_paths(),
        mode="multi-shot-optimized-candidate",
        ontology=ontology,
        workflow_input_dims=workflow_input_dims,
        tool_output_dims=tool_output_dims,
        progress_callback=progress_callback,
        base_grounding_callback=base_grounding_callback,
        horizon_record_callback=horizon_record_callback,
        initial_step_program="step_initial",
        solve_all_horizons=False,
        stop_on_solution=False,
        horizon_parts_builder=lambda horizon: _dynamic_horizon_parts(
            horizon,
            initial_step_program="step_initial",
            initial_seed_program=None,
        ),
        capture_raw_models=capture_raw_models,
        diagnostic_counts_enabled=diagnostic_counts_enabled,
        solve_start_horizon=facts.earliest_solution_step,
        parallel_mode=parallel_mode,
        project_models=project_models,
    )


def solve_multi_shot_compressed_candidate(
    config: SnakeConfig,
    facts: FactBundle,
    *,
    ontology: Ontology | None = None,
    workflow_input_dims: dict[str, dict[str, tuple[str, ...]]] | None = None,
    tool_output_dims: dict[tuple[str, int], dict[str, tuple[str, ...]]] | None = None,
    progress_callback: ProgressCallback = None,
    base_grounding_callback: BaseGroundingCallback = None,
    horizon_record_callback: HorizonRecordCallback = None,
    capture_raw_models: bool = False,
    diagnostic_counts_enabled: bool = True,
    parallel_mode: str | None = None,
    project_models: bool = False,
) -> SolveOutput:
    """Compatibility wrapper for the legacy optimized backend entrypoint."""

    return solve_multi_shot_optimized_candidate(
        config,
        facts,
        ontology=ontology,
        workflow_input_dims=workflow_input_dims,
        tool_output_dims=tool_output_dims,
        progress_callback=progress_callback,
        base_grounding_callback=base_grounding_callback,
        horizon_record_callback=horizon_record_callback,
        capture_raw_models=capture_raw_models,
        diagnostic_counts_enabled=diagnostic_counts_enabled,
        parallel_mode=parallel_mode,
        project_models=project_models,
    )


def ground_multi_shot_optimized_candidate(
    config: SnakeConfig,
    facts: FactBundle,
    *,
    stage: str = "base",
    progress_callback: ProgressCallback = None,
    base_grounding_callback: BaseGroundingCallback = None,
    horizon_record_callback: HorizonRecordCallback = None,
) -> GroundingOutput:
    """Ground the optimized-candidate multi-shot encoding without solving."""

    control = _make_grounding_control()
    for program_path in _multi_shot_optimized_candidate_program_paths():
        control.load(str(program_path))
    control.add("base", [], facts.facts)
    return _ground_multi_shot_control(
        control,
        config,
        stage=stage,
        progress_callback=progress_callback,
        base_grounding_callback=base_grounding_callback,
        horizon_record_callback=horizon_record_callback,
        initial_step_program="step_initial",
        horizon_parts_builder=lambda horizon: _dynamic_grounding_horizon_parts(
            horizon,
            initial_step_program="step_initial",
            initial_seed_program=None,
        ),
    )


def ground_multi_shot_compressed_candidate(
    config: SnakeConfig,
    facts: FactBundle,
    *,
    stage: str = "base",
    progress_callback: ProgressCallback = None,
    base_grounding_callback: BaseGroundingCallback = None,
    horizon_record_callback: HorizonRecordCallback = None,
) -> GroundingOutput:
    """Compatibility wrapper for the legacy optimized grounding entrypoint."""

    return ground_multi_shot_optimized_candidate(
        config,
        facts,
        stage=stage,
        progress_callback=progress_callback,
        base_grounding_callback=base_grounding_callback,
        horizon_record_callback=horizon_record_callback,
    )
