"""Multi-shot solver internals: incremental grounding plus per-horizon solve loop."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter, perf_counter_ns
from typing import Callable

import clingo

from clindaws.core.models import FactBundle, HorizonRecord, SnakeConfig
from clindaws.core.ontology import Ontology
from clindaws.core.runtime_stats import current_peak_rss_mb
from clindaws.core.workflow import (
    canonicalize_shown_symbols,
    extract_canonical_workflow_keys,
)
from clindaws.execution.solver_control import (
    BaseGroundingCallback,
    HorizonRecordCallback,
    ProgressCallback,
    _load_control_programs,
    _make_solve_control,
)
from clindaws.execution.solver_optimized_candidate import (
    _add_exact_model_blocking_clause,
    _exact_candidate_sequence_symbols,
    _optimized_certificate_horizon_parts,
    _optimized_exact_incremental_horizon_parts,
    _optimized_full_solve_horizon_parts,
    _optimized_model_blocking_mode,
    _optimized_query_assumptions,
    _optimized_structural_probe_horizons,
    _run_feasibility_precheck,
    _run_optimized_exact_certificate,
    _smart_expansion_enabled,
)
from clindaws.execution.solver_solutions import (
    GroundingOutput,
    SolveOutput,
    _SolvePassMetrics,
    _stored_solution_quota_reached,
    _stored_workflow_key,
)
from clindaws.execution.solver_utils import (
    _collect_direct_multishot_metrics,
    _format_progress_counts,
    _ground_program_parts,
    _interrupt_guard,
    _legacy_direct_multishot_horizon_metrics,
    _report,
    _run_interruptible,
)


def _default_horizon_parts(
    horizon: int,
    *,
    initial_step_program: str | None,
    initial_seed_program: str | None,
    grounding_only: bool = False,
) -> tuple[tuple[str, tuple[clingo.Symbol, ...]], ...]:
    parts: list[tuple[str, tuple[clingo.Symbol, ...]]] = []
    if initial_step_program is not None and horizon == 1:
        parts.append((initial_step_program, (clingo.Number(horizon),)))
    else:
        if initial_seed_program is not None and horizon > 1:
            parts.append((initial_seed_program, (clingo.Number(horizon - 1),)))
        parts.append(("step", (clingo.Number(horizon),)))
    if not grounding_only:
        # constraint_step / check are only useful when query(t) will be assigned
        # True during solving; skip them for grounding-only runs.
        parts.append(("constraint_step", (clingo.Number(horizon),)))
        parts.append(("check", (clingo.Number(horizon),)))
    return tuple(parts)


def _dynamic_horizon_parts(
    horizon: int,
    *,
    initial_step_program: str | None,
    initial_seed_program: str | None,
) -> tuple[tuple[str, tuple[clingo.Symbol, ...]], ...]:
    parts: list[tuple[str, tuple[clingo.Symbol, ...]]] = []
    if initial_step_program is not None and horizon == 1:
        parts.append((initial_step_program, (clingo.Number(horizon),)))
    else:
        if initial_seed_program is not None and horizon > 1:
            parts.append((initial_seed_program, (clingo.Number(horizon - 1),)))
        parts.append(("step", (clingo.Number(horizon),)))
        parts.append(("step_query", (clingo.Number(horizon),)))
    parts.append(("possible", (clingo.Number(horizon),)))
    parts.append(("constraint_step", (clingo.Number(horizon),)))
    parts.append(("check", (clingo.Number(horizon),)))
    parts.append(("check_usage", (clingo.Number(horizon),)))
    return tuple(parts)


def _multi_shot_horizon_parts(horizon: int) -> tuple[tuple[str, tuple[clingo.Symbol, ...]], ...]:
    parts: list[tuple[str, tuple[clingo.Symbol, ...]]] = []
    if horizon == 1:
        parts.append(("init", ()))
    parts.append(("step", (clingo.Number(horizon),)))
    parts.append(("constraint_step", (clingo.Number(horizon),)))
    parts.append(("check", (clingo.Number(horizon),)))
    return tuple(parts)


def _multi_shot_grounding_horizon_parts(
    horizon: int,
) -> tuple[tuple[str, tuple[clingo.Symbol, ...]], ...]:
    parts: list[tuple[str, tuple[clingo.Symbol, ...]]] = []
    if horizon == 1:
        parts.append(("init", ()))
    parts.append(("step", (clingo.Number(horizon),)))
    return tuple(parts)


def _dynamic_grounding_horizon_parts(
    horizon: int,
    *,
    initial_step_program: str | None,
    initial_seed_program: str | None,
) -> tuple[tuple[str, tuple[clingo.Symbol, ...]], ...]:
    """Lean horizon parts for grounding-only dynamic runs.

    step_query / constraint_step / check / check_usage are all guarded by
    query(t), which is never assigned during grounding-only runs, so they
    produce no useful atoms and are omitted entirely.
    """
    parts: list[tuple[str, tuple[clingo.Symbol, ...]]] = []
    if initial_step_program is not None and horizon == 1:
        parts.append((initial_step_program, (clingo.Number(horizon),)))
    else:
        if initial_seed_program is not None and horizon > 1:
            parts.append((initial_seed_program, (clingo.Number(horizon - 1),)))
        parts.append(("step", (clingo.Number(horizon),)))
    return tuple(parts)


def _solve_on_control(
    control: clingo.Control,
    *,
    config: SnakeConfig,
    ontology: Ontology | None = None,
    tool_input_signatures: dict[str, tuple[tuple[tuple[str, tuple[str, ...]], ...], ...]],
    workflow_input_dims: dict[str, dict[str, tuple[str, ...]]] | None = None,
    tool_output_dims: dict[tuple[str, int], dict[str, tuple[str, ...]]] | None = None,
    horizon: int,
    assumptions: list[tuple[clingo.Symbol, bool]] | None,
    clause_blocking_mode: str | None,
    capture_raw_models: bool,
    diagnostic_counts_enabled: bool,
    solve_all_horizons: bool,
    raw_collected: list[tuple[clingo.Symbol, ...]],
    unique_collected: list[tuple[clingo.Symbol, ...]],
    stored_unique_keys: set[tuple[object, ...]],
    progress_callback: ProgressCallback,
    is_interrupted: Callable[[], bool],
) -> _SolvePassMetrics:
    assumptions = list(assumptions or ())
    any_model_seen = False
    models_seen = 0
    models_stored = 0
    unique_workflows_seen = 0
    unique_workflows_stored = 0
    model_callback_sec = 0.0
    shown_symbols_sec = 0.0
    workflow_signature_key_sec = 0.0
    canonicalization_sec = 0.0
    clause_constraints_added = 0
    seen_unique_keys: set[tuple[object, ...]] = set()
    seen_tool_sequence_keys: set[tuple[object, ...]] = set()
    stored_tool_sequence_keys: set[tuple[object, ...]] = set()

    start = perf_counter()

    def _solve() -> None:
        nonlocal any_model_seen, models_seen, models_stored
        nonlocal unique_workflows_seen, unique_workflows_stored
        nonlocal model_callback_sec, shown_symbols_sec
        nonlocal workflow_signature_key_sec, canonicalization_sec
        nonlocal clause_constraints_added

        if clause_blocking_mode == "candidate_sequence_clause":
            while True:
                model_found = False
                blocking_symbols: tuple[clingo.Symbol, ...] = ()
                with control.solve(yield_=True, assumptions=assumptions) as handle:
                    for model in handle:
                        model_found = True
                        callback_start = perf_counter()
                        any_model_seen = True
                        if diagnostic_counts_enabled:
                            models_seen += 1
                        shown_start = perf_counter()
                        shown_symbols = tuple(model.symbols(shown=True))
                        shown_symbols_sec += perf_counter() - shown_start
                        key_start = perf_counter()
                        tool_sequence_key, workflow_key = extract_canonical_workflow_keys(
                            shown_symbols,
                            tool_input_signatures,
                            workflow_input_dims,
                            tool_output_dims,
                            ontology,
                            use_binding_target_abstraction=(
                                not config.tool_seq_repeat
                            ),
                        )
                        workflow_signature_key_sec += perf_counter() - key_start
                        workflow_storage_key = _stored_workflow_key(
                            config=config,
                            tool_sequence_key=tool_sequence_key,
                            workflow_key=workflow_key,
                        )
                        in_length_window = (
                            config.solution_length_min
                            <= horizon
                            <= config.solution_length_max
                        )
                        if diagnostic_counts_enabled:
                            if tool_sequence_key not in seen_tool_sequence_keys:
                                seen_tool_sequence_keys.add(tool_sequence_key)
                            if workflow_storage_key not in seen_unique_keys:
                                seen_unique_keys.add(workflow_storage_key)
                                unique_workflows_seen += 1
                        store_raw = (
                            in_length_window
                            and capture_raw_models
                            and len(raw_collected) < config.solutions
                        )
                        store_unique = (
                            in_length_window
                            and workflow_storage_key not in stored_unique_keys
                            and len(unique_collected) < config.solutions
                        )
                        if store_raw:
                            raw_collected.append(shown_symbols)
                            models_stored += 1
                        if store_unique:
                            canonical_start = perf_counter()
                            canonical_shown = canonicalize_shown_symbols(
                                shown_symbols,
                                tool_input_signatures,
                                workflow_input_dims,
                                tool_output_dims,
                            )
                            canonicalization_sec += perf_counter() - canonical_start
                            stored_unique_keys.add(workflow_storage_key)
                            stored_tool_sequence_keys.add(tool_sequence_key)
                            unique_collected.append(canonical_shown)
                            unique_workflows_stored += 1
                        blocking_symbols = _exact_candidate_sequence_symbols(model)
                        model_callback_sec += perf_counter() - callback_start
                        break
                if not model_found:
                    break
                if blocking_symbols and _add_exact_model_blocking_clause(control, symbols=blocking_symbols):
                    clause_constraints_added += 1
                else:
                    break
                if not solve_all_horizons and len(unique_collected) >= config.solutions:
                    break
                if solve_all_horizons and len(unique_collected) >= config.solutions:
                    break
        else:
            while True:
                with control.solve(yield_=True, assumptions=assumptions) as handle:
                    for model in handle:
                        callback_start = perf_counter()
                        any_model_seen = True
                        if diagnostic_counts_enabled:
                            models_seen += 1
                        shown_start = perf_counter()
                        shown_symbols = tuple(model.symbols(shown=True))
                        shown_symbols_sec += perf_counter() - shown_start
                        key_start = perf_counter()
                        tool_sequence_key, workflow_key = extract_canonical_workflow_keys(
                            shown_symbols,
                            tool_input_signatures,
                            workflow_input_dims,
                            tool_output_dims,
                            ontology,
                            use_binding_target_abstraction=(
                                not config.tool_seq_repeat
                            ),
                        )
                        workflow_signature_key_sec += perf_counter() - key_start
                        workflow_storage_key = _stored_workflow_key(
                            config=config,
                            tool_sequence_key=tool_sequence_key,
                            workflow_key=workflow_key,
                        )
                        in_length_window = (
                            config.solution_length_min
                            <= horizon
                            <= config.solution_length_max
                        )
                        if diagnostic_counts_enabled:
                            if tool_sequence_key not in seen_tool_sequence_keys:
                                seen_tool_sequence_keys.add(tool_sequence_key)
                            if workflow_storage_key not in seen_unique_keys:
                                seen_unique_keys.add(workflow_storage_key)
                                unique_workflows_seen += 1
                        store_raw = (
                            in_length_window
                            and capture_raw_models
                            and len(raw_collected) < config.solutions
                        )
                        store_unique = (
                            in_length_window
                            and workflow_storage_key not in stored_unique_keys
                            and len(unique_collected) < config.solutions
                        )
                        if store_raw:
                            raw_collected.append(shown_symbols)
                            models_stored += 1
                        if store_unique:
                            canonical_start = perf_counter()
                            canonical_shown = canonicalize_shown_symbols(
                                shown_symbols,
                                tool_input_signatures,
                                workflow_input_dims,
                                tool_output_dims,
                            )
                            canonicalization_sec += perf_counter() - canonical_start
                            stored_unique_keys.add(workflow_storage_key)
                            stored_tool_sequence_keys.add(tool_sequence_key)
                            unique_collected.append(canonical_shown)
                            unique_workflows_stored += 1
                        elif solve_all_horizons and len(unique_collected) >= config.solutions:
                            model_callback_sec += perf_counter() - callback_start
                            break
                        if not solve_all_horizons and len(unique_collected) >= config.solutions:
                            model_callback_sec += perf_counter() - callback_start
                            break
                        model_callback_sec += perf_counter() - callback_start
                    break

    _run_interruptible(_solve, is_interrupted)
    solve_elapsed = perf_counter() - start
    return _SolvePassMetrics(
        any_model_seen=any_model_seen,
        models_seen=models_seen,
        models_stored=models_stored,
        unique_workflows_seen=unique_workflows_seen,
        unique_workflows_stored=unique_workflows_stored,
        model_callback_sec=model_callback_sec,
        shown_symbols_sec=shown_symbols_sec,
        workflow_signature_key_sec=workflow_signature_key_sec,
        canonicalization_sec=canonicalization_sec,
        clause_constraints_added=clause_constraints_added,
        seen_tool_sequence_count=len(seen_tool_sequence_keys),
        stored_tool_sequence_count=len(stored_tool_sequence_keys),
        solve_elapsed=solve_elapsed,
    )


def _solve_multi_shot_with_programs(
    config: SnakeConfig,
    facts: FactBundle,
    program_paths: tuple[Path, ...],
    mode: str,
    *,
    ontology: Ontology | None = None,
    workflow_input_dims: dict[str, dict[str, tuple[str, ...]]] | None = None,
    tool_output_dims: dict[tuple[str, int], dict[str, tuple[str, ...]]] | None = None,
    progress_callback: ProgressCallback = None,
    base_grounding_callback: BaseGroundingCallback = None,
    horizon_record_callback: HorizonRecordCallback = None,
    initial_step_program: str | None = None,
    initial_seed_program: str | None = None,
    solve_all_horizons: bool = False,
    stop_on_solution: bool = True,
    horizon_parts_builder: Callable[[int], tuple[tuple[str, tuple[clingo.Symbol, ...]], ...]] | None = None,
    capture_raw_models: bool = False,
    diagnostic_counts_enabled: bool = True,
    solve_start_horizon: int | None = None,
    parallel_mode: str | None = None,
    project_models: bool = False,
) -> SolveOutput:
    control = _make_solve_control(
        parallel_mode=parallel_mode,
        project_models=project_models,
    )
    _load_control_programs(
        control,
        program_paths=program_paths,
        facts=facts,
        mode=mode,
        project_models=project_models,
    )
    total_grounding = 0.0
    total_solving = 0.0
    base_grounding_peak_rss_mb = 0.0
    base_grounding_sec = 0.0
    raw_collected: list[tuple[clingo.Symbol, ...]] = []
    unique_collected: list[tuple[clingo.Symbol, ...]] = []
    stored_unique_keys: set[tuple[object, ...]] = set()
    horizon_records: list[HorizonRecord] = []
    solving_started = False
    tool_input_signatures = dict(facts.tool_input_signatures)
    effective_solve_start = solve_start_horizon if solve_start_horizon is not None else config.solution_length_min
    collect_horizon_metrics = _collect_direct_multishot_metrics(facts)
    structural_probe_horizons = set(_optimized_structural_probe_horizons(facts))

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

            horizon = 1
            while horizon <= config.solution_length_max:
                if not solve_all_horizons and _stored_solution_quota_reached(
                    unique_count=len(unique_collected),
                    solution_limit=config.solutions,
                ):
                    break

                _report(progress_callback, f"Grounding: horizon {horizon}...")
                optimized_two_phase = _smart_expansion_enabled(facts)
                if optimized_two_phase:
                    feasibility_ground_parts = _optimized_exact_incremental_horizon_parts(
                        horizon,
                        initial_step_program=initial_step_program,
                        initial_seed_program=initial_seed_program,
                    )
                    full_ground_parts = _optimized_full_solve_horizon_parts(horizon)
                elif horizon_parts_builder is None:
                    feasibility_ground_parts = _default_horizon_parts(
                        horizon,
                        initial_step_program=initial_step_program,
                        initial_seed_program=initial_seed_program,
                    )
                    full_ground_parts = ()
                else:
                    feasibility_ground_parts = horizon_parts_builder(horizon)
                    full_ground_parts = ()
                feasibility_ground_elapsed, feasibility_grounding_parts = _ground_program_parts(
                    control,
                    feasibility_ground_parts,
                    is_interrupted=is_interrupted,
                    progress_callback=progress_callback,
                    horizon=horizon,
                )
                ground_elapsed = feasibility_ground_elapsed
                grounding_parts = list(feasibility_grounding_parts)
                horizon_metrics = (
                    _legacy_direct_multishot_horizon_metrics(control, horizon=horizon)
                    if collect_horizon_metrics
                    else {}
                )
                total_grounding += ground_elapsed
                if optimized_two_phase:
                    _report(
                        progress_callback,
                        f"Grounding progress: horizon {horizon} feasibility parts finished after "
                        f"{ground_elapsed:.3f}s.",
                    )
                else:
                    _report(
                        progress_callback,
                        f"Grounding progress: horizon {horizon} finished after {ground_elapsed:.3f}s.",
                    )

                if _smart_expansion_enabled(facts) and structural_probe_horizons and horizon >= effective_solve_start and horizon not in structural_probe_horizons:
                    record = HorizonRecord(
                        horizon=horizon,
                        grounding_sec=ground_elapsed,
                        solving_sec=0.0,
                        peak_rss_mb=current_peak_rss_mb(),
                        satisfiable=False,
                        models_seen=0,
                        models_stored=0,
                        unique_workflows_seen=0,
                        unique_workflows_stored=0,
                        diagnostic_counts_enabled=diagnostic_counts_enabled,
                        available_artifacts_at_step=horizon_metrics.get("available_artifacts_at_step"),
                        eligible_artifacts_at_step=horizon_metrics.get("eligible_artifacts_at_step"),
                        eligible_workflow_inputs_at_step=horizon_metrics.get("eligible_workflow_inputs_at_step"),
                        eligible_produced_outputs_at_step=horizon_metrics.get("eligible_produced_outputs_at_step"),
                        bind_choice_domain_size_at_step=horizon_metrics.get("bind_choice_domain_size_at_step"),
                        feasibility_checked=False,
                        feasibility_possible=None,
                        feasibility_sec=None,
                        feasibility_grounding_sec=feasibility_ground_elapsed,
                        certificate_grounding_sec=0.0,
                        certificate_solving_sec=None,
                        full_grounding_sec=0.0,
                        full_solve_performed=False,
                        structural_skip_only=True,
                        solve_skipped_reason="structural_goal_window",
                        grounding_parts=tuple(grounding_parts),
                    )
                    horizon_records.append(record)
                    if horizon_record_callback is not None:
                        horizon_record_callback(record)
                    if progress_callback is not None:
                        progress_callback(
                            {
                                "event": "horizon_complete",
                                "horizon": horizon,
                                "timestamp_ns": perf_counter_ns(),
                            }
                        )
                    _report(
                        progress_callback,
                        f"Solving skipped: horizon {horizon} is outside the current structural probe horizons.",
                    )
                    horizon += 1
                    continue

                if horizon < effective_solve_start:
                    record = HorizonRecord(
                        horizon=horizon,
                        grounding_sec=ground_elapsed,
                        solving_sec=0.0,
                        peak_rss_mb=current_peak_rss_mb(),
                        satisfiable=False,
                        models_seen=0,
                        models_stored=0,
                        unique_workflows_seen=0,
                        unique_workflows_stored=0,
                        diagnostic_counts_enabled=diagnostic_counts_enabled,
                        available_artifacts_at_step=horizon_metrics.get("available_artifacts_at_step"),
                        eligible_artifacts_at_step=horizon_metrics.get("eligible_artifacts_at_step"),
                        eligible_workflow_inputs_at_step=horizon_metrics.get("eligible_workflow_inputs_at_step"),
                        eligible_produced_outputs_at_step=horizon_metrics.get("eligible_produced_outputs_at_step"),
                        bind_choice_domain_size_at_step=horizon_metrics.get("bind_choice_domain_size_at_step"),
                        feasibility_checked=False,
                        feasibility_possible=None,
                        feasibility_sec=None,
                        feasibility_grounding_sec=feasibility_ground_elapsed,
                        certificate_grounding_sec=0.0,
                        certificate_solving_sec=None,
                        full_grounding_sec=0.0,
                        full_solve_performed=False,
                        structural_skip_only=True,
                        solve_skipped_reason="structural_lower_bound",
                        grounding_parts=tuple(grounding_parts),
                    )
                    horizon_records.append(record)
                    if horizon_record_callback is not None:
                        horizon_record_callback(record)
                    if progress_callback is not None:
                        progress_callback(
                            {
                                "event": "horizon_complete",
                                "horizon": horizon,
                                "timestamp_ns": perf_counter_ns(),
                            }
                        )
                    _report(
                        progress_callback,
                        f"Solving skipped: horizon {horizon} is below earliest feasible horizon {effective_solve_start}.",
                    )
                    horizon += 1
                    continue

                feasibility_checked = False
                feasibility_possible: bool | None = None
                feasibility_sec: float | None = None
                feasibility_stage_timings: tuple[tuple[str, float], ...] = ()
                feasibility_failure_category: str | None = None
                feasibility_failure_details: tuple[str, ...] = ()
                full_ground_elapsed = 0.0
                full_solve_performed = True
                solve_skipped_reason: str | None = None
                solve_elapsed = 0.0
                if optimized_two_phase:
                    feasibility_checked = True
                    _report(progress_callback, f"Feasibility: horizon {horizon}...")
                    (
                        feasibility_possible,
                        feasibility_sec,
                        feasibility_stage_timings,
                        feasibility_failure_category,
                        feasibility_failure_details,
                    ) = _run_feasibility_precheck(
                        facts=facts,
                        horizon=horizon,
                    )
                    _report(
                        progress_callback,
                        f"Feasibility progress: horizon {horizon} finished after {feasibility_sec:.3f}s, "
                        f"possible={'yes' if feasibility_possible else 'no'}.",
                    )
                    if not feasibility_possible:
                        full_solve_performed = False
                        solve_skipped_reason = "feasibility_precheck"
                        reason_suffix = (
                            f", category={feasibility_failure_category}"
                            if feasibility_failure_category is not None
                            else ""
                        )
                        detail_suffix = (
                            f", details={'; '.join(feasibility_failure_details[:3])}"
                            if feasibility_failure_details
                            else ""
                        )
                        _report(
                            progress_callback,
                            f"Feasibility skip: horizon {horizon} failed{reason_suffix}{detail_suffix}.",
                        )
                        record = HorizonRecord(
                            horizon=horizon,
                            grounding_sec=ground_elapsed,
                            solving_sec=0.0,
                            peak_rss_mb=current_peak_rss_mb(),
                            satisfiable=False,
                            models_seen=0,
                            models_stored=0,
                            unique_workflows_seen=0,
                            unique_workflows_stored=0,
                            diagnostic_counts_enabled=diagnostic_counts_enabled,
                            available_artifacts_at_step=horizon_metrics.get("available_artifacts_at_step"),
                            eligible_artifacts_at_step=horizon_metrics.get("eligible_artifacts_at_step"),
                            eligible_workflow_inputs_at_step=horizon_metrics.get("eligible_workflow_inputs_at_step"),
                            eligible_produced_outputs_at_step=horizon_metrics.get("eligible_produced_outputs_at_step"),
                            bind_choice_domain_size_at_step=horizon_metrics.get("bind_choice_domain_size_at_step"),
                            feasibility_checked=feasibility_checked,
                            feasibility_possible=feasibility_possible,
                            feasibility_sec=feasibility_sec,
                            feasibility_stage_timings=feasibility_stage_timings,
                            feasibility_failure_category=feasibility_failure_category,
                            feasibility_failure_details=feasibility_failure_details,
                            feasibility_grounding_sec=feasibility_ground_elapsed,
                            certificate_grounding_sec=0.0,
                            certificate_solving_sec=None,
                            full_grounding_sec=0.0,
                            full_solve_performed=full_solve_performed,
                            structural_skip_only=False,
                            solve_skipped_reason=solve_skipped_reason,
                            grounding_parts=tuple(grounding_parts),
                        )
                        horizon_records.append(record)
                        if horizon_record_callback is not None:
                            horizon_record_callback(record)
                        if progress_callback is not None:
                            progress_callback(
                                {
                                    "event": "horizon_complete",
                                    "horizon": horizon,
                                    "timestamp_ns": perf_counter_ns(),
                                }
                            )
                        horizon += 1
                        continue

                clause_blocking_mode = (
                    _optimized_model_blocking_mode(facts)
                    if _smart_expansion_enabled(facts)
                    else None
                )
                any_model_seen = False
                models_seen = 0
                models_stored = 0
                unique_workflows_seen = 0
                unique_workflows_stored = 0
                model_callback_sec = 0.0
                shown_symbols_sec = 0.0
                workflow_signature_key_sec = 0.0
                canonicalization_sec = 0.0
                clause_constraints_added = 0
                seen_tool_sequence_count = 0
                stored_tool_sequence_count = 0
                if not solving_started:
                    _report(progress_callback, "Step 3: solving started.")
                    solving_started = True
                assumptions = (
                    _optimized_query_assumptions(
                        horizon=horizon,
                        grounded_horizon=horizon,
                        query_active=True,
                    )
                    if _smart_expansion_enabled(facts)
                    else None
                )
                query_external_symbol = (
                    None
                    if _smart_expansion_enabled(facts)
                    else clingo.Function("query", [clingo.Number(horizon)])
                )

                if query_external_symbol is not None:
                    control.assign_external(query_external_symbol, True)
                try:
                    if optimized_two_phase and full_ground_parts:
                        certificate_ground_elapsed, certificate_grounding_parts = _ground_program_parts(
                            control,
                            _optimized_certificate_horizon_parts(horizon),
                            is_interrupted=is_interrupted,
                            progress_callback=progress_callback,
                            horizon=horizon,
                        )
                        full_ground_elapsed += certificate_ground_elapsed
                        ground_elapsed += certificate_ground_elapsed
                        total_grounding += certificate_ground_elapsed
                        grounding_parts.extend(
                            tuple((f"exact_{name}", elapsed) for name, elapsed in certificate_grounding_parts)
                        )
                        _report(progress_callback, f"Certificate: horizon {horizon}...")
                        certificate_ok, certificate_sec = _run_optimized_exact_certificate(
                            control,
                            horizon=horizon,
                            grounded_horizon=horizon,
                            is_interrupted=is_interrupted,
                        )
                        _report(
                            progress_callback,
                            f"Certificate progress: horizon {horizon} finished after {certificate_sec:.3f}s, "
                            f"possible={'yes' if certificate_ok else 'no'}.",
                        )
                        if not certificate_ok:
                            full_solve_performed = False
                            solve_skipped_reason = "exact_certificate"
                            record = HorizonRecord(
                                horizon=horizon,
                                grounding_sec=ground_elapsed,
                                solving_sec=0.0,
                                peak_rss_mb=current_peak_rss_mb(),
                                satisfiable=False,
                                models_seen=0,
                                models_stored=0,
                                unique_workflows_seen=0,
                                unique_workflows_stored=0,
                                diagnostic_counts_enabled=diagnostic_counts_enabled,
                                available_artifacts_at_step=horizon_metrics.get("available_artifacts_at_step"),
                                eligible_artifacts_at_step=horizon_metrics.get("eligible_artifacts_at_step"),
                                eligible_workflow_inputs_at_step=horizon_metrics.get("eligible_workflow_inputs_at_step"),
                                eligible_produced_outputs_at_step=horizon_metrics.get("eligible_produced_outputs_at_step"),
                                bind_choice_domain_size_at_step=horizon_metrics.get("bind_choice_domain_size_at_step"),
                                feasibility_checked=feasibility_checked,
                                feasibility_possible=feasibility_possible,
                                feasibility_sec=feasibility_sec,
                                feasibility_stage_timings=feasibility_stage_timings,
                                feasibility_failure_category=feasibility_failure_category,
                                feasibility_failure_details=feasibility_failure_details,
                                feasibility_grounding_sec=feasibility_ground_elapsed,
                                certificate_grounding_sec=certificate_ground_elapsed,
                                certificate_solving_sec=certificate_sec,
                                full_grounding_sec=full_ground_elapsed,
                                full_solve_performed=full_solve_performed,
                                structural_skip_only=False,
                                solve_skipped_reason=solve_skipped_reason,
                                grounding_parts=tuple(grounding_parts),
                            )
                            horizon_records.append(record)
                            if horizon_record_callback is not None:
                                horizon_record_callback(record)
                            if progress_callback is not None:
                                progress_callback(
                                    {
                                        "event": "horizon_complete",
                                        "horizon": horizon,
                                        "timestamp_ns": perf_counter_ns(),
                                    }
                                )
                            horizon += 1
                            continue
                        _report(progress_callback, f"Grounding: horizon {horizon} exact_query...")
                        deferred_ground_elapsed, deferred_grounding_parts = _ground_program_parts(
                            control,
                            full_ground_parts,
                            is_interrupted=is_interrupted,
                            progress_callback=progress_callback,
                            horizon=horizon,
                        )
                        full_ground_elapsed += deferred_ground_elapsed
                        ground_elapsed += deferred_ground_elapsed
                        total_grounding += deferred_ground_elapsed
                        grounding_parts.extend(
                            tuple((f"exact_{name}", elapsed) for name, elapsed in deferred_grounding_parts)
                        )
                        _report(
                            progress_callback,
                            f"Grounding progress: horizon {horizon} deferred solve parts finished after "
                            f"{deferred_ground_elapsed:.3f}s.",
                        )

                        _report(progress_callback, f"Solving: horizon {horizon}...")
                        solve_metrics = _solve_on_control(
                            control,
                            config=config,
                            ontology=ontology,
                            tool_input_signatures=tool_input_signatures,
                            workflow_input_dims=workflow_input_dims,
                            tool_output_dims=tool_output_dims,
                            horizon=horizon,
                            assumptions=assumptions,
                            clause_blocking_mode=clause_blocking_mode,
                            capture_raw_models=capture_raw_models,
                            diagnostic_counts_enabled=diagnostic_counts_enabled,
                            solve_all_horizons=solve_all_horizons,
                            raw_collected=raw_collected,
                            unique_collected=unique_collected,
                            stored_unique_keys=stored_unique_keys,
                            progress_callback=progress_callback,
                            is_interrupted=is_interrupted,
                        )
                    else:
                        if full_ground_parts:
                            deferred_ground_elapsed, deferred_grounding_parts = _ground_program_parts(
                                control,
                                full_ground_parts,
                                is_interrupted=is_interrupted,
                                progress_callback=progress_callback,
                                horizon=horizon,
                            )
                            full_ground_elapsed += deferred_ground_elapsed
                            ground_elapsed += deferred_ground_elapsed
                            total_grounding += deferred_ground_elapsed
                            grounding_parts.extend(deferred_grounding_parts)
                            _report(
                                progress_callback,
                                f"Grounding progress: horizon {horizon} deferred solve parts finished after "
                                f"{deferred_ground_elapsed:.3f}s.",
                            )

                        _report(progress_callback, f"Solving: horizon {horizon}...")
                        solve_metrics = _solve_on_control(
                            control,
                            config=config,
                            ontology=ontology,
                            tool_input_signatures=tool_input_signatures,
                            workflow_input_dims=workflow_input_dims,
                            tool_output_dims=tool_output_dims,
                            horizon=horizon,
                            assumptions=assumptions,
                            clause_blocking_mode=clause_blocking_mode,
                            capture_raw_models=capture_raw_models,
                            diagnostic_counts_enabled=diagnostic_counts_enabled,
                            solve_all_horizons=solve_all_horizons,
                            raw_collected=raw_collected,
                            unique_collected=unique_collected,
                            stored_unique_keys=stored_unique_keys,
                            progress_callback=progress_callback,
                            is_interrupted=is_interrupted,
                        )
                finally:
                    if query_external_symbol is not None:
                        control.release_external(query_external_symbol)

                solve_elapsed = solve_metrics.solve_elapsed
                total_solving += solve_elapsed
                any_model_seen = solve_metrics.any_model_seen
                models_seen = solve_metrics.models_seen
                models_stored = solve_metrics.models_stored
                unique_workflows_seen = solve_metrics.unique_workflows_seen
                unique_workflows_stored = solve_metrics.unique_workflows_stored
                model_callback_sec = solve_metrics.model_callback_sec
                shown_symbols_sec = solve_metrics.shown_symbols_sec
                workflow_signature_key_sec = solve_metrics.workflow_signature_key_sec
                canonicalization_sec = solve_metrics.canonicalization_sec
                clause_constraints_added = solve_metrics.clause_constraints_added
                seen_tool_sequence_count = solve_metrics.seen_tool_sequence_count
                stored_tool_sequence_count = solve_metrics.stored_tool_sequence_count

                record = HorizonRecord(
                    horizon=horizon,
                    grounding_sec=ground_elapsed,
                    solving_sec=solve_elapsed,
                    peak_rss_mb=current_peak_rss_mb(),
                    satisfiable=any_model_seen,
                    models_seen=models_seen,
                    models_stored=models_stored,
                    unique_workflows_seen=unique_workflows_seen,
                    unique_workflows_stored=unique_workflows_stored,
                    diagnostic_counts_enabled=diagnostic_counts_enabled,
                    available_artifacts_at_step=horizon_metrics.get("available_artifacts_at_step"),
                    eligible_artifacts_at_step=horizon_metrics.get("eligible_artifacts_at_step"),
                    eligible_workflow_inputs_at_step=horizon_metrics.get("eligible_workflow_inputs_at_step"),
                    eligible_produced_outputs_at_step=horizon_metrics.get("eligible_produced_outputs_at_step"),
                    bind_choice_domain_size_at_step=horizon_metrics.get("bind_choice_domain_size_at_step"),
                    model_callback_sec=model_callback_sec,
                    shown_symbols_sec=shown_symbols_sec,
                    workflow_signature_key_sec=workflow_signature_key_sec,
                    canonicalization_sec=canonicalization_sec,
                    feasibility_checked=feasibility_checked,
                    feasibility_possible=feasibility_possible,
                    feasibility_sec=feasibility_sec,
                    feasibility_stage_timings=feasibility_stage_timings,
                    feasibility_failure_category=feasibility_failure_category,
                    feasibility_failure_details=feasibility_failure_details,
                    feasibility_grounding_sec=feasibility_ground_elapsed,
                    certificate_grounding_sec=certificate_ground_elapsed if optimized_two_phase else 0.0,
                    certificate_solving_sec=certificate_sec if optimized_two_phase else None,
                    full_grounding_sec=full_ground_elapsed,
                    full_solve_performed=full_solve_performed,
                    structural_skip_only=False,
                    solve_skipped_reason=solve_skipped_reason,
                    clause_blocking_mode=clause_blocking_mode,
                    clause_constraints_added=clause_constraints_added,
                    grounding_parts=tuple(grounding_parts),
                )
                horizon_records.append(record)
                if horizon_record_callback is not None:
                    horizon_record_callback(record)
                if progress_callback is not None:
                    progress_callback(
                        {
                            "event": "horizon_complete",
                            "horizon": horizon,
                            "timestamp_ns": perf_counter_ns(),
                        }
                    )
                _report(
                    progress_callback,
                    f"Solving progress: horizon {horizon} finished after {solve_elapsed:.3f}s, "
                    f"{_format_progress_counts(
                        diagnostic_counts_enabled=diagnostic_counts_enabled,
                        capture_raw_models=capture_raw_models,
                        models_seen=models_seen,
                        models_stored=models_stored,
                        unique_workflows_seen=unique_workflows_seen,
                        unique_workflows_stored=unique_workflows_stored,
                        seen_tool_sequence_count=seen_tool_sequence_count,
                        stored_tool_sequence_count=stored_tool_sequence_count,
                    )}, "
                    f"satisfiable={'yes' if any_model_seen else 'no'}.",
                )
                if not solve_all_horizons and _stored_solution_quota_reached(
                    unique_count=len(unique_collected),
                    solution_limit=config.solutions,
                ):
                    break
                if unique_collected and not solve_all_horizons and stop_on_solution:
                    break
                horizon += 1
    except KeyboardInterrupt:
        _report(progress_callback, "Interrupted/timeout: returning partial results.")
    finally:
        control.cleanup()

    return SolveOutput(
        raw_solutions=tuple(raw_collected),
        solutions=tuple(unique_collected),
        base_grounding_peak_rss_mb=base_grounding_peak_rss_mb,
        base_grounding_sec=base_grounding_sec,
        grounding_sec=total_grounding,
        solving_sec=total_solving,
        horizon_records=tuple(horizon_records),
    )


def _ground_multi_shot_control(
    control: clingo.Control,
    config: SnakeConfig,
    *,
    stage: str,
    collect_horizon_metrics: bool = False,
    progress_callback: ProgressCallback = None,
    base_grounding_callback: BaseGroundingCallback = None,
    horizon_record_callback: HorizonRecordCallback = None,
    initial_step_program: str | None = None,
    initial_seed_program: str | None = None,
    horizon_parts_builder: Callable[[int], tuple[tuple[str, tuple[clingo.Symbol, ...]], ...]] | None = None,
) -> GroundingOutput:
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
                for horizon in range(1, config.solution_length_max + 1):
                    _report(progress_callback, f"Grounding: horizon {horizon}...")
                    if horizon_parts_builder is None:
                        ground_parts = _default_horizon_parts(
                            horizon,
                            initial_step_program=initial_step_program,
                            initial_seed_program=initial_seed_program,
                        )
                    else:
                        ground_parts = horizon_parts_builder(horizon)
                    elapsed, grounding_parts = _ground_program_parts(
                        control,
                        ground_parts,
                        is_interrupted=is_interrupted,
                        progress_callback=progress_callback,
                        horizon=horizon,
                    )
                    horizon_metrics = (
                        _legacy_direct_multishot_horizon_metrics(control, horizon=horizon)
                        if collect_horizon_metrics
                        else {}
                    )
                    total_grounding += elapsed
                    grounded_horizons.append(horizon)
                    record = HorizonRecord(
                        horizon=horizon,
                        grounding_sec=elapsed,
                        solving_sec=0.0,
                        peak_rss_mb=current_peak_rss_mb(),
                        satisfiable=False,
                        models_seen=0,
                        models_stored=0,
                        unique_workflows_seen=0,
                        unique_workflows_stored=0,
                        available_artifacts_at_step=horizon_metrics.get("available_artifacts_at_step"),
                        eligible_artifacts_at_step=horizon_metrics.get("eligible_artifacts_at_step"),
                        eligible_workflow_inputs_at_step=horizon_metrics.get("eligible_workflow_inputs_at_step"),
                        eligible_produced_outputs_at_step=horizon_metrics.get("eligible_produced_outputs_at_step"),
                        bind_choice_domain_size_at_step=horizon_metrics.get("bind_choice_domain_size_at_step"),
                        grounding_parts=grounding_parts,
                    )
                    horizon_records.append(record)
                    if horizon_record_callback is not None:
                        horizon_record_callback(record)
                    _report(
                        progress_callback,
                        f"Grounding progress: horizon {horizon} finished after {elapsed:.3f}s.",
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
