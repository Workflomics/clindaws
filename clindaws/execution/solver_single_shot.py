"""Single-shot solver internals: ground-once-and-solve over horizon slices."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter, perf_counter_ns

import clingo

from clindaws.core.models import FactBundle, HorizonRecord, SnakeConfig
from clindaws.core.ontology import Ontology
from clindaws.core.runtime_stats import current_peak_rss_mb
from clindaws.core.workflow import (
    canonicalize_shown_symbols,
    extract_canonical_workflow_keys,
    workflow_signature_length,
)
from clindaws.execution.solver_control import (
    BaseGroundingCallback,
    HorizonRecordCallback,
    ProgressCallback,
    _make_solve_control,
    _projection_runtime_facts,
)
from clindaws.execution.solver_solutions import (
    SolveOutput,
    _stored_solution_quota_reached,
    _stored_workflow_key,
)
from clindaws.execution.solver_utils import (
    _format_progress_counts,
    _interrupt_guard,
    _report,
    _run_interruptible,
)


SINGLE_SHOT_OVERLAY_PREFIX = ""


def _single_shot_overlay(min_length: int, horizon: int) -> str:
    """Return one-shot constraints layered on top of the shared multi-shot core.

    The overlay enforces that the final grounded horizon reaches the goal and
    exposes external atoms used by Python to solve exact ``(goal_time,
    run_count)`` slices in a deterministic shortest-first order.
    """
    overlay = [
        SINGLE_SHOT_OVERLAY_PREFIX,
        f":- not holds({horizon}, goal).",
        ":- occurs(T, run(_)), not occurs(T-1, run(_)), T > 1.",
        ":- occurs(T, run(ToolA)), occurs(T, run(ToolB)), ToolA < ToolB.",
        "",
        "% One-shot ordering layer: solve exact (goal_time, run_count) slices on one",
        "% grounded control object, lexicographically by earlier goal then shorter",
        "% active prefix.",
        "single_shot_goal_time(1) :- holds(1, goal).",
        "single_shot_goal_time(T) :- T > 1, holds(T, goal), not holds(T-1, goal).",
        "single_shot_run_count(N) :- N = #count { T : occurs(T, run(_)) }.",
        "#external single_shot_target_goal_time(T) : time(T).",
        "#external single_shot_target_run_count(N) : time(N).",
        ":- single_shot_target_goal_time(T), not single_shot_goal_time(T).",
        ":- single_shot_target_goal_time(Target), single_shot_goal_time(Actual), Actual != Target.",
        ":- single_shot_target_run_count(N), not single_shot_run_count(N).",
        ":- single_shot_target_run_count(Target), single_shot_run_count(Actual), Actual != Target.",
    ]
    if min_length > 1:
        overlay.append(f":- holds({min_length - 1}, goal).")
    return "\n".join(overlay) + "\n"


def _single_shot_full_ground_parts(
    horizon: int,
) -> tuple[tuple[str, tuple[clingo.Symbol, ...]], ...]:
    parts: list[tuple[str, tuple[clingo.Symbol, ...]]] = [("init", ())]
    for current_horizon in range(1, horizon + 1):
        parts.append(("step", (clingo.Number(current_horizon),)))
        parts.append(("constraint_step", (clingo.Number(current_horizon),)))
    parts.append(("check", (clingo.Number(horizon),)))
    parts.append(("single_shot", ()))
    return tuple(parts)


def _solve_single_shot_with_programs(
    config: SnakeConfig,
    facts: FactBundle,
    program_paths: tuple[Path, ...],
    *,
    ontology: Ontology | None = None,
    workflow_input_dims: dict[str, dict[str, tuple[str, ...]]] | None = None,
    tool_output_dims: dict[tuple[str, int], dict[str, tuple[str, ...]]] | None = None,
    progress_callback: ProgressCallback = None,
    base_grounding_callback: BaseGroundingCallback = None,
    horizon_record_callback: HorizonRecordCallback = None,
    solve_all_horizons: bool = False,
    capture_raw_models: bool = False,
    diagnostic_counts_enabled: bool = True,
    parallel_mode: str | None = None,
    project_models: bool = False,
) -> SolveOutput:
    """Solve using the single-shot encoding by iterating horizons."""

    raw_solutions: list[tuple[clingo.Symbol, ...]] = []
    unique_solutions: list[tuple[clingo.Symbol, ...]] = []
    stored_unique_keys: set[tuple[object, ...]] = set()
    total_grounding = 0.0
    total_solving = 0.0
    base_grounding_peak_rss_mb = 0.0
    base_grounding_sec = 0.0
    horizon_records: list[HorizonRecord] = []
    solving_started = False
    tool_input_signatures = dict(facts.tool_input_signatures)
    horizon = config.solution_length_min
    while horizon <= config.solution_length_max:
        if not solve_all_horizons and _stored_solution_quota_reached(
            unique_count=len(unique_solutions),
            solution_limit=config.solutions,
        ):
            break

        control = _make_solve_control(
            parallel_mode=parallel_mode,
            project_models=project_models,
        )
        for program_path in program_paths:
            control.load(str(program_path))
        control.add("base", [], facts.facts)
        runtime_facts = _projection_runtime_facts(mode="single-shot", project_models=project_models)
        if runtime_facts:
            control.add("base", [], runtime_facts)
        if facts.python_precomputed_facts:
            control.add("base", [], facts.python_precomputed_facts)
        control.add("single_shot", [], _single_shot_overlay(config.solution_length_min, horizon))

        _horizon_interrupted = False
        try:
            with _interrupt_guard(control) as is_interrupted:
                if horizon == config.solution_length_min:
                    _report(progress_callback, "Step 2: grounding started.")
                _report(progress_callback, f"Grounding: single-shot horizon {horizon}...")
                grounding_parts: list[tuple[str, float]] = []
                start = perf_counter()
                _run_interruptible(lambda: control.ground([("base", [])]), is_interrupted)
                base_elapsed = perf_counter() - start
                ground_elapsed = base_elapsed
                grounding_parts.append(("base", base_elapsed))
                total_grounding += base_elapsed
                if horizon == config.solution_length_min:
                    base_grounding_sec += base_elapsed
                    base_grounding_peak_rss_mb = current_peak_rss_mb()
                    if base_grounding_callback is not None:
                        base_grounding_callback(base_grounding_sec, base_grounding_peak_rss_mb)
                full_ground_parts = _single_shot_full_ground_parts(horizon)
                start = perf_counter()
                _run_interruptible(
                    lambda: control.ground(
                        [(name, list(args)) for name, args in full_ground_parts]
                    ),
                    is_interrupted,
                )
                full_ground_elapsed = perf_counter() - start
                ground_elapsed += full_ground_elapsed
                grounding_parts.append(("single_shot_full", full_ground_elapsed))
                total_grounding += full_ground_elapsed
                _report(
                    progress_callback,
                    f"Grounding progress: single-shot horizon {horizon} finished after {ground_elapsed:.3f}s.",
                )

                if not solving_started:
                    _report(progress_callback, "Step 3: solving started.")
                    solving_started = True
                _report(progress_callback, f"Solving: single-shot horizon {horizon}...")
                any_model_seen = False
                models_seen = 0
                models_stored = 0
                unique_workflows_seen = 0
                unique_workflows_stored = 0
                model_callback_sec = 0.0
                shown_symbols_sec = 0.0
                workflow_signature_key_sec = 0.0
                canonicalization_sec = 0.0
                seen_unique_keys: set[tuple[object, ...]] = set()
                seen_tool_sequence_keys: set[tuple[object, ...]] = set()
                stored_tool_sequence_keys: set[tuple[object, ...]] = set()
                start = perf_counter()
                query_symbol = clingo.Function("query", [clingo.Number(horizon)])
                control.assign_external(query_symbol, True)

                def _solve() -> None:
                    with control.solve(yield_=True) as handle:
                        for model in handle:
                            nonlocal models_seen, models_stored, unique_workflows_seen, unique_workflows_stored
                            nonlocal model_callback_sec, shown_symbols_sec
                            nonlocal workflow_signature_key_sec, canonicalization_sec
                            nonlocal any_model_seen
                            callback_start = perf_counter()
                            any_model_seen = True
                            if diagnostic_counts_enabled:
                                models_seen += 1
                            sample_start = perf_counter()
                            shown_symbols = tuple(model.symbols(shown=True))
                            shown_symbols_sec += perf_counter() - sample_start
                            sample_start = perf_counter()
                            tool_sequence_key, workflow_key = extract_canonical_workflow_keys(
                                shown_symbols,
                                tool_input_signatures,
                                workflow_input_dims,
                                tool_output_dims,
                                ontology=ontology,
                                use_binding_target_abstraction=(
                                    not config.tool_seq_repeat
                                ),
                            )
                            workflow_signature_key_sec += perf_counter() - sample_start
                            workflow_storage_key = _stored_workflow_key(
                                config=config,
                                tool_sequence_key=tool_sequence_key,
                                workflow_key=workflow_key,
                            )
                            workflow_length = workflow_signature_length(workflow_key)
                            in_length_window = (
                                config.solution_length_min
                                <= workflow_length
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
                                and
                                capture_raw_models
                                and len(raw_solutions) < config.solutions
                            )
                            store_unique = (
                                in_length_window
                                and
                                workflow_storage_key not in stored_unique_keys
                                and len(unique_solutions) < config.solutions
                            )
                            if store_raw:
                                raw_solutions.append(shown_symbols)
                                models_stored += 1
                            if store_unique:
                                sample_start = perf_counter()
                                canonical_shown = canonicalize_shown_symbols(
                                    shown_symbols,
                                    tool_input_signatures,
                                    workflow_input_dims,
                                    tool_output_dims,
                                )
                                canonicalization_sec += perf_counter() - sample_start
                                stored_unique_keys.add(workflow_storage_key)
                                stored_tool_sequence_keys.add(tool_sequence_key)
                                unique_solutions.append(canonical_shown)
                                unique_workflows_stored += 1
                            elif (
                                solve_all_horizons
                                and _stored_solution_quota_reached(
                                    unique_count=len(unique_solutions),
                                    solution_limit=config.solutions,
                                )
                            ):
                                model_callback_sec += perf_counter() - callback_start
                                break
                            if not solve_all_horizons and _stored_solution_quota_reached(
                                unique_count=len(unique_solutions),
                                solution_limit=config.solutions,
                            ):
                                model_callback_sec += perf_counter() - callback_start
                                break
                            model_callback_sec += perf_counter() - callback_start

                try:
                    _run_interruptible(_solve, is_interrupted)
                    solve_elapsed = perf_counter() - start
                    total_solving += solve_elapsed
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
                        model_callback_sec=model_callback_sec,
                        shown_symbols_sec=shown_symbols_sec,
                        workflow_signature_key_sec=workflow_signature_key_sec,
                        canonicalization_sec=canonicalization_sec,
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
                        f"Solving progress: single-shot horizon {horizon} finished after {solve_elapsed:.3f}s, "
                        f"{_format_progress_counts(
                            diagnostic_counts_enabled=diagnostic_counts_enabled,
                            capture_raw_models=capture_raw_models,
                            models_seen=models_seen,
                            models_stored=models_stored,
                            unique_workflows_seen=unique_workflows_seen,
                            unique_workflows_stored=unique_workflows_stored,
                            seen_tool_sequence_count=len(seen_tool_sequence_keys),
                            stored_tool_sequence_count=len(stored_tool_sequence_keys),
                        )}, "
                        f"satisfiable={'yes' if any_model_seen else 'no'}.",
                    )
                finally:
                    control.release_external(query_symbol)
        except KeyboardInterrupt:
            _report(progress_callback, "Interrupted/timeout: returning partial results.")
            _horizon_interrupted = True
        finally:
            control.cleanup()

        if _horizon_interrupted:
            break
        if not solve_all_horizons and _stored_solution_quota_reached(
            unique_count=len(unique_solutions),
            solution_limit=config.solutions,
        ):
            break
        horizon += 1

    return SolveOutput(
        raw_solutions=tuple(raw_solutions),
        solutions=tuple(unique_solutions),
        base_grounding_peak_rss_mb=base_grounding_peak_rss_mb,
        base_grounding_sec=base_grounding_sec,
        grounding_sec=total_grounding,
        solving_sec=total_solving,
        horizon_records=tuple(horizon_records),
    )


def _solve_single_shot_once(
    config: SnakeConfig,
    facts: FactBundle,
    program_paths: tuple[Path, ...],
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
    """Ground the plain multi-shot programs once and solve at max horizon.

    Stored results stop at ``config.solutions`` unique canonical workflows.
    Raw answer sets remain optional diagnostics and do not control termination.
    """

    raw_solutions: list[tuple[clingo.Symbol, ...]] = []
    unique_solutions: list[tuple[clingo.Symbol, ...]] = []
    stored_unique_keys: set[tuple[object, ...]] = set()
    tool_input_signatures = dict(facts.tool_input_signatures)
    horizon = config.solution_length_max

    static_overlay = _single_shot_overlay(config.solution_length_min, horizon)

    control = _make_solve_control(
        parallel_mode=parallel_mode,
        project_models=project_models,
    )
    for program_path in program_paths:
        control.load(str(program_path))
    control.add("base", [], facts.facts)
    runtime_facts = _projection_runtime_facts(mode="single-shot", project_models=project_models)
    if runtime_facts:
        control.add("base", [], runtime_facts)
    if facts.python_precomputed_facts:
        control.add("base", [], facts.python_precomputed_facts)
    control.add("single_shot", [], static_overlay)

    ground_elapsed = 0.0
    solve_elapsed = 0.0
    base_grounding_peak_rss_mb = 0.0
    base_grounding_sec = 0.0
    models_seen = 0
    models_stored = 0
    unique_workflows_seen = 0
    unique_workflows_stored = 0
    any_model_seen = False
    model_callback_sec = 0.0
    shown_symbols_sec = 0.0
    workflow_signature_key_sec = 0.0
    canonicalization_sec = 0.0
    grounding_parts: list[tuple[str, float]] = []
    seen_unique_keys: set[tuple[object, ...]] = set()
    seen_tool_sequence_keys: set[tuple[object, ...]] = set()
    stored_tool_sequence_keys: set[tuple[object, ...]] = set()

    try:
        with _interrupt_guard(control) as is_interrupted:
            _report(progress_callback, "Step 2: grounding started.")
            _report(progress_callback, "Grounding: base program...")
            start = perf_counter()
            _run_interruptible(lambda: control.ground([("base", [])]), is_interrupted)
            base_grounding_sec = perf_counter() - start
            ground_elapsed = base_grounding_sec
            base_grounding_peak_rss_mb = current_peak_rss_mb()
            if base_grounding_callback is not None:
                base_grounding_callback(base_grounding_sec, base_grounding_peak_rss_mb)
            _report(
                progress_callback,
                f"Grounding progress: base program finished after {base_grounding_sec:.3f}s.",
            )
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
            ground_elapsed += full_ground_elapsed
            grounding_parts.append(("single_shot_full", full_ground_elapsed))
            _report(
                progress_callback,
                f"Grounding progress: single-shot full program finished after {full_ground_elapsed:.3f}s.",
            )
            _report(
                progress_callback,
                f"Grounding progress: single-shot finished after {ground_elapsed:.3f}s.",
            )

            _report(progress_callback, "Step 3: solving started.")
            _report(progress_callback, "Solving: single-shot...")
            start = perf_counter()
            query_symbol = clingo.Function("query", [clingo.Number(horizon)])
            control.assign_external(query_symbol, True)
            goal_time_symbols = {
                goal_time: clingo.Function("single_shot_target_goal_time", [clingo.Number(goal_time)])
                for goal_time in range(config.solution_length_min, horizon + 1)
            }
            run_count_symbols = {
                run_count: clingo.Function("single_shot_target_run_count", [clingo.Number(run_count)])
                for run_count in range(config.solution_length_min, horizon + 1)
            }

            def _solve_layer() -> None:
                nonlocal models_seen, models_stored, unique_workflows_seen, unique_workflows_stored
                nonlocal model_callback_sec, shown_symbols_sec
                nonlocal workflow_signature_key_sec, canonicalization_sec
                with control.solve(yield_=True) as handle:
                    for model in handle:
                        nonlocal any_model_seen
                        callback_start = perf_counter()
                        any_model_seen = True
                        if diagnostic_counts_enabled:
                            models_seen += 1
                        sample_start = perf_counter()
                        shown_symbols = tuple(model.symbols(shown=True))
                        shown_symbols_sec += perf_counter() - sample_start
                        sample_start = perf_counter()
                        tool_sequence_key, workflow_key = extract_canonical_workflow_keys(
                            shown_symbols,
                            tool_input_signatures,
                            workflow_input_dims,
                            tool_output_dims,
                            ontology=ontology,
                            use_binding_target_abstraction=(
                                not config.tool_seq_repeat
                            ),
                        )
                        workflow_signature_key_sec += perf_counter() - sample_start
                        workflow_storage_key = _stored_workflow_key(
                            config=config,
                            tool_sequence_key=tool_sequence_key,
                            workflow_key=workflow_key,
                        )
                        workflow_length = workflow_signature_length(workflow_key)
                        in_length_window = (
                            config.solution_length_min
                            <= workflow_length
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
                            and len(raw_solutions) < config.solutions
                        )
                        store_unique = (
                            in_length_window
                            and workflow_storage_key not in stored_unique_keys
                            and len(unique_solutions) < config.solutions
                        )
                        if store_raw:
                            raw_solutions.append(shown_symbols)
                            models_stored += 1
                        if store_unique:
                            sample_start = perf_counter()
                            canonical_shown = canonicalize_shown_symbols(
                                shown_symbols,
                                tool_input_signatures,
                                workflow_input_dims,
                                tool_output_dims,
                            )
                            canonicalization_sec += perf_counter() - sample_start
                            unique_workflows_stored += 1
                            stored_unique_keys.add(workflow_storage_key)
                            stored_tool_sequence_keys.add(tool_sequence_key)
                            unique_solutions.append(canonical_shown)

                        model_callback_sec += perf_counter() - callback_start
                        if _stored_solution_quota_reached(
                            unique_count=len(unique_solutions),
                            solution_limit=config.solutions,
                        ):
                            break

            try:
                for goal_time in range(config.solution_length_min, horizon + 1):
                    if _stored_solution_quota_reached(
                        unique_count=len(unique_solutions),
                        solution_limit=config.solutions,
                    ):
                        break
                    goal_time_symbol = goal_time_symbols[goal_time]
                    control.assign_external(goal_time_symbol, True)
                    try:
                        for run_count in range(goal_time, horizon + 1):
                            if _stored_solution_quota_reached(
                                unique_count=len(unique_solutions),
                                solution_limit=config.solutions,
                            ):
                                break
                            run_count_symbol = run_count_symbols[run_count]
                            control.assign_external(run_count_symbol, True)
                            try:
                                _run_interruptible(_solve_layer, is_interrupted)
                            finally:
                                control.assign_external(run_count_symbol, False)
                    finally:
                        control.assign_external(goal_time_symbol, False)
                solve_elapsed = perf_counter() - start
            finally:
                control.release_external(query_symbol)
                control.cleanup()
            _report(
                progress_callback,
                f"Solving progress: single-shot finished after {solve_elapsed:.3f}s, "
                f"{_format_progress_counts(
                    diagnostic_counts_enabled=diagnostic_counts_enabled,
                    capture_raw_models=capture_raw_models,
                    models_seen=models_seen,
                    models_stored=models_stored,
                    unique_workflows_seen=unique_workflows_seen,
                    unique_workflows_stored=unique_workflows_stored,
                    seen_tool_sequence_count=len(seen_tool_sequence_keys),
                    stored_tool_sequence_count=len(stored_tool_sequence_keys),
                )}, "
                f"satisfiable={'yes' if any_model_seen else 'no'}.",
            )
    except KeyboardInterrupt:
        _report(progress_callback, "Interrupted/timeout: returning partial results.")
    finally:
        control.cleanup()

    peak_rss_mb = max(base_grounding_peak_rss_mb, current_peak_rss_mb())

    record = HorizonRecord(
        horizon=horizon,
        grounding_sec=ground_elapsed,
        solving_sec=solve_elapsed,
        peak_rss_mb=peak_rss_mb,
        satisfiable=any_model_seen,
        models_seen=models_seen,
        models_stored=models_stored,
        unique_workflows_seen=unique_workflows_seen,
        unique_workflows_stored=unique_workflows_stored,
        diagnostic_counts_enabled=diagnostic_counts_enabled,
        model_callback_sec=model_callback_sec,
        shown_symbols_sec=shown_symbols_sec,
        workflow_signature_key_sec=workflow_signature_key_sec,
        canonicalization_sec=canonicalization_sec,
        grounding_parts=tuple(grounding_parts),
    )
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

    return SolveOutput(
        raw_solutions=tuple(raw_solutions),
        solutions=tuple(unique_solutions),
        base_grounding_peak_rss_mb=base_grounding_peak_rss_mb,
        base_grounding_sec=base_grounding_sec,
        grounding_sec=ground_elapsed,
        solving_sec=solve_elapsed,
        horizon_records=(record,),
    )
