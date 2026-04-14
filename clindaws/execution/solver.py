"""Clingo solver orchestration.

This module maps the public CLI modes onto concrete ASP program families and
then drives grounding/solving in clingo.

- ``multi-shot`` uses the direct incremental encoding family.
- ``multi-shot --optimized`` uses the compressed-candidate incremental family.
- ``single-shot`` currently reuses the direct ``multi_shot`` encoding files but
  grounds them once over ``time(1..max_length)`` and solves on that single
  control object.

Canonical workflow candidates are the default stored result surface. Raw answer
sets remain optional diagnostics for debugging multiplicity and callback cost.
"""

from __future__ import annotations

import os
import signal
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter, perf_counter_ns
from typing import Callable, Iterator

import clingo

from clindaws.core.models import FactBundle, HorizonRecord, SnakeConfig
from clindaws.core.runtime_stats import current_peak_rss_mb
from clindaws.core.workflow import (
    canonicalize_shown_symbols,
    extract_canonical_workflow_keys,
    workflow_signature_length,
)


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
ENCODINGS_ROOT = PACKAGE_ROOT / "encodings"
ProgressCallback = Callable[[object], None] | None
BaseGroundingCallback = Callable[[float, float], None] | None
HorizonRecordCallback = Callable[[HorizonRecord], None] | None


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



def _make_solve_control(
    *,
    parallel_mode: str | None = None,
    project_models: bool = False,
) -> clingo.Control:
    """Create a clingo control for solve-time model enumeration.

    The public runtime keeps clingo's native model bound open and enforces the
    stored workflow quota in Python after canonicalization. That preserves the
    meaning of ``solutions`` as a cap on unique stored workflows rather than a
    cap on raw pre-canonical answer sets.
    """

    args = ["0", "--warn=none"]
    if project_models:
        args.append("--project")
    if parallel_mode:
        args.append(f"--parallel-mode={parallel_mode}")
    return clingo.Control(args)


def _make_grounding_control() -> clingo.Control:
    """Create a clingo control for grounding-only operations."""

    return clingo.Control(["0", "--warn=none"])


def _projection_runtime_facts(*, mode: str, project_models: bool) -> str:
    """Return runtime fact toggles derived from the chosen solve policy."""

    facts: list[str] = []
    if mode in {"multi-shot", "single-shot"} and not project_models:
        facts.append("full_workflow_input_witnesses.\n")
    return "".join(facts)


def _format_progress_counts(
    *,
    diagnostic_counts_enabled: bool,
    capture_raw_models: bool,
    models_seen: int,
    models_stored: int,
    unique_workflows_seen: int,
    unique_workflows_stored: int,
    seen_tool_sequence_count: int,
    stored_tool_sequence_count: int,
) -> str:
    """Format solve progress counts based on the active reporting mode.

    Normal runs only report stored canonical workflow candidates. Diagnostic
    mode adds raw-answer-set and seen/stored counters so dense benchmarks can be
    analyzed without changing the stored result artifact.
    """

    if not diagnostic_counts_enabled:
        return (
            f"workflow candidates stored={unique_workflows_stored}, "
            f"unique tool sequences stored={stored_tool_sequence_count}"
        )
    parts = [f"raw models seen={models_seen}"]
    if capture_raw_models:
        parts.append(f"raw models stored={models_stored}")
    parts.extend(
        [
            f"workflow candidates seen={unique_workflows_seen}",
            f"workflow candidates stored={unique_workflows_stored}",
            f"unique tool sequences seen={seen_tool_sequence_count}",
            f"unique tool sequences stored={stored_tool_sequence_count}",
        ]
    )
    return ", ".join(parts)


@dataclass(frozen=True)
class SolveOutput:
    """Raw solver output."""

    raw_solutions: tuple[tuple[clingo.Symbol, ...], ...]
    solutions: tuple[tuple[clingo.Symbol, ...], ...]
    base_grounding_peak_rss_mb: float
    base_grounding_sec: float
    grounding_sec: float
    solving_sec: float
    horizon_records: tuple[HorizonRecord, ...]


@dataclass(frozen=True)
class GroundingOutput:
    """Grounding-only solver output."""

    base_grounding_peak_rss_mb: float
    base_grounding_sec: float
    grounding_sec: float
    grounded_horizons: tuple[int, ...]
    horizon_records: tuple[HorizonRecord, ...]


def _single_shot_program_paths(*, optimized: bool = False) -> tuple[Path, ...]:
    """Return the active program set for public single-shot mode.

    The current single-shot runtime deliberately reuses the direct multi-shot
    encoding family. Solver behavior differs by grounding/solving strategy, not
    by a separate active ASP program family.
    """
    if optimized:
        raise ValueError("--optimized is not yet supported for single-shot.")
    return _multi_shot_program_paths()


def _multi_shot_program_paths() -> tuple[Path, ...]:
    """Return the plain direct incremental multi-shot program set."""
    base = ENCODINGS_ROOT / "multi_shot"
    return (
        base / "base.lp",
        base / "init.lp",
        base / "step.lp",
        base / "constraints.lp",
        base / "check.lp",
        base / "ape_extract.lp",
        base / "tool_inclusion.lp",
        base / "tool_dependency.lp",
        base / "temporal_constraint.lp",
        base / "input_usage.lp",
        base / "output_usage.lp",
    )



def _multi_shot_compressed_candidate_program_paths() -> tuple[Path, ...]:
    """Return the optimized compressed-candidate incremental program set."""
    base = ENCODINGS_ROOT / "multi_shot_compressed_candidate"
    return (
        base / "base.lp",
        base / "step_initial.lp",
        base / "step.lp",
        base / "step_query.lp",
        base / "constraints.lp",
        base / "check.lp",
        base / "ape_extract.lp",
        base / "tool_inclusion.lp",
        base / "input_usage.lp",
        base / "output_usage.lp",
    )


def program_paths_for_mode(
    mode: str,
    *,
    optimized: bool = False,
) -> tuple[Path, ...]:
    """Return the concrete ASP program family for one effective solver mode."""

    if mode in {"single-shot", "single-shot-sliding-window"}:
        return _single_shot_program_paths(optimized=optimized)
    if mode == "multi-shot":
        if optimized:
            return _multi_shot_compressed_candidate_program_paths()
        return _multi_shot_program_paths()
    if mode == "multi-shot-compressed-candidate":
        return _multi_shot_compressed_candidate_program_paths()
    raise ValueError(f"Unsupported mode: {mode}")


def _report(progress_callback: ProgressCallback, message: str) -> None:
    if progress_callback is not None:
        progress_callback(message)


def _stored_solution_quota_reached(
    *,
    unique_count: int,
    solution_limit: int,
) -> bool:
    """Return whether stored solution collection has reached its configured cap."""

    return unique_count >= solution_limit


def _artifact_is_produced_output(symbol: clingo.Symbol) -> bool:
    return (
        symbol.type == clingo.SymbolType.Function
        and symbol.name == "out"
        and len(symbol.arguments) == 3
    )


def _legacy_direct_multishot_horizon_metrics(
    control: clingo.Control,
    *,
    horizon: int,
) -> dict[str, int]:
    """Collect grounded-domain counts for direct multi-shot eligibility/binding."""

    metrics = {
        "available_artifacts_at_step": 0,
        "eligible_artifacts_at_step": 0,
        "eligible_workflow_inputs_at_step": 0,
        "eligible_produced_outputs_at_step": 0,
        "bind_choice_domain_size_at_step": 0,
    }
    horizon_symbol = clingo.Number(horizon)
    available_artifacts: set[str] = set()

    for atom in control.symbolic_atoms.by_signature("holds", 2):
        symbol = atom.symbol
        if symbol.arguments[0] != horizon_symbol:
            continue
        state = symbol.arguments[1]
        if (
            state.type == clingo.SymbolType.Function
            and state.name == "avail"
            and len(state.arguments) == 1
        ):
            available_artifacts.add(str(state.arguments[0]))

    for atom in control.symbolic_atoms.by_signature("available", 1):
        symbol = atom.symbol
        available_artifacts.add(str(symbol.arguments[0]))

    metrics["available_artifacts_at_step"] = len(available_artifacts)

    for atom in control.symbolic_atoms.by_signature("eligible", 3):
        symbol = atom.symbol
        if symbol.arguments[0] != horizon_symbol:
            continue
        metrics["eligible_artifacts_at_step"] += 1
        artifact = symbol.arguments[2]
        if _artifact_is_produced_output(artifact):
            metrics["eligible_produced_outputs_at_step"] += 1
        else:
            metrics["eligible_workflow_inputs_at_step"] += 1

    for atom in control.symbolic_atoms.by_signature("occurs", 2):
        symbol = atom.symbol
        if symbol.arguments[0] != horizon_symbol:
            continue
        event = symbol.arguments[1]
        if (
            event.type == clingo.SymbolType.Function
            and event.name == "bind"
            and len(event.arguments) == 3
        ):
            metrics["bind_choice_domain_size_at_step"] += 1

    return metrics


def _collect_direct_multishot_metrics(facts: FactBundle) -> bool:
    return facts.internal_solver_mode == "multi-shot"


@contextmanager
def _interrupt_guard(control: clingo.Control) -> Iterator[Callable[[], bool]]:
    interrupted = False
    previous_handler = None

    try:
        previous_handler = signal.getsignal(signal.SIGINT)

        def _handle_sigint(signum, frame) -> None:  # type: ignore[override]
            nonlocal interrupted
            if interrupted:
                try:
                    os.write(2, b"Second interrupt received, forcing shutdown.\n")
                except OSError:
                    pass
                signal.default_int_handler(signum, frame)
            interrupted = True
            try:
                os.write(2, b"Interrupt requested, stopping current Clingo operation...\n")
            except OSError:
                pass
            control.interrupt()

        signal.signal(signal.SIGINT, _handle_sigint)
    except ValueError:
        previous_handler = None

    try:
        yield lambda: interrupted
    finally:
        if previous_handler is not None:
            signal.signal(signal.SIGINT, previous_handler)


def _raise_if_interrupted(is_interrupted: Callable[[], bool]) -> None:
    if is_interrupted():
        raise KeyboardInterrupt


def _run_interruptible(operation: Callable[[], None], is_interrupted: Callable[[], bool]) -> None:
    try:
        operation()
    except RuntimeError as exc:
        if is_interrupted():
            raise KeyboardInterrupt from exc
        raise
    _raise_if_interrupted(is_interrupted)


def _ground_program_parts(
    control: clingo.Control,
    parts: tuple[tuple[str, tuple[clingo.Symbol, ...]], ...],
    *,
    is_interrupted: Callable[[], bool],
    progress_callback: ProgressCallback,
    horizon: int,
) -> tuple[float, tuple[tuple[str, float], ...]]:
    total_elapsed = 0.0
    part_timings: list[tuple[str, float]] = []
    for name, args in parts:
        _report(progress_callback, f"Grounding: horizon {horizon} {name}...")
        start = perf_counter()
        _run_interruptible(
            lambda program_name=name, program_args=args: control.ground([(program_name, list(program_args))]),
            is_interrupted,
        )
        elapsed = perf_counter() - start
        total_elapsed += elapsed
        part_timings.append((name, elapsed))
        _report(
            progress_callback,
            f"Grounding progress: horizon {horizon} {name} finished after {elapsed:.3f}s.",
        )
    return total_elapsed, tuple(part_timings)


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


def _single_shot_horizon_parts(
    horizon: int,
) -> tuple[tuple[str, tuple[clingo.Symbol, ...]], ...]:
    parts: list[tuple[str, tuple[clingo.Symbol, ...]]] = []
    if horizon == 1:
        parts.append(("init", ()))
    parts.append(("step", (clingo.Number(horizon),)))
    parts.append(("constraint_step", (clingo.Number(horizon),)))
    return tuple(parts)


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


def _has_translated_constraints(facts: FactBundle) -> bool:
    """Return whether the translated fact bundle contains any constraint facts."""
    return any(
        count > 0 and predicate.startswith("constraint_")
        for predicate, count in facts.predicate_counts.items()
    )


def _solve_multi_shot_with_programs(
    config: SnakeConfig,
    facts: FactBundle,
    program_paths: tuple[Path, ...],
    mode: str,
    *,
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
    for program_path in program_paths:
        control.load(str(program_path))
    control.add("base", [], facts.facts)
    runtime_facts = _projection_runtime_facts(mode=mode, project_models=project_models)
    if runtime_facts:
        control.add("base", [], runtime_facts)
    if facts.python_precomputed_facts:
        control.add("base", [], facts.python_precomputed_facts)

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
                if horizon_parts_builder is None:
                    ground_parts = _default_horizon_parts(
                        horizon,
                        initial_step_program=initial_step_program,
                        initial_seed_program=initial_seed_program,
                    )
                else:
                    ground_parts = horizon_parts_builder(horizon)
                ground_elapsed, grounding_parts = _ground_program_parts(
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
                total_grounding += ground_elapsed
                _report(
                    progress_callback,
                    f"Grounding progress: horizon {horizon} finished after {ground_elapsed:.3f}s.",
                )

                if horizon < effective_solve_start:
                    horizon += 1
                    continue

                query_symbol = clingo.Function("query", [clingo.Number(horizon)])
                control.assign_external(query_symbol, True)
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
                try:
                    if not solving_started:
                        _report(progress_callback, "Step 3: solving started.")
                        solving_started = True
                    _report(progress_callback, f"Solving: horizon {horizon}...")
                    start = perf_counter()

                    def _solve() -> None:
                        while True:
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
                                    start = perf_counter()
                                    shown_symbols = tuple(model.symbols(shown=True))
                                    shown_symbols_sec += perf_counter() - start
                                    start = perf_counter()
                                    tool_sequence_key, workflow_key = extract_canonical_workflow_keys(
                                        shown_symbols,
                                        tool_input_signatures,
                                    )
                                    workflow_signature_key_sec += perf_counter() - start
                                    workflow_length = workflow_signature_length(workflow_key)
                                    in_length_window = (
                                        config.solution_length_min
                                        <= workflow_length
                                        <= config.solution_length_max
                                    )
                                    if diagnostic_counts_enabled:
                                        if tool_sequence_key not in seen_tool_sequence_keys:
                                            seen_tool_sequence_keys.add(tool_sequence_key)
                                        if workflow_key not in seen_unique_keys:
                                            seen_unique_keys.add(workflow_key)
                                            unique_workflows_seen += 1
                                    store_raw = (
                                        in_length_window
                                        and
                                        capture_raw_models
                                        and len(raw_collected) < config.solutions
                                    )
                                    store_unique = (
                                        in_length_window
                                        and
                                        workflow_key not in stored_unique_keys
                                        and len(unique_collected) < config.solutions
                                    )
                                    if store_raw:
                                        raw_collected.append(shown_symbols)
                                        models_stored += 1
                                    if store_unique:
                                        start = perf_counter()
                                        canonical_shown = canonicalize_shown_symbols(
                                            shown_symbols,
                                            tool_input_signatures,
                                        )
                                        canonicalization_sec += perf_counter() - start
                                        stored_unique_keys.add(workflow_key)
                                        stored_tool_sequence_keys.add(tool_sequence_key)
                                        unique_collected.append(canonical_shown)
                                        unique_workflows_stored += 1
                                    elif (
                                        solve_all_horizons
                                        and len(unique_collected) >= config.solutions
                                    ):
                                        model_callback_sec += perf_counter() - callback_start
                                        break
                                    if not solve_all_horizons and len(unique_collected) >= config.solutions:
                                        model_callback_sec += perf_counter() - callback_start
                                        break
                                    model_callback_sec += perf_counter() - callback_start
                                break

                    _run_interruptible(_solve, is_interrupted)
                    solve_elapsed = perf_counter() - start
                    total_solving += solve_elapsed
                finally:
                    # Permanently retire the external: clingo can now
                    # garbage-collect all check(t)/step_query(t) rules whose
                    # bodies referenced query(t).
                    control.release_external(query_symbol)
                    control.cleanup()

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
                    grounding_parts=grounding_parts,
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
                        seen_tool_sequence_count=len(seen_tool_sequence_keys),
                        stored_tool_sequence_count=len(stored_tool_sequence_keys),
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


def _solve_single_shot_with_programs(
    config: SnakeConfig,
    facts: FactBundle,
    program_paths: tuple[Path, ...],
    *,
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
                            )
                            workflow_signature_key_sec += perf_counter() - sample_start
                            workflow_length = workflow_signature_length(workflow_key)
                            in_length_window = (
                                config.solution_length_min
                                <= workflow_length
                                <= config.solution_length_max
                            )
                            if diagnostic_counts_enabled:
                                if tool_sequence_key not in seen_tool_sequence_keys:
                                    seen_tool_sequence_keys.add(tool_sequence_key)
                                if workflow_key not in seen_unique_keys:
                                    seen_unique_keys.add(workflow_key)
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
                                workflow_key not in stored_unique_keys
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
                                )
                                canonicalization_sec += perf_counter() - sample_start
                                stored_unique_keys.add(workflow_key)
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
                        )
                        workflow_signature_key_sec += perf_counter() - sample_start
                        workflow_length = workflow_signature_length(workflow_key)
                        in_length_window = (
                            config.solution_length_min
                            <= workflow_length
                            <= config.solution_length_max
                        )
                        if diagnostic_counts_enabled:
                            if tool_sequence_key not in seen_tool_sequence_keys:
                                seen_tool_sequence_keys.add(tool_sequence_key)
                            if workflow_key not in seen_unique_keys:
                                seen_unique_keys.add(workflow_key)
                                unique_workflows_seen += 1

                        store_raw = (
                            in_length_window
                            and capture_raw_models
                            and len(raw_solutions) < config.solutions
                        )
                        store_unique = (
                            in_length_window
                            and workflow_key not in stored_unique_keys
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
                            )
                            canonicalization_sec += perf_counter() - sample_start
                            unique_workflows_stored += 1
                            stored_unique_keys.add(workflow_key)
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



def solve_single_shot(
    config: SnakeConfig,
    facts: FactBundle,
    *,
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


def solve_multi_shot(
    config: SnakeConfig,
    facts: FactBundle,
    *,
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



def solve_multi_shot_compressed_candidate(
    config: SnakeConfig,
    facts: FactBundle,
    *,
    progress_callback: ProgressCallback = None,
    base_grounding_callback: BaseGroundingCallback = None,
    horizon_record_callback: HorizonRecordCallback = None,
    capture_raw_models: bool = False,
    diagnostic_counts_enabled: bool = True,
    parallel_mode: str | None = None,
    project_models: bool = False,
) -> SolveOutput:
    """Solve using the compressed-candidate multi-shot encoding."""

    return _solve_multi_shot_with_programs(
        config,
        facts,
        _multi_shot_compressed_candidate_program_paths(),
        mode="multi-shot-compressed-candidate",
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
        parallel_mode=parallel_mode,
        project_models=project_models,
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
    """Ground the compressed-candidate multi-shot encoding without solving."""

    control = _make_grounding_control()
    for program_path in _multi_shot_compressed_candidate_program_paths():
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
