"""Clingo solver orchestration."""

from __future__ import annotations

import os
import signal
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Callable, Iterator

import clingo

from .models import FactBundle, HorizonRecord, SnakeConfig
from .runtime_stats import current_peak_rss_mb
from .workflow import (
    canonicalize_shown_symbols,
    extract_workflow_signature_key,
    workflow_signature_length,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENCODINGS_ROOT = PROJECT_ROOT / "encodings"
ProgressCallback = Callable[[str], None] | None
BaseGroundingCallback = Callable[[float, float], None] | None
HorizonRecordCallback = Callable[[HorizonRecord], None] | None


SINGLE_SHOT_OVERLAY_PREFIX = """
#show tool_at_time/2.
#show ape_bind/3.
#show ape_holds_dim/3.

ape_bind(T, Port, WF) :- occurs(T, bind(_, Port, WF)).
ape_holds_dim(WF, V, Cat) :- holds(0, avail(WF)), holds(0, dim(WF, V, Cat)).
ape_holds_dim(out(T, Tool, Port), V, Cat) :-
    occurs(T, output(Tool, _, Port)),
    holds(T, dim(out(T, Tool, Port), V, Cat)).
"""


def _single_shot_overlay(horizon: int) -> str:
    return (
        SINGLE_SHOT_OVERLAY_PREFIX
        + f"""
:- not holds({horizon}, goal).
:- {horizon} > 1, holds({horizon - 1}, goal).
:- time(T), 2 {{ occurs(T, run(_)) }}.
"""
    )


def _make_solve_control(
    *,
    parallel_mode: str | None = None,
    project_models: bool = False,
) -> clingo.Control:
    """Create a clingo control for full model enumeration during solving."""

    args = ["0", "--warn=none"]
    if project_models:
        args.append("--project")
    if parallel_mode:
        args.append(f"--parallel-mode={parallel_mode}")
    return clingo.Control(args)


def _make_grounding_control() -> clingo.Control:
    """Create a clingo control for grounding-only operations."""

    return clingo.Control(["0", "--warn=none"])


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


def _single_shot_program_paths(*, python_precompute_direct: bool = False) -> tuple[Path, ...]:
    base = ENCODINGS_ROOT / "single_shot"
    return (
        base / "show.lp",
        base / ("pre_compute_python.lp" if python_precompute_direct else "pre_compute.lp"),
        base / "propagation.lp",
        base / ("tool_choice_python.lp" if python_precompute_direct else "tool_choice.lp"),
        base / "output_production.lp",
        base / "reachability.lp",
        base / "goal.lp",
        base / "usefulness.lp",
        base / "constraints.lp",
        base / "tool_taxonomy_logic.lp",
        base / "user_constraints.lp",
        base / "generated_constraints.lp",
        base / "plan_constraints.lp",
        base / "temporal_constraint.lp",
        base / "tool_inclusion_constraints.lp",
        base / "input_usage_constraints.lp",
    )


def _multi_shot_program_paths(*, python_precompute_direct: bool = False) -> tuple[Path, ...]:
    base = ENCODINGS_ROOT / "multi_shot"
    return (
        base / ("base_python.lp" if python_precompute_direct else "base.lp"),
        base / ("init_python.lp" if python_precompute_direct else "init.lp"),
        base / ("step_python.lp" if python_precompute_direct else "step.lp"),
        base / "constraints.lp",
        base / ("check_python.lp" if python_precompute_direct else "check.lp"),
        base / "ape_extract.lp",
        base / "tool_inclusion.lp",
        base / "tool_dependency.lp",
        base / "temporal_constraint.lp",
        base / "input_usage.lp",
        base / "output_usage.lp",
    )



def _multi_shot_lazy_program_paths() -> tuple[Path, ...]:
    base = ENCODINGS_ROOT / "multi_shot_lazy"
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
    python_precompute_direct: bool = False,
) -> tuple[Path, ...]:
    """Return encoding program paths for a solver mode."""

    if mode == "single-shot":
        return _single_shot_program_paths(python_precompute_direct=python_precompute_direct)
    if mode == "multi-shot":
        return _multi_shot_program_paths(python_precompute_direct=python_precompute_direct)
    if mode == "multi-shot-lazy":
        return _multi_shot_lazy_program_paths()
    raise ValueError(f"Unsupported mode: {mode}")


def _report(progress_callback: ProgressCallback, message: str) -> None:
    if progress_callback is not None:
        progress_callback(message)


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


def _lazy_horizon_parts(
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


def _lazy_prefix_horizon_parts(
    horizon: int,
    *,
    initial_step_program: str | None,
    initial_seed_program: str | None,
    include_constraint_step: bool,
) -> tuple[tuple[str, tuple[clingo.Symbol, ...]], ...]:
    """Lean lazy horizon parts for prefix horizons below solution_length.min.

    Prefix horizons still need to build the reachable workflow state for later
    solved horizons, but they do not need query-time validation. Constraint
    summaries only need to be accumulated here when translated constraints are
    present.
    """
    parts: list[tuple[str, tuple[clingo.Symbol, ...]]] = []
    if initial_step_program is not None and horizon == 1:
        parts.append((initial_step_program, (clingo.Number(horizon),)))
    else:
        if initial_seed_program is not None and horizon > 1:
            parts.append((initial_seed_program, (clingo.Number(horizon - 1),)))
        parts.append(("step", (clingo.Number(horizon),)))
    if include_constraint_step:
        parts.append(("constraint_step", (clingo.Number(horizon),)))
    return tuple(parts)


def _multi_shot_horizon_parts(horizon: int) -> tuple[tuple[str, tuple[clingo.Symbol, ...]], ...]:
    parts: list[tuple[str, tuple[clingo.Symbol, ...]]] = []
    if horizon == 1:
        parts.append(("init", ()))
    parts.append(("step", (clingo.Number(horizon),)))
    parts.append(("check", (clingo.Number(horizon),)))
    return tuple(parts)


def _multi_shot_prefix_horizon_parts(
    horizon: int,
) -> tuple[tuple[str, tuple[clingo.Symbol, ...]], ...]:
    """Lean legacy multi-shot horizon parts below the first solvable horizon."""

    parts: list[tuple[str, tuple[clingo.Symbol, ...]]] = []
    if horizon == 1:
        parts.append(("init", ()))
    parts.append(("step", (clingo.Number(horizon),)))
    return tuple(parts)


def _multi_shot_grounding_horizon_parts(
    horizon: int,
) -> tuple[tuple[str, tuple[clingo.Symbol, ...]], ...]:
    parts: list[tuple[str, tuple[clingo.Symbol, ...]]] = []
    if horizon == 1:
        parts.append(("init", ()))
    parts.append(("step", (clingo.Number(horizon),)))
    return tuple(parts)


def _lazy_grounding_horizon_parts(
    horizon: int,
    *,
    initial_step_program: str | None,
    initial_seed_program: str | None,
) -> tuple[tuple[str, tuple[clingo.Symbol, ...]], ...]:
    """Lean horizon parts for grounding-only lazy runs.

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
                if not solve_all_horizons and len(unique_collected) >= config.solutions:
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
                models_seen = 0
                models_stored = 0
                unique_workflows_seen = 0
                unique_workflows_stored = 0
                seen_unique_keys: set[tuple[object, ...]] = set()
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
                                    models_seen += 1
                                    shown_symbols = tuple(model.symbols(shown=True))
                                    workflow_key = extract_workflow_signature_key(shown_symbols)
                                    workflow_length = workflow_signature_length(workflow_key)
                                    in_length_window = (
                                        config.solution_length_min
                                        <= workflow_length
                                        <= config.solution_length_max
                                    )
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
                                    canonical_shown: tuple[clingo.Symbol, ...] | None = None
                                    if store_unique:
                                        canonical_shown = canonicalize_shown_symbols(
                                            shown_symbols,
                                            tool_input_signatures,
                                        )
                                    if store_raw:
                                        raw_collected.append(shown_symbols)
                                        models_stored += 1
                                    if store_unique:
                                        stored_unique_keys.add(workflow_key)
                                        unique_collected.append(canonical_shown if canonical_shown is not None else shown_symbols)
                                        unique_workflows_stored += 1
                                    elif (
                                        solve_all_horizons
                                        and (not capture_raw_models or len(raw_collected) >= config.solutions)
                                        and len(unique_collected) >= config.solutions
                                    ):
                                        break
                                    if not solve_all_horizons and len(unique_collected) >= config.solutions:
                                        break
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
                    satisfiable=models_seen > 0,
                    models_seen=models_seen,
                    models_stored=models_stored,
                    unique_workflows_seen=unique_workflows_seen,
                    unique_workflows_stored=unique_workflows_stored,
                    grounding_parts=grounding_parts,
                )
                horizon_records.append(record)
                if horizon_record_callback is not None:
                    horizon_record_callback(record)
                _report(
                    progress_callback,
                    f"Solving progress: horizon {horizon} finished after {solve_elapsed:.3f}s, "
                    f"raw models seen={models_seen}, raw models stored={models_stored}, "
                    f"unique tool sequences seen={unique_workflows_seen}, "
                    f"unique tool sequences stored={unique_workflows_stored}, "
                    f"satisfiable={'yes' if models_seen > 0 else 'no'}.",
                )
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
        if not solve_all_horizons and len(unique_solutions) >= config.solutions:
            break

        control = _make_solve_control(
            parallel_mode=parallel_mode,
            project_models=project_models,
        )
        for program_path in program_paths:
            control.load(str(program_path))
        control.add("base", [], facts.facts)
        control.add("base", [], f"time(1..{horizon}).\n")
        control.add("base", [], _single_shot_overlay(horizon))

        _horizon_interrupted = False
        try:
            with _interrupt_guard(control) as is_interrupted:
                if horizon == config.solution_length_min:
                    _report(progress_callback, "Step 2: grounding started.")
                _report(progress_callback, f"Grounding: single-shot horizon {horizon}...")
                start = perf_counter()
                _run_interruptible(lambda: control.ground([("base", [])]), is_interrupted)
                ground_elapsed = perf_counter() - start
                total_grounding += ground_elapsed
                if horizon == config.solution_length_min:
                    base_grounding_sec += ground_elapsed
                    base_grounding_peak_rss_mb = current_peak_rss_mb()
                    if base_grounding_callback is not None:
                        base_grounding_callback(base_grounding_sec, base_grounding_peak_rss_mb)
                _report(
                    progress_callback,
                    f"Grounding progress: single-shot horizon {horizon} finished after {ground_elapsed:.3f}s.",
                )

                if not solving_started:
                    _report(progress_callback, "Step 3: solving started.")
                    solving_started = True
                _report(progress_callback, f"Solving: single-shot horizon {horizon}...")
                models_seen = 0
                models_stored = 0
                unique_workflows_seen = 0
                unique_workflows_stored = 0
                seen_unique_keys: set[tuple[object, ...]] = set()
                start = perf_counter()

                def _solve() -> None:
                    with control.solve(yield_=True) as handle:
                        for model in handle:
                            nonlocal models_seen, models_stored, unique_workflows_seen, unique_workflows_stored
                            models_seen += 1
                            shown_symbols = tuple(model.symbols(shown=True))
                            workflow_key = extract_workflow_signature_key(shown_symbols)
                            workflow_length = workflow_signature_length(workflow_key)
                            in_length_window = (
                                config.solution_length_min
                                <= workflow_length
                                <= config.solution_length_max
                            )
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
                            canonical_shown: tuple[clingo.Symbol, ...] | None = None
                            if store_unique:
                                canonical_shown = canonicalize_shown_symbols(
                                    shown_symbols,
                                    tool_input_signatures,
                                )
                            if store_raw:
                                raw_solutions.append(shown_symbols)
                                models_stored += 1
                            if store_unique:
                                stored_unique_keys.add(workflow_key)
                                unique_solutions.append(canonical_shown if canonical_shown is not None else shown_symbols)
                                unique_workflows_stored += 1
                            elif (
                                solve_all_horizons
                                and (not capture_raw_models or len(raw_solutions) >= config.solutions)
                                and len(unique_solutions) >= config.solutions
                            ):
                                break
                            if not solve_all_horizons and len(unique_solutions) >= config.solutions:
                                break

                _run_interruptible(_solve, is_interrupted)
                solve_elapsed = perf_counter() - start
                total_solving += solve_elapsed
                record = HorizonRecord(
                    horizon=horizon,
                    grounding_sec=ground_elapsed,
                    solving_sec=solve_elapsed,
                    peak_rss_mb=current_peak_rss_mb(),
                    satisfiable=models_seen > 0,
                    models_seen=models_seen,
                    models_stored=models_stored,
                    unique_workflows_seen=unique_workflows_seen,
                    unique_workflows_stored=unique_workflows_stored,
                )
                horizon_records.append(record)
                if horizon_record_callback is not None:
                    horizon_record_callback(record)
                _report(
                    progress_callback,
                    f"Solving progress: single-shot horizon {horizon} finished after {solve_elapsed:.3f}s, "
                    f"raw models seen={models_seen}, raw models stored={models_stored}, "
                    f"unique tool sequences seen={unique_workflows_seen}, "
                    f"unique tool sequences stored={unique_workflows_stored}, "
                    f"satisfiable={'yes' if models_seen > 0 else 'no'}.",
                )
        except KeyboardInterrupt:
            _report(progress_callback, "Interrupted/timeout: returning partial results.")
            _horizon_interrupted = True
        finally:
            control.cleanup()

        if _horizon_interrupted:
            break
        if unique_solutions and not solve_all_horizons:
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
    parallel_mode: str | None = None,
    project_models: bool = False,
) -> SolveOutput:
    """Ground once with time(1..max_length) and solve in a single shot."""

    raw_solutions: list[tuple[clingo.Symbol, ...]] = []
    unique_solutions: list[tuple[clingo.Symbol, ...]] = []
    stored_unique_keys: set[tuple[object, ...]] = set()
    tool_input_signatures = dict(facts.tool_input_signatures)
    horizon = config.solution_length_max

    static_overlay = (
        SINGLE_SHOT_OVERLAY_PREFIX
        + ":- time(T), 2 { occurs(T, run(_)) }.\n"
    )

    control = _make_solve_control(
        parallel_mode=parallel_mode,
        project_models=project_models,
    )
    for program_path in program_paths:
        control.load(str(program_path))
    control.add("base", [], facts.facts)
    if facts.python_precomputed_facts:
        control.add("base", [], facts.python_precomputed_facts)
    control.add("base", [], f"time(1..{horizon}).\n")
    control.add("base", [], static_overlay)

    ground_elapsed = 0.0
    solve_elapsed = 0.0
    base_grounding_peak_rss_mb = 0.0
    models_seen = 0
    models_stored = 0
    unique_workflows_seen = 0
    unique_workflows_stored = 0

    try:
        with _interrupt_guard(control) as is_interrupted:
            _report(progress_callback, "Step 2: grounding started.")
            _report(progress_callback, f"Grounding: single-shot (time 1..{horizon})...")
            start = perf_counter()
            _run_interruptible(lambda: control.ground([("base", [])]), is_interrupted)
            ground_elapsed = perf_counter() - start
            base_grounding_peak_rss_mb = current_peak_rss_mb()
            if base_grounding_callback is not None:
                base_grounding_callback(ground_elapsed, base_grounding_peak_rss_mb)
            _report(
                progress_callback,
                f"Grounding progress: single-shot finished after {ground_elapsed:.3f}s.",
            )

            _report(progress_callback, "Step 3: solving started.")
            _report(progress_callback, "Solving: single-shot...")
            seen_unique_keys: set[tuple[object, ...]] = set()
            start = perf_counter()

            def _solve() -> None:
                nonlocal models_seen, models_stored, unique_workflows_seen, unique_workflows_stored
                with control.solve(yield_=True) as handle:
                    for model in handle:
                        models_seen += 1
                        shown_symbols = tuple(model.symbols(shown=True))
                        workflow_key = extract_workflow_signature_key(shown_symbols)
                        workflow_length = workflow_signature_length(workflow_key)
                        in_length_window = (
                            config.solution_length_min
                            <= workflow_length
                            <= config.solution_length_max
                        )
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
                        canonical_shown: tuple[clingo.Symbol, ...] | None = None
                        if store_unique:
                            canonical_shown = canonicalize_shown_symbols(
                                shown_symbols,
                                tool_input_signatures,
                            )
                        if store_raw:
                            raw_solutions.append(shown_symbols)
                            models_stored += 1
                        if store_unique:
                            stored_unique_keys.add(workflow_key)
                            unique_solutions.append(canonical_shown if canonical_shown is not None else shown_symbols)
                            unique_workflows_stored += 1
                        if len(unique_solutions) >= config.solutions:
                            break

            _run_interruptible(_solve, is_interrupted)
            solve_elapsed = perf_counter() - start
            _report(
                progress_callback,
                f"Solving progress: single-shot finished after {solve_elapsed:.3f}s, "
                f"raw models seen={models_seen}, raw models stored={models_stored}, "
                f"unique tool sequences seen={unique_workflows_seen}, "
                f"unique tool sequences stored={unique_workflows_stored}, "
                f"satisfiable={'yes' if models_seen > 0 else 'no'}.",
            )
    except KeyboardInterrupt:
        _report(progress_callback, "Interrupted/timeout: returning partial results.")
    finally:
        control.cleanup()

    record = HorizonRecord(
        horizon=horizon,
        grounding_sec=ground_elapsed,
        solving_sec=solve_elapsed,
        peak_rss_mb=base_grounding_peak_rss_mb,
        satisfiable=models_seen > 0,
        models_seen=models_seen,
        models_stored=models_stored,
        unique_workflows_seen=unique_workflows_seen,
        unique_workflows_stored=unique_workflows_stored,
    )
    if horizon_record_callback is not None:
        horizon_record_callback(record)

    return SolveOutput(
        raw_solutions=tuple(raw_solutions),
        solutions=tuple(unique_solutions),
        base_grounding_peak_rss_mb=base_grounding_peak_rss_mb,
        base_grounding_sec=ground_elapsed,
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
    parallel_mode: str | None = None,
    project_models: bool = False,
) -> SolveOutput:
    """Solve single-shot mode as one grounding over time(1..max_length)."""

    return _solve_single_shot_once(
        config,
        facts,
        _single_shot_program_paths(python_precompute_direct=facts.python_precompute_enabled),
        progress_callback=progress_callback,
        base_grounding_callback=base_grounding_callback,
        horizon_record_callback=horizon_record_callback,
        capture_raw_models=capture_raw_models,
        parallel_mode=parallel_mode,
        project_models=project_models,
    )


def _ground_multi_shot_control(
    control: clingo.Control,
    config: SnakeConfig,
    *,
    stage: str,
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
    parallel_mode: str | None = None,
    project_models: bool = False,
) -> SolveOutput:
    """Solve using the multi-shot encoding."""
    effective_solve_start = max(config.solution_length_min, facts.earliest_solution_step)
    if effective_solve_start > config.solution_length_min:
        _report(
            progress_callback,
            f"Lower-bound solve start: skipping query/check before horizon {effective_solve_start}.",
        )
    return _solve_multi_shot_with_programs(
        config,
        facts,
        _multi_shot_program_paths(python_precompute_direct=facts.python_precompute_enabled),
        progress_callback=progress_callback,
        base_grounding_callback=base_grounding_callback,
        horizon_record_callback=horizon_record_callback,
        solve_all_horizons=False,
        stop_on_solution=False,
        horizon_parts_builder=lambda horizon: (
            _multi_shot_prefix_horizon_parts(horizon)
            if horizon < effective_solve_start
            else _multi_shot_horizon_parts(horizon)
        ),
        capture_raw_models=capture_raw_models,
        solve_start_horizon=effective_solve_start,
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
    for program_path in _multi_shot_program_paths(python_precompute_direct=facts.python_precompute_enabled):
        control.load(str(program_path))
    control.add("base", [], facts.facts)
    if facts.python_precomputed_facts:
        control.add("base", [], facts.python_precomputed_facts)
    return _ground_multi_shot_control(
        control,
        config,
        stage=stage,
        progress_callback=progress_callback,
        base_grounding_callback=base_grounding_callback,
        horizon_record_callback=horizon_record_callback,
        horizon_parts_builder=_multi_shot_grounding_horizon_parts,
    )



def solve_multi_shot_lazy(
    config: SnakeConfig,
    facts: FactBundle,
    *,
    progress_callback: ProgressCallback = None,
    base_grounding_callback: BaseGroundingCallback = None,
    horizon_record_callback: HorizonRecordCallback = None,
    capture_raw_models: bool = False,
    parallel_mode: str | None = None,
    project_models: bool = False,
) -> SolveOutput:
    """Solve using the lazy multi-shot encoding."""

    has_translated_constraints = _has_translated_constraints(facts)
    effective_solve_start = max(config.solution_length_min, facts.earliest_solution_step)

    return _solve_multi_shot_with_programs(
        config,
        facts,
        _multi_shot_lazy_program_paths(),
        progress_callback=progress_callback,
        base_grounding_callback=base_grounding_callback,
        horizon_record_callback=horizon_record_callback,
        initial_step_program="step_initial",
        solve_all_horizons=False,
        stop_on_solution=False,
        horizon_parts_builder=lambda horizon: (
            _lazy_prefix_horizon_parts(
                horizon,
                initial_step_program="step_initial",
                initial_seed_program=None,
                include_constraint_step=has_translated_constraints,
            )
            if horizon < effective_solve_start
            else _lazy_horizon_parts(
                horizon,
                initial_step_program="step_initial",
                initial_seed_program=None,
            )
        ),
        capture_raw_models=capture_raw_models,
        solve_start_horizon=effective_solve_start,
        parallel_mode=parallel_mode,
        project_models=project_models,
    )



def ground_multi_shot_lazy(
    config: SnakeConfig,
    facts: FactBundle,
    *,
    stage: str = "base",
    progress_callback: ProgressCallback = None,
    base_grounding_callback: BaseGroundingCallback = None,
    horizon_record_callback: HorizonRecordCallback = None,
) -> GroundingOutput:
    """Ground the lazy multi-shot encoding without solving."""

    control = _make_grounding_control()
    for program_path in _multi_shot_lazy_program_paths():
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
        horizon_parts_builder=lambda horizon: _lazy_grounding_horizon_parts(
            horizon,
            initial_step_program="step_initial",
            initial_seed_program=None,
        ),
    )
