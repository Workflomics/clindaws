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
from .workflow import reconstruct_solution


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENCODINGS_ROOT = PROJECT_ROOT / "encodings"
ProgressCallback = Callable[[str], None] | None
BaseGroundingCallback = Callable[[float, float], None] | None
HorizonRecordCallback = Callable[[HorizonRecord], None] | None


SINGLE_SHOT_OVERLAY_PREFIX = """
#show tool_at_time/2.
#show ape_bind/3.
#show ape_holds_dim/3.
#show ape_goal_out/3.

ape_bind(T, Port, WF) :- occurs(T, bind(_, Port, WF)).
ape_holds_dim(WF, V, Cat) :- holds(0, avail(WF)), holds(0, dim(WF, V, Cat)).
ape_holds_dim(out(T, Tool, Port), V, Cat) :-
    occurs(T, output(Tool, _, Port)),
    holds(T, dim(out(T, Tool, Port), V, Cat)).

goal_dim_match_at(T, GoalID, WF, GoalV, Cat) :-
    holds(T, goal_time(T)),
    goal_output(GoalID, GoalV, Cat),
    holds(T, avail(WF)),
    holds(T, dim(WF, ActualV, Cat)),
    compatible(ActualV, GoalV).

goal_dim_missing_at(T, GoalID, WF) :-
    holds(T, goal_time(T)),
    goal_output(GoalID, GoalV, Cat),
    holds(T, avail(WF)),
    not goal_dim_match_at(T, GoalID, WF, GoalV, Cat).

ape_goal_out(T, GoalID, WF) :-
    holds(T, goal_time(T)),
    goal_output(GoalID, _, _),
    holds(T, avail(WF)),
    goal_dim_match_at(T, GoalID, WF, _, _),
    not goal_dim_missing_at(T, GoalID, WF).
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


def _single_shot_program_paths() -> tuple[Path, ...]:
    base = ENCODINGS_ROOT / "single_shot"
    return (
        base / "show.lp",
        base / "pre_compute.lp",
        base / "propagation.lp",
        base / "tool_choice.lp",
        base / "output_production.lp",
        base / "reachability.lp",
        base / "goal.lp",
        base / "usefulness.lp",
        base / "constraints.lp",
        base / "tool_taxonomy_logic.lp",
        base / "user_constraints.lp",
        base / "plan_constraints.lp",
        base / "temporal_constraint.lp",
        base / "tool_inclusion_constraints.lp",
        base / "tool_repetition_constraints.lp",
        base / "input_usage_constraints.lp",
    )


def _single_shot_opt_program_paths() -> tuple[Path, ...]:
    base = ENCODINGS_ROOT / "single_shot_opt"
    return (
        base / "show.lp",
        base / "pre_compute.lp",
        base / "propagation.lp",
        base / "tool_choice.lp",
        base / "output_production.lp",
        base / "reachability.lp",
        base / "goal.lp",
        base / "usefulness.lp",
        base / "constraints.lp",
        base / "tool_taxonomy_logic.lp",
        base / "user_constraints.lp",
        base / "plan_constraints.lp",
        base / "temporal_constraint.lp",
        base / "tool_inclusion_constraints.lp",
        base / "tool_repetition_constraints.lp",
        base / "input_usage_constraints.lp",
    )


def _single_shot_lazy_program_paths() -> tuple[Path, ...]:
    base = ENCODINGS_ROOT / "single_shot_lazy"
    return (
        base / "show.lp",
        base / "pre_compute.lp",
        base / "propagation.lp",
        base / "tool_choice.lp",
        base / "output_production.lp",
        base / "reachability.lp",
        base / "goal.lp",
        base / "usefulness.lp",
        base / "constraints.lp",
        base / "tool_taxonomy_logic.lp",
        base / "user_constraints.lp",
        base / "plan_constraints.lp",
        base / "temporal_constraint.lp",
        base / "tool_inclusion_constraints.lp",
        base / "tool_repetition_constraints.lp",
        base / "input_usage_constraints.lp",
    )


def _multi_shot_program_paths() -> tuple[Path, ...]:
    base = ENCODINGS_ROOT / "multi_shot"
    return (
        base / "base.lp",
        base / "step.lp",
        base / "check.lp",
        base / "ape_extract.lp",
        base / "tool_inclusion.lp",
        base / "tool_dependency.lp",
        base / "temporal_constraint.lp",
        base / "input_usage.lp",
        base / "output_usage.lp",
    )


def _grounding_opt_program_paths() -> tuple[Path, ...]:
    base = ENCODINGS_ROOT / "multi_shot_opt"
    return (
        base / "base.lp",
        base / "step.lp",
        base / "check.lp",
        base / "ape_extract.lp",
        base / "tool_inclusion.lp",
        base / "input_usage.lp",
        base / "output_usage.lp",
    )


def _lazy_program_paths() -> tuple[Path, ...]:
    base = ENCODINGS_ROOT / "multi_shot_lazy"
    return (
        base / "base.lp",
        base / "step_initial.lp",
        base / "step.lp",
        base / "check.lp",
        base / "ape_extract.lp",
        base / "tool_inclusion.lp",
        base / "input_usage.lp",
        base / "output_usage.lp",
    )


def program_paths_for_mode(mode: str) -> tuple[Path, ...]:
    """Return encoding program paths for a solver mode."""

    if mode == "single-shot":
        return _single_shot_program_paths()
    if mode == "single-shot-opt":
        return _single_shot_opt_program_paths()
    if mode == "single-shot-lazy":
        return _single_shot_lazy_program_paths()
    if mode == "multi-shot":
        return _multi_shot_program_paths()
    if mode == "multi-shot-opt":
        return _grounding_opt_program_paths()
    if mode == "multi-shot-lazy":
        return _lazy_program_paths()
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


def _solve_multi_shot_with_programs(
    config: SnakeConfig,
    facts: FactBundle,
    program_paths: tuple[Path, ...],
    *,
    progress_callback: ProgressCallback = None,
    base_grounding_callback: BaseGroundingCallback = None,
    horizon_record_callback: HorizonRecordCallback = None,
    initial_step_program: str | None = None,
    solve_all_horizons: bool = False,
) -> SolveOutput:
    control = clingo.Control(["0", "--warn=none"])
    for program_path in program_paths:
        control.load(str(program_path))
    control.add("base", [], facts.facts)

    total_grounding = 0.0
    total_solving = 0.0
    base_grounding_peak_rss_mb = 0.0
    base_grounding_sec = 0.0
    raw_collected: list[tuple[clingo.Symbol, ...]] = []
    unique_collected: list[tuple[clingo.Symbol, ...]] = []
    stored_unique_keys: set[tuple[object, ...]] = set()
    horizon_records: list[HorizonRecord] = []
    solving_started = False
    tool_labels = dict(facts.tool_labels)

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

            horizon = config.solution_length_min
            while horizon <= config.solution_length_max:
                if not solve_all_horizons and len(unique_collected) >= config.solutions:
                    break

                _report(progress_callback, f"Grounding: horizon {horizon}...")
                start = perf_counter()
                ground_parts = [("check", [clingo.Number(horizon)])]
                if initial_step_program is not None and horizon == config.solution_length_min:
                    ground_parts.insert(0, (initial_step_program, [clingo.Number(horizon)]))
                else:
                    ground_parts.insert(0, ("step", [clingo.Number(horizon)]))
                _run_interruptible(
                    lambda parts=tuple(ground_parts): control.ground(list(parts)),
                    is_interrupted,
                )
                ground_elapsed = perf_counter() - start
                total_grounding += ground_elapsed
                _report(
                    progress_callback,
                    f"Grounding progress: horizon {horizon} finished after {ground_elapsed:.3f}s.",
                )

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
                        with control.solve(yield_=True) as handle:
                            for model in handle:
                                nonlocal models_seen, models_stored, unique_workflows_seen, unique_workflows_stored
                                models_seen += 1
                                shown = tuple(model.symbols(shown=True))
                                solution = reconstruct_solution(0, shown, tool_labels)
                                canonical_key = solution.canonical_key
                                if canonical_key not in seen_unique_keys:
                                    seen_unique_keys.add(canonical_key)
                                    unique_workflows_seen += 1
                                if len(raw_collected) < config.solutions:
                                    raw_collected.append(shown)
                                    models_stored += 1
                                if canonical_key not in stored_unique_keys and len(unique_collected) < config.solutions:
                                    stored_unique_keys.add(canonical_key)
                                    unique_collected.append(shown)
                                    unique_workflows_stored += 1
                                elif solve_all_horizons and len(raw_collected) >= config.solutions and len(unique_collected) >= config.solutions:
                                    break
                                if not solve_all_horizons and len(unique_collected) >= config.solutions:
                                    break

                    _run_interruptible(_solve, is_interrupted)
                    solve_elapsed = perf_counter() - start
                    total_solving += solve_elapsed
                finally:
                    control.assign_external(query_symbol, False)

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
                    f"Solving progress: horizon {horizon} finished after {solve_elapsed:.3f}s, "
                    f"raw models seen={models_seen}, raw models stored={models_stored}, "
                    f"unique workflows seen={unique_workflows_seen}, "
                    f"unique workflows stored={unique_workflows_stored}, "
                    f"satisfiable={'yes' if models_seen > 0 else 'no'}.",
                )
                if unique_collected and not solve_all_horizons:
                    break
                horizon += 1
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
    tool_labels = dict(facts.tool_labels)

    horizon = config.solution_length_min
    while horizon <= config.solution_length_max:
        if not solve_all_horizons and len(unique_solutions) >= config.solutions:
            break

        control = clingo.Control(["0", "--warn=none"])
        for program_path in program_paths:
            control.load(str(program_path))
        control.add("base", [], facts.facts)
        control.add("base", [], f"time(1..{horizon}).\n")
        control.add("base", [], _single_shot_overlay(horizon))

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
                            shown = tuple(model.symbols(shown=True))
                            solution = reconstruct_solution(0, shown, tool_labels)
                            canonical_key = solution.canonical_key
                            if canonical_key not in seen_unique_keys:
                                seen_unique_keys.add(canonical_key)
                                unique_workflows_seen += 1
                            if len(raw_solutions) < config.solutions:
                                raw_solutions.append(shown)
                                models_stored += 1
                            if canonical_key not in stored_unique_keys and len(unique_solutions) < config.solutions:
                                stored_unique_keys.add(canonical_key)
                                unique_solutions.append(shown)
                                unique_workflows_stored += 1
                            elif solve_all_horizons and len(raw_solutions) >= config.solutions and len(unique_solutions) >= config.solutions:
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
                    f"unique workflows seen={unique_workflows_seen}, "
                    f"unique workflows stored={unique_workflows_stored}, "
                    f"satisfiable={'yes' if models_seen > 0 else 'no'}.",
                )
        finally:
            control.cleanup()

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


def solve_single_shot(
    config: SnakeConfig,
    facts: FactBundle,
    *,
    progress_callback: ProgressCallback = None,
    base_grounding_callback: BaseGroundingCallback = None,
    horizon_record_callback: HorizonRecordCallback = None,
) -> SolveOutput:
    """Solve using the legacy single-shot encoding by iterating horizons."""

    return _solve_single_shot_with_programs(
        config,
        facts,
        _single_shot_program_paths(),
        progress_callback=progress_callback,
        base_grounding_callback=base_grounding_callback,
        horizon_record_callback=horizon_record_callback,
        solve_all_horizons=True,
    )


def solve_single_shot_opt(
    config: SnakeConfig,
    facts: FactBundle,
    *,
    progress_callback: ProgressCallback = None,
    base_grounding_callback: BaseGroundingCallback = None,
    horizon_record_callback: HorizonRecordCallback = None,
) -> SolveOutput:
    """Solve using the optimized candidate single-shot encoding."""

    return _solve_single_shot_with_programs(
        config,
        facts,
        _single_shot_opt_program_paths(),
        progress_callback=progress_callback,
        base_grounding_callback=base_grounding_callback,
        horizon_record_callback=horizon_record_callback,
        solve_all_horizons=True,
    )


def solve_single_shot_lazy(
    config: SnakeConfig,
    facts: FactBundle,
    *,
    progress_callback: ProgressCallback = None,
    base_grounding_callback: BaseGroundingCallback = None,
    horizon_record_callback: HorizonRecordCallback = None,
) -> SolveOutput:
    """Solve using the lazy candidate single-shot encoding."""

    return _solve_single_shot_with_programs(
        config,
        facts,
        _single_shot_lazy_program_paths(),
        progress_callback=progress_callback,
        base_grounding_callback=base_grounding_callback,
        horizon_record_callback=horizon_record_callback,
        solve_all_horizons=True,
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
                for horizon in range(config.solution_length_min, config.solution_length_max + 1):
                    _report(progress_callback, f"Grounding: horizon {horizon}...")
                    start = perf_counter()
                    ground_parts = [("check", [clingo.Number(horizon)])]
                    if initial_step_program is not None and horizon == config.solution_length_min:
                        ground_parts.insert(0, (initial_step_program, [clingo.Number(horizon)]))
                    else:
                        ground_parts.insert(0, ("step", [clingo.Number(horizon)]))
                    _run_interruptible(
                        lambda parts=tuple(ground_parts): control.ground(list(parts)),
                        is_interrupted,
                    )
                    elapsed = perf_counter() - start
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
) -> SolveOutput:
    """Solve using the multi-shot encoding."""
    return _solve_multi_shot_with_programs(
        config,
        facts,
        _multi_shot_program_paths(),
        progress_callback=progress_callback,
        base_grounding_callback=base_grounding_callback,
        horizon_record_callback=horizon_record_callback,
        solve_all_horizons=True,
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

    control = clingo.Control(["0", "--warn=none"])
    for program_path in _multi_shot_program_paths():
        control.load(str(program_path))
    control.add("base", [], facts.facts)
    return _ground_multi_shot_control(
        control,
        config,
        stage=stage,
        progress_callback=progress_callback,
        base_grounding_callback=base_grounding_callback,
        horizon_record_callback=horizon_record_callback,
    )


def solve_multi_shot_grounding_opt(
    config: SnakeConfig,
    facts: FactBundle,
    *,
    progress_callback: ProgressCallback = None,
    base_grounding_callback: BaseGroundingCallback = None,
    horizon_record_callback: HorizonRecordCallback = None,
) -> SolveOutput:
    """Solve using the grounding-optimised multi-shot encoding.

    Uses pre-expanded ``tool_candidate`` / ``candidate_in`` / ``candidate_out``
    facts so the per-step choice rules reduce to a single aggregate constraint,
    eliminating the ``compatible/2`` join from eligibility checks.
    """
    return _solve_multi_shot_with_programs(
        config,
        facts,
        _grounding_opt_program_paths(),
        progress_callback=progress_callback,
        base_grounding_callback=base_grounding_callback,
        horizon_record_callback=horizon_record_callback,
        solve_all_horizons=True,
    )


def solve_multi_shot_lazy(
    config: SnakeConfig,
    facts: FactBundle,
    *,
    progress_callback: ProgressCallback = None,
    base_grounding_callback: BaseGroundingCallback = None,
    horizon_record_callback: HorizonRecordCallback = None,
) -> SolveOutput:
    """Solve using the lazy multi-shot encoding."""

    return _solve_multi_shot_with_programs(
        config,
        facts,
        _lazy_program_paths(),
        progress_callback=progress_callback,
        base_grounding_callback=base_grounding_callback,
        horizon_record_callback=horizon_record_callback,
        initial_step_program="step_initial",
        solve_all_horizons=True,
    )


def ground_multi_shot_grounding_opt(
    config: SnakeConfig,
    facts: FactBundle,
    *,
    stage: str = "base",
    progress_callback: ProgressCallback = None,
    base_grounding_callback: BaseGroundingCallback = None,
    horizon_record_callback: HorizonRecordCallback = None,
) -> GroundingOutput:
    """Ground the grounding-optimised multi-shot encoding without solving."""

    control = clingo.Control(["0", "--warn=none"])
    for program_path in _grounding_opt_program_paths():
        control.load(str(program_path))
    control.add("base", [], facts.facts)
    return _ground_multi_shot_control(
        control,
        config,
        stage=stage,
        progress_callback=progress_callback,
        base_grounding_callback=base_grounding_callback,
        horizon_record_callback=horizon_record_callback,
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

    control = clingo.Control(["0", "--warn=none"])
    for program_path in _lazy_program_paths():
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
    )
