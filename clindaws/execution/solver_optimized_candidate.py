"""Optimized-candidate-specific solver helpers: assumptions, feasibility, blocking."""

from __future__ import annotations

import os
from time import perf_counter
from typing import Callable

import clingo

from clindaws.core.models import FactBundle
from clindaws.execution.solver_utils import _run_interruptible


def _optimized_full_solve_horizon_parts(
    horizon: int,
) -> tuple[tuple[str, tuple[clingo.Symbol, ...]], ...]:
    return (
        ("step_query", (clingo.Number(horizon),)),
        ("check", (clingo.Number(horizon),)),
        ("check_usage", (clingo.Number(horizon),)),
    )


def _optimized_certificate_horizon_parts(
    horizon: int,
) -> tuple[tuple[str, tuple[clingo.Symbol, ...]], ...]:
    return (("certificate_check", (clingo.Number(horizon),)),)


def _optimized_exact_incremental_horizon_parts(
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
    parts.append(("constraint_step", (clingo.Number(horizon),)))
    return tuple(parts)


def _smart_expansion_enabled(facts: FactBundle) -> bool:
    return facts.internal_solver_mode in {
        "multi-shot-optimized-candidate",
        "multi-shot-compressed-candidate",
    }


def _optimized_query_assumptions(
    *,
    horizon: int,
    grounded_horizon: int,
    query_active: bool,
    certificate_active: bool = False,
    possible_enforcement: dict[str, bool] | None = None,
) -> list[tuple[clingo.Symbol, bool]]:
    """Build a complete assumption set for optimized query activation."""

    assumptions: list[tuple[clingo.Symbol, bool]] = []
    enforcement = possible_enforcement or {
        "goal": False,
        "input_support": False,
        "forced_binding": False,
        "must_run_tool": False,
        "must_run_candidate": False,
    }
    for current_horizon in range(1, grounded_horizon + 1):
        assumptions.append(
            (
                clingo.Function("query_assumption", [clingo.Number(current_horizon)]),
                query_active and current_horizon == horizon,
            )
        )
        assumptions.append(
            (
                clingo.Function("certificate_query_assumption", [clingo.Number(current_horizon)]),
                certificate_active and current_horizon == horizon,
            )
        )
        assumptions.append(
            (
                clingo.Function("possible_query_assumption", [clingo.Number(current_horizon)]),
                (not query_active) and (not certificate_active) and current_horizon == horizon,
            )
        )
        assumptions.append(
            (
                clingo.Function("possible_enforce_goal_assumption", [clingo.Number(current_horizon)]),
                (not query_active) and (not certificate_active) and current_horizon == horizon and enforcement["goal"],
            )
        )
        assumptions.append(
            (
                clingo.Function("possible_enforce_input_support_assumption", [clingo.Number(current_horizon)]),
                (not query_active) and (not certificate_active) and current_horizon == horizon and enforcement["input_support"],
            )
        )
        assumptions.append(
            (
                clingo.Function("possible_enforce_forced_binding_assumption", [clingo.Number(current_horizon)]),
                (not query_active) and (not certificate_active) and current_horizon == horizon and enforcement["forced_binding"],
            )
        )
        assumptions.append(
            (
                clingo.Function("possible_enforce_must_run_tool_assumption", [clingo.Number(current_horizon)]),
                (not query_active) and (not certificate_active) and current_horizon == horizon and enforcement["must_run_tool"],
            )
        )
        assumptions.append(
            (
                clingo.Function("possible_enforce_must_run_candidate_assumption", [clingo.Number(current_horizon)]),
                (not query_active) and (not certificate_active) and current_horizon == horizon and enforcement["must_run_candidate"],
            )
        )
    return assumptions


def _optimized_structural_probe_horizons(facts: FactBundle) -> tuple[int, ...]:
    smart_expansion = facts.backend_stats.get("smart_expansion", {})
    probe_horizons = smart_expansion.get("structural_probe_horizons", [])
    return tuple(int(horizon) for horizon in probe_horizons)


def _optimized_model_blocking_mode(facts: FactBundle) -> str | None:
    mode = os.environ.get("CLINDAWS_OPTIMIZED_MODEL_BLOCKING", "").strip()
    return mode or None


def _exact_candidate_sequence_symbols(model: clingo.Model) -> tuple[clingo.Symbol, ...]:
    """Return the exact selected candidate atoms for one optimized model."""

    candidate_symbols: list[clingo.Symbol] = []
    for symbol in model.symbols(atoms=True):
        if symbol.type != clingo.SymbolType.Function or symbol.name != "occurs":
            continue
        if len(symbol.arguments) != 2:
            continue
        action = symbol.arguments[1]
        if (
            action.type == clingo.SymbolType.Function
            and action.name == "use_dynamic_candidate"
            and len(action.arguments) == 2
        ):
            candidate_symbols.append(symbol)
    return tuple(sorted(candidate_symbols, key=str))


def _add_exact_model_blocking_clause(
    control: clingo.Control,
    *,
    symbols: tuple[clingo.Symbol, ...],
) -> bool:
    """Block the exact conjunction of selected candidate atoms in future solves."""

    if not symbols:
        return False
    body_literals: list[int] = []
    for symbol in symbols:
        symbolic_atom = control.symbolic_atoms[symbol]
        if symbolic_atom is None:
            continue
        body_literals.append(symbolic_atom.literal)
    if not body_literals:
        return False
    with control.backend() as backend:
        backend.add_rule([], body_literals)
    return True


def _run_feasibility_precheck(
    *,
    facts: FactBundle,
    horizon: int,
) -> tuple[bool, float, tuple[tuple[str, float], ...], str | None, tuple[str, ...]]:
    """Run translator-side structural feasibility checks for one optimized horizon."""

    start = perf_counter()
    smart_expansion = facts.backend_stats.get("smart_expansion", {})
    goal_support_counts_by_horizon = smart_expansion.get("goal_support_candidate_counts_by_horizon", {})
    goal_support_goal_counts_by_horizon = smart_expansion.get("goal_support_goal_counts_by_horizon", {})
    goal_support_missing_goals_by_horizon = smart_expansion.get("goal_support_missing_goals_by_horizon", {})
    supportable_candidate_counts_by_horizon = smart_expansion.get("supportable_candidate_counts_by_horizon", {})
    unsupported_input_counts_by_horizon = smart_expansion.get("unsupported_input_counts_by_horizon", {})
    unsupported_input_samples_by_horizon = smart_expansion.get("unsupported_input_samples_by_horizon", {})
    supportable_goal_counts_by_horizon = smart_expansion.get("supportable_goal_counts_by_horizon", {})
    supportable_missing_goals_by_horizon = smart_expansion.get("supportable_missing_goals_by_horizon", {})
    forced_associations_global = int(smart_expansion.get("forced_associations_global", 0))
    must_run_tools_global = int(smart_expansion.get("must_run_tools_global", 0))
    must_run_candidates_global = int(smart_expansion.get("must_run_candidates_global", 0))
    if int(goal_support_counts_by_horizon.get(horizon, 0)) == 0:
        return False, perf_counter() - start, (), "goal_support", ()
    structural_stage_start = perf_counter()
    missing_goals = tuple(
        str(goal_id)
        for goal_id in goal_support_missing_goals_by_horizon.get(horizon, ())
    )
    structural_stage_elapsed = perf_counter() - structural_stage_start
    stage_timings: list[tuple[str, float]] = [("goal_only_structural", structural_stage_elapsed)]
    if missing_goals or int(goal_support_goal_counts_by_horizon.get(horizon, 0)) == 0:
        return (
            False,
            perf_counter() - start,
            tuple(stage_timings),
            "goal_only",
            missing_goals,
        )
    input_support_stage_start = perf_counter()
    unsupported_input_count = int(unsupported_input_counts_by_horizon.get(horizon, 0))
    supportable_missing_goals = tuple(
        str(goal_id)
        for goal_id in supportable_missing_goals_by_horizon.get(horizon, ())
    )
    unsupported_input_samples = tuple(
        str(value)
        for value in unsupported_input_samples_by_horizon.get(horizon, ())
    )
    input_support_stage_elapsed = perf_counter() - input_support_stage_start
    stage_timings.append(("input_support_structural", input_support_stage_elapsed))
    if (
        unsupported_input_count > 0
        or int(supportable_candidate_counts_by_horizon.get(horizon, 0)) < int(goal_support_counts_by_horizon.get(horizon, 0))
        or supportable_missing_goals
        or int(supportable_goal_counts_by_horizon.get(horizon, 0)) == 0
    ):
        return (
            False,
            perf_counter() - start,
            tuple(stage_timings),
            "input_support",
            tuple((*supportable_missing_goals, *unsupported_input_samples)),
        )
    if forced_associations_global > 0:
        stage_timings.append(("forced_binding_structural", 0.0))
    if must_run_tools_global > 0 or must_run_candidates_global > 0:
        stage_timings.append(("must_run_structural", 0.0))
    return True, perf_counter() - start, tuple(stage_timings), None, ()


def _run_optimized_exact_certificate(
    control: clingo.Control,
    *,
    horizon: int,
    grounded_horizon: int,
    is_interrupted: Callable[[], bool],
) -> tuple[bool, float]:
    """Check whether the exact horizon is satisfiable before full query grounding."""

    start = perf_counter()
    certificate_ok = False

    def _solve() -> None:
        nonlocal certificate_ok
        assumptions = _optimized_query_assumptions(
            horizon=horizon,
            grounded_horizon=grounded_horizon,
            query_active=False,
            certificate_active=True,
        )
        with control.solve(yield_=True, assumptions=assumptions) as handle:
            for _model in handle:
                certificate_ok = True
                break

    _run_interruptible(_solve, is_interrupted)
    return certificate_ok, perf_counter() - start
