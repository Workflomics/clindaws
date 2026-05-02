"""Solver output dataclasses and stored-solution accounting helpers."""

from __future__ import annotations

from dataclasses import dataclass

import clingo

from clindaws.core.models import HorizonRecord, SnakeConfig


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


@dataclass(frozen=True)
class _SolvePassMetrics:
    any_model_seen: bool
    models_seen: int
    models_stored: int
    unique_workflows_seen: int
    unique_workflows_stored: int
    model_callback_sec: float
    shown_symbols_sec: float
    workflow_signature_key_sec: float
    canonicalization_sec: float
    clause_constraints_added: int
    seen_tool_sequence_count: int
    stored_tool_sequence_count: int
    solve_elapsed: float


def _stored_solution_quota_reached(
    *,
    unique_count: int,
    solution_limit: int,
) -> bool:
    """Return whether stored solution collection has reached its configured cap."""

    return unique_count >= solution_limit


def _stored_workflow_key(
    *,
    config: SnakeConfig,
    tool_sequence_key: tuple[object, ...],
    workflow_key: tuple[object, ...],
) -> tuple[object, ...]:
    """Return the deduplication key used for stored workflow candidates.

    For non-repeating use-all-generated-data=ALL workflows, APE parity is at
    the tool-sequence family level rather than the finer binding/target family.
    Shortest-first collection must therefore stop on unique tool sequences, or
    duplicate same-sequence witnesses exhaust the solution cap before later
    horizons are reached.
    """

    if not config.tool_seq_repeat and config.use_all_generated_data == "ALL":
        return tool_sequence_key
    if (
        not config.tool_seq_repeat
        and config.use_all_generated_data == "ONE"
        and (
            config.constraints_path is None
            or config.constraints_path.name == "constraints_empty.json"
        )
    ):
        return tool_sequence_key
    return workflow_key


def _artifact_is_produced_output(symbol: clingo.Symbol) -> bool:
    return (
        symbol.type == clingo.SymbolType.Function
        and symbol.name == "out"
        and len(symbol.arguments) == 3
    )
