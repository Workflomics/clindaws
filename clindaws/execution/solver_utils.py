"""Solver utilities: progress reporting, interrupt handling, grounding helpers."""

from __future__ import annotations

import os
import signal
from contextlib import contextmanager
from time import perf_counter
from typing import Callable, Iterator

import clingo

from clindaws.core.models import FactBundle
from clindaws.execution.solver_control import ProgressCallback
from clindaws.execution.solver_solutions import _artifact_is_produced_output


def _report(progress_callback: ProgressCallback, message: str) -> None:
    if progress_callback is not None:
        progress_callback(message)


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
