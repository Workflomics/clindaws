"""Worker-process orchestration: symbol (de)serialization, IPC, timeouts."""

from __future__ import annotations

import multiprocessing
import queue
import traceback
from pathlib import Path
from time import perf_counter
from typing import Callable

import clingo

from clindaws.core.models import RunResult, TimingBreakdown
from clindaws.core.runtime_stats import ProcessTreePeakMonitor
from clindaws.execution.runner_modes import ProgressCallback, _SOLVER_DISPATCH
from clindaws.execution.runner_utils import _report


def _serialize_symbol_collection(
    collection: tuple[tuple[object, ...], ...],
) -> tuple[tuple[str, ...], ...]:
    return tuple(tuple(str(symbol) for symbol in symbols) for symbols in collection)


def _deserialize_symbol_collection(
    collection: tuple[tuple[str, ...], ...],
) -> tuple[tuple[clingo.Symbol, ...], ...]:
    return tuple(
        tuple(clingo.parse_term(symbol_text) for symbol_text in symbols)
        for symbols in collection
    )


def _serialize_solve_output(solve_output) -> dict[str, object]:
    return {
        "raw_solutions": _serialize_symbol_collection(solve_output.raw_solutions),
        "solutions": _serialize_symbol_collection(solve_output.solutions),
        "base_grounding_peak_rss_mb": solve_output.base_grounding_peak_rss_mb,
        "base_grounding_sec": solve_output.base_grounding_sec,
        "grounding_sec": solve_output.grounding_sec,
        "solving_sec": solve_output.solving_sec,
        "horizon_records": solve_output.horizon_records,
    }


def _deserialize_solve_output(payload: dict[str, object]):
    from clindaws.execution.solver_solutions import SolveOutput

    return SolveOutput(
        raw_solutions=_deserialize_symbol_collection(payload["raw_solutions"]),
        solutions=_deserialize_symbol_collection(payload["solutions"]),
        base_grounding_peak_rss_mb=float(payload["base_grounding_peak_rss_mb"]),
        base_grounding_sec=float(payload["base_grounding_sec"]),
        grounding_sec=float(payload["grounding_sec"]),
        solving_sec=float(payload["solving_sec"]),
        horizon_records=tuple(payload["horizon_records"]),
    )


def _empty_solve_output():
    from clindaws.execution.solver_solutions import SolveOutput

    return SolveOutput(
        raw_solutions=(),
        solutions=(),
        base_grounding_peak_rss_mb=0.0,
        base_grounding_sec=0.0,
        grounding_sec=0.0,
        solving_sec=0.0,
        horizon_records=(),
    )


def _timed_out_run_result(
    *,
    config,
    mode: str,
    grounding_strategy: str,
    fact_bundle,
    translation_sec: float,
    translation_peak_rss_mb: float,
    combined_peak_rss_mb: float,
    solve_start: float,
    completed_stage: str,
    run_log_path: Path | None = None,
    run_summary_path: Path | None = None,
) -> RunResult:
    solving_sec = max(0.0, perf_counter() - solve_start) if completed_stage == "run_timeout" else 0.0
    return RunResult(
        config=config,
        mode=mode,
        grounding_strategy=grounding_strategy,
        fact_bundle=fact_bundle,
        solutions=(),
        timings=TimingBreakdown(
            translation_sec=translation_sec,
            grounding_sec=0.0,
            solving_sec=solving_sec,
            rendering_sec=0.0,
        ),
        translation_peak_rss_mb=translation_peak_rss_mb,
        combined_peak_rss_mb=combined_peak_rss_mb,
        base_grounding_peak_rss_mb=0.0,
        base_grounding_sec=0.0,
        horizon_records=(),
        translation_path=None,
        answer_set_path=None,
        solution_summary_path=None,
        workflow_signature_path=None,
        graph_paths=(),
        raw_models_seen=0,
        raw_answer_sets_found=0,
        unique_solutions_found=0,
        diagnostic_counts_enabled=False,
        timed_out=True,
        completed_stage=completed_stage,
        run_log_path=run_log_path,
        run_summary_path=run_summary_path,
    )


def _drain_progress_queue(
    progress_queue: multiprocessing.queues.Queue | None,
    progress_callback: ProgressCallback,
    event_callback: Callable[[dict[str, object]], None] | None = None,
) -> None:
    if progress_queue is None:
        return
    while True:
        try:
            message = progress_queue.get_nowait()
        except queue.Empty:
            return
        if message is None:
            return
        if isinstance(message, dict):
            if event_callback is not None:
                event_callback(message)
            continue
        _report(progress_callback, str(message))


def _solve_worker_entrypoint(
    *,
    mode: str,
    config,
    ontology,
    fact_bundle,
    workflow_input_dims,
    tool_output_dims,
    capture_raw_models: bool,
    diagnostic_counts_enabled: bool,
    parallel_mode: str | None,
    project_models: bool,
    result_queue: multiprocessing.queues.Queue,
    progress_queue: multiprocessing.queues.Queue,
) -> None:
    def _worker_progress(message: object) -> None:
        progress_queue.put(message)

    try:
        solve_output = _SOLVER_DISPATCH[mode](
            config,
            fact_bundle,
            ontology=ontology,
            workflow_input_dims=workflow_input_dims,
            tool_output_dims=tool_output_dims,
            progress_callback=_worker_progress,
            capture_raw_models=capture_raw_models,
            diagnostic_counts_enabled=diagnostic_counts_enabled,
            parallel_mode=parallel_mode,
            project_models=project_models,
        )
        result_queue.put(
            {
                "ok": True,
                "solve_output": _serialize_solve_output(solve_output),
            }
        )
    except KeyError:
        result_queue.put(
            {
                "ok": False,
                "error_type": "ValueError",
                "error_message": f"Unsupported mode: {mode}",
                "traceback": traceback.format_exc(),
            }
        )
    except BaseException as exc:
        result_queue.put(
            {
                "ok": False,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
    finally:
        progress_queue.put(None)


def _run_solve_in_worker(
    *,
    mode: str,
    config,
    ontology,
    fact_bundle,
    workflow_input_dims,
    tool_output_dims,
    capture_raw_models: bool,
    diagnostic_counts_enabled: bool,
    parallel_mode: str | None,
    project_models: bool,
    remaining_timeout: float,
    progress_callback: ProgressCallback,
    memory_monitor: ProcessTreePeakMonitor | None = None,
) -> tuple[object, bool, dict[int, float]]:
    ctx = multiprocessing.get_context("spawn")
    result_queue = ctx.Queue()
    progress_queue = ctx.Queue()
    process = ctx.Process(
        target=_solve_worker_entrypoint,
        kwargs={
            "mode": mode,
            "config": config,
            "ontology": ontology,
            "fact_bundle": fact_bundle,
            "workflow_input_dims": workflow_input_dims,
            "tool_output_dims": tool_output_dims,
            "capture_raw_models": capture_raw_models,
            "diagnostic_counts_enabled": diagnostic_counts_enabled,
            "parallel_mode": parallel_mode,
            "project_models": project_models,
            "result_queue": result_queue,
            "progress_queue": progress_queue,
        },
    )
    process.start()
    deadline = perf_counter() + remaining_timeout
    timed_out = False
    payload: dict[str, object] | None = None
    horizon_peak_rss_by_horizon: dict[int, float] = {}

    def _handle_progress_event(message: dict[str, object]) -> None:
        if message.get("event") != "horizon_complete":
            return
        horizon = message.get("horizon")
        timestamp_ns = message.get("timestamp_ns")
        if not isinstance(horizon, int):
            return
        if memory_monitor is None or not isinstance(timestamp_ns, int):
            horizon_peak_rss_by_horizon[horizon] = 0.0
            return
        horizon_peak_rss_by_horizon[horizon] = memory_monitor.peak_at(timestamp_ns)

    while True:
        _drain_progress_queue(
            progress_queue,
            progress_callback,
            event_callback=_handle_progress_event,
        )
        if payload is None:
            try:
                payload = result_queue.get_nowait()
            except queue.Empty:
                payload = None
        process_alive = process.is_alive()
        if payload is not None and not process_alive:
            break
        if payload is None:
            remaining = deadline - perf_counter()
            if remaining <= 0:
                timed_out = True
                break
            process.join(min(0.1, remaining))
        else:
            process.join(0.1)

    if timed_out:
        _report(
            progress_callback,
            f"Timeout of {config.timeout_sec}s reached, terminating solver worker...",
        )
        process.terminate()
        process.join(1.0)
        if process.is_alive():
            _report(progress_callback, "Solver worker did not exit after terminate; forcing kill.")
            process.kill()
            process.join()
        _drain_progress_queue(
            progress_queue,
            progress_callback,
            event_callback=_handle_progress_event,
        )
        return _empty_solve_output(), True, horizon_peak_rss_by_horizon

    process.join()
    _drain_progress_queue(
        progress_queue,
        progress_callback,
        event_callback=_handle_progress_event,
    )

    if payload is None:
        try:
            payload = result_queue.get(timeout=1.0)
        except queue.Empty as exc:
            raise RuntimeError("Solver worker exited without returning a result.") from exc

    if not payload.get("ok", False):
        raise RuntimeError(
            "Solver worker failed with "
            f"{payload.get('error_type', 'unknown error')}: {payload.get('error_message', '')}\n"
            f"{payload.get('traceback', '')}".rstrip()
        )

    return _deserialize_solve_output(payload["solve_output"]), False, horizon_peak_rss_by_horizon
