"""High-level execution entrypoints.

The runner owns the end-to-end lifecycle around the lower-level clingo solver:

- load and normalize configuration,
- choose the effective translation/backend family,
- build the fact bundle and translation summaries,
- execute grounding/solving in a worker process with timeout control,
- reconstruct canonical workflow candidates and write run artifacts.

Public modes stay small (`single-shot`, `single-shot-sliding-window`, `multi-shot`), while this module maps
them onto the internal backends used by translation and solving.
"""

from __future__ import annotations

import csv
import json
import multiprocessing
import os
import queue
import re
import traceback
from datetime import datetime, timezone
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from time import perf_counter
from typing import Callable
from uuid import uuid4

import clingo

from clindaws.core.config import load_config, SnakeConfig
from clindaws.execution.precompute import apply_precompute
from clindaws.core.models import (
    FactBundle,
    GroundingRunResult,
    HorizonRecord,
    RunResult,
    TimingBreakdown,
    TranslationBuilder,
    TranslationRunResult,
)
from clindaws.core.ontology import Ontology
from clindaws.core.workflow_input_compression import (
    build_workflow_input_compression_plan,
    workflow_input_compression_stats,
)
from clindaws.rendering.rendering import (
    render_solution_graphs,
    write_workflow_signatures,
)
from clindaws.core.runtime_stats import (
    ProcessTreePeakMonitor,
    current_peak_rss_mb,
)
from clindaws.execution.solver import (
    ground_single_shot,
    ground_multi_shot,
    ground_multi_shot_optimized_candidate,
    ground_multi_shot_compressed_candidate,
    program_paths_for_mode,
    solve_multi_shot,
    solve_multi_shot_optimized_candidate,
    solve_multi_shot_compressed_candidate,
    solve_single_shot,
    solve_single_shot_sliding_window,
)
from clindaws.core.tool_annotations import (
    load_candidate_tool_annotations,
    load_direct_tool_annotations,
    load_multi_shot_tool_annotations,
)
from clindaws.translators.translator_direct import (
    build_fact_bundle,
    build_fact_bundle_ape_multi_shot,
)
from clindaws.translators.translator_optimized_candidate import build_optimized_candidate_fact_bundle
from clindaws.core.workflow import reconstruct_solution


SCHEMA_PREDICATES = (
    "tool_input",
    "input_port",
    "tool_output",
    "output_port",
    "tool_candidate",
    "candidate_in",
    "candidate_out",
    "dynamic_tool_candidate",
    "dynamic_initial_bindable",
    "dynamic_candidate_input_port",
    "dynamic_candidate_input_signature_id",
    "optimized_candidate_input_support_class",
    "dynamic_candidate_input_association_class",
    "dynamic_signature_support_class",
    "dynamic_support_class_bindable_producer_port",
    "dynamic_association_class_bindable_producer_port",
    "optimized_candidate_input_profile",
    "dynamic_signature_profile",
    "dynamic_profile_accepts",
    "optimized_goal_requirement_profile",
    "dynamic_forced_produced_bind",
    "dynamic_candidate_output_port",
    "dynamic_candidate_output_multiplicity",
    "dynamic_candidate_total_output_multiplicity",
    "dynamic_candidate_output_singleton",
    "dynamic_candidate_output_choice_value",
    "dynamic_candidate_output_declared_type",
)
RUNTIME_TRANSLATION_BUILDER = "runtime_legacy"
OPTIMIZED_CANDIDATE_TRANSLATION_BUILDER = "candidate_optimized"
COMPRESSED_CANDIDATE_TRANSLATION_BUILDER = "candidate_compressed"
ProgressCallback = Callable[[object], None] | None


@dataclass(frozen=True)
class _ModeConfig:
    solver_family: str
    solver_approach: str
    translation_pathway: str
    translation_builder: TranslationBuilder
    supports_ground_only: bool


_MODE_CONFIGS = {
    # Public single-shot currently shares the APE-style multi-shot translation
    # surface, then changes only the grounding/solving strategy downstream.
    "single-shot": _ModeConfig(
        solver_family="single-shot",
        solver_approach="one-shot",
        translation_pathway="ape_multi_shot",
        translation_builder=RUNTIME_TRANSLATION_BUILDER,
        supports_ground_only=True,
    ),
    "single-shot-sliding-window": _ModeConfig(
        solver_family="single-shot",
        solver_approach="sliding_window",
        translation_pathway="ape_multi_shot",
        translation_builder=RUNTIME_TRANSLATION_BUILDER,
        supports_ground_only=False,
    ),
    "multi-shot": _ModeConfig(
        solver_family="multi-shot",
        solver_approach="legacy",
        translation_pathway="ape_multi_shot",
        translation_builder=RUNTIME_TRANSLATION_BUILDER,
        supports_ground_only=True,
    ),
    # Optimized multi-shot is an explicit optimized-candidate backend rather
    # than a small variation of the direct multi-shot encoding family.
    "multi-shot-optimized-candidate": _ModeConfig(
        solver_family="multi-shot",
        solver_approach="optimized_candidate",
        translation_pathway="optimized_candidate",
        translation_builder=OPTIMIZED_CANDIDATE_TRANSLATION_BUILDER,
        supports_ground_only=True,
    ),
    "multi-shot-compressed-candidate": _ModeConfig(
        solver_family="multi-shot",
        solver_approach="optimized_candidate",
        translation_pathway="optimized_candidate",
        translation_builder=OPTIMIZED_CANDIDATE_TRANSLATION_BUILDER,
        supports_ground_only=True,
    ),
}

_SOLVER_DISPATCH = {
    "single-shot": solve_single_shot,
    "single-shot-sliding-window": solve_single_shot_sliding_window,
    "multi-shot": solve_multi_shot,
    "multi-shot-optimized-candidate": solve_multi_shot_optimized_candidate,
    "multi-shot-compressed-candidate": solve_multi_shot_compressed_candidate,
}

_GROUNDER_DISPATCH = {
    "single-shot": ground_single_shot,
    "multi-shot": ground_multi_shot,
    "multi-shot-optimized-candidate": ground_multi_shot_optimized_candidate,
    "multi-shot-compressed-candidate": ground_multi_shot_compressed_candidate,
}


@dataclass(frozen=True)
class RunContext:
    """All translation-phase results, passed from _prepare_run_context to callers."""

    config: SnakeConfig
    solution_dir: Path
    fact_bundle: FactBundle
    workflow_input_dims: dict[str, dict[str, tuple[str, ...]]]
    tool_output_dims: dict[tuple[str, int], dict[str, tuple[str, ...]]]
    translation_sec: float
    translation_peak_rss_mb: float
    translation_peak_combined_rss_mb: float
    effective_translation_strategy: str
    resolved_translation_builder: str
    run_metadata: dict[str, object]


def _mode_config(mode: str) -> _ModeConfig:
    try:
        return _MODE_CONFIGS[mode]
    except KeyError as exc:
        raise ValueError(f"Unsupported mode: {mode}") from exc


def _effective_translation_strategy(mode: str, grounding_strategy: str) -> str:
    translation_pathway = _mode_config(mode).translation_pathway
    if translation_pathway == "optimized_candidate":
        return "python"
    if translation_pathway == "ape_multi_shot":
        return "ape_clingo_legacy"
    return grounding_strategy


def _workflow_input_dims_from_config(config: SnakeConfig) -> dict[str, dict[str, tuple[str, ...]]]:
    return {
        f"wf_input_{index}": {
            str(dim): tuple(str(value) for value in values)
            for dim, values in item.items()
        }
        for index, item in enumerate(config.inputs)
    }


def _tool_output_dims_lookup(tools: tuple) -> dict[tuple[str, int], dict[str, tuple[str, ...]]]:
    output_dims: dict[tuple[str, int], dict[str, tuple[str, ...]]] = {}
    for tool in tools:
        for port_index, output_spec in enumerate(getattr(tool, "outputs", ())):
            output_dims[(str(tool.mode_id), port_index)] = {
                str(dim): tuple(str(value) for value in values)
                for dim, values in output_spec.dimensions.items()
            }
    return output_dims


def _load_tools_for_mode(config, translation_pathway: str):
    if translation_pathway == "optimized_candidate":
        return load_candidate_tool_annotations(config.tool_annotations_path, config.ontology_prefix)
    if translation_pathway == "ape_multi_shot":
        return load_multi_shot_tool_annotations(config.tool_annotations_path, config.ontology_prefix)
    return load_direct_tool_annotations(config.tool_annotations_path, config.ontology_prefix)


def _solver_family(mode: str) -> str:
    return _mode_config(mode).solver_family


def _solver_approach(mode: str) -> str:
    return _mode_config(mode).solver_approach


def _report(progress_callback: ProgressCallback, message: str) -> None:
    if progress_callback is not None:
        progress_callback(message)


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
    from clindaws.execution.solver import SolveOutput

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
    from clindaws.execution.solver import SolveOutput

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


def _combined_peak_mb(memory_monitor: ProcessTreePeakMonitor | None) -> float:
    """Return the latest sampled combined-process peak, sampling once first."""

    if memory_monitor is None:
        return 0.0
    memory_monitor.sample_now()
    return memory_monitor.current_peak_mb()


def _records_with_combined_peak_rss(
    records: tuple[HorizonRecord, ...],
    *,
    memory_monitor: ProcessTreePeakMonitor | None,
    peaks_by_horizon: dict[int, float] | None = None,
) -> tuple[HorizonRecord, ...]:
    """Replace per-process RSS fields with cumulative combined-process peaks."""

    updated_records: list[HorizonRecord] = []
    cumulative_peak_mb = 0.0
    horizon_peaks = peaks_by_horizon or {}
    for record in records:
        combined_peak_mb = horizon_peaks.get(record.horizon, 0.0)
        if combined_peak_mb <= 0.0 and memory_monitor is not None:
            combined_peak_mb = _combined_peak_mb(memory_monitor)
        if combined_peak_mb <= 0.0:
            combined_peak_mb = record.peak_rss_mb
        cumulative_peak_mb = max(cumulative_peak_mb, combined_peak_mb)
        updated_records.append(replace(record, peak_rss_mb=cumulative_peak_mb))
    return tuple(updated_records)


def _solve_worker_entrypoint(
    *,
    mode: str,
    config,
    fact_bundle,
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
    fact_bundle,
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
            "fact_bundle": fact_bundle,
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


def _constraint_metadata(config) -> tuple[str | None, int]:
    if config.constraints_path is None:
        return None, 0

    count = 0
    if config.constraints_path.exists():
        raw = json.loads(config.constraints_path.read_text(encoding="utf-8"))
        count = len(raw.get("constraints", []))
    return str(config.constraints_path), count


def _run_metadata_payload(*, config, ontology, tools) -> dict[str, object]:
    constraints_used, constraint_count = _constraint_metadata(config)
    return {
        "config_path": str(config.config_path),
        "ontology_used": str(config.ontology_path),
        "ontology_entry_count": len(ontology.nodes),
        "tool_annotation_used": str(config.tool_annotations_path),
        "tool_count": len(tools),
        "constraints_used": constraints_used,
        "constraint_count": constraint_count,
    }


def _compressed_candidate_engaged(fact_bundle) -> bool:
    return fact_bundle.internal_solver_mode in {
        "multi-shot-optimized-candidate",
        "multi-shot-compressed-candidate",
    }


def _effective_solve_start_horizon(*, config, fact_bundle) -> int:
    if _compressed_candidate_engaged(fact_bundle):
        return max(config.solution_length_min, fact_bundle.earliest_solution_step)
    return config.solution_length_min


def _sanitize_filename_token(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    sanitized = re.sub(r"_+", "_", sanitized).strip("._-")
    return sanitized or "default"


def _answer_sets_filename(
    *,
    config,
    mode: str,
    optimized_enabled: bool,
    effective_parallel_mode: str | None,
) -> str:
    parts = [
        "answer_sets",
        _sanitize_filename_token(config.config_path.stem),
        _sanitize_filename_token(mode),
        "opt" if optimized_enabled else "noopt",
    ]
    if effective_parallel_mode:
        parts.append(f"parallel_{_sanitize_filename_token(effective_parallel_mode)}")
    return "__".join(parts) + ".txt"


def _workflow_signatures_filename(
    *,
    config,
    mode: str,
    optimized_enabled: bool,
    effective_parallel_mode: str | None,
) -> str:
    parts = [
        "workflow_signatures",
        _sanitize_filename_token(config.config_path.stem),
        _sanitize_filename_token(mode),
        "opt" if optimized_enabled else "noopt",
    ]
    if effective_parallel_mode:
        parts.append(f"parallel_{_sanitize_filename_token(effective_parallel_mode)}")
    return "__".join(parts) + ".json"


def _horizon_record_payload(records: tuple[HorizonRecord, ...]) -> list[dict[str, object]]:
    return [asdict(record) for record in records]


def _ensure_csv_header(csv_path: Path, fieldnames: list[str]) -> None:
    if not csv_path.exists():
        with csv_path.open("a", encoding="utf-8", newline="") as handle:
            csv.DictWriter(handle, fieldnames=fieldnames).writeheader()
        return

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        existing_header = next(reader, None)

    if existing_header == fieldnames:
        return

    rows: list[dict[str, str]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        dict_reader = csv.DictReader(handle)
        for row in dict_reader:
            migrated_row: dict[str, str] = {}
            for name in fieldnames:
                value = row.get(name, "")
                if not value and name == "workflow_candidates_found":
                    value = row.get("solutions_found", "")
                migrated_row[name] = value
            rows.append(migrated_row)

    temp_path = csv_path.with_suffix(csv_path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temp_path.replace(csv_path)


class _RunLogWriter:
    """APE-style per-horizon CSV logger."""

    fieldnames = [
        "test",
        "mode",
        "horizon",
        "optimized_enabled",
        "effective_parallel_mode",
        "compressed_candidate_engaged",
        "translation_ms",
        "setup_grounding_ms",
        "solving_ms",
        "memory_used_mb",
        "workflow_candidates_found",
        "constraints_used",
        "timed_out",
    ]

    def __init__(
        self,
        *,
        csv_path: Path,
        mode: str,
        grounding_strategy: str,
        fact_count: int,
        run_metadata: dict[str, object],
        translation_builder: TranslationBuilder,
        translation_schema: str,
        run_id: str,
        timestamp_utc: str,
        optimized_enabled: bool,
        effective_parallel_mode: str | None,
        compressed_candidate_engaged: bool,
    ) -> None:
        self.csv_path = csv_path
        self.cumulative_unique_solutions = 0
        self.test_name = Path(str(run_metadata["config_path"])).resolve().parent.name
        self.constraints_used = str(bool(run_metadata["constraints_used"])).lower()
        self.mode = mode
        self.optimized_enabled = optimized_enabled
        self.effective_parallel_mode = effective_parallel_mode or ""
        self.compressed_candidate_engaged = str(compressed_candidate_engaged).lower()
        self.base_grounding_ms = 0
        self.base_grounding_peak_rss_mb = 0.0
        self._translation_ms = 0
        _ensure_csv_header(self.csv_path, self.fieldnames)

    def _write_row(self, row: dict[str, object]) -> None:
        with self.csv_path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fieldnames)
            writer.writerow(row)

    def log_translation(self, *, translation_sec: float, translation_peak_rss_mb: float) -> None:
        self._translation_ms = round(translation_sec * 1000)

    def log_base_grounding(self, *, base_grounding_sec: float, base_grounding_peak_rss_mb: float) -> None:
        self.base_grounding_ms = round(base_grounding_sec * 1000)
        self.base_grounding_peak_rss_mb = base_grounding_peak_rss_mb

    def log_horizon(self, record: HorizonRecord) -> None:
        self.cumulative_unique_solutions += record.unique_workflows_stored
        setup_grounding_ms = round(record.grounding_sec * 1000)
        memory_used_mb = record.peak_rss_mb or 0.0
        if record.horizon == 1:
            setup_grounding_ms += self.base_grounding_ms
        self._write_row(
            {
                "test": self.test_name,
                "mode": self.mode,
                "horizon": record.horizon,
                "optimized_enabled": str(self.optimized_enabled).lower(),
                "effective_parallel_mode": self.effective_parallel_mode,
                "compressed_candidate_engaged": self.compressed_candidate_engaged,
                "translation_ms": self._translation_ms,
                "setup_grounding_ms": setup_grounding_ms,
                "solving_ms": round(record.solving_sec * 1000),
                "memory_used_mb": f"{memory_used_mb:.2f}" if memory_used_mb else "",
                "workflow_candidates_found": self.cumulative_unique_solutions,
                "constraints_used": self.constraints_used,
                "timed_out": "false",
            }
        )

    def log_timeout(self, *, elapsed_ms: int, memory_used_mb: float | None = None) -> None:
        """Write a sentinel row marking the step that was interrupted by timeout."""
        self._write_row(
            {
                "test": self.test_name,
                "mode": self.mode,
                "horizon": "timeout",
                "optimized_enabled": str(self.optimized_enabled).lower(),
                "effective_parallel_mode": self.effective_parallel_mode,
                "compressed_candidate_engaged": self.compressed_candidate_engaged,
                "translation_ms": self._translation_ms,
                "setup_grounding_ms": elapsed_ms,
                "solving_ms": "",
                "memory_used_mb": f"{memory_used_mb:.2f}" if memory_used_mb else "",
                "workflow_candidates_found": self.cumulative_unique_solutions,
                "constraints_used": self.constraints_used,
                "timed_out": "true",
            }
        )


class _RunSummaryWriter:
    """Append-only per-run summary CSV logger."""

    fieldnames = [
        "run_id",
        "timestamp_utc",
        "mode",
        "solver_family",
        "solver_approach",
        "grounding_strategy",
        "config_path",
        "ontology_used",
        "ontology_entry_count",
        "tool_annotation_used",
        "tool_count",
        "constraints_used",
        "constraint_count",
        "translation_builder",
        "translation_schema",
        "completed_stage",
        "fact_count",
        "translation_sec",
        "translation_peak_rss_mb",
        "combined_peak_rss_mb",
        "base_grounding_sec",
        "base_grounding_peak_rss_mb",
        "grounding_sec_total",
        "solving_sec_total",
        "rendering_sec_total",
        "total_sec",
        "raw_models_seen",
        "raw_solutions_found",
        "solutions_found",
        "grounded_horizons",
        "first_satisfiable_horizon",
        "optimized_enabled",
        "effective_parallel_mode",
        "compressed_candidate_engaged",
    ]

    def __init__(
        self,
        *,
        csv_path: Path,
        mode: str,
        grounding_strategy: str,
        fact_count: int,
        run_metadata: dict[str, object],
        translation_builder: TranslationBuilder,
        translation_schema: str,
        run_id: str,
        timestamp_utc: str,
        optimized_enabled: bool,
        compressed_candidate_engaged: bool,
    ) -> None:
        self.csv_path = csv_path
        self.base_row = {
            "run_id": run_id,
            "timestamp_utc": timestamp_utc,
            "mode": mode,
            "solver_family": _solver_family(mode),
            "solver_approach": _solver_approach(mode),
            "grounding_strategy": grounding_strategy,
            "config_path": run_metadata["config_path"],
            "ontology_used": run_metadata["ontology_used"],
            "ontology_entry_count": run_metadata["ontology_entry_count"],
            "tool_annotation_used": run_metadata["tool_annotation_used"],
            "tool_count": run_metadata["tool_count"],
            "constraints_used": run_metadata["constraints_used"],
            "constraint_count": run_metadata["constraint_count"],
            "translation_builder": translation_builder,
            "translation_schema": translation_schema,
            "fact_count": fact_count,
            "optimized_enabled": str(optimized_enabled).lower(),
            "compressed_candidate_engaged": str(compressed_candidate_engaged).lower(),
        }
        _ensure_csv_header(self.csv_path, self.fieldnames)

    def log_summary(
        self,
        *,
        completed_stage: str,
        timings: TimingBreakdown,
        translation_peak_rss_mb: float,
        combined_peak_rss_mb: float,
        base_grounding_sec: float,
        base_grounding_peak_rss_mb: float,
        horizon_records: tuple[HorizonRecord, ...],
        raw_models_seen: int,
        raw_solutions_found: int,
        solutions_found: int,
        grounded_horizons: tuple[int, ...] = (),
        effective_parallel_mode: str | None = None,
    ) -> None:
        first_satisfiable = next((record.horizon for record in horizon_records if record.satisfiable), None)
        grounded_horizons_text = ",".join(str(horizon) for horizon in grounded_horizons)
        row = {
            **self.base_row,
            "completed_stage": completed_stage,
            "translation_sec": f"{timings.translation_sec:.6f}",
            "translation_peak_rss_mb": f"{translation_peak_rss_mb:.3f}" if translation_peak_rss_mb else "",
            "combined_peak_rss_mb": f"{combined_peak_rss_mb:.3f}" if combined_peak_rss_mb else "",
            "base_grounding_sec": f"{base_grounding_sec:.6f}" if base_grounding_sec else "",
            "base_grounding_peak_rss_mb": f"{base_grounding_peak_rss_mb:.3f}" if base_grounding_peak_rss_mb else "",
            "grounding_sec_total": f"{timings.grounding_sec:.6f}",
            "solving_sec_total": f"{timings.solving_sec:.6f}",
            "rendering_sec_total": f"{timings.rendering_sec:.6f}",
            "total_sec": f"{timings.total_sec:.6f}",
            "raw_models_seen": raw_models_seen,
            "raw_solutions_found": raw_solutions_found,
            "solutions_found": solutions_found,
            "grounded_horizons": grounded_horizons_text,
            "first_satisfiable_horizon": first_satisfiable if first_satisfiable is not None else "",
            "effective_parallel_mode": effective_parallel_mode or "",
        }
        with self.csv_path.open("a", encoding="utf-8", newline="") as handle:
            csv.DictWriter(handle, fieldnames=self.fieldnames).writerow(row)


class _RunCsvWriters:
    """Per-run CSV writers sharing one run id."""

    def __init__(
        self,
        *,
        csv_dir: Path,
        mode: str,
        grounding_strategy: str,
        fact_count: int,
        run_metadata: dict[str, object],
        translation_builder: TranslationBuilder,
        translation_schema: str,
        optimized_enabled: bool,
        effective_parallel_mode: str | None,
        compressed_candidate_engaged: bool,
    ) -> None:
        run_id = str(uuid4())
        timestamp_utc = datetime.now(timezone.utc).isoformat()
        csv_dir.mkdir(parents=True, exist_ok=True)
        self.step_writer = _RunLogWriter(
            csv_path=csv_dir / "asp_run_log.csv",
            mode=mode,
            grounding_strategy=grounding_strategy,
            fact_count=fact_count,
            run_metadata=run_metadata,
            translation_builder=translation_builder,
            translation_schema=translation_schema,
            run_id=run_id,
            timestamp_utc=timestamp_utc,
            optimized_enabled=optimized_enabled,
            effective_parallel_mode=effective_parallel_mode,
            compressed_candidate_engaged=compressed_candidate_engaged,
        )
        self.summary_writer = _RunSummaryWriter(
            csv_path=csv_dir / "asp_run_summary.csv",
            mode=mode,
            grounding_strategy=grounding_strategy,
            fact_count=fact_count,
            run_metadata=run_metadata,
            translation_builder=translation_builder,
            translation_schema=translation_schema,
            run_id=run_id,
            timestamp_utc=timestamp_utc,
            optimized_enabled=optimized_enabled,
            compressed_candidate_engaged=compressed_candidate_engaged,
        )

    @property
    def run_log_path(self) -> Path:
        return self.step_writer.csv_path

    @property
    def run_summary_path(self) -> Path:
        return self.summary_writer.csv_path


def _translation_schema(fact_bundle) -> str:
    if any(
        fact_bundle.predicate_counts.get(name, 0)
        for name in (
            "dynamic_tool_candidate",
            "dynamic_signature_support_class",
            "dynamic_signature_profile",
            "dynamic_profile_accepts",
            "dynamic_candidate_output_singleton",
            "dynamic_candidate_output_choice_value",
        )
    ):
        return "optimized_candidate"
    if any(fact_bundle.predicate_counts.get(name, 0) for name in ("tool_candidate", "candidate_in", "candidate_out")):
        return "candidate"
    if any(fact_bundle.predicate_counts.get(name, 0) for name in ("tool_input", "input_port", "tool_output", "output_port")):
        return "legacy"
    return "unknown"


def _encoding_schema_summary(
    mode: str,
    *,
    optimized: bool = False,
) -> dict[str, object]:
    program_paths = program_paths_for_mode(
        mode,
        optimized=optimized,
    )
    predicate_presence: dict[str, bool] = {}
    for predicate in SCHEMA_PREDICATES:
        predicate_presence[predicate] = False

    for program_path in program_paths:
        text = program_path.read_text(encoding="utf-8")
        for predicate in SCHEMA_PREDICATES:
            if not predicate_presence[predicate] and f"{predicate}(" in text:
                predicate_presence[predicate] = True

    return {
        "program_paths": [str(path) for path in program_paths],
        "predicate_presence": predicate_presence,
        "schema": (
            "optimized_candidate"
            if any(
                predicate_presence[name]
                for name in (
                    "dynamic_tool_candidate",
                    "dynamic_signature_support_class",
                    "dynamic_signature_profile",
                    "dynamic_profile_accepts",
                    "dynamic_candidate_output_singleton",
                    "dynamic_candidate_output_choice_value",
                )
            )
            else
            "candidate"
            if any(predicate_presence[name] for name in ("tool_candidate", "candidate_in", "candidate_out"))
            else "legacy"
            if any(predicate_presence[name] for name in ("tool_input", "input_port", "tool_output", "output_port"))
            else "unknown"
        ),
    }


def _translation_warnings(
    *,
    mode: str,
    fact_bundle,
    encoding_summary: dict[str, object],
) -> list[str]:
    warnings: list[str] = []
    translation_schema = _translation_schema(fact_bundle)
    encoding_presence = encoding_summary["predicate_presence"]
    translation_pathway = _mode_config(mode).translation_pathway

    if translation_pathway in {"dynamic", "optimized_candidate"} and translation_schema != "optimized_candidate":
        warnings.append(
            f"{mode} expects optimized-candidate translation, but the emitted translation schema is {translation_schema}."
        )
    if translation_schema == "optimized_candidate" and encoding_summary["schema"] != "optimized_candidate":
        warnings.append(
            "Optimized-candidate translation is not compatible with the selected encoding."
        )

    if translation_schema == "candidate" and not any(
        encoding_presence[name] for name in ("tool_candidate", "candidate_in", "candidate_out")
    ):
        warnings.append(
            "Translated facts use candidate predicates, but the selected encoding does not reference candidate predicates."
        )

    if translation_schema == "optimized_candidate" and not any(
        encoding_presence[name]
        for name in (
            "dynamic_tool_candidate",
            "dynamic_signature_support_class",
            "dynamic_signature_profile",
            "dynamic_profile_accepts",
            "dynamic_candidate_output_singleton",
            "dynamic_candidate_output_choice_value",
        )
    ):
        warnings.append(
            "Translated facts use optimized-candidate predicates, but the selected encoding does not reference the optimized-candidate predicate family."
        )

    if translation_schema == "legacy" and not any(
        encoding_presence[name] for name in ("tool_input", "input_port", "tool_output", "output_port")
    ):
        warnings.append(
            "Translated facts use legacy tool_input/tool_output predicates, but the selected encoding does not reference the legacy schema."
        )

    if encoding_summary["schema"] != "unknown" and translation_schema != "unknown" and encoding_summary["schema"] != translation_schema:
        warnings.append(
            f"Translation schema ({translation_schema}) does not match encoding schema ({encoding_summary['schema']})."
        )

    return warnings


def _workflow_input_compression_payload(
    *,
    config: SnakeConfig,
    mode: str,
    internal_solver_mode: str,
    compression_active: bool | None = None,
) -> dict[str, object] | None:
    if mode != "multi-shot" or internal_solver_mode != "multi-shot":
        return None

    workflow_input_dimensions = {
        f"wf_input_{index}": dimensions
        for index, dimensions in enumerate(config.inputs)
    }
    plan = build_workflow_input_compression_plan(workflow_input_dimensions)
    payload: dict[str, object] = dict(sorted(workflow_input_compression_stats(plan).items()))
    if compression_active is not None:
        payload["workflow_input_compression_active"] = compression_active
        payload["workflow_input_planner_visible_count_effective"] = (
            plan.planner_visible_count_if_compressed
            if compression_active
            else plan.planner_visible_count_if_uncompressed
        )
    return payload


def _solve_callback_profile_payload(
    records: tuple[HorizonRecord, ...],
    *,
    solving_sec: float,
) -> dict[str, float] | None:
    if not any(record.model_callback_sec is not None for record in records):
        return None

    model_callback_sec = sum(record.model_callback_sec or 0.0 for record in records)
    shown_symbols_sec = sum(record.shown_symbols_sec or 0.0 for record in records)
    workflow_signature_key_sec = sum(
        record.workflow_signature_key_sec or 0.0
        for record in records
    )
    canonicalization_sec = sum(record.canonicalization_sec or 0.0 for record in records)
    other_callback_sec = max(
        0.0,
        model_callback_sec
        - shown_symbols_sec
        - workflow_signature_key_sec
        - canonicalization_sec,
    )
    payload = {
        "model_callback_sec": model_callback_sec,
        "shown_symbols_sec": shown_symbols_sec,
        "workflow_signature_key_sec": workflow_signature_key_sec,
        "canonicalization_sec": canonicalization_sec,
        "other_callback_sec": other_callback_sec,
    }
    if solving_sec > 0:
        payload["model_callback_share_of_solving_sec"] = model_callback_sec / solving_sec
    return payload


def _translation_summary_payload(
    *,
    config: SnakeConfig,
    mode: str,
    grounding_strategy: str,
    translation_builder: TranslationBuilder,
    effective_translation_strategy: str,
    fact_bundle,
    translation_path: Path | None,
    translation_sec: float,
) -> dict[str, object]:
    internal_mode = fact_bundle.internal_solver_mode or mode
    encoding_summary = _encoding_schema_summary(
        internal_mode,
        optimized=fact_bundle.python_precompute_enabled,
    )
    schema_presence = {
        predicate: fact_bundle.predicate_counts.get(predicate, 0) > 0
        for predicate in SCHEMA_PREDICATES
    }
    candidate_total = sum(
        stat.candidate_count or 0
        for stat in fact_bundle.tool_stats
    )
    dynamic_input_value_total = sum(stat.dynamic_input_value_count or 0 for stat in fact_bundle.tool_stats)
    dynamic_output_value_total = sum(stat.dynamic_output_value_count or 0 for stat in fact_bundle.tool_stats)
    dynamic_cross_product_estimate = sum(stat.dynamic_cross_product_estimate or 0 for stat in fact_bundle.tool_stats)

    return {
        "mode": mode,
        "solver_family": _solver_family(mode),
        "solver_approach": _solver_approach(mode),
        "internal_solver_mode": internal_mode,
        "internal_schema": fact_bundle.internal_schema,
        "grounding_strategy": grounding_strategy,
        "translation_builder": translation_builder,
        "effective_translation_strategy": effective_translation_strategy,
        "translation_schema": _translation_schema(fact_bundle),
        "fact_count": fact_bundle.fact_count,
        "tool_count": len(fact_bundle.tool_stats),
        "goal_count": fact_bundle.goal_count,
        "workflow_input_count": len(fact_bundle.workflow_input_ids),
        "earliest_solution_step": fact_bundle.earliest_solution_step,
        "python_precompute_enabled": fact_bundle.python_precompute_enabled,
        "python_precompute_fact_count": fact_bundle.python_precompute_fact_count,
        "python_precompute_stats": dict(sorted(fact_bundle.python_precompute_stats.items())),
        "workflow_input_compression": _workflow_input_compression_payload(
            config=config,
            mode=mode,
            internal_solver_mode=internal_mode,
        ),
        "translation_path": str(translation_path) if translation_path else None,
        "translation_sec": translation_sec,
        "predicate_counts": dict(sorted(fact_bundle.predicate_counts.items())),
        "translation_cache_stats": dict(sorted(fact_bundle.cache_stats.items())),
        "translation_emit_stats": dict(sorted(fact_bundle.emit_stats.items())),
        "backend_stats": fact_bundle.backend_stats,
        "translation_schema_predicates": schema_presence,
        "encoding_schema": encoding_summary,
        "expansion_totals": {
            "input_variant_total": sum(stat.input_variant_count for stat in fact_bundle.tool_stats),
            "output_variant_total": sum(stat.output_variant_count for stat in fact_bundle.tool_stats),
            "candidate_total": candidate_total if candidate_total else None,
            "dynamic_input_value_total": dynamic_input_value_total if dynamic_input_value_total else None,
            "dynamic_output_value_total": dynamic_output_value_total if dynamic_output_value_total else None,
            "dynamic_cross_product_estimate": dynamic_cross_product_estimate if dynamic_cross_product_estimate else None,
        },
        "per_tool_port_value_counts": [
            {
                "tool_id": stat.tool_id,
                "tool_label": stat.tool_label,
                "input_port_value_counts": list(stat.dynamic_input_port_value_counts),
                "output_port_value_counts": list(stat.dynamic_output_port_value_counts),
            }
            for stat in fact_bundle.tool_stats
            if stat.dynamic_input_port_value_counts or stat.dynamic_output_port_value_counts
        ],
        "warnings": _translation_warnings(
            mode=internal_mode,
            fact_bundle=fact_bundle,
            encoding_summary=encoding_summary,
        ),
    }


def _write_translation_summary(
    *,
    config: SnakeConfig,
    solution_dir: Path,
    mode: str,
    grounding_strategy: str,
    translation_builder: TranslationBuilder,
    effective_translation_strategy: str,
    fact_bundle,
    translation_path: Path | None,
    translation_sec: float,
) -> tuple[Path, dict[str, object]]:
    translation_summary_path = solution_dir / "translation_summary.json"
    payload = _translation_summary_payload(
        config=config,
        mode=mode,
        grounding_strategy=grounding_strategy,
        translation_builder=translation_builder,
        effective_translation_strategy=effective_translation_strategy,
        fact_bundle=fact_bundle,
        translation_path=None,
        translation_sec=translation_sec,
    )
    translation_summary_path.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    return translation_summary_path, payload


def _default_solution_dir(config: SnakeConfig) -> Path:
    """Return the default output directory when none is provided."""

    return config.solutions_dir_path


def _effective_project_models(mode: str, project_models: bool | None) -> bool:
    """Resolve the effective projection policy for one run."""

    if project_models is not None:
        return project_models
    return False


def _effective_parallel_mode(
    mode: str,
    parallel_mode: str | None,
    fact_bundle,
) -> str | None:
    """Resolve the effective solve parallel mode for one run."""

    if parallel_mode is not None:
        return parallel_mode
    if mode != "multi-shot" or fact_bundle.internal_schema not in {
        "direct_precompute_legacy",
        "optimized_candidate",
        "compressed_candidate_optimized",
    }:
        return None
    cpu_count = os.cpu_count() or 1
    if cpu_count < 2 or len(fact_bundle.tool_labels) < 200:
        return None
    workers = min(cpu_count, 6)
    return f"{workers},compete"


def _ape_multi_shot_direct_bundle(
    config: SnakeConfig,
    ontology: Ontology,
    tools,
    *,
    internal_solver_mode: str,
):
    return replace(
        build_fact_bundle_ape_multi_shot(config, ontology, tools),
        internal_schema="legacy_direct",
        internal_solver_mode=internal_solver_mode,
    )


def _legacy_direct_bundle(
    config: SnakeConfig,
    ontology: Ontology,
    tools,
):
    return _ape_multi_shot_direct_bundle(
        config,
        ontology,
        tools,
        internal_solver_mode="multi-shot",
    )


def _optimized_candidate_internal_bundle(
    config: SnakeConfig,
    ontology: Ontology,
    tools,
    *,
    max_workers: int = 1,
):
    return replace(
        build_optimized_candidate_fact_bundle(config, ontology, tools, max_workers=max_workers),
        internal_schema="optimized_candidate",
        internal_solver_mode="multi-shot-optimized-candidate",
    )


def _compressed_candidate_internal_bundle(
    config: SnakeConfig,
    ontology: Ontology,
    tools,
    *,
    max_workers: int = 1,
):
    return _optimized_candidate_internal_bundle(
        config,
        ontology,
        tools,
        max_workers=max_workers,
    )


def _effective_internal_solver_mode(mode: str, fact_bundle) -> str:
    return fact_bundle.internal_solver_mode or mode


def _is_clasp_id_overflow(exc: BaseException) -> bool:
    message = str(exc)
    return (
        "Id out of range" in message
        or "Value too large to be stored in data type" in message
    )


def _validate_run_config(config: SnakeConfig) -> None:
    """Validate derived run bounds before translation/solving."""

    if config.solution_length_min < 1:
        raise ValueError("solution_length.min must be at least 1.")
    if config.solution_length_max < config.solution_length_min:
        raise ValueError("solution_length.max must be greater than or equal to solution_length.min.")
    if config.solutions < 1:
        raise ValueError("solutions must be at least 1.")
    if config.timeout_sec < 0:
        raise ValueError("timeout_sec must be non-negative.")


def _select_fact_bundle(
    *,
    mode_config: _ModeConfig,
    mode: str,
    config: SnakeConfig,
    ontology: Ontology,
    tools: tuple,
    optimized: bool,
    effective_translation_strategy: str,
    progress_callback: ProgressCallback,
    max_workers: int = 1,
) -> tuple[FactBundle, str]:
    """Select and build the fact bundle, applying backend fallback logic.

    Returns (fact_bundle, resolved_translation_builder).
    """
    resolved_translation_builder = mode_config.translation_builder

    if mode_config.translation_pathway == "ape_multi_shot":
        # Plain multi-shot and the public single-shot modes both start from the
        # APE-style direct fact surface. Optional precompute augments that
        # bundle, while optimized multi-shot swaps to the optimized-candidate
        # internal bundle entirely.
        fact_bundle = _ape_multi_shot_direct_bundle(
            config,
            ontology,
            tools,
            internal_solver_mode=(
                "single-shot"
                if mode == "single-shot"
                else "single-shot-sliding-window"
                if mode == "single-shot-sliding-window"
                else "multi-shot"
            ),
        )
        if optimized:
            if mode_config.solver_family == "single-shot":
                raise ValueError("--optimized is not yet supported for single-shot modes.")
            optimized_candidate_tools = load_candidate_tool_annotations(
                config.tool_annotations_path,
                config.ontology_prefix,
            )
            _report(progress_callback, "Step 1b: optimized-candidate translation started.")
            fact_bundle = _optimized_candidate_internal_bundle(
                config,
                ontology,
                optimized_candidate_tools,
                max_workers=max_workers,
            )
            resolved_translation_builder = OPTIMIZED_CANDIDATE_TRANSLATION_BUILDER
            _report(
                progress_callback,
                "Step 1b complete: selected optimized-candidate schema "
                f"with {fact_bundle.fact_count} facts.",
            )
        else:
            fact_bundle = apply_precompute(
                mode,
                config,
                ontology,
                tools,
                fact_bundle,
                optimized_programs=False,
            )

    elif mode_config.translation_pathway == "normal":
        # Reserved for the legacy direct translation family where optimized mode
        # means "add Python-emitted helper facts" rather than "switch backend".
        fact_bundle = replace(
            build_fact_bundle(config, ontology, tools, effective_translation_strategy),
            internal_schema="legacy_direct",
            internal_solver_mode="single-shot",
        )
        if optimized:
            _report(progress_callback, "Step 1b: Python precompute started.")
            fact_bundle = apply_precompute(
                mode,
                config,
                ontology,
                tools,
                fact_bundle,
                optimized_programs=True,
            )
            _report(
                progress_callback,
                "Step 1b complete: Python precompute emitted "
                f"{fact_bundle.python_precompute_fact_count} helper facts.",
            )

    else:
        raise ValueError(f"Unsupported translation pathway: {mode_config.translation_pathway}")

    return fact_bundle, resolved_translation_builder


def _prepare_run_context(
    config_path: str | Path,
    *,
    mode: str,
    grounding_strategy: str,
    output_dir: str | Path | None = None,
    solutions: int | None = None,
    min_length: int | None = None,
    max_length: int | None = None,
    progress_callback: ProgressCallback = None,
    optimized: bool = False,
    max_workers: int = 1,
    memory_monitor: ProcessTreePeakMonitor | None = None,
) -> RunContext:
    """Load config and build the translation-phase context for one run.

    The returned context is the hand-off point between translation and the later
    solve/render phases. It packages the resolved config, output directory,
    backend-specific fact bundle, and translation diagnostics in one named
    object so callers do not need to thread anonymous tuples around.
    """

    mode_config = _mode_config(mode)
    if optimized and mode_config.solver_family == "single-shot":
        raise ValueError("--optimized is not yet supported for single-shot modes.")
    if optimized and mode != "multi-shot":
        raise ValueError("--optimized supports only multi-shot.")
    config = load_config(config_path)
    config = config.model_copy(update={
        "solutions": solutions if solutions is not None else config.solutions,
        "solution_length_min": min_length if min_length is not None else config.solution_length_min,
        "solution_length_max": max_length if max_length is not None else config.solution_length_max,
    })
    _validate_run_config(config)
    solution_dir = Path(output_dir).resolve() if output_dir else _default_solution_dir(config)
    solution_dir.mkdir(parents=True, exist_ok=True)

    effective_translation_strategy = _effective_translation_strategy(mode, grounding_strategy)

    _report(progress_callback, "Step 1: translation started.")
    start = perf_counter()
    ontology = Ontology.from_file(config.ontology_path, config.ontology_prefix)
    tools = _load_tools_for_mode(config, mode_config.translation_pathway)
    run_metadata = _run_metadata_payload(config=config, ontology=ontology, tools=tools)
    fact_bundle, resolved_translation_builder = _select_fact_bundle(
        mode_config=mode_config,
        mode=mode,
        config=config,
        ontology=ontology,
        tools=tools,
        optimized=optimized,
        effective_translation_strategy=effective_translation_strategy,
        progress_callback=progress_callback,
        max_workers=max_workers,
    )
    translation_sec = perf_counter() - start
    translation_peak_rss_mb = current_peak_rss_mb()
    if memory_monitor is not None:
        memory_monitor.sample_now()
        translation_peak_combined_rss_mb = memory_monitor.current_peak_mb()
    else:
        translation_peak_combined_rss_mb = translation_peak_rss_mb
    _report(progress_callback, f"Step 1 complete: translation finished after {translation_sec:.3f}s.")

    return RunContext(
        config=config,
        solution_dir=solution_dir,
        fact_bundle=fact_bundle,
        workflow_input_dims=_workflow_input_dims_from_config(config),
        tool_output_dims=_tool_output_dims_lookup(tools),
        translation_sec=translation_sec,
        translation_peak_rss_mb=translation_peak_rss_mb,
        translation_peak_combined_rss_mb=translation_peak_combined_rss_mb,
        effective_translation_strategy=effective_translation_strategy,
        resolved_translation_builder=resolved_translation_builder,
        run_metadata=run_metadata,
    )


def run_translate_only(
    config_path: str | Path,
    *,
    mode: str,
    grounding_strategy: str,
    output_dir: str | Path | None = None,
    solutions: int | None = None,
    min_length: int | None = None,
    max_length: int | None = None,
    progress_callback: ProgressCallback = None,
    optimized: bool = False,
    max_workers: int = 1,
) -> TranslationRunResult:
    """Run translation only and write translation diagnostics."""

    with ProcessTreePeakMonitor() as memory_monitor:
        ctx = _prepare_run_context(
            config_path,
            mode=mode,
            grounding_strategy=grounding_strategy,
            output_dir=output_dir,
            solutions=solutions,
            min_length=min_length,
            max_length=max_length,
            progress_callback=progress_callback,
            optimized=optimized,
            max_workers=max_workers,
            memory_monitor=memory_monitor,
        )
        csv_writers = _RunCsvWriters(
            csv_dir=ctx.config.solutions_dir_path,
            mode=mode,
            grounding_strategy=grounding_strategy,
            fact_count=ctx.fact_bundle.fact_count,
            run_metadata=ctx.run_metadata,
            translation_builder=ctx.resolved_translation_builder,
            translation_schema=_translation_schema(ctx.fact_bundle),
            optimized_enabled=optimized,
            effective_parallel_mode=None,
            compressed_candidate_engaged=_compressed_candidate_engaged(ctx.fact_bundle),
        )
        csv_writers.step_writer.log_translation(
            translation_sec=ctx.translation_sec,
            translation_peak_rss_mb=ctx.translation_peak_rss_mb,
        )

        translation_summary_path, _ = _write_translation_summary(
            config=ctx.config,
            solution_dir=ctx.solution_dir,
            mode=mode,
            grounding_strategy=grounding_strategy,
            translation_builder=ctx.resolved_translation_builder,
            effective_translation_strategy=ctx.effective_translation_strategy,
            fact_bundle=ctx.fact_bundle,
            translation_path=None,
            translation_sec=ctx.translation_sec,
        )
        combined_peak_rss_mb = _combined_peak_mb(memory_monitor)
        csv_writers.summary_writer.log_summary(
            completed_stage="translate_only",
            timings=TimingBreakdown(
                translation_sec=ctx.translation_sec,
                grounding_sec=0.0,
                solving_sec=0.0,
                rendering_sec=0.0,
            ),
            translation_peak_rss_mb=ctx.translation_peak_combined_rss_mb,
            combined_peak_rss_mb=combined_peak_rss_mb,
            base_grounding_sec=0.0,
            base_grounding_peak_rss_mb=0.0,
            horizon_records=(),
            raw_solutions_found=0,
            raw_models_seen=0,
            solutions_found=0,
            effective_parallel_mode=None,
        )
        run_log_path = csv_writers.run_log_path
        run_summary_path = csv_writers.run_summary_path

        return TranslationRunResult(
            config=ctx.config,
            mode=mode,
            grounding_strategy=grounding_strategy,
            translation_builder=ctx.resolved_translation_builder,
            effective_translation_strategy=ctx.effective_translation_strategy,
            fact_bundle=ctx.fact_bundle,
            timings=TimingBreakdown(
                translation_sec=ctx.translation_sec,
                grounding_sec=0.0,
                solving_sec=0.0,
                rendering_sec=0.0,
            ),
            translation_peak_rss_mb=ctx.translation_peak_combined_rss_mb,
            combined_peak_rss_mb=combined_peak_rss_mb,
            translation_path=None,
            translation_summary_path=translation_summary_path,
            run_log_path=run_log_path,
            run_summary_path=run_summary_path,
        )


def run_once(
    config_path: str | Path,
    *,
    mode: str,
    grounding_strategy: str,
    output_dir: str | Path | None = None,
    solutions: int | None = None,
    min_length: int | None = None,
    max_length: int | None = None,
    parallel_mode: str | None = None,
    project_models: bool | None = None,
    graph_format: str = "png",
    render_graphs: bool = True,
    write_raw_answer_sets: bool = False,
    debug: bool = False,
    progress_callback: ProgressCallback = None,
    optimized: bool = False,
    max_workers: int = 1,
) -> RunResult:
    """Run one snakeAPE execution."""

    with ProcessTreePeakMonitor() as memory_monitor:
        ctx = _prepare_run_context(
            config_path,
            mode=mode,
            grounding_strategy=grounding_strategy,
            output_dir=output_dir,
            solutions=solutions,
            min_length=min_length,
            max_length=max_length,
            progress_callback=progress_callback,
            optimized=optimized,
            max_workers=max_workers,
            memory_monitor=memory_monitor,
        )
        config = ctx.config
        solution_dir = ctx.solution_dir
        fact_bundle = ctx.fact_bundle
        translation_sec = ctx.translation_sec
        translation_peak_rss_mb = ctx.translation_peak_combined_rss_mb
        run_metadata = ctx.run_metadata
        _translation_builder = ctx.resolved_translation_builder
        effective_parallel_mode = _effective_parallel_mode(mode, parallel_mode, fact_bundle)
        internal_solver_mode = _effective_internal_solver_mode(mode, fact_bundle)
        csv_writers = _RunCsvWriters(
            csv_dir=config.solutions_dir_path,
            mode=mode,
            grounding_strategy=grounding_strategy,
            fact_count=fact_bundle.fact_count,
            run_metadata=run_metadata,
            translation_builder=_translation_builder,
            translation_schema=_translation_schema(fact_bundle),
            optimized_enabled=optimized,
            effective_parallel_mode=effective_parallel_mode,
            compressed_candidate_engaged=_compressed_candidate_engaged(fact_bundle),
        )
        csv_writers.step_writer.log_translation(
            translation_sec=translation_sec,
            translation_peak_rss_mb=translation_peak_rss_mb,
        )

        effective_project_models = _effective_project_models(mode, project_models)
        diagnostic_counts_enabled = bool(debug or write_raw_answer_sets)
        capture_raw_models = bool(write_raw_answer_sets)
        if effective_parallel_mode:
            _report(progress_callback, f"Step 3a: effective solve parallel mode is {effective_parallel_mode}.")
        remaining_timeout = config.timeout_sec - translation_sec
        _timed_out = False
        _solve_start = perf_counter()
        if remaining_timeout <= 0:
            _timed_out = True
            _report(
                progress_callback,
                f"Translation already exceeded timeout ({config.timeout_sec}s); skipping solve.",
            )
            combined_peak_rss_mb = _combined_peak_mb(memory_monitor)
            csv_writers.step_writer.log_timeout(elapsed_ms=0, memory_used_mb=combined_peak_rss_mb)
            csv_writers.summary_writer.log_summary(
                completed_stage="translation_timeout",
                timings=TimingBreakdown(
                    translation_sec=translation_sec,
                    grounding_sec=0.0,
                    solving_sec=0.0,
                    rendering_sec=0.0,
                ),
                translation_peak_rss_mb=translation_peak_rss_mb,
                combined_peak_rss_mb=combined_peak_rss_mb,
                base_grounding_sec=0.0,
                base_grounding_peak_rss_mb=0.0,
                horizon_records=(),
                raw_solutions_found=0,
                raw_models_seen=0,
                solutions_found=0,
                effective_parallel_mode=effective_parallel_mode,
            )
            return _timed_out_run_result(
                config=config,
                mode=mode,
                grounding_strategy=grounding_strategy,
                fact_bundle=fact_bundle,
                translation_sec=translation_sec,
                translation_peak_rss_mb=translation_peak_rss_mb,
                combined_peak_rss_mb=combined_peak_rss_mb,
                solve_start=_solve_start,
                completed_stage="translation_timeout",
                run_log_path=csv_writers.run_log_path,
                run_summary_path=csv_writers.run_summary_path,
            )

        try:
            solve_output, _timed_out, horizon_peak_rss_by_horizon = _run_solve_in_worker(
                mode=internal_solver_mode,
                config=config,
                fact_bundle=fact_bundle,
                capture_raw_models=capture_raw_models,
                diagnostic_counts_enabled=diagnostic_counts_enabled,
                parallel_mode=effective_parallel_mode,
                project_models=effective_project_models,
                remaining_timeout=remaining_timeout,
                progress_callback=progress_callback,
                memory_monitor=memory_monitor,
            )
        except RuntimeError as exc:
            if (
                mode == "multi-shot"
                and not optimized
                and _is_clasp_id_overflow(exc)
            ):
                _report(
                    progress_callback,
                    "Legacy multi-shot grounding exceeded clasp's internal id limit; "
                    "retrying with optimized direct precompute.",
                )
                return run_once(
                    config_path,
                    mode=mode,
                    grounding_strategy=grounding_strategy,
                    output_dir=output_dir,
                    solutions=solutions,
                    min_length=min_length,
                    max_length=max_length,
                    parallel_mode=parallel_mode,
                    project_models=project_models,
                    graph_format=graph_format,
                    render_graphs=render_graphs,
                    write_raw_answer_sets=write_raw_answer_sets,
                    debug=debug,
                    progress_callback=progress_callback,
                    optimized=True,
                    max_workers=max_workers,
                )
            raise

        if _timed_out:
            _report(progress_callback, "Configured timeout reached; stopping run immediately.")
            combined_peak_rss_mb = _combined_peak_mb(memory_monitor)
            csv_writers.step_writer.log_timeout(
                elapsed_ms=round(max(0.0, perf_counter() - _solve_start) * 1000),
                memory_used_mb=combined_peak_rss_mb,
            )
            csv_writers.summary_writer.log_summary(
                completed_stage="run_timeout",
                timings=TimingBreakdown(
                    translation_sec=translation_sec,
                    grounding_sec=0.0,
                    solving_sec=max(0.0, perf_counter() - _solve_start),
                    rendering_sec=0.0,
                ),
                translation_peak_rss_mb=translation_peak_rss_mb,
                combined_peak_rss_mb=combined_peak_rss_mb,
                base_grounding_sec=0.0,
                base_grounding_peak_rss_mb=0.0,
                horizon_records=(),
                raw_solutions_found=0,
                raw_models_seen=0,
                solutions_found=0,
                effective_parallel_mode=effective_parallel_mode,
            )
            return _timed_out_run_result(
                config=config,
                mode=mode,
                grounding_strategy=grounding_strategy,
                fact_bundle=fact_bundle,
                translation_sec=translation_sec,
                translation_peak_rss_mb=translation_peak_rss_mb,
                combined_peak_rss_mb=combined_peak_rss_mb,
                solve_start=_solve_start,
                completed_stage="run_timeout",
                run_log_path=csv_writers.run_log_path,
                run_summary_path=csv_writers.run_summary_path,
            )

        horizon_records = _records_with_combined_peak_rss(
            solve_output.horizon_records,
            memory_monitor=memory_monitor,
            peaks_by_horizon=horizon_peak_rss_by_horizon,
        )
        if solve_output.base_grounding_sec or solve_output.base_grounding_peak_rss_mb:
            csv_writers.step_writer.log_base_grounding(
                base_grounding_sec=solve_output.base_grounding_sec,
                base_grounding_peak_rss_mb=solve_output.base_grounding_peak_rss_mb,
            )
        for record in horizon_records:
            csv_writers.step_writer.log_horizon(record)

        candidate_solutions = tuple(
            reconstruct_solution(
                index + 1,
                symbols,
                dict(fact_bundle.tool_labels),
                workflow_input_dims=ctx.workflow_input_dims,
                tool_output_dims=ctx.tool_output_dims,
            )
            for index, symbols in enumerate(solve_output.raw_solutions)
        )
        solutions = tuple(
            reconstruct_solution(
                index + 1,
                symbols,
                dict(fact_bundle.tool_labels),
                workflow_input_dims=ctx.workflow_input_dims,
                tool_output_dims=ctx.tool_output_dims,
            )
            for index, symbols in enumerate(solve_output.solutions)
        )
        # ``solutions`` is the canonical stored result surface. ``candidate_solutions``
        # mirrors raw callback order and is only useful when raw diagnostics are
        # requested.

        answer_set_path: Path | None = None
        workflow_signature_path: Path | None = None
        graph_paths: tuple[Path, ...] = ()
        rendering_sec = 0.0

        _report(progress_callback, "Step 4: writing outputs and rendering artifacts...")
        render_start = perf_counter()
        if write_raw_answer_sets:
            # Raw answer sets are a debug artifact. The default machine-readable
            # result surface is the config/mode-specific workflow-signature file below.
            answer_set_path = solution_dir / _answer_sets_filename(
                config=config,
                mode=mode,
                optimized_enabled=optimized,
                effective_parallel_mode=effective_parallel_mode,
            )
            if solve_output.raw_solutions:
                _answer_set_content = "".join(
                    f"Answer Set {index}\n"
                    + " ".join(sorted(str(symbol) for symbol in symbols))
                    + "\n\n"
                    for index, symbols in enumerate(solve_output.raw_solutions, start=1)
                )
                answer_set_path.write_text(_answer_set_content, encoding="utf-8")
            else:
                _answer_set_content = "No answer sets found.\n"
                answer_set_path.write_text(_answer_set_content, encoding="utf-8")
        workflow_signature_path = write_workflow_signatures(
            solution_dir
            / _workflow_signatures_filename(
                config=config,
                mode=mode,
                optimized_enabled=optimized,
                effective_parallel_mode=effective_parallel_mode,
            ),
            solutions,
        )
        graph_path_list: list[Path] = []
        if render_graphs:
            figures_dir = solution_dir / "Figures"
            max_graphs = config.number_of_generated_graphs
            for solution in solutions[:max_graphs]:
                graph_path_list.extend(render_solution_graphs(figures_dir, solution, graph_format))
        graph_paths = tuple(graph_path_list)
        rendering_sec = perf_counter() - render_start
        _report(progress_callback, f"Step 4 complete: output writing/rendering finished after {rendering_sec:.3f}s.")

        combined_peak_rss_mb = _combined_peak_mb(memory_monitor)
        horizon_summary_path = solution_dir / "horizon_summary.json"
        horizon_summary_path.write_text(
            json.dumps(
                {
                    "mode": mode,
                    "grounding_strategy": grounding_strategy,
                    "internal_schema": fact_bundle.internal_schema,
                    "internal_solver_mode": internal_solver_mode,
                    "earliest_solution_step": fact_bundle.earliest_solution_step,
                    "effective_solve_start_horizon": _effective_solve_start_horizon(
                        config=config,
                        fact_bundle=fact_bundle,
                    ),
                    "python_precompute_enabled": fact_bundle.python_precompute_enabled,
                    "python_precompute_fact_count": fact_bundle.python_precompute_fact_count,
                    "python_precompute_stats": dict(sorted(fact_bundle.python_precompute_stats.items())),
                    "backend_stats": fact_bundle.backend_stats,
                    "workflow_input_compression": _workflow_input_compression_payload(
                        config=config,
                        mode=mode,
                        internal_solver_mode=internal_solver_mode,
                        compression_active=effective_project_models,
                    ),
                    "solve_callback_profile": _solve_callback_profile_payload(
                        horizon_records,
                        solving_sec=solve_output.solving_sec,
                    ),
                    "effective_parallel_mode": effective_parallel_mode,
                    "translation_peak_rss_mb": translation_peak_rss_mb,
                    "combined_peak_rss_mb": combined_peak_rss_mb,
                    "base_grounding_peak_rss_mb": solve_output.base_grounding_peak_rss_mb,
                    "base_grounding_sec": solve_output.base_grounding_sec,
                    "timed_out": _timed_out,
                    "horizons": _horizon_record_payload(horizon_records),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        csv_writers.summary_writer.log_summary(
            completed_stage="run",
            timings=TimingBreakdown(
                translation_sec=translation_sec,
                grounding_sec=solve_output.grounding_sec,
                solving_sec=solve_output.solving_sec,
                rendering_sec=rendering_sec,
            ),
            translation_peak_rss_mb=translation_peak_rss_mb,
            combined_peak_rss_mb=combined_peak_rss_mb,
            base_grounding_sec=solve_output.base_grounding_sec,
            base_grounding_peak_rss_mb=solve_output.base_grounding_peak_rss_mb,
            horizon_records=horizon_records,
            raw_solutions_found=len(solve_output.raw_solutions) if diagnostic_counts_enabled else 0,
            raw_models_seen=(
                sum(record.models_seen for record in horizon_records)
                if diagnostic_counts_enabled
                else 0
            ),
            solutions_found=len(solutions),
            grounded_horizons=tuple(record.horizon for record in horizon_records),
            effective_parallel_mode=effective_parallel_mode,
        )
        run_log_path = csv_writers.run_log_path
        run_summary_path = csv_writers.run_summary_path

        return RunResult(
            config=config,
            mode=mode,
            grounding_strategy=grounding_strategy,
            fact_bundle=fact_bundle,
            solutions=solutions,
            timings=TimingBreakdown(
                translation_sec=translation_sec,
                grounding_sec=solve_output.grounding_sec,
                solving_sec=solve_output.solving_sec,
                rendering_sec=rendering_sec,
            ),
            translation_peak_rss_mb=translation_peak_rss_mb,
            combined_peak_rss_mb=combined_peak_rss_mb,
            base_grounding_peak_rss_mb=solve_output.base_grounding_peak_rss_mb,
            base_grounding_sec=solve_output.base_grounding_sec,
            horizon_records=horizon_records,
            translation_path=None,
            answer_set_path=answer_set_path,
            solution_summary_path=None,
            workflow_signature_path=workflow_signature_path,
            graph_paths=graph_paths,
            raw_models_seen=(
                sum(record.models_seen for record in horizon_records)
                if diagnostic_counts_enabled
                else 0
            ),
            raw_answer_sets_found=len(solve_output.raw_solutions) if diagnostic_counts_enabled else 0,
            unique_solutions_found=len(solutions),
            diagnostic_counts_enabled=diagnostic_counts_enabled,
            timed_out=False,
            completed_stage="run",
            run_log_path=run_log_path,
            run_summary_path=run_summary_path,
        )


def run_ground_only(
    config_path: str | Path,
    *,
    mode: str,
    grounding_strategy: str,
    stage: str = "base",
    output_dir: str | Path | None = None,
    solutions: int | None = None,
    min_length: int | None = None,
    max_length: int | None = None,
    progress_callback: ProgressCallback = None,
    optimized: bool = False,
    max_workers: int = 1,
) -> GroundingRunResult:
    """Run translation plus grounding without solving."""

    with ProcessTreePeakMonitor() as memory_monitor:
        ctx = _prepare_run_context(
            config_path,
            mode=mode,
            grounding_strategy=grounding_strategy,
            output_dir=output_dir,
            solutions=solutions,
            min_length=min_length,
            max_length=max_length,
            progress_callback=progress_callback,
            optimized=optimized,
            max_workers=max_workers,
            memory_monitor=memory_monitor,
        )
        config = ctx.config
        solution_dir = ctx.solution_dir
        fact_bundle = ctx.fact_bundle
        translation_sec = ctx.translation_sec
        translation_peak_rss_mb = ctx.translation_peak_combined_rss_mb
        translation_builder = ctx.resolved_translation_builder
        effective_translation_strategy = ctx.effective_translation_strategy
        run_metadata = ctx.run_metadata
        csv_writers = _RunCsvWriters(
            csv_dir=config.solutions_dir_path,
            mode=mode,
            grounding_strategy=grounding_strategy,
            fact_count=fact_bundle.fact_count,
            run_metadata=run_metadata,
            translation_builder=translation_builder,
            translation_schema=_translation_schema(fact_bundle),
            optimized_enabled=optimized,
            effective_parallel_mode=None,
            compressed_candidate_engaged=_compressed_candidate_engaged(fact_bundle),
        )
        csv_writers.step_writer.log_translation(
            translation_sec=translation_sec,
            translation_peak_rss_mb=translation_peak_rss_mb,
        )

        translation_summary_path, translation_summary = _write_translation_summary(
            config=config,
            solution_dir=solution_dir,
            mode=mode,
            grounding_strategy=grounding_strategy,
            translation_builder=translation_builder,
            effective_translation_strategy=effective_translation_strategy,
            fact_bundle=fact_bundle,
            translation_path=None,
            translation_sec=translation_sec,
        )

        mode_config = _mode_config(mode)
        if not mode_config.supports_ground_only:
            raise ValueError(f"Ground-only runs do not support mode {mode}.")

        internal_solver_mode = _effective_internal_solver_mode(mode, fact_bundle)
        grounded_horizon_peaks: dict[int, float] = {}

        def _log_ground_horizon(record: HorizonRecord) -> None:
            combined_peak_mb = _combined_peak_mb(memory_monitor)
            grounded_horizon_peaks[record.horizon] = combined_peak_mb
            csv_writers.step_writer.log_horizon(replace(record, peak_rss_mb=combined_peak_mb))

        grounding_output = _GROUNDER_DISPATCH[internal_solver_mode](
            config,
            fact_bundle,
            stage=stage,
            progress_callback=progress_callback,
            base_grounding_callback=lambda sec, peak: csv_writers.step_writer.log_base_grounding(
                base_grounding_sec=sec,
                base_grounding_peak_rss_mb=peak,
            ),
            horizon_record_callback=_log_ground_horizon,
        )

        _report(
            progress_callback,
            f"Step 2 complete: grounding finished after {grounding_output.grounding_sec:.3f}s.",
        )

        horizon_records = _records_with_combined_peak_rss(
            grounding_output.horizon_records,
            memory_monitor=memory_monitor,
            peaks_by_horizon=grounded_horizon_peaks,
        )
        combined_peak_rss_mb = _combined_peak_mb(memory_monitor)
        grounding_summary_path = solution_dir / "grounding_summary.json"
        grounding_summary_path.write_text(
            json.dumps(
                {
                    "mode": mode,
                    "grounding_strategy": grounding_strategy,
                    "stage": stage,
                    "internal_schema": fact_bundle.internal_schema,
                    "internal_solver_mode": internal_solver_mode,
                    "fact_count": fact_bundle.fact_count,
                    "translation_path": None,
                    "translation_summary_path": str(translation_summary_path),
                    "earliest_solution_step": fact_bundle.earliest_solution_step,
                    "python_precompute_enabled": fact_bundle.python_precompute_enabled,
                    "python_precompute_fact_count": fact_bundle.python_precompute_fact_count,
                    "python_precompute_stats": dict(sorted(fact_bundle.python_precompute_stats.items())),
                    "backend_stats": fact_bundle.backend_stats,
                    "workflow_input_compression": _workflow_input_compression_payload(
                        config=config,
                        mode=mode,
                        internal_solver_mode=internal_solver_mode,
                    ),
                    "grounded_horizons": list(grounding_output.grounded_horizons),
                    "translation_peak_rss_mb": translation_peak_rss_mb,
                    "combined_peak_rss_mb": combined_peak_rss_mb,
                    "base_grounding_peak_rss_mb": grounding_output.base_grounding_peak_rss_mb,
                    "base_grounding_sec": grounding_output.base_grounding_sec,
                    "horizon_records": _horizon_record_payload(horizon_records),
                    "translation_sec": translation_sec,
                    "grounding_sec": grounding_output.grounding_sec,
                    "total_sec": translation_sec + grounding_output.grounding_sec,
                    "translation_summary": translation_summary,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        csv_writers.summary_writer.log_summary(
            completed_stage="ground_only_full" if stage == "full" else "ground_only_base",
            timings=TimingBreakdown(
                translation_sec=translation_sec,
                grounding_sec=grounding_output.grounding_sec,
                solving_sec=0.0,
                rendering_sec=0.0,
            ),
            translation_peak_rss_mb=translation_peak_rss_mb,
            combined_peak_rss_mb=combined_peak_rss_mb,
            base_grounding_sec=grounding_output.base_grounding_sec,
            base_grounding_peak_rss_mb=grounding_output.base_grounding_peak_rss_mb,
            horizon_records=horizon_records,
            raw_models_seen=0,
            raw_solutions_found=0,
            solutions_found=0,
            grounded_horizons=grounding_output.grounded_horizons,
            effective_parallel_mode=None,
        )
        run_log_path = csv_writers.run_log_path
        run_summary_path = csv_writers.run_summary_path

        return GroundingRunResult(
            config=config,
            mode=mode,
            grounding_strategy=grounding_strategy,
            stage=stage,
            fact_bundle=fact_bundle,
            timings=TimingBreakdown(
                translation_sec=translation_sec,
                grounding_sec=grounding_output.grounding_sec,
                solving_sec=0.0,
                rendering_sec=0.0,
            ),
            translation_peak_rss_mb=translation_peak_rss_mb,
            combined_peak_rss_mb=combined_peak_rss_mb,
            base_grounding_peak_rss_mb=grounding_output.base_grounding_peak_rss_mb,
            base_grounding_sec=grounding_output.base_grounding_sec,
            horizon_records=horizon_records,
            translation_path=None,
            translation_summary_path=translation_summary_path,
            grounding_summary_path=grounding_summary_path,
            grounded_horizons=grounding_output.grounded_horizons,
            run_log_path=run_log_path,
            run_summary_path=run_summary_path,
        )
