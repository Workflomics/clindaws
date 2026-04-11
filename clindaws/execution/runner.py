"""High-level execution entrypoints."""

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
from clindaws.execution.direct_python_precompute import apply_direct_python_precompute
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
from clindaws.rendering.rendering import (
    render_solution_graphs,
    write_workflow_signatures,
)
from clindaws.core.runtime_stats import current_peak_rss_mb
from clindaws.execution.solver import (
    ground_multi_shot,
    ground_multi_shot_compressed_candidate,
    program_paths_for_mode,
    solve_multi_shot,
    solve_multi_shot_compressed_candidate,
    solve_single_shot,
)
from clindaws.core.tool_annotations import (
    load_candidate_tool_annotations,
    load_direct_tool_annotations,
)
from clindaws.translators.translator_direct import (
    build_fact_bundle,
    build_fact_bundle_ape_multi_shot,
)
from clindaws.translators.translator_compressed_candidate import build_compressed_candidate_fact_bundle
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
    "dynamic_signature_support_class",
    "dynamic_support_class_bindable_producer_port",
    "dynamic_signature_profile",
    "dynamic_profile_accepts",
    "dynamic_candidate_output_port",
    "dynamic_candidate_output_multiplicity",
    "dynamic_candidate_total_output_multiplicity",
    "dynamic_candidate_output_singleton",
    "dynamic_candidate_output_choice_value",
    "dynamic_candidate_output_declared_type",
)
RUNTIME_TRANSLATION_BUILDER = "runtime_legacy"
COMPRESSED_CANDIDATE_TRANSLATION_BUILDER = "candidate_compressed"
ProgressCallback = Callable[[str], None] | None


@dataclass(frozen=True)
class _ModeConfig:
    solver_family: str
    solver_approach: str
    translation_pathway: str
    translation_builder: TranslationBuilder
    supports_ground_only: bool


_MODE_CONFIGS = {
    "single-shot": _ModeConfig(
        solver_family="single-shot",
        solver_approach="legacy",
        translation_pathway="normal",
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
    "multi-shot-compressed-candidate": _ModeConfig(
        solver_family="multi-shot",
        solver_approach="compressed_candidate",
        translation_pathway="compressed_candidate",
        translation_builder=COMPRESSED_CANDIDATE_TRANSLATION_BUILDER,
        supports_ground_only=True,
    ),
}

_SOLVER_DISPATCH = {
    "single-shot": solve_single_shot,
    "multi-shot": solve_multi_shot,
    "multi-shot-compressed-candidate": solve_multi_shot_compressed_candidate,
}

_GROUNDER_DISPATCH = {
    "multi-shot": ground_multi_shot,
    "multi-shot-compressed-candidate": ground_multi_shot_compressed_candidate,
}


@dataclass(frozen=True)
class RunContext:
    """All translation-phase results, passed from _prepare_run_context to callers."""

    config: SnakeConfig
    solution_dir: Path
    fact_bundle: FactBundle
    translation_sec: float
    translation_peak_rss_mb: float
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
    if translation_pathway == "compressed_candidate":
        return "python"
    if translation_pathway == "ape_multi_shot":
        return "ape_clingo_legacy"
    return grounding_strategy


def _load_tools_for_mode(config, translation_pathway: str):
    if translation_pathway == "compressed_candidate":
        return load_candidate_tool_annotations(config.tool_annotations_path, config.ontology_prefix)
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


def _drain_progress_queue(
    progress_queue: multiprocessing.queues.Queue | None,
    progress_callback: ProgressCallback,
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
        _report(progress_callback, str(message))


def _solve_worker_entrypoint(
    *,
    mode: str,
    config,
    fact_bundle,
    capture_raw_models: bool,
    parallel_mode: str | None,
    project_models: bool,
    result_queue: multiprocessing.queues.Queue,
    progress_queue: multiprocessing.queues.Queue,
) -> None:
    def _worker_progress(message: str) -> None:
        progress_queue.put(message)

    try:
        solve_output = _SOLVER_DISPATCH[mode](
            config,
            fact_bundle,
            progress_callback=_worker_progress,
            capture_raw_models=capture_raw_models,
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
    parallel_mode: str | None,
    project_models: bool,
    remaining_timeout: float,
    progress_callback: ProgressCallback,
) -> tuple[object, bool]:
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

    while True:
        _drain_progress_queue(progress_queue, progress_callback)
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
        _drain_progress_queue(progress_queue, progress_callback)
        return _empty_solve_output(), True

    process.join()
    _drain_progress_queue(progress_queue, progress_callback)

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

    return _deserialize_solve_output(payload["solve_output"]), False


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
    return fact_bundle.internal_schema == "compressed_candidate_fallback"


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
            rows.append({name: row.get(name, "") for name in fieldnames})

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
        "raw_models_seen",
        "raw_solutions_found",
        "solutions_found",
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
        self.cumulative_raw_models_seen = 0
        self.cumulative_raw_solutions = 0
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
        self.cumulative_raw_models_seen += record.models_seen
        self.cumulative_raw_solutions += record.models_stored
        self.cumulative_unique_solutions += record.unique_workflows_stored
        setup_grounding_ms = round(record.grounding_sec * 1000)
        memory_used_mb = record.peak_rss_mb or 0.0
        if record.horizon == 1:
            setup_grounding_ms += self.base_grounding_ms
            memory_used_mb = max(memory_used_mb, self.base_grounding_peak_rss_mb)
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
                "raw_models_seen": self.cumulative_raw_models_seen,
                "raw_solutions_found": self.cumulative_raw_solutions,
                "solutions_found": self.cumulative_unique_solutions,
                "constraints_used": self.constraints_used,
                "timed_out": "false",
            }
        )

    def log_timeout(self, *, elapsed_ms: int) -> None:
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
                "memory_used_mb": "",
                "raw_models_seen": self.cumulative_raw_models_seen,
                "raw_solutions_found": self.cumulative_raw_solutions,
                "solutions_found": self.cumulative_unique_solutions,
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
        return "compressed_candidate"
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
            "compressed_candidate"
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

    if translation_pathway in {"dynamic", "compressed_candidate"} and translation_schema != "compressed_candidate":
        warnings.append(
            f"{mode} expects compressed-candidate translation, but the emitted translation schema is {translation_schema}."
        )
    if translation_schema == "compressed_candidate" and encoding_summary["schema"] != "compressed_candidate":
        warnings.append(
            "Compressed-candidate translation is not compatible with the selected encoding."
        )

    if translation_schema == "candidate" and not any(
        encoding_presence[name] for name in ("tool_candidate", "candidate_in", "candidate_out")
    ):
        warnings.append(
            "Translated facts use candidate predicates, but the selected encoding does not reference candidate predicates."
        )

    if translation_schema == "compressed_candidate" and not any(
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
            "Translated facts use compressed-candidate predicates, but the selected encoding does not reference the compressed-candidate predicate family."
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





def _translation_summary_payload(
    *,
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
        "translation_path": str(translation_path) if translation_path else None,
        "translation_sec": translation_sec,
        "predicate_counts": dict(sorted(fact_bundle.predicate_counts.items())),
        "translation_cache_stats": dict(sorted(fact_bundle.cache_stats.items())),
        "translation_emit_stats": dict(sorted(fact_bundle.emit_stats.items())),
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
        "compressed_candidate_fallback",
    }:
        return None
    cpu_count = os.cpu_count() or 1
    if cpu_count < 2 or len(fact_bundle.tool_labels) < 200:
        return None
    workers = min(cpu_count, 4)
    return f"{workers},compete"


def _legacy_direct_bundle(
    config: SnakeConfig,
    ontology: Ontology,
    tools,
):
    return replace(
        build_fact_bundle_ape_multi_shot(config, ontology, tools),
        internal_schema="legacy_direct",
        internal_solver_mode="multi-shot",
    )


def _compressed_candidate_internal_bundle(
    config: SnakeConfig,
    ontology: Ontology,
    tools,
    *,
    max_workers: int = 1,
):
    return replace(
        build_compressed_candidate_fact_bundle(config, ontology, tools, max_workers=max_workers),
        internal_schema="compressed_candidate_fallback",
        internal_solver_mode="multi-shot-compressed-candidate",
    )


def _should_consider_direct_candidate_fallback(
    mode: str,
    *,
    optimized: bool,
    tools,
) -> bool:
    return (
        mode == "multi-shot"
        and optimized
        and len(tools) >= 200
    )


def _should_force_direct_candidate_fallback(tools) -> bool:
    return len(tools) >= 500


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
        fact_bundle = _legacy_direct_bundle(config, ontology, tools)
        if optimized:
            if _should_consider_direct_candidate_fallback(mode, optimized=optimized, tools=tools):
                compressed_candidate_tools = load_candidate_tool_annotations(
                    config.tool_annotations_path,
                    config.ontology_prefix,
                )
                if _should_force_direct_candidate_fallback(tools):
                    _report(progress_callback, "Step 1b: forcing compressed-candidate internal fallback for heavy direct run.")
                    fact_bundle = _compressed_candidate_internal_bundle(config, ontology, compressed_candidate_tools, max_workers=max_workers)
                    resolved_translation_builder = COMPRESSED_CANDIDATE_TRANSLATION_BUILDER
                    _report(
                        progress_callback,
                        "Step 1b complete: selected compressed-candidate internal schema "
                        f"with {fact_bundle.fact_count} facts.",
                    )
                else:
                    _report(progress_callback, "Step 1b: Python direct precompute started.")
                    optimized_direct_bundle = apply_direct_python_precompute(mode, config, ontology, tools, fact_bundle)
                    _report(
                        progress_callback,
                        "Step 1b complete: Python direct precompute emitted "
                        f"{optimized_direct_bundle.python_precompute_fact_count} helper facts.",
                    )
                    fact_bundle = optimized_direct_bundle
                    _report(progress_callback, "Step 1c: evaluating compressed-candidate internal fallback.")
                    compressed_candidate_bundle = _compressed_candidate_internal_bundle(config, ontology, compressed_candidate_tools, max_workers=max_workers)
                    if compressed_candidate_bundle.fact_count < fact_bundle.fact_count:
                        fact_bundle = compressed_candidate_bundle
                        resolved_translation_builder = COMPRESSED_CANDIDATE_TRANSLATION_BUILDER
                        _report(
                            progress_callback,
                            "Step 1c complete: selected compressed-candidate internal schema "
                            f"({compressed_candidate_bundle.fact_count} facts vs {optimized_direct_bundle.fact_count}).",
                        )
                    else:
                        _report(
                            progress_callback,
                            "Step 1c complete: kept legacy direct optimization path "
                            f"({optimized_direct_bundle.fact_count} facts vs {compressed_candidate_bundle.fact_count}).",
                        )
            else:
                _report(progress_callback, "Step 1b: Python direct precompute started.")
                optimized_direct_bundle = apply_direct_python_precompute(mode, config, ontology, tools, fact_bundle)
                _report(
                    progress_callback,
                    "Step 1b complete: Python direct precompute emitted "
                    f"{optimized_direct_bundle.python_precompute_fact_count} helper facts.",
                )
                fact_bundle = optimized_direct_bundle

    elif mode_config.translation_pathway == "normal":
        fact_bundle = replace(
            build_fact_bundle(config, ontology, tools, effective_translation_strategy),
            internal_schema="legacy_direct",
            internal_solver_mode="single-shot",
        )
        if optimized:
            _report(progress_callback, "Step 1b: Python direct precompute started.")
            fact_bundle = apply_direct_python_precompute(mode, config, ontology, tools, fact_bundle)
            _report(
                progress_callback,
                "Step 1b complete: Python direct precompute emitted "
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
) -> RunContext:
    """Load config and build the fact bundle for a run."""

    mode_config = _mode_config(mode)
    if optimized and mode not in {"single-shot", "multi-shot"}:
        raise ValueError("--optimized supports only single-shot and multi-shot.")
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
    _report(progress_callback, f"Step 1 complete: translation finished after {translation_sec:.3f}s.")

    return RunContext(
        config=config,
        solution_dir=solution_dir,
        fact_bundle=fact_bundle,
        translation_sec=translation_sec,
        translation_peak_rss_mb=translation_peak_rss_mb,
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
        solution_dir=ctx.solution_dir,
        mode=mode,
        grounding_strategy=grounding_strategy,
        translation_builder=ctx.resolved_translation_builder,
        effective_translation_strategy=ctx.effective_translation_strategy,
        fact_bundle=ctx.fact_bundle,
        translation_path=None,
        translation_sec=ctx.translation_sec,
    )
    csv_writers.summary_writer.log_summary(
        completed_stage="translate_only",
        timings=TimingBreakdown(
            translation_sec=ctx.translation_sec,
            grounding_sec=0.0,
            solving_sec=0.0,
            rendering_sec=0.0,
        ),
        translation_peak_rss_mb=ctx.translation_peak_rss_mb,
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
        translation_peak_rss_mb=ctx.translation_peak_rss_mb,
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
    progress_callback: ProgressCallback = None,
    optimized: bool = False,
    max_workers: int = 1,
) -> RunResult:
    """Run one snakeAPE execution."""

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
    )
    config = ctx.config
    solution_dir = ctx.solution_dir
    fact_bundle = ctx.fact_bundle
    translation_sec = ctx.translation_sec
    translation_peak_rss_mb = ctx.translation_peak_rss_mb
    run_metadata = ctx.run_metadata
    _translation_builder = ctx.resolved_translation_builder
    effective_parallel_mode = _effective_parallel_mode(mode, parallel_mode, fact_bundle)
    internal_solver_mode = _effective_internal_solver_mode(mode, fact_bundle)

    effective_project_models = _effective_project_models(mode, project_models)
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

    if remaining_timeout <= 0:
        solve_output = _empty_solve_output()
    else:
        try:
            solve_output, _timed_out = _run_solve_in_worker(
                mode=internal_solver_mode,
                config=config,
                fact_bundle=fact_bundle,
                capture_raw_models=True,
                parallel_mode=effective_parallel_mode,
                project_models=effective_project_models,
                remaining_timeout=remaining_timeout,
                progress_callback=progress_callback,
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
                    progress_callback=progress_callback,
                    optimized=True,
                    max_workers=max_workers,
                )
            raise

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

    if solve_output.base_grounding_sec or solve_output.base_grounding_peak_rss_mb:
        csv_writers.step_writer.log_base_grounding(
            base_grounding_sec=solve_output.base_grounding_sec,
            base_grounding_peak_rss_mb=solve_output.base_grounding_peak_rss_mb,
        )
    for record in solve_output.horizon_records:
        csv_writers.step_writer.log_horizon(record)

    candidate_solutions = tuple(
        reconstruct_solution(index + 1, symbols, dict(fact_bundle.tool_labels))
        for index, symbols in enumerate(solve_output.raw_solutions)
    )
    solutions = tuple(
        reconstruct_solution(index + 1, symbols, dict(fact_bundle.tool_labels))
        for index, symbols in enumerate(solve_output.solutions)
    )

    answer_set_path: Path | None = None
    
    workflow_signature_path: Path | None = None
    graph_paths: tuple[Path, ...] = ()
    rendering_sec = 0.0

    horizon_summary_path = solution_dir / "horizon_summary.json"
    horizon_summary_path.write_text(
        json.dumps(
            {
                "mode": mode,
                "grounding_strategy": grounding_strategy,
                "internal_schema": fact_bundle.internal_schema,
                "internal_solver_mode": internal_solver_mode,
                "earliest_solution_step": fact_bundle.earliest_solution_step,
                "effective_solve_start_horizon": max(
                    config.solution_length_min,
                    fact_bundle.earliest_solution_step,
                ),
                "python_precompute_enabled": fact_bundle.python_precompute_enabled,
                "python_precompute_fact_count": fact_bundle.python_precompute_fact_count,
                "python_precompute_stats": dict(sorted(fact_bundle.python_precompute_stats.items())),
                "effective_parallel_mode": effective_parallel_mode,
                "translation_peak_rss_mb": translation_peak_rss_mb,
                "base_grounding_peak_rss_mb": solve_output.base_grounding_peak_rss_mb,
                "base_grounding_sec": solve_output.base_grounding_sec,
                "timed_out": _timed_out,
                "horizons": _horizon_record_payload(solve_output.horizon_records),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    if not _timed_out:
        _report(progress_callback, "Step 4: writing outputs and rendering artifacts...")
        render_start = perf_counter()
        if write_raw_answer_sets or config.debug_mode:
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
            solution_dir / "workflow_signatures.json",
            solutions,
        )
        graph_path_list: list[Path] = []
        if render_graphs:
            figures_dir = solution_dir / "Figures"
            max_graphs = config.number_of_generated_graphs
            for solution in candidate_solutions[:max_graphs]:
                graph_path_list.extend(render_solution_graphs(figures_dir, solution, graph_format))
        graph_paths = tuple(graph_path_list)
        rendering_sec = perf_counter() - render_start
        _report(progress_callback, f"Step 4 complete: output writing/rendering finished after {rendering_sec:.3f}s.")

    if _timed_out:
        _elapsed_since_solve_start_ms = round((perf_counter() - _solve_start) * 1000)
        csv_writers.step_writer.log_timeout(elapsed_ms=_elapsed_since_solve_start_ms)
    _timeout_solving_sec = (
        max((perf_counter() - _solve_start), solve_output.solving_sec)
        if _timed_out and remaining_timeout > 0
        else solve_output.solving_sec
    )
    _completed_stage = (
        "translation_timeout" if _timed_out and remaining_timeout <= 0
        else "run_timeout" if _timed_out
        else "run"
    )
    csv_writers.summary_writer.log_summary(
        completed_stage=_completed_stage,
        timings=TimingBreakdown(
            translation_sec=translation_sec,
            grounding_sec=solve_output.grounding_sec,
            solving_sec=_timeout_solving_sec,
            rendering_sec=rendering_sec,
        ),
        translation_peak_rss_mb=translation_peak_rss_mb,
        base_grounding_sec=solve_output.base_grounding_sec,
        base_grounding_peak_rss_mb=solve_output.base_grounding_peak_rss_mb,
        horizon_records=solve_output.horizon_records,
        raw_solutions_found=len(solve_output.raw_solutions),
        raw_models_seen=sum(record.models_seen for record in solve_output.horizon_records),
        solutions_found=len(solutions),
        grounded_horizons=tuple(record.horizon for record in solve_output.horizon_records),
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
            solving_sec=_timeout_solving_sec,
            rendering_sec=rendering_sec,
        ),
        translation_peak_rss_mb=translation_peak_rss_mb,
        base_grounding_peak_rss_mb=solve_output.base_grounding_peak_rss_mb,
        base_grounding_sec=solve_output.base_grounding_sec,
        horizon_records=solve_output.horizon_records,
        translation_path=None,
        answer_set_path=answer_set_path,
        solution_summary_path=None,
        workflow_signature_path=workflow_signature_path,
        graph_paths=graph_paths,
        raw_models_seen=sum(record.models_seen for record in solve_output.horizon_records),
        raw_answer_sets_found=len(solve_output.raw_solutions),
        unique_solutions_found=len(solutions),
        timed_out=_timed_out,
        completed_stage=_completed_stage,
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
    )
    config = ctx.config
    solution_dir = ctx.solution_dir
    fact_bundle = ctx.fact_bundle
    translation_sec = ctx.translation_sec
    translation_peak_rss_mb = ctx.translation_peak_rss_mb
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
        raise ValueError("Ground-only runs support only multi-shot mode.")

    internal_solver_mode = _effective_internal_solver_mode(mode, fact_bundle)

    grounding_output = _GROUNDER_DISPATCH[internal_solver_mode](
        config,
        fact_bundle,
        stage=stage,
        progress_callback=progress_callback,
        base_grounding_callback=lambda sec, peak: csv_writers.step_writer.log_base_grounding(
            base_grounding_sec=sec,
            base_grounding_peak_rss_mb=peak,
        ),
        horizon_record_callback=csv_writers.step_writer.log_horizon,
    )

    _report(
        progress_callback,
        f"Step 2 complete: grounding finished after {grounding_output.grounding_sec:.3f}s.",
    )

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
                "grounded_horizons": list(grounding_output.grounded_horizons),
                "translation_peak_rss_mb": translation_peak_rss_mb,
                "base_grounding_peak_rss_mb": grounding_output.base_grounding_peak_rss_mb,
                "base_grounding_sec": grounding_output.base_grounding_sec,
                "horizon_records": _horizon_record_payload(grounding_output.horizon_records),
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
        base_grounding_sec=grounding_output.base_grounding_sec,
        base_grounding_peak_rss_mb=grounding_output.base_grounding_peak_rss_mb,
        horizon_records=grounding_output.horizon_records,
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
        base_grounding_peak_rss_mb=grounding_output.base_grounding_peak_rss_mb,
        base_grounding_sec=grounding_output.base_grounding_sec,
        horizon_records=grounding_output.horizon_records,
        translation_path=None,
        translation_summary_path=translation_summary_path,
        grounding_summary_path=grounding_summary_path,
        grounded_horizons=grounding_output.grounded_horizons,
        run_log_path=run_log_path,
        run_summary_path=run_summary_path,
    )


