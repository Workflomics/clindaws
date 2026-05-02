"""Small pure helpers for runner: progress, peaks, filenames, validation."""

from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path

from clindaws.core.config import SnakeConfig
from clindaws.core.models import HorizonRecord
from clindaws.core.runtime_stats import ProcessTreePeakMonitor
from clindaws.execution.runner_modes import ProgressCallback


def _report(progress_callback: ProgressCallback, message: str) -> None:
    if progress_callback is not None:
        progress_callback(message)


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


def _compressed_candidate_engaged(fact_bundle) -> bool:
    return fact_bundle.internal_solver_mode == "multi-shot-optimized-candidate"


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
    optimized_enabled: bool,
) -> str | None:
    """Resolve the effective solve parallel mode for one run."""

    if parallel_mode is not None:
        return parallel_mode

    if optimized_enabled:
        return "2,split"

    return None


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
