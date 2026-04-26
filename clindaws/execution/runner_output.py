"""Per-horizon CSV log writers and per-run summary writers."""

from __future__ import annotations

import csv
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from clindaws.core.models import HorizonRecord, TimingBreakdown, TranslationBuilder
from clindaws.execution.runner_modes import _solver_approach, _solver_family


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
        run_metadata: dict[str, object],
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

    def log_translation(self, *, translation_sec: float) -> None:
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
            run_metadata=run_metadata,
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
