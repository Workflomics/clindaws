"""Core data models for snakeAPE."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping

from clindaws.core.config import SnakeConfig as SnakeConfig  # re-export


GroundingStrategy = str
SolverMode = str
GroundOnlyStage = str
TranslationBuilder = str


@dataclass(frozen=True)
class ToolExpansionStat:
    """Translation expansion metrics for one tool."""

    tool_id: str
    tool_label: str
    input_ports: int
    output_ports: int
    input_variant_count: int
    output_variant_count: int
    candidate_count: int | None = None
    dynamic_input_value_count: int | None = None
    dynamic_output_value_count: int | None = None
    dynamic_input_port_value_counts: tuple[int, ...] = ()
    dynamic_output_port_value_counts: tuple[int, ...] = ()
    dynamic_cross_product_estimate: int | None = None

    @property
    def expansion_score(self) -> int:
        if self.candidate_count is not None:
            return self.candidate_count
        if self.dynamic_cross_product_estimate is not None:
            return self.dynamic_cross_product_estimate
        return self.input_variant_count * max(self.output_variant_count, 1)


@dataclass(frozen=True)
class ToolPortSpec:
    """A tool input or output port specification."""

    dimensions: Mapping[str, tuple[str, ...]]

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Iterable[str]]) -> "ToolPortSpec":
        normalized = {
            str(name): tuple(str(value) for value in values)
            for name, values in mapping.items()
        }
        return cls(dimensions=normalized)


@dataclass(frozen=True)
class ToolMode:
    """A tool mode from the annotation JSON."""

    label: str
    mode_id: str
    taxonomy_operations: tuple[str, ...]
    inputs: tuple[ToolPortSpec, ...] = ()
    outputs: tuple[ToolPortSpec, ...] = ()
    implementation: str | None = None



@dataclass(frozen=True)
class FactBundle:
    """Facts and metadata prepared for the solver."""

    facts: str
    fact_count: int
    tool_labels: Mapping[str, str]
    tool_input_signatures: Mapping[str, tuple[tuple[tuple[str, tuple[str, ...]], ...], ...]]
    workflow_input_ids: tuple[str, ...]
    goal_count: int
    predicate_counts: Mapping[str, int]
    tool_stats: tuple[ToolExpansionStat, ...]
    cache_stats: Mapping[str, int]
    emit_stats: Mapping[str, int]
    earliest_solution_step: int = 1
    python_precomputed_facts: str = ""
    python_precompute_enabled: bool = False
    python_precompute_fact_count: int = 0
    python_precompute_stats: Mapping[str, int] = field(default_factory=dict)
    internal_schema: str = ""
    internal_solver_mode: str = ""


@dataclass(frozen=True)
class ArtifactRef:
    """Reference to a workflow artifact."""

    ref_id: str
    created_at: int
    created_by_tool: str | None = None
    created_by_label: str | None = None
    dims: Mapping[str, set[str]] = field(default_factory=dict)


@dataclass(frozen=True)
class Binding:
    """A step input binding."""

    port_id: str
    artifact_id: str


@dataclass(frozen=True)
class WorkflowStep:
    """A concrete tool execution step."""

    time: int
    tool_id: str
    tool_label: str
    bindings: tuple[Binding, ...]
    outputs: tuple[str, ...]


@dataclass(frozen=True)
class WorkflowSolution:
    """A reconstructed workflow solution."""

    index: int
    steps: tuple[WorkflowStep, ...]
    artifacts: Mapping[str, ArtifactRef]
    goal_outputs: Mapping[int, str]
    signature_bindings: tuple[str, ...]
    goal_bindings: tuple[str, ...]
    workflow_signature_key: tuple[object, ...]
    canonical_key: tuple[object, ...]

    @property
    def tool_sequence(self) -> tuple[str, ...]:
        return tuple(step.tool_label for step in self.steps)

    @property
    def length(self) -> int:
        return len(self.steps)


@dataclass(frozen=True)
class TimingBreakdown:
    """Timing data for a run."""

    translation_sec: float
    grounding_sec: float
    solving_sec: float
    rendering_sec: float

    @property
    def total_sec(self) -> float:
        return (
            self.translation_sec
            + self.grounding_sec
            + self.solving_sec
            + self.rendering_sec
        )


@dataclass(frozen=True)
class HorizonRecord:
    """Per-horizon grounding and solving metrics."""

    horizon: int
    grounding_sec: float
    solving_sec: float
    peak_rss_mb: float
    satisfiable: bool
    models_seen: int
    models_stored: int
    unique_workflows_seen: int
    unique_workflows_stored: int
    available_artifacts_at_step: int | None = None
    eligible_artifacts_at_step: int | None = None
    eligible_workflow_inputs_at_step: int | None = None
    eligible_produced_outputs_at_step: int | None = None
    bind_choice_domain_size_at_step: int | None = None
    grounding_parts: tuple[tuple[str, float], ...] = ()


@dataclass(frozen=True)
class RunResult:
    """End-to-end execution result."""

    config: SnakeConfig
    mode: SolverMode
    grounding_strategy: GroundingStrategy
    fact_bundle: FactBundle
    solutions: tuple[WorkflowSolution, ...]
    timings: TimingBreakdown
    translation_peak_rss_mb: float
    base_grounding_peak_rss_mb: float
    base_grounding_sec: float
    horizon_records: tuple[HorizonRecord, ...]
    translation_path: Path | None
    answer_set_path: Path | None
    solution_summary_path: Path | None
    workflow_signature_path: Path | None
    graph_paths: tuple[Path, ...]
    raw_models_seen: int = 0
    raw_answer_sets_found: int = 0
    unique_solutions_found: int = 0
    timed_out: bool = False
    completed_stage: str = "run"
    run_log_path: Path | None = None
    run_summary_path: Path | None = None


@dataclass(frozen=True)
class GroundingRunResult:
    """Translation + grounding-only execution result."""

    config: SnakeConfig
    mode: SolverMode
    grounding_strategy: GroundingStrategy
    stage: GroundOnlyStage
    fact_bundle: FactBundle
    timings: TimingBreakdown
    translation_peak_rss_mb: float
    base_grounding_peak_rss_mb: float
    base_grounding_sec: float
    horizon_records: tuple[HorizonRecord, ...]
    translation_path: Path | None
    translation_summary_path: Path
    grounding_summary_path: Path
    grounded_horizons: tuple[int, ...]
    run_log_path: Path | None = None
    run_summary_path: Path | None = None


@dataclass(frozen=True)
class TranslationRunResult:
    """Translation-only execution result."""

    config: SnakeConfig
    mode: SolverMode
    grounding_strategy: GroundingStrategy
    translation_builder: TranslationBuilder
    effective_translation_strategy: GroundingStrategy
    fact_bundle: FactBundle
    timings: TimingBreakdown
    translation_peak_rss_mb: float
    translation_path: Path | None
    translation_summary_path: Path
    run_log_path: Path | None = None
    run_summary_path: Path | None = None
