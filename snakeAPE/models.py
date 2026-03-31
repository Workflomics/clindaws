"""Core data models for snakeAPE."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Mapping


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
    lazy_input_value_count: int | None = None
    lazy_output_value_count: int | None = None
    lazy_input_port_value_counts: tuple[int, ...] = ()
    lazy_output_port_value_counts: tuple[int, ...] = ()
    lazy_cross_product_estimate: int | None = None

    @property
    def expansion_score(self) -> int:
        if self.candidate_count is not None:
            return self.candidate_count
        if self.lazy_cross_product_estimate is not None:
            return self.lazy_cross_product_estimate
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
class SnakeConfig:
    """Normalized configuration resolved from an APE-style config."""

    config_path: Path
    base_dir: Path
    ontology_path: Path
    tool_annotations_path: Path
    constraints_path: Path | None
    solutions_dir_path: Path
    ontology_prefix: str
    tools_taxonomy_root: str
    data_dimensions_taxonomy_roots: tuple[str, ...]
    strict_tool_annotations: bool
    solution_length_min: int
    solution_length_max: int
    solutions: int
    timeout_sec: int
    number_of_generated_graphs: int
    inputs: tuple[Mapping[str, tuple[str, ...]], ...]
    outputs: tuple[Mapping[str, tuple[str, ...]], ...]
    use_workflow_input: str
    use_all_generated_data: str
    tool_seq_repeat: bool
    debug_mode: bool


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
    translation_path: Path
    answer_set_path: Path
    solution_summary_path: Path
    graph_paths: tuple[Path, ...]
    raw_answer_sets_found: int = 0
    unique_solutions_found: int = 0
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
    translation_path: Path
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
    translation_path: Path
    translation_summary_path: Path
    run_log_path: Path | None = None
    run_summary_path: Path | None = None


@dataclass(frozen=True)
class BenchmarkRecord:
    """Benchmark data for one strategy execution."""

    mode: SolverMode
    strategy: GroundingStrategy
    translation_sec: float
    grounding_sec: float
    solving_sec: float
    rendering_sec: float
    total_sec: float
    fact_count: int
    solutions_found: int
    raw_solutions_found: int
    lengths: tuple[int, ...]
    repetition: int


@dataclass(frozen=True)
class BenchmarkResult:
    """Grounding benchmark output."""

    records: tuple[BenchmarkRecord, ...]
    output_path: Path
