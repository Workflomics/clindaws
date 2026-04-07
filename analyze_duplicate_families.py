#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from compare_solutions import (
    ANSWER_SET_RE,
    extract_local_name,
    infer_config_path,
    parse_snake_port_index,
    strip_snake_suffix,
)


SNAKE_TOOL_RE = re.compile(r'tool_at_time\((\d+),"([^"]+)"\)')
SNAKE_BIND_WF_RE = re.compile(r'ape_bind\((\d+),"([^"]+)","wf_input_(\d+)"\)')
SNAKE_BIND_OUT_RE = re.compile(
    r'ape_bind\((\d+),"([^"]+)",out\((\d+),"([^"]+)","([^"]+)"\)\)'
)

MOTIF_PRIORITY = (
    "repeated_run_chain_swap",
    "symmetric_join_output_role_swap",
    "ancestor_output_vs_descendant_output_substitution",
    "workflow_input_vs_ancestor_output_substitution",
    "output_reuse_depth_variation",
    "workflow_input_vs_output_substitution",
    "workflow_input_permutation",
    "other",
)

MOTIF_RULE_SKETCHES = {
    "repeated_run_chain_swap": {
        "rule_sketch": (
            "Constrain repeated runs of the same variant to consume canonically ordered "
            "artifacts on the reused port, but only when both sibling outputs are consumed downstream."
        ),
        "grounding_risk": "medium",
    },
    "symmetric_join_output_role_swap": {
        "rule_sketch": (
            "Order sibling producer outputs on symmetric join ports using a narrow guard "
            "that only fires when the join consumes the same repeated producer family."
        ),
        "grounding_risk": "medium",
    },
    "ancestor_output_vs_descendant_output_substitution": {
        "rule_sketch": (
            "Canonicalize direct ancestor-vs-descendant output choices in the same chain, "
            "preferring the older representative when the downstream tool sequence is unchanged."
        ),
        "grounding_risk": "medium",
    },
    "workflow_input_vs_ancestor_output_substitution": {
        "rule_sketch": (
            "Canonicalize workflow-input vs older chain-output choices only when that older "
            "output is clearly an ancestor in an already-materialized output chain."
        ),
        "grounding_risk": "medium",
    },
    "output_reuse_depth_variation": {
        "rule_sketch": (
            "Choose one canonical reuse depth when an output chain can be consumed either "
            "directly or via a later derived sibling artifact."
        ),
        "grounding_risk": "high",
    },
    "workflow_input_vs_output_substitution": {
        "rule_sketch": (
            "Prefer either the older produced artifact or the delayed workflow input only "
            "inside repeated-run plus downstream-consumer shapes."
        ),
        "grounding_risk": "high",
    },
    "workflow_input_permutation": {
        "rule_sketch": (
            "Canonicalize equivalent workflow-input choices on repeated same-signature "
            "ports or repeated-run ports where downstream use is unchanged."
        ),
        "grounding_risk": "low",
    },
    "other": {
        "rule_sketch": "Inspect representative families manually before adding any rule.",
        "grounding_risk": "unknown",
    },
}


@dataclass(frozen=True)
class SourceRef:
    kind: str
    workflow_input: int | None = None
    producer_step: int | None = None
    producer_tool: str | None = None
    producer_port: int | None = None

    def short(self) -> str:
        if self.kind == "workflow_input":
            return f"wf_input_{self.workflow_input}"
        return f"t{self.producer_step}:{self.producer_tool}:out{self.producer_port}"


@dataclass(frozen=True)
class BindingRef:
    consumer_step: int
    consumer_tool: str
    consumer_port: int
    source: SourceRef

    @property
    def target_key(self) -> tuple[int, int]:
        return (self.consumer_step, self.consumer_port)

    def target_label(self) -> str:
        return f"t{self.consumer_step}:{self.consumer_tool}:in{self.consumer_port}"


@dataclass(frozen=True)
class AnswerSetModel:
    index: int
    horizon: int
    tool_sequence: tuple[str, ...]
    tools_by_step: dict[int, str]
    bindings: dict[tuple[int, int], BindingRef]

    def binding_signature(self) -> tuple[str, ...]:
        parts = []
        for target_key in sorted(self.bindings):
            binding = self.bindings[target_key]
            parts.append(f"{binding.target_label()}<-{binding.source.short()}")
        return tuple(parts)

    def source_for(self, target_key: tuple[int, int]) -> SourceRef | None:
        binding = self.bindings.get(target_key)
        return None if binding is None else binding.source

    def dependency_edges(self) -> dict[int, set[int]]:
        edges: dict[int, set[int]] = defaultdict(set)
        for binding in self.bindings.values():
            if binding.source.kind == "output" and binding.source.producer_step is not None:
                edges[binding.consumer_step].add(binding.source.producer_step)
        return edges


@dataclass(frozen=True)
class Family:
    horizon: int
    tool_sequence: tuple[str, ...]
    models: tuple[AnswerSetModel, ...]

    @property
    def duplicate_models(self) -> int:
        return max(0, len(self.models) - 1)

    def label(self) -> str:
        return " -> ".join(self.tool_sequence)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Classify duplicate raw answer sets within the same ordered tool sequence and "
            "report likely motif families."
        )
    )
    parser.add_argument("answer_sets", help="Path to answer_sets.txt from a snakeAPE run.")
    parser.add_argument(
        "--workflow-signatures",
        help="Optional path to workflow_signatures.json. Defaults to a sibling file when present.",
    )
    parser.add_argument(
        "--config",
        help="Optional config path used to canonicalize tool labels.",
    )
    parser.add_argument(
        "--json-out",
        help="Optional path for the JSON report.",
    )
    parser.add_argument(
        "--top-families",
        type=int,
        default=12,
        help="Number of duplicate families to include in the report.",
    )
    parser.add_argument(
        "--top-motifs",
        type=int,
        default=8,
        help="Number of motif buckets to print in the text summary.",
    )
    return parser


def load_answer_set_models(answer_sets_path: Path, config_path: Path | None) -> list[AnswerSetModel]:
    models: list[AnswerSetModel] = []
    current_index: int | None = None
    current_lines: list[str] = []

    def flush() -> None:
        nonlocal current_index, current_lines
        if current_index is None:
            return
        block = " ".join(line.strip() for line in current_lines if line.strip())
        def exact_tool_name(raw_tool_name: str) -> str:
            return strip_snake_suffix(extract_local_name(raw_tool_name))

        tools_by_step = {
            int(step_str): exact_tool_name(tool_name)
            for step_str, tool_name in SNAKE_TOOL_RE.findall(block)
        }
        tool_sequence = tuple(tool_name for _, tool_name in sorted(tools_by_step.items()))
        bindings: dict[tuple[int, int], BindingRef] = {}

        for step_str, raw_consumer_port, wf_index in SNAKE_BIND_WF_RE.findall(block):
            step = int(step_str)
            consumer_port = parse_snake_port_index("", raw_consumer_port)
            if consumer_port is None or step not in tools_by_step:
                continue
            source = SourceRef(kind="workflow_input", workflow_input=int(wf_index))
            binding = BindingRef(
                consumer_step=step,
                consumer_tool=tools_by_step[step],
                consumer_port=consumer_port,
                source=source,
            )
            bindings[binding.target_key] = binding

        for step_str, raw_consumer_port, producer_step_str, raw_producer_tool, raw_producer_port in SNAKE_BIND_OUT_RE.findall(block):
            step = int(step_str)
            producer_step = int(producer_step_str)
            consumer_port = parse_snake_port_index("", raw_consumer_port)
            producer_port = parse_snake_port_index("", raw_producer_port)
            if consumer_port is None or producer_port is None or step not in tools_by_step:
                continue
            producer_tool = tools_by_step.get(
                producer_step,
                exact_tool_name(raw_producer_tool),
            )
            source = SourceRef(
                kind="output",
                producer_step=producer_step,
                producer_tool=producer_tool,
                producer_port=producer_port,
            )
            binding = BindingRef(
                consumer_step=step,
                consumer_tool=tools_by_step[step],
                consumer_port=consumer_port,
                source=source,
            )
            bindings[binding.target_key] = binding

        models.append(
            AnswerSetModel(
                index=current_index,
                horizon=len(tool_sequence),
                tool_sequence=tool_sequence,
                tools_by_step=tools_by_step,
                bindings=bindings,
            )
        )
        current_index = None
        current_lines = []

    for raw_line in answer_sets_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        header_match = ANSWER_SET_RE.match(line)
        if header_match:
            flush()
            current_index = int(header_match.group(1))
            current_lines = []
            continue
        if current_index is not None:
            current_lines.append(line)
    flush()
    return models


def load_workflow_signature_counts(path: Path | None) -> dict[int, int]:
    if path is None or not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    counts: Counter[int] = Counter()
    for entry in data.get("workflow_signatures", []):
        try:
            length = int(entry["length"])
        except (KeyError, TypeError, ValueError):
            continue
        counts[length] += 1
    return dict(counts)


def load_horizon_summary_counts(path: Path | None) -> dict[int, dict[str, int]]:
    if path is None or not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    counts: dict[int, dict[str, int]] = {}
    for entry in data.get("horizons", []):
        try:
            horizon = int(entry["horizon"])
        except (KeyError, TypeError, ValueError):
            continue
        counts[horizon] = {
            "models_seen": int(entry.get("models_seen", 0)),
            "unique_workflows_seen": int(entry.get("unique_workflows_seen", 0)),
            "models_stored": int(entry.get("models_stored", 0)),
            "unique_workflows_stored": int(entry.get("unique_workflows_stored", 0)),
        }
    return counts


def group_families(models: Iterable[AnswerSetModel]) -> list[Family]:
    grouped: dict[tuple[int, tuple[str, ...]], list[AnswerSetModel]] = defaultdict(list)
    for model in models:
        grouped[(model.horizon, model.tool_sequence)].append(model)
    families = [
        Family(horizon=horizon, tool_sequence=tool_sequence, models=tuple(group))
        for (horizon, tool_sequence), group in grouped.items()
    ]
    families.sort(key=lambda family: (-family.duplicate_models, family.horizon, family.tool_sequence))
    return families


def compute_ancestors(model: AnswerSetModel) -> dict[int, set[int]]:
    reverse_edges = model.dependency_edges()
    ancestors: dict[int, set[int]] = defaultdict(set)
    for consumer_step, producer_steps in reverse_edges.items():
        stack = list(producer_steps)
        while stack:
            step = stack.pop()
            if step in ancestors[consumer_step]:
                continue
            ancestors[consumer_step].add(step)
            stack.extend(reverse_edges.get(step, ()))
    return ancestors


def detect_swap_on_consumer_step(
    base: AnswerSetModel,
    other: AnswerSetModel,
    changed_targets: list[tuple[int, int]],
) -> bool:
    changed_by_step: dict[int, list[tuple[SourceRef, SourceRef]]] = defaultdict(list)
    for target_key in changed_targets:
        base_source = base.source_for(target_key)
        other_source = other.source_for(target_key)
        if base_source is None or other_source is None:
            continue
        changed_by_step[target_key[0]].append((base_source, other_source))

    for source_pairs in changed_by_step.values():
        if len(source_pairs) < 2:
            continue
        base_outputs = [source.short() for source, _ in source_pairs if source.kind == "output"]
        other_outputs = [source.short() for _, source in source_pairs if source.kind == "output"]
        if len(base_outputs) < 2 or len(other_outputs) < 2:
            continue
        if Counter(base_outputs) == Counter(other_outputs) and base_outputs != other_outputs:
            return True
    return False


def source_has_descendant_output(
    model: AnswerSetModel,
    source: SourceRef,
    ancestors: dict[int, set[int]],
) -> bool:
    if source.kind != "output" or source.producer_step is None:
        return False
    producer_step = source.producer_step
    for binding in model.bindings.values():
        other = binding.source
        if other.kind != "output" or other.producer_step is None:
            continue
        if producer_step != other.producer_step and producer_step in ancestors.get(other.producer_step, set()):
            return True
    return False


def outputs_are_ancestor_descendant(
    left: SourceRef,
    right: SourceRef,
    left_ancestors: dict[int, set[int]],
    right_ancestors: dict[int, set[int]],
) -> bool:
    if (
        left.kind != "output"
        or right.kind != "output"
        or left.producer_step is None
        or right.producer_step is None
    ):
        return False
    left_step = left.producer_step
    right_step = right.producer_step
    return (
        left_step in right_ancestors.get(right_step, set())
        or right_step in left_ancestors.get(left_step, set())
        or right_step in right_ancestors.get(left_step, set())
        or left_step in left_ancestors.get(right_step, set())
    )


def classify_difference(base: AnswerSetModel, other: AnswerSetModel) -> set[str]:
    motifs: set[str] = set()
    changed_targets = sorted(
        target_key
        for target_key in set(base.bindings) | set(other.bindings)
        if base.source_for(target_key) != other.source_for(target_key)
    )
    if not changed_targets:
        return motifs

    repeated_tools = {
        tool_name
        for tool_name, count in Counter(base.tool_sequence).items()
        if count > 1
    }
    base_ancestors = compute_ancestors(base)
    other_ancestors = compute_ancestors(other)

    for target_key in changed_targets:
        base_source = base.source_for(target_key)
        other_source = other.source_for(target_key)
        consumer_tool = base.tools_by_step.get(target_key[0], "")

        if consumer_tool in repeated_tools:
            motifs.add("repeated_run_chain_swap")

        if base_source is None or other_source is None:
            continue

        if base_source.kind == other_source.kind == "workflow_input":
            if base_source.workflow_input != other_source.workflow_input:
                motifs.add("workflow_input_permutation")
                if consumer_tool in repeated_tools:
                    motifs.add("repeated_run_chain_swap")
            continue

        if base_source.kind != other_source.kind:
            output_source = base_source if base_source.kind == "output" else other_source
            if source_has_descendant_output(base, output_source, base_ancestors) or source_has_descendant_output(
                other, output_source, other_ancestors
            ):
                motifs.add("workflow_input_vs_ancestor_output_substitution")
            else:
                motifs.add("workflow_input_vs_output_substitution")
            if consumer_tool in repeated_tools:
                motifs.add("repeated_run_chain_swap")
            continue

        if base_source.kind == other_source.kind == "output":
            if (
                base_source.producer_tool == other_source.producer_tool
                and base_source.producer_step != other_source.producer_step
            ):
                motifs.add("repeated_run_chain_swap")

            base_producer = base_source.producer_step
            other_producer = other_source.producer_step
            if base_producer is not None and other_producer is not None:
                if outputs_are_ancestor_descendant(base_source, other_source, base_ancestors, other_ancestors):
                    motifs.add("ancestor_output_vs_descendant_output_substitution")
                elif base_producer != other_producer:
                    motifs.add("output_reuse_depth_variation")

    if detect_swap_on_consumer_step(base, other, changed_targets):
        motifs.add("symmetric_join_output_role_swap")

    if not motifs:
        motifs.add("other")
    return motifs


def choose_primary_motif(motifs: set[str]) -> str:
    for motif in MOTIF_PRIORITY:
        if motif in motifs:
            return motif
    return "other"


def select_canonical_model(models: tuple[AnswerSetModel, ...]) -> AnswerSetModel:
    return min(models, key=lambda model: model.binding_signature())


def describe_changed_bindings(base: AnswerSetModel, other: AnswerSetModel, limit: int = 8) -> list[dict[str, str]]:
    rows = []
    for target_key in sorted(set(base.bindings) | set(other.bindings)):
        base_source = base.source_for(target_key)
        other_source = other.source_for(target_key)
        if base_source == other_source:
            continue
        binding = base.bindings.get(target_key) or other.bindings.get(target_key)
        if binding is None:
            continue
        rows.append(
            {
                "target": binding.target_label(),
                "canonical": "-" if base_source is None else base_source.short(),
                "variant": "-" if other_source is None else other_source.short(),
            }
        )
        if len(rows) >= limit:
            break
    return rows


def family_examples(family: Family) -> list[dict[str, object]]:
    canonical = select_canonical_model(family.models)
    examples = []
    for model in sorted(family.models, key=lambda item: item.binding_signature()):
        if model.index == canonical.index:
            continue
        motifs = classify_difference(canonical, model)
        examples.append(
            {
                "canonical_index": canonical.index,
                "variant_index": model.index,
                "variant_primary_motif": choose_primary_motif(motifs),
                "variant_motifs": sorted(motifs),
                "changed_bindings": describe_changed_bindings(canonical, model),
            }
        )
        if len(examples) >= 3:
            break
    return examples


def build_report(
    answer_sets_path: Path,
    workflow_signatures_path: Path | None,
    horizon_summary_path: Path | None,
    models: list[AnswerSetModel],
    top_families: int,
) -> dict[str, object]:
    families = group_families(models)
    duplicate_families = [family for family in families if family.duplicate_models > 0]

    workflow_signature_counts = load_workflow_signature_counts(workflow_signatures_path)
    horizon_summary_counts = load_horizon_summary_counts(horizon_summary_path)

    motif_totals: dict[str, dict[str, object]] = defaultdict(
        lambda: {
            "duplicate_models": 0,
            "affected_families": set(),
            "horizons": Counter(),
            "examples": [],
            "rule_sketch": "",
            "grounding_risk": "",
        }
    )
    horizon_motif_totals: dict[int, Counter[str]] = defaultdict(Counter)
    family_reports = []

    for family in duplicate_families:
        canonical = select_canonical_model(family.models)
        variant_primary_counts: Counter[str] = Counter()
        family_motifs: set[str] = set()

        for model in family.models:
            if model.index == canonical.index:
                continue
            motifs = classify_difference(canonical, model)
            primary_motif = choose_primary_motif(motifs)
            variant_primary_counts[primary_motif] += 1
            family_motifs.update(motifs)

        family_primary_motif = choose_primary_motif(set(variant_primary_counts))
        if variant_primary_counts:
            family_primary_motif = max(
                variant_primary_counts,
                key=lambda motif: (variant_primary_counts[motif], -MOTIF_PRIORITY.index(motif)),
            )

        for motif in {family_primary_motif}:
            motif_totals[motif]["duplicate_models"] += family.duplicate_models
            motif_totals[motif]["affected_families"].add((family.horizon, family.tool_sequence))
            motif_totals[motif]["horizons"][family.horizon] += family.duplicate_models
            motif_totals[motif]["rule_sketch"] = MOTIF_RULE_SKETCHES[motif]["rule_sketch"]
            motif_totals[motif]["grounding_risk"] = MOTIF_RULE_SKETCHES[motif]["grounding_risk"]
            if len(motif_totals[motif]["examples"]) < 5:
                motif_totals[motif]["examples"].append(
                    {
                        "horizon": family.horizon,
                        "tool_sequence": list(family.tool_sequence),
                        "duplicate_models": family.duplicate_models,
                        "family_motifs": sorted(family_motifs),
                        "examples": family_examples(family),
                    }
                )
            horizon_motif_totals[family.horizon][motif] += family.duplicate_models

        family_reports.append(
            {
                "horizon": family.horizon,
                "tool_sequence": list(family.tool_sequence),
                "raw_models": len(family.models),
                "duplicate_models": family.duplicate_models,
                "primary_motif": family_primary_motif,
                "family_motifs": sorted(family_motifs),
                "examples": family_examples(family),
            }
        )

    horizons_report = {}
    models_by_horizon = defaultdict(list)
    for model in models:
        models_by_horizon[model.horizon].append(model)

    for horizon in sorted(models_by_horizon):
        sequence_counter = Counter(model.tool_sequence for model in models_by_horizon[horizon])
        family_count = len(sequence_counter)
        duplicates = sum(count - 1 for count in sequence_counter.values())
        run_summary = horizon_summary_counts.get(horizon)
        workflow_signature_count = workflow_signature_counts.get(horizon)
        captured_models_ratio = None
        captured_unique_ratio = None
        coverage_limited = False
        if run_summary:
            models_seen = run_summary.get("models_seen", 0)
            unique_seen = run_summary.get("unique_workflows_seen", 0)
            if models_seen:
                captured_models_ratio = len(models_by_horizon[horizon]) / models_seen
                coverage_limited = coverage_limited or len(models_by_horizon[horizon]) != models_seen
            if unique_seen:
                captured_unique_ratio = family_count / unique_seen
                coverage_limited = coverage_limited or family_count != unique_seen
        elif workflow_signature_count is not None and workflow_signature_count:
            captured_unique_ratio = family_count / workflow_signature_count
            coverage_limited = coverage_limited or family_count != workflow_signature_count
        horizons_report[horizon] = {
            "raw_models_in_answer_sets": len(models_by_horizon[horizon]),
            "unique_tool_sequences_in_answer_sets": family_count,
            "duplicate_models_in_answer_sets": duplicates,
            "workflow_signatures": workflow_signature_count,
            "run_summary": run_summary,
            "captured_models_ratio": captured_models_ratio,
            "captured_unique_ratio": captured_unique_ratio,
            "coverage_limited": coverage_limited,
            "top_primary_motifs": [
                {"motif": motif, "duplicate_models": count}
                for motif, count in horizon_motif_totals[horizon].most_common()
            ],
        }

    serializable_motif_totals = {}
    for motif, data in motif_totals.items():
        serializable_motif_totals[motif] = {
            "duplicate_models": data["duplicate_models"],
            "affected_families": len(data["affected_families"]),
            "horizons": dict(sorted(data["horizons"].items())),
            "rule_sketch": data["rule_sketch"],
            "grounding_risk": data["grounding_risk"],
            "examples": data["examples"],
        }

    report = {
        "answer_sets_path": str(answer_sets_path),
        "workflow_signatures_path": None if workflow_signatures_path is None else str(workflow_signatures_path),
        "horizon_summary_path": None if horizon_summary_path is None else str(horizon_summary_path),
        "summary": {
            "raw_models_in_answer_sets": len(models),
            "unique_tool_sequences_in_answer_sets": len(families),
            "duplicate_models_in_answer_sets": sum(family.duplicate_models for family in families),
            "families_with_duplicates": len(duplicate_families),
        },
        "horizons": horizons_report,
        "motifs": dict(
            sorted(
                serializable_motif_totals.items(),
                key=lambda item: (-item[1]["duplicate_models"], item[0]),
            )
        ),
        "top_duplicate_families": sorted(
            family_reports,
            key=lambda family: (-family["duplicate_models"], family["horizon"], family["tool_sequence"]),
        )[:top_families],
    }
    return report


def print_summary(report: dict[str, object], top_motifs: int) -> None:
    summary = report["summary"]
    print(
        "Summary: "
        f"raw={summary['raw_models_in_answer_sets']} "
        f"unique={summary['unique_tool_sequences_in_answer_sets']} "
        f"duplicates={summary['duplicate_models_in_answer_sets']} "
        f"families={summary['families_with_duplicates']}"
    )
    print("Horizons:")
    for horizon, data in sorted(report["horizons"].items()):
        run_summary = data.get("run_summary") or {}
        run_text = ""
        if run_summary:
            run_text = (
                f" run_summary(raw={run_summary.get('models_seen')}, "
                f"unique={run_summary.get('unique_workflows_seen')})"
            )
        coverage_text = ""
        if data.get("coverage_limited"):
            captured_models_ratio = data.get("captured_models_ratio")
            captured_unique_ratio = data.get("captured_unique_ratio")
            ratio_bits = []
            if captured_models_ratio is not None:
                ratio_bits.append(f"raw_capture={captured_models_ratio:.3f}")
            if captured_unique_ratio is not None:
                ratio_bits.append(f"unique_capture={captured_unique_ratio:.3f}")
            coverage_text = " coverage_limited(" + ", ".join(ratio_bits) + ")"
        print(
            f"  H{horizon}: raw={data['raw_models_in_answer_sets']} "
            f"unique={data['unique_tool_sequences_in_answer_sets']} "
            f"duplicates={data['duplicate_models_in_answer_sets']}{run_text}{coverage_text}"
        )

    print("Top motifs:")
    for motif, data in list(report["motifs"].items())[:top_motifs]:
        print(
            f"  {motif}: duplicates={data['duplicate_models']} "
            f"families={data['affected_families']} "
            f"risk={data['grounding_risk']}"
        )


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    answer_sets_path = Path(args.answer_sets).resolve()
    workflow_signatures_path = (
        Path(args.workflow_signatures).resolve()
        if args.workflow_signatures
        else (answer_sets_path.parent / "workflow_signatures.json").resolve()
    )
    if not workflow_signatures_path.exists():
        workflow_signatures_path = None

    horizon_summary_path = (answer_sets_path.parent / "horizon_summary.json").resolve()
    if not horizon_summary_path.exists():
        horizon_summary_path = None

    config_path = Path(args.config).resolve() if args.config else infer_config_path(answer_sets_path)
    models = load_answer_set_models(answer_sets_path, config_path)
    report = build_report(
        answer_sets_path=answer_sets_path,
        workflow_signatures_path=workflow_signatures_path,
        horizon_summary_path=horizon_summary_path,
        models=models,
        top_families=args.top_families,
    )
    print_summary(report, top_motifs=args.top_motifs)

    if args.json_out:
        Path(args.json_out).resolve().write_text(
            json.dumps(report, indent=2) + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
