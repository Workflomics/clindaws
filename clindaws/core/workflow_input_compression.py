"""Workflow-input equivalence helpers shared across translation paths."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping


def _normalize_profile(
    dimensions: Mapping[str, Iterable[str]],
    units: Iterable[str] = (),
) -> tuple[tuple[tuple[str, tuple[str, ...]], ...], tuple[str, ...]]:
    return (
        tuple(
            sorted(
                (
                    str(category),
                    tuple(sorted(str(value) for value in values)),
                )
                for category, values in dimensions.items()
            )
        ),
        tuple(sorted(str(unit) for unit in units)),
    )


@dataclass(frozen=True)
class WorkflowInputEquivalenceClass:
    """One deterministic workflow-input equivalence class."""

    representative: str
    members: tuple[str, ...]

    @property
    def repeated(self) -> bool:
        return len(self.members) > 1

    @property
    def slot_count(self) -> int:
        return len(self.members) if self.repeated else 0


@dataclass(frozen=True)
class WorkflowInputCompressionPlan:
    """Static workflow-input equivalence partition."""

    classes: tuple[WorkflowInputEquivalenceClass, ...]

    @property
    def workflow_input_count(self) -> int:
        return sum(len(group.members) for group in self.classes)

    @property
    def equivalence_class_count(self) -> int:
        return len(self.classes)

    @property
    def repeated_equivalence_class_count(self) -> int:
        return sum(1 for group in self.classes if group.repeated)

    @property
    def collapsed_member_count(self) -> int:
        return sum(len(group.members) - 1 for group in self.classes if group.repeated)

    @property
    def slot_count(self) -> int:
        return sum(group.slot_count for group in self.classes)

    @property
    def planner_visible_count_if_compressed(self) -> int:
        return sum(group.slot_count if group.repeated else 1 for group in self.classes)

    @property
    def planner_visible_count_if_uncompressed(self) -> int:
        return self.workflow_input_count

    @property
    def planner_visible_reduction_if_compressed(self) -> int:
        return (
            self.planner_visible_count_if_uncompressed
            - self.planner_visible_count_if_compressed
        )


def build_workflow_input_compression_plan(
    workflow_input_dimensions: Mapping[str, Mapping[str, Iterable[str]]],
    workflow_input_units: Mapping[str, Iterable[str]] | None = None,
) -> WorkflowInputCompressionPlan:
    """Group workflow inputs by identical dim/unit profile."""

    unit_map = workflow_input_units or {}
    classes: dict[
        tuple[tuple[tuple[str, tuple[str, ...]], ...], tuple[str, ...]],
        list[str],
    ] = {}
    for workflow_input_id in sorted(workflow_input_dimensions):
        profile = _normalize_profile(
            workflow_input_dimensions[workflow_input_id],
            unit_map.get(workflow_input_id, ()),
        )
        classes.setdefault(profile, []).append(workflow_input_id)

    return WorkflowInputCompressionPlan(
        classes=tuple(
            WorkflowInputEquivalenceClass(
                representative=members[0],
                members=tuple(members),
            )
            for _, members in sorted(classes.items(), key=lambda item: item[1][0])
        )
    )


def workflow_input_compression_stats(
    plan: WorkflowInputCompressionPlan,
) -> dict[str, int]:
    """Return summary counters for one workflow-input compression plan."""

    return {
        "workflow_input_equivalence_classes": plan.equivalence_class_count,
        "workflow_input_repeated_equivalence_classes": plan.repeated_equivalence_class_count,
        "workflow_input_collapsed_members": plan.collapsed_member_count,
        "workflow_input_slots": plan.slot_count,
        "workflow_input_planner_visible_count_uncompressed": plan.planner_visible_count_if_uncompressed,
        "workflow_input_planner_visible_count_compressed": plan.planner_visible_count_if_compressed,
        "workflow_input_planner_visible_reduction_if_compressed": plan.planner_visible_reduction_if_compressed,
    }
