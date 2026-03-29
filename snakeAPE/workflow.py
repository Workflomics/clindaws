"""Workflow reconstruction from shown Clingo atoms."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

import clingo

from .models import ArtifactRef, Binding, WorkflowSolution, WorkflowStep


@dataclass
class _MutableArtifact:
    ref_id: str
    created_at: int
    created_by_tool: str | None
    created_by_label: str | None
    dims: dict[str, set[str]]


def _symbol_text(symbol: clingo.Symbol) -> str:
    if symbol.type == clingo.SymbolType.String:
        return symbol.string
    return str(symbol)


def _workflow_key(symbol: clingo.Symbol) -> str:
    return str(symbol)


def _artifact_from_key(ref_id: str, tool_labels: dict[str, str]) -> tuple[int, str | None, str | None]:
    if ref_id.startswith('"wf_input_') and ref_id.endswith('"'):
        return 0, None, None
    if ref_id.startswith("out(") and ref_id.endswith(")"):
        inner = ref_id[4:-1]
        parts = [part.strip() for part in inner.split(",", 2)]
        time = int(parts[0])
        tool_id = parts[1].strip('"')
        return time, tool_id, tool_labels.get(tool_id, tool_id)
    return 0, None, None


def _canonical_solution_key(
    steps: tuple[WorkflowStep, ...],
    artifacts: dict[str, ArtifactRef],
    goal_outputs: dict[int, str],
) -> tuple[object, ...]:
    step_key = tuple(
        (
            step.time,
            step.tool_id,
            tuple((binding.port_id, binding.artifact_id) for binding in step.bindings),
            tuple(step.outputs),
        )
        for step in steps
    )
    artifact_key = tuple(
        sorted(
            (
                ref_id,
                artifact.created_at,
                artifact.created_by_tool or "",
                tuple(
                    sorted(
                        (dim, tuple(sorted(values)))
                        for dim, values in artifact.dims.items()
                    )
                ),
            )
            for ref_id, artifact in artifacts.items()
        )
    )
    goal_key = tuple(sorted(goal_outputs.items()))
    return ("steps", step_key, "artifacts", artifact_key, "goals", goal_key)


def reconstruct_solution(
    index: int,
    symbols: Iterable[clingo.Symbol],
    tool_labels: dict[str, str],
) -> WorkflowSolution:
    """Reconstruct a workflow solution from shown symbols."""

    step_tools: dict[int, tuple[str, str]] = {}
    bindings_by_step: dict[int, list[Binding]] = defaultdict(list)
    artifacts: dict[str, _MutableArtifact] = {}
    goal_candidates: dict[int, list[str]] = defaultdict(list)

    for symbol in symbols:
        if symbol.type != clingo.SymbolType.Function:
            continue
        function = symbol
        name = function.name
        args = function.arguments

        if name == "tool_at_time":
            time = int(str(args[0]))
            tool_id = _symbol_text(args[1])
            step_tools[time] = (tool_id, tool_labels.get(tool_id, tool_id))
        elif name == "ape_bind":
            time = int(str(args[0]))
            bindings_by_step[time].append(
                Binding(
                    port_id=_symbol_text(args[1]),
                    artifact_id=_workflow_key(args[2]),
                )
            )
        elif name == "ape_holds_dim":
            ref_id = _workflow_key(args[0])
            value = _symbol_text(args[1])
            dim = _symbol_text(args[2])
            if ref_id not in artifacts:
                created_at, created_by_tool, created_by_label = _artifact_from_key(ref_id, tool_labels)
                artifacts[ref_id] = _MutableArtifact(
                    ref_id=ref_id,
                    created_at=created_at,
                    created_by_tool=created_by_tool,
                    created_by_label=created_by_label,
                    dims=defaultdict(set),
                )
            artifacts[ref_id].dims.setdefault(dim, set()).add(value)
        elif name == "ape_goal_out":
            goal_index = int(str(args[1]))
            goal_candidates[goal_index].append(_workflow_key(args[2]))

    for artifact_id in {binding.artifact_id for bindings in bindings_by_step.values() for binding in bindings}:
        if artifact_id not in artifacts:
            created_at, created_by_tool, created_by_label = _artifact_from_key(artifact_id, tool_labels)
            artifacts[artifact_id] = _MutableArtifact(
                ref_id=artifact_id,
                created_at=created_at,
                created_by_tool=created_by_tool,
                created_by_label=created_by_label,
                dims=defaultdict(set),
            )

    outputs_by_step: dict[int, list[str]] = defaultdict(list)
    for artifact in artifacts.values():
        if artifact.created_at > 0:
            outputs_by_step[artifact.created_at].append(artifact.ref_id)

    steps = []
    for time in sorted(step_tools):
        tool_id, tool_label = step_tools[time]
        steps.append(
            WorkflowStep(
                time=time,
                tool_id=tool_id,
                tool_label=tool_label,
                bindings=tuple(sorted(bindings_by_step.get(time, []), key=lambda item: item.port_id)),
                outputs=tuple(sorted(outputs_by_step.get(time, []))),
            )
        )

    goal_outputs = {}
    for goal_index, candidates in goal_candidates.items():
        ranked = sorted(
            candidates,
            key=lambda ref: (
                artifacts.get(ref).created_at if ref in artifacts else 0,
                0 if ref.startswith('"wf_input_') else 1,
                ref,
            ),
            reverse=True,
        )
        if ranked:
            goal_outputs[goal_index] = ranked[0]

    frozen_artifacts = {
        ref_id: ArtifactRef(
            ref_id=artifact.ref_id,
            created_at=artifact.created_at,
            created_by_tool=artifact.created_by_tool,
            created_by_label=artifact.created_by_label,
            dims={dim: set(values) for dim, values in artifact.dims.items()},
        )
        for ref_id, artifact in artifacts.items()
    }

    return WorkflowSolution(
        index=index,
        steps=tuple(steps),
        artifacts=frozen_artifacts,
        goal_outputs=goal_outputs,
        canonical_key=_canonical_solution_key(tuple(steps), frozen_artifacts, goal_outputs),
    )
