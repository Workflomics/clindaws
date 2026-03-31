"""Workflow reconstruction from shown Clingo atoms."""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from functools import cmp_to_key
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


_LEGACY_PORT_SUFFIX_RE = re.compile(r"_p(\d+)$")
_STRUCTURED_PORT_SUFFIX_RE = re.compile(r",\s*(\d+)\)$")


def _symbol_text(symbol: clingo.Symbol) -> str:
    if symbol.type == clingo.SymbolType.String:
        return symbol.string
    return str(symbol)


def _workflow_key(symbol: clingo.Symbol) -> str:
    return str(symbol)


def _port_index(port_id: str) -> int | None:
    try:
        return int(port_id)
    except ValueError:
        pass

    match = _LEGACY_PORT_SUFFIX_RE.search(port_id)
    if match is not None:
        return int(match.group(1))

    match = _STRUCTURED_PORT_SUFFIX_RE.search(port_id)
    if match is not None:
        return int(match.group(1))

    return None


def _port_sort_key(port_id: str) -> tuple[int, int, str]:
    port_index = _port_index(port_id)
    if port_index is not None:
        return (0, port_index, port_id)
    return (1, 0, port_id)


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


def _artifact_sort_tuple(ref_id: str) -> tuple[int, int, str]:
    created_at, _, _ = _artifact_from_key(ref_id, {})
    return (
        created_at,
        0 if ref_id.startswith('"wf_input_') and ref_id.endswith('"') else 1,
        ref_id,
    )


def _is_descendant(
    artifact_id: str,
    ancestor_id: str,
    parents: dict[str, tuple[str, ...]],
    memo: dict[tuple[str, str], bool],
) -> bool:
    key = (artifact_id, ancestor_id)
    if key in memo:
        return memo[key]
    for parent_id in parents.get(artifact_id, ()):
        if parent_id == ancestor_id or _is_descendant(parent_id, ancestor_id, parents, memo):
            memo[key] = True
            return True
    memo[key] = False
    return False


def canonicalize_shown_symbols(
    symbols: Iterable[clingo.Symbol],
    tool_input_signatures: dict[str, tuple[tuple[tuple[str, tuple[str, ...]], ...], ...]],
) -> tuple[clingo.Symbol, ...]:
    """Normalize shown ape_bind atoms to a SAT-style canonical ordering."""

    symbol_list = list(symbols)
    step_tools: dict[int, str] = {}
    bindings_by_step: dict[int, dict[str, str]] = defaultdict(dict)
    artifact_symbols: dict[str, clingo.Symbol] = {}
    output_refs: set[str] = set()
    other_symbols: list[clingo.Symbol] = []

    for symbol in symbol_list:
        if symbol.type != clingo.SymbolType.Function:
            other_symbols.append(symbol)
            continue
        name = symbol.name
        args = symbol.arguments
        if name == "tool_at_time":
            step_tools[int(str(args[0]))] = _symbol_text(args[1])
            other_symbols.append(symbol)
        elif name == "ape_bind":
            time = int(str(args[0]))
            port_id = _symbol_text(args[1])
            artifact_id = _workflow_key(args[2])
            bindings_by_step[time][port_id] = artifact_id
            artifact_symbols[artifact_id] = args[2]
        else:
            if name == "ape_holds_dim":
                artifact_id = _workflow_key(args[0])
                artifact_symbols.setdefault(artifact_id, args[0])
                if artifact_id.startswith("out(") and artifact_id.endswith(")"):
                    output_refs.add(artifact_id)
            elif name == "ape_goal_out":
                artifact_id = _workflow_key(args[2])
                artifact_symbols.setdefault(artifact_id, args[2])
            other_symbols.append(symbol)

    parents: dict[str, tuple[str, ...]] = {}
    for output_ref in output_refs:
        created_at, _, _ = _artifact_from_key(output_ref, {})
        if created_at > 0 and created_at in bindings_by_step:
            parents[output_ref] = tuple(bindings_by_step[created_at].values())

    descendant_memo: dict[tuple[str, str], bool] = {}

    def _compare_artifacts(left: str, right: str) -> int:
        if left == right:
            return 0
        if _is_descendant(left, right, parents, descendant_memo):
            return 1
        if _is_descendant(right, left, parents, descendant_memo):
            return -1
        left_key = _artifact_sort_tuple(left)
        right_key = _artifact_sort_tuple(right)
        if left_key < right_key:
            return -1
        return 1

    def _signature(tool_id: str | None, port_id: str) -> tuple[tuple[str, tuple[str, ...]], ...] | None:
        if tool_id is None:
            return None
        port_index = _port_index(port_id)
        if port_index is None:
            return None
        signatures = tool_input_signatures.get(tool_id, ())
        if 0 <= port_index < len(signatures):
            return signatures[port_index]
        return None

    # Canonicalize same-signature join ports directly.
    for time in sorted(bindings_by_step):
        tool_id = step_tools.get(time)
        if tool_id is None:
            continue
        sig_groups: dict[tuple[tuple[str, tuple[str, ...]], ...], list[str]] = defaultdict(list)
        for port_id in bindings_by_step[time]:
            signature = _signature(tool_id, port_id)
            if signature is not None:
                sig_groups[signature].append(port_id)
        for ports in sig_groups.values():
            if len(ports) < 2:
                continue
            ordered_ports = sorted(ports, key=_port_sort_key)
            ordered_artifacts = sorted(
                (bindings_by_step[time][port_id] for port_id in ordered_ports),
                key=cmp_to_key(_compare_artifacts),
            )
            for port_id, artifact_id in zip(ordered_ports, ordered_artifacts):
                bindings_by_step[time][port_id] = artifact_id

    # Canonicalize paired single-input producers based on the later symmetric join
    # they feed: the lower join port gets the older source artifact.
    for time in sorted(bindings_by_step):
        tool_id = step_tools.get(time)
        if tool_id is None:
            continue
        sig_groups: dict[tuple[tuple[str, tuple[str, ...]], ...], list[str]] = defaultdict(list)
        for port_id in bindings_by_step[time]:
            signature = _signature(tool_id, port_id)
            if signature is not None:
                sig_groups[signature].append(port_id)
        for ports in sig_groups.values():
            if len(ports) < 2:
                continue
            consumer_ports = sorted(ports, key=_port_sort_key)
            producer_infos: list[tuple[str, int, str, str, str]] = []
            producer_signatures: set[tuple[tuple[str, tuple[str, ...]], ...]] = set()
            producer_tools: set[str] = set()
            for port_id in consumer_ports:
                artifact_id = bindings_by_step[time][port_id]
                created_at, producer_tool, _ = _artifact_from_key(artifact_id, {})
                if created_at <= 0 or producer_tool is None:
                    producer_infos = []
                    break
                producer_bindings = bindings_by_step.get(created_at)
                producer_tool_at_time = step_tools.get(created_at)
                if not producer_bindings or producer_tool_at_time is None or len(producer_bindings) != 1:
                    producer_infos = []
                    break
                producer_port_id, source_artifact = next(iter(producer_bindings.items()))
                producer_signature = _signature(producer_tool_at_time, producer_port_id)
                if producer_signature is None:
                    producer_infos = []
                    break
                producer_infos.append((port_id, created_at, producer_tool_at_time, producer_port_id, source_artifact))
                producer_signatures.add(producer_signature)
                producer_tools.add(producer_tool_at_time)
            if len(producer_infos) < 2 or len(producer_signatures) != 1 or len(producer_tools) < 2:
                continue
            ordered_infos = sorted(producer_infos, key=lambda item: _port_sort_key(item[0]))
            ordered_sources = sorted(
                (item[4] for item in ordered_infos),
                key=cmp_to_key(_compare_artifacts),
            )
            for (_, producer_time, _, producer_port_id, _), source_artifact in zip(ordered_infos, ordered_sources):
                bindings_by_step[producer_time][producer_port_id] = source_artifact

    canonical_binds = [
        clingo.Function(
            "ape_bind",
            [
                clingo.Number(time),
                clingo.Number(port_index) if port_index is not None else clingo.String(port_id),
                artifact_symbols[artifact_id],
            ],
        )
        for time in sorted(bindings_by_step)
        for port_id, artifact_id in sorted(bindings_by_step[time].items(), key=lambda item: _port_sort_key(item[0]))
        for port_index in [_port_index(port_id)]
    ]
    return tuple(sorted([*other_symbols, *canonical_binds], key=str))


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
                bindings=tuple(sorted(bindings_by_step.get(time, []), key=lambda item: _port_sort_key(item.port_id))),
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
