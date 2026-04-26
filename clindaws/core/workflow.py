"""Workflow reconstruction from shown Clingo atoms."""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Mapping

import clingo

from clindaws.core.models import ArtifactRef, Binding, WorkflowSolution, WorkflowStep
from clindaws.core.ontology import Ontology


@dataclass
class _MutableArtifact:
    ref_id: str
    created_at: int
    created_by_tool: str | None
    created_by_label: str | None
    dims: dict[str, set[str]]


@dataclass(frozen=True)
class WorkflowKeyBundle:
    tool_sequence_key: tuple[object, ...]
    structural_workflow_key: tuple[object, ...]
    parity_workflow_key: tuple[object, ...]


_LEGACY_PORT_SUFFIX_RE = re.compile(r"_p(?:ort_)?(\d+)$")
_STRUCTURED_PORT_SUFFIX_RE = re.compile(r"_port_(\d+)$")
_OUTPUT_SLOT_RE = re.compile(r"_out_(\d+)_port_(\d+)$")


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


def _artifact_output_index(port_id: str) -> int | None:
    match = _OUTPUT_SLOT_RE.search(port_id)
    if match is not None:
        return int(match.group(1))
    return _port_index(port_id)


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


def _parse_shown_workflow_symbols(
    symbols: Iterable[clingo.Symbol],
) -> tuple[
    dict[int, str],
    dict[int, dict[str, str]],
    dict[str, clingo.Symbol],
    dict[str, dict[str, set[str]]],
    set[str],
    list[tuple[int, str]],
    list[clingo.Symbol],
]:
    step_tools: dict[int, str] = {}
    bindings_by_step: dict[int, dict[str, str]] = defaultdict(dict)
    artifact_symbols: dict[str, clingo.Symbol] = {}
    artifact_dims: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    output_refs: set[str] = set()
    goal_refs: list[tuple[int, str]] = []
    other_symbols: list[clingo.Symbol] = []

    for symbol in symbols:
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
                artifact_dims[artifact_id][_symbol_text(args[2])].add(_symbol_text(args[1]))
                if artifact_id.startswith("out(") and artifact_id.endswith(")"):
                    output_refs.add(artifact_id)
            elif name == "ape_output":
                artifact_id = _workflow_key(args[0])
                artifact_symbols.setdefault(artifact_id, args[0])
                if artifact_id.startswith("out(") and artifact_id.endswith(")"):
                    output_refs.add(artifact_id)
            elif name == "ape_goal_out":
                artifact_id = _workflow_key(args[2])
                artifact_symbols.setdefault(artifact_id, args[2])
                goal_refs.append((int(str(args[1])), artifact_id))
            other_symbols.append(symbol)

    return step_tools, bindings_by_step, artifact_symbols, artifact_dims, output_refs, goal_refs, other_symbols


def _input_signature(
    tool_input_signatures: dict[str, tuple[tuple[tuple[str, tuple[str, ...]], ...], ...]],
    tool_id: str | None,
    port_id: str,
) -> tuple[tuple[str, tuple[str, ...]], ...] | None:
    if tool_id is None:
        return None
    port_index = _port_index(port_id)
    if port_index is None:
        return None
    signatures = tool_input_signatures.get(tool_id, ())
    if 0 <= port_index < len(signatures):
        return signatures[port_index]
    return None


def _artifact_metadata_dimensions(
    ref_id: str,
    *,
    workflow_input_dims: Mapping[str, Mapping[str, tuple[str, ...]]],
    tool_output_dims: Mapping[tuple[str, int], Mapping[str, tuple[str, ...]]],
) -> Mapping[str, tuple[str, ...]]:
    stripped_ref = ref_id.strip('"')
    if stripped_ref in workflow_input_dims:
        return workflow_input_dims[stripped_ref]
    if ref_id in workflow_input_dims:
        return workflow_input_dims[ref_id]
    created_at, tool_id, _ = _artifact_from_key(ref_id, {})
    if created_at <= 0 or tool_id is None:
        return {}
    inner = ref_id[4:-1]
    parts = [part.strip() for part in inner.split(",", 2)]
    port_id = parts[2].strip().strip('"')
    port_index = _artifact_output_index(port_id)
    if port_index is None:
        return {}
    return tool_output_dims.get((tool_id, port_index), {})


def _merge_metadata_artifact_dims(
    artifact_dims: dict[str, dict[str, set[str]]],
    *,
    artifact_refs: Iterable[str],
    workflow_input_dims: Mapping[str, Mapping[str, tuple[str, ...]]],
    tool_output_dims: Mapping[tuple[str, int], Mapping[str, tuple[str, ...]]],
) -> None:
    for artifact_id in artifact_refs:
        metadata_dims = _artifact_metadata_dimensions(
            artifact_id,
            workflow_input_dims=workflow_input_dims,
            tool_output_dims=tool_output_dims,
        )
        for dim, values in metadata_dims.items():
            artifact_dims[artifact_id][str(dim)].update(str(value) for value in values)


def _canonicalize_binding_assignments(
    step_tools: dict[int, str],
    bindings_by_step: dict[int, dict[str, str]],
    artifact_dims: dict[str, dict[str, set[str]]],
    output_refs: set[str],
    goal_refs: list[tuple[int, str]],
    tool_input_signatures: dict[str, tuple[tuple[tuple[str, tuple[str, ...]], ...], ...]],
) -> None:
    usage_by_artifact: dict[str, list[tuple[object, ...]]] = defaultdict(list)
    for consumer_time, bindings in bindings_by_step.items():
        consumer_tool = step_tools.get(consumer_time, "")
        for port_id, artifact_id in bindings.items():
            port_index = _port_index(port_id)
            usage_by_artifact[artifact_id].append(
                (
                    "bind",
                    consumer_time,
                    consumer_tool,
                    port_index if port_index is not None else port_id,
                )
            )
    for goal_index, artifact_id in goal_refs:
        usage_by_artifact[artifact_id].append(("goal", goal_index))

    alias_groups: dict[
        tuple[int, str, tuple[tuple[str, tuple[str, ...]], ...], tuple[tuple[object, ...], ...]],
        list[str],
    ] = defaultdict(list)
    for output_ref in sorted(output_refs):
        created_at, producer_tool, _ = _artifact_from_key(output_ref, {})
        if created_at <= 0 or producer_tool is None:
            continue
        dims_key = tuple(
            sorted(
                (dim, tuple(sorted(values)))
                for dim, values in artifact_dims.get(output_ref, {}).items()
            )
        )
        usage_key = tuple(sorted(usage_by_artifact.get(output_ref, ())))
        alias_groups[(created_at, producer_tool, dims_key, usage_key)].append(output_ref)

    symmetric_output_steps = {
        created_at
        for (created_at, _producer_tool, _dims_key, _usage_key), members in alias_groups.items()
        if len(members) > 1
    }
    for time, bindings in bindings_by_step.items():
        if time not in symmetric_output_steps:
            continue
        tool_id = step_tools.get(time)
        if tool_id is None:
            continue
        signature_groups: dict[tuple[tuple[str, tuple[str, ...]], ...], list[str]] = defaultdict(list)
        for port_id in bindings:
            signature = _input_signature(tool_input_signatures, tool_id, port_id)
            if signature is not None:
                signature_groups[signature].append(port_id)
        for ports in signature_groups.values():
            if len(ports) < 2:
                continue
            ordered_ports = sorted(ports, key=_port_sort_key)
            ordered_artifacts = sorted((bindings[port_id] for port_id in ordered_ports), key=_artifact_sort_tuple)
            for port_id, artifact_id in zip(ordered_ports, ordered_artifacts):
                bindings[port_id] = artifact_id

    equivalent_output_ref: dict[str, str] = {}
    for members in alias_groups.values():
        if len(members) < 2:
            continue
        representative = min(members, key=_artifact_sort_tuple)
        for artifact_id in members:
            equivalent_output_ref[artifact_id] = representative

    if not equivalent_output_ref:
        return

    original_output_refs = tuple(output_refs)
    for time, bindings in tuple(bindings_by_step.items()):
        bindings_by_step[time] = {
            port_id: equivalent_output_ref.get(artifact_id, artifact_id)
            for port_id, artifact_id in bindings.items()
        }
    output_refs.clear()
    output_refs.update(
        equivalent_output_ref.get(output_ref, output_ref)
        for output_ref in original_output_refs
    )
    for index, (goal_index, artifact_id) in enumerate(goal_refs):
        goal_refs[index] = (goal_index, equivalent_output_ref.get(artifact_id, artifact_id))

def canonicalize_shown_symbols(
    symbols: Iterable[clingo.Symbol],
    tool_input_signatures: dict[str, tuple[tuple[tuple[str, tuple[str, ...]], ...], ...]],
    workflow_input_dims: Mapping[str, Mapping[str, tuple[str, ...]]] | None = None,
    tool_output_dims: Mapping[tuple[str, int], Mapping[str, tuple[str, ...]]] | None = None,
) -> tuple[clingo.Symbol, ...]:
    """Normalize shown ape_bind atoms to a SAT-style canonical ordering."""

    step_tools, bindings_by_step, artifact_symbols, artifact_dims, output_refs, goal_refs, other_symbols = _parse_shown_workflow_symbols(symbols)
    _merge_metadata_artifact_dims(
        artifact_dims,
        artifact_refs={
            artifact_id
            for bindings in bindings_by_step.values()
            for artifact_id in bindings.values()
        }
        | set(output_refs),
        workflow_input_dims=workflow_input_dims or {},
        tool_output_dims=tool_output_dims or {},
    )
    _canonicalize_binding_assignments(
        step_tools,
        bindings_by_step,
        artifact_dims,
        output_refs,
        goal_refs,
        tool_input_signatures,
    )

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


def extract_canonical_workflow_keys(
    symbols: Iterable[clingo.Symbol],
    tool_input_signatures: dict[str, tuple[tuple[tuple[str, tuple[str, ...]], ...], ...]],
    workflow_input_dims: Mapping[str, Mapping[str, tuple[str, ...]]] | None = None,
    tool_output_dims: Mapping[tuple[str, int], Mapping[str, tuple[str, ...]]] | None = None,
    ontology: Ontology | None = None,
    use_binding_target_abstraction: bool = False,
) -> tuple[tuple[object, ...], tuple[object, ...]]:
    """Extract canonical tool-sequence and parity workflow-candidate keys."""

    keys = extract_workflow_key_bundle(
        symbols,
        tool_input_signatures,
        workflow_input_dims,
        tool_output_dims,
        ontology,
        use_binding_target_abstraction=use_binding_target_abstraction,
    )
    return keys.tool_sequence_key, keys.parity_workflow_key


def _workflow_candidate_key(
    *,
    step_tools: Mapping[int, str],
    bindings_by_step: Mapping[int, Mapping[str, str]],
    output_refs: Iterable[str],
) -> tuple[object, ...]:
    ordered_steps = tuple(tool_id for _, tool_id in sorted(step_tools.items()))
    binding_signatures = tuple(
        sorted(
            _binding_signature_with_target(
                time,
                port_id,
                _normalize_artifact_ref(artifact_id),
            )
            for time in sorted(bindings_by_step)
            for port_id, artifact_id in sorted(
                bindings_by_step[time].items(),
                key=lambda item: _port_sort_key(item[0]),
            )
        )
    )
    output_signatures = tuple(sorted(_normalize_artifact_ref(ref_id) for ref_id in output_refs))
    return (
        "steps",
        ordered_steps,
        "bindings",
        binding_signatures,
        "outputs",
        output_signatures,
    )


def _parity_workflow_candidate_key(
    *,
    step_tools: Mapping[int, str],
    bindings_by_step: Mapping[int, Mapping[str, str]],
    output_refs: Iterable[str],
    artifact_dims: Mapping[str, Mapping[str, set[str]]],
    tool_input_signatures: Mapping[str, tuple[tuple[tuple[str, tuple[str, ...]], ...], ...]],
    ontology: Ontology | None = None,
) -> tuple[object, ...]:
    artifact_labels = _artifact_equivalence_labels(artifact_dims)
    ordered_steps = tuple(tool_id for _, tool_id in sorted(step_tools.items()))
    binding_signatures = tuple(
        sorted(
            _binding_signature_with_target(
                time,
                port_id,
                _binding_target_label(
                    tool_id=step_tools.get(time),
                    port_id=port_id,
                    artifact_id=artifact_id,
                    artifact_dims=artifact_dims,
                    tool_input_signatures=tool_input_signatures,
                    artifact_labels=artifact_labels,
                    ontology=ontology,
                ),
            )
            for time in sorted(bindings_by_step)
            for port_id, artifact_id in sorted(
                bindings_by_step[time].items(),
                key=lambda item: _port_sort_key(item[0]),
            )
        )
    )
    output_signatures = tuple(sorted(_normalize_artifact_ref(ref_id) for ref_id in output_refs))
    return (
        "steps",
        ordered_steps,
        "bindings",
        binding_signatures,
        "outputs",
        output_signatures,
    )


def extract_workflow_key_bundle(
    symbols: Iterable[clingo.Symbol],
    tool_input_signatures: dict[str, tuple[tuple[tuple[str, tuple[str, ...]], ...], ...]],
    workflow_input_dims: Mapping[str, Mapping[str, tuple[str, ...]]] | None = None,
    tool_output_dims: Mapping[tuple[str, int], Mapping[str, tuple[str, ...]]] | None = None,
    ontology: Ontology | None = None,
    use_binding_target_abstraction: bool = False,
) -> WorkflowKeyBundle:
    """Extract both exact structural and current parity workflow keys.

    The structural key keeps the raw destination-specific binding surface from
    the shown model. The parity key applies the current canonicalization rules
    used for stored workflow-candidate counting.
    """

    step_tools, bindings_by_step, _, artifact_dims, output_refs, goal_refs, _ = _parse_shown_workflow_symbols(symbols)
    _merge_metadata_artifact_dims(
        artifact_dims,
        artifact_refs={
            artifact_id
            for bindings in bindings_by_step.values()
            for artifact_id in bindings.values()
        }
        | set(output_refs),
        workflow_input_dims=workflow_input_dims or {},
        tool_output_dims=tool_output_dims or {},
    )
    structural_bindings_by_step = {
        time: dict(bindings)
        for time, bindings in bindings_by_step.items()
    }
    structural_output_refs = set(output_refs)
    structural_workflow_key = _workflow_candidate_key(
        step_tools=step_tools,
        bindings_by_step=structural_bindings_by_step,
        output_refs=structural_output_refs,
    )
    _canonicalize_binding_assignments(
        step_tools,
        bindings_by_step,
        artifact_dims,
        output_refs,
        goal_refs,
        tool_input_signatures,
    )
    ordered_steps = tuple(tool_id for _, tool_id in sorted(step_tools.items()))
    tool_sequence_key = ("steps", ordered_steps)
    if use_binding_target_abstraction:
        parity_workflow_key = _parity_workflow_candidate_key(
            step_tools=step_tools,
            bindings_by_step=bindings_by_step,
            output_refs=output_refs,
            artifact_dims=artifact_dims,
            tool_input_signatures=tool_input_signatures,
            ontology=ontology,
        )
    else:
        parity_workflow_key = _workflow_candidate_key(
            step_tools=step_tools,
            bindings_by_step=bindings_by_step,
            output_refs=output_refs,
        )
    return WorkflowKeyBundle(
        tool_sequence_key=tool_sequence_key,
        structural_workflow_key=structural_workflow_key,
        parity_workflow_key=parity_workflow_key,
    )


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


def _normalize_artifact_ref(ref_id: str) -> str:
    if ref_id.startswith('"wf_input_') and ref_id.endswith('"'):
        return ref_id.strip('"')
    if ref_id.startswith("out(") and ref_id.endswith(")"):
        inner = ref_id[4:-1]
        parts = [part.strip() for part in inner.split(",", 2)]
        time = int(parts[0])
        port_id = parts[2].strip().strip('"')
        port_index = _artifact_output_index(port_id)
        if port_index is not None:
            return f"t{time}:out{port_index}"
        return f"t{time}:out[{port_id}]"
    return ref_id.strip('"')


def _binding_signature(time: int, binding: Binding) -> str:
    port_index = _port_index(binding.port_id)
    if port_index is not None:
        port_text = f"in{port_index}"
    else:
        port_text = f"in[{binding.port_id}]"
    return f"t{time}:{port_text}<-{_normalize_artifact_ref(binding.artifact_id)}"


def _binding_signature_with_target(time: int, port_id: str, target: str) -> str:
    port_index = _port_index(port_id)
    if port_index is not None:
        port_text = f"in{port_index}"
    else:
        port_text = f"in[{port_id}]"
    return f"t{time}:{port_text}<-{target}"


def _artifact_equivalence_labels(
    artifact_dims: Mapping[str, Mapping[str, set[str]]],
) -> dict[str, str]:
    class_members: dict[tuple[tuple[str, tuple[str, ...]], ...], list[str]] = defaultdict(list)
    for artifact_id, dims in artifact_dims.items():
        dims_key = tuple(
            sorted(
                (dim, tuple(sorted(values)))
                for dim, values in dims.items()
            )
        )
        if not dims_key:
            continue
        class_members[dims_key].append(artifact_id)

    labels: dict[str, str] = {}
    for index, (_dims_key, members) in enumerate(
        sorted(class_members.items(), key=lambda item: (len(item[0]), item[0])),
        start=1,
    ):
        representative = min(members, key=_artifact_sort_tuple)
        label = f"artifact_class_{index}:{_normalize_artifact_ref(representative)}"
        for artifact_id in members:
            labels[artifact_id] = label
    return labels


def _binding_target_label(
    *,
    tool_id: str | None,
    port_id: str,
    artifact_id: str,
    artifact_dims: Mapping[str, Mapping[str, set[str]]],
    tool_input_signatures: Mapping[str, tuple[tuple[tuple[str, tuple[str, ...]], ...], ...]],
    artifact_labels: Mapping[str, str],
    ontology: Ontology | None = None,
) -> str:
    signature = _input_signature(tool_input_signatures, tool_id, port_id)
    if signature is None:
        return artifact_labels.get(artifact_id, _normalize_artifact_ref(artifact_id))
    dims = artifact_dims.get(artifact_id, {})
    projection_items: list[tuple[str, bool]] = []
    for category, accepted_values in signature:
        actual_values = dims.get(category, set())
        if ontology is None:
            matched = bool(actual_values)
        else:
            matched = any(
                required_value in ontology.ancestors_of(actual_value)
                for required_value in accepted_values
                for actual_value in actual_values
            )
        projection_items.append((category, matched))
    projection = tuple(projection_items)
    if any(matched for _category, matched in projection):
        return f"binding_class:{projection}"
    return artifact_labels.get(artifact_id, _normalize_artifact_ref(artifact_id))


def _goal_signature(goal_index: int, artifact_id: str) -> str:
    return f"g{goal_index}<-{_normalize_artifact_ref(artifact_id)}"


def _workflow_signature_key(
    steps: tuple[WorkflowStep, ...],
) -> tuple[object, ...]:
    return ("steps", tuple(step.tool_id for step in steps))


def workflow_signature_length(workflow_key: tuple[object, ...]) -> int:
    """Return the workflow length encoded in the default signature key."""

    if len(workflow_key) >= 2 and workflow_key[0] == "steps" and isinstance(workflow_key[1], tuple):
        return len(workflow_key[1])
    return 0


def reconstruct_solution(
    index: int,
    symbols: Iterable[clingo.Symbol],
    tool_labels: dict[str, str],
    workflow_input_dims: Mapping[str, Mapping[str, tuple[str, ...]]] | None = None,
    tool_output_dims: Mapping[tuple[str, int], Mapping[str, tuple[str, ...]]] | None = None,
) -> WorkflowSolution:
    """Reconstruct a workflow solution from shown symbols."""

    step_tools: dict[int, tuple[str, str]] = {}
    bindings_by_step: dict[int, list[Binding]] = defaultdict(list)
    artifacts: dict[str, _MutableArtifact] = {}
    goal_candidates: dict[int, list[str]] = defaultdict(list)
    workflow_input_dims = workflow_input_dims or {}
    tool_output_dims = tool_output_dims or {}

    def _ensure_artifact(
        ref_id: str,
        *,
        initial_dims: Mapping[str, tuple[str, ...]] | None = None,
    ) -> _MutableArtifact:
        artifact = artifacts.get(ref_id)
        if artifact is None:
            created_at, created_by_tool, created_by_label = _artifact_from_key(ref_id, tool_labels)
            dims = defaultdict(set)
            if initial_dims is not None:
                for dim, values in initial_dims.items():
                    dims[dim].update(values)
            artifact = _MutableArtifact(
                ref_id=ref_id,
                created_at=created_at,
                created_by_tool=created_by_tool,
                created_by_label=created_by_label,
                dims=dims,
            )
            artifacts[ref_id] = artifact
        elif initial_dims is not None:
            for dim, values in initial_dims.items():
                artifact.dims.setdefault(dim, set()).update(values)
        return artifact

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
        elif name == "ape_output":
            ref_id = _workflow_key(args[0])
            _ensure_artifact(
                ref_id,
                initial_dims=_artifact_metadata_dimensions(
                    ref_id,
                    workflow_input_dims=workflow_input_dims,
                    tool_output_dims=tool_output_dims,
                ),
            )
        elif name == "ape_holds_dim":
            ref_id = _workflow_key(args[0])
            value = _symbol_text(args[1])
            dim = _symbol_text(args[2])
            _ensure_artifact(ref_id).dims.setdefault(dim, set()).add(value)
        elif name == "ape_goal_out":
            goal_index = int(str(args[1]))
            goal_candidates[goal_index].append(_workflow_key(args[2]))

    for artifact_id in {binding.artifact_id for bindings in bindings_by_step.values() for binding in bindings}:
        if artifact_id not in artifacts:
            _ensure_artifact(
                artifact_id,
                initial_dims=_artifact_metadata_dimensions(
                    artifact_id,
                    workflow_input_dims=workflow_input_dims,
                    tool_output_dims=tool_output_dims,
                ),
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
    signature_bindings = tuple(
        _binding_signature(step.time, binding)
        for step in steps
        for binding in step.bindings
    )
    goal_bindings = tuple(
        _goal_signature(goal_index, artifact_id)
        for goal_index, artifact_id in sorted(goal_outputs.items())
    )
    steps_tuple = tuple(steps)

    return WorkflowSolution(
        index=index,
        steps=steps_tuple,
        artifacts=frozen_artifacts,
        goal_outputs=goal_outputs,
        signature_bindings=signature_bindings,
        goal_bindings=goal_bindings,
        workflow_signature_key=_workflow_signature_key(steps_tuple),
        canonical_key=_canonical_solution_key(steps_tuple, frozen_artifacts, goal_outputs),
    )
