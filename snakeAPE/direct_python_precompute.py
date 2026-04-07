"""Optional Python-side precompute layer for direct encodings."""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Mapping
from dataclasses import dataclass, replace
from itertools import combinations

import clingo

from .models import FactBundle, SnakeConfig, ToolMode
from .ontology import Ontology
from .translator import (
    _FactWriter,
    _artifact_profile_terminal_sets,
    _artifact_satisfies_port_requirements,
    _build_roots,
    _port_requirement_terminal_sets,
    _quote,
)


def _slot_term(rep: str, wf: str) -> str:
    return f"slot({_quote(rep)}, {_quote(wf)})"


def _artifact_term(artifact_id: str) -> str:
    if artifact_id.startswith("slot("):
        return artifact_id
    return _quote(artifact_id)


def _symbol_string(symbol: clingo.Symbol) -> str:
    return str(symbol.string)


def _tuple_value(symbol: clingo.Symbol) -> tuple[str, str]:
    return (_symbol_string(symbol.arguments[0]), _symbol_string(symbol.arguments[1]))


@dataclass(frozen=True)
class _ParsedDirectFacts:
    tool_input_variants: Mapping[str, tuple[str, ...]]
    variant_tool_by_id: Mapping[str, str]
    input_ports_by_variant: Mapping[str, tuple[str, ...]]
    tool_output_groups: Mapping[str, tuple[str, ...]]
    output_group_tool_by_id: Mapping[str, str]
    output_ports_by_group: Mapping[str, tuple[str, ...]]
    input_dimensions_by_port: Mapping[str, Mapping[str, tuple[str, ...]]]
    output_dimensions_by_port: Mapping[str, Mapping[str, tuple[str, ...]]]
    workflow_input_dims: Mapping[str, Mapping[str, tuple[str, ...]]]
    workflow_input_units: Mapping[str, tuple[str, ...]]


def _parse_direct_facts(facts: str) -> _ParsedDirectFacts:
    tool_input_variants: dict[str, list[str]] = defaultdict(list)
    variant_tool_by_id: dict[str, str] = {}
    input_ports_by_variant: dict[str, list[str]] = defaultdict(list)
    tool_output_groups: dict[str, list[str]] = defaultdict(list)
    output_group_tool_by_id: dict[str, str] = {}
    output_ports_by_group: dict[str, list[str]] = defaultdict(list)
    all_dimensions_by_port: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    workflow_input_dims: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    workflow_input_units: dict[str, list[str]] = defaultdict(list)

    input_port_ids: set[str] = set()
    output_port_ids: set[str] = set()

    for raw_line in facts.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("%") or ":-" in line or not line.endswith("."):
            continue
        atom = clingo.parse_term(line[:-1])
        if atom.name == "tool_input":
            tool_id = _symbol_string(atom.arguments[0])
            variant_id = _symbol_string(atom.arguments[1])
            tool_input_variants[tool_id].append(variant_id)
            variant_tool_by_id[variant_id] = tool_id
            continue
        if atom.name == "tool_output":
            tool_id = _symbol_string(atom.arguments[0])
            output_id = _symbol_string(atom.arguments[1])
            tool_output_groups[tool_id].append(output_id)
            output_group_tool_by_id[output_id] = tool_id
            continue
        if atom.name == "input_port":
            variant_id = _symbol_string(atom.arguments[0])
            port_id = _symbol_string(atom.arguments[1])
            input_ports_by_variant[variant_id].append(port_id)
            input_port_ids.add(port_id)
            continue
        if atom.name == "output_port":
            output_id = _symbol_string(atom.arguments[0])
            port_id = _symbol_string(atom.arguments[1])
            output_ports_by_group[output_id].append(port_id)
            output_port_ids.add(port_id)
            continue
        if atom.name == "dimension":
            port_id = _symbol_string(atom.arguments[0])
            value, category = _tuple_value(atom.arguments[1])
            all_dimensions_by_port[port_id][category].append(value)
            continue
        if atom.name != "holds" or atom.arguments[0].number != 0:
            continue
        inner = atom.arguments[1]
        if inner.name == "dim":
            wf = _symbol_string(inner.arguments[0])
            value = _symbol_string(inner.arguments[1])
            category = _symbol_string(inner.arguments[2])
            workflow_input_dims[wf][category].append(value)
        elif inner.name == "unit":
            wf = _symbol_string(inner.arguments[0])
            workflow_input_units[wf].append(_symbol_string(inner.arguments[1]))

    input_dimensions_by_port = {
        port_id: {
            category: tuple(sorted(dict.fromkeys(values)))
            for category, values in sorted(all_dimensions_by_port.get(port_id, {}).items())
        }
        for port_id in sorted(input_port_ids)
    }
    output_dimensions_by_port = {
        port_id: {
            category: tuple(sorted(dict.fromkeys(values)))
            for category, values in sorted(all_dimensions_by_port.get(port_id, {}).items())
        }
        for port_id in sorted(output_port_ids)
    }

    return _ParsedDirectFacts(
        tool_input_variants={
            tool_id: tuple(sorted(variant_ids))
            for tool_id, variant_ids in sorted(tool_input_variants.items())
        },
        variant_tool_by_id=dict(sorted(variant_tool_by_id.items())),
        input_ports_by_variant={
            variant_id: tuple(sorted(port_ids))
            for variant_id, port_ids in sorted(input_ports_by_variant.items())
        },
        tool_output_groups={
            tool_id: tuple(sorted(output_ids))
            for tool_id, output_ids in sorted(tool_output_groups.items())
        },
        output_group_tool_by_id=dict(sorted(output_group_tool_by_id.items())),
        output_ports_by_group={
            output_id: tuple(sorted(port_ids))
            for output_id, port_ids in sorted(output_ports_by_group.items())
        },
        input_dimensions_by_port=input_dimensions_by_port,
        output_dimensions_by_port=output_dimensions_by_port,
        workflow_input_dims={
            wf: {
                category: tuple(sorted(dict.fromkeys(values)))
                for category, values in sorted(dimensions.items())
            }
            for wf, dimensions in sorted(workflow_input_dims.items())
        },
        workflow_input_units={
            wf: tuple(sorted(dict.fromkeys(units)))
            for wf, units in sorted(workflow_input_units.items())
        },
    )


def _signature_key(dimensions: Mapping[str, tuple[str, ...]]) -> tuple[tuple[str, tuple[str, ...]], ...]:
    return tuple(sorted((category, tuple(values)) for category, values in dimensions.items()))


def _profile_key(
    dims: Mapping[str, tuple[str, ...]],
    units: tuple[str, ...] = (),
) -> tuple[tuple[tuple[str, tuple[str, ...]], ...], tuple[str, ...]]:
    return (_signature_key(dims), tuple(units))


def _emit_port_signature_facts(
    writer: _FactWriter,
    parsed: _ParsedDirectFacts,
) -> dict[str, int]:
    repeated_variants = 0
    signature_pairs = 0

    signature_by_port = {
        port_id: _signature_key(dimensions)
        for port_id, dimensions in parsed.input_dimensions_by_port.items()
    }

    for port_id, dimensions in sorted(parsed.input_dimensions_by_port.items()):
        for category, values in sorted(dimensions.items()):
            for value in values:
                writer.emit_fact("port_requires", _quote(port_id), _quote(value), _quote(category))

    for port_a, signature_a in sorted(signature_by_port.items()):
        for port_b, signature_b in sorted(signature_by_port.items()):
            if signature_a == signature_b:
                writer.emit_fact("same_input_signature", _quote(port_a), _quote(port_b))
                signature_pairs += 1

    for variant_id, ports in sorted(parsed.input_ports_by_variant.items()):
        if len(ports) == 1:
            writer.emit_fact("variant_single_input", _quote(variant_id))
        has_repeated_signature = False
        for port_low, port_high in combinations(sorted(ports), 2):
            if signature_by_port[port_low] == signature_by_port[port_high]:
                writer.emit_fact(
                    "variant_sym_ports",
                    _quote(variant_id),
                    _quote(port_high),
                    _quote(port_low),
                )
                has_repeated_signature = True
        if has_repeated_signature:
            writer.emit_fact("variant_has_repeated_signature_inputs", _quote(variant_id))
            repeated_variants += 1

    return {
        "precompute_port_requires": sum(len(values) for dimensions in parsed.input_dimensions_by_port.values() for values in dimensions.values()),
        "precompute_same_input_signature": signature_pairs,
        "precompute_variant_repeated_signature_inputs": repeated_variants,
    }


def _emit_single_shot_workflow_input_facts(
    writer: _FactWriter,
    parsed: _ParsedDirectFacts,
) -> dict[str, int]:
    emitted_pairs = 0
    profiles = {
        wf: _profile_key(dimensions)
        for wf, dimensions in parsed.workflow_input_dims.items()
    }
    for wf_a, wf_b in combinations(sorted(profiles), 2):
        if profiles[wf_a] == profiles[wf_b]:
            writer.emit_fact("equivalent_workflow_inputs", _quote(wf_a), _quote(wf_b))
            emitted_pairs += 1
    return {
        "precompute_equivalent_workflow_inputs": emitted_pairs,
    }


def _emit_multi_shot_workflow_input_facts(
    writer: _FactWriter,
    parsed: _ParsedDirectFacts,
) -> tuple[dict[str, dict[str, tuple[str, ...]]], dict[str, int]]:
    classes: dict[tuple[tuple[tuple[str, tuple[str, ...]], ...], tuple[str, ...]], list[str]] = defaultdict(list)
    for wf, dimensions in sorted(parsed.workflow_input_dims.items()):
        classes[_profile_key(dimensions, parsed.workflow_input_units.get(wf, ()))].append(wf)

    planner_artifact_profiles: dict[str, dict[str, tuple[str, ...]]] = {}
    repeated_class_count = 0
    slot_count = 0
    for members in classes.values():
        ordered_members = sorted(members)
        rep = ordered_members[0]
        writer.emit_fact("canonical_workflow_input", _quote(rep))
        if len(ordered_members) == 1:
            writer.emit_fact("workflow_input_class_member", _quote(rep), _quote(rep))
            writer.emit_fact("planner_workflow_input", _quote(rep))
            planner_artifact_profiles[rep] = dict(parsed.workflow_input_dims[rep])
            continue

        repeated_class_count += 1
        writer.emit_fact("workflow_input_class_repeated", _quote(rep))
        writer.emit_fact("workflow_input_slot_class_size", _quote(rep), str(len(ordered_members)))
        for rank, wf in enumerate(ordered_members, start=1):
            slot_term = _slot_term(rep, wf)
            writer.emit_fact("workflow_input_class_member", _quote(rep), _quote(wf))
            writer.emit_fact("workflow_input_collapsed_member", _quote(wf))
            writer.emit_fact("workflow_input_slot", slot_term, _quote(rep), str(rank))
            writer.emit_fact("workflow_input_slot_source", slot_term, _quote(wf))
            writer.emit_fact("planner_workflow_input", slot_term)
            planner_artifact_profiles[slot_term] = dict(parsed.workflow_input_dims[wf])
            slot_count += 1
        for wf_a in ordered_members:
            for wf_b in ordered_members:
                if wf_a != wf_b:
                    writer.emit_fact("equivalent_workflow_input_pair", _quote(wf_a), _quote(wf_b))

    return planner_artifact_profiles, {
        "precompute_workflow_input_classes": len(classes),
        "precompute_repeated_workflow_input_classes": repeated_class_count,
        "precompute_workflow_input_slots": slot_count,
    }


def _emit_bindability_facts(
    writer: _FactWriter,
    *,
    config: SnakeConfig,
    ontology: Ontology,
    planner_artifact_profiles: Mapping[str, Mapping[str, tuple[str, ...]]],
    parsed: _ParsedDirectFacts,
) -> dict[str, int]:
    roots = _build_roots(config, ontology)
    port_requirements = {
        port_id: _port_requirement_terminal_sets(ontology, roots, dimensions)
        for port_id, dimensions in parsed.input_dimensions_by_port.items()
    }
    base_profiles = {
        artifact_id: _artifact_profile_terminal_sets(ontology, roots, dimensions)
        for artifact_id, dimensions in planner_artifact_profiles.items()
    }
    output_profiles = {
        port_id: _artifact_profile_terminal_sets(ontology, roots, dimensions)
        for port_id, dimensions in parsed.output_dimensions_by_port.items()
    }

    base_bindable_count = 0
    for port_id, requirements in sorted(port_requirements.items()):
        for artifact_id, profile in sorted(base_profiles.items()):
            if _artifact_satisfies_port_requirements(profile, requirements):
                writer.emit_fact(
                    "precomputed_base_artifact_bindable",
                    _quote(port_id),
                    _artifact_term(artifact_id),
                )
                base_bindable_count += 1

    output_bindable_count = 0
    for consumer_port, requirements in sorted(port_requirements.items()):
        for producer_port, profile in sorted(output_profiles.items()):
            if _artifact_satisfies_port_requirements(profile, requirements):
                writer.emit_fact(
                    "precomputed_output_port_bindable",
                    _quote(consumer_port),
                    _quote(producer_port),
                )
                output_bindable_count += 1

    return {
        "precompute_base_artifact_bindable": base_bindable_count,
        "precompute_output_port_bindable": output_bindable_count,
    }


def _compute_tool_step_windows(
    *,
    config: SnakeConfig,
    ontology: Ontology,
    tools: tuple[ToolMode, ...],
    planner_artifact_profiles: Mapping[str, Mapping[str, tuple[str, ...]]],
    parsed: _ParsedDirectFacts,
    mode: str,
) -> tuple[
    dict[str, int],
    dict[str, int],
    dict[str, dict[object, tuple[tuple[str, object], ...]]],
]:
    roots = _build_roots(config, ontology)
    planner_profiles = tuple(
        _artifact_profile_terminal_sets(ontology, roots, dimensions)
        for dimensions in planner_artifact_profiles.values()
    )
    input_requirements_by_tool = {
        tool.mode_id: tuple(
            _port_requirement_terminal_sets(ontology, roots, port.dimensions)
            for port in tool.inputs
        )
        for tool in tools
    }
    output_profiles_by_tool = {
        tool.mode_id: tuple(
            (port_idx, _artifact_profile_terminal_sets(ontology, roots, port.dimensions))
            for port_idx, port in enumerate(tool.outputs)
        )
        for tool in tools
    }
    goal_requirements = tuple(
        _port_requirement_terminal_sets(ontology, roots, goal_dimensions)
        for goal_dimensions in config.outputs
    )

    producer_support_by_tool: dict[str, dict[object, tuple[tuple[str, object], ...]]] = defaultdict(dict)
    input_port_refs_by_tool: dict[str, tuple[object, ...]] = {}
    reverse_edges: dict[str, set[str]] = defaultdict(set)
    goal_tools: set[str] = set()

    for tool in tools:
        tool_id = tool.mode_id
        if mode == "multi-shot":
            input_variants = parsed.tool_input_variants.get(tool_id, ())
            input_port_ids = parsed.input_ports_by_variant.get(input_variants[0], ()) if input_variants else ()
            output_group_ids = parsed.tool_output_groups.get(tool_id, ())
            output_port_ids = tuple(
                parsed.output_ports_by_group[group_id][0]
                for group_id in output_group_ids
                if parsed.output_ports_by_group.get(group_id)
            )
        else:
            input_port_ids = tuple(range(len(input_requirements_by_tool[tool_id])))
            output_port_ids = tuple(range(len(output_profiles_by_tool[tool_id])))
        input_port_refs_by_tool[tool_id] = tuple(input_port_ids)

        for port_idx, requirements in enumerate(input_requirements_by_tool[tool_id]):
            consumer_port_ref: object = input_port_ids[port_idx] if port_idx < len(input_port_ids) else port_idx
            supported_producers: list[tuple[str, object]] = []
            for producer_tool_id, output_profiles in output_profiles_by_tool.items():
                producer_output_ids: tuple[object, ...]
                if mode == "multi-shot":
                    producer_group_ids = parsed.tool_output_groups.get(producer_tool_id, ())
                    producer_output_ids = tuple(
                        parsed.output_ports_by_group[group_id][0]
                        for group_id in producer_group_ids
                        if parsed.output_ports_by_group.get(group_id)
                    )
                else:
                    producer_output_ids = tuple(range(len(output_profiles)))
                for producer_port_idx, output_profile in output_profiles:
                    if _artifact_satisfies_port_requirements(output_profile, requirements):
                        producer_port_ref: object = (
                            producer_output_ids[producer_port_idx]
                            if producer_port_idx < len(producer_output_ids)
                            else producer_port_idx
                        )
                        supported_producers.append((producer_tool_id, producer_port_ref))
                        reverse_edges[tool_id].add(producer_tool_id)
            producer_support_by_tool[tool_id][consumer_port_ref] = tuple(sorted(set(supported_producers)))

        if goal_requirements and all(
            any(
                _artifact_satisfies_port_requirements(output_profile, requirements)
                for _port_idx, output_profile in output_profiles_by_tool[tool_id]
            )
            for requirements in goal_requirements
        ):
            goal_tools.add(tool_id)

    min_step_by_tool: dict[str, int] = {}
    changed = True
    while changed:
        changed = False
        for tool in tools:
            tool_id = tool.mode_id
            input_requirements = input_requirements_by_tool[tool_id]
            if not input_requirements:
                tool_step = 1
            else:
                port_steps: list[int] = []
                for port_idx, requirements in enumerate(input_requirements):
                    input_port_ref = input_port_refs_by_tool.get(tool_id, ())
                    consumer_port_ref: object = (
                        input_port_ref[port_idx]
                        if port_idx < len(input_port_ref)
                        else port_idx
                    )
                    earliest_source_step: int | None = 0 if any(
                        _artifact_satisfies_port_requirements(profile, requirements)
                        for profile in planner_profiles
                    ) else None
                    for producer_tool_id, _producer_port_idx in producer_support_by_tool[tool_id].get(consumer_port_ref, ()):
                        producer_step = min_step_by_tool.get(producer_tool_id)
                        if producer_step is None:
                            continue
                        if earliest_source_step is None or producer_step < earliest_source_step:
                            earliest_source_step = producer_step
                    if earliest_source_step is None:
                        port_steps = []
                        break
                    port_steps.append(earliest_source_step)
                if not port_steps:
                    continue
                tool_step = max(port_steps) + 1

            current = min_step_by_tool.get(tool_id)
            if current is None or tool_step < current:
                min_step_by_tool[tool_id] = tool_step
                changed = True

    goal_distance_by_tool: dict[str, int] = {}
    frontier: deque[str] = deque(sorted(goal_tools))
    for tool_id in frontier:
        goal_distance_by_tool[tool_id] = 0
    while frontier:
        consumer_tool_id = frontier.popleft()
        next_distance = goal_distance_by_tool[consumer_tool_id] + 1
        for producer_tool_id in sorted(reverse_edges.get(consumer_tool_id, set())):
            if producer_tool_id in goal_distance_by_tool:
                continue
            goal_distance_by_tool[producer_tool_id] = next_distance
            frontier.append(producer_tool_id)

    return min_step_by_tool, goal_distance_by_tool, {
        tool_id: dict(ports)
        for tool_id, ports in producer_support_by_tool.items()
    }


def _emit_step_window_facts(
    writer: _FactWriter,
    *,
    config: SnakeConfig,
    min_step_by_tool: Mapping[str, int],
    goal_distance_by_tool: Mapping[str, int],
    producer_support_by_tool: Mapping[str, Mapping[object, tuple[tuple[str, object], ...]]],
    mode: str,
) -> dict[str, int]:
    tool_allowed_count = 0

    for tool_id, min_step in sorted(min_step_by_tool.items()):
        max_step = config.solution_length_max
        if config.use_all_generated_data != "NONE":
            goal_distance = goal_distance_by_tool.get(tool_id)
            if goal_distance is None:
                continue
            max_step = min(max_step, config.solution_length_max - goal_distance)
        if max_step < min_step:
            continue
        for step_index in range(min_step, max_step + 1):
            writer.emit_fact("precomputed_tool_allowed_at_step", _quote(tool_id), str(step_index))
            tool_allowed_count += 1

    for consumer_tool_id, ports in sorted(producer_support_by_tool.items()):
        consumer_min_step = min_step_by_tool.get(consumer_tool_id)
        if consumer_min_step is None:
            continue
    return {
        "precompute_tool_min_step_entries": len(min_step_by_tool),
        "precompute_goal_distance_entries": len(goal_distance_by_tool),
        "precompute_tool_allowed_at_step": tool_allowed_count,
    }


def apply_direct_python_precompute(
    mode: str,
    config: SnakeConfig,
    ontology: Ontology,
    tools: tuple[ToolMode, ...],
    fact_bundle: FactBundle,
) -> FactBundle:
    """Augment a direct fact bundle with Python-precomputed helper facts."""

    if mode not in {"single-shot", "multi-shot"}:
        return fact_bundle

    parsed = _parse_direct_facts(fact_bundle.facts)
    writer = _FactWriter()
    stats: dict[str, int] = {}
    stats.update(_emit_port_signature_facts(writer, parsed))

    if mode == "single-shot":
        stats.update(_emit_single_shot_workflow_input_facts(writer, parsed))
        planner_artifact_profiles = dict(parsed.workflow_input_dims)
    else:
        planner_artifact_profiles, workflow_stats = _emit_multi_shot_workflow_input_facts(writer, parsed)
        stats.update(workflow_stats)

    stats.update(
        _emit_bindability_facts(
            writer,
            config=config,
            ontology=ontology,
            planner_artifact_profiles=planner_artifact_profiles,
            parsed=parsed,
        )
    )
    min_step_by_tool, goal_distance_by_tool, producer_support_by_tool = _compute_tool_step_windows(
        config=config,
        ontology=ontology,
        tools=tools,
        planner_artifact_profiles=planner_artifact_profiles,
        parsed=parsed,
        mode=mode,
    )
    stats.update(
        _emit_step_window_facts(
            writer,
            config=config,
            min_step_by_tool=min_step_by_tool,
            goal_distance_by_tool=goal_distance_by_tool,
            producer_support_by_tool=producer_support_by_tool,
            mode=mode,
        )
    )

    if writer.fact_count == 0:
        return fact_bundle

    merged_predicates = dict(fact_bundle.predicate_counts)
    for name, count in writer.predicate_counts.items():
        merged_predicates[name] = merged_predicates.get(name, 0) + count

    merged_emit_stats = dict(fact_bundle.emit_stats)
    for name, count in writer.stats().items():
        merged_emit_stats[f"python_precompute:{name}"] = count

    return replace(
        fact_bundle,
        fact_count=fact_bundle.fact_count + writer.fact_count,
        predicate_counts=merged_predicates,
        emit_stats=merged_emit_stats,
        python_precomputed_facts=writer.text(),
        python_precompute_enabled=True,
        python_precompute_fact_count=writer.fact_count,
        python_precompute_stats=dict(sorted(stats.items())),
    )
