"""Optional Python-side precompute layer for legacy direct encodings.

This module emits helper facts that are static or monotone enough to compute in
Python once instead of reconstructing them in ASP at every horizon. The main
uses are:

- workflow-input equivalence compression,
- bindability facts for direct backends,
- output-choice compression for dense output domains,
- safe lower bounds on when a tool or workflow can first become feasible.

The optimized compressed-candidate backend has its own dedicated optimization
pipeline; this module focuses on augmenting the direct fact surface.
"""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Mapping
from dataclasses import dataclass, replace
from itertools import combinations

import clingo

from clindaws.core.models import FactBundle, SnakeConfig, ToolMode
from clindaws.core.ontology import Ontology
from clindaws.core.workflow_input_compression import (
    build_workflow_input_compression_plan,
    workflow_input_compression_stats,
)
from clindaws.translators.fact_writer import (
    _FactWriter,
)
from clindaws.translators.ports import (
    _artifact_profile_terminal_sets,
    _artifact_satisfies_port_requirements,
    _port_requirement_terminal_sets,
)
from clindaws.translators.signatures import (
    _prefer_less_specific_value,
)
from clindaws.translators.builder import (
    _build_roots,
)
from clindaws.translators.utils import (
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


def _symbol_text(symbol: clingo.Symbol) -> str:
    if symbol.type == clingo.SymbolType.Number:
        return str(symbol.number)
    if symbol.type == clingo.SymbolType.String:
        return str(symbol.string)
    return str(symbol)


def _tuple_value(symbol: clingo.Symbol) -> tuple[str, str]:
    return (_symbol_string(symbol.arguments[0]), _symbol_string(symbol.arguments[1]))


def _workflow_input_satisfies_port_requirements(
    artifact_profile: Mapping[str, frozenset[str]],
    port_requirements: Mapping[str, tuple[frozenset[str], ...]],
) -> bool:
    """Return whether a workflow input can satisfy a port.

    Missing workflow-input dimensions are treated as unconstrained rather than
    incompatible. This matches the compressed-candidate backend and APE's
    handling of underspecified inputs such as ImageMagick's Content input.
    """

    for dimension, required_value_sets in port_requirements.items():
        artifact_values = artifact_profile.get(dimension, frozenset())
        if not artifact_values:
            continue
        if not any(artifact_values.intersection(required_values) for required_values in required_value_sets):
            return False
    return True


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
    workflow_input_ids: tuple[str, ...]
    workflow_input_dims: Mapping[str, Mapping[str, tuple[str, ...]]]
    workflow_input_units: Mapping[str, tuple[str, ...]]
    goal_dimensions_by_id: Mapping[str, Mapping[str, tuple[str, ...]]]


@dataclass(frozen=True)
class _PortSignatureFacts:
    signature_by_port: Mapping[str, str]
    signature_dimensions_by_id: Mapping[str, Mapping[str, tuple[str, ...]]]
    stats: Mapping[str, int]


def _parse_direct_facts(facts: str) -> _ParsedDirectFacts:
    tool_input_variants: dict[str, list[str]] = defaultdict(list)
    variant_tool_by_id: dict[str, str] = {}
    input_ports_by_variant: dict[str, list[str]] = defaultdict(list)
    tool_output_groups: dict[str, list[str]] = defaultdict(list)
    output_group_tool_by_id: dict[str, str] = {}
    output_ports_by_group: dict[str, list[str]] = defaultdict(list)
    all_dimensions_by_port: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    workflow_input_ids: set[str] = set()
    workflow_input_dims: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    workflow_input_units: dict[str, list[str]] = defaultdict(list)
    goal_dimensions_by_id: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))

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
        if atom.name == "goal_output":
            goal_id = _symbol_text(atom.arguments[0])
            value = _symbol_string(atom.arguments[1])
            category = _symbol_string(atom.arguments[2])
            goal_dimensions_by_id[goal_id][category].append(value)
            continue
        if atom.name != "holds" or atom.arguments[0].number != 0:
            continue
        inner = atom.arguments[1]
        if inner.name == "avail":
            workflow_input_ids.add(_symbol_string(inner.arguments[0]))
        elif inner.name == "dim":
            wf = _symbol_string(inner.arguments[0])
            workflow_input_ids.add(wf)
            value = _symbol_string(inner.arguments[1])
            category = _symbol_string(inner.arguments[2])
            workflow_input_dims[wf][category].append(value)
        elif inner.name == "unit":
            wf = _symbol_string(inner.arguments[0])
            workflow_input_ids.add(wf)
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
        workflow_input_ids=tuple(sorted(workflow_input_ids)),
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
        goal_dimensions_by_id={
            goal_id: {
                category: tuple(sorted(dict.fromkeys(values)))
                for category, values in sorted(dimensions.items())
            }
            for goal_id, dimensions in sorted(goal_dimensions_by_id.items())
        },
    )


def _signature_key(dimensions: Mapping[str, tuple[str, ...]]) -> tuple[tuple[str, tuple[str, ...]], ...]:
    return tuple(sorted((category, tuple(values)) for category, values in dimensions.items()))


def _profile_key(
    dims: Mapping[str, tuple[str, ...]],
    units: tuple[str, ...] = (),
) -> tuple[tuple[tuple[str, tuple[str, ...]], ...], tuple[str, ...]]:
    return (_signature_key(dims), tuple(units))


def _compress_direct_output_profile_candidates(
    ontology: Ontology,
    *,
    output_profile_terminal_sets: Mapping[str, Mapping[str, frozenset[str]]],
    signature_requirements: Mapping[str, Mapping[str, tuple[frozenset[str], ...]]],
    goal_requirements: Mapping[str, Mapping[str, tuple[frozenset[str], ...]]],
) -> tuple[dict[str, dict[str, tuple[str, ...]]], dict[str, int]]:
    """Compress output-choice terminals by consumer/goal equivalence profile."""

    compressed_candidates: dict[str, dict[str, tuple[str, ...]]] = {}
    dense_candidate_count = 0
    emitted_candidate_count = 0
    equivalence_class_count = 0
    dropped_candidate_count = 0
    compressed_category_count = 0

    for profile_id, profile in sorted(output_profile_terminal_sets.items()):
        category_candidates: dict[str, tuple[str, ...]] = {}
        for category, terminal_values in sorted(profile.items()):
            ordered_values = tuple(sorted(terminal_values))
            dense_candidate_count += len(ordered_values)
            if len(ordered_values) <= 1:
                category_candidates[category] = ordered_values
                emitted_candidate_count += len(ordered_values)
                equivalence_class_count += len(ordered_values)
                continue

            representatives: dict[tuple[tuple[str, ...], tuple[str, ...]], str] = {}
            for terminal_value in ordered_values:
                supported_signatures = tuple(
                    signature_id
                    for signature_id, requirements in sorted(signature_requirements.items())
                    if any(
                        terminal_value in requirement_set
                        for requirement_set in requirements.get(category, ())
                    )
                )
                supported_goals = tuple(
                    goal_id
                    for goal_id, requirements in sorted(goal_requirements.items())
                    if any(
                        terminal_value in requirement_set
                        for requirement_set in requirements.get(category, ())
                    )
                )
                profile_key = (supported_signatures, supported_goals)
                current = representatives.get(profile_key)
                if current is None:
                    representatives[profile_key] = terminal_value
                else:
                    representatives[profile_key] = _prefer_less_specific_value(
                        ontology,
                        current,
                        terminal_value,
                    )

            retained_values = tuple(representatives.values())
            category_candidates[category] = retained_values
            emitted_candidate_count += len(retained_values)
            equivalence_class_count += len(retained_values)
            dropped_candidate_count += len(ordered_values) - len(retained_values)
            if len(retained_values) < len(ordered_values):
                compressed_category_count += 1

        compressed_candidates[profile_id] = category_candidates

    return compressed_candidates, {
        "precompute_output_profile_candidates_dense": dense_candidate_count,
        "precompute_output_profile_candidates": emitted_candidate_count,
        "precompute_output_profile_candidate_equivalence_classes": equivalence_class_count,
        "precompute_output_profile_candidates_dropped": dropped_candidate_count,
        "precompute_output_profile_candidate_categories_compressed": compressed_category_count,
    }


def _compress_plain_output_port_candidates(
    *,
    output_port_terminal_sets: Mapping[str, Mapping[str, frozenset[str]]],
    port_requirements: Mapping[str, Mapping[str, tuple[frozenset[str], ...]]],
    goal_requirements: Mapping[str, Mapping[str, tuple[frozenset[str], ...]]],
) -> tuple[dict[str, dict[str, tuple[str, ...]]], dict[str, int]]:
    """Compress legacy multi-shot output terminals by consumer/goal support."""

    def _build_support_index(
        requirements_by_id: Mapping[str, Mapping[str, tuple[frozenset[str], ...]]],
    ) -> dict[str, dict[str, tuple[str, ...]]]:
        support_index: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
        for item_id, category_requirements in sorted(requirements_by_id.items()):
            for category, requirement_sets in sorted(category_requirements.items()):
                supported_values = set().union(*requirement_sets) if requirement_sets else set()
                for terminal_value in supported_values:
                    support_index[category][terminal_value].add(item_id)
        return {
            category: {
                terminal_value: tuple(sorted(item_ids))
                for terminal_value, item_ids in sorted(value_index.items())
            }
            for category, value_index in sorted(support_index.items())
        }

    compressed_candidates: dict[str, dict[str, tuple[str, ...]]] = {}
    dense_candidate_count = 0
    emitted_candidate_count = 0
    dropped_candidate_count = 0
    compressed_category_count = 0
    retained_category_count = 0
    requiring_check_category_count = 0
    requiring_check_candidate_count = 0
    ports_by_category_and_terminal_value = _build_support_index(port_requirements)
    goals_by_category_and_terminal_value = _build_support_index(goal_requirements)

    for port_id, profile in sorted(output_port_terminal_sets.items()):
        category_candidates: dict[str, tuple[str, ...]] = {}
        for category, terminal_values in sorted(profile.items()):
            retained_category_count += 1
            ordered_values = tuple(sorted(terminal_values))
            dense_candidate_count += len(ordered_values)
            if len(ordered_values) <= 1:
                category_candidates[category] = ordered_values
                emitted_candidate_count += len(ordered_values)
                continue

            representatives: dict[tuple[tuple[str, ...], tuple[str, ...]], str] = {}
            for terminal_value in ordered_values:
                supported_ports = ports_by_category_and_terminal_value.get(category, {}).get(
                    terminal_value,
                    (),
                )
                supported_goals = goals_by_category_and_terminal_value.get(category, {}).get(
                    terminal_value,
                    (),
                )
                profile_key = (supported_ports, supported_goals)
                current = representatives.get(profile_key)
                if current is None or terminal_value < current:
                    representatives[profile_key] = terminal_value

            retained_values = tuple(sorted(representatives.values()))
            category_candidates[category] = retained_values
            emitted_candidate_count += len(retained_values)
            dropped_candidate_count += len(ordered_values) - len(retained_values)
            if len(retained_values) < len(ordered_values):
                compressed_category_count += 1
            if len(retained_values) > 1:
                requiring_check_category_count += 1
                requiring_check_candidate_count += len(retained_values)

        compressed_candidates[port_id] = category_candidates

    return compressed_candidates, {
        "precompute_plain_output_port_candidate_categories_retained": retained_category_count,
        "precompute_plain_output_port_candidate_categories_requiring_check": requiring_check_category_count,
        "precompute_plain_output_port_candidates_dense": dense_candidate_count,
        "precompute_plain_output_port_candidates": emitted_candidate_count,
        "precompute_plain_output_port_candidates_dropped": dropped_candidate_count,
        "precompute_plain_output_port_candidate_categories_compressed": compressed_category_count,
        "precompute_plain_output_port_candidates_requiring_check": requiring_check_candidate_count,
    }


def _emit_plain_multi_shot_check_facts(
    writer: _FactWriter,
    *,
    config: SnakeConfig,
    ontology: Ontology,
    parsed: _ParsedDirectFacts,
) -> dict[str, int]:
    roots = _build_roots(config, ontology)
    output_port_terminal_sets = {
        port_id: _artifact_profile_terminal_sets(ontology, roots, dimensions)
        for port_id, dimensions in parsed.output_dimensions_by_port.items()
    }
    port_requirements = {
        port_id: _port_requirement_terminal_sets(ontology, roots, dimensions)
        for port_id, dimensions in parsed.input_dimensions_by_port.items()
    }
    goal_requirements = {
        goal_id: _port_requirement_terminal_sets(ontology, roots, dimensions)
        for goal_id, dimensions in parsed.goal_dimensions_by_id.items()
    }
    compressed_candidates, stats = _compress_plain_output_port_candidates(
        output_port_terminal_sets=output_port_terminal_sets,
        port_requirements=port_requirements,
        goal_requirements=goal_requirements,
    )

    for port_id, profile in sorted(compressed_candidates.items()):
        for category, terminal_values in sorted(profile.items()):
            for terminal_value in terminal_values:
                writer.emit_fact(
                    "precomputed_output_port_candidate",
                    _quote(port_id),
                    _quote(terminal_value),
                    _quote(category),
                )
            if len(terminal_values) <= 1:
                continue
            for terminal_value in terminal_values:
                writer.emit_fact(
                    "precomputed_output_port_candidate_requires_check",
                    _quote(port_id),
                    _quote(terminal_value),
                    _quote(category),
                )

    return stats


def _emit_direct_output_profile_check_facts(
    writer: _FactWriter,
    *,
    compressed_output_profile_candidates: Mapping[str, Mapping[str, tuple[str, ...]]],
    candidate_signature_supports: Mapping[tuple[str, str, str], frozenset[str]],
    candidate_goal_supports: Mapping[tuple[str, str, str], frozenset[str]],
) -> dict[str, int]:
    retained_category_count = 0
    ambiguous_category_count = 0
    ambiguous_candidate_count = 0

    for profile_id, profile in sorted(compressed_output_profile_candidates.items()):
        for category, terminal_values in sorted(profile.items()):
            retained_category_count += 1
            if len(terminal_values) <= 1:
                continue

            support_profiles = {
                (
                    tuple(sorted(candidate_signature_supports.get((profile_id, terminal_value, category), frozenset()))),
                    tuple(sorted(candidate_goal_supports.get((profile_id, terminal_value, category), frozenset()))),
                )
                for terminal_value in terminal_values
            }
            if len(support_profiles) <= 1:
                continue

            writer.emit_fact(
                "direct_output_profile_category_requires_check",
                _quote(profile_id),
                _quote(category),
            )
            ambiguous_category_count += 1
            for terminal_value in terminal_values:
                writer.emit_fact(
                    "direct_output_profile_candidate_requires_check",
                    _quote(profile_id),
                    _quote(terminal_value),
                    _quote(category),
                )
                ambiguous_candidate_count += 1

    return {
        "precompute_output_profile_candidate_categories_retained": retained_category_count,
        "precompute_output_profile_candidate_categories_requiring_check": ambiguous_category_count,
        "precompute_output_profile_candidates_requiring_check": ambiguous_candidate_count,
    }


def _emit_port_signature_facts(
    writer: _FactWriter,
    parsed: _ParsedDirectFacts,
) -> _PortSignatureFacts:
    repeated_variants = 0
    signature_pairs = 0

    signature_key_by_port = {
        port_id: _signature_key(dimensions)
        for port_id, dimensions in parsed.input_dimensions_by_port.items()
    }
    signature_id_by_key: dict[tuple[tuple[str, tuple[str, ...]], ...], str] = {}
    signature_dimensions_by_id: dict[str, Mapping[str, tuple[str, ...]]] = {}
    signature_by_port: dict[str, str] = {}

    for index, signature_key in enumerate(sorted(set(signature_key_by_port.values())), start=1):
        signature_id = f"sig_{index}"
        signature_id_by_key[signature_key] = signature_id
        signature_dimensions_by_id[signature_id] = {
            category: tuple(values)
            for category, values in signature_key
        }
        writer.emit_fact("direct_signature", _quote(signature_id))
        for category, values in signature_key:
            writer.emit_fact("direct_signature_category", _quote(signature_id), _quote(category))
            for value in values:
                writer.emit_fact(
                    "direct_signature_requires",
                    _quote(signature_id),
                    _quote(value),
                    _quote(category),
                )

    for port_id, signature_key in sorted(signature_key_by_port.items()):
        signature_id = signature_id_by_key[signature_key]
        signature_by_port[port_id] = signature_id
        writer.emit_fact("direct_port_signature", _quote(port_id), _quote(signature_id))

    for port_id, dimensions in sorted(parsed.input_dimensions_by_port.items()):
        for category, values in sorted(dimensions.items()):
            for value in values:
                writer.emit_fact("port_requires", _quote(port_id), _quote(value), _quote(category))

    for port_a, signature_a in sorted(signature_key_by_port.items()):
        for port_b, signature_b in sorted(signature_key_by_port.items()):
            if signature_a == signature_b:
                writer.emit_fact("same_input_signature", _quote(port_a), _quote(port_b))
                signature_pairs += 1

    for variant_id, ports in sorted(parsed.input_ports_by_variant.items()):
        if len(ports) == 1:
            writer.emit_fact("variant_single_input", _quote(variant_id))
        has_repeated_signature = False
        for port_low, port_high in combinations(sorted(ports), 2):
            if signature_key_by_port[port_low] == signature_key_by_port[port_high]:
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

    return _PortSignatureFacts(
        signature_by_port=dict(sorted(signature_by_port.items())),
        signature_dimensions_by_id=dict(sorted(signature_dimensions_by_id.items())),
        stats={
            "precompute_port_requires": sum(
                len(values)
                for dimensions in parsed.input_dimensions_by_port.values()
                for values in dimensions.values()
            ),
            "precompute_same_input_signature": signature_pairs,
            "precompute_variant_repeated_signature_inputs": repeated_variants,
            "precompute_unique_input_signatures": len(signature_dimensions_by_id),
        },
    )


def _emit_single_shot_workflow_input_facts(
    writer: _FactWriter,
    parsed: _ParsedDirectFacts,
) -> dict[str, int]:
    emitted_pairs = 0
    profiles = {
        wf: _profile_key(
            parsed.workflow_input_dims.get(wf, {}),
            parsed.workflow_input_units.get(wf, ()),
        )
        for wf in parsed.workflow_input_ids
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
    plan = build_workflow_input_compression_plan(
        parsed.workflow_input_dims,
        parsed.workflow_input_units,
    )

    planner_artifact_profiles: dict[str, dict[str, tuple[str, ...]]] = {}
    for group in plan.classes:
        rep = group.representative
        for wf in group.members:
            planner_artifact_profiles[wf] = dict(parsed.workflow_input_dims.get(wf, {}))
        writer.emit_fact("canonical_workflow_input", _quote(rep))
        if not group.repeated:
            writer.emit_fact("workflow_input_class_member", _quote(rep), _quote(rep))
            continue

        writer.emit_fact("workflow_input_class_repeated", _quote(rep))
        writer.emit_fact("workflow_input_slot_class_size", _quote(rep), str(len(group.members)))
        for rank, wf in enumerate(group.members, start=1):
            slot_term = _slot_term(rep, wf)
            writer.emit_fact("workflow_input_class_member", _quote(rep), _quote(wf))
            writer.emit_fact("workflow_input_collapsed_member", _quote(wf))
            writer.emit_fact("workflow_input_slot", slot_term, _quote(rep), str(rank))
            writer.emit_fact("workflow_input_slot_source", slot_term, _quote(wf))
            planner_artifact_profiles[slot_term] = dict(parsed.workflow_input_dims.get(wf, {}))
        for wf_a in group.members:
            for wf_b in group.members:
                if wf_a != wf_b:
                    writer.emit_fact("equivalent_workflow_input_pair", _quote(wf_a), _quote(wf_b))

    stats = {
        "precompute_workflow_input_classes": plan.equivalence_class_count,
        "precompute_repeated_workflow_input_classes": plan.repeated_equivalence_class_count,
        "precompute_workflow_input_slots": plan.slot_count,
        "precompute_workflow_input_collapsed_members": plan.collapsed_member_count,
        "precompute_workflow_input_planner_visible_count_uncompressed": (
            plan.planner_visible_count_if_uncompressed
        ),
        "precompute_workflow_input_planner_visible_count_compressed": (
            plan.planner_visible_count_if_compressed
        ),
        "precompute_workflow_input_planner_visible_reduction_if_compressed": (
            plan.planner_visible_reduction_if_compressed
        ),
    }
    stats.update(
        {
            f"precompute_{name}": value
            for name, value in workflow_input_compression_stats(plan).items()
        }
    )
    return planner_artifact_profiles, stats


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
            if _workflow_input_satisfies_port_requirements(profile, requirements):
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


def _emit_multi_shot_signature_compatibility_facts(
    writer: _FactWriter,
    *,
    config: SnakeConfig,
    ontology: Ontology,
    planner_artifact_profiles: Mapping[str, Mapping[str, tuple[str, ...]]],
    parsed: _ParsedDirectFacts,
    port_signatures: _PortSignatureFacts,
) -> dict[str, int]:
    roots = _build_roots(config, ontology)
    signature_requirements = {
        signature_id: _port_requirement_terminal_sets(ontology, roots, dimensions)
        for signature_id, dimensions in port_signatures.signature_dimensions_by_id.items()
    }
    goal_requirements = {
        goal_id: _port_requirement_terminal_sets(ontology, roots, dimensions)
        for goal_id, dimensions in parsed.goal_dimensions_by_id.items()
    }
    base_profiles = {
        artifact_id: _artifact_profile_terminal_sets(ontology, roots, dimensions)
        for artifact_id, dimensions in planner_artifact_profiles.items()
    }

    output_profile_dimensions: dict[
        tuple[tuple[str, tuple[str, ...]], ...],
        Mapping[str, tuple[str, ...]],
    ] = {}
    output_profile_ports: dict[tuple[tuple[str, tuple[str, ...]], ...], list[str]] = defaultdict(list)
    for port_id, dimensions in sorted(parsed.output_dimensions_by_port.items()):
        profile_key = _signature_key(dimensions)
        output_profile_dimensions[profile_key] = dimensions
        output_profile_ports[profile_key].append(port_id)

    output_profile_id_by_key: dict[tuple[tuple[str, tuple[str, ...]], ...], str] = {}
    output_profile_terminal_sets: dict[str, Mapping[str, frozenset[str]]] = {}
    output_profile_port_counts: dict[str, int] = {}
    output_profile_by_port: dict[str, str] = {}
    signature_port_counts: dict[str, int] = defaultdict(int)
    ports_by_signature: dict[str, list[str]] = defaultdict(list)
    for signature_id in port_signatures.signature_by_port.values():
        signature_port_counts[signature_id] += 1
    for port_id, signature_id in sorted(port_signatures.signature_by_port.items()):
        ports_by_signature[signature_id].append(port_id)

    for index, profile_key in enumerate(sorted(output_profile_dimensions), start=1):
        profile_id = f"profile_{index}"
        output_profile_id_by_key[profile_key] = profile_id
        output_profile_terminal_sets[profile_id] = _artifact_profile_terminal_sets(
            ontology,
            roots,
            output_profile_dimensions[profile_key],
        )
        output_profile_port_counts[profile_id] = len(output_profile_ports[profile_key])
        writer.emit_fact("direct_output_profile", _quote(profile_id))
        for port_id in sorted(output_profile_ports[profile_key]):
            output_profile_by_port[port_id] = profile_id
            writer.emit_fact(
                "direct_output_port_profile",
                _quote(port_id),
                _quote(profile_id),
            )

    compatible_profiles_by_signature: dict[str, tuple[str, ...]] = {}
    dense_output_bindable_estimate = 0
    base_bindable_count = 0
    port_initial_artifact_count = 0
    port_bindable_output_profile_count = 0
    profile_signature_candidate_support_count = 0
    profile_goal_candidate_support_count = 0
    goal_category_count = 0
    candidate_signature_support_map: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    candidate_goal_support_map: dict[tuple[str, str, str], set[str]] = defaultdict(set)

    compressed_output_profile_candidates, candidate_compression_stats = (
        _compress_direct_output_profile_candidates(
            ontology,
            output_profile_terminal_sets=output_profile_terminal_sets,
            signature_requirements=signature_requirements,
            goal_requirements=goal_requirements,
        )
    )

    for goal_id, requirements in sorted(goal_requirements.items()):
        for category in sorted(requirements):
            writer.emit_fact("direct_goal_category", goal_id, _quote(category))
            goal_category_count += 1

    for profile_id, profile in sorted(compressed_output_profile_candidates.items()):
        for category, terminal_values in sorted(profile.items()):
            for terminal_value in terminal_values:
                writer.emit_fact(
                    "direct_output_profile_candidate",
                    _quote(profile_id),
                    _quote(terminal_value),
                    _quote(category),
                )

    for signature_id, requirements in sorted(signature_requirements.items()):
        compatible_profiles = tuple(
            sorted(
                profile_id
                for profile_id, profile in output_profile_terminal_sets.items()
                if _artifact_satisfies_port_requirements(profile, requirements)
            )
        )
        compatible_profiles_by_signature[signature_id] = compatible_profiles
        for port_id in sorted(ports_by_signature.get(signature_id, ())):
            for profile_id in compatible_profiles:
                writer.emit_fact(
                    "direct_port_bindable_output_profile",
                    _quote(port_id),
                    _quote(profile_id),
                )
                port_bindable_output_profile_count += 1
        dense_output_bindable_estimate += sum(
            output_profile_port_counts[profile_id]
            for profile_id in compatible_profiles
        ) * signature_port_counts[signature_id]
        compatible_initial_artifacts = tuple(
            artifact_id
            for artifact_id, profile in sorted(base_profiles.items())
            if _workflow_input_satisfies_port_requirements(profile, requirements)
        )
        for artifact_id in compatible_initial_artifacts:
            for port_id in sorted(ports_by_signature.get(signature_id, ())):
                writer.emit_fact(
                    "direct_port_initial_artifact",
                    _quote(port_id),
                    _artifact_term(artifact_id),
                )
                port_initial_artifact_count += 1
        for artifact_id in compatible_initial_artifacts:
            writer.emit_fact(
                "direct_signature_initial_artifact",
                _quote(signature_id),
                _artifact_term(artifact_id),
            )
            base_bindable_count += 1
        for profile_id in compatible_profiles:
            profile = compressed_output_profile_candidates[profile_id]
            for category, requirement_sets in sorted(requirements.items()):
                supported_values = tuple(
                    sorted(
                        terminal_value
                        for terminal_value in profile.get(category, ())
                        if any(terminal_value in requirement_set for requirement_set in requirement_sets)
                    )
                )
                for terminal_value in supported_values:
                    candidate_signature_support_map[(profile_id, terminal_value, category)].add(signature_id)
                    writer.emit_fact(
                        "direct_output_profile_candidate_supports_signature",
                        _quote(profile_id),
                        _quote(signature_id),
                        _quote(terminal_value),
                        _quote(category),
                    )
                    profile_signature_candidate_support_count += 1

    for goal_id, requirements in sorted(goal_requirements.items()):
        for profile_id, profile in sorted(compressed_output_profile_candidates.items()):
            for category, requirement_sets in sorted(requirements.items()):
                supported_values = tuple(
                    sorted(
                        terminal_value
                        for terminal_value in profile.get(category, ())
                        if any(terminal_value in requirement_set for requirement_set in requirement_sets)
                    )
                )
                for terminal_value in supported_values:
                    candidate_goal_support_map[(profile_id, terminal_value, category)].add(goal_id)
                    writer.emit_fact(
                        "direct_output_profile_candidate_supports_goal",
                        _quote(profile_id),
                        goal_id,
                        _quote(terminal_value),
                        _quote(category),
                    )
                    profile_goal_candidate_support_count += 1

    check_fact_stats = _emit_direct_output_profile_check_facts(
        writer,
        compressed_output_profile_candidates=compressed_output_profile_candidates,
        candidate_signature_supports={
            key: frozenset(value)
            for key, value in candidate_signature_support_map.items()
        },
        candidate_goal_supports={
            key: frozenset(value)
            for key, value in candidate_goal_support_map.items()
        },
    )

    support_class_id_by_profiles: dict[tuple[str, ...], str] = {}
    support_class_profiles: dict[str, tuple[str, ...]] = {}
    for signature_id, compatible_profiles in sorted(compatible_profiles_by_signature.items()):
        support_class_id = support_class_id_by_profiles.get(compatible_profiles)
        if support_class_id is None:
            support_class_id = f"support_class_{len(support_class_id_by_profiles) + 1}"
            support_class_id_by_profiles[compatible_profiles] = support_class_id
            support_class_profiles[support_class_id] = compatible_profiles
        writer.emit_fact(
            "direct_signature_support_class",
            _quote(signature_id),
            _quote(support_class_id),
        )

    support_class_profile_count = 0
    for support_class_id, compatible_profiles in sorted(support_class_profiles.items()):
        writer.emit_fact("direct_support_class", _quote(support_class_id))
        for profile_id in compatible_profiles:
            writer.emit_fact(
                "direct_support_class_profile",
                _quote(support_class_id),
                _quote(profile_id),
            )
            support_class_profile_count += 1

    compact_fact_count = (
        len(port_signatures.signature_by_port)
        + len(port_signatures.signature_dimensions_by_id)
        + len(output_profile_by_port)
        + len(support_class_profiles)
        + support_class_profile_count
        + base_bindable_count
        + port_initial_artifact_count
        + port_bindable_output_profile_count
    )

    return {
        "precompute_base_artifact_bindable": base_bindable_count,
        "precompute_port_initial_artifact_bindable": port_initial_artifact_count,
        "precompute_port_bindable_output_profiles": port_bindable_output_profile_count,
        "precompute_unique_output_profiles": len(output_profile_terminal_sets),
        "precompute_signature_support_classes": len(support_class_profiles),
        "precompute_support_class_profiles": support_class_profile_count,
        "precompute_profile_signature_candidate_support": profile_signature_candidate_support_count,
        "precompute_profile_goal_candidate_support": profile_goal_candidate_support_count,
        "precompute_goal_requirement_sets": len(goal_requirements),
        "precompute_goal_categories": goal_category_count,
        "precompute_dense_output_bindable_estimate": dense_output_bindable_estimate,
        "precompute_compact_output_bindable_estimate": compact_fact_count,
        "precompute_compact_output_compatibility_mode_selected": 1,
        **check_fact_stats,
        **candidate_compression_stats,
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
                        _workflow_input_satisfies_port_requirements(profile, requirements)
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
            if goal_distance is not None:
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


def apply_precompute(
    mode: str,
    config: SnakeConfig,
    ontology: Ontology,
    tools: tuple[ToolMode, ...],
    fact_bundle: FactBundle,
    *,
    optimized_programs: bool,
) -> FactBundle:
    """Augment a legacy direct fact bundle with Python-precomputed helper facts."""

    if mode not in {"single-shot", "multi-shot"}:
        return fact_bundle

    if not optimized_programs and mode not in {"single-shot", "multi-shot"}:
        return fact_bundle

    parsed = _parse_direct_facts(fact_bundle.facts)
    writer = _FactWriter()
    stats: dict[str, int] = {}
    if optimized_programs:
        port_signatures = _emit_port_signature_facts(writer, parsed)
        stats.update(port_signatures.stats)

        if mode == "single-shot":
            stats.update(_emit_single_shot_workflow_input_facts(writer, parsed))
            planner_artifact_profiles = dict(parsed.workflow_input_dims)
        else:
            planner_artifact_profiles, workflow_stats = _emit_multi_shot_workflow_input_facts(writer, parsed)
            stats.update(workflow_stats)

        if mode == "single-shot":
            stats.update(
                _emit_bindability_facts(
                    writer,
                    config=config,
                    ontology=ontology,
                    planner_artifact_profiles=planner_artifact_profiles,
                    parsed=parsed,
                )
            )
        else:
            stats.update(
                _emit_multi_shot_signature_compatibility_facts(
                    writer,
                    config=config,
                    ontology=ontology,
                    planner_artifact_profiles=planner_artifact_profiles,
                    parsed=parsed,
                    port_signatures=port_signatures,
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
    else:
        stats.update(
            _emit_plain_multi_shot_check_facts(
                writer,
                config=config,
                ontology=ontology,
                parsed=parsed,
            )
        )

    if writer.fact_count == 0:
        return fact_bundle

    merged_predicates = dict(fact_bundle.predicate_counts)
    for name, count in writer.predicate_counts.items():
        merged_predicates[name] = merged_predicates.get(name, 0) + count

    merged_emit_stats = dict(fact_bundle.emit_stats)
    for name, count in writer.stats().items():
        prefix = "python_precompute" if optimized_programs else "precompute"
        merged_emit_stats[f"{prefix}:{name}"] = count

    if optimized_programs:
        return replace(
            fact_bundle,
            fact_count=fact_bundle.fact_count + writer.fact_count,
            predicate_counts=merged_predicates,
            emit_stats=merged_emit_stats,
            python_precomputed_facts=writer.text(),
            python_precompute_enabled=True,
            python_precompute_fact_count=writer.fact_count,
            python_precompute_stats=dict(sorted(stats.items())),
            internal_schema="direct_precompute_legacy",
            internal_solver_mode="multi-shot" if mode == "multi-shot" else mode,
        )

    merged_facts = fact_bundle.facts
    if merged_facts and not merged_facts.endswith("\n"):
        merged_facts += "\n"
    merged_facts += writer.text()

    return replace(
        fact_bundle,
        facts=merged_facts,
        fact_count=fact_bundle.fact_count + writer.fact_count,
        predicate_counts=merged_predicates,
        emit_stats=merged_emit_stats,
        python_precompute_fact_count=writer.fact_count,
        python_precompute_stats=dict(sorted(stats.items())),
        internal_schema="legacy_direct_check_precompute",
    )


