from __future__ import annotations
from collections.abc import Iterable, Mapping

from clindaws.core.models import ToolMode
from clindaws.core.ontology import Ontology

from clindaws.translators.utils import _dedupe_stable, _normalize_dim_map



def _prefer_less_specific_value(
    ontology: Ontology,
    current: str,
    candidate: str,
) -> str:
    current_depth = len(ontology.ancestors_of(current))
    candidate_depth = len(ontology.ancestors_of(candidate))
    if candidate_depth < current_depth:
        return candidate
    if candidate_depth > current_depth:
        return current
    return candidate if candidate < current else current
def _reduce_requirement_values(
    ontology: Ontology,
    values: Iterable[str],
) -> tuple[str, ...]:
    """Drop requirement values subsumed by a more general retained value."""

    ordered_values = _dedupe_stable(values)
    if len(ordered_values) <= 1:
        return ordered_values
    if all(not ontology.child_map.get(value, frozenset()) for value in ordered_values):
        return ordered_values

    reduced: list[str] = []
    for value in ordered_values:
        value_ancestors = ontology.ancestors_of(value)
        if any(existing in value_ancestors for existing in reduced):
            continue
        reduced = [
            existing
            for existing in reduced
            if value not in ontology.ancestors_of(existing)
        ]
        reduced.append(value)
    return tuple(reduced)
def _reduce_signature_requirements(
    ontology: Ontology,
    requirements: Mapping[str, tuple[str, ...]],
) -> dict[str, tuple[str, ...]]:
    reduced: dict[str, tuple[str, ...]] = {}
    for dim, values in sorted(requirements.items()):
        reduced_values = _reduce_requirement_values(ontology, values)
        if reduced_values:
            reduced[dim] = reduced_values
    return reduced
def _signature_key(
    requirements: Mapping[str, tuple[str, ...]],
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    return tuple(
        (dim, tuple(values))
        for dim, values in sorted(requirements.items())
    )
def _assign_dynamic_signature_profiles(
    ontology: Ontology,
    roots: Mapping[str, frozenset[str]],
    input_ports: Iterable[dict[str, object]],
) -> tuple[
    dict[int, dict[str, tuple[int, tuple[str, ...]]]],
    dict[int, tuple[str, ...]],
    dict[int, tuple[str, ...]],
    dict[str, int],
]:
    """Assign reduced signature/profile ids to dynamic input ports."""

    raw_requirement_count = 0
    reduced_requirement_count = 0
    signature_ids_by_key: dict[tuple[tuple[str, tuple[str, ...]], ...], int] = {}
    profile_ids_by_key: dict[tuple[str, tuple[str, ...]], int] = {}
    signature_profiles_by_id: dict[int, dict[str, tuple[int, tuple[str, ...]]]] = {}
    profile_values_by_id: dict[int, tuple[str, ...]] = {}
    profile_accepts_by_id: dict[int, tuple[str, ...]] = {}

    for input_port in input_ports:
        port_requirements = input_port["port_values_by_dimension"]
        assert isinstance(port_requirements, Mapping)
        raw_requirement_count += sum(len(values) for values in port_requirements.values())
        reduced_requirements = _reduce_signature_requirements(ontology, port_requirements)
        reduced_requirement_count += sum(len(values) for values in reduced_requirements.values())
        input_port["signature_requirements"] = reduced_requirements

        signature_key = _signature_key(reduced_requirements)
        signature_id = signature_ids_by_key.setdefault(signature_key, len(signature_ids_by_key))
        input_port["signature_id"] = signature_id

    for signature_key, signature_id in sorted(signature_ids_by_key.items(), key=lambda item: item[1]):
        category_profiles: dict[str, tuple[int, tuple[str, ...]]] = {}
        for category, values in signature_key:
            profile_key = (category, values)
            profile_id = profile_ids_by_key.setdefault(profile_key, len(profile_ids_by_key))
            profile_values_by_id.setdefault(profile_id, values)
            profile_accepts_by_id.setdefault(
                profile_id,
                _dedupe_stable(
                    actual_value
                    for required_value in values
                    for actual_value in ontology.terminal_descendants_of(
                        required_value,
                        within=roots.get(category, frozenset()),
                    )
                ),
            )
            category_profiles[category] = (profile_id, values)
        signature_profiles_by_id[signature_id] = category_profiles

    return (
        signature_profiles_by_id,
        profile_values_by_id,
        profile_accepts_by_id,
        {
            "dynamic_raw_signature_requirement_count": raw_requirement_count,
            "dynamic_reduced_signature_requirement_count": reduced_requirement_count,
            "dynamic_signature_count": len(signature_profiles_by_id),
            "dynamic_profile_count": len(profile_values_by_id),
        },
    )
def _port_signature(spec) -> tuple[tuple[str, tuple[str, ...]], ...]:
    return tuple(
        (str(dim), tuple(str(value) for value in values))
        for dim, values in sorted(_normalize_dim_map(spec.dimensions).items())
    )
def _tool_input_signatures(tools: tuple[ToolMode, ...]) -> dict[str, tuple[tuple[tuple[str, tuple[str, ...]], ...], ...]]:
    signatures: dict[str, tuple[tuple[tuple[str, tuple[str, ...]], ...], ...]] = {}
    for tool in tools:
        signatures[tool.mode_id] = tuple(_port_signature(port) for port in tool.inputs)
    return signatures
