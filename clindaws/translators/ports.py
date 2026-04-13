from __future__ import annotations
from collections import defaultdict
from collections.abc import Iterable, Mapping

from clindaws.core.ontology import Ontology

from clindaws.translators.utils import _dedupe_stable, _normalize_dim_map
from clindaws.translators.signatures import _prefer_less_specific_value
from clindaws.translators.resolvers import _ExpansionResolver



def _terminal_values_for_declared_values(
    ontology: Ontology,
    roots: Mapping[str, frozenset[str]],
    dimension: str,
    values: Iterable[str],
) -> frozenset[str]:
    """Return all terminal values an input/output declaration may realize."""

    allowed = roots.get(dimension)
    terminals: set[str] = set()
    for value in values:
        terminals.update(
            ontology.terminal_descendants_of(
                value,
                within=allowed if allowed else None,
            )
        )
    return frozenset(terminals)
def _artifact_profile_terminal_sets(
    ontology: Ontology,
    roots: Mapping[str, frozenset[str]],
    dimensions: Mapping[str, Iterable[str]],
) -> dict[str, frozenset[str]]:
    """Convert declared artifact dimensions into terminal-value capability sets."""

    normalized = _normalize_dim_map(dimensions)
    return {
        dimension: _terminal_values_for_declared_values(ontology, roots, dimension, values)
        for dimension, values in normalized.items()
    }
def _port_requirement_terminal_sets(
    ontology: Ontology,
    roots: Mapping[str, frozenset[str]],
    dimensions: Mapping[str, Iterable[str]],
) -> dict[str, tuple[frozenset[str], ...]]:
    """Convert an input/goal declaration into satisfiable terminal requirement sets."""

    normalized = _normalize_dim_map(dimensions)
    return {
        dimension: tuple(
            _terminal_values_for_declared_values(ontology, roots, dimension, (value,))
            for value in values
        )
        for dimension, values in normalized.items()
    }
def _artifact_profile_key(profile: Mapping[str, frozenset[str]]) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Return a stable key for a terminalized artifact capability profile."""

    return tuple(
        sorted((dimension, tuple(sorted(values))) for dimension, values in profile.items())
    )
def _artifact_satisfies_port_requirements(
    artifact_profile: Mapping[str, frozenset[str]],
    port_requirements: Mapping[str, tuple[frozenset[str], ...]],
) -> bool:
    """Return whether an optimistic artifact profile can satisfy a port/goal."""

    for dimension, required_value_sets in port_requirements.items():
        artifact_values = artifact_profile.get(dimension, frozenset())
        if not artifact_values:
            return False
        if not any(artifact_values.intersection(required_values) for required_values in required_value_sets):
            return False
    return True
def _dynamic_port_expansion(
    resolver: _ExpansionResolver,
    dims: Mapping[str, tuple[str, ...]],
    *,
    expand_outputs: bool,
) -> tuple[tuple[tuple[str, str], ...], int]:
    """Expand one port without materializing cross-products across ports."""

    port_values: list[tuple[str, str]] = []
    variant_cardinality = 1

    for dim, values in sorted(dims.items()):
        expanded_values = _dedupe_stable(
            expanded_value
            for value in values
            for expanded_value in resolver.expanded_values(
                dim,
                value,
                expand_outputs=expand_outputs,
            )
        )
        variant_cardinality *= max(len(expanded_values), 1)
        port_values.extend((dim, expanded_value) for expanded_value in expanded_values)

    return tuple(port_values), variant_cardinality
def _group_port_values_by_dimension(
    port_values: tuple[tuple[str, str], ...],
) -> dict[str, tuple[str, ...]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for dim, value in port_values:
        grouped[dim].append(value)
    return {
        dim: tuple(values)
        for dim, values in grouped.items()
    }
def _output_port_group_key(
    output_port: Mapping[str, object],
) -> tuple[tuple[tuple[str, tuple[str, ...]], ...], tuple[tuple[str, tuple[str, ...]], ...]]:
    declared_dims = output_port["declared_dims"]
    port_values_by_dimension = output_port["port_values_by_dimension"]
    assert isinstance(declared_dims, Mapping)
    assert isinstance(port_values_by_dimension, Mapping)
    return (
        tuple((str(dim), tuple(str(value) for value in values)) for dim, values in sorted(declared_dims.items())),
        tuple(
            (str(dim), tuple(str(value) for value in values))
            for dim, values in sorted(port_values_by_dimension.items())
        ),
    )
def _compress_duplicate_output_ports(
    output_ports: tuple[dict[str, object], ...],
) -> tuple[dict[str, object], ...]:
    grouped: list[dict[str, object]] = []
    grouped_by_key: dict[
        tuple[tuple[tuple[str, tuple[str, ...]], ...], tuple[tuple[str, tuple[str, ...]], ...]],
        dict[str, object],
    ] = {}

    for output_port in output_ports:
        group_key = _output_port_group_key(output_port)
        existing = grouped_by_key.get(group_key)
        multiplicity = int(output_port.get("multiplicity", 1))
        source_port_indices = tuple(
            sorted(
                int(port_idx)
                for port_idx in output_port.get(
                    "source_port_indices",
                    (int(output_port["port_idx"]),),
                )
            )
        )
        if existing is None:
            grouped_port = {
                **output_port,
                "multiplicity": multiplicity,
                "source_port_indices": source_port_indices,
            }
            grouped.append(grouped_port)
            grouped_by_key[group_key] = grouped_port
        else:
            existing["multiplicity"] = int(existing.get("multiplicity", 1)) + multiplicity
            existing["source_port_indices"] = tuple(
                sorted(
                    set(int(port_idx) for port_idx in existing.get("source_port_indices", ()))
                    | set(source_port_indices)
                )
            )

    return tuple(grouped)
def _workflow_input_matches_dynamic_port(
    ontology: Ontology,
    workflow_input: Mapping[str, tuple[str, ...]],
    port_values_by_dimension: Mapping[str, tuple[str, ...]],
) -> bool:
    for dim, accepted_values in port_values_by_dimension.items():
        actual_values = workflow_input.get(dim, ())
        # A workflow input that omits a dimension leaves that dimension
        # unconstrained rather than making the port incompatible.
        if not actual_values:
            continue
        if not any(
            actual_value in ontology.ancestors_of(required_value)
            for actual_value in actual_values
            for required_value in accepted_values
        ):
            return False
    return True
def _dynamic_output_matches_dynamic_input(
    output_values_by_dimension: Mapping[str, tuple[str, ...]],
    input_values_by_dimension: Mapping[str, tuple[str, ...]],
) -> bool:
    for dim, required_values in input_values_by_dimension.items():
        produced_values = output_values_by_dimension.get(dim, ())
        if not produced_values:
            return False
        if not set(required_values).intersection(produced_values):
            return False
    return True
def _dynamic_output_matches_dynamic_input_fset(
    output_fsets: Mapping[str, frozenset[str]],
    input_fsets: Mapping[str, frozenset[str]],
) -> bool:
    for dim, required_fset in input_fsets.items():
        produced_fset = output_fsets.get(dim)
        if not produced_fset:
            return False
        if required_fset.isdisjoint(produced_fset):
            return False
    return True
def _compress_dynamic_output_choice_values(
    ontology: Ontology,
    output_values_by_dimension: Mapping[str, tuple[str, ...]],
    output_fsets: Mapping[str, frozenset[str]],
    bindable_input_ports: tuple[Mapping[str, object], ...],
    goal_port_values: tuple[Mapping[str, tuple[str, ...]], ...],
    goal_fsets: tuple[Mapping[str, frozenset[str]], ...],
    *,
    preserve_goal_profiles: bool,
) -> dict[str, tuple[str, ...]]:
    globally_bindable_input_ports = tuple(
        input_port
        for input_port in bindable_input_ports
        if _dynamic_output_matches_dynamic_input_fset(
            output_fsets,
            input_port["port_values_fset"],
        )
    )
    globally_bindable_goal_ids = tuple(
        goal_id
        for goal_id, goal_fset in enumerate(goal_fsets)
        if _dynamic_output_matches_dynamic_input_fset(output_fsets, goal_fset)
    )

    default_consumer_signatures_by_dimension: dict[str, set[int]] = defaultdict(set)
    consumer_signatures_by_dimension_value: dict[str, dict[str, set[int]]] = defaultdict(
        lambda: defaultdict(set)
    )
    for input_port in globally_bindable_input_ports:
        signature_id = int(input_port["signature_id"])
        input_dims = input_port.get("signature_requirements", input_port["port_values_by_dimension"])
        assert isinstance(input_dims, Mapping)
        for dim in output_values_by_dimension:
            required_values = input_dims.get(dim)
            if required_values is None:
                default_consumer_signatures_by_dimension[dim].add(signature_id)
                continue
            for value in required_values:
                consumer_signatures_by_dimension_value[dim][value].add(signature_id)

    default_goal_ids_by_dimension: dict[str, set[int]] = defaultdict(set)
    goal_ids_by_dimension_value: dict[str, dict[str, set[int]]] = defaultdict(
        lambda: defaultdict(set)
    )
    for goal_id in globally_bindable_goal_ids:
        goal_dims = goal_port_values[goal_id]
        for dim in output_values_by_dimension:
            required_values = goal_dims.get(dim)
            if required_values is None:
                default_goal_ids_by_dimension[dim].add(goal_id)
                continue
            for value in required_values:
                goal_ids_by_dimension_value[dim][value].add(goal_id)

    compressed: dict[str, tuple[str, ...]] = {}
    for dim, values in output_values_by_dimension.items():
        if len(values) <= 1:
            compressed[dim] = values
            continue

        representatives: dict[tuple[tuple[int, ...], tuple[int, ...]], str] = {}
        for value in values:
            consumer_profile = tuple(
                sorted(
                    default_consumer_signatures_by_dimension.get(dim, set())
                    | consumer_signatures_by_dimension_value[dim].get(value, set())
                )
            )
            goal_profile = tuple(
                sorted(
                    default_goal_ids_by_dimension.get(dim, set())
                    | goal_ids_by_dimension_value[dim].get(value, set())
                )
            )
            profile_key = (
                consumer_profile,
                goal_profile if preserve_goal_profiles else (),
            )
            if profile_key in representatives:
                representatives[profile_key] = _prefer_less_specific_value(
                    ontology,
                    representatives[profile_key],
                    value,
                )
            else:
                representatives[profile_key] = value

        compressed[dim] = tuple(representatives.values())

    return compressed
def _compressed_output_supports_signature(
    output_fsets: Mapping[str, frozenset[str]],
    signature_id: int,
    signature_profiles_by_id: Mapping[int, Mapping[str, tuple[int, tuple[str, ...]]]],
    profile_accepts_by_id: Mapping[int, tuple[str, ...]],
) -> bool:
    category_profiles = signature_profiles_by_id.get(signature_id, {})
    for dim, (profile_id, _values) in category_profiles.items():
        produced_fset = output_fsets.get(dim)
        if not produced_fset:
            return False
        if produced_fset.isdisjoint(profile_accepts_by_id.get(profile_id, ())):
            return False
    return True
