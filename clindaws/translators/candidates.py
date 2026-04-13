from __future__ import annotations
from collections import defaultdict
from collections.abc import Iterable, Mapping

from clindaws.core.models import SnakeConfig, ToolMode
from clindaws.core.ontology import Ontology

from clindaws.translators.ports import _port_requirement_terminal_sets, _artifact_profile_key, _artifact_profile_terminal_sets, _artifact_satisfies_port_requirements, _compressed_output_supports_signature



def _compute_ape_multi_shot_earliest_solution_step(
    config: SnakeConfig,
    ontology: Ontology,
    tools: tuple[ToolMode, ...],
    roots: Mapping[str, frozenset[str]],
) -> int:
    """Compute a safe optimistic lower bound for the first solvable horizon."""

    input_requirements_by_tool = {
        tool.mode_id: tuple(
            _port_requirement_terminal_sets(ontology, roots, port.dimensions)
            for port in tool.inputs
        )
        for tool in tools
    }
    output_profiles_by_tool = {
        tool.mode_id: tuple(
            _artifact_profile_terminal_sets(ontology, roots, port.dimensions)
            for port in tool.outputs
        )
        for tool in tools
    }
    goal_requirements = tuple(
        _port_requirement_terminal_sets(ontology, roots, goal_dimensions)
        for goal_dimensions in config.outputs
    )

    profile_steps: dict[tuple[tuple[str, tuple[str, ...]], ...], int] = {}
    profiles_by_key: dict[tuple[tuple[str, tuple[str, ...]], ...], dict[str, frozenset[str]]] = {}

    def _register_profile(profile: Mapping[str, frozenset[str]], step: int) -> bool:
        key = _artifact_profile_key(profile)
        current = profile_steps.get(key)
        if current is not None and current <= step:
            return False
        profile_steps[key] = step
        profiles_by_key[key] = dict(profile)
        return True

    for workflow_input in config.inputs:
        _register_profile(
            _artifact_profile_terminal_sets(ontology, roots, workflow_input),
            0,
        )

    changed = True
    while changed:
        changed = False
        available_profiles = tuple(
            (profiles_by_key[key], step)
            for key, step in profile_steps.items()
        )
        for tool in tools:
            input_requirements = input_requirements_by_tool[tool.mode_id]
            if not input_requirements:
                tool_step = 1
            else:
                port_steps: list[int] = []
                for requirements in input_requirements:
                    earliest_port_step = min(
                        (
                            step
                            for profile, step in available_profiles
                            if _artifact_satisfies_port_requirements(profile, requirements)
                        ),
                        default=config.solution_length_max + 1,
                    )
                    if earliest_port_step > config.solution_length_max:
                        port_steps = []
                        break
                    port_steps.append(earliest_port_step)
                if not port_steps:
                    continue
                tool_step = max(port_steps) + 1

            if tool_step > config.solution_length_max:
                continue
            for output_profile in output_profiles_by_tool[tool.mode_id]:
                changed = _register_profile(output_profile, tool_step) or changed

    earliest_goal_step = 1
    for requirements in goal_requirements:
        goal_step = min(
            (
                step
                for profile, step in (
                    (profiles_by_key[key], step)
                    for key, step in profile_steps.items()
                )
                if _artifact_satisfies_port_requirements(profile, requirements)
            ),
            default=config.solution_length_max + 1,
        )
        earliest_goal_step = max(earliest_goal_step, goal_step)

    return max(1, earliest_goal_step)
def _compute_dynamic_candidate_min_steps(
    candidate_records: Iterable[Mapping[str, object]],
    workflow_bindable_ports: Mapping[str, set[int]],
    produced_bindable_ports: Mapping[str, Mapping[int, set[str]]],
) -> dict[str, int]:
    """Compute the earliest feasible step for each dynamic candidate.

    A candidate can run at step 1 if every input port can be bound from a
    workflow input. Otherwise, each non-workflow-bound port must be fed by a
    producer candidate from a strictly earlier step.
    """
    candidate_input_ports: dict[str, tuple[int, ...]] = {}
    for record in candidate_records:
        candidate_id = str(record["candidate_id"])
        candidate_input_ports[candidate_id] = tuple(
            int(port["port_idx"])
            for port in tuple(record["input_ports"])
        )

    min_required_producers_by_candidate: dict[str, int] = {}
    for candidate_id, input_ports in candidate_input_ports.items():
        uncovered_ports = tuple(
            port_idx
            for port_idx in input_ports
            if port_idx not in workflow_bindable_ports.get(candidate_id, set())
        )
        if not uncovered_ports:
            min_required_producers_by_candidate[candidate_id] = 0
            continue

        port_bit_index = {port_idx: bit for bit, port_idx in enumerate(uncovered_ports)}
        full_mask = (1 << len(uncovered_ports)) - 1
        producer_to_mask: dict[str, int] = {}
        for port_idx in uncovered_ports:
            for producer_candidate in produced_bindable_ports.get(candidate_id, {}).get(port_idx, set()):
                producer_to_mask[producer_candidate] = (
                    producer_to_mask.get(producer_candidate, 0)
                    | (1 << port_bit_index[port_idx])
                )
        producer_masks = {mask for mask in producer_to_mask.values() if mask}
        best_cover = len(uncovered_ports)
        frontier = {0: 0}
        for producer_mask in sorted(producer_masks):
            next_frontier = dict(frontier)
            for covered_mask, used_count in frontier.items():
                new_mask = covered_mask | producer_mask
                new_count = used_count + 1
                old_count = next_frontier.get(new_mask)
                if old_count is None or new_count < old_count:
                    next_frontier[new_mask] = new_count
            frontier = next_frontier
        best_cover = frontier.get(full_mask, best_cover)
        min_required_producers_by_candidate[candidate_id] = best_cover

    min_step_by_candidate: dict[str, int] = {}
    changed = True
    while changed:
        changed = False
        for candidate_id, input_ports in candidate_input_ports.items():
            if not input_ports:
                candidate_step = 1
            else:
                port_steps: list[int] = []
                for port_idx in input_ports:
                    feasible_steps: list[int] = []
                    if port_idx in workflow_bindable_ports.get(candidate_id, set()):
                        feasible_steps.append(1)
                    for producer_candidate in produced_bindable_ports.get(candidate_id, {}).get(port_idx, set()):
                        producer_step = min_step_by_candidate.get(producer_candidate)
                        if producer_step is not None:
                            feasible_steps.append(producer_step + 1)
                    if not feasible_steps:
                        port_steps = []
                        break
                    port_steps.append(min(feasible_steps))
                if not port_steps:
                    continue
                candidate_step = max(
                    max(port_steps),
                    1 + min_required_producers_by_candidate.get(candidate_id, 0),
                )

            current_step = min_step_by_candidate.get(candidate_id)
            if current_step is None or candidate_step < current_step:
                min_step_by_candidate[candidate_id] = candidate_step
                changed = True

    return min_step_by_candidate
def _collect_compressed_dynamic_bindability_surface(
    bindable_pairs: Iterable[tuple[str, int, str, int]],
    *,
    relevant_records: Iterable[Mapping[str, object]],
    signature_profiles_by_id: Mapping[int, Mapping[str, tuple[int, tuple[str, ...]]]],
    profile_accepts_by_id: Mapping[int, tuple[str, ...]],
) -> tuple[
    dict[str, set[str]],
    dict[str, dict[int, set[str]]],
    dict[int, set[tuple[str, int]]],
    dict[str, int],
]:
    relevant_candidate_ids = {
        str(record["candidate_id"])
        for record in relevant_records
    }
    input_signature_by_port: dict[tuple[str, int], int] = {}
    output_port_by_source: dict[tuple[str, int], Mapping[str, object]] = {}

    for record in relevant_records:
        candidate_id = str(record["candidate_id"])
        for input_port in tuple(record["input_ports"]):
            input_signature_by_port[(candidate_id, int(input_port["port_idx"]))] = int(input_port["signature_id"])
        for output_port in tuple(record["output_ports"]):
            for source_port_idx in output_port.get(
                "source_port_indices",
                (int(output_port["port_idx"]),),
            ):
                output_port_by_source[(candidate_id, int(source_port_idx))] = output_port

    reverse_edges: dict[str, set[str]] = defaultdict(set)
    produced_bindable_ports: dict[str, dict[int, set[str]]] = defaultdict(lambda: defaultdict(set))
    signature_bindable_ports: dict[int, set[tuple[str, int]]] = defaultdict(set)
    considered_pairs = 0
    dropped_pairs = 0

    for producer_candidate, producer_port, consumer_candidate, consumer_port in bindable_pairs:
        if (
            producer_candidate not in relevant_candidate_ids
            or consumer_candidate not in relevant_candidate_ids
        ):
            continue
        considered_pairs += 1
        retained_output_port = output_port_by_source.get((producer_candidate, producer_port))
        if retained_output_port is None:
            dropped_pairs += 1
            continue
        signature_id = input_signature_by_port.get((consumer_candidate, consumer_port))
        if signature_id is None:
            dropped_pairs += 1
            continue
        output_fsets = retained_output_port["port_values_fset"]
        assert isinstance(output_fsets, Mapping)
        if not _compressed_output_supports_signature(
            output_fsets,
            signature_id,
            signature_profiles_by_id,
            profile_accepts_by_id,
        ):
            dropped_pairs += 1
            continue
        retained_producer_port = int(retained_output_port["port_idx"])
        produced_bindable_ports[consumer_candidate][consumer_port].add(producer_candidate)
        reverse_edges[consumer_candidate].add(producer_candidate)
        signature_bindable_ports[signature_id].add((producer_candidate, retained_producer_port))

    return (
        reverse_edges,
        produced_bindable_ports,
        signature_bindable_ports,
        {
            "dynamic_bindable_pairs_considered": considered_pairs,
            "dynamic_bindable_candidate_edges_internal": sum(
                len(producer_candidates)
                for ports_by_source in produced_bindable_ports.values()
                for producer_candidates in ports_by_source.values()
            ),
            "dynamic_signature_bindable_ports_emitted": sum(
                len(producer_ports)
                for producer_ports in signature_bindable_ports.values()
            ),
            "dynamic_bindable_pairs_dropped": dropped_pairs,
        },
    )
def _dynamic_dim_values_cache_key(
    dim_values: Mapping[str, Iterable[str]],
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    return tuple(
        sorted(
            (str(dim), tuple(str(value) for value in values))
            for dim, values in dim_values.items()
        )
    )
