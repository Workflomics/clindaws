"""Optimized-candidate precomputation and optimization pipeline.

This is the translation-time optimization layer behind ``multi-shot
--optimized``. It expands tools into candidate records, prunes candidates that
cannot contribute to any satisfiable workflow, compresses equivalent output
choice values, and emits the indexed support surface consumed by the optimized
ASP encodings.
"""

from __future__ import annotations
from clindaws.translators.constraints import _collect_dynamic_forbidden_tool_ids, _collect_dynamic_backward_relevant_candidates, _collect_dynamic_selector_lower_bounds, _collect_dynamic_exact_prefix_lower_bound
from clindaws.translators.utils import _dedupe_stable, _normalize_dim_map
from clindaws.translators.ports import _group_port_values_by_dimension, _compress_dynamic_output_choice_values, _compress_duplicate_output_ports, _dynamic_output_matches_dynamic_input_fset, _workflow_input_matches_dynamic_port, _dynamic_port_expansion
from clindaws.translators.signatures import _assign_dynamic_signature_profiles
from clindaws.translators.candidates import _collect_compressed_dynamic_bindability_surface, _compute_dynamic_candidate_min_steps, _dynamic_dim_values_cache_key
from clindaws.translators.resolvers import _ExpansionResolver
from clindaws.translators.builder import _build_roots

from collections import defaultdict, deque
from collections.abc import Iterable, Mapping
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from time import perf_counter

from clindaws.core.models import SnakeConfig, ToolExpansionStat, ToolMode
from clindaws.core.ontology import Ontology


MAX_GLOBAL_MUST_RUN_REMOVAL_TESTS = 512


@dataclass(frozen=True)
class CompressedCandidateOptimizationResult:
    """Optimized-candidate surface ready for fact emission."""

    tool_stats: tuple[ToolExpansionStat, ...]
    relevant_records: tuple[dict[str, object], ...]
    relevant_tools: tuple[ToolMode, ...]
    query_goal_candidates: tuple[str, ...]
    query_goal_tools: tuple[str, ...]
    min_goal_distance_by_candidate: dict[str, int]
    min_step_by_candidate: dict[str, int]
    max_step_by_candidate: dict[str, int]
    goal_support_candidates_by_horizon: dict[int, tuple[str, ...]]
    goal_support_tools_by_horizon: dict[int, tuple[str, ...]]
    goal_support_inputs_by_horizon: dict[int, tuple[tuple[str, int], ...]]
    smart_expansion_rounds_by_horizon: dict[int, int]
    structurally_supportable_candidates_by_horizon: dict[int, tuple[str, ...]]
    structurally_unsupported_inputs_by_horizon: dict[int, tuple[tuple[str, int], ...]]
    input_support_rounds_by_horizon: dict[int, int]
    allowed_candidates_by_step: dict[int, tuple[str, ...]]
    allowed_tools_by_step: dict[int, tuple[str, ...]]
    check_relevant_output_categories_by_horizon: dict[int, tuple[tuple[str, int, str], ...]]
    check_required_profile_classes_by_horizon: dict[int, tuple[tuple[str, int, str, int], ...]]
    check_profile_class_members: dict[int, tuple[int, ...]]
    signature_profiles_by_id: dict[int, dict[str, tuple[int, tuple[str, ...]]]]
    profile_accepts_by_id: dict[int, tuple[str, ...]]
    goal_requirement_profiles_by_id: dict[int, tuple[tuple[int, str, int], ...]]
    signature_support_class_by_id: dict[int, int]
    support_class_bindable_ports: dict[int, tuple[tuple[str, int], ...]]
    association_class_by_input: dict[tuple[str, int], int]
    association_class_bindable_ports: dict[int, tuple[tuple[str, int], ...]]
    canonical_producers: Mapping[tuple[int, int], tuple[int, int]]
    cache_stats: dict[str, int]
    must_run_tools_global: tuple[str, ...]
    must_run_candidates_global: tuple[str, ...]
    must_run_tools_by_step: dict[int, tuple[str, ...]]
    must_run_candidates_by_step: dict[int, tuple[str, ...]]
    forced_associations_global: tuple[tuple[str, int, str, int], ...]
    forced_associations_by_step: dict[int, tuple[tuple[str, int, str, int], ...]]
    fixpoint_rounds: int
    structural_probe_horizons: tuple[int, ...]
    structural_horizon_skip_count: int
    earliest_solution_step: int
    phase_timings: dict[str, float]


def _compute_goal_support_by_horizon(
    *,
    config: SnakeConfig,
    candidate_records_by_id: Mapping[str, Mapping[str, object]],
    query_goal_candidates: Iterable[str],
    feasible_associations_by_input: Mapping[tuple[str, int], tuple[tuple[str, int], ...]],
    min_step_by_candidate: Mapping[str, int],
    max_step_by_candidate: Mapping[str, int],
    min_goal_distance_by_candidate: Mapping[str, int],
) -> tuple[
    dict[int, tuple[str, ...]],
    dict[int, tuple[str, ...]],
    dict[int, tuple[tuple[str, int], ...]],
]:
    """Build a conservative horizon-indexed backward support surface for goals.

    The surface is translator-side on purpose: the fast feasibility encoding can
    then avoid recursive backward expansion over large association-class joins
    and instead operate only on the horizon-specific candidate/input subset that
    can still contribute to a goal.
    """

    query_goal_candidate_set = set(query_goal_candidates)
    goal_support_candidates_by_horizon: dict[int, tuple[str, ...]] = {}
    goal_support_tools_by_horizon: dict[int, tuple[str, ...]] = {}
    goal_support_inputs_by_horizon: dict[int, tuple[tuple[str, int], ...]] = {}

    for horizon in range(config.solution_length_min, config.solution_length_max + 1):
        goal_support_candidates: set[str] = set()
        goal_support_inputs: set[tuple[str, int]] = set()
        frontier: deque[str] = deque()

        for candidate_id in sorted(query_goal_candidate_set):
            min_step = min_step_by_candidate.get(candidate_id)
            max_step = max_step_by_candidate.get(candidate_id)
            goal_distance = min_goal_distance_by_candidate.get(candidate_id)
            if min_step is None or max_step is None or goal_distance is None:
                continue
            if min_step > horizon or horizon > max_step:
                continue
            if min_step + goal_distance > horizon:
                continue
            goal_support_candidates.add(candidate_id)
            frontier.append(candidate_id)

        while frontier:
            consumer_candidate = frontier.popleft()
            consumer_min_step = min_step_by_candidate.get(consumer_candidate)
            consumer_max_step = max_step_by_candidate.get(consumer_candidate)
            if consumer_min_step is None or consumer_max_step is None:
                continue
            consumer_latest_step = min(horizon, consumer_max_step)
            for input_port in tuple(candidate_records_by_id[consumer_candidate]["input_ports"]):
                consumer_port = int(input_port["port_idx"])
                workflow_input_matches = tuple(input_port.get("workflow_input_matches", ()))
                if workflow_input_matches:
                    goal_support_inputs.add((consumer_candidate, consumer_port))
                    continue
                producer_ports = feasible_associations_by_input.get((consumer_candidate, consumer_port), ())
                support_found = False
                for producer_candidate, _producer_port in producer_ports:
                    producer_min_step = min_step_by_candidate.get(producer_candidate)
                    producer_goal_distance = min_goal_distance_by_candidate.get(producer_candidate)
                    if producer_min_step is None or producer_goal_distance is None:
                        continue
                    if producer_min_step + producer_goal_distance > horizon:
                        continue
                    if producer_min_step >= consumer_latest_step:
                        continue
                    support_found = True
                    if producer_candidate not in goal_support_candidates:
                        goal_support_candidates.add(producer_candidate)
                        frontier.append(producer_candidate)
                if support_found:
                    goal_support_inputs.add((consumer_candidate, consumer_port))

        if not goal_support_candidates:
            continue

        goal_support_candidates_by_horizon[horizon] = tuple(sorted(goal_support_candidates))
        goal_support_tools_by_horizon[horizon] = tuple(
            sorted(
                {
                    str(candidate_records_by_id[candidate_id]["tool"].mode_id)
                    for candidate_id in goal_support_candidates
                }
            )
        )
        goal_support_inputs_by_horizon[horizon] = tuple(sorted(goal_support_inputs))

    return (
        goal_support_candidates_by_horizon,
        goal_support_tools_by_horizon,
        goal_support_inputs_by_horizon,
    )


def _build_goal_support_for_horizon(
    *,
    horizon: int,
    candidate_records_by_id: Mapping[str, Mapping[str, object]],
    query_goal_candidates: Iterable[str],
    feasible_associations_by_input: Mapping[tuple[str, int], tuple[tuple[str, int], ...]],
    min_step_by_candidate: Mapping[str, int],
    max_step_by_candidate: Mapping[str, int],
    min_goal_distance_by_candidate: Mapping[str, int],
) -> tuple[set[str], set[tuple[str, int]]]:
    """Build horizon-local goal-support candidates and inputs for one horizon."""

    query_goal_candidate_set = set(query_goal_candidates)
    goal_support_candidates: set[str] = set()
    goal_support_inputs: set[tuple[str, int]] = set()
    frontier: deque[str] = deque()

    for candidate_id in sorted(query_goal_candidate_set):
        min_step = min_step_by_candidate.get(candidate_id)
        max_step = max_step_by_candidate.get(candidate_id)
        goal_distance = min_goal_distance_by_candidate.get(candidate_id)
        if min_step is None or max_step is None or goal_distance is None:
            continue
        if min_step > horizon or horizon > max_step:
            continue
        if min_step + goal_distance > horizon:
            continue
        goal_support_candidates.add(candidate_id)
        frontier.append(candidate_id)

    while frontier:
        consumer_candidate = frontier.popleft()
        consumer_max_step = max_step_by_candidate.get(consumer_candidate)
        if consumer_max_step is None:
            continue
        consumer_latest_step = min(horizon, consumer_max_step)
        for input_port in tuple(candidate_records_by_id[consumer_candidate]["input_ports"]):
            consumer_port = int(input_port["port_idx"])
            workflow_input_matches = tuple(input_port.get("workflow_input_matches", ()))
            if workflow_input_matches:
                goal_support_inputs.add((consumer_candidate, consumer_port))
                continue
            producer_ports = feasible_associations_by_input.get((consumer_candidate, consumer_port), ())
            support_found = False
            for producer_candidate, _producer_port in producer_ports:
                producer_min_step = min_step_by_candidate.get(producer_candidate)
                producer_goal_distance = min_goal_distance_by_candidate.get(producer_candidate)
                if producer_min_step is None or producer_goal_distance is None:
                    continue
                if producer_min_step + producer_goal_distance > horizon:
                    continue
                if producer_min_step >= consumer_latest_step:
                    continue
                support_found = True
                if producer_candidate not in goal_support_candidates:
                    goal_support_candidates.add(producer_candidate)
                    frontier.append(producer_candidate)
            if support_found:
                goal_support_inputs.add((consumer_candidate, consumer_port))

    return goal_support_candidates, goal_support_inputs


def _compute_smart_expansion_by_horizon(
    *,
    config: SnakeConfig,
    candidate_records_by_id: Mapping[str, Mapping[str, object]],
    query_goal_candidates: Iterable[str],
    feasible_associations_by_input: Mapping[tuple[str, int], tuple[tuple[str, int], ...]],
    min_step_by_candidate: Mapping[str, int],
    max_step_by_candidate: Mapping[str, int],
    min_goal_distance_by_candidate: Mapping[str, int],
) -> tuple[
    dict[int, tuple[str, ...]],
    dict[int, tuple[str, ...]],
    dict[int, tuple[tuple[str, int], ...]],
    dict[int, int],
]:
    """Compute fixed-point smart-expansion surfaces for each horizon.

    This keeps only the horizon-local candidate/input closure in Python. The
    feasibility encoding applies temporal and reachability filtering over the
    global association classes at solve time.
    """

    goal_support_candidates_by_horizon: dict[int, tuple[str, ...]] = {}
    goal_support_tools_by_horizon: dict[int, tuple[str, ...]] = {}
    goal_support_inputs_by_horizon: dict[int, tuple[tuple[str, int], ...]] = {}
    smart_expansion_rounds_by_horizon: dict[int, int] = {}

    for horizon in range(config.solution_length_min, config.solution_length_max + 1):
        goal_support_candidates, goal_support_inputs = _build_goal_support_for_horizon(
            horizon=horizon,
            candidate_records_by_id=candidate_records_by_id,
            query_goal_candidates=query_goal_candidates,
            feasible_associations_by_input=feasible_associations_by_input,
            min_step_by_candidate=min_step_by_candidate,
            max_step_by_candidate=max_step_by_candidate,
            min_goal_distance_by_candidate=min_goal_distance_by_candidate,
        )
        smart_expansion_rounds_by_horizon[horizon] = 1
        if not goal_support_candidates:
            continue

        goal_support_candidates_by_horizon[horizon] = tuple(sorted(goal_support_candidates))
        goal_support_tools_by_horizon[horizon] = tuple(
            sorted(
                {
                    str(candidate_records_by_id[candidate_id]["tool"].mode_id)
                    for candidate_id in goal_support_candidates
                }
            )
        )
        goal_support_inputs_by_horizon[horizon] = tuple(sorted(goal_support_inputs))

    return (
        goal_support_candidates_by_horizon,
        goal_support_tools_by_horizon,
        goal_support_inputs_by_horizon,
        smart_expansion_rounds_by_horizon,
    )


def _compute_check_surface_by_horizon(
    *,
    goal_support_candidates_by_horizon: Mapping[int, tuple[str, ...]],
    goal_support_inputs_by_horizon: Mapping[int, tuple[tuple[str, int], ...]],
    candidate_records_by_id: Mapping[str, Mapping[str, object]],
    feasible_associations_by_input: Mapping[tuple[str, int], tuple[tuple[str, int], ...]],
    signature_profiles_by_id: Mapping[int, Mapping[str, tuple[int, tuple[str, ...]]]],
    query_goal_candidates: Iterable[str],
    goal_requirement_profiles_by_id: Mapping[int, tuple[tuple[int, str, int], ...]],
    goal_fsets: tuple[Mapping[str, frozenset[str]], ...],
) -> tuple[
    dict[int, tuple[tuple[str, int, str], ...]],
    dict[int, tuple[tuple[str, int, str, int], ...]],
    dict[int, tuple[int, ...]],
]:
    """Precompute the exact output/category/profile surface consumed by check.

    This keeps the exact semantics in ASP, but removes the expensive rebuild of
    all producer/category/profile combinations from raw bind occurrences.
    """

    query_goal_candidate_set = set(query_goal_candidates)
    input_profiles_by_port: dict[tuple[str, int], tuple[tuple[str, int], ...]] = {}
    goal_categories_by_output: dict[tuple[str, int], tuple[str, ...]] = {}

    for candidate_id, record in candidate_records_by_id.items():
        for input_port in tuple(record["input_ports"]):
            port_idx = int(input_port["port_idx"])
            signature_id = int(input_port["signature_id"])
            input_profiles_by_port[(candidate_id, port_idx)] = tuple(
                sorted(
                    (str(category), int(profile_id))
                    for category, (profile_id, _values) in
                    signature_profiles_by_id.get(signature_id, {}).items()
                )
            )

        if candidate_id not in query_goal_candidate_set:
            continue

        for output_port in tuple(record["output_ports"]):
            port_idx = int(output_port["port_idx"])
            port_fset = output_port["port_values_fset"]
            categories: set[str] = set()
            for goal_id, goal_fset in enumerate(goal_fsets):
                if not _dynamic_output_matches_dynamic_input_fset(port_fset, goal_fset):
                    continue
                for _requirement_id, category, _profile_id in goal_requirement_profiles_by_id.get(goal_id, ()):
                    categories.add(str(category))
            if categories:
                goal_categories_by_output[(candidate_id, port_idx)] = tuple(sorted(categories))

    profile_class_id_by_members: dict[tuple[int, ...], int] = {}
    profile_class_members: dict[int, tuple[int, ...]] = {}
    next_profile_class_id = 0
    relevant_output_categories_by_horizon: dict[int, tuple[tuple[str, int, str], ...]] = {}
    required_profile_classes_by_horizon: dict[int, tuple[tuple[str, int, str, int], ...]] = {}

    for horizon, goal_support_candidates in sorted(goal_support_candidates_by_horizon.items()):
        active_candidates = set(goal_support_candidates)
        relevant_output_categories: set[tuple[str, int, str]] = set()
        profile_ids_by_output_category: dict[tuple[str, int, str], set[int]] = defaultdict(set)

        for consumer_candidate, consumer_port in goal_support_inputs_by_horizon.get(horizon, ()):
            for producer_candidate, producer_port in feasible_associations_by_input.get((consumer_candidate, consumer_port), ()):
                if producer_candidate not in active_candidates:
                    continue
                for category, profile_id in input_profiles_by_port.get((consumer_candidate, consumer_port), ()):
                    relevant_output_categories.add((producer_candidate, producer_port, category))
                    profile_ids_by_output_category[(producer_candidate, producer_port, category)].add(profile_id)

        for candidate_id in sorted(active_candidates & query_goal_candidate_set):
            for (goal_candidate_id, port_idx), categories in goal_categories_by_output.items():
                if goal_candidate_id != candidate_id:
                    continue
                for category in categories:
                    relevant_output_categories.add((candidate_id, port_idx, category))

        relevant_output_categories_by_horizon[horizon] = tuple(sorted(relevant_output_categories))

        horizon_profile_classes: set[tuple[str, int, str, int]] = set()
        for (producer_candidate, producer_port, category), profile_ids in sorted(profile_ids_by_output_category.items()):
            members = tuple(sorted(profile_ids))
            if not members:
                continue
            profile_class_id = profile_class_id_by_members.get(members)
            if profile_class_id is None:
                profile_class_id = next_profile_class_id
                next_profile_class_id += 1
                profile_class_id_by_members[members] = profile_class_id
                profile_class_members[profile_class_id] = members
            horizon_profile_classes.add((producer_candidate, producer_port, category, profile_class_id))
        required_profile_classes_by_horizon[horizon] = tuple(sorted(horizon_profile_classes))

    return (
        relevant_output_categories_by_horizon,
        required_profile_classes_by_horizon,
        dict(sorted(profile_class_members.items())),
    )


def _compute_structural_input_support_by_horizon(
    *,
    goal_support_candidates_by_horizon: Mapping[int, tuple[str, ...]],
    goal_support_inputs_by_horizon: Mapping[int, tuple[tuple[str, int], ...]],
    candidate_records_by_id: Mapping[str, Mapping[str, object]],
    association_class_by_input: Mapping[tuple[str, int], int],
    association_class_bindable_ports: Mapping[int, tuple[tuple[str, int], ...]],
    min_step_by_candidate: Mapping[str, int],
    max_step_by_candidate: Mapping[str, int],
) -> tuple[
    dict[int, tuple[str, ...]],
    dict[int, tuple[tuple[str, int], ...]],
    dict[int, int],
]:
    """Compute horizon-local structural supportability for goal-support candidates."""

    producer_candidates_by_input: dict[tuple[str, int], tuple[str, ...]] = {}
    for input_key, class_id in sorted(association_class_by_input.items()):
        producer_candidates_by_input[input_key] = tuple(
            sorted(
                {
                    producer_candidate
                    for producer_candidate, _producer_port in association_class_bindable_ports.get(class_id, ())
                }
            )
        )

    input_records_by_key: dict[tuple[str, int], Mapping[str, object]] = {}
    for candidate_id, record in candidate_records_by_id.items():
        for input_port in tuple(record["input_ports"]):
            input_records_by_key[(candidate_id, int(input_port["port_idx"]))] = input_port

    structurally_supportable_candidates_by_horizon: dict[int, tuple[str, ...]] = {}
    structurally_unsupported_inputs_by_horizon: dict[int, tuple[tuple[str, int], ...]] = {}
    input_support_rounds_by_horizon: dict[int, int] = {}

    for horizon, candidate_ids in sorted(goal_support_candidates_by_horizon.items()):
        candidate_set = set(candidate_ids)
        goal_inputs = set(goal_support_inputs_by_horizon.get(horizon, ()))
        candidate_required_inputs: dict[str, list[tuple[str, int]]] = {
            candidate_id: []
            for candidate_id in candidate_ids
        }
        producer_options_by_input: dict[tuple[str, int], tuple[str, ...]] = {}

        for candidate_id, port_idx in sorted(goal_inputs):
            input_record = input_records_by_key.get((candidate_id, port_idx))
            if input_record is None:
                continue
            if tuple(input_record.get("workflow_input_matches", ())):
                continue
            consumer_latest_step = min(
                horizon,
                max_step_by_candidate.get(candidate_id, horizon),
            )
            filtered_producers = tuple(
                sorted(
                    producer_candidate
                    for producer_candidate in producer_candidates_by_input.get((candidate_id, port_idx), ())
                    if producer_candidate in candidate_set
                    and min_step_by_candidate.get(producer_candidate) is not None
                    and min_step_by_candidate[producer_candidate] < consumer_latest_step
                )
            )
            candidate_required_inputs[candidate_id].append((candidate_id, port_idx))
            producer_options_by_input[(candidate_id, port_idx)] = filtered_producers

        supportable_candidates = {
            candidate_id
            for candidate_id, required_inputs in candidate_required_inputs.items()
            if not required_inputs
        }
        rounds = 0
        changed = True
        while changed:
            changed = False
            rounds += 1
            for candidate_id in candidate_ids:
                if candidate_id in supportable_candidates:
                    continue
                required_inputs = candidate_required_inputs.get(candidate_id, [])
                if all(
                    any(producer_candidate in supportable_candidates for producer_candidate in producer_options_by_input.get(input_key, ()))
                    for input_key in required_inputs
                ):
                    supportable_candidates.add(candidate_id)
                    changed = True

        unsupported_inputs = tuple(
            sorted(
                input_key
                for input_key, producer_candidates in producer_options_by_input.items()
                if not any(producer_candidate in supportable_candidates for producer_candidate in producer_candidates)
            )
        )
        structurally_supportable_candidates_by_horizon[horizon] = tuple(sorted(supportable_candidates))
        structurally_unsupported_inputs_by_horizon[horizon] = unsupported_inputs
        input_support_rounds_by_horizon[horizon] = max(rounds, 1)

    return (
        structurally_supportable_candidates_by_horizon,
        structurally_unsupported_inputs_by_horizon,
        input_support_rounds_by_horizon,
    )


def _reverse_edges_from_produced_bindable_ports(
    produced_bindable_ports: Mapping[str, Mapping[int, set[str]]],
) -> dict[str, set[str]]:
    reverse_edges: dict[str, set[str]] = defaultdict(set)
    for consumer_candidate, by_port in produced_bindable_ports.items():
        for producers in by_port.values():
            for producer_candidate in producers:
                reverse_edges[consumer_candidate].add(producer_candidate)
    return reverse_edges


def _close_relevant_candidates(
    *,
    candidate_records_by_id: Mapping[str, Mapping[str, object]],
    workflow_bindable_ports: Mapping[str, set[int]],
    produced_bindable_ports: Mapping[str, Mapping[int, set[str]]],
    goal_candidates: Iterable[str],
    active_candidates: Iterable[str] | None = None,
) -> set[str]:
    """Compute a fixed-point of forward feasible and backward goal-relevant candidates."""

    allowed = set(candidate_records_by_id) if active_candidates is None else set(active_candidates)
    reverse_edges = _reverse_edges_from_produced_bindable_ports(produced_bindable_ports)

    stable = set(allowed)
    while True:
        forward_reachable: set[str] = set()
        changed = True
        while changed:
            changed = False
            for candidate_id in sorted(stable):
                record = candidate_records_by_id[candidate_id]
                input_ports = tuple(record["input_ports"])
                if all(
                    int(port["port_idx"]) in workflow_bindable_ports.get(candidate_id, set())
                    or any(
                        producer_candidate in forward_reachable
                        for producer_candidate in produced_bindable_ports.get(candidate_id, {}).get(int(port["port_idx"]), set())
                    )
                    for port in input_ports
                ):
                    if candidate_id not in forward_reachable:
                        forward_reachable.add(candidate_id)
                        changed = True

        backward_relevant: set[str] = set()
        frontier = deque(
            sorted(
                candidate_id
                for candidate_id in goal_candidates
                if candidate_id in forward_reachable
            )
        )
        while frontier:
            candidate_id = frontier.popleft()
            if candidate_id in backward_relevant:
                continue
            backward_relevant.add(candidate_id)
            for producer_candidate in sorted(reverse_edges.get(candidate_id, set())):
                if producer_candidate in forward_reachable and producer_candidate not in backward_relevant:
                    frontier.append(producer_candidate)

        new_stable = forward_reachable & backward_relevant
        if new_stable == stable:
            return new_stable
        stable = new_stable


def _filter_bindability_by_temporal_overlap(
    *,
    produced_bindable_ports: Mapping[str, Mapping[int, set[str]]],
    min_step_by_candidate: Mapping[str, int],
    max_step_by_candidate: Mapping[str, int],
) -> tuple[dict[str, dict[int, set[str]]], int]:
    """Drop producer-consumer associations that can never respect temporal order."""

    filtered: dict[str, dict[int, set[str]]] = defaultdict(lambda: defaultdict(set))
    dropped = 0
    for consumer_candidate, by_port in produced_bindable_ports.items():
        consumer_max_step = max_step_by_candidate.get(consumer_candidate)
        if consumer_max_step is None:
            continue
        for consumer_port, producer_candidates in by_port.items():
            for producer_candidate in producer_candidates:
                producer_min_step = min_step_by_candidate.get(producer_candidate)
                if producer_min_step is None or producer_min_step >= consumer_max_step:
                    dropped += 1
                    continue
                filtered[consumer_candidate][consumer_port].add(producer_candidate)
    return (
        {
            consumer_candidate: {
                consumer_port: set(producer_candidates)
                for consumer_port, producer_candidates in by_port.items()
                if producer_candidates
            }
            for consumer_candidate, by_port in filtered.items()
            if any(by_port.values())
        },
        dropped,
    )


def _filter_signature_bindable_ports(
    signature_bindable_ports: Mapping[int, set[tuple[str, int]]],
    *,
    active_candidates: Iterable[str],
) -> dict[int, set[tuple[str, int]]]:
    """Filter signature support classes to the surviving candidate universe."""

    active = set(active_candidates)
    return {
        signature_id: {
            (producer_candidate, producer_port)
            for producer_candidate, producer_port in producer_ports
            if producer_candidate in active
        }
        for signature_id, producer_ports in signature_bindable_ports.items()
    }


def _candidate_set_supports_any_goal(
    *,
    candidate_records_by_id: Mapping[str, Mapping[str, object]],
    workflow_bindable_ports: Mapping[str, set[int]],
    produced_bindable_ports: Mapping[str, Mapping[int, set[str]]],
    query_goal_candidates: Iterable[str],
    active_candidates: Iterable[str],
) -> bool:
    closed = _close_relevant_candidates(
        candidate_records_by_id=candidate_records_by_id,
        workflow_bindable_ports=workflow_bindable_ports,
        produced_bindable_ports=produced_bindable_ports,
        goal_candidates=query_goal_candidates,
        active_candidates=active_candidates,
    )
    return any(candidate_id in closed for candidate_id in query_goal_candidates)


def _compute_global_must_run_candidates(
    *,
    relevant_records: Iterable[Mapping[str, object]],
    workflow_bindable_ports: Mapping[str, set[int]],
    produced_bindable_ports: Mapping[str, Mapping[int, set[str]]],
    query_goal_candidates: Iterable[str],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Derive conservative global must-run candidates/tools by removal testing."""

    candidate_records_by_id = {
        str(record["candidate_id"]): record
        for record in relevant_records
    }
    active_candidates = tuple(sorted(candidate_records_by_id))
    if len(active_candidates) > MAX_GLOBAL_MUST_RUN_REMOVAL_TESTS:
        return (), ()
    must_run_candidates: list[str] = []

    for candidate_id in active_candidates:
        remaining_candidates = tuple(other for other in active_candidates if other != candidate_id)
        if not _candidate_set_supports_any_goal(
            candidate_records_by_id=candidate_records_by_id,
            workflow_bindable_ports=workflow_bindable_ports,
            produced_bindable_ports=produced_bindable_ports,
            query_goal_candidates=query_goal_candidates,
            active_candidates=remaining_candidates,
        ):
            must_run_candidates.append(candidate_id)

    must_run_tools = tuple(
        sorted(
            {
                str(candidate_records_by_id[candidate_id]["tool"].mode_id)
                for candidate_id in must_run_candidates
            }
        )
    )
    return tuple(sorted(must_run_candidates)), must_run_tools


def _factor_signature_support_classes(
    signature_bindable_ports: Mapping[int, set[tuple[str, int]]],
) -> tuple[
    dict[int, int],
    dict[int, tuple[tuple[str, int], ...]],
    dict[str, int],
]:
    """Compress identical signature support sets into shared support classes."""

    support_class_by_key: dict[tuple[tuple[str, int], ...], int] = {}
    signature_support_class_by_id: dict[int, int] = {}
    support_class_bindable_ports: dict[int, tuple[tuple[str, int], ...]] = {}

    for signature_id, producer_ports in sorted(signature_bindable_ports.items()):
        support_key = tuple(sorted((producer_candidate, int(producer_port)) for producer_candidate, producer_port in producer_ports))
        support_class_id = support_class_by_key.setdefault(support_key, len(support_class_by_key))
        signature_support_class_by_id[signature_id] = support_class_id
        support_class_bindable_ports.setdefault(support_class_id, support_key)

    raw_signature_supports = len(signature_bindable_ports)
    factored_supports = len(support_class_bindable_ports)
    return (
        signature_support_class_by_id,
        support_class_bindable_ports,
        {
            "dynamic_signature_supports_raw": raw_signature_supports,
            "dynamic_signature_support_classes": factored_supports,
            "dynamic_signature_supports_collapsed": raw_signature_supports - factored_supports,
            "dynamic_support_class_bindable_ports_emitted": sum(
                len(producer_ports)
                for producer_ports in support_class_bindable_ports.values()
            ),
        },
    )


def _factor_input_association_classes(
    *,
    relevant_records: Iterable[Mapping[str, object]],
    workflow_bindable_ports: Mapping[str, set[int]],
    signature_bindable_ports: Mapping[int, set[tuple[str, int]]],
    min_step_by_candidate: Mapping[str, int],
    max_step_by_candidate: Mapping[str, int],
) -> tuple[
    dict[tuple[str, int], int],
    dict[int, tuple[tuple[str, int], ...]],
    dict[tuple[str, int], tuple[tuple[str, int], ...]],
    dict[str, int],
]:
    """Factor feasible producer-port options per concrete consumer input port."""

    association_class_by_key: dict[tuple[tuple[str, int], ...], int] = {}
    association_class_by_input: dict[tuple[str, int], int] = {}
    association_class_bindable_ports: dict[int, tuple[tuple[str, int], ...]] = {}
    feasible_associations_by_input: dict[tuple[str, int], tuple[tuple[str, int], ...]] = {}
    raw_port_options = 0

    for record in relevant_records:
        candidate_id = str(record["candidate_id"])
        consumer_max_step = max_step_by_candidate.get(candidate_id)
        if consumer_max_step is None:
            continue
        for input_port in tuple(record["input_ports"]):
            port_idx = int(input_port["port_idx"])
            signature_id = int(input_port["signature_id"])
            feasible_producer_ports = tuple(
                sorted(
                    (producer_candidate, int(producer_port))
                    for producer_candidate, producer_port in signature_bindable_ports.get(signature_id, set())
                    if producer_candidate != candidate_id
                    and producer_candidate in min_step_by_candidate
                    and min_step_by_candidate[producer_candidate] < consumer_max_step
                )
            )
            feasible_associations_by_input[(candidate_id, port_idx)] = feasible_producer_ports
            raw_port_options += len(feasible_producer_ports)
            if not feasible_producer_ports:
                continue
            class_id = association_class_by_key.setdefault(
                feasible_producer_ports,
                len(association_class_by_key),
            )
            association_class_by_input[(candidate_id, port_idx)] = class_id
            association_class_bindable_ports.setdefault(class_id, feasible_producer_ports)

    return (
        association_class_by_input,
        association_class_bindable_ports,
        feasible_associations_by_input,
        {
            "dynamic_association_classes": len(association_class_bindable_ports),
            "dynamic_association_port_options": raw_port_options,
            "dynamic_association_port_options_collapsed": max(
                0,
                raw_port_options - len(association_class_bindable_ports),
            ),
        },
    )


def _propagate_forced_associations(
    *,
    candidate_records_by_id: Mapping[str, Mapping[str, object]],
    workflow_bindable_ports: Mapping[str, set[int]],
    feasible_associations_by_input: Mapping[tuple[str, int], tuple[tuple[str, int], ...]],
    seed_required_candidates: Iterable[str],
) -> tuple[tuple[str, ...], tuple[tuple[str, int, str, int], ...], int]:
    """Propagate unique producer choices from required candidates until stable."""

    required_candidates = set(seed_required_candidates)
    forced_associations: dict[tuple[str, int], tuple[str, int]] = {}
    fixpoint_rounds = 0

    while True:
        changed = False
        fixpoint_rounds += 1
        for candidate_id in sorted(required_candidates):
            record = candidate_records_by_id.get(candidate_id)
            if record is None:
                continue
            for input_port in tuple(record["input_ports"]):
                port_idx = int(input_port["port_idx"])
                if port_idx in workflow_bindable_ports.get(candidate_id, set()):
                    continue
                feasible = feasible_associations_by_input.get((candidate_id, port_idx), ())
                if len(feasible) != 1:
                    continue
                producer_candidate, producer_port = feasible[0]
                key = (candidate_id, port_idx)
                if forced_associations.get(key) == (producer_candidate, producer_port):
                    continue
                forced_associations[key] = (producer_candidate, producer_port)
                if producer_candidate not in required_candidates:
                    required_candidates.add(producer_candidate)
                changed = True
        if not changed:
            break

    return (
        tuple(sorted(required_candidates)),
        tuple(
            sorted(
                (candidate_id, port_idx, producer_candidate, producer_port)
                for (candidate_id, port_idx), (producer_candidate, producer_port) in forced_associations.items()
            )
        ),
        fixpoint_rounds,
    )


def _expand_single_tool(
    args: tuple,
) -> tuple[dict[str, object], ToolExpansionStat]:
    """Expand one tool into its candidate record and stats.

    Accepts a single tuple argument so it can be used with executor.map.
    Each call creates its own _ExpansionResolver so workers are independent.
    """
    tool, candidate_id, config, ontology, roots = args
    resolver = _ExpansionResolver(ontology, roots, "python")

    input_port_value_counts: list[int] = []
    output_port_value_counts: list[int] = []
    input_variant_count = 1
    output_variant_count = 1
    dynamic_input_value_count = 0
    dynamic_output_value_count = 0
    input_ports: list[dict[str, object]] = []
    output_ports: list[dict[str, object]] = []

    for port_idx, port in enumerate(tool.inputs):
        port_values, port_variant_count = _dynamic_port_expansion(
            resolver,
            _normalize_dim_map(port.dimensions),
            expand_outputs=False,
        )
        port_values_by_dimension = _group_port_values_by_dimension(port_values)
        workflow_input_matches = [
            wf_index
            for wf_index, workflow_input in enumerate(config.inputs)
            if _workflow_input_matches_dynamic_port(ontology, workflow_input, port_values_by_dimension)
        ]
        input_ports.append(
            {
                "port_idx": port_idx,
                "port_values": port_values,
                "port_values_by_dimension": port_values_by_dimension,
                "port_values_fset": {dim: frozenset(vals) for dim, vals in port_values_by_dimension.items()},
                "workflow_input_matches": workflow_input_matches,
            }
        )
        emitted_count = len(port_values)
        input_port_value_counts.append(emitted_count)
        dynamic_input_value_count += emitted_count
        input_variant_count *= port_variant_count

    for port_idx, port in enumerate(tool.outputs):
        declared_dims = _normalize_dim_map(port.dimensions)
        port_values, port_variant_count = _dynamic_port_expansion(
            resolver,
            declared_dims,
            expand_outputs=True,
        )
        port_values_by_dimension = _group_port_values_by_dimension(port_values)
        output_ports.append(
            {
                "port_idx": port_idx,
                "declared_dims": declared_dims,
                "port_values_by_dimension": port_values_by_dimension,
                "port_values_fset": {dim: frozenset(vals) for dim, vals in port_values_by_dimension.items()},
            }
        )
        emitted_count = sum(len(values) for values in port_values_by_dimension.values())
        output_port_value_counts.append(emitted_count)
        dynamic_output_value_count += emitted_count
        output_variant_count *= port_variant_count

    stat = ToolExpansionStat(
        tool_id=tool.mode_id,
        tool_label=tool.label,
        input_ports=len(tool.inputs),
        output_ports=len(tool.outputs),
        input_variant_count=input_variant_count,
        output_variant_count=output_variant_count,
        dynamic_input_value_count=dynamic_input_value_count,
        dynamic_output_value_count=dynamic_output_value_count,
        dynamic_input_port_value_counts=tuple(input_port_value_counts),
        dynamic_output_port_value_counts=tuple(output_port_value_counts),
        dynamic_cross_product_estimate=input_variant_count * output_variant_count,
    )
    record: dict[str, object] = {
        "tool": tool,
        "candidate_id": candidate_id,
        "input_ports": tuple(input_ports),
        "output_ports": _compress_duplicate_output_ports(tuple(output_ports)),
    }
    return record, stat


def optimize_compressed_candidates(
    config: SnakeConfig,
    ontology: Ontology,
    tools: tuple[ToolMode, ...],
    *,
    max_workers: int = 1,
) -> CompressedCandidateOptimizationResult:
    """Precompute the optimized-candidate surface before fact emission.

    The resulting record is intentionally close to the optimized ASP schema:
    candidate-step eligibility, signature support classes, goal distances, and
    retained output choices are all materialized here so the encoding can stay
    focused on workflow search rather than rediscovering that structure.
    """

    roots = _build_roots(config, ontology)
    resolver = _ExpansionResolver(ontology, roots, "python")

    forbidden_tool_ids = _collect_dynamic_forbidden_tool_ids(config, ontology, tools)
    candidate_source_tools = tuple(
        tool
        for tool in tools
        if tool.mode_id not in forbidden_tool_ids
    )

    # Pre-assign candidate IDs before optional parallel expansion so candidate
    # order remains deterministic regardless of worker count.
    offset_tracker: dict[str, int] = defaultdict(int)
    work_items: list[tuple] = []
    for tool in candidate_source_tools:
        candidate_id = f"{tool.mode_id}_lc{offset_tracker[tool.mode_id]}"
        offset_tracker[tool.mode_id] += 1
        work_items.append((tool, candidate_id, config, ontology, roots))

    if max_workers == 1:
        results = [_expand_single_tool(item) for item in work_items]
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(_expand_single_tool, work_items))

    candidate_records: list[dict[str, object]] = [r[0] for r in results]
    tool_stats: list[ToolExpansionStat] = [r[1] for r in results]

    goal_port_values: list[dict[str, tuple[str, ...]]] = []
    goal_requirement_accepts_by_id: dict[int, tuple[tuple[int, str, tuple[str, ...]], ...]] = {}
    for goal_item in config.outputs:
        goal_index = len(goal_port_values)
        goal_dims: dict[str, tuple[str, ...]] = {}
        goal_requirement_accepts: list[tuple[int, str, tuple[str, ...]]] = []
        requirement_index = 0
        for dim, values in sorted(goal_item.items()):
            expanded_values: list[str] = []
            for value in values:
                expanded_requirement = _dedupe_stable(
                    resolver.expanded_values(
                        dim,
                        value,
                        expand_outputs=True,
                    )
                )
                goal_requirement_accepts.append(
                    (requirement_index, str(dim), expanded_requirement)
                )
                requirement_index += 1
                expanded_values.extend(expanded_requirement)
            goal_dims[dim] = _dedupe_stable(expanded_values)
        goal_port_values.append(goal_dims)
        goal_requirement_accepts_by_id[goal_index] = tuple(goal_requirement_accepts)
    goal_port_values_tuple = tuple(goal_port_values)
    goal_fsets: tuple[dict[str, frozenset[str]], ...] = tuple(
        {dim: frozenset(vals) for dim, vals in g.items()} for g in goal_port_values
    )

    t0 = perf_counter()

    bindable_pairs: set[tuple[str, int, str, int]] = set()
    candidate_count_before_pruning = len(candidate_records)
    reverse_edges: dict[str, set[str]] = defaultdict(set)
    candidate_records_by_id = {
        str(record["candidate_id"]): record
        for record in candidate_records
    }
    workflow_inputs = tuple(
        {
            str(dim): tuple(str(value) for value in values)
            for dim, values in item.items()
        }
        for item in config.inputs
    )
    direct_goal_candidates: set[str] = set()

    for record in candidate_records:
        candidate_id = str(record["candidate_id"])
        for output_port in tuple(record["output_ports"]):
            output_fset = output_port["port_values_fset"]
            if any(_dynamic_output_matches_dynamic_input_fset(output_fset, gf) for gf in goal_fsets):
                direct_goal_candidates.add(candidate_id)

    t1 = perf_counter()

    for producer_record in candidate_records:
        producer_candidate = str(producer_record["candidate_id"])
        for output_port in tuple(producer_record["output_ports"]):
            producer_port = int(output_port["port_idx"])
            output_fset = output_port["port_values_fset"]
            for consumer_record in candidate_records:
                consumer_candidate = str(consumer_record["candidate_id"])
                for input_port in tuple(consumer_record["input_ports"]):
                    consumer_port = int(input_port["port_idx"])
                    if _dynamic_output_matches_dynamic_input_fset(output_fset, input_port["port_values_fset"]):
                        bindable_pairs.add(
                            (producer_candidate, producer_port, consumer_candidate, consumer_port)
                        )
                        reverse_edges[consumer_candidate].add(producer_candidate)

    t2 = perf_counter()

    workflow_bindable_ports: dict[str, set[int]] = defaultdict(set)
    produced_bindable_ports: dict[str, dict[int, set[str]]] = defaultdict(lambda: defaultdict(set))
    for record in candidate_records:
        candidate_id = str(record["candidate_id"])
        for input_port in tuple(record["input_ports"]):
            port_idx = int(input_port["port_idx"])
            if any(
                _workflow_input_matches_dynamic_port(
                    ontology,
                    workflow_input,
                    input_port["port_values_by_dimension"],
                )
                for workflow_input in workflow_inputs
            ):
                workflow_bindable_ports[candidate_id].add(port_idx)

    for producer_candidate, _, consumer_candidate, consumer_port in bindable_pairs:
        produced_bindable_ports[consumer_candidate][consumer_port].add(producer_candidate)

    relevant_candidates: set[str] = set()
    frontier: deque[str] = deque()
    for record in candidate_records:
        candidate_id = str(record["candidate_id"])
        input_ports = tuple(record["input_ports"])
        if all(int(port["port_idx"]) in workflow_bindable_ports[candidate_id] for port in input_ports):
            relevant_candidates.add(candidate_id)
            frontier.append(candidate_id)

    while frontier:
        producer_candidate = frontier.popleft()
        for consumer_candidate, ports_by_source in produced_bindable_ports.items():
            if consumer_candidate in relevant_candidates:
                continue
            consumer_record = candidate_records_by_id[consumer_candidate]
            input_ports = tuple(consumer_record["input_ports"])
            if all(
                int(port["port_idx"]) in workflow_bindable_ports[consumer_candidate]
                or any(
                    source_candidate in relevant_candidates
                    for source_candidate in produced_bindable_ports[consumer_candidate].get(int(port["port_idx"]), set())
                )
                for port in input_ports
            ):
                relevant_candidates.add(consumer_candidate)
                frontier.append(consumer_candidate)

    t3 = perf_counter()

    min_anchor_distance_by_candidate: dict[str, int] = {}
    backward_relevant_candidates, min_anchor_distance_by_candidate = _collect_dynamic_backward_relevant_candidates(
        config,
        ontology,
        tools,
        candidate_records=candidate_records,
        reverse_edges=reverse_edges,
        direct_goal_candidates=direct_goal_candidates,
    )
    if config.use_all_generated_data == "ALL":
        relevant_candidates &= backward_relevant_candidates
    else:
        # Positive temporal/tool constraints can still require candidates that do
        # not sit on the strict goal-only backward closure. Keep that surface
        # available so exact solve is not starved before check runs.
        relevant_candidates |= backward_relevant_candidates

    relevant_records = [
        record
        for record in candidate_records
        if str(record["candidate_id"]) in relevant_candidates
    ]
    output_compression_cache: dict[
        tuple[tuple[tuple[str, tuple[str, ...]], ...], bool],
        dict[str, tuple[str, ...]],
    ] = {}
    relevant_input_ports = tuple(
        input_port
        for record in relevant_records
        for input_port in tuple(record["input_ports"])
    )
    unique_input_signatures: dict[int, tuple[Mapping[str, object], Mapping[str, frozenset[str]]]] = {}
    signature_profiles_by_id, _profile_values_by_id, profile_accepts_by_id, dynamic_schema_stats = _assign_dynamic_signature_profiles(
        ontology,
        roots,
        relevant_input_ports,
    )
    for input_port in relevant_input_ports:
        signature_id = int(input_port["signature_id"])
        unique_input_signatures.setdefault(
            signature_id,
            (
                input_port.get("signature_requirements", input_port["port_values_by_dimension"]),
                input_port["port_values_fset"],
            ),
        )
    bindable_input_signatures = tuple(
        (signature_id, signature_requirements, signature_fsets)
        for signature_id, (signature_requirements, signature_fsets) in sorted(unique_input_signatures.items())
    )
    goal_requirement_profiles_by_id: dict[int, tuple[tuple[int, str, int], ...]] = {}
    profile_id_by_accepts = {
        tuple(values): profile_id
        for profile_id, values in profile_accepts_by_id.items()
    }
    next_profile_id = max(profile_accepts_by_id, default=-1) + 1
    goal_profile_stats = {
        "compressed_goal_requirement_profiles": 0,
        "compressed_goal_requirement_profile_reused": 0,
        "compressed_goal_requirement_profile_created": 0,
    }
    for goal_id, requirement_accepts in sorted(goal_requirement_accepts_by_id.items()):
        goal_profiles: list[tuple[int, str, int]] = []
        for requirement_id, category, accepted_values in requirement_accepts:
            profile_id = profile_id_by_accepts.get(accepted_values)
            if profile_id is None:
                profile_id = next_profile_id
                next_profile_id += 1
                profile_id_by_accepts[accepted_values] = profile_id
                profile_accepts_by_id[profile_id] = accepted_values
                goal_profile_stats["compressed_goal_requirement_profile_created"] += 1
            else:
                goal_profile_stats["compressed_goal_requirement_profile_reused"] += 1
            goal_profiles.append((requirement_id, category, profile_id))
            goal_profile_stats["compressed_goal_requirement_profiles"] += 1
        goal_requirement_profiles_by_id[goal_id] = tuple(goal_profiles)
    for record in relevant_records:
        compressed_output_ports: list[dict[str, object]] = []
        for output_port in tuple(record["output_ports"]):
            preserve_goal_profiles = str(record["candidate_id"]) in direct_goal_candidates
            compression_cache_key = (
                _dynamic_dim_values_cache_key(output_port["port_values_by_dimension"]),
                preserve_goal_profiles,
            )
            compressed_vals = output_compression_cache.get(compression_cache_key)
            if compressed_vals is None:
                compressed_vals = _compress_dynamic_output_choice_values(
                    ontology,
                    output_port["port_values_by_dimension"],
                    output_port["port_values_fset"],
                    bindable_input_signatures,
                    goal_port_values_tuple,
                    goal_fsets,
                    preserve_goal_profiles=preserve_goal_profiles,
                )
                output_compression_cache[compression_cache_key] = compressed_vals
            compressed_fset = {dim: frozenset(vals) for dim, vals in compressed_vals.items()}
            compressed_output_ports.append(
                {
                    **output_port,
                    "port_values_by_dimension": compressed_vals,
                    "port_values_fset": compressed_fset,
                }
            )
        record["output_ports"] = _compress_duplicate_output_ports(tuple(compressed_output_ports))

    compressed_reverse_edges, compressed_produced_bindable_ports, signature_bindable_ports, bindability_stats = (
        _collect_compressed_dynamic_bindability_surface(
            bindable_pairs,
            relevant_records=relevant_records,
            signature_profiles_by_id=signature_profiles_by_id,
            profile_accepts_by_id=profile_accepts_by_id,
        )
    )
    candidate_records_by_id = {
        str(record["candidate_id"]): record
        for record in relevant_records
    }

    query_goal_candidates: set[str] = set()
    for record in relevant_records:
        candidate_id = str(record["candidate_id"])
        output_ports = tuple(record["output_ports"])
        goals_satisfied = all(
            any(
                _dynamic_output_matches_dynamic_input_fset(
                    output_port["port_values_fset"],
                    gf,
                )
                for output_port in output_ports
            )
            for gf in goal_fsets
        )
        if not goals_satisfied:
            continue
        total_output_multiplicity = sum(
            int(output_port.get("multiplicity", 1))
            for output_port in output_ports
        )
        if config.use_all_generated_data == "ALL" and any(
            not any(
                _dynamic_output_matches_dynamic_input_fset(
                    output_port["port_values_fset"],
                    gf,
                )
                for gf in goal_fsets
            )
            for output_port in output_ports
        ):
            continue
        if config.use_all_generated_data == "ALL" and total_output_multiplicity != len(goal_fsets):
            continue
        query_goal_candidates.add(candidate_id)
    query_goal_tools = {
        str(record["tool"].mode_id)
        for record in relevant_records
        if str(record["candidate_id"]) in query_goal_candidates
    }

    min_step_by_candidate = _compute_dynamic_candidate_min_steps(
        relevant_records,
        workflow_bindable_ports,
        compressed_produced_bindable_ports,
    )
    tool_min_steps: dict[str, int] = {}
    for record in relevant_records:
        candidate_id = str(record["candidate_id"])
        tool_id = str(record["tool"].mode_id)
        min_step = min_step_by_candidate.get(candidate_id)
        if min_step is None:
            continue
        existing_step = tool_min_steps.get(tool_id)
        if existing_step is None or min_step < existing_step:
            tool_min_steps[tool_id] = min_step
    must_use_min_steps, at_step_lower_bounds = _collect_dynamic_selector_lower_bounds(
        config,
        ontology,
        tools,
        tool_min_steps=tool_min_steps,
    )
    exact_prefix_lower_bound = _collect_dynamic_exact_prefix_lower_bound(
        config,
        ontology,
        tools,
        candidate_records=relevant_records,
        workflow_bindable_ports=workflow_bindable_ports,
        produced_bindable_ports=compressed_produced_bindable_ports,
        query_goal_candidates=query_goal_candidates,
    )
    allowed_candidates_by_step: dict[int, set[str]] = defaultdict(set)
    allowed_tools_by_step: dict[int, set[str]] = defaultdict(set)
    max_step_by_candidate: dict[str, int] = {}
    for record in relevant_records:
        candidate_id = str(record["candidate_id"])
        tool_id = str(record["tool"].mode_id)
        min_step = min_step_by_candidate.get(candidate_id)
        if min_step is None:
            continue
        max_step = config.solution_length_max
        if config.use_all_generated_data == "ALL":
            anchor_distance = min_anchor_distance_by_candidate.get(candidate_id)
            if anchor_distance is not None:
                max_step = min(
                    max_step,
                    config.solution_length_max - anchor_distance,
                )
        if max_step < min_step:
            continue
        max_step_by_candidate[candidate_id] = max_step

    compressed_produced_bindable_ports, temporally_dropped_associations = _filter_bindability_by_temporal_overlap(
        produced_bindable_ports=compressed_produced_bindable_ports,
        min_step_by_candidate=min_step_by_candidate,
        max_step_by_candidate=max_step_by_candidate,
    )
    compressed_reverse_edges = _reverse_edges_from_produced_bindable_ports(compressed_produced_bindable_ports)

    closure_goal_candidates = set(query_goal_candidates) | set(backward_relevant_candidates)
    brave_closed_candidates = _close_relevant_candidates(
        candidate_records_by_id=candidate_records_by_id,
        workflow_bindable_ports=workflow_bindable_ports,
        produced_bindable_ports=compressed_produced_bindable_ports,
        goal_candidates=closure_goal_candidates,
    )
    relevant_records = [
        record
        for record in relevant_records
        if str(record["candidate_id"]) in brave_closed_candidates
    ]
    candidate_records_by_id = {
        str(record["candidate_id"]): record
        for record in relevant_records
    }
    query_goal_candidates &= brave_closed_candidates
    query_goal_tools = {
        str(record["tool"].mode_id)
        for record in relevant_records
        if str(record["candidate_id"]) in query_goal_candidates
    }
    compressed_produced_bindable_ports = {
        consumer_candidate: {
            consumer_port: {
                producer_candidate
                for producer_candidate in producer_candidates
                if producer_candidate in brave_closed_candidates
            }
            for consumer_port, producer_candidates in by_port.items()
            if any(producer_candidate in brave_closed_candidates for producer_candidate in producer_candidates)
        }
        for consumer_candidate, by_port in compressed_produced_bindable_ports.items()
        if consumer_candidate in brave_closed_candidates
    }
    compressed_reverse_edges = _reverse_edges_from_produced_bindable_ports(compressed_produced_bindable_ports)
    min_step_by_candidate = _compute_dynamic_candidate_min_steps(
        relevant_records,
        workflow_bindable_ports,
        compressed_produced_bindable_ports,
    )
    max_step_by_candidate = {
        candidate_id: max_step
        for candidate_id, max_step in max_step_by_candidate.items()
        if candidate_id in brave_closed_candidates
    }
    signature_bindable_ports = _filter_signature_bindable_ports(
        signature_bindable_ports,
        active_candidates=brave_closed_candidates,
    )
    (
        signature_support_class_by_id,
        support_class_bindable_ports,
        support_class_stats,
    ) = _factor_signature_support_classes(signature_bindable_ports)
    (
        association_class_by_input,
        association_class_bindable_ports,
        feasible_associations_by_input,
        association_class_stats,
    ) = _factor_input_association_classes(
        relevant_records=relevant_records,
        workflow_bindable_ports=workflow_bindable_ports,
        signature_bindable_ports=signature_bindable_ports,
        min_step_by_candidate=min_step_by_candidate,
        max_step_by_candidate=max_step_by_candidate,
    )

    # Still intentionally empty; canonical producers remain a solve-time notion.
    canonical_producers: dict[tuple[int, int], tuple[int, int]] = {}

    relevant_tools = tuple(record["tool"] for record in relevant_records)
    min_goal_distance_by_candidate: dict[str, int] = {}
    frontier = deque(sorted(query_goal_candidates))
    for candidate_id in frontier:
        min_goal_distance_by_candidate[candidate_id] = 0
    while frontier:
        consumer_candidate = frontier.popleft()
        next_distance = min_goal_distance_by_candidate[consumer_candidate] + 1
        for producer_candidate in sorted(compressed_reverse_edges.get(consumer_candidate, set())):
            if producer_candidate in min_goal_distance_by_candidate:
                continue
            if producer_candidate not in candidate_records_by_id:
                continue
            min_goal_distance_by_candidate[producer_candidate] = next_distance
            frontier.append(producer_candidate)

    must_run_candidates_global, must_run_tools_global = _compute_global_must_run_candidates(
        relevant_records=relevant_records,
        workflow_bindable_ports=workflow_bindable_ports,
        produced_bindable_ports=compressed_produced_bindable_ports,
        query_goal_candidates=query_goal_candidates,
    )
    (
        propagated_required_candidates,
        forced_associations_global,
        fixpoint_rounds,
    ) = _propagate_forced_associations(
        candidate_records_by_id=candidate_records_by_id,
        workflow_bindable_ports=workflow_bindable_ports,
        feasible_associations_by_input=feasible_associations_by_input,
        seed_required_candidates=must_run_candidates_global,
    )
    must_run_candidates_global = propagated_required_candidates
    must_run_tools_global = tuple(
        sorted(
            {
                str(candidate_records_by_id[candidate_id]["tool"].mode_id)
                for candidate_id in must_run_candidates_global
                if candidate_id in candidate_records_by_id
            }
        )
    )

    (
        goal_support_candidates_by_horizon,
        goal_support_tools_by_horizon,
        goal_support_inputs_by_horizon,
        smart_expansion_rounds_by_horizon,
    ) = _compute_smart_expansion_by_horizon(
        config=config,
        candidate_records_by_id=candidate_records_by_id,
        query_goal_candidates=query_goal_candidates,
        feasible_associations_by_input=feasible_associations_by_input,
        min_step_by_candidate=min_step_by_candidate,
        max_step_by_candidate=max_step_by_candidate,
        min_goal_distance_by_candidate=min_goal_distance_by_candidate,
    )
    (
        structurally_supportable_candidates_by_horizon,
        structurally_unsupported_inputs_by_horizon,
        input_support_rounds_by_horizon,
    ) = _compute_structural_input_support_by_horizon(
        goal_support_candidates_by_horizon=goal_support_candidates_by_horizon,
        goal_support_inputs_by_horizon=goal_support_inputs_by_horizon,
        candidate_records_by_id=candidate_records_by_id,
        association_class_by_input=association_class_by_input,
        association_class_bindable_ports=association_class_bindable_ports,
        min_step_by_candidate=min_step_by_candidate,
        max_step_by_candidate=max_step_by_candidate,
    )
    (
        check_relevant_output_categories_by_horizon,
        check_required_profile_classes_by_horizon,
        check_profile_class_members,
    ) = _compute_check_surface_by_horizon(
        goal_support_candidates_by_horizon=goal_support_candidates_by_horizon,
        goal_support_inputs_by_horizon=goal_support_inputs_by_horizon,
        candidate_records_by_id=candidate_records_by_id,
        feasible_associations_by_input=feasible_associations_by_input,
        signature_profiles_by_id=signature_profiles_by_id,
        query_goal_candidates=query_goal_candidates,
        goal_requirement_profiles_by_id=goal_requirement_profiles_by_id,
        goal_fsets=goal_fsets,
    )

    t4 = perf_counter()

    for record in relevant_records:
        candidate_id = str(record["candidate_id"])
        tool_id = str(record["tool"].mode_id)
        min_step = min_step_by_candidate.get(candidate_id)
        max_step = max_step_by_candidate.get(candidate_id)
        if min_step is None or max_step is None or max_step < min_step:
            continue
        for step_index in range(min_step, max_step + 1):
            allowed_candidates_by_step[step_index].add(candidate_id)
            allowed_tools_by_step[step_index].add(tool_id)

    t5 = perf_counter()

    must_run_tools_by_step: dict[int, tuple[str, ...]] = {}
    must_run_candidates_by_step: dict[int, tuple[str, ...]] = {}
    forced_associations_by_step: dict[int, tuple[tuple[str, int, str, int], ...]] = {}
    for step_index, tool_ids in sorted(allowed_tools_by_step.items()):
        if len(tool_ids) == 1:
            must_run_tools_by_step[step_index] = tuple(sorted(tool_ids))
    for step_index, candidate_ids in sorted(allowed_candidates_by_step.items()):
        if len(candidate_ids) == 1:
            must_run_candidates_by_step[step_index] = tuple(sorted(candidate_ids))
        step_forced_associations = tuple(
            sorted(
                association
                for association in forced_associations_global
                if association[0] in candidate_ids
            )
        )
        if step_forced_associations:
            forced_associations_by_step[step_index] = step_forced_associations

    earliest_goal_step = min(
        (
            min_step_by_candidate[candidate_id]
            for candidate_id in query_goal_candidates
            if candidate_id in min_step_by_candidate
        ),
        default=config.solution_length_max + 1,
    )
    earliest_solution_step = max(
        1,
        earliest_goal_step,
        exact_prefix_lower_bound,
        *must_use_min_steps,
        *at_step_lower_bounds,
    )
    structural_horizon_skip_count = max(0, earliest_solution_step - config.solution_length_min)
    candidate_pruned_count = max(0, candidate_count_before_pruning - len(relevant_records))
    structural_probe_horizons = tuple(
        sorted(
            step_index
            for step_index, candidate_ids in allowed_candidates_by_step.items()
            if step_index >= earliest_solution_step
            and any(candidate_id in query_goal_candidates for candidate_id in candidate_ids)
        )
    )
    structural_horizon_skip_count += max(
        0,
        sum(
            1
            for step_index in range(max(config.solution_length_min, earliest_solution_step), config.solution_length_max + 1)
            if step_index not in set(structural_probe_horizons)
        )
    )

    return CompressedCandidateOptimizationResult(
        tool_stats=tuple(tool_stats),
        relevant_records=tuple(relevant_records),
        relevant_tools=relevant_tools,
        query_goal_candidates=tuple(sorted(query_goal_candidates)),
        query_goal_tools=tuple(sorted(query_goal_tools)),
        min_goal_distance_by_candidate=min_goal_distance_by_candidate,
        min_step_by_candidate=min_step_by_candidate,
        max_step_by_candidate=max_step_by_candidate,
        goal_support_candidates_by_horizon=goal_support_candidates_by_horizon,
        goal_support_tools_by_horizon=goal_support_tools_by_horizon,
        goal_support_inputs_by_horizon=goal_support_inputs_by_horizon,
        smart_expansion_rounds_by_horizon=smart_expansion_rounds_by_horizon,
        structurally_supportable_candidates_by_horizon=structurally_supportable_candidates_by_horizon,
        structurally_unsupported_inputs_by_horizon=structurally_unsupported_inputs_by_horizon,
        input_support_rounds_by_horizon=input_support_rounds_by_horizon,
        allowed_candidates_by_step={
            step_index: tuple(sorted(candidate_ids))
            for step_index, candidate_ids in sorted(allowed_candidates_by_step.items())
        },
        allowed_tools_by_step={
            step_index: tuple(sorted(tool_ids))
            for step_index, tool_ids in sorted(allowed_tools_by_step.items())
        },
        check_relevant_output_categories_by_horizon=check_relevant_output_categories_by_horizon,
        check_required_profile_classes_by_horizon=check_required_profile_classes_by_horizon,
        check_profile_class_members=check_profile_class_members,
        signature_profiles_by_id=signature_profiles_by_id,
        profile_accepts_by_id=profile_accepts_by_id,
        goal_requirement_profiles_by_id=goal_requirement_profiles_by_id,
        signature_support_class_by_id=signature_support_class_by_id,
        support_class_bindable_ports=support_class_bindable_ports,
        canonical_producers=canonical_producers,
        cache_stats={
            **resolver.stats(),
            **dynamic_schema_stats,
            **bindability_stats,
            **support_class_stats,
            **association_class_stats,
            **goal_profile_stats,
            "dynamic_candidates_before_pruning": candidate_count_before_pruning,
            "dynamic_candidates_after_pruning": len(relevant_records),
            "dynamic_candidates_pruned": candidate_pruned_count,
            "dynamic_brave_relevant_candidates_after_closure": len(brave_closed_candidates),
            "dynamic_temporally_impossible_associations_pruned": temporally_dropped_associations,
            "dynamic_check_profile_classes": len(check_profile_class_members),
            "dynamic_check_relevant_output_categories": sum(
                len(entries) for entries in check_relevant_output_categories_by_horizon.values()
            ),
            "dynamic_check_required_profile_classes": sum(
                len(entries) for entries in check_required_profile_classes_by_horizon.values()
            ),
            "dynamic_forced_associations_global": len(forced_associations_global),
            "dynamic_fixpoint_rounds": fixpoint_rounds,
            "dynamic_goal_support_horizons": len(goal_support_candidates_by_horizon),
            "dynamic_goal_support_candidates": sum(
                len(candidate_ids)
                for candidate_ids in goal_support_candidates_by_horizon.values()
            ),
            "dynamic_goal_support_inputs": sum(
                len(input_ports)
                for input_ports in goal_support_inputs_by_horizon.values()
            ),
            "dynamic_structurally_supportable_candidates": sum(
                len(candidate_ids)
                for candidate_ids in structurally_supportable_candidates_by_horizon.values()
            ),
            "dynamic_structurally_unsupported_inputs": sum(
                len(input_ports)
                for input_ports in structurally_unsupported_inputs_by_horizon.values()
            ),
            "dynamic_smart_expansion_rounds_total": sum(
                smart_expansion_rounds_by_horizon.values()
            ),
            "dynamic_input_support_rounds_total": sum(
                input_support_rounds_by_horizon.values()
            ),
        },
        must_run_tools_global=must_run_tools_global,
        must_run_candidates_global=must_run_candidates_global,
        must_run_tools_by_step=must_run_tools_by_step,
        must_run_candidates_by_step=must_run_candidates_by_step,
        forced_associations_global=forced_associations_global,
        forced_associations_by_step=forced_associations_by_step,
        association_class_by_input=association_class_by_input,
        association_class_bindable_ports=association_class_bindable_ports,
        fixpoint_rounds=fixpoint_rounds,
        structural_probe_horizons=structural_probe_horizons,
        structural_horizon_skip_count=structural_horizon_skip_count,
        earliest_solution_step=earliest_solution_step,
        phase_timings={
            "goal_check": t1 - t0,
            "bindable_pairs": t2 - t1,
            "bfs_pruning": t3 - t2,
            "compression": t4 - t3,
            "step_indexing": t5 - t4,
        },
    )
