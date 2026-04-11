"""Dynamic-mode precomputation and optimization pipeline."""

from __future__ import annotations
from clindaws.translators.constraints import _collect_dynamic_forbidden_tool_ids, _collect_dynamic_backward_relevant_candidates, _collect_dynamic_selector_lower_bounds, _collect_dynamic_exact_prefix_lower_bound
from clindaws.translators.utils import _dedupe_stable, _normalize_dim_map
from clindaws.translators.ports import _group_port_values_by_dimension, _compress_dynamic_output_choice_values, _compress_duplicate_output_ports, _dynamic_output_matches_dynamic_input_fset, _workflow_input_matches_dynamic_port, _dynamic_port_expansion
from clindaws.translators.signatures import _assign_dynamic_signature_profiles
from clindaws.translators.candidates import _collect_compressed_dynamic_bindability_surface, _compute_dynamic_candidate_min_steps, _dynamic_dim_values_cache_key
from clindaws.translators.resolvers import _ExpansionResolver
from clindaws.translators.builder import _build_roots

from collections import defaultdict, deque
from collections.abc import Mapping
from dataclasses import dataclass
from time import perf_counter

from clindaws.core.models import SnakeConfig, ToolExpansionStat, ToolMode
from clindaws.core.ontology import Ontology


@dataclass(frozen=True)
class DynamicOptimizationResult:
    """Optimized dynamic candidate surface ready for fact emission."""

    tool_stats: tuple[ToolExpansionStat, ...]
    relevant_records: tuple[dict[str, object], ...]
    relevant_tools: tuple[ToolMode, ...]
    query_goal_candidates: tuple[str, ...]
    query_goal_tools: tuple[str, ...]
    min_goal_distance_by_candidate: dict[str, int]
    allowed_candidates_by_step: dict[int, tuple[str, ...]]
    allowed_tools_by_step: dict[int, tuple[str, ...]]
    signature_profiles_by_id: dict[int, dict[str, tuple[int, tuple[str, ...]]]]
    profile_accepts_by_id: dict[int, tuple[str, ...]]
    signature_support_class_by_id: dict[int, int]
    support_class_bindable_ports: dict[int, tuple[tuple[str, int], ...]]
    cache_stats: dict[str, int]
    earliest_solution_step: int
    phase_timings: dict[str, float]


def _factor_signature_support_classes(
    signature_bindable_ports: Mapping[tuple[int, str], int],
) -> tuple[
    dict[int, int],
    dict[int, tuple[tuple[str, int], ...]],
    dict[str, int],
]:
    """Compress identical signature support sets into shared support classes."""

    ports_by_signature: dict[int, list[tuple[str, int]]] = defaultdict(list)
    for (signature_id, producer_candidate), producer_port in signature_bindable_ports.items():
        ports_by_signature[signature_id].append((producer_candidate, int(producer_port)))

    support_class_by_key: dict[tuple[tuple[str, int], ...], int] = {}
    signature_support_class_by_id: dict[int, int] = {}
    support_class_bindable_ports: dict[int, tuple[tuple[str, int], ...]] = {}

    for signature_id, producer_ports in sorted(ports_by_signature.items()):
        support_key = tuple(sorted(producer_ports))
        support_class_id = support_class_by_key.setdefault(support_key, len(support_class_by_key))
        signature_support_class_by_id[signature_id] = support_class_id
        support_class_bindable_ports.setdefault(support_class_id, support_key)

    raw_signature_supports = len(ports_by_signature)
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


def optimize_dynamic_candidates(
    config: SnakeConfig,
    ontology: Ontology,
    tools: tuple[ToolMode, ...],
) -> DynamicOptimizationResult:
    """Precompute the dynamic candidate surface before fact emission."""

    roots = _build_roots(config, ontology)
    resolver = _ExpansionResolver(ontology, roots, "python")
    tool_stats: list[ToolExpansionStat] = []
    candidate_records: list[dict[str, object]] = []

    dynamic_offset: dict[str, int] = defaultdict(int)

    forbidden_tool_ids = _collect_dynamic_forbidden_tool_ids(config, ontology, tools)
    candidate_source_tools = tuple(
        tool
        for tool in tools
        if tool.mode_id not in forbidden_tool_ids
    )

    for tool in candidate_source_tools:
        candidate_index = dynamic_offset[tool.mode_id]
        candidate_id = f"{tool.mode_id}_lc{candidate_index}"
        dynamic_offset[tool.mode_id] += 1

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

        tool_stats.append(
            ToolExpansionStat(
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
        )
        candidate_records.append(
            {
                "tool": tool,
                "candidate_id": candidate_id,
                "input_ports": tuple(input_ports),
                "output_ports": _compress_duplicate_output_ports(tuple(output_ports)),
            }
        )

    goal_port_values: list[dict[str, tuple[str, ...]]] = []
    for goal_item in config.outputs:
        goal_dims: dict[str, tuple[str, ...]] = {}
        for dim, values in sorted(goal_item.items()):
            goal_dims[dim] = _dedupe_stable(
                expanded_value
                for value in values
                for expanded_value in resolver.expanded_values(
                    dim,
                    value,
                    expand_outputs=True,
                )
            )
        goal_port_values.append(goal_dims)
    goal_port_values_tuple = tuple(goal_port_values)
    goal_fsets: tuple[dict[str, frozenset[str]], ...] = tuple(
        {dim: frozenset(vals) for dim, vals in g.items()} for g in goal_port_values
    )

    t0 = perf_counter()

    bindable_pairs: set[tuple[str, int, str, int]] = set()
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
    if config.use_all_generated_data == "ALL":
        backward_relevant_candidates, min_anchor_distance_by_candidate = _collect_dynamic_backward_relevant_candidates(
            config,
            ontology,
            tools,
            candidate_records=(
                record
                for record in candidate_records
                if str(record["candidate_id"]) in relevant_candidates
            ),
            reverse_edges=reverse_edges,
            direct_goal_candidates=direct_goal_candidates,
        )
        relevant_candidates &= backward_relevant_candidates

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
    signature_profiles_by_id, _profile_values_by_id, profile_accepts_by_id, dynamic_schema_stats = _assign_dynamic_signature_profiles(
        ontology,
        roots,
        relevant_input_ports,
    )
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
                    relevant_input_ports,
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
    (
        signature_support_class_by_id,
        support_class_bindable_ports,
        support_class_stats,
    ) = _factor_signature_support_classes(signature_bindable_ports)

    t4 = perf_counter()

    relevant_tools = tuple(record["tool"] for record in relevant_records)
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
            if producer_candidate not in relevant_candidates:
                continue
            min_goal_distance_by_candidate[producer_candidate] = next_distance
            frontier.append(producer_candidate)

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
        for step_index in range(min_step, max_step + 1):
            allowed_candidates_by_step[step_index].add(candidate_id)
            allowed_tools_by_step[step_index].add(tool_id)

    t5 = perf_counter()

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

    return DynamicOptimizationResult(
        tool_stats=tuple(tool_stats),
        relevant_records=tuple(relevant_records),
        relevant_tools=relevant_tools,
        query_goal_candidates=tuple(sorted(query_goal_candidates)),
        query_goal_tools=tuple(sorted(query_goal_tools)),
        min_goal_distance_by_candidate=min_goal_distance_by_candidate,
        allowed_candidates_by_step={
            step_index: tuple(sorted(candidate_ids))
            for step_index, candidate_ids in sorted(allowed_candidates_by_step.items())
        },
        allowed_tools_by_step={
            step_index: tuple(sorted(tool_ids))
            for step_index, tool_ids in sorted(allowed_tools_by_step.items())
        },
        signature_profiles_by_id=signature_profiles_by_id,
        profile_accepts_by_id=profile_accepts_by_id,
        signature_support_class_by_id=signature_support_class_by_id,
        support_class_bindable_ports=support_class_bindable_ports,
        cache_stats={
            **resolver.stats(),
            **dynamic_schema_stats,
            **bindability_stats,
            **support_class_stats,
        },
        earliest_solution_step=earliest_solution_step,
        phase_timings={
            "goal_check": t1 - t0,
            "bindable_pairs": t2 - t1,
            "bfs_pruning": t3 - t2,
            "compression": t4 - t3,
            "step_indexing": t5 - t4,
        },
    )
