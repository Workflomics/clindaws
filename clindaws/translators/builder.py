from __future__ import annotations
from collections import defaultdict, deque
from itertools import product
from time import perf_counter

from clindaws.core.models import FactBundle, SnakeConfig, ToolExpansionStat, ToolMode
from clindaws.core.ontology import Ontology

from clindaws.translators.utils import _dedupe_stable, _product, _normalize_dim_map, _quote
from clindaws.translators.signatures import _assign_dynamic_signature_profiles, _tool_input_signatures
from clindaws.translators.candidates import _dynamic_dim_values_cache_key, _collect_compressed_dynamic_bindability_surface, _compute_ape_multi_shot_earliest_solution_step, _compute_dynamic_candidate_min_steps
from clindaws.translators.ports import _workflow_input_matches_dynamic_port, _group_port_values_by_dimension, _compress_dynamic_output_choice_values, _dynamic_port_expansion, _compress_duplicate_output_ports, _dynamic_output_matches_dynamic_input_fset
from clindaws.translators.fact_writer import _FactWriter
from clindaws.translators.constraints import _collect_dynamic_exact_prefix_lower_bound, _emit_dynamic_constraints, _collect_dynamic_forbidden_tool_ids, _collect_dynamic_backward_relevant_candidates, _collect_dynamic_selector_lower_bounds
from clindaws.translators.resolvers import _ExpansionResolver



def _bundle_metadata(
    config: SnakeConfig,
    tools: tuple[ToolMode, ...],
) -> tuple[dict[str, str], dict[str, tuple[tuple[tuple[str, tuple[str, ...]], ...], ...]], tuple[str, ...]]:
    return (
        {tool.mode_id: tool.label for tool in tools},
        _tool_input_signatures(tools),
        tuple(f"wf_input_{i}" for i in range(len(config.inputs))),
    )
def _finalize_fact_bundle(
    writer: "_FactWriter",
    *,
    config: SnakeConfig,
    tools: tuple[ToolMode, ...],
    tool_stats: list[ToolExpansionStat],
    cache_stats: dict[str, int],
    backend_stats: dict[str, object] | None = None,
    earliest_solution_step: int = 1,
) -> FactBundle:
    tool_labels, tool_input_signatures, workflow_input_ids = _bundle_metadata(config, tools)
    return FactBundle(
        facts=writer.text(),
        fact_count=writer.fact_count,
        tool_labels=tool_labels,
        tool_input_signatures=tool_input_signatures,
        workflow_input_ids=workflow_input_ids,
        goal_count=len(config.outputs),
        predicate_counts=dict(writer.predicate_counts),
        tool_stats=tuple(tool_stats),
        cache_stats=cache_stats,
        emit_stats=writer.stats(),
        backend_stats=backend_stats or {},
        earliest_solution_step=earliest_solution_step,
    )
def _build_roots(
    config: SnakeConfig,
    ontology: Ontology,
) -> dict[str, frozenset[str]]:
    tool_taxonomy_nodes = ontology.descendants_of(config.tools_taxonomy_root)
    raw_roots = {
        root: frozenset(
            node
            for node in ontology.descendants_of(root)
            if node not in tool_taxonomy_nodes
        )
        for root in config.data_dimensions_taxonomy_roots
    }
    roots: dict[str, frozenset[str]] = {}
    for root in config.data_dimensions_taxonomy_roots:
        excluded: set[str] = set()
        for other_root in config.data_dimensions_taxonomy_roots:
            if other_root == root:
                continue
            if other_root in raw_roots[root]:
                excluded.update(raw_roots[other_root])
        roots[root] = frozenset(node for node in raw_roots[root] if node not in excluded)
    return roots
def _build_common_facts(
    writer: _FactWriter,
    config: SnakeConfig,
    ontology: Ontology,
    tools: tuple[ToolMode, ...],
) -> dict[str, frozenset[str]]:
    """Build taxonomy/tool/workflow/goal facts shared by all bundles."""
    roots = _build_roots(config, ontology)
    tool_taxonomy_nodes = ontology.descendants_of(config.tools_taxonomy_root)

    for dimension_root in config.data_dimensions_taxonomy_roots:
        allowed = roots[dimension_root]
        for child, parent in ontology.edges:
            if child in allowed and parent in allowed:
                writer.emit_fact(
                    "taxonomy",
                    _quote("ape"),
                    _quote(dimension_root),
                    f"({_quote(dimension_root)}, {_quote(child)}, {_quote(parent)})",
                )

    # Pre-compute compatible(V, Ancestor) for every terminal V in each dimension
    # subtree. Python's cached BFS (ontology.ancestors_of) replaces the O(n²)
    # recursive ancestor/2 transitive closure that clingo would otherwise derive
    # at grounding time. ancestors_of(V) includes V itself, so compatible(V, V)
    # is covered without a separate rule.
    for dimension_root in config.data_dimensions_taxonomy_roots:
        allowed = roots[dimension_root]
        for terminal in ontology.terminal_descendants_of(dimension_root, within=allowed):
            for anc in ontology.ancestors_of(terminal):
                if anc in allowed:
                    writer.emit_fact("compatible", _quote(terminal), _quote(anc))

    for child, parent in ontology.edges:
        if child in tool_taxonomy_nodes and parent in tool_taxonomy_nodes:
            writer.emit_fact(
                "tool_taxonomy",
                _quote("ape"),
                f"({_quote(child)}, {_quote(parent)})",
            )

    for tool in tools:
        writer.emit_fact("tool", _quote(tool.mode_id))
        for tax_op in tool.taxonomy_operations:
            writer.emit_fact(
                "tool_taxonomy",
                _quote("ape"),
                f"({_quote(tool.mode_id)}, {_quote(tax_op)})",
            )

    for index, item in enumerate(config.inputs):
        wf_id = f"wf_input_{index}"
        writer.emit_fact("holds", "0", f"avail({_quote(wf_id)})")
        for dim, values in sorted(item.items()):
            allowed = roots.get(dim, frozenset())
            for value in values:
                for tval in ontology.terminal_descendants_of(value, within=allowed):
                    writer.emit_fact(
                        "holds",
                        "0",
                        f"dim({_quote(wf_id)}, {_quote(tval)}, {_quote(dim)})",
                    )
                    writer.emit_fact(
                        "ape_holds_dim",
                        _quote(wf_id),
                        _quote(tval),
                        _quote(dim),
                    )

    for goal_index, item in enumerate(config.outputs):
        for dim, values in sorted(item.items()):
            for value in values:
                writer.emit_fact(
                    "goal_output",
                    str(goal_index),
                    _quote(value),
                    _quote(dim),
                )

    if config.use_workflow_input == "ALL":
        writer.emit_atom("enable_all_inputs_used")
    elif config.use_workflow_input == "ONE":
        writer.emit_atom("enable_some_input_used")

    if config.use_all_generated_data == "ALL":
        writer.emit_atom("enable_all_outputs_consumed")
        writer.emit_atom("enable_usefulness_pruning")
    elif config.use_all_generated_data == "ONE":
        writer.emit_atom("enable_primary_output_consumed")
        writer.emit_atom("enable_usefulness_pruning")

    if config.tool_seq_repeat:
        writer.emit_rule("multi_run", "multi_run(Tool) :- tool(Tool).")

    return roots
def build_dynamic_fact_bundle(
    config: SnakeConfig,
    ontology: Ontology,
    tools: tuple[ToolMode, ...],
) -> FactBundle:
    """Compatibility wrapper for the dynamic builder implementation."""
    from clindaws.translators.translator_dynamic import build_dynamic_fact_bundle as _impl

    return _impl(config, ontology, tools)

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

    _t0 = perf_counter()

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
        output_ports = tuple(record["output_ports"])
        for output_port in output_ports:
            output_fset = output_port["port_values_fset"]
            if any(_dynamic_output_matches_dynamic_input_fset(output_fset, gf) for gf in goal_fsets):
                direct_goal_candidates.add(candidate_id)

    _t1 = perf_counter()

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

    _t2 = perf_counter()

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

    _t3 = perf_counter()

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
    signature_profiles_by_id, profile_values_by_id, profile_accepts_by_id, dynamic_schema_stats = _assign_dynamic_signature_profiles(
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

    _t4 = perf_counter()
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
    frontier: deque[str] = deque(sorted(query_goal_candidates))
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
    writer = _FactWriter()
    _build_common_facts(writer, config, ontology, relevant_tools)
    _emit_dynamic_constraints(writer, config, ontology, tools)

    for candidate_id in sorted(query_goal_candidates):
        writer.emit_fact("dynamic_query_goal_candidate", _quote(candidate_id))
    for tool_id in sorted(query_goal_tools):
        writer.emit_fact("dynamic_query_goal_tool", _quote(tool_id))
    for candidate_id, goal_distance in sorted(min_goal_distance_by_candidate.items()):
        writer.emit_fact("dynamic_candidate_goal_distance", _quote(candidate_id), str(goal_distance))
    for step_index, candidate_ids in sorted(allowed_candidates_by_step.items()):
        for candidate_id in sorted(candidate_ids):
            writer.emit_fact(
                "dynamic_candidate_allowed_at_step",
                _quote(candidate_id),
                str(step_index),
            )
    for step_index, tool_ids in sorted(allowed_tools_by_step.items()):
        for tool_id in sorted(tool_ids):
            writer.emit_fact(
                "dynamic_step_allowed_tool",
                _quote(tool_id),
                str(step_index),
            )
    _t5 = perf_counter()

    for record in relevant_records:
        tool = record["tool"]
        candidate_id = str(record["candidate_id"])
        writer.emit_fact("dynamic_tool_candidate", _quote(candidate_id), _quote(tool.mode_id))

        for input_port in tuple(record["input_ports"]):
            port_idx = int(input_port["port_idx"])
            writer.emit_fact(
                "dynamic_candidate_input_port",
                _quote(candidate_id),
                str(port_idx),
            )
            writer.emit_fact(
                "dynamic_candidate_input_signature_id",
                _quote(candidate_id),
                str(port_idx),
                str(input_port["signature_id"]),
            )
            for wf_index in input_port["workflow_input_matches"]:
                writer.emit_fact(
                    "dynamic_initial_bindable",
                    _quote(candidate_id),
                    str(port_idx),
                    _quote(f"wf_input_{wf_index}"),
                )

        for output_port in tuple(record["output_ports"]):
            port_idx = int(output_port["port_idx"])
            writer.emit_fact(
                "dynamic_candidate_output_port",
                _quote(candidate_id),
                str(port_idx),
            )
            writer.emit_fact(
                "dynamic_candidate_output_multiplicity",
                _quote(candidate_id),
                str(port_idx),
                str(int(output_port.get("multiplicity", 1))),
            )
            for dim, declared_values in output_port["declared_dims"].items():
                for declared_value in declared_values:
                    writer.emit_fact(
                        "dynamic_candidate_output_declared_type",
                        _quote(candidate_id),
                        str(port_idx),
                        _quote(declared_value),
                        _quote(dim),
                    )
            for dim, values in sorted(output_port["port_values_by_dimension"].items()):
                if len(values) == 1:
                    writer.emit_fact(
                        "dynamic_candidate_output_singleton",
                        _quote(candidate_id),
                        str(port_idx),
                        _quote(values[0]),
                        _quote(dim),
                    )
                else:
                    for value in values:
                        writer.emit_fact(
                            "dynamic_candidate_output_choice_value",
                            _quote(candidate_id),
                            str(port_idx),
                            _quote(value),
                            _quote(dim),
                        )
        writer.emit_fact(
            "dynamic_candidate_total_output_multiplicity",
            _quote(candidate_id),
            str(sum(int(output_port.get("multiplicity", 1)) for output_port in tuple(record["output_ports"]))),
        )

    for (signature_id, producer_candidate), producer_port in sorted(signature_bindable_ports.items()):
        writer.emit_fact(
            "dynamic_signature_bindable_producer_port",
            str(signature_id),
            _quote(producer_candidate),
            str(producer_port),
        )
    for signature_id, category_profiles in sorted(signature_profiles_by_id.items()):
        for dim, (profile_id, _values) in sorted(category_profiles.items()):
            writer.emit_fact(
                "dynamic_signature_profile",
                str(signature_id),
                _quote(dim),
                str(profile_id),
            )
    for profile_id, values in sorted(profile_values_by_id.items()):
        for value in values:
            writer.emit_fact(
                "dynamic_profile_value",
                str(profile_id),
                _quote(value),
            )
    for profile_id, values in sorted(profile_accepts_by_id.items()):
        for value in values:
            writer.emit_fact(
                "dynamic_profile_accepts",
                str(profile_id),
                _quote(value),
            )
    _t6 = perf_counter()
    print(
        f"  dynamic builder phases: "
        f"goal_check={_t1-_t0:.2f}s "
        f"bindable_pairs={_t2-_t1:.2f}s "
        f"bfs_pruning={_t3-_t2:.2f}s "
        f"compression={_t4-_t3:.2f}s "
        f"step_indexing={_t5-_t4:.2f}s "
        f"fact_emission={_t6-_t5:.2f}s"
    )

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

    return _finalize_fact_bundle(
        writer,
        config=config,
        tools=relevant_tools,
        tool_stats=tool_stats,
        cache_stats={**ontology.cache_stats(), **resolver.stats(), **dynamic_schema_stats, **bindability_stats},
        earliest_solution_step=earliest_solution_step,
    )
def build_fact_bundle(
    config: SnakeConfig,
    ontology: Ontology,
    tools: tuple[ToolMode, ...],
    strategy: str,
) -> FactBundle:
    """Build a solver-ready fact bundle."""

    writer = _FactWriter()
    roots = _build_common_facts(writer, config, ontology, tools)
    resolver = _ExpansionResolver(ontology, roots, strategy)
    tool_stats: list[ToolExpansionStat] = []

    # Per-mode_id offsets to avoid ID collisions when multiple ToolMode objects
    # share the same mode_id (e.g. relax_structure with Bulk vs Slab variants).
    _variant_offset: dict[str, int] = defaultdict(int)
    _output_offset: dict[str, int] = defaultdict(int)

    for tool in tools:
        input_port_variants = tuple(
            tuple(
                resolver.iter_dimension_maps(
                    _normalize_dim_map(port.dimensions),
                    expand_outputs=False,
                )
            )
            for port in tool.inputs
        )
        input_variant_count = _product(len(variants) for variants in input_port_variants) if input_port_variants else 1
        output_variant_count = len(tool.outputs)
        tool_stats.append(
            ToolExpansionStat(
                tool_id=tool.mode_id,
                tool_label=tool.label,
                input_ports=len(tool.inputs),
                output_ports=len(tool.outputs),
                input_variant_count=input_variant_count,
                output_variant_count=output_variant_count,
            )
        )
        if tool.inputs:
            v_base = _variant_offset[tool.mode_id]
            for variant_index, port_specs in enumerate(product(*input_port_variants)):
                variant_id = f"{tool.mode_id}_v{v_base + variant_index}"
                writer.emit_fact("tool_input", _quote(tool.mode_id), _quote(variant_id))
                for port_index, dims in enumerate(port_specs):
                    port_id = f"{variant_id}_p{port_index}"
                    writer.emit_fact("input_port", _quote(variant_id), _quote(port_id))
                    for dim, value in sorted(dims.items()):
                        writer.emit_fact(
                            "dimension",
                            _quote(port_id),
                            f"({_quote(value)}, {_quote(dim)})",
                        )
            _variant_offset[tool.mode_id] += input_variant_count

        o_base = _output_offset[tool.mode_id]
        for output_index, port in enumerate(tool.outputs):
            output_id = f"{tool.mode_id}_out_{o_base + output_index}"
            port_id = f"{output_id}_port_0"
            writer.emit_fact("tool_output", _quote(tool.mode_id), _quote(output_id))
            writer.emit_fact("output_port", _quote(output_id), _quote(port_id))
            for dim, values in sorted(_normalize_dim_map(port.dimensions).items()):
                for value in values:
                    writer.emit_fact(
                        "dimension",
                        _quote(port_id),
                        f"({_quote(value)}, {_quote(dim)})",
                    )
        _output_offset[tool.mode_id] += output_variant_count

    _emit_dynamic_constraints(writer, config, ontology, tools)
    return _finalize_fact_bundle(
        writer,
        config=config,
        tools=tools,
        tool_stats=tool_stats,
        cache_stats={**ontology.cache_stats(), **resolver.stats()},
    )
def build_fact_bundle_ape_multi_shot(
    config: SnakeConfig,
    ontology: Ontology,
    tools: tuple[ToolMode, ...],
) -> FactBundle:
    """Build facts matching APE's Java ClingoSynthesisEngine contract.

    This intentionally avoids any legacy variant expansion:
    - one `tool_input(..., "<tool>_v0")` per tool with inputs
    - one `tool_output(..., "<tool>_out_<i>")` per declared output port
    - raw declared dimension values only
    """

    writer = _FactWriter()
    roots = _build_common_facts(writer, config, ontology, tools)
    tool_stats: list[ToolExpansionStat] = []

    for tool in tools:
        tool_stats.append(
            ToolExpansionStat(
                tool_id=tool.mode_id,
                tool_label=tool.label,
                input_ports=len(tool.inputs),
                output_ports=len(tool.outputs),
                input_variant_count=1 if tool.inputs else 0,
                output_variant_count=len(tool.outputs),
            )
        )

        if tool.inputs:
            variant_id = f"{tool.mode_id}_v0"
            writer.emit_fact("tool_input", _quote(tool.mode_id), _quote(variant_id))
            for port_index, port in enumerate(tool.inputs):
                port_id = f"{variant_id}_p{port_index}"
                writer.emit_fact("input_port", _quote(variant_id), _quote(port_id))
                for dim, values in sorted(_normalize_dim_map(port.dimensions).items()):
                    for value in values:
                        writer.emit_fact(
                            "dimension",
                            _quote(port_id),
                            f"({_quote(value)}, {_quote(dim)})",
                        )

        for output_index, port in enumerate(tool.outputs):
            output_id = f"{tool.mode_id}_out_{output_index}"
            port_id = f"{output_id}_port_0"
            writer.emit_fact("tool_output", _quote(tool.mode_id), _quote(output_id))
            writer.emit_fact("output_port", _quote(output_id), _quote(port_id))
            for dim, values in sorted(_normalize_dim_map(port.dimensions).items()):
                for value in values:
                    writer.emit_fact(
                        "dimension",
                        _quote(port_id),
                        f"({_quote(value)}, {_quote(dim)})",
                    )

    _emit_dynamic_constraints(writer, config, ontology, tools)
    earliest_solution_step = _compute_ape_multi_shot_earliest_solution_step(
        config,
        ontology,
        tools,
        roots,
    )
    return _finalize_fact_bundle(
        writer,
        config=config,
        tools=tools,
        tool_stats=tool_stats,
        cache_stats=dict(ontology.cache_stats()),
        earliest_solution_step=earliest_solution_step,
    )
