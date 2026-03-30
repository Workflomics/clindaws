"""Translate config, OWL, and tool annotations into encoding-compatible facts."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from io import StringIO
from itertools import product
from typing import Iterable, Mapping

from .models import FactBundle, SnakeConfig, ToolExpansionStat, ToolMode
from .ontology import Ontology


def _quote(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _fact(name: str, *args: str) -> str:
    return f"{name}({', '.join(args)})."


def _product(values: Iterable[int]) -> int:
    total = 1
    for value in values:
        total *= value
    return total


def _normalize_dim_map(mapping: Mapping[str, Iterable[str]]) -> dict[str, tuple[str, ...]]:
    return {
        str(dim): tuple(str(value) for value in values)
        for dim, values in mapping.items()
    }


def _dedupe_stable(values: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return tuple(ordered)


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


@dataclass
class _FactWriter:
    """Incrementally build fact text and counts."""

    buffer: StringIO = field(default_factory=StringIO)
    predicate_counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    fact_count: int = 0

    def emit_fact(self, name: str, *args: str) -> None:
        self.buffer.write(_fact(name, *args))
        self.buffer.write("\n")
        self.predicate_counts[name] += 1
        self.fact_count += 1

    def emit_atom(self, name: str) -> None:
        self.buffer.write(f"{name}.\n")
        self.predicate_counts[name] += 1
        self.fact_count += 1

    def emit_rule(self, name: str, text: str) -> None:
        self.buffer.write(text)
        if not text.endswith("\n"):
            self.buffer.write("\n")
        self.predicate_counts[name] += 1
        self.fact_count += 1

    def text(self) -> str:
        return self.buffer.getvalue()

    def stats(self) -> dict[str, int]:
        return {
            "fact_count": self.fact_count,
            "text_chars": self.buffer.tell(),
        }


@dataclass
class _ExpansionResolver:
    """Reuse resolved terminal expansions across translation."""

    ontology: Ontology
    roots: Mapping[str, frozenset[str]]
    strategy: str
    _value_cache: dict[tuple[str, bool, str, str], tuple[str, ...]] = field(default_factory=dict)
    _stats: dict[str, int] = field(
        default_factory=lambda: {
            "value_expansion_hits": 0,
            "value_expansion_misses": 0,
        }
    )

    def _should_expand(self, expand_outputs: bool) -> bool:
        return self.strategy == "python" or (self.strategy == "hybrid" and expand_outputs)

    def expanded_values(self, dim: str, value: str, *, expand_outputs: bool) -> tuple[str, ...]:
        allowed_nodes = self.roots.get(dim, frozenset())
        if not self._should_expand(expand_outputs) or value not in self.ontology.nodes or not allowed_nodes:
            return (value,)

        cache_key = (self.strategy, expand_outputs, dim, value)
        cached = self._value_cache.get(cache_key)
        if cached is not None:
            self._stats["value_expansion_hits"] += 1
            return cached

        self._stats["value_expansion_misses"] += 1
        expanded = self.ontology.terminal_descendants_of(value, within=allowed_nodes)
        self._value_cache[cache_key] = expanded
        return expanded

    def iter_dimension_maps(
        self,
        dims: Mapping[str, tuple[str, ...]],
        *,
        expand_outputs: bool,
    ):
        per_dimension: list[tuple[tuple[str, str], ...]] = []
        for dim, values in dims.items():
            choices: list[tuple[str, str]] = []
            for value in values:
                for expanded_value in self.expanded_values(dim, value, expand_outputs=expand_outputs):
                    choices.append((dim, expanded_value))
            per_dimension.append(tuple(choices))

        if not per_dimension:
            yield {}
            return

        for choice in product(*per_dimension):
            combo: dict[str, str] = {}
            valid = True
            for dim, value in choice:
                if dim in combo and combo[dim] != value:
                    valid = False
                    break
                combo[dim] = value
            if valid:
                yield combo

    def stats(self) -> dict[str, int]:
        return dict(self._stats)


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
            for value in values:
                writer.emit_fact(
                    "holds",
                    "0",
                    f"dim({_quote(wf_id)}, {_quote(value)}, {_quote(dim)})",
                )
                writer.emit_fact(
                    "ape_holds_dim",
                    _quote(wf_id),
                    _quote(value),
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
        writer.emit_atom("enable_inputs_used_once")

    if config.use_all_generated_data == "ALL":
        writer.emit_atom("enable_all_outputs_consumed")
    elif config.use_all_generated_data == "ONE":
        writer.emit_atom("enable_primary_output_consumed")

    if config.tool_seq_repeat:
        writer.emit_rule("multi_run", "multi_run(Tool) :- tool(Tool).")

    writer.emit_atom("exact_horizon_mode")

    return roots


def _lazy_port_expansion(
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


def _workflow_input_matches_lazy_port(
    ontology: Ontology,
    workflow_input: Mapping[str, tuple[str, ...]],
    port_values_by_dimension: Mapping[str, tuple[str, ...]],
) -> bool:
    for dim, accepted_values in port_values_by_dimension.items():
        actual_values = workflow_input.get(dim, ())
        if not any(
            actual_value in ontology.ancestors_of(required_value)
            for actual_value in actual_values
            for required_value in accepted_values
        ):
            return False
    return True


def _lazy_output_matches_lazy_input(
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


def _compress_lazy_output_choice_values(
    output_values_by_dimension: Mapping[str, tuple[str, ...]],
    bindable_input_ports: tuple[Mapping[str, object], ...],
    goal_port_values: tuple[Mapping[str, tuple[str, ...]], ...],
) -> dict[str, tuple[str, ...]]:
    globally_bindable_input_ports = tuple(
        input_port
        for input_port in bindable_input_ports
        if _lazy_output_matches_lazy_input(
            output_values_by_dimension,
            input_port["port_values_by_dimension"],
        )
    )
    globally_bindable_goal_ids = tuple(
        goal_id
        for goal_id, goal_dims in enumerate(goal_port_values)
        if _lazy_output_matches_lazy_input(output_values_by_dimension, goal_dims)
    )

    default_consumer_signatures_by_dimension: dict[str, set[int]] = defaultdict(set)
    consumer_signatures_by_dimension_value: dict[str, dict[str, set[int]]] = defaultdict(
        lambda: defaultdict(set)
    )
    for input_port in globally_bindable_input_ports:
        signature_id = int(input_port["signature_id"])
        input_dims = input_port["port_values_by_dimension"]
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
            representatives.setdefault((consumer_profile, goal_profile), value)

        compressed[dim] = tuple(representatives.values())

    return compressed


def build_fact_bundle_grounding_opt(
    config: SnakeConfig,
    ontology: Ontology,
    tools: tuple[ToolMode, ...],
) -> FactBundle:
    """Build a solver-ready fact bundle using the grounding-optimised candidate schema.

    Each (tool, input_variant, output_variant) triple is pre-expanded to a flat
    ``tool_candidate`` fact with ``candidate_in`` / ``candidate_out`` facts carrying
    fully-resolved terminal dimension values.  This eliminates the multi-layer choice
    rules and the ``compatible/2`` join from the eligibility check in the ASP encoding.
    """
    writer = _FactWriter()
    roots = _build_common_facts(writer, config, ontology, tools)
    resolver = _ExpansionResolver(ontology, roots, "python")
    tool_labels = {tool.mode_id: tool.label for tool in tools}
    tool_input_signatures = _tool_input_signatures(tools)
    tool_stats: list[ToolExpansionStat] = []

    _cand_offset: dict[str, int] = defaultdict(int)

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
        output_port_variants = tuple(
            tuple(
                resolver.iter_dimension_maps(
                    _normalize_dim_map(port.dimensions),
                    expand_outputs=True,
                )
            )
            for port in tool.outputs
        )
        input_variant_count = _product(len(variants) for variants in input_port_variants) if input_port_variants else 1
        output_variant_count = _product(len(variants) for variants in output_port_variants) if output_port_variants else 1
        tool_stats.append(
            ToolExpansionStat(
                tool_id=tool.mode_id,
                tool_label=tool.label,
                input_ports=len(tool.inputs),
                output_ports=len(tool.outputs),
                input_variant_count=input_variant_count,
                output_variant_count=output_variant_count,
                candidate_count=input_variant_count * output_variant_count,
            )
        )

        base = _cand_offset[tool.mode_id]
        counter = 0
        input_variant_iter = product(*input_port_variants) if input_port_variants else [()]
        for port_specs in input_variant_iter:
            for out_combo in (product(*output_port_variants) if output_port_variants else [()]):
                cand_id = f"{tool.mode_id}_c{base + counter}"
                counter += 1
                writer.emit_fact("tool_candidate", _quote(cand_id), _quote(tool.mode_id))
                for port_idx, dims in enumerate(port_specs):
                    for dim, value in sorted(dims.items()):
                        writer.emit_fact(
                            "candidate_in",
                            _quote(cand_id),
                            str(port_idx),
                            _quote(value),
                            _quote(dim),
                        )
                # out_combo is () (empty tuple) when the tool has no outputs,
                # otherwise a tuple of per-output-port dim dicts.
                for out_idx, out_dims in enumerate(out_combo):
                    for dim, value in sorted(out_dims.items()):
                        writer.emit_fact(
                            "candidate_out",
                            _quote(cand_id),
                            str(out_idx),
                            _quote(value),
                            _quote(dim),
                        )
        _cand_offset[tool.mode_id] += counter

    workflow_input_ids = [f"wf_input_{i}" for i in range(len(config.inputs))]
    facts = writer.text()

    return FactBundle(
        facts=facts,
        fact_count=writer.fact_count,
        tool_labels=tool_labels,
        tool_input_signatures=tool_input_signatures,
        workflow_input_ids=tuple(workflow_input_ids),
        goal_count=len(config.outputs),
        predicate_counts=dict(writer.predicate_counts),
        tool_stats=tuple(tool_stats),
        cache_stats={**ontology.cache_stats(), **resolver.stats()},
        emit_stats=writer.stats(),
    )


def build_fact_bundle_grounding_opt_lazy(
    config: SnakeConfig,
    ontology: Ontology,
    tools: tuple[ToolMode, ...],
) -> FactBundle:
    """Build a diagnostic lazy candidate bundle without full variant expansion."""
    roots = _build_roots(config, ontology)
    resolver = _ExpansionResolver(ontology, roots, "python")
    tool_stats: list[ToolExpansionStat] = []
    candidate_records: list[dict[str, object]] = []

    lazy_offset: dict[str, int] = defaultdict(int)
    _input_sig_to_id: dict[frozenset, int] = {}

    for tool in tools:
        candidate_index = lazy_offset[tool.mode_id]
        candidate_id = f"{tool.mode_id}_lc{candidate_index}"
        lazy_offset[tool.mode_id] += 1

        input_port_value_counts: list[int] = []
        output_port_value_counts: list[int] = []
        input_variant_count = 1
        output_variant_count = 1
        lazy_input_value_count = 0
        lazy_output_value_count = 0
        input_ports: list[dict[str, object]] = []
        output_ports: list[dict[str, object]] = []

        for port_idx, port in enumerate(tool.inputs):
            port_values, port_variant_count = _lazy_port_expansion(
                resolver,
                _normalize_dim_map(port.dimensions),
                expand_outputs=False,
            )
            port_values_by_dimension = _group_port_values_by_dimension(port_values)
            sig = frozenset(port_values)
            if sig not in _input_sig_to_id:
                _input_sig_to_id[sig] = len(_input_sig_to_id)
            workflow_input_matches = [
                wf_index
                for wf_index, workflow_input in enumerate(config.inputs)
                if _workflow_input_matches_lazy_port(ontology, workflow_input, port_values_by_dimension)
            ]
            input_ports.append(
                {
                    "port_idx": port_idx,
                    "port_values": port_values,
                    "port_values_by_dimension": port_values_by_dimension,
                    "signature_id": _input_sig_to_id[sig],
                    "workflow_input_matches": workflow_input_matches,
                }
            )
            emitted_count = len(port_values)
            input_port_value_counts.append(emitted_count)
            lazy_input_value_count += emitted_count
            input_variant_count *= port_variant_count

        for port_idx, port in enumerate(tool.outputs):
            declared_dims = _normalize_dim_map(port.dimensions)
            port_values, port_variant_count = _lazy_port_expansion(
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
                }
            )
            emitted_count = sum(len(values) for values in port_values_by_dimension.values())
            output_port_value_counts.append(emitted_count)
            lazy_output_value_count += emitted_count
            output_variant_count *= port_variant_count

        tool_stats.append(
            ToolExpansionStat(
                tool_id=tool.mode_id,
                tool_label=tool.label,
                input_ports=len(tool.inputs),
                output_ports=len(tool.outputs),
                input_variant_count=input_variant_count,
                output_variant_count=output_variant_count,
                lazy_input_value_count=lazy_input_value_count,
                lazy_output_value_count=lazy_output_value_count,
                lazy_input_port_value_counts=tuple(input_port_value_counts),
                lazy_output_port_value_counts=tuple(output_port_value_counts),
                lazy_cross_product_estimate=input_variant_count * output_variant_count,
            )
        )
        candidate_records.append(
            {
                "tool": tool,
                "candidate_id": candidate_id,
                "input_ports": tuple(input_ports),
                "output_ports": tuple(output_ports),
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

    bindable_pairs: set[tuple[str, int, str, int]] = set()
    reverse_edges: dict[str, set[str]] = defaultdict(set)
    direct_goal_candidates: set[str] = set()

    for record in candidate_records:
        candidate_id = str(record["candidate_id"])
        output_ports = tuple(record["output_ports"])
        for output_port in output_ports:
            output_dims = output_port["port_values_by_dimension"]
            if any(_lazy_output_matches_lazy_input(output_dims, goal_dims) for goal_dims in goal_port_values):
                direct_goal_candidates.add(candidate_id)

    for producer_record in candidate_records:
        producer_candidate = str(producer_record["candidate_id"])
        for output_port in tuple(producer_record["output_ports"]):
            producer_port = int(output_port["port_idx"])
            output_dims = output_port["port_values_by_dimension"]
            for consumer_record in candidate_records:
                consumer_candidate = str(consumer_record["candidate_id"])
                for input_port in tuple(consumer_record["input_ports"]):
                    consumer_port = int(input_port["port_idx"])
                    input_dims = input_port["port_values_by_dimension"]
                    if _lazy_output_matches_lazy_input(output_dims, input_dims):
                        bindable_pairs.add(
                            (producer_candidate, producer_port, consumer_candidate, consumer_port)
                        )
                        reverse_edges[consumer_candidate].add(producer_candidate)

    relevant_candidates = set(direct_goal_candidates)
    frontier = list(direct_goal_candidates)
    while frontier:
        consumer_candidate = frontier.pop()
        for producer_candidate in reverse_edges.get(consumer_candidate, set()):
            if producer_candidate in relevant_candidates:
                continue
            relevant_candidates.add(producer_candidate)
            frontier.append(producer_candidate)

    relevant_records = [
        record
        for record in candidate_records
        if str(record["candidate_id"]) in relevant_candidates
    ]
    relevant_input_ports = tuple(
        input_port
        for record in relevant_records
        for input_port in tuple(record["input_ports"])
    )
    for record in relevant_records:
        compressed_output_ports: list[dict[str, object]] = []
        for output_port in tuple(record["output_ports"]):
            compressed_output_ports.append(
                {
                    **output_port,
                    "port_values_by_dimension": _compress_lazy_output_choice_values(
                        output_port["port_values_by_dimension"],
                        relevant_input_ports,
                        goal_port_values_tuple,
                    ),
                }
            )
        record["output_ports"] = tuple(compressed_output_ports)
    relevant_tools = tuple(record["tool"] for record in relevant_records)

    writer = _FactWriter()
    _build_common_facts(writer, config, ontology, relevant_tools)
    tool_labels = {tool.mode_id: tool.label for tool in relevant_tools}
    tool_input_signatures = _tool_input_signatures(relevant_tools)

    for record in relevant_records:
        tool = record["tool"]
        candidate_id = str(record["candidate_id"])
        writer.emit_fact("lazy_tool_candidate", _quote(candidate_id), _quote(tool.mode_id))

        for input_port in tuple(record["input_ports"]):
            port_idx = int(input_port["port_idx"])
            writer.emit_fact(
                "lazy_candidate_input_port",
                _quote(candidate_id),
                str(port_idx),
            )
            for dim, value in input_port["port_values"]:
                writer.emit_fact(
                    "lazy_candidate_input_value",
                    _quote(candidate_id),
                    str(port_idx),
                    _quote(value),
                    _quote(dim),
                )
            writer.emit_fact(
                "lazy_candidate_input_signature_id",
                _quote(candidate_id),
                str(port_idx),
                str(input_port["signature_id"]),
            )
            for wf_index in input_port["workflow_input_matches"]:
                writer.emit_fact(
                    "lazy_initial_bindable",
                    _quote(candidate_id),
                    str(port_idx),
                    _quote(f"wf_input_{wf_index}"),
                )

        for output_port in tuple(record["output_ports"]):
            port_idx = int(output_port["port_idx"])
            writer.emit_fact(
                "lazy_candidate_output_port",
                _quote(candidate_id),
                str(port_idx),
            )
            for dim, declared_values in output_port["declared_dims"].items():
                for declared_value in declared_values:
                    writer.emit_fact(
                        "lazy_candidate_output_declared_type",
                        _quote(candidate_id),
                        str(port_idx),
                        _quote(declared_value),
                        _quote(dim),
                    )
            for dim, values in sorted(output_port["port_values_by_dimension"].items()):
                if len(values) == 1:
                    writer.emit_fact(
                        "lazy_candidate_output_singleton",
                        _quote(candidate_id),
                        str(port_idx),
                        _quote(values[0]),
                        _quote(dim),
                    )
                else:
                    for value in values:
                        writer.emit_fact(
                            "lazy_candidate_output_choice_value",
                            _quote(candidate_id),
                            str(port_idx),
                            _quote(value),
                            _quote(dim),
                        )

    for producer_candidate, producer_port, consumer_candidate, consumer_port in sorted(bindable_pairs):
        if producer_candidate not in relevant_candidates or consumer_candidate not in relevant_candidates:
            continue
        writer.emit_fact(
            "lazy_candidate_port_bindable",
            _quote(producer_candidate),
            str(producer_port),
            _quote(consumer_candidate),
            str(consumer_port),
        )

    workflow_input_ids = [f"wf_input_{i}" for i in range(len(config.inputs))]
    facts = writer.text()

    return FactBundle(
        facts=facts,
        fact_count=writer.fact_count,
        tool_labels=tool_labels,
        tool_input_signatures=tool_input_signatures,
        workflow_input_ids=tuple(workflow_input_ids),
        goal_count=len(config.outputs),
        predicate_counts=dict(writer.predicate_counts),
        tool_stats=tuple(tool_stats),
        cache_stats={**ontology.cache_stats(), **resolver.stats()},
        emit_stats=writer.stats(),
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
    tool_labels = {tool.mode_id: tool.label for tool in tools}
    tool_input_signatures = _tool_input_signatures(tools)
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
        output_port_variants = tuple(
            tuple(
                resolver.iter_dimension_maps(
                    _normalize_dim_map(port.dimensions),
                    expand_outputs=True,
                )
            )
            for port in tool.outputs
        )
        input_variant_count = _product(len(variants) for variants in input_port_variants) if input_port_variants else 1
        output_variant_count = sum(len(expanded_ports) for expanded_ports in output_port_variants)
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
        for output_index, expanded_ports in enumerate(output_port_variants):
            for variant_index, dims in enumerate(expanded_ports):
                output_id = f"{tool.mode_id}_out_{o_base + output_index}_{variant_index}"
                port_id = f"{output_id}_port_0"
                writer.emit_fact("tool_output", _quote(tool.mode_id), _quote(output_id))
                writer.emit_fact("output_port", _quote(output_id), _quote(port_id))
                for dim, value in sorted(dims.items()):
                    writer.emit_fact(
                        "dimension",
                        _quote(port_id),
                        f"({_quote(value)}, {_quote(dim)})",
                    )
        _output_offset[tool.mode_id] += len(output_port_variants)

    workflow_input_ids = [f"wf_input_{i}" for i in range(len(config.inputs))]
    facts = writer.text()

    return FactBundle(
        facts=facts,
        fact_count=writer.fact_count,
        tool_labels=tool_labels,
        tool_input_signatures=tool_input_signatures,
        workflow_input_ids=tuple(workflow_input_ids),
        goal_count=len(config.outputs),
        predicate_counts=dict(writer.predicate_counts),
        tool_stats=tuple(tool_stats),
        cache_stats={**ontology.cache_stats(), **resolver.stats()},
        emit_stats=writer.stats(),
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
    _build_common_facts(writer, config, ontology, tools)
    tool_labels = {tool.mode_id: tool.label for tool in tools}
    tool_input_signatures = _tool_input_signatures(tools)
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

    workflow_input_ids = [f"wf_input_{i}" for i in range(len(config.inputs))]
    facts = writer.text()

    return FactBundle(
        facts=facts,
        fact_count=writer.fact_count,
        tool_labels=tool_labels,
        tool_input_signatures=tool_input_signatures,
        workflow_input_ids=tuple(workflow_input_ids),
        goal_count=len(config.outputs),
        predicate_counts=dict(writer.predicate_counts),
        tool_stats=tuple(tool_stats),
        cache_stats=dict(ontology.cache_stats()),
        emit_stats=writer.stats(),
    )
