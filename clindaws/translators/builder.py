"""Fact-bundle builders for the direct translation families.

This module turns normalized config/ontology/tool data into the ground facts
consumed by the direct ASP encodings. It provides:

- shared base facts used by all direct bundles,
- the APE-style multi-shot/direct surface used by public multi-shot and the
  current single-shot runtime,
- helper metadata that the Python runner uses for reporting and reconstruction.

Compressed-candidate optimized translation lives in its own translator module.
"""

from __future__ import annotations
from collections import defaultdict
from itertools import product

from clindaws.core.models import FactBundle, SnakeConfig, ToolExpansionStat, ToolMode
from clindaws.core.ontology import Ontology

from clindaws.translators.utils import _product, _normalize_dim_map, _quote
from clindaws.translators.signatures import _tool_input_signatures
from clindaws.translators.candidates import _compute_ape_multi_shot_earliest_solution_step
from clindaws.translators.fact_writer import _FactWriter
from clindaws.translators.constraints import _emit_dynamic_constraints
from clindaws.translators.resolvers import _ExpansionResolver



def _bundle_metadata(
    config: SnakeConfig,
    tools: tuple[ToolMode, ...],
) -> tuple[dict[str, str], dict[str, tuple[tuple[tuple[str, tuple[str, ...]], ...], ...]], tuple[str, ...]]:
    """Return Python-side metadata reused after solving.

    These structures are not part of the ASP fact text itself, but the runner
    needs them later to reconstruct workflow candidates and report tool labels.
    """
    tool_labels = {tool.mode_id: tool.label for tool in tools}
    tool_input_signatures = _tool_input_signatures(tools)

    signatures_by_label: dict[str, set[tuple[tuple[tuple[str, tuple[str, ...]], ...], ...]]] = defaultdict(set)
    for tool in tools:
        signatures_by_label[tool.label].add(tool_input_signatures[tool.mode_id])
    for label, signatures in signatures_by_label.items():
        if len(signatures) == 1:
            tool_input_signatures.setdefault(label, next(iter(signatures)))

    return (
        tool_labels,
        tool_input_signatures,
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
    """Build disjoint ontology subtrees for the configured data dimensions."""
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
        # Workflow inputs are emitted as timestep-0 available artifacts so both
        # direct multi-shot and the current single-shot runtime can bootstrap
        # artifact propagation from the same initial surface.
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
    # This builder is the direct baseline for public multi-shot and also feeds
    # the current public single-shot runtime, which changes solver strategy but
    # not the underlying fact surface.

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
