"""Lazy translation entrypoints for multi-shot-lazy mode."""

from __future__ import annotations

from time import perf_counter

from .lazy_optimization import LazyOptimizationResult, optimize_lazy_candidates
from .models import FactBundle, SnakeConfig, ToolMode
from .ontology import Ontology
from . import translator_core as core


def build_lazy_fact_bundle(
    config: SnakeConfig,
    ontology: Ontology,
    tools: tuple[ToolMode, ...],
) -> FactBundle:
    """Build lazy facts using a dedicated optimization layer."""

    optimization = optimize_lazy_candidates(config, ontology, tools)

    writer = core._FactWriter()
    core._build_common_facts(writer, config, ontology, optimization.relevant_tools)
    core._emit_lazy_constraints(writer, config, ontology, tools)

    for candidate_id in optimization.query_goal_candidates:
        writer.emit_fact("lazy_query_goal_candidate", core._quote(candidate_id))
    for tool_id in optimization.query_goal_tools:
        writer.emit_fact("lazy_query_goal_tool", core._quote(tool_id))
    for candidate_id, goal_distance in sorted(optimization.min_goal_distance_by_candidate.items()):
        writer.emit_fact("lazy_candidate_goal_distance", core._quote(candidate_id), str(goal_distance))
    for step_index, candidate_ids in optimization.allowed_candidates_by_step.items():
        for candidate_id in candidate_ids:
            writer.emit_fact(
                "lazy_candidate_allowed_at_step",
                core._quote(candidate_id),
                str(step_index),
            )
    for step_index, tool_ids in optimization.allowed_tools_by_step.items():
        for tool_id in tool_ids:
            writer.emit_fact(
                "lazy_step_allowed_tool",
                core._quote(tool_id),
                str(step_index),
            )

    emit_start = perf_counter()

    for record in optimization.relevant_records:
        tool = record["tool"]
        candidate_id = str(record["candidate_id"])
        writer.emit_fact("lazy_tool_candidate", core._quote(candidate_id), core._quote(tool.mode_id))

        for input_port in tuple(record["input_ports"]):
            port_idx = int(input_port["port_idx"])
            writer.emit_fact(
                "lazy_candidate_input_port",
                core._quote(candidate_id),
                str(port_idx),
            )
            writer.emit_fact(
                "lazy_candidate_input_signature_id",
                core._quote(candidate_id),
                str(port_idx),
                str(input_port["signature_id"]),
            )
            for wf_index in input_port["workflow_input_matches"]:
                writer.emit_fact(
                    "lazy_initial_bindable",
                    core._quote(candidate_id),
                    str(port_idx),
                    core._quote(f"wf_input_{wf_index}"),
                )

        for output_port in tuple(record["output_ports"]):
            port_idx = int(output_port["port_idx"])
            writer.emit_fact(
                "lazy_candidate_output_port",
                core._quote(candidate_id),
                str(port_idx),
            )
            writer.emit_fact(
                "lazy_candidate_output_multiplicity",
                core._quote(candidate_id),
                str(port_idx),
                str(int(output_port.get("multiplicity", 1))),
            )
            for dim, declared_values in output_port["declared_dims"].items():
                for declared_value in declared_values:
                    writer.emit_fact(
                        "lazy_candidate_output_declared_type",
                        core._quote(candidate_id),
                        str(port_idx),
                        core._quote(declared_value),
                        core._quote(dim),
                    )
            for dim, values in sorted(output_port["port_values_by_dimension"].items()):
                if len(values) == 1:
                    writer.emit_fact(
                        "lazy_candidate_output_singleton",
                        core._quote(candidate_id),
                        str(port_idx),
                        core._quote(values[0]),
                        core._quote(dim),
                    )
                else:
                    for value in values:
                        writer.emit_fact(
                            "lazy_candidate_output_choice_value",
                            core._quote(candidate_id),
                            str(port_idx),
                            core._quote(value),
                            core._quote(dim),
                        )
        writer.emit_fact(
            "lazy_candidate_total_output_multiplicity",
            core._quote(candidate_id),
            str(sum(int(output_port.get("multiplicity", 1)) for output_port in tuple(record["output_ports"]))),
        )

    for signature_id, support_class_id in sorted(optimization.signature_support_class_by_id.items()):
        writer.emit_fact(
            "lazy_signature_support_class",
            str(signature_id),
            str(support_class_id),
        )
    for support_class_id, producer_ports in sorted(optimization.support_class_bindable_ports.items()):
        for producer_candidate, producer_port in producer_ports:
            writer.emit_fact(
                "lazy_support_class_bindable_producer_port",
                str(support_class_id),
                core._quote(producer_candidate),
                str(producer_port),
            )
    for signature_id, category_profiles in sorted(optimization.signature_profiles_by_id.items()):
        for dim, (profile_id, _values) in sorted(category_profiles.items()):
            writer.emit_fact(
                "lazy_signature_profile",
                str(signature_id),
                core._quote(dim),
                str(profile_id),
            )
    for profile_id, values in sorted(optimization.profile_accepts_by_id.items()):
        for value in values:
            writer.emit_fact(
                "lazy_profile_accepts",
                str(profile_id),
                core._quote(value),
            )

    emit_elapsed = perf_counter() - emit_start
    print(
        f"  lazy builder phases: "
        f"goal_check={optimization.phase_timings['goal_check']:.2f}s "
        f"bindable_pairs={optimization.phase_timings['bindable_pairs']:.2f}s "
        f"bfs_pruning={optimization.phase_timings['bfs_pruning']:.2f}s "
        f"compression={optimization.phase_timings['compression']:.2f}s "
        f"step_indexing={optimization.phase_timings['step_indexing']:.2f}s "
        f"fact_emission={emit_elapsed:.2f}s"
    )

    return core._finalize_fact_bundle(
        writer,
        config=config,
        tools=optimization.relevant_tools,
        tool_stats=list(optimization.tool_stats),
        cache_stats={**ontology.cache_stats(), **optimization.cache_stats},
        earliest_solution_step=optimization.earliest_solution_step,
    )


__all__ = (
    "build_lazy_fact_bundle",
    "optimize_lazy_candidates",
    "LazyOptimizationResult",
    "FactBundle",
    "SnakeConfig",
    "ToolMode",
    "Ontology",
)
