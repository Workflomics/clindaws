"""Compressed-candidate translation entrypoints for optimized multi-shot runs."""

from __future__ import annotations
from clindaws.translators.utils import _quote
from clindaws.translators.fact_writer import _FactWriter
from clindaws.translators.builder import _build_common_facts, _finalize_fact_bundle
from clindaws.translators.constraints import _emit_dynamic_constraints

from time import perf_counter

from clindaws.execution.compressed_candidate_optimization import (
    CompressedCandidateOptimizationResult,
    optimize_compressed_candidates,
)
from clindaws.core.models import FactBundle, SnakeConfig, ToolMode
from clindaws.core.ontology import Ontology


def build_compressed_candidate_fact_bundle(
    config: SnakeConfig,
    ontology: Ontology,
    tools: tuple[ToolMode, ...],
    *,
    max_workers: int = 1,
) -> FactBundle:
    """Build compressed-candidate facts using a dedicated optimization layer."""

    optimization = optimize_compressed_candidates(config, ontology, tools, max_workers=max_workers)

    writer = _FactWriter()
    _build_common_facts(writer, config, ontology, optimization.relevant_tools)
    _emit_dynamic_constraints(writer, config, ontology, tools)

    for candidate_id in optimization.query_goal_candidates:
        writer.emit_fact("dynamic_query_goal_candidate", _quote(candidate_id))
    for tool_id in optimization.query_goal_tools:
        writer.emit_fact("dynamic_query_goal_tool", _quote(tool_id))
    for candidate_id, goal_distance in sorted(optimization.min_goal_distance_by_candidate.items()):
        writer.emit_fact("dynamic_candidate_goal_distance", _quote(candidate_id), str(goal_distance))
    for step_index, candidate_ids in optimization.allowed_candidates_by_step.items():
        for candidate_id in candidate_ids:
            writer.emit_fact(
                "dynamic_candidate_allowed_at_step",
                _quote(candidate_id),
                str(step_index),
            )
    for step_index, tool_ids in optimization.allowed_tools_by_step.items():
        for tool_id in tool_ids:
            writer.emit_fact(
                "dynamic_step_allowed_tool",
                _quote(tool_id),
                str(step_index),
            )

    emit_start = perf_counter()

    for record in optimization.relevant_records:
        tool = record["tool"]
        candidate_id = str(record["candidate_id"])
        writer.emit_fact("dynamic_tool_candidate", _quote(candidate_id), _quote(tool.mode_id))

        for input_port in tuple(record["input_ports"]):
            port_idx = int(input_port["port_idx"])
            signature_id = int(input_port["signature_id"])
            writer.emit_fact(
                "dynamic_candidate_input_port",
                _quote(candidate_id),
                str(port_idx),
            )
            writer.emit_fact(
                "dynamic_candidate_input_signature_id",
                _quote(candidate_id),
                str(port_idx),
                str(signature_id),
            )
            support_class_id = optimization.signature_support_class_by_id.get(signature_id)
            if support_class_id is not None:
                writer.emit_fact(
                    "compressed_candidate_input_support_class",
                    _quote(candidate_id),
                    str(port_idx),
                    str(support_class_id),
                )
                # Helper for single-shot mode
                writer.emit_fact(
                    "precomputed_candidate_input_support_class",
                    _quote(candidate_id),
                    str(port_idx),
                    str(support_class_id),
                )

            for dim, (profile_id, _values) in sorted(
                optimization.signature_profiles_by_id.get(signature_id, {}).items()
            ):
                writer.emit_fact(
                    "compressed_candidate_input_profile",
                    _quote(candidate_id),
                    str(port_idx),
                    _quote(dim),
                    str(profile_id),
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
            output_id = optimization.candidate_output_id_map.get((candidate_id, port_idx))
            writer.emit_fact(
                "dynamic_candidate_output_port",
                _quote(candidate_id),
                str(port_idx),
            )
            if output_id is not None:
                writer.emit_fact(
                    "dynamic_candidate_output_id",
                    _quote(candidate_id),
                    str(port_idx),
                    str(output_id),
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

    for signature_id, support_class_id in sorted(optimization.signature_support_class_by_id.items()):
        writer.emit_fact(
            "dynamic_signature_support_class",
            str(signature_id),
            str(support_class_id),
        )
    for support_class_id, producer_ports in sorted(optimization.support_class_bindable_ports.items()):
        for producer_candidate, producer_port in producer_ports:
            writer.emit_fact(
                "dynamic_support_class_bindable_producer_port",
                str(support_class_id),
                _quote(producer_candidate),
                str(producer_port),
            )
    for signature_id, category_profiles in sorted(optimization.signature_profiles_by_id.items()):
        for dim, (profile_id, _values) in sorted(category_profiles.items()):
            writer.emit_fact(
                "dynamic_signature_profile",
                str(signature_id),
                _quote(dim),
                str(profile_id),
            )
    for profile_id, values in sorted(optimization.profile_accepts_by_id.items()):
        for value in values:
            writer.emit_fact(
                "dynamic_profile_accepts",
                str(profile_id),
                _quote(value),
            )
    for goal_id, goal_profiles in sorted(optimization.goal_requirement_profiles_by_id.items()):
        for requirement_id, category, profile_id in goal_profiles:
            writer.emit_fact(
                "compressed_goal_requirement_profile",
                str(goal_id),
                str(requirement_id),
                _quote(category),
                str(profile_id),
            )

    emit_elapsed = perf_counter() - emit_start
    print(
        f"  compressed-candidate builder phases: "
        f"goal_check={optimization.phase_timings['goal_check']:.2f}s "
        f"bindable_pairs={optimization.phase_timings['bindable_pairs']:.2f}s "
        f"bfs_pruning={optimization.phase_timings['bfs_pruning']:.2f}s "
        f"compression={optimization.phase_timings['compression']:.2f}s "
        f"step_indexing={optimization.phase_timings['step_indexing']:.2f}s "
        f"fact_emission={emit_elapsed:.2f}s"
    )

    return _finalize_fact_bundle(
        writer,
        config=config,
        tools=optimization.relevant_tools,
        tool_stats=list(optimization.tool_stats),
        cache_stats={**ontology.cache_stats(), **optimization.cache_stats},
        backend_stats={
            "compressed_candidate_phase_timings": dict(sorted(optimization.phase_timings.items())),
            "translation_workers": max_workers,
        },
        earliest_solution_step=optimization.earliest_solution_step,
    )


__all__ = (
    "build_compressed_candidate_fact_bundle",
    "optimize_compressed_candidates",
    "CompressedCandidateOptimizationResult",
    "FactBundle",
    "SnakeConfig",
    "ToolMode",
    "Ontology",
)
