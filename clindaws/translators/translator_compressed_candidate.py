"""Optimized-candidate translation entrypoints for optimized multi-shot runs.

This translator builds the fact surface consumed by the optimized
``multi_shot_optimized_candidate`` encodings. Most of the heavy work happens in
the optimization layer first; this module mainly turns those precomputed
records, support classes, and step windows into emitted facts.
"""

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
    """Build optimized-candidate facts using a dedicated optimization layer.

    The emitted facts are intentionally close to the optimized ASP schema so
    step grounding can work from preindexed candidates, bindability classes, and
    retained output-choice values instead of rediscovering them from scratch.
    """

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
    for step_index, tool_ids in optimization.must_run_tools_by_step.items():
        for tool_id in tool_ids:
            writer.emit_fact(
                "dynamic_must_run_tool_at_step",
                _quote(tool_id),
                str(step_index),
            )
    for step_index, candidate_ids in optimization.must_run_candidates_by_step.items():
        for candidate_id in candidate_ids:
            writer.emit_fact(
                "dynamic_must_run_candidate_at_step",
                _quote(candidate_id),
                str(step_index),
            )
    for step_index, associations in optimization.forced_associations_by_step.items():
        for candidate_id, port_idx, producer_candidate, producer_port in associations:
            writer.emit_fact(
                "dynamic_forced_produced_bind_at_step",
                _quote(candidate_id),
                str(port_idx),
                _quote(producer_candidate),
                str(producer_port),
                str(step_index),
            )
    for tool_id in optimization.must_run_tools_global:
        writer.emit_fact(
            "dynamic_must_run_tool_global",
            _quote(tool_id),
        )
    for candidate_id in optimization.must_run_candidates_global:
        writer.emit_fact(
            "dynamic_must_run_candidate_global",
            _quote(candidate_id),
        )
    for candidate_id, port_idx, producer_candidate, producer_port in optimization.forced_associations_global:
        writer.emit_fact(
            "dynamic_forced_produced_bind",
            _quote(candidate_id),
            str(port_idx),
            _quote(producer_candidate),
            str(producer_port),
        )

    # Fact emission is kept separate from optimization timing so translation
    # summaries can distinguish Python search/compression cost from plain output
    # serialization cost.
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
                    "optimized_candidate_input_support_class",
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
            association_class_id = optimization.association_class_by_input.get((candidate_id, port_idx))
            if association_class_id is not None:
                writer.emit_fact(
                    "dynamic_candidate_input_association_class",
                    _quote(candidate_id),
                    str(port_idx),
                    str(association_class_id),
                )

            for dim, (profile_id, _values) in sorted(
                optimization.signature_profiles_by_id.get(signature_id, {}).items()
            ):
                writer.emit_fact(
                    "optimized_candidate_input_profile",
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
    for association_class_id, producer_ports in sorted(optimization.association_class_bindable_ports.items()):
        for producer_candidate, producer_port in producer_ports:
            writer.emit_fact(
                "dynamic_association_class_bindable_producer_port",
                str(association_class_id),
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
                "optimized_goal_requirement_profile",
                str(goal_id),
                str(requirement_id),
                _quote(category),
                str(profile_id),
            )

    emit_elapsed = perf_counter() - emit_start
    print(
        f"  optimized-candidate builder phases: "
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
            "optimized_candidate_phase_timings": dict(sorted(optimization.phase_timings.items())),
            "translation_workers": max_workers,
            "smart_expansion": {
                "earliest_feasible_horizon": optimization.earliest_solution_step,
                "structural_horizon_skip_count": optimization.structural_horizon_skip_count,
                "structural_probe_horizons": list(optimization.structural_probe_horizons),
                "query_control_mode": "assumptions",
                "must_run_tools_global": len(optimization.must_run_tools_global),
                "must_run_candidates_global": len(optimization.must_run_candidates_global),
                "must_run_tool_steps": len(optimization.must_run_tools_by_step),
                "must_run_candidate_steps": len(optimization.must_run_candidates_by_step),
                "forced_associations_global": len(optimization.forced_associations_global),
                "forced_association_steps": len(optimization.forced_associations_by_step),
                "association_class_count": len(optimization.association_class_bindable_ports),
                "fixpoint_rounds": optimization.fixpoint_rounds,
            },
        },
        earliest_solution_step=optimization.earliest_solution_step,
    )


__all__ = (
    "build_optimized_candidate_fact_bundle",
    "build_compressed_candidate_fact_bundle",
    "optimize_compressed_candidates",
    "CompressedCandidateOptimizationResult",
    "FactBundle",
    "SnakeConfig",
    "ToolMode",
    "Ontology",
)


def build_optimized_candidate_fact_bundle(
    config: SnakeConfig,
    ontology: Ontology,
    tools: tuple[ToolMode, ...],
    *,
    max_workers: int = 1,
) -> FactBundle:
    """Canonical optimized-candidate translation entrypoint."""

    return build_compressed_candidate_fact_bundle(
        config,
        ontology,
        tools,
        max_workers=max_workers,
    )
