"""Optimized-candidate translation entrypoints for optimized multi-shot runs.

This translator builds the fact surface consumed by the optimized
``multi_shot_optimized_candidate`` encodings. Most of the heavy work happens in
the optimization layer first; this module mainly turns those precomputed
records, support classes, and step windows into emitted facts.
"""

from __future__ import annotations
from clindaws.translators.utils import _quote
from clindaws.translators.fact_writer import _FactWriter
from clindaws.translators.builder import _build_common_facts, _finalize_fact_bundle, _build_roots
from clindaws.translators.constraints import _emit_dynamic_constraints
from clindaws.translators.ports import (
    _artifact_satisfies_port_requirements,
    _artifact_profile_terminal_sets,
    _port_requirement_terminal_sets,
)

from time import perf_counter

from clindaws.execution.compressed_candidate_optimization import (
    CompressedCandidateOptimizationResult,
    optimize_compressed_candidates,
)
from clindaws.core.models import FactBundle, SnakeConfig, ToolMode
from clindaws.core.ontology import Ontology


def _candidate_satisfies_goal(
    goal_requirements: dict[str, tuple[frozenset[str], ...]],
    artifact_profile: dict[str, frozenset[str]],
) -> bool:
    return _artifact_satisfies_port_requirements(artifact_profile, goal_requirements)


def _goal_support_goal_stats(
    *,
    config: SnakeConfig,
    ontology: Ontology,
    candidate_ids_by_horizon: dict[int, tuple[str, ...]],
    relevant_records: tuple[dict[str, object], ...],
) -> tuple[dict[int, int], dict[int, tuple[int, ...]]]:
    roots = _build_roots(config, ontology)
    goal_requirements_by_id = {
        goal_id: _port_requirement_terminal_sets(ontology, roots, goal_item)
        for goal_id, goal_item in enumerate(config.outputs)
    }
    candidate_goal_ids: dict[str, frozenset[int]] = {}
    for record in relevant_records:
        candidate_id = str(record["candidate_id"])
        covered_goals: set[int] = set()
        for output_port in tuple(record["output_ports"]):
            artifact_profile = _artifact_profile_terminal_sets(
                ontology,
                roots,
                output_port["port_values_by_dimension"],
            )
            for goal_id, goal_requirements in goal_requirements_by_id.items():
                if goal_id in covered_goals:
                    continue
                if _candidate_satisfies_goal(
                    goal_requirements,
                    artifact_profile,
                ):
                    covered_goals.add(goal_id)
        candidate_goal_ids[candidate_id] = frozenset(covered_goals)

    goal_support_goal_counts_by_horizon: dict[int, int] = {}
    goal_support_missing_goals_by_horizon: dict[int, tuple[int, ...]] = {}

    for horizon, candidate_ids in sorted(candidate_ids_by_horizon.items()):
        covered_goals: set[int] = set()
        for candidate_id in candidate_ids:
            covered_goals.update(candidate_goal_ids.get(candidate_id, frozenset()))
        goal_support_goal_counts_by_horizon[horizon] = len(covered_goals)
        goal_support_missing_goals_by_horizon[horizon] = tuple(
            goal_id
            for goal_id in range(len(config.outputs))
            if goal_id not in covered_goals
        )

    return goal_support_goal_counts_by_horizon, goal_support_missing_goals_by_horizon


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
    for candidate_id, min_step in sorted(optimization.min_step_by_candidate.items()):
        writer.emit_fact("dynamic_candidate_min_step", _quote(candidate_id), str(min_step))
    for candidate_id, max_step in sorted(optimization.max_step_by_candidate.items()):
        writer.emit_fact("dynamic_candidate_max_step", _quote(candidate_id), str(max_step))
    earliest_solve_horizon = optimization.earliest_solution_step
    for horizon, candidate_ids in sorted(optimization.goal_support_candidates_by_horizon.items()):
        if horizon < earliest_solve_horizon:
            continue
        for candidate_id in candidate_ids:
            writer.emit_fact(
                "dynamic_goal_support_candidate_at_horizon",
                _quote(candidate_id),
                str(horizon),
            )
    for horizon, tool_ids in sorted(optimization.goal_support_tools_by_horizon.items()):
        if horizon < earliest_solve_horizon:
            continue
        for tool_id in tool_ids:
            writer.emit_fact(
                "dynamic_goal_support_tool_at_horizon",
                _quote(tool_id),
                str(horizon),
            )
    for horizon, input_ports in sorted(optimization.goal_support_inputs_by_horizon.items()):
        if horizon < earliest_solve_horizon:
            continue
        for candidate_id, port_idx in input_ports:
            writer.emit_fact(
                "dynamic_goal_support_input_at_horizon",
                _quote(candidate_id),
                str(port_idx),
                str(horizon),
            )
    for horizon, candidate_ids in sorted(optimization.structurally_supportable_candidates_by_horizon.items()):
        if horizon < earliest_solve_horizon:
            continue
        for candidate_id in candidate_ids:
            writer.emit_fact(
                "dynamic_structurally_supportable_candidate_at_horizon",
                _quote(candidate_id),
                str(horizon),
            )
    for horizon, input_ports in sorted(optimization.structurally_unsupported_inputs_by_horizon.items()):
        if horizon < earliest_solve_horizon:
            continue
        for candidate_id, port_idx in input_ports:
            writer.emit_fact(
                "dynamic_structurally_unsupported_input_at_horizon",
                _quote(candidate_id),
                str(port_idx),
                str(horizon),
            )
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

    output_value_class_by_key: dict[tuple[tuple[str, tuple[str, ...]], ...], int] = {}
    emitted_output_value_classes: set[int] = set()

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
            multiplicity = int(output_port.get("multiplicity", 1))
            writer.emit_fact(
                "dynamic_candidate_output_port",
                _quote(candidate_id),
                str(port_idx),
            )

            writer.emit_fact(
                "dynamic_candidate_output_multiplicity",
                _quote(candidate_id),
                str(port_idx),
                str(multiplicity),
            )
            if multiplicity == 1:
                writer.emit_fact(
                    "dynamic_candidate_output_single_use",
                    _quote(candidate_id),
                    str(port_idx),
                )
            elif multiplicity > 1:
                writer.emit_fact(
                    "dynamic_candidate_output_multi_use",
                    _quote(candidate_id),
                    str(port_idx),
                    str(multiplicity),
                )
            choice_values_by_dimension = {
                str(dim): tuple(str(value) for value in values)
                for dim, values in sorted(output_port["port_values_by_dimension"].items())
                if len(values) > 1
            }
            if choice_values_by_dimension:
                value_class_key = tuple(
                    sorted(
                        (dim, tuple(values))
                        for dim, values in choice_values_by_dimension.items()
                    )
                )
                value_class_id = output_value_class_by_key.setdefault(
                    value_class_key,
                    len(output_value_class_by_key),
                )
                writer.emit_fact(
                    "dynamic_candidate_output_value_class",
                    _quote(candidate_id),
                    str(port_idx),
                    str(value_class_id),
                )
                if value_class_id not in emitted_output_value_classes:
                    emitted_output_value_classes.add(value_class_id)
                    for dim, values in value_class_key:
                        for value in values:
                            writer.emit_fact(
                                "dynamic_output_value_class_choice_value",
                                str(value_class_id),
                                _quote(value),
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
    # Solver skips any horizon not in structural_probe_horizons (solver.py:1214),
    # so check-surface emissions above the max probe horizon are dead weight.
    max_probe_horizon = (
        max(optimization.structural_probe_horizons)
        if optimization.structural_probe_horizons
        else None
    )
    for horizon, output_categories in sorted(optimization.check_relevant_output_categories_by_horizon.items()):
        if horizon < earliest_solve_horizon:
            continue
        if max_probe_horizon is not None and horizon > max_probe_horizon:
            continue
        for candidate_id, port_idx, category in output_categories:
            writer.emit_fact(
                "dynamic_check_relevant_output_category_at_horizon",
                _quote(candidate_id),
                str(port_idx),
                _quote(category),
                str(horizon),
            )
    for horizon, output_profile_classes in sorted(optimization.check_required_profile_classes_by_horizon.items()):
        if horizon < earliest_solve_horizon:
            continue
        if max_probe_horizon is not None and horizon > max_probe_horizon:
            continue
        for candidate_id, port_idx, category, profile_class_id in output_profile_classes:
            writer.emit_fact(
                "dynamic_check_required_profile_class_at_horizon",
                _quote(candidate_id),
                str(port_idx),
                _quote(category),
                str(profile_class_id),
                str(horizon),
            )
    for horizon, output_values in sorted(optimization.check_output_values_by_horizon.items()):
        if horizon < earliest_solve_horizon:
            continue
        if max_probe_horizon is not None and horizon > max_probe_horizon:
            continue
        for candidate_id, port_idx, category, value in output_values:
            writer.emit_fact(
                "dynamic_check_output_value_at_horizon",
                _quote(candidate_id),
                str(port_idx),
                _quote(category),
                str(horizon),
                _quote(value),
            )
    for profile_class_id, profile_ids in sorted(optimization.check_profile_class_members.items()):
        for profile_id in profile_ids:
            writer.emit_fact(
                "dynamic_check_profile_class_member",
                str(profile_class_id),
                str(profile_id),
            )

    emit_elapsed = perf_counter() - emit_start
    goal_support_goal_counts_by_horizon, goal_support_missing_goals_by_horizon = (
        _goal_support_goal_stats(
            config=config,
            ontology=ontology,
            candidate_ids_by_horizon=optimization.goal_support_candidates_by_horizon,
            relevant_records=optimization.relevant_records,
        )
    )
    supportable_goal_counts_by_horizon, supportable_missing_goals_by_horizon = (
        _goal_support_goal_stats(
            config=config,
            ontology=ontology,
            candidate_ids_by_horizon=optimization.structurally_supportable_candidates_by_horizon,
            relevant_records=optimization.relevant_records,
        )
    )
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
                "available_model_blocking_modes": ["candidate_sequence_clause"],
                "smart_expansion_rounds_by_horizon": dict(sorted(optimization.smart_expansion_rounds_by_horizon.items())),
                "goal_support_candidate_counts_by_horizon": {
                    int(horizon): len(candidate_ids)
                    for horizon, candidate_ids in sorted(
                        optimization.goal_support_candidates_by_horizon.items()
                    )
                },
                "goal_support_input_counts_by_horizon": {
                    int(horizon): len(input_ports)
                    for horizon, input_ports in sorted(
                        optimization.goal_support_inputs_by_horizon.items()
                    )
                },
                "check_relevant_output_category_counts_by_horizon": {
                    int(horizon): len(entries)
                    for horizon, entries in sorted(
                        optimization.check_relevant_output_categories_by_horizon.items()
                    )
                },
                "check_required_profile_class_counts_by_horizon": {
                    int(horizon): len(entries)
                    for horizon, entries in sorted(
                        optimization.check_required_profile_classes_by_horizon.items()
                    )
                },
                "check_output_value_counts_by_horizon": {
                    int(horizon): len(entries)
                    for horizon, entries in sorted(
                        optimization.check_output_values_by_horizon.items()
                    )
                },
                "check_profile_class_count": len(optimization.check_profile_class_members),
                "goal_support_tool_counts_by_horizon": {
                    int(horizon): len(tool_ids)
                    for horizon, tool_ids in sorted(
                        optimization.goal_support_tools_by_horizon.items()
                    )
                },
                "goal_support_goal_counts_by_horizon": dict(
                    sorted(goal_support_goal_counts_by_horizon.items())
                ),
                "goal_support_missing_goals_by_horizon": {
                    int(horizon): list(goal_ids)
                    for horizon, goal_ids in sorted(goal_support_missing_goals_by_horizon.items())
                },
                "supportable_candidate_counts_by_horizon": {
                    int(horizon): len(candidate_ids)
                    for horizon, candidate_ids in sorted(
                        optimization.structurally_supportable_candidates_by_horizon.items()
                    )
                },
                "unsupported_input_counts_by_horizon": {
                    int(horizon): len(input_ports)
                    for horizon, input_ports in sorted(
                        optimization.structurally_unsupported_inputs_by_horizon.items()
                    )
                },
                "unsupported_input_samples_by_horizon": {
                    int(horizon): [f"{candidate_id}:{port_idx}" for candidate_id, port_idx in input_ports[:8]]
                    for horizon, input_ports in sorted(
                        optimization.structurally_unsupported_inputs_by_horizon.items()
                    )
                },
                "supportable_goal_counts_by_horizon": dict(
                    sorted(supportable_goal_counts_by_horizon.items())
                ),
                "supportable_missing_goals_by_horizon": {
                    int(horizon): list(goal_ids)
                    for horizon, goal_ids in sorted(supportable_missing_goals_by_horizon.items())
                },
                "input_support_rounds_by_horizon": dict(
                    sorted(optimization.input_support_rounds_by_horizon.items())
                ),
                "must_run_tools_global": len(optimization.must_run_tools_global),
                "must_run_candidates_global": len(optimization.must_run_candidates_global),
                "must_run_tool_steps": len(optimization.must_run_tools_by_step),
                "must_run_candidate_steps": len(optimization.must_run_candidates_by_step),
                "forced_associations_global": len(optimization.forced_associations_global),
                "forced_association_steps": len(optimization.forced_associations_by_step),
                "association_class_count": len(optimization.association_class_bindable_ports),
                "association_class_candidate_edges": sum(
                    len({candidate_id for candidate_id, _producer_port in producer_ports})
                    for producer_ports in optimization.association_class_bindable_ports.values()
                ),
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
