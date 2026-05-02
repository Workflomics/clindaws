"""Mode metadata, dispatch tables, and translation-pathway resolvers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from clindaws.core.models import TranslationBuilder
from clindaws.execution.solver import (
    ground_multi_shot,
    ground_multi_shot_optimized_candidate,
    ground_single_shot,
    solve_multi_shot,
    solve_multi_shot_optimized_candidate,
    solve_single_shot,
    solve_single_shot_sliding_window,
)


SCHEMA_PREDICATES = (
    "tool_input",
    "input_port",
    "tool_output",
    "output_port",
    "tool_candidate",
    "candidate_in",
    "candidate_out",
    "dynamic_tool_candidate",
    "dynamic_candidate_min_step",
    "dynamic_candidate_max_step",
    "dynamic_goal_support_candidate_at_horizon",
    "dynamic_goal_support_tool_at_horizon",
    "dynamic_goal_support_input_at_horizon",
    "dynamic_structurally_supportable_candidate_at_horizon",
    "dynamic_structurally_unsupported_input_at_horizon",
    "dynamic_initial_bindable",
    "dynamic_candidate_input_port",
    "dynamic_candidate_input_signature_id",
    "optimized_candidate_input_support_class",
    "dynamic_candidate_input_association_class",
    "dynamic_signature_support_class",
    "dynamic_support_class_bindable_producer_port",
    "dynamic_association_class_bindable_producer_port",
    "optimized_candidate_input_profile",
    "dynamic_signature_profile",
    "dynamic_profile_accepts",
    "optimized_goal_requirement_profile",
    "dynamic_check_relevant_output_category_at_horizon",
    "dynamic_check_required_profile_class_at_horizon",
    "dynamic_check_profile_class_member",
    "dynamic_forced_produced_bind",
    "dynamic_candidate_output_port",
    "dynamic_candidate_output_multiplicity",
    "dynamic_candidate_output_single_use",
    "dynamic_candidate_output_multi_use",
    "dynamic_candidate_total_output_multiplicity",
    "dynamic_candidate_output_singleton",
    "dynamic_candidate_output_choice_value",
)
RUNTIME_TRANSLATION_BUILDER = "runtime_legacy"
OPTIMIZED_CANDIDATE_TRANSLATION_BUILDER = "candidate_optimized"
ProgressCallback = Callable[[object], None] | None


@dataclass(frozen=True)
class _ModeConfig:
    solver_family: str
    solver_approach: str
    translation_pathway: str
    translation_builder: TranslationBuilder
    supports_ground_only: bool


_MODE_CONFIGS = {
    # Public single-shot currently shares the APE-style multi-shot translation
    # surface, then changes only the grounding/solving strategy downstream.
    "single-shot": _ModeConfig(
        solver_family="single-shot",
        solver_approach="one-shot",
        translation_pathway="ape_multi_shot",
        translation_builder=RUNTIME_TRANSLATION_BUILDER,
        supports_ground_only=True,
    ),
    "single-shot-sliding-window": _ModeConfig(
        solver_family="single-shot",
        solver_approach="sliding_window",
        translation_pathway="ape_multi_shot",
        translation_builder=RUNTIME_TRANSLATION_BUILDER,
        supports_ground_only=False,
    ),
    "multi-shot": _ModeConfig(
        solver_family="multi-shot",
        solver_approach="legacy",
        translation_pathway="ape_multi_shot",
        translation_builder=RUNTIME_TRANSLATION_BUILDER,
        supports_ground_only=True,
    ),
    # Optimized multi-shot is an explicit optimized-candidate backend rather
    # than a small variation of the direct multi-shot encoding family.
    "multi-shot-optimized-candidate": _ModeConfig(
        solver_family="multi-shot",
        solver_approach="optimized_candidate",
        translation_pathway="optimized_candidate",
        translation_builder=OPTIMIZED_CANDIDATE_TRANSLATION_BUILDER,
        supports_ground_only=True,
    ),
}

_SOLVER_DISPATCH = {
    "single-shot": solve_single_shot,
    "single-shot-sliding-window": solve_single_shot_sliding_window,
    "multi-shot": solve_multi_shot,
    "multi-shot-optimized-candidate": solve_multi_shot_optimized_candidate,
}

_GROUNDER_DISPATCH = {
    "single-shot": ground_single_shot,
    "multi-shot": ground_multi_shot,
    "multi-shot-optimized-candidate": ground_multi_shot_optimized_candidate,
}


def _mode_config(mode: str) -> _ModeConfig:
    try:
        return _MODE_CONFIGS[mode]
    except KeyError as exc:
        raise ValueError(f"Unsupported mode: {mode}") from exc


def _effective_translation_strategy(mode: str, grounding_strategy: str) -> str:
    translation_pathway = _mode_config(mode).translation_pathway
    if translation_pathway == "optimized_candidate":
        return "python"
    if translation_pathway == "ape_multi_shot":
        return "ape_clingo_legacy"
    return grounding_strategy


def _solver_family(mode: str) -> str:
    return _mode_config(mode).solver_family


def _solver_approach(mode: str) -> str:
    return _mode_config(mode).solver_approach
