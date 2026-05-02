"""Fact-bundle building, tool loading, and per-mode bundle selection."""

from __future__ import annotations

from dataclasses import replace

from clindaws.core.config import SnakeConfig
from clindaws.core.models import FactBundle
from clindaws.core.ontology import Ontology
from clindaws.core.tool_annotations import (
    load_candidate_tool_annotations,
    load_direct_tool_annotations,
    load_multi_shot_tool_annotations,
)
from clindaws.execution.precompute import apply_precompute
from clindaws.execution.runner_modes import (
    OPTIMIZED_CANDIDATE_TRANSLATION_BUILDER,
    ProgressCallback,
    _ModeConfig,
)
from clindaws.execution.runner_utils import _report
from clindaws.translators.translator_direct import build_fact_bundle_ape_multi_shot
from clindaws.translators.translator_optimized_candidate import (
    build_optimized_candidate_fact_bundle,
)


def _workflow_input_dims_from_config(config: SnakeConfig) -> dict[str, dict[str, tuple[str, ...]]]:
    return {
        f"wf_input_{index}": {
            str(dim): tuple(str(value) for value in values)
            for dim, values in item.items()
        }
        for index, item in enumerate(config.inputs)
    }


def _tool_output_dims_lookup(tools: tuple) -> dict[tuple[str, int], dict[str, tuple[str, ...]]]:
    output_dims: dict[tuple[str, int], dict[str, tuple[str, ...]]] = {}
    output_dims_by_label: dict[str, dict[int, set[tuple[tuple[str, tuple[str, ...]], ...]]]] = {}
    for tool in tools:
        label_ports = output_dims_by_label.setdefault(str(tool.label), {})
        for port_index, output_spec in enumerate(getattr(tool, "outputs", ())):
            dims = {
                str(dim): tuple(str(value) for value in values)
                for dim, values in output_spec.dimensions.items()
            }
            output_dims[(str(tool.mode_id), port_index)] = dims
            label_ports.setdefault(port_index, set()).add(tuple(sorted(dims.items())))
    for label, port_variants in output_dims_by_label.items():
        for port_index, variants in port_variants.items():
            if len(variants) == 1:
                output_dims[(label, port_index)] = dict(next(iter(variants)))
    return output_dims


def _load_tools_for_mode(config, translation_pathway: str):
    if translation_pathway == "optimized_candidate":
        return load_candidate_tool_annotations(config.tool_annotations_path, config.ontology_prefix)
    if translation_pathway == "ape_multi_shot":
        return load_multi_shot_tool_annotations(config.tool_annotations_path, config.ontology_prefix)
    return load_direct_tool_annotations(config.tool_annotations_path, config.ontology_prefix)


def _ape_multi_shot_direct_bundle(
    config: SnakeConfig,
    ontology: Ontology,
    tools,
    *,
    internal_solver_mode: str,
):
    return replace(
        build_fact_bundle_ape_multi_shot(config, ontology, tools),
        internal_schema="legacy_direct",
        internal_solver_mode=internal_solver_mode,
    )


def _optimized_candidate_internal_bundle(
    config: SnakeConfig,
    ontology: Ontology,
    tools,
    *,
    max_workers: int = 1,
):
    return replace(
        build_optimized_candidate_fact_bundle(config, ontology, tools, max_workers=max_workers),
        internal_schema="optimized_candidate",
        internal_solver_mode="multi-shot-optimized-candidate",
    )


def _select_fact_bundle(
    *,
    mode_config: _ModeConfig,
    mode: str,
    config: SnakeConfig,
    ontology: Ontology,
    tools: tuple,
    optimized: bool,
    progress_callback: ProgressCallback,
    max_workers: int = 1,
) -> tuple[FactBundle, str]:
    """Select and build the fact bundle, applying backend fallback logic.

    Returns (fact_bundle, resolved_translation_builder).
    """
    resolved_translation_builder = mode_config.translation_builder

    if mode_config.translation_pathway == "ape_multi_shot":
        # Plain multi-shot and the public single-shot modes both start from the
        # APE-style direct fact surface. Optional precompute augments that
        # bundle, while optimized multi-shot swaps to the optimized-candidate
        # internal bundle entirely.
        fact_bundle = _ape_multi_shot_direct_bundle(
            config,
            ontology,
            tools,
            internal_solver_mode=(
                "single-shot"
                if mode == "single-shot"
                else "single-shot-sliding-window"
                if mode == "single-shot-sliding-window"
                else "multi-shot"
            ),
        )
        if optimized:
            if mode_config.solver_family == "single-shot":
                raise ValueError("--optimized is not yet supported for single-shot modes.")
            optimized_candidate_tools = load_candidate_tool_annotations(
                config.tool_annotations_path,
                config.ontology_prefix,
            )
            _report(progress_callback, "Step 1b: optimized-candidate translation started.")
            fact_bundle = _optimized_candidate_internal_bundle(
                config,
                ontology,
                optimized_candidate_tools,
                max_workers=max_workers,
            )
            resolved_translation_builder = OPTIMIZED_CANDIDATE_TRANSLATION_BUILDER
            _report(
                progress_callback,
                "Step 1b complete: selected optimized-candidate schema "
                f"with {fact_bundle.fact_count} facts.",
            )
        else:
            fact_bundle = apply_precompute(
                mode,
                config,
                ontology,
                tools,
                fact_bundle,
                optimized_programs=False,
            )

    else:
        raise ValueError(f"Unsupported translation pathway: {mode_config.translation_pathway}")

    return fact_bundle, resolved_translation_builder
