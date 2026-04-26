"""Translation summary, metadata, schema warnings, and JSON writers."""

from __future__ import annotations

import json
from pathlib import Path

from clindaws.core.config import SnakeConfig
from clindaws.core.models import HorizonRecord, TranslationBuilder
from clindaws.core.workflow_input_compression import (
    build_workflow_input_compression_plan,
    workflow_input_compression_stats,
)
from clindaws.execution.runner_modes import (
    SCHEMA_PREDICATES,
    _mode_config,
    _solver_approach,
    _solver_family,
)
from clindaws.execution.solver_control import program_paths_for_mode


def _constraint_metadata(config) -> tuple[str | None, int]:
    if config.constraints_path is None:
        return None, 0

    count = 0
    if config.constraints_path.exists():
        raw = json.loads(config.constraints_path.read_text(encoding="utf-8"))
        count = len(raw.get("constraints", []))
    return str(config.constraints_path), count


def _run_metadata_payload(*, config, ontology, tools) -> dict[str, object]:
    constraints_used, constraint_count = _constraint_metadata(config)
    return {
        "config_path": str(config.config_path),
        "ontology_used": str(config.ontology_path),
        "ontology_entry_count": len(ontology.nodes),
        "tool_annotation_used": str(config.tool_annotations_path),
        "tool_count": len(tools),
        "constraints_used": constraints_used,
        "constraint_count": constraint_count,
    }


def _translation_schema(fact_bundle) -> str:
    if any(
        fact_bundle.predicate_counts.get(name, 0)
        for name in (
            "dynamic_tool_candidate",
            "dynamic_signature_support_class",
            "dynamic_signature_profile",
            "dynamic_profile_accepts",
            "dynamic_candidate_output_singleton",
            "dynamic_candidate_output_choice_value",
        )
    ):
        return "optimized_candidate"
    if any(fact_bundle.predicate_counts.get(name, 0) for name in ("tool_candidate", "candidate_in", "candidate_out")):
        return "candidate"
    if any(fact_bundle.predicate_counts.get(name, 0) for name in ("tool_input", "input_port", "tool_output", "output_port")):
        return "legacy"
    return "unknown"


def _encoding_schema_summary(
    mode: str,
    *,
    optimized: bool = False,
) -> dict[str, object]:
    program_paths = program_paths_for_mode(
        mode,
        optimized=optimized,
    )
    predicate_presence: dict[str, bool] = {}
    for predicate in SCHEMA_PREDICATES:
        predicate_presence[predicate] = False

    for program_path in program_paths:
        text = program_path.read_text(encoding="utf-8")
        for predicate in SCHEMA_PREDICATES:
            if not predicate_presence[predicate] and f"{predicate}(" in text:
                predicate_presence[predicate] = True

    return {
        "program_paths": [str(path) for path in program_paths],
        "predicate_presence": predicate_presence,
        "schema": (
            "optimized_candidate"
            if any(
                predicate_presence[name]
                for name in (
                    "dynamic_tool_candidate",
                    "dynamic_signature_support_class",
                    "dynamic_signature_profile",
                    "dynamic_profile_accepts",
                    "dynamic_candidate_output_singleton",
                    "dynamic_candidate_output_choice_value",
                )
            )
            else
            "candidate"
            if any(predicate_presence[name] for name in ("tool_candidate", "candidate_in", "candidate_out"))
            else "legacy"
            if any(predicate_presence[name] for name in ("tool_input", "input_port", "tool_output", "output_port"))
            else "unknown"
        ),
    }


def _translation_warnings(
    *,
    mode: str,
    fact_bundle,
    encoding_summary: dict[str, object],
) -> list[str]:
    warnings: list[str] = []
    translation_schema = _translation_schema(fact_bundle)
    encoding_presence = encoding_summary["predicate_presence"]
    translation_pathway = _mode_config(mode).translation_pathway

    if translation_pathway in {"dynamic", "optimized_candidate"} and translation_schema != "optimized_candidate":
        warnings.append(
            f"{mode} expects optimized-candidate translation, but the emitted translation schema is {translation_schema}."
        )
    if translation_schema == "optimized_candidate" and encoding_summary["schema"] != "optimized_candidate":
        warnings.append(
            "Optimized-candidate translation is not compatible with the selected encoding."
        )

    if translation_schema == "candidate" and not any(
        encoding_presence[name] for name in ("tool_candidate", "candidate_in", "candidate_out")
    ):
        warnings.append(
            "Translated facts use candidate predicates, but the selected encoding does not reference candidate predicates."
        )

    if translation_schema == "optimized_candidate" and not any(
        encoding_presence[name]
        for name in (
            "dynamic_tool_candidate",
            "dynamic_signature_support_class",
            "dynamic_signature_profile",
            "dynamic_profile_accepts",
            "dynamic_candidate_output_singleton",
            "dynamic_candidate_output_choice_value",
        )
    ):
        warnings.append(
            "Translated facts use optimized-candidate predicates, but the selected encoding does not reference the optimized-candidate predicate family."
        )

    if translation_schema == "legacy" and not any(
        encoding_presence[name] for name in ("tool_input", "input_port", "tool_output", "output_port")
    ):
        warnings.append(
            "Translated facts use legacy tool_input/tool_output predicates, but the selected encoding does not reference the legacy schema."
        )

    if encoding_summary["schema"] != "unknown" and translation_schema != "unknown" and encoding_summary["schema"] != translation_schema:
        warnings.append(
            f"Translation schema ({translation_schema}) does not match encoding schema ({encoding_summary['schema']})."
        )

    return warnings


def _workflow_input_compression_payload(
    *,
    config: SnakeConfig,
    mode: str,
    internal_solver_mode: str,
    compression_active: bool | None = None,
) -> dict[str, object] | None:
    if mode != "multi-shot" or internal_solver_mode != "multi-shot":
        return None

    workflow_input_dimensions = {
        f"wf_input_{index}": dimensions
        for index, dimensions in enumerate(config.inputs)
    }
    plan = build_workflow_input_compression_plan(workflow_input_dimensions)
    payload: dict[str, object] = dict(sorted(workflow_input_compression_stats(plan).items()))
    if compression_active is not None:
        payload["workflow_input_compression_active"] = compression_active
        payload["workflow_input_planner_visible_count_effective"] = (
            plan.planner_visible_count_if_compressed
            if compression_active
            else plan.planner_visible_count_if_uncompressed
        )
    return payload


def _solve_callback_profile_payload(
    records: tuple[HorizonRecord, ...],
    *,
    solving_sec: float,
) -> dict[str, float] | None:
    if not any(record.model_callback_sec is not None for record in records):
        return None

    model_callback_sec = sum(record.model_callback_sec or 0.0 for record in records)
    shown_symbols_sec = sum(record.shown_symbols_sec or 0.0 for record in records)
    workflow_signature_key_sec = sum(
        record.workflow_signature_key_sec or 0.0
        for record in records
    )
    canonicalization_sec = sum(record.canonicalization_sec or 0.0 for record in records)
    other_callback_sec = max(
        0.0,
        model_callback_sec
        - shown_symbols_sec
        - workflow_signature_key_sec
        - canonicalization_sec,
    )
    payload = {
        "model_callback_sec": model_callback_sec,
        "shown_symbols_sec": shown_symbols_sec,
        "workflow_signature_key_sec": workflow_signature_key_sec,
        "canonicalization_sec": canonicalization_sec,
        "other_callback_sec": other_callback_sec,
    }
    if solving_sec > 0:
        payload["model_callback_share_of_solving_sec"] = model_callback_sec / solving_sec
    return payload


def _translation_summary_payload(
    *,
    config: SnakeConfig,
    mode: str,
    grounding_strategy: str,
    translation_builder: TranslationBuilder,
    effective_translation_strategy: str,
    fact_bundle,
    translation_path: Path | None,
    translation_sec: float,
) -> dict[str, object]:
    internal_mode = fact_bundle.internal_solver_mode or mode
    encoding_summary = _encoding_schema_summary(
        internal_mode,
        optimized=fact_bundle.python_precompute_enabled,
    )
    schema_presence = {
        predicate: fact_bundle.predicate_counts.get(predicate, 0) > 0
        for predicate in SCHEMA_PREDICATES
    }
    candidate_total = sum(
        stat.candidate_count or 0
        for stat in fact_bundle.tool_stats
    )
    dynamic_input_value_total = sum(stat.dynamic_input_value_count or 0 for stat in fact_bundle.tool_stats)
    dynamic_output_value_total = sum(stat.dynamic_output_value_count or 0 for stat in fact_bundle.tool_stats)
    dynamic_cross_product_estimate = sum(stat.dynamic_cross_product_estimate or 0 for stat in fact_bundle.tool_stats)

    return {
        "mode": mode,
        "solver_family": _solver_family(mode),
        "solver_approach": _solver_approach(mode),
        "internal_solver_mode": internal_mode,
        "internal_schema": fact_bundle.internal_schema,
        "grounding_strategy": grounding_strategy,
        "translation_builder": translation_builder,
        "effective_translation_strategy": effective_translation_strategy,
        "translation_schema": _translation_schema(fact_bundle),
        "fact_count": fact_bundle.fact_count,
        "tool_count": len(fact_bundle.tool_stats),
        "goal_count": fact_bundle.goal_count,
        "workflow_input_count": len(fact_bundle.workflow_input_ids),
        "earliest_solution_step": fact_bundle.earliest_solution_step,
        "python_precompute_enabled": fact_bundle.python_precompute_enabled,
        "python_precompute_fact_count": fact_bundle.python_precompute_fact_count,
        "python_precompute_stats": dict(sorted(fact_bundle.python_precompute_stats.items())),
        "workflow_input_compression": _workflow_input_compression_payload(
            config=config,
            mode=mode,
            internal_solver_mode=internal_mode,
        ),
        "translation_path": str(translation_path) if translation_path else None,
        "translation_sec": translation_sec,
        "predicate_counts": dict(sorted(fact_bundle.predicate_counts.items())),
        "translation_cache_stats": dict(sorted(fact_bundle.cache_stats.items())),
        "translation_emit_stats": dict(sorted(fact_bundle.emit_stats.items())),
        "backend_stats": fact_bundle.backend_stats,
        "translation_schema_predicates": schema_presence,
        "encoding_schema": encoding_summary,
        "expansion_totals": {
            "input_variant_total": sum(stat.input_variant_count for stat in fact_bundle.tool_stats),
            "output_variant_total": sum(stat.output_variant_count for stat in fact_bundle.tool_stats),
            "candidate_total": candidate_total if candidate_total else None,
            "dynamic_input_value_total": dynamic_input_value_total if dynamic_input_value_total else None,
            "dynamic_output_value_total": dynamic_output_value_total if dynamic_output_value_total else None,
            "dynamic_cross_product_estimate": dynamic_cross_product_estimate if dynamic_cross_product_estimate else None,
        },
        "per_tool_port_value_counts": [
            {
                "tool_id": stat.tool_id,
                "tool_label": stat.tool_label,
                "input_port_value_counts": list(stat.dynamic_input_port_value_counts),
                "output_port_value_counts": list(stat.dynamic_output_port_value_counts),
            }
            for stat in fact_bundle.tool_stats
            if stat.dynamic_input_port_value_counts or stat.dynamic_output_port_value_counts
        ],
        "warnings": _translation_warnings(
            mode=internal_mode,
            fact_bundle=fact_bundle,
            encoding_summary=encoding_summary,
        ),
    }


def _write_translation_summary(
    *,
    config: SnakeConfig,
    solution_dir: Path,
    mode: str,
    grounding_strategy: str,
    translation_builder: TranslationBuilder,
    effective_translation_strategy: str,
    fact_bundle,
    translation_path: Path | None,
    translation_sec: float,
) -> tuple[Path, dict[str, object]]:
    translation_summary_path = solution_dir / "translation_summary.json"
    payload = _translation_summary_payload(
        config=config,
        mode=mode,
        grounding_strategy=grounding_strategy,
        translation_builder=translation_builder,
        effective_translation_strategy=effective_translation_strategy,
        fact_bundle=fact_bundle,
        translation_path=None,
        translation_sec=translation_sec,
    )
    translation_summary_path.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    return translation_summary_path, payload
