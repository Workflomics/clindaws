"""Runtime mode and backend resolution."""

from __future__ import annotations

from dataclasses import dataclass

from clindaws.core.models import TranslationBuilder


RUNTIME_TRANSLATION_BUILDER = "runtime_legacy"
OPTIMIZED_CANDIDATE_TRANSLATION_BUILDER = "candidate_optimized"


@dataclass(frozen=True)
class ModeConfig:
    solver_family: str
    solver_approach: str
    translation_pathway: str
    translation_builder: TranslationBuilder
    supports_ground_only: bool


MODE_CONFIGS = {
    "single-shot": ModeConfig(
        solver_family="single-shot",
        solver_approach="one-shot",
        translation_pathway="ape_multi_shot",
        translation_builder=RUNTIME_TRANSLATION_BUILDER,
        supports_ground_only=True,
    ),
    "multi-shot": ModeConfig(
        solver_family="multi-shot",
        solver_approach="legacy",
        translation_pathway="ape_multi_shot",
        translation_builder=RUNTIME_TRANSLATION_BUILDER,
        supports_ground_only=True,
    ),
    "optimized": ModeConfig(
        solver_family="multi-shot",
        solver_approach="optimized_candidate",
        translation_pathway="optimized_candidate",
        translation_builder=OPTIMIZED_CANDIDATE_TRANSLATION_BUILDER,
        supports_ground_only=True,
    ),
    "multi-shot-optimized-candidate": ModeConfig(
        solver_family="multi-shot",
        solver_approach="optimized_candidate",
        translation_pathway="optimized_candidate",
        translation_builder=OPTIMIZED_CANDIDATE_TRANSLATION_BUILDER,
        supports_ground_only=True,
    ),
}


def normalize_decompression_mode(value: str | None) -> str | None:
    if value in {"1n", "1:n"}:
        return "one-to-n"
    if value in {None, "kcluster", "one-to-n"}:
        return value
    raise ValueError(f"Unsupported decompression mode: {value}")


def mode_config(mode: str) -> ModeConfig:
    try:
        return MODE_CONFIGS[mode]
    except KeyError as exc:
        raise ValueError(f"Unsupported mode: {mode}") from exc


def validate_mode_request(
    *,
    mode: str,
    optimized: bool = False,
    decompression_mode: str | None = None,
) -> tuple[str, str | None]:
    decompression_mode = normalize_decompression_mode(decompression_mode)
    config = mode_config(mode)
    if optimized and config.solver_family == "single-shot":
        raise ValueError("--optimized is not yet supported for single-shot modes.")
    if optimized and mode not in {"multi-shot", "optimized"}:
        raise ValueError("--optimized supports only multi-shot/optimized.")
    if decompression_mode is not None and mode != "optimized":
        raise ValueError("--decomp requires --mode optimized.")
    return effective_public_mode(mode, optimized=optimized), decompression_mode


def effective_public_mode(mode: str, *, optimized: bool = False) -> str:
    if optimized and mode == "multi-shot":
        return "optimized"
    return mode


def effective_translation_strategy(mode: str, grounding_strategy: str) -> str:
    translation_pathway = mode_config(mode).translation_pathway
    if translation_pathway == "optimized_candidate":
        return "python"
    if translation_pathway == "ape_multi_shot":
        return "ape_clingo_legacy"
    return grounding_strategy


def solver_family(mode: str) -> str:
    return mode_config(mode).solver_family


def solver_approach(mode: str) -> str:
    return mode_config(mode).solver_approach


def approach_label(mode: str, optimized: bool) -> str:
    family = mode_config(mode).solver_family
    if family == "single-shot":
        return "single_shot"
    if optimized or mode == "optimized" or mode_config(mode).solver_approach == "optimized_candidate":
        return "optimized_multi_shot"
    return "multi-shot"
