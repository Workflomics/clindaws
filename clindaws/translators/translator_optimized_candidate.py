"""Canonical optimized-candidate translation entrypoints."""

from __future__ import annotations

from clindaws.translators.translator_compressed_candidate import (
    CompressedCandidateOptimizationResult,
    FactBundle,
    Ontology,
    SnakeConfig,
    ToolMode,
    build_compressed_candidate_fact_bundle,
    build_optimized_candidate_fact_bundle,
    optimize_compressed_candidates,
)


def optimize_optimized_candidates(*args, **kwargs):
    """Canonical alias for the optimized-candidate precompute pipeline."""

    return optimize_compressed_candidates(*args, **kwargs)


__all__ = (
    "build_optimized_candidate_fact_bundle",
    "build_compressed_candidate_fact_bundle",
    "optimize_optimized_candidates",
    "optimize_compressed_candidates",
    "CompressedCandidateOptimizationResult",
    "FactBundle",
    "SnakeConfig",
    "ToolMode",
    "Ontology",
)
