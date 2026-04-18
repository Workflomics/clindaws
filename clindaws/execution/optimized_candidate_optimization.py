"""Canonical optimized-candidate precomputation entrypoints."""

from __future__ import annotations

from clindaws.execution.compressed_candidate_optimization import (
    CompressedCandidateOptimizationResult,
    optimize_compressed_candidates,
)


def optimize_optimized_candidates(*args, **kwargs):
    """Canonical alias for the optimized-candidate precompute pipeline."""

    return optimize_compressed_candidates(*args, **kwargs)


__all__ = (
    "CompressedCandidateOptimizationResult",
    "optimize_optimized_candidates",
    "optimize_compressed_candidates",
)
