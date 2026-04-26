"""Clingo control creation, fact loading, and ASP program path resolution."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import clingo

from clindaws.core.models import FactBundle, HorizonRecord


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
ENCODINGS_ROOT = PACKAGE_ROOT / "encodings"
ProgressCallback = Callable[[object], None] | None
BaseGroundingCallback = Callable[[float, float], None] | None
HorizonRecordCallback = Callable[[HorizonRecord], None] | None


def _make_solve_control(
    *,
    parallel_mode: str | None = None,
    project_models: bool = False,
) -> clingo.Control:
    """Create a clingo control for solve-time model enumeration.

    The public runtime keeps clingo's native model bound open and enforces the
    stored workflow quota in Python after canonicalization. That preserves the
    meaning of ``solutions`` as a cap on unique stored workflows rather than a
    cap on raw pre-canonical answer sets.
    """

    args = ["0", "--warn=none"]
    if project_models:
        args.append("--project")
    if parallel_mode:
        args.append(f"--parallel-mode={parallel_mode}")
    return clingo.Control(args)


def _make_grounding_control() -> clingo.Control:
    """Create a clingo control for grounding-only operations."""

    return clingo.Control(["0", "--warn=none"])


def _projection_runtime_facts(*, mode: str, project_models: bool) -> str:
    """Return runtime fact toggles derived from the chosen solve policy."""

    facts: list[str] = []
    if mode in {"multi-shot", "single-shot"} and not project_models:
        facts.append("full_workflow_input_witnesses.\n")
    return "".join(facts)


def _load_control_programs(
    control: clingo.Control,
    *,
    program_paths: tuple[Path, ...],
    facts: FactBundle,
    mode: str,
    project_models: bool,
) -> None:
    for program_path in program_paths:
        control.load(str(program_path))
    control.add("base", [], facts.facts)
    runtime_facts = _projection_runtime_facts(mode=mode, project_models=project_models)
    if runtime_facts:
        control.add("base", [], runtime_facts)
    if facts.python_precomputed_facts:
        control.add("base", [], facts.python_precomputed_facts)


def _single_shot_program_paths(*, optimized: bool = False) -> tuple[Path, ...]:
    """Return the active program set for public single-shot mode.

    The current single-shot runtime deliberately reuses the direct multi-shot
    encoding family. Solver behavior differs by grounding/solving strategy, not
    by a separate active ASP program family.
    """
    if optimized:
        raise ValueError("--optimized is not yet supported for single-shot.")
    return _multi_shot_program_paths()


def _multi_shot_program_paths() -> tuple[Path, ...]:
    """Return the plain direct incremental multi-shot program set."""
    base = ENCODINGS_ROOT / "multi_shot"
    return (
        base / "base.lp",
        base / "init.lp",
        base / "step.lp",
        base / "constraints.lp",
        base / "check.lp",
        base / "ape_extract.lp",
        base / "tool_inclusion.lp",
        base / "tool_dependency.lp",
        base / "temporal_constraint.lp",
        base / "input_usage.lp",
        base / "output_usage.lp",
    )


def _multi_shot_optimized_candidate_program_paths() -> tuple[Path, ...]:
    """Return the optimized-candidate incremental program set."""
    base = ENCODINGS_ROOT / "multi_shot_optimized_candidate"
    return (
        base / "base.lp",
        base / "step_initial.lp",
        base / "step.lp",
        base / "step_query.lp",
        base / "exact_certificate.lp",
        base / "possible.lp",
        base / "constraints.lp",
        base / "check.lp",
        base / "ape_extract.lp",
        base / "tool_inclusion.lp",
        base / "input_usage.lp",
        base / "output_usage.lp",
    )


def program_paths_for_mode(
    mode: str,
    *,
    optimized: bool = False,
) -> tuple[Path, ...]:
    """Return the concrete ASP program family for one effective solver mode."""

    if mode in {"single-shot", "single-shot-sliding-window"}:
        return _single_shot_program_paths(optimized=optimized)
    if mode == "multi-shot":
        if optimized:
            return _multi_shot_optimized_candidate_program_paths()
        return _multi_shot_program_paths()
    if mode in {"multi-shot-optimized-candidate", "multi-shot-compressed-candidate"}:
        return _multi_shot_optimized_candidate_program_paths()
    raise ValueError(f"Unsupported mode: {mode}")
