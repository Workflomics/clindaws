"""ASP program family selection."""

from __future__ import annotations

from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
ENCODINGS_ROOT = PACKAGE_ROOT / "encodings"


SINGLE_SHOT_OVERLAY_PREFIX = ""


def single_shot_overlay(min_length: int, horizon: int) -> str:
    """Return one-shot constraints layered on top of the shared multi-shot core."""

    overlay = [
        SINGLE_SHOT_OVERLAY_PREFIX,
        f":- not holds({horizon}, goal).",
        ":- occurs(T, run(_)), not occurs(T-1, run(_)), T > 1.",
        ":- occurs(T, run(ToolA)), occurs(T, run(ToolB)), ToolA < ToolB.",
        "",
        "% One-shot ordering layer: solve exact (goal_time, run_count) slices on one",
        "% grounded control object, lexicographically by earlier goal then shorter",
        "% active prefix.",
        "single_shot_goal_time(1) :- holds(1, goal).",
        "single_shot_goal_time(T) :- T > 1, holds(T, goal), not holds(T-1, goal).",
        "single_shot_run_count(N) :- N = #count { T : occurs(T, run(_)) }.",
        "#external single_shot_target_goal_time(T) : time(T).",
        "#external single_shot_target_run_count(N) : time(N).",
        ":- single_shot_target_goal_time(T), not single_shot_goal_time(T).",
        ":- single_shot_target_goal_time(Target), single_shot_goal_time(Actual), Actual != Target.",
        ":- single_shot_target_run_count(N), not single_shot_run_count(N).",
        ":- single_shot_target_run_count(Target), single_shot_run_count(Actual), Actual != Target.",
    ]
    if min_length > 1:
        overlay.append(f":- holds({min_length - 1}, goal).")
    return "\n".join(overlay) + "\n"


def single_shot_program_paths(*, optimized: bool = False) -> tuple[Path, ...]:
    """Return the active program set for public single-shot modes."""

    if optimized:
        raise ValueError("--optimized is not yet supported for single-shot.")
    return multi_shot_program_paths()


def multi_shot_program_paths() -> tuple[Path, ...]:
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


def optimized_candidate_program_paths() -> tuple[Path, ...]:
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

    if mode == "single-shot":
        return single_shot_program_paths(optimized=optimized)
    if mode == "multi-shot":
        if optimized:
            return optimized_candidate_program_paths()
        return multi_shot_program_paths()
    if mode == "multi-shot-optimized-candidate":
        return optimized_candidate_program_paths()
    raise ValueError(f"Unsupported mode: {mode}")
