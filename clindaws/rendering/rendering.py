"""DOT generation and artifact writing."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess

from clindaws.core.models import ArtifactRef, WorkflowSolution


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _artifact_node_name(ref_id: str) -> str:
    return "artifact_" + "".join(
        char if char.isalnum() else "_"
        for char in ref_id
    )


def _tool_node_name(time: int) -> str:
    return f"tool_{time}"


def _artifact_label(artifact: ArtifactRef) -> str:
    type_values = ", ".join(sorted(artifact.dims.get("Type", set())))
    format_values = ", ".join(sorted(artifact.dims.get("Format", set())))
    if type_values and format_values:
        return f"{type_values}\\n{format_values}"
    if type_values:
        return type_values
    if format_values:
        return format_values
    return artifact.ref_id.strip('"')


def build_dot(solution: WorkflowSolution, title: str) -> str:
    """Build a DOT graph for a workflow solution."""

    lines = [
        "digraph workflow {",
        '  rankdir=TB;',
        '  graph [fontname="Helvetica", labelloc="t", labeljust="l", fontsize=18];',
        '  node [fontname="Helvetica", fontsize=11];',
        '  edge [fontname="Helvetica", fontsize=10, color="#566573"];',
        f'  label="{_escape(title)}";',
    ]

    input_artifacts = sorted(
        (artifact for artifact in solution.artifacts.values() if artifact.created_at == 0),
        key=lambda item: item.ref_id,
    )
    if input_artifacts:
        lines.append('  subgraph cluster_inputs {')
        lines.append('    label="Workflow Inputs";')
        lines.append('    color="#cfd8dc";')
        for artifact in input_artifacts:
            lines.append(
                f'    {_artifact_node_name(artifact.ref_id)} '
                f'[shape=box, style="rounded,filled", fillcolor="#f7fbff", color="#607d8b", '
                f'label="{_escape(_artifact_label(artifact))}"];'
            )
        lines.append("  }")

    for step in solution.steps:
        lines.append(
            f'  {_tool_node_name(step.time)} '
            f'[shape=box, style="rounded,filled", fillcolor="#fff3e0", color="#ef6c00", '
            f'label="{_escape(f"{step.time}. {step.tool_label}")}"];'
        )

    output_artifacts = sorted(
        (artifact for artifact in solution.artifacts.values() if artifact.created_at > 0),
        key=lambda item: (item.created_at, item.ref_id),
    )
    for artifact in output_artifacts:
        lines.append(
            f'  {_artifact_node_name(artifact.ref_id)} '
            f'[shape=ellipse, style="filled", fillcolor="#e8f5e9", color="#2e7d32", '
            f'label="{_escape(_artifact_label(artifact))}"];'
        )

    if solution.goal_outputs:
        lines.append('  subgraph cluster_outputs {')
        lines.append('    label="Workflow Outputs";')
        lines.append('    color="#cfd8dc";')
        for goal_index, artifact_id in sorted(solution.goal_outputs.items()):
            node_name = f"goal_{goal_index}"
            lines.append(
                f'    {node_name} [shape=box, style="rounded,dashed", color="#455a64", '
                f'label="{_escape(f"Goal {goal_index}")}"];'
            )
            lines.append(
                f'    {_artifact_node_name(artifact_id)} -> {node_name} [label="goal"];'
            )
        lines.append("  }")

    for step in solution.steps:
        for binding in step.bindings:
            lines.append(
                f'  {_artifact_node_name(binding.artifact_id)} -> {_tool_node_name(step.time)} '
                f'[label="{_escape(binding.port_id)}"];'
            )
        for artifact_id in step.outputs:
            lines.append(
                f'  {_tool_node_name(step.time)} -> {_artifact_node_name(artifact_id)};'
            )

    lines.append("}")
    return "\n".join(lines) + "\n"


def render_solution_graphs(
    output_dir: Path,
    solution: WorkflowSolution,
    graph_format: str,
) -> tuple[Path, ...]:
    """Write DOT and rendered graph files for one solution."""

    output_dir.mkdir(parents=True, exist_ok=True)
    base_name = f"candidate_workflow_{solution.index}"
    dot_content = build_dot(solution, f"Candidate Workflow {solution.index}")

    if graph_format == "dot":
        dot_path = output_dir / f"{base_name}.dot"
        dot_path.write_text(dot_content, encoding="utf-8")
        return (dot_path,)

    rendered_path = output_dir / f"{base_name}.{graph_format}"
    subprocess.run(
        ["dot", f"-T{graph_format}", "-o", str(rendered_path)],
        input=dot_content,
        text=True,
        check=True,
    )
    return (rendered_path,)


def write_solution_summary(output_path: Path, solutions: tuple[WorkflowSolution, ...]) -> Path:
    """Write a readable solution summary."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not solutions:
        output_path.write_text("No solutions found.\n", encoding="utf-8")
        return output_path
    lines = []
    for solution in solutions:
        tools = " -> ".join(solution.tool_sequence)
        lines.append(f"Candidate {solution.index}")
        lines.append(f"Length: {solution.length}")
        lines.append(f"Tools: {tools}")
        if solution.signature_bindings:
            lines.append(f"Bindings: {' | '.join(solution.signature_bindings)}")
        if solution.goal_bindings:
            lines.append(f"Goals: {' | '.join(solution.goal_bindings)}")
        lines.append("")
    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return output_path


def write_workflow_signatures(output_path: Path, solutions: tuple[WorkflowSolution, ...]) -> Path:
    """Write machine-readable workflow signatures."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "workflow_signatures": [
            {
                "index": solution.index,
                "length": solution.length,
                "tools": list(solution.tool_sequence),
                "bindings": list(solution.signature_bindings),
                "goals": list(solution.goal_bindings),
            }
            for solution in solutions
        ]
    }
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return output_path
