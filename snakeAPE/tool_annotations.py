"""Tool annotation loading."""

from __future__ import annotations

import json
from pathlib import Path

from .models import ToolMode, ToolPortSpec


def _strip_prefix(value: str, prefix: str) -> str:
    if prefix and value.startswith(prefix):
        return value[len(prefix):]
    if "#" in value:
        return value.rsplit("#", 1)[1]
    return value


def load_tool_annotations(path: Path, prefix: str) -> tuple[ToolMode, ...]:
    """Load tool annotations from JSON."""

    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    tools = []
    for item in raw.get("functions", []):
        inputs = tuple(
            ToolPortSpec.from_mapping(
                {
                    dim: tuple(_strip_prefix(value, prefix) for value in values)
                    for dim, values in port.items()
                }
            )
            for port in item.get("inputs") or []
        )
        outputs = tuple(
            ToolPortSpec.from_mapping(
                {
                    dim: tuple(_strip_prefix(value, prefix) for value in values)
                    for dim, values in port.items()
                }
            )
            for port in item.get("outputs") or []
        )
        taxonomy_operations = tuple(
            _strip_prefix(value, prefix) for value in item.get("taxonomyOperations", [])
        )
        tools.append(
            ToolMode(
                label=str(item["label"]),
                mode_id=str(item["id"]),
                taxonomy_operations=taxonomy_operations,
                inputs=inputs,
                outputs=outputs,
                implementation=(item.get("implementation") or {}).get("code"),
            )
        )
    return tuple(tools)
