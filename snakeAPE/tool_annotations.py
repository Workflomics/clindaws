"""Tool annotation loading."""

from __future__ import annotations

import json
from pathlib import Path
from collections import Counter, defaultdict

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

    functions = raw.get("functions", [])
    id_counts = Counter(str(item["id"]) for item in functions)
    id_offsets: dict[str, int] = defaultdict(int)

    tools = []
    for item in functions:
        raw_id = str(item["id"])
        internal_id = raw_id
        if id_counts[raw_id] > 1:
            internal_id = f"{raw_id}__ann{id_offsets[raw_id]}"
            id_offsets[raw_id] += 1
        inputs = tuple(
            ToolPortSpec.from_mapping(
                {
                    dim: tuple(_strip_prefix(value, prefix) for value in values)
                    for dim, values in port.items()
                }
            )
            for port in item.get("inputs") or []
        )
        _seen_output_keys: set[frozenset] = set()
        _deduped_outputs: list[ToolPortSpec] = []
        for port in item.get("outputs") or []:
            spec = ToolPortSpec.from_mapping(
                {
                    dim: tuple(_strip_prefix(value, prefix) for value in values)
                    for dim, values in port.items()
                }
            )
            key = frozenset(
                (dim, frozenset(vals)) for dim, vals in spec.dimensions.items()
            )
            if key not in _seen_output_keys:
                _seen_output_keys.add(key)
                _deduped_outputs.append(spec)
        outputs = tuple(_deduped_outputs)
        taxonomy_operations = tuple(
            _strip_prefix(value, prefix) for value in item.get("taxonomyOperations", [])
        )
        tools.append(
            ToolMode(
                label=str(item["label"]),
                mode_id=internal_id,
                taxonomy_operations=taxonomy_operations,
                inputs=inputs,
                outputs=outputs,
                implementation=(item.get("implementation") or {}).get("code"),
            )
        )
    return tuple(tools)
