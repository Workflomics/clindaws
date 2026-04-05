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


def _normalize_output_ports(
    raw_ports,
    *,
    prefix: str,
    dedupe_duplicate_outputs: bool,
) -> tuple[ToolPortSpec, ...]:
    ports = tuple(
        ToolPortSpec.from_mapping(
            {
                dim: tuple(_strip_prefix(value, prefix) for value in values)
                for dim, values in port.items()
            }
        )
        for port in raw_ports or []
    )
    if not dedupe_duplicate_outputs:
        return ports

    seen: set[frozenset[tuple[str, frozenset[str]]]] = set()
    deduped: list[ToolPortSpec] = []
    for spec in ports:
        key = frozenset(
            (dim, frozenset(vals)) for dim, vals in spec.dimensions.items()
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(spec)
    return tuple(deduped)


def load_tool_annotations(
    path: Path,
    prefix: str,
    *,
    dedupe_duplicate_outputs: bool = False,
) -> tuple[ToolMode, ...]:
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
        outputs = _normalize_output_ports(
            item.get("outputs") or [],
            prefix=prefix,
            dedupe_duplicate_outputs=dedupe_duplicate_outputs,
        )
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


def load_direct_tool_annotations(path: Path, prefix: str) -> tuple[ToolMode, ...]:
    """Load tool annotations for direct modes with legacy duplicate-output dedupe."""

    return load_tool_annotations(
        path,
        prefix,
        dedupe_duplicate_outputs=True,
    )


def load_lazy_tool_annotations(path: Path, prefix: str) -> tuple[ToolMode, ...]:
    """Load tool annotations for lazy mode preserving duplicate outputs."""

    return load_tool_annotations(
        path,
        prefix,
        dedupe_duplicate_outputs=False,
    )
