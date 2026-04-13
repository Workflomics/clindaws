from __future__ import annotations
from collections.abc import Iterable, Mapping





def _quote(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'
def _fact(name: str, *args: str) -> str:
    return f"{name}({', '.join(args)})."
def _product(values: Iterable[int]) -> int:
    total = 1
    for value in values:
        total *= value
    return total
def _normalize_dim_map(mapping: Mapping[str, Iterable[str]]) -> dict[str, tuple[str, ...]]:
    return {
        str(dim): tuple(str(value) for value in values)
        for dim, values in mapping.items()
    }
def _dedupe_stable(values: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return tuple(ordered)
