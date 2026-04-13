from __future__ import annotations
from collections.abc import Mapping
from dataclasses import dataclass, field
from itertools import product

from clindaws.core.ontology import Ontology




@dataclass
class _ExpansionResolver:
    """Reuse resolved terminal expansions across translation."""

    ontology: Ontology
    roots: Mapping[str, frozenset[str]]
    strategy: str
    _value_cache: dict[tuple[str, bool, str, str], tuple[str, ...]] = field(default_factory=dict)
    _stats: dict[str, int] = field(
        default_factory=lambda: {
            "value_expansion_hits": 0,
            "value_expansion_misses": 0,
        }
    )

    def _should_expand(self, expand_outputs: bool) -> bool:
        return self.strategy == "python" or (self.strategy == "hybrid" and expand_outputs)

    def expanded_values(self, dim: str, value: str, *, expand_outputs: bool) -> tuple[str, ...]:
        allowed_nodes = self.roots.get(dim, frozenset())
        if not self._should_expand(expand_outputs) or value not in self.ontology.nodes or not allowed_nodes:
            return (value,)

        cache_key = (self.strategy, expand_outputs, dim, value)
        cached = self._value_cache.get(cache_key)
        if cached is not None:
            self._stats["value_expansion_hits"] += 1
            return cached

        self._stats["value_expansion_misses"] += 1
        expanded = self.ontology.terminal_descendants_of(value, within=allowed_nodes)
        self._value_cache[cache_key] = expanded
        return expanded

    def iter_dimension_maps(
        self,
        dims: Mapping[str, tuple[str, ...]],
        *,
        expand_outputs: bool,
    ):
        per_dimension: list[tuple[tuple[str, str], ...]] = []
        for dim, values in dims.items():
            choices: list[tuple[str, str]] = []
            for value in values:
                for expanded_value in self.expanded_values(dim, value, expand_outputs=expand_outputs):
                    choices.append((dim, expanded_value))
            per_dimension.append(tuple(choices))

        if not per_dimension:
            yield {}
            return

        for choice in product(*per_dimension):
            combo: dict[str, str] = {}
            valid = True
            for dim, value in choice:
                if dim in combo and combo[dim] != value:
                    valid = False
                    break
                combo[dim] = value
            if valid:
                yield combo

    def stats(self) -> dict[str, int]:
        return dict(self._stats)
