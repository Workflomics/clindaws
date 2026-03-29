"""OWL parsing and taxonomy helpers."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
import re
import subprocess
from typing import Iterable


def _strip_prefix(value: str, prefix: str) -> str:
    if value.startswith(prefix):
        return value[len(prefix):]
    if "#" in value:
        return value.rsplit("#", 1)[1]
    if "/" in value:
        return value.rsplit("/", 1)[1]
    return value


@dataclass(frozen=True)
class Ontology:
    """Ontology edges and lookup helpers."""

    edges: tuple[tuple[str, str], ...]
    parent_map: dict[str, frozenset[str]]
    child_map: dict[str, frozenset[str]]
    nodes: frozenset[str]
    _descendants_cache: dict[str, frozenset[str]] = field(default_factory=dict, init=False, repr=False, compare=False)
    _ancestors_cache: dict[str, frozenset[str]] = field(default_factory=dict, init=False, repr=False, compare=False)
    _terminal_descendants_cache: dict[tuple[str, frozenset[str] | None], tuple[str, ...]] = field(default_factory=dict, init=False, repr=False, compare=False)
    _cache_stats: dict[str, int] = field(
        default_factory=lambda: {
            "descendants_hits": 0,
            "descendants_misses": 0,
            "ancestors_hits": 0,
            "ancestors_misses": 0,
            "terminal_descendants_hits": 0,
            "terminal_descendants_misses": 0,
        },
        init=False,
        repr=False,
        compare=False,
    )

    @classmethod
    def from_file(cls, path: Path, prefix: str) -> "Ontology":
        result = subprocess.run(
            ["/opt/homebrew/bin/xmllint", "--format", str(path)],
            check=True,
            capture_output=True,
            text=True,
        )
        xml_text = result.stdout
        edge_set: set[tuple[str, str]] = set()
        nodes: set[str] = set()

        class_pattern = re.compile(
            r"<(?:owl:Class|rdf:Description)\b[^>]*rdf:about=\"([^\"]+)\"[^>]*(?<!/)>(.*?)</(?:owl:Class|rdf:Description)>",
            re.DOTALL,
        )
        direct_parent_pattern = re.compile(
            r"<rdfs:subClassOf\b[^>]*rdf:resource=\"([^\"]+)\"\s*/?>",
            re.DOTALL,
        )
        nested_parent_pattern = re.compile(
            r"<rdfs:subClassOf\b[^>]*>.*?<rdf:Description\b[^>]*rdf:about=\"([^\"]+)\"[^>]*/?>.*?</rdfs:subClassOf>",
            re.DOTALL,
        )

        for about, body in class_pattern.findall(xml_text):
            child = _strip_prefix(about, prefix)
            nodes.add(child)
            for parent_ref in direct_parent_pattern.findall(body):
                parent = _strip_prefix(parent_ref, prefix)
                edge_set.add((child, parent))
                nodes.add(parent)
            for parent_ref in nested_parent_pattern.findall(body):
                parent = _strip_prefix(parent_ref, prefix)
                edge_set.add((child, parent))
                nodes.add(parent)

        self_closing_pattern = re.compile(
            r"<(?:owl:Class|rdf:Description)\b[^>]*rdf:about=\"([^\"]+)\"[^>]*/>"
        )
        for about in self_closing_pattern.findall(xml_text):
            nodes.add(_strip_prefix(about, prefix))

        parent_map: dict[str, set[str]] = defaultdict(set)
        child_map: dict[str, set[str]] = defaultdict(set)
        for child, parent in edge_set:
            parent_map[child].add(parent)
            child_map[parent].add(child)

        return cls(
            edges=tuple(sorted(edge_set)),
            parent_map={key: frozenset(value) for key, value in parent_map.items()},
            child_map={key: frozenset(value) for key, value in child_map.items()},
            nodes=frozenset(nodes),
        )

    def descendants_of(self, root: str) -> frozenset[str]:
        """Return descendants of ``root``, including ``root``."""

        cached = self._descendants_cache.get(root)
        if cached is not None:
            self._cache_stats["descendants_hits"] += 1
            return cached
        self._cache_stats["descendants_misses"] += 1
        seen: set[str] = set()
        queue: deque[str] = deque([root])
        while queue:
            current = queue.popleft()
            if current in seen:
                continue
            seen.add(current)
            for child in self.child_map.get(current, frozenset()):
                queue.append(child)
        descendants = frozenset(seen)
        self._descendants_cache[root] = descendants
        return descendants

    def ancestors_of(self, node: str) -> frozenset[str]:
        """Return ancestors of ``node``, including ``node``."""

        cached = self._ancestors_cache.get(node)
        if cached is not None:
            self._cache_stats["ancestors_hits"] += 1
            return cached
        self._cache_stats["ancestors_misses"] += 1
        seen: set[str] = set()
        queue: deque[str] = deque([node])
        while queue:
            current = queue.popleft()
            if current in seen:
                continue
            seen.add(current)
            for parent in self.parent_map.get(current, frozenset()):
                queue.append(parent)
        ancestors = frozenset(seen)
        self._ancestors_cache[node] = ancestors
        return ancestors

    def terminal_descendants_of(self, node: str, *, within: Iterable[str] | None = None) -> tuple[str, ...]:
        """Return terminal descendants of ``node`` within a subtree."""

        cache_key = (node, frozenset(within) if within is not None else None)
        cached = self._terminal_descendants_cache.get(cache_key)
        if cached is not None:
            self._cache_stats["terminal_descendants_hits"] += 1
            return cached
        self._cache_stats["terminal_descendants_misses"] += 1

        allowed = cache_key[1] if cache_key[1] is not None else self.descendants_of(node)
        terminals = []
        for descendant in sorted(self.descendants_of(node)):
            if descendant not in allowed:
                continue
            children = [
                child
                for child in self.child_map.get(descendant, frozenset())
                if child in allowed
            ]
            if not children:
                terminals.append(descendant)
        result = tuple(terminals or [node])
        self._terminal_descendants_cache[cache_key] = result
        return result

    def cache_stats(self) -> dict[str, int]:
        """Return ontology cache usage counters."""

        return dict(self._cache_stats)
