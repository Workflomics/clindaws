"""OWL parsing and taxonomy helpers."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
import re
import subprocess
from typing import Iterable
import xml.etree.ElementTree as ET


def _strip_prefix(value: str, prefix: str) -> str:
    if value.startswith(prefix):
        return value[len(prefix):]
    if value.startswith("#"):
        return value[1:]
    if "#" in value:
        return value.rsplit("#", 1)[1]
    if "/" in value:
        return value.rsplit("/", 1)[1]
    return value


def _parse_rdf_xml_ontology(xml_text: str, prefix: str) -> tuple[set[tuple[str, str]], set[str]]:
    edge_set: set[tuple[str, str]] = set()
    nodes: set[str] = set()

    ns = {
        "owl": "http://www.w3.org/2002/07/owl#",
        "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
        "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
    }
    rdf_about = f"{{{ns['rdf']}}}about"
    rdf_resource = f"{{{ns['rdf']}}}resource"
    owl_class = f"{{{ns['owl']}}}Class"
    rdf_description = f"{{{ns['rdf']}}}Description"
    rdfs_subclass = f"{{{ns['rdfs']}}}subClassOf"
    root = ET.fromstring(xml_text)

    named_tags = {owl_class, rdf_description}
    for element in root.iter():
        if element.tag not in named_tags:
            continue
        about = element.attrib.get(rdf_about)
        if not about:
            continue
        child = _strip_prefix(about, prefix)
        nodes.add(child)
        for subclass_of in element.findall(rdfs_subclass):
            direct_parent_ref = subclass_of.attrib.get(rdf_resource)
            if direct_parent_ref:
                parent = _strip_prefix(direct_parent_ref, prefix)
                edge_set.add((child, parent))
                nodes.add(parent)
                continue

            for nested in subclass_of.iter():
                if nested is subclass_of:
                    continue
                if nested.tag not in named_tags:
                    continue
                parent_ref = nested.attrib.get(rdf_about)
                if not parent_ref:
                    continue
                parent = _strip_prefix(parent_ref, prefix)
                edge_set.add((child, parent))
                nodes.add(parent)

    return edge_set, nodes


def _parse_owl_xml_ontology(xml_text: str, prefix: str) -> tuple[set[tuple[str, str]], set[str]]:
    edge_set: set[tuple[str, str]] = set()
    nodes: set[str] = set()

    declaration_pattern = re.compile(
        r"<Declaration>\s*<Class\s+IRI=\"([^\"]+)\"\s*/>\s*</Declaration>",
        re.DOTALL,
    )
    subclass_pattern = re.compile(
        r"<SubClassOf>\s*<Class\s+IRI=\"([^\"]+)\"\s*/>\s*<Class\s+IRI=\"([^\"]+)\"\s*/>\s*</SubClassOf>",
        re.DOTALL,
    )

    for iri in declaration_pattern.findall(xml_text):
        nodes.add(_strip_prefix(iri, prefix))

    for child_ref, parent_ref in subclass_pattern.findall(xml_text):
        child = _strip_prefix(child_ref, prefix)
        parent = _strip_prefix(parent_ref, prefix)
        edge_set.add((child, parent))
        nodes.add(child)
        nodes.add(parent)

    return edge_set, nodes


def _detect_ontology_format(xml_text: str) -> str:
    if "<rdf:RDF" in xml_text or "<owl:Class" in xml_text or "<rdfs:subClassOf" in xml_text:
        return "rdf_xml"
    if "<Ontology" in xml_text and ("<Declaration>" in xml_text or "<SubClassOf>" in xml_text):
        return "owl_xml"
    snippet = " ".join(xml_text.splitlines()[:5]).strip()
    raise ValueError(
        "Unsupported ontology XML format. "
        f"Expected RDF/XML or OWL/XML, got leading content: {snippet[:200]!r}"
    )


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
        ontology_format = _detect_ontology_format(xml_text)
        if ontology_format == "rdf_xml":
            edge_set, nodes = _parse_rdf_xml_ontology(xml_text, prefix)
        else:
            edge_set, nodes = _parse_owl_xml_ontology(xml_text, prefix)

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
