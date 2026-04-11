"""Direct translation entrypoints for single-shot and multi-shot modes."""

from __future__ import annotations

from clindaws.core.models import FactBundle, SnakeConfig, ToolMode
from clindaws.core.ontology import Ontology
from clindaws.translators.builder import build_fact_bundle, build_fact_bundle_ape_multi_shot

__all__ = (
    "build_fact_bundle",
    "build_fact_bundle_ape_multi_shot",
    "FactBundle",
    "SnakeConfig",
    "ToolMode",
    "Ontology",
)
