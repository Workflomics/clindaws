"""Direct translation entrypoints for single-shot and multi-shot modes."""

from __future__ import annotations

from .models import FactBundle, SnakeConfig, ToolMode
from .ontology import Ontology
from .translator_core import build_fact_bundle, build_fact_bundle_ape_multi_shot

__all__ = (
    "build_fact_bundle",
    "build_fact_bundle_ape_multi_shot",
    "FactBundle",
    "SnakeConfig",
    "ToolMode",
    "Ontology",
)
