"""Direct translation entrypoints for single-shot and multi-shot modes.

The actual builder logic lives in ``builder.py``. This module exists so the
runner can import a small, mode-oriented surface without knowing whether the
direct backend is being used for public multi-shot or the current single-shot
runtime.
"""

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
