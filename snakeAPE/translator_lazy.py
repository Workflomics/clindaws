"""Lazy translation entrypoints for multi-shot-lazy mode."""

from __future__ import annotations

from .models import FactBundle, SnakeConfig, ToolMode
from .ontology import Ontology
from .translator import build_lazy_fact_bundle

__all__ = (
    "build_lazy_fact_bundle",
    "FactBundle",
    "SnakeConfig",
    "ToolMode",
    "Ontology",
)
