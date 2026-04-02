"""Configuration loading and normalization."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from .models import SnakeConfig


TRUE_VALUES = {"1", "true", "yes", "on"}


def _to_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in TRUE_VALUES


def _to_int(value: Any, default: int) -> int:
    if value is None:
        return default
    return int(value)


def _resolve_path(base_dir: Path, value: str, *, must_exist: bool) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    candidates = [(base_dir / path).resolve()]
    for parent in base_dir.parents:
        candidates.append((parent / path).resolve())
    for candidate in candidates:
        if must_exist and candidate.exists():
            return candidate
        if not must_exist and candidate.parent.exists():
            return candidate
    return candidates[0]


def _resolve_solution_dir_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return path.resolve()


def _normalize_io(items: Iterable[Mapping[str, Iterable[str]]]) -> tuple[Mapping[str, tuple[str, ...]], ...]:
    normalized = []
    for item in items:
        normalized.append(
            {
                str(dim): tuple(str(value) for value in values)
                for dim, values in item.items()
            }
        )
    return tuple(normalized)


def load_config(config_path: str | Path) -> SnakeConfig:
    """Load and normalize an APE-style config file."""

    config_file = Path(config_path).resolve()
    with config_file.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    base_dir = config_file.parent
    solution_length = raw.get("solution_length", {})

    return SnakeConfig(
        config_path=config_file,
        base_dir=base_dir,
        ontology_path=_resolve_path(base_dir, raw["ontology_path"], must_exist=True),
        tool_annotations_path=_resolve_path(base_dir, raw["tool_annotations_path"], must_exist=True),
        constraints_path=(
            _resolve_path(base_dir, raw["constraints_path"], must_exist=True)
            if raw.get("constraints_path")
            else None
        ),
        solutions_dir_path=_resolve_solution_dir_path(raw["solutions_dir_path"]),
        ontology_prefix=str(raw["ontologyPrefixIRI"]),
        tools_taxonomy_root=str(raw["toolsTaxonomyRoot"]),
        data_dimensions_taxonomy_roots=tuple(
            str(value) for value in raw.get("dataDimensionsTaxonomyRoots", [])
        ),
        strict_tool_annotations=_to_bool(raw.get("strict_tool_annotations"), False),
        solution_length_min=_to_int(solution_length.get("min"), 1),
        solution_length_max=_to_int(solution_length.get("max"), 1),
        solutions=_to_int(raw.get("solutions"), 1),
        timeout_sec=_to_int(raw.get("timeout_sec"), 300),
        number_of_generated_graphs=_to_int(raw.get("number_of_generated_graphs"), 0),
        inputs=_normalize_io(raw.get("inputs", [])),
        outputs=_normalize_io(raw.get("outputs", [])),
        use_workflow_input=str(raw.get("use_workflow_input", "none")).upper(),
        use_all_generated_data=str(raw.get("use_all_generated_data", "none")).upper(),
        tool_seq_repeat=_to_bool(raw.get("tool_seq_repeat"), False),
        debug_mode=_to_bool(raw.get("debug_mode"), False),
    )
