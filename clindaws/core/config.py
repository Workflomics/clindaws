"""Configuration loading and normalization."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from pydantic import BaseModel, ConfigDict, model_validator


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


def _strip_prefix(value: str, prefix: str) -> str:
    if prefix and value.startswith(prefix):
        return value[len(prefix):]
    if "#" in value:
        return value.rsplit("#", 1)[1]
    if "/" in value:
        return value.rsplit("/", 1)[1]
    return value


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


def _normalize_io(
    items: Iterable[Mapping[str, Iterable[str]]],
    *,
    prefix: str,
) -> tuple[Mapping[str, tuple[str, ...]], ...]:
    normalized = []
    for item in items:
        normalized.append(
            {
                str(dim): tuple(_strip_prefix(str(value), prefix) for value in values)
                for dim, values in item.items()
            }
        )
    return tuple(normalized)


class SnakeConfig(BaseModel):
    """Normalized configuration resolved from an APE-style config."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    config_path: Path
    base_dir: Path
    ontology_path: Path
    tool_annotations_path: Path
    constraints_path: Path | None
    solutions_dir_path: Path
    ontology_prefix: str
    tools_taxonomy_root: str
    data_dimensions_taxonomy_roots: tuple[str, ...]
    strict_tool_annotations: bool
    solution_length_min: int
    solution_length_max: int
    solutions: int
    timeout_sec: int
    number_of_generated_graphs: int
    inputs: tuple[Mapping[str, tuple[str, ...]], ...]
    outputs: tuple[Mapping[str, tuple[str, ...]], ...]
    use_workflow_input: str
    use_all_generated_data: str
    tool_seq_repeat: bool
    debug_mode: bool

    @model_validator(mode="before")
    @classmethod
    def _normalize_from_raw(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        # If already normalized (all Path fields present), pass through.
        if isinstance(data.get("config_path"), Path):
            return data

        data = dict(data)  # copy to avoid mutating caller's dict
        config_file: Path = data.pop("_config_file")
        base_dir = config_file.parent
        solution_length = data.get("solution_length", {})
        try:
            ontology_prefix = str(data["ontologyPrefixIRI"])
        except KeyError as exc:
            raise ValueError(f"Missing required config field: {exc}") from exc

        try:
            ontology_path_raw = data["ontology_path"]
            tool_annotations_path_raw = data["tool_annotations_path"]
            solutions_dir_path_raw = data["solutions_dir_path"]
            tools_taxonomy_root_raw = data["toolsTaxonomyRoot"]
        except KeyError as exc:
            raise ValueError(f"Missing required config field: {exc}") from exc

        return {
            "config_path": config_file,
            "base_dir": base_dir,
            "ontology_path": _resolve_path(base_dir, ontology_path_raw, must_exist=True),
            "tool_annotations_path": _resolve_path(base_dir, tool_annotations_path_raw, must_exist=True),
            "constraints_path": (
                _resolve_path(base_dir, data["constraints_path"], must_exist=True)
                if data.get("constraints_path")
                else None
            ),
            "solutions_dir_path": _resolve_solution_dir_path(solutions_dir_path_raw),
            "ontology_prefix": ontology_prefix,
            "tools_taxonomy_root": _strip_prefix(str(tools_taxonomy_root_raw), ontology_prefix),
            "data_dimensions_taxonomy_roots": tuple(
                _strip_prefix(str(value), ontology_prefix)
                for value in data.get("dataDimensionsTaxonomyRoots", [])
            ),
            "strict_tool_annotations": _to_bool(data.get("strict_tool_annotations"), False),
            "solution_length_min": _to_int(solution_length.get("min"), 1),
            "solution_length_max": _to_int(solution_length.get("max"), 1),
            "solutions": _to_int(data.get("solutions"), 1),
            "timeout_sec": _to_int(data.get("timeout_sec"), 300),
            "number_of_generated_graphs": _to_int(data.get("number_of_generated_graphs"), 0),
            "inputs": _normalize_io(data.get("inputs", []), prefix=ontology_prefix),
            "outputs": _normalize_io(data.get("outputs", []), prefix=ontology_prefix),
            "use_workflow_input": str(data.get("use_workflow_input", "none")).upper(),
            "use_all_generated_data": str(data.get("use_all_generated_data", "none")).upper(),
            "tool_seq_repeat": _to_bool(data.get("tool_seq_repeat"), False),
            "debug_mode": _to_bool(data.get("debug_mode"), False),
        }


def load_config(config_path: str | Path) -> SnakeConfig:
    """Load and normalize an APE-style config file."""
    config_file = Path(config_path).resolve()
    raw = json.loads(config_file.read_text(encoding="utf-8"))
    return SnakeConfig.model_validate({"_config_file": config_file, **raw})
