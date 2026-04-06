"""Translate config, OWL, tool annotations, and supported constraints into facts."""

from __future__ import annotations

import json
import re
from collections import defaultdict, deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from io import StringIO
from itertools import product
from time import perf_counter
from typing import Any

from .models import FactBundle, SnakeConfig, ToolExpansionStat, ToolMode
from .ontology import Ontology


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


def _prefer_less_specific_value(
    ontology: Ontology,
    current: str,
    candidate: str,
) -> str:
    current_depth = len(ontology.ancestors_of(current))
    candidate_depth = len(ontology.ancestors_of(candidate))
    if candidate_depth < current_depth:
        return candidate
    if candidate_depth > current_depth:
        return current
    return candidate if candidate < current else current


def _reduce_requirement_values(
    ontology: Ontology,
    values: Iterable[str],
) -> tuple[str, ...]:
    """Drop requirement values subsumed by a more general retained value."""

    ordered_values = _dedupe_stable(values)
    if len(ordered_values) <= 1:
        return ordered_values
    if all(not ontology.child_map.get(value, frozenset()) for value in ordered_values):
        return ordered_values

    reduced: list[str] = []
    for value in ordered_values:
        value_ancestors = ontology.ancestors_of(value)
        if any(existing in value_ancestors for existing in reduced):
            continue
        reduced = [
            existing
            for existing in reduced
            if value not in ontology.ancestors_of(existing)
        ]
        reduced.append(value)
    return tuple(reduced)


def _reduce_signature_requirements(
    ontology: Ontology,
    requirements: Mapping[str, tuple[str, ...]],
) -> dict[str, tuple[str, ...]]:
    reduced: dict[str, tuple[str, ...]] = {}
    for dim, values in sorted(requirements.items()):
        reduced_values = _reduce_requirement_values(ontology, values)
        if reduced_values:
            reduced[dim] = reduced_values
    return reduced


def _signature_key(
    requirements: Mapping[str, tuple[str, ...]],
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    return tuple(
        (dim, tuple(values))
        for dim, values in sorted(requirements.items())
    )


def _assign_lazy_signature_profiles(
    ontology: Ontology,
    roots: Mapping[str, frozenset[str]],
    input_ports: Iterable[dict[str, object]],
) -> tuple[
    dict[int, dict[str, tuple[int, tuple[str, ...]]]],
    dict[int, tuple[str, ...]],
    dict[int, tuple[str, ...]],
    dict[str, int],
]:
    """Assign reduced signature/profile ids to lazy input ports."""

    raw_requirement_count = 0
    reduced_requirement_count = 0
    signature_ids_by_key: dict[tuple[tuple[str, tuple[str, ...]], ...], int] = {}
    profile_ids_by_key: dict[tuple[str, tuple[str, ...]], int] = {}
    signature_profiles_by_id: dict[int, dict[str, tuple[int, tuple[str, ...]]]] = {}
    profile_values_by_id: dict[int, tuple[str, ...]] = {}
    profile_accepts_by_id: dict[int, tuple[str, ...]] = {}

    for input_port in input_ports:
        port_requirements = input_port["port_values_by_dimension"]
        assert isinstance(port_requirements, Mapping)
        raw_requirement_count += sum(len(values) for values in port_requirements.values())
        reduced_requirements = _reduce_signature_requirements(ontology, port_requirements)
        reduced_requirement_count += sum(len(values) for values in reduced_requirements.values())
        input_port["signature_requirements"] = reduced_requirements

        signature_key = _signature_key(reduced_requirements)
        signature_id = signature_ids_by_key.setdefault(signature_key, len(signature_ids_by_key))
        input_port["signature_id"] = signature_id

    for signature_key, signature_id in sorted(signature_ids_by_key.items(), key=lambda item: item[1]):
        category_profiles: dict[str, tuple[int, tuple[str, ...]]] = {}
        for category, values in signature_key:
            profile_key = (category, values)
            profile_id = profile_ids_by_key.setdefault(profile_key, len(profile_ids_by_key))
            profile_values_by_id.setdefault(profile_id, values)
            profile_accepts_by_id.setdefault(
                profile_id,
                _dedupe_stable(
                    actual_value
                    for required_value in values
                    for actual_value in ontology.terminal_descendants_of(
                        required_value,
                        within=roots.get(category, frozenset()),
                    )
                ),
            )
            category_profiles[category] = (profile_id, values)
        signature_profiles_by_id[signature_id] = category_profiles

    return (
        signature_profiles_by_id,
        profile_values_by_id,
        profile_accepts_by_id,
        {
            "lazy_raw_signature_requirement_count": raw_requirement_count,
            "lazy_reduced_signature_requirement_count": reduced_requirement_count,
            "lazy_signature_count": len(signature_profiles_by_id),
            "lazy_profile_count": len(profile_values_by_id),
        },
    )


def _port_signature(spec) -> tuple[tuple[str, tuple[str, ...]], ...]:
    return tuple(
        (str(dim), tuple(str(value) for value in values))
        for dim, values in sorted(_normalize_dim_map(spec.dimensions).items())
    )


def _tool_input_signatures(tools: tuple[ToolMode, ...]) -> dict[str, tuple[tuple[tuple[str, tuple[str, ...]], ...], ...]]:
    signatures: dict[str, tuple[tuple[tuple[str, tuple[str, ...]], ...], ...]] = {}
    for tool in tools:
        signatures[tool.mode_id] = tuple(_port_signature(port) for port in tool.inputs)
    return signatures


def _bundle_metadata(
    config: SnakeConfig,
    tools: tuple[ToolMode, ...],
) -> tuple[dict[str, str], dict[str, tuple[tuple[tuple[str, tuple[str, ...]], ...], ...]], tuple[str, ...]]:
    return (
        {tool.mode_id: tool.label for tool in tools},
        _tool_input_signatures(tools),
        tuple(f"wf_input_{i}" for i in range(len(config.inputs))),
    )


def _finalize_fact_bundle(
    writer: "_FactWriter",
    *,
    config: SnakeConfig,
    tools: tuple[ToolMode, ...],
    tool_stats: list[ToolExpansionStat],
    cache_stats: dict[str, int],
    earliest_solution_step: int = 1,
) -> FactBundle:
    tool_labels, tool_input_signatures, workflow_input_ids = _bundle_metadata(config, tools)
    return FactBundle(
        facts=writer.text(),
        fact_count=writer.fact_count,
        tool_labels=tool_labels,
        tool_input_signatures=tool_input_signatures,
        workflow_input_ids=workflow_input_ids,
        goal_count=len(config.outputs),
        predicate_counts=dict(writer.predicate_counts),
        tool_stats=tuple(tool_stats),
        cache_stats=cache_stats,
        emit_stats=writer.stats(),
        earliest_solution_step=earliest_solution_step,
    )


@dataclass
class _FactWriter:
    """Incrementally build fact text and counts."""

    buffer: StringIO = field(default_factory=StringIO)
    predicate_counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    fact_count: int = 0

    def emit_fact(self, name: str, *args: str) -> None:
        self.buffer.write(_fact(name, *args))
        self.buffer.write("\n")
        self.predicate_counts[name] += 1
        self.fact_count += 1

    def emit_atom(self, name: str) -> None:
        self.buffer.write(f"{name}.\n")
        self.predicate_counts[name] += 1
        self.fact_count += 1

    def emit_rule(self, name: str, text: str) -> None:
        self.buffer.write(text)
        if not text.endswith("\n"):
            self.buffer.write("\n")
        self.predicate_counts[name] += 1
        self.fact_count += 1

    def emit_comment(self, text: str) -> None:
        self.buffer.write(f"% {text}\n")

    def text(self) -> str:
        return self.buffer.getvalue()

    def stats(self) -> dict[str, int]:
        return {
            "fact_count": self.fact_count,
            "text_chars": self.buffer.tell(),
        }


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


def _build_roots(
    config: SnakeConfig,
    ontology: Ontology,
) -> dict[str, frozenset[str]]:
    tool_taxonomy_nodes = ontology.descendants_of(config.tools_taxonomy_root)
    raw_roots = {
        root: frozenset(
            node
            for node in ontology.descendants_of(root)
            if node not in tool_taxonomy_nodes
        )
        for root in config.data_dimensions_taxonomy_roots
    }
    roots: dict[str, frozenset[str]] = {}
    for root in config.data_dimensions_taxonomy_roots:
        excluded: set[str] = set()
        for other_root in config.data_dimensions_taxonomy_roots:
            if other_root == root:
                continue
            if other_root in raw_roots[root]:
                excluded.update(raw_roots[other_root])
        roots[root] = frozenset(node for node in raw_roots[root] if node not in excluded)
    return roots


def _build_common_facts(
    writer: _FactWriter,
    config: SnakeConfig,
    ontology: Ontology,
    tools: tuple[ToolMode, ...],
) -> dict[str, frozenset[str]]:
    """Build taxonomy/tool/workflow/goal facts shared by all bundles."""
    roots = _build_roots(config, ontology)
    tool_taxonomy_nodes = ontology.descendants_of(config.tools_taxonomy_root)

    for dimension_root in config.data_dimensions_taxonomy_roots:
        allowed = roots[dimension_root]
        for child, parent in ontology.edges:
            if child in allowed and parent in allowed:
                writer.emit_fact(
                    "taxonomy",
                    _quote("ape"),
                    _quote(dimension_root),
                    f"({_quote(dimension_root)}, {_quote(child)}, {_quote(parent)})",
                )

    # Pre-compute compatible(V, Ancestor) for every terminal V in each dimension
    # subtree. Python's cached BFS (ontology.ancestors_of) replaces the O(n²)
    # recursive ancestor/2 transitive closure that clingo would otherwise derive
    # at grounding time. ancestors_of(V) includes V itself, so compatible(V, V)
    # is covered without a separate rule.
    for dimension_root in config.data_dimensions_taxonomy_roots:
        allowed = roots[dimension_root]
        for terminal in ontology.terminal_descendants_of(dimension_root, within=allowed):
            for anc in ontology.ancestors_of(terminal):
                if anc in allowed:
                    writer.emit_fact("compatible", _quote(terminal), _quote(anc))

    for child, parent in ontology.edges:
        if child in tool_taxonomy_nodes and parent in tool_taxonomy_nodes:
            writer.emit_fact(
                "tool_taxonomy",
                _quote("ape"),
                f"({_quote(child)}, {_quote(parent)})",
            )

    for tool in tools:
        writer.emit_fact("tool", _quote(tool.mode_id))
        for tax_op in tool.taxonomy_operations:
            writer.emit_fact(
                "tool_taxonomy",
                _quote("ape"),
                f"({_quote(tool.mode_id)}, {_quote(tax_op)})",
            )

    for index, item in enumerate(config.inputs):
        wf_id = f"wf_input_{index}"
        writer.emit_fact("holds", "0", f"avail({_quote(wf_id)})")
        for dim, values in sorted(item.items()):
            allowed = roots.get(dim, frozenset())
            for value in values:
                for tval in ontology.terminal_descendants_of(value, within=allowed):
                    writer.emit_fact(
                        "holds",
                        "0",
                        f"dim({_quote(wf_id)}, {_quote(tval)}, {_quote(dim)})",
                    )
                    writer.emit_fact(
                        "ape_holds_dim",
                        _quote(wf_id),
                        _quote(tval),
                        _quote(dim),
                    )

    for goal_index, item in enumerate(config.outputs):
        for dim, values in sorted(item.items()):
            for value in values:
                writer.emit_fact(
                    "goal_output",
                    str(goal_index),
                    _quote(value),
                    _quote(dim),
                )

    if config.use_workflow_input == "ALL":
        writer.emit_atom("enable_all_inputs_used")
    elif config.use_workflow_input == "ONE":
        writer.emit_atom("enable_some_input_used")

    if config.use_all_generated_data == "ALL":
        writer.emit_atom("enable_all_outputs_consumed")
        writer.emit_atom("enable_usefulness_pruning")
    elif config.use_all_generated_data == "ONE":
        writer.emit_atom("enable_primary_output_consumed")
        writer.emit_atom("enable_usefulness_pruning")

    if config.tool_seq_repeat:
        writer.emit_rule("multi_run", "multi_run(Tool) :- tool(Tool).")

    return roots


_SELECTOR_SINGLE_ARG_CONSTRAINTS = {
    "use_m",
    "nuse_m",
    "unique_inputs",
    "first_m",
    "not_consecutive",
}

_SELECTOR_SINGLE_ARG_WITH_INT_CONSTRAINTS = {
    "at_step",
    "max_uses",
}

_SELECTOR_DOUBLE_ARG_CONSTRAINTS = {
    "ite_m",
    "depend_m",
    "itn_m",
    "next_m",
    "used_iff_used",
    "mutex_tools",
    "connected_op",
}

_SELECTOR_TEMPLATE_ALIASES: dict[str, tuple[str, tuple[str, ...]]] = {}
for _name in (
    _SELECTOR_SINGLE_ARG_CONSTRAINTS
    | _SELECTOR_SINGLE_ARG_WITH_INT_CONSTRAINTS
    | _SELECTOR_DOUBLE_ARG_CONSTRAINTS
):
    _arg_count = 2 if _name in _SELECTOR_DOUBLE_ARG_CONSTRAINTS else 1
    _auto = tuple("auto" for _ in range(_arg_count))
    _class = tuple("class_transitive" for _ in range(_arg_count))
    _tool = tuple("tool_exact" for _ in range(_arg_count))

    _SELECTOR_TEMPLATE_ALIASES[_name] = (_name, _auto)
    _SELECTOR_TEMPLATE_ALIASES[f"{_name}_c"] = (_name, _class)
    _SELECTOR_TEMPLATE_ALIASES[f"{_name}_tool"] = (_name, _tool)
    if _name.endswith("_m"):
        _short = _name[:-2]
        _SELECTOR_TEMPLATE_ALIASES[f"{_short}_c"] = (_name, _class)
        _SELECTOR_TEMPLATE_ALIASES[f"{_short}_tool"] = (_name, _tool)
    if _name in _SELECTOR_DOUBLE_ARG_CONSTRAINTS:
        _left_class = ("class_transitive", "auto")
        _right_class = ("auto", "class_transitive")
        _left_tool = ("tool_exact", "auto")
        _right_tool = ("auto", "tool_exact")
        _SELECTOR_TEMPLATE_ALIASES[f"{_name}_left_c"] = (_name, _left_class)
        _SELECTOR_TEMPLATE_ALIASES[f"{_name}_right_c"] = (_name, _right_class)
        _SELECTOR_TEMPLATE_ALIASES[f"{_name}_left_tool"] = (_name, _left_tool)
        _SELECTOR_TEMPLATE_ALIASES[f"{_name}_right_tool"] = (_name, _right_tool)
        if _name.endswith("_m"):
            _short = _name[:-2]
            _SELECTOR_TEMPLATE_ALIASES[f"{_short}_left_c"] = (_name, _left_class)
            _SELECTOR_TEMPLATE_ALIASES[f"{_short}_right_c"] = (_name, _right_class)
            _SELECTOR_TEMPLATE_ALIASES[f"{_short}_left_tool"] = (_name, _left_tool)
            _SELECTOR_TEMPLATE_ALIASES[f"{_short}_right_tool"] = (_name, _right_tool)

_LAZY_SUPPORTED_CONSTRAINTS = (
    set(_SELECTOR_TEMPLATE_ALIASES)
    | {
        "use_t",
        "operationInput",
    }
)

_LAZY_NATIVE_CONSTRAINTS = _LAZY_SUPPORTED_CONSTRAINTS | {
    "operation_input",
}

_CONSTRAINT_ATOM_PATTERN = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\((.*)\)\s*$")

_CONSTRAINT_SELECTOR_MODE_BY_TEMPLATE: dict[str, str] = {
    "use_m": "transitive",
    "nuse_m": "transitive",
    "unique_inputs": "transitive",
    "first_m": "transitive",
    "not_consecutive": "transitive",
    "at_step": "transitive",
    "ite_m": "transitive",
    "depend_m": "transitive",
    "itn_m": "transitive",
    "next_m": "transitive",
    "used_iff_used": "transitive",
    "max_uses": "transitive",
    "mutex_tools": "transitive",
    "connected_op": "transitive",
    "operationInput": "transitive",
    "operation_input": "transitive",
}

_SELECTOR_POLICY_LABELS = {
    "auto": "tool or class selector",
    "class_transitive": "abstract class selector",
    "tool_exact": "concrete tool selector",
}


def _strip_constraint_value(value: str, prefix: str) -> str:
    if prefix and value.startswith(prefix):
        return value[len(prefix):]
    if "#" in value:
        return value.rsplit("#", 1)[1]
    return value


def _extract_constraint_selector(
    parameter: Mapping[str, Any],
    *,
    prefix: str,
) -> str:
    if not parameter:
        raise ValueError("empty parameter")

    for raw_values in parameter.values():
        if isinstance(raw_values, str):
            return _strip_constraint_value(raw_values, prefix)
        if isinstance(raw_values, Iterable):
            for raw_value in raw_values:
                return _strip_constraint_value(str(raw_value), prefix)
        break
    raise ValueError("parameter did not contain a selector value")


def _constraint_selector_kind(
    selector: str,
    *,
    tool_ids: set[str],
    operation_ids: set[str],
) -> str:
    if selector in tool_ids:
        return "tool"
    if selector in operation_ids or selector.startswith("operation_"):
        return "operation"
    return "class"


def _constraint_selector_mode(
    constraint_name: str,
    *,
    selector_kind: str,
    selector_policy: str = "auto",
) -> str:
    if selector_policy == "class_transitive":
        return "transitive"
    if selector_policy == "tool_exact":
        return "exact"
    mode = _CONSTRAINT_SELECTOR_MODE_BY_TEMPLATE.get(constraint_name, "transitive")
    if mode == "exact" and selector_kind == "class":
        return "transitive"
    return mode


def _resolve_constraint_template_name(constraint_name: str) -> tuple[str, tuple[str, ...]]:
    return _SELECTOR_TEMPLATE_ALIASES.get(constraint_name, (constraint_name, ("auto",)))


def _parse_constraint_atom(text: str) -> tuple[str, tuple[str | int, ...]]:
    match = _CONSTRAINT_ATOM_PATTERN.fullmatch(text.strip())
    if match is None:
        raise ValueError("expected atom syntax name(arg1, arg2, ...)")

    atom_name = match.group(1)
    raw_args = match.group(2).strip()
    if not raw_args:
        return atom_name, ()

    args: list[str | int] = []
    current: list[str] = []
    quote: str | None = None
    escaped = False
    for char in raw_args:
        if escaped:
            current.append(char)
            escaped = False
            continue
        if quote is not None:
            if char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            else:
                current.append(char)
            continue
        if char in {"'", '"'}:
            quote = char
            continue
        if char == ",":
            token = "".join(current).strip()
            if not token:
                raise ValueError("empty argument")
            args.append(int(token) if token.lstrip("-").isdigit() else token)
            current = []
            continue
        current.append(char)

    if quote is not None:
        raise ValueError("unterminated quoted string")
    if escaped:
        raise ValueError("dangling escape sequence")

    token = "".join(current).strip()
    if not token:
        raise ValueError("empty argument")
    args.append(int(token) if token.lstrip("-").isdigit() else token)
    return atom_name, tuple(args)


def _lazy_allowed_selectors(
    config: SnakeConfig,
    ontology: Ontology,
    tools: tuple[ToolMode, ...],
) -> set[str]:
    allowed_selectors = set(ontology.descendants_of(config.tools_taxonomy_root))
    allowed_selectors.add(config.tools_taxonomy_root)
    for tool in tools:
        allowed_selectors.add(tool.mode_id)
        allowed_selectors.update(tool.taxonomy_operations)
    return allowed_selectors


def _lazy_allowed_data_selectors(
    config: SnakeConfig,
    ontology: Ontology,
    tools: tuple[ToolMode, ...],
) -> set[str]:
    allowed_values = set(ontology.nodes)

    def _add_value(raw_value: str) -> None:
        value = str(raw_value).strip()
        if not value:
            return
        allowed_values.add(value)
        allowed_values.add(_strip_constraint_value(value, config.ontology_prefix))

    for item in config.inputs:
        for values in item.values():
            for value in values:
                _add_value(value)
    for item in config.outputs:
        for values in item.values():
            for value in values:
                _add_value(value)
    for tool in tools:
        for port in (*tool.inputs, *tool.outputs):
            for values in port.dimensions.values():
                for value in values:
                    _add_value(value)

    return allowed_values


def _data_selector_aliases(value: str, *, prefix: str) -> tuple[str, ...]:
    aliases = [value.strip()]
    stripped = _strip_constraint_value(value.strip(), prefix)
    if stripped and stripped not in aliases:
        aliases.append(stripped)
    return tuple(alias for alias in aliases if alias)


def _emit_lazy_constraint(
    writer: _FactWriter,
    *,
    config: SnakeConfig,
    constraint_name: str,
    args: tuple[str | int, ...],
    allowed_selectors: set[str],
    allowed_data_selectors: set[str],
    selector_ids: dict[tuple[str, str], str],
    data_selector_ids: dict[str, str],
    tool_ids: set[str],
    operation_ids: set[str],
    source_name: str,
    index: int,
) -> None:
    base_constraint_name, selector_policies = _resolve_constraint_template_name(constraint_name)

    def _skip(reason: str) -> None:
        writer.emit_comment(f"skipping {source_name} constraint {index}: {reason}")

    def _selector_id_for(selector: str, *, position: int = 0) -> str:
        selector_kind = _constraint_selector_kind(
            selector,
            tool_ids=tool_ids,
            operation_ids=operation_ids,
        )
        selector_policy = selector_policies[position] if position < len(selector_policies) else "auto"
        selector_mode = _constraint_selector_mode(
            base_constraint_name,
            selector_kind=selector_kind,
            selector_policy=selector_policy,
        )
        selector_key = (selector, selector_mode)
        selector_id = selector_ids.get(selector_key)
        if selector_id is None:
            selector_id = f"constraint_selector_{len(selector_ids)}"
            selector_ids[selector_key] = selector_id
            writer.emit_fact("constraint_selector", _quote(selector_id), _quote(selector))
            writer.emit_fact(
                "constraint_selector_kind",
                _quote(selector_id),
                _quote(selector_kind),
            )
            writer.emit_fact(
                "constraint_selector_mode",
                _quote(selector_id),
                _quote(selector_mode),
            )
        return selector_id

    def _data_selector_id_for(raw_selector: str) -> str:
        selector_id = data_selector_ids.get(raw_selector)
        if selector_id is None:
            selector_id = f"constraint_data_selector_{len(data_selector_ids)}"
            data_selector_ids[raw_selector] = selector_id
            for alias in _data_selector_aliases(raw_selector, prefix=config.ontology_prefix):
                writer.emit_fact("constraint_data_selector", _quote(selector_id), _quote(alias))
        return selector_id

    def _selector_arg(position: int) -> str | None:
        if position >= len(args):
            _skip(f"{constraint_name} is missing selector argument {position + 1}")
            return None
        raw_value = args[position]
        if not isinstance(raw_value, str):
            _skip(f"{constraint_name} expects selector argument {position + 1}")
            return None
        selector = _strip_constraint_value(raw_value, prefix=config.ontology_prefix).strip()
        if not selector:
            _skip(f"{constraint_name} has an empty selector argument")
            return None
        if selector not in allowed_selectors:
            _skip(f"unknown selector {selector}")
            return None
        selector_kind = _constraint_selector_kind(
            selector,
            tool_ids=tool_ids,
            operation_ids=operation_ids,
        )
        selector_policy = selector_policies[position] if position < len(selector_policies) else "auto"
        if selector_policy == "class_transitive" and selector_kind != "class":
            _skip(
                f"{constraint_name} expects an abstract class selector, got {selector_kind} {selector}"
            )
            return None
        if selector_policy == "tool_exact" and selector_kind != "tool":
            _skip(
                f"{constraint_name} expects a concrete tool selector, got {selector_kind} {selector}"
            )
            return None
        return selector

    def _data_selector_arg(position: int) -> str | None:
        if position >= len(args):
            _skip(f"{constraint_name} is missing data selector argument {position + 1}")
            return None
        raw_value = args[position]
        if not isinstance(raw_value, str):
            _skip(f"{constraint_name} expects data selector argument {position + 1}")
            return None
        selector = raw_value.strip()
        if not selector:
            _skip(f"{constraint_name} has an empty data selector argument")
            return None
        aliases = _data_selector_aliases(selector, prefix=config.ontology_prefix)
        if not any(alias in allowed_data_selectors for alias in aliases):
            _skip(f"unknown data selector {selector}")
            return None
        return selector

    def _int_arg(position: int, *, minimum: int | None = None) -> int | None:
        if position >= len(args):
            _skip(f"{constraint_name} is missing integer argument {position + 1}")
            return None
        raw_value = args[position]
        if not isinstance(raw_value, int):
            _skip(f"{constraint_name} expects integer argument {position + 1}")
            return None
        if minimum is not None and raw_value < minimum:
            _skip(
                f"{constraint_name} expects argument {position + 1} >= {minimum}"
            )
            return None
        return raw_value

    if base_constraint_name == "use_m":
        if len(args) != 1:
            _skip("use_m expects 1 selector")
            return
        selector = _selector_arg(0)
        if selector is not None:
            writer.emit_fact("constraint_must_use", _quote(_selector_id_for(selector, position=0)))
        return

    if base_constraint_name == "nuse_m":
        if len(args) != 1:
            _skip("nuse_m expects 1 selector")
            return
        selector = _selector_arg(0)
        if selector is not None:
            writer.emit_fact("constraint_must_not_use", _quote(_selector_id_for(selector, position=0)))
        return

    if base_constraint_name == "unique_inputs":
        if len(args) != 1:
            _skip("unique_inputs expects 1 selector")
            return
        selector = _selector_arg(0)
        if selector is not None:
            writer.emit_fact(
                "constraint_unique_inputs",
                _quote(_selector_id_for(selector, position=0)),
            )
        return

    if base_constraint_name == "first_m":
        if len(args) != 1:
            _skip("first_m expects 1 selector")
            return
        selector = _selector_arg(0)
        if selector is not None:
            writer.emit_fact(
                "constraint_first",
                _quote(_selector_id_for(selector, position=0)),
            )
        return

    if base_constraint_name == "not_consecutive":
        if len(args) != 1:
            _skip("not_consecutive expects 1 selector")
            return
        selector = _selector_arg(0)
        if selector is not None:
            writer.emit_fact(
                "constraint_not_consecutive",
                _quote(_selector_id_for(selector, position=0)),
            )
        return

    if base_constraint_name == "use_t":
        if len(args) != 1:
            _skip("use_t expects 1 data selector")
            return
        selector = _data_selector_arg(0)
        if selector is not None:
            writer.emit_fact(
                "constraint_use_data",
                _quote(_data_selector_id_for(selector)),
            )
        return

    if base_constraint_name == "at_step":
        if len(args) != 2:
            _skip("at_step expects selector and step")
            return
        selector = _selector_arg(0)
        step = _int_arg(1, minimum=1)
        if step is not None and selector is not None:
            writer.emit_fact(
                "constraint_tool_at_step",
                str(step),
                _quote(_selector_id_for(selector, position=0)),
            )
        return

    if base_constraint_name == "max_uses":
        if len(args) != 2:
            _skip("max_uses expects selector and limit")
            return
        selector = _selector_arg(0)
        limit = _int_arg(1, minimum=0)
        if selector is not None and limit is not None:
            writer.emit_fact(
                "constraint_max_uses",
                _quote(_selector_id_for(selector, position=0)),
                str(limit),
            )
        return

    if base_constraint_name in {"operation_input", "operationInput"}:
        if len(args) != 2:
            _skip(f"{constraint_name} expects module selector and data selector")
            return
        selector = _selector_arg(0)
        data_selector = _data_selector_arg(1)
        if selector is not None and data_selector is not None:
            writer.emit_fact(
                "constraint_operation_input",
                _quote(_selector_id_for(selector, position=0)),
                _quote(_data_selector_id_for(data_selector)),
            )
        return

    if len(args) != 2:
        _skip(f"{constraint_name} expects 2 selectors")
        return

    selector_a = _selector_arg(0)
    selector_b = _selector_arg(1)
    if selector_a is None or selector_b is None:
        return

    selector_a_id = _quote(_selector_id_for(selector_a, position=0))
    selector_b_id = _quote(_selector_id_for(selector_b, position=1))
    if base_constraint_name == "ite_m":
        writer.emit_fact("constraint_implies_future", selector_a_id, selector_b_id)
    elif base_constraint_name == "depend_m":
        writer.emit_fact("constraint_depends_prior", selector_a_id, selector_b_id)
    elif base_constraint_name == "itn_m":
        writer.emit_fact("constraint_forbid_later", selector_a_id, selector_b_id)
    elif base_constraint_name == "next_m":
        writer.emit_fact("constraint_next", selector_a_id, selector_b_id)
    elif base_constraint_name == "used_iff_used":
        writer.emit_fact("constraint_used_iff_used", selector_a_id, selector_b_id)
    elif base_constraint_name == "mutex_tools":
        writer.emit_fact("constraint_mutex", selector_a_id, selector_b_id)
    elif base_constraint_name == "connected_op":
        writer.emit_fact("constraint_connected", selector_a_id, selector_b_id)
    else:
        _skip(f"unsupported constraint {constraint_name}")


def _emit_lazy_template_constraints(
    writer: _FactWriter,
    *,
    config: SnakeConfig,
    constraints_path,
    constraints: list[Mapping[str, Any]],
    allowed_selectors: set[str],
    allowed_data_selectors: set[str],
    selector_ids: dict[tuple[str, str], str],
    data_selector_ids: dict[str, str],
    tool_ids: set[str],
    operation_ids: set[str],
) -> None:
    writer.emit_comment(
        f"lazy constraint translation from {constraints_path.name}"
    )

    for index, raw_constraint in enumerate(constraints):
        constraint_id = str(raw_constraint.get("constraintid", "")).strip()
        if not constraint_id:
            writer.emit_comment(
                f"skipping {constraints_path.name} constraint {index}: missing constraintid"
            )
            continue
        if constraint_id == "SLTLx":
            writer.emit_comment(
                f"skipping {constraints_path.name} constraint {index}: SLTLx is unsupported"
            )
            continue
        if constraint_id not in _LAZY_SUPPORTED_CONSTRAINTS:
            writer.emit_comment(
                f"skipping {constraints_path.name} constraint {index}: unsupported template {constraint_id}"
            )
            continue

        raw_parameters = raw_constraint.get("parameters") or []
        try:
            selectors = tuple(
                _extract_constraint_selector(parameter, prefix=config.ontology_prefix)
                for parameter in raw_parameters
            )
        except ValueError as exc:
            writer.emit_comment(
                f"skipping {constraints_path.name} constraint {index}: invalid parameters ({exc})"
            )
            continue

        _emit_lazy_constraint(
            writer,
            config=config,
            constraint_name=constraint_id,
            args=selectors,
            allowed_selectors=allowed_selectors,
            allowed_data_selectors=allowed_data_selectors,
            selector_ids=selector_ids,
            data_selector_ids=data_selector_ids,
            tool_ids=tool_ids,
            operation_ids=operation_ids,
            source_name=constraints_path.name,
            index=index,
        )


def _emit_lazy_native_constraints(
    writer: _FactWriter,
    *,
    config: SnakeConfig,
    constraints_path,
    constraints: list[str],
    allowed_selectors: set[str],
    allowed_data_selectors: set[str],
    selector_ids: dict[tuple[str, str], str],
    data_selector_ids: dict[str, str],
    tool_ids: set[str],
    operation_ids: set[str],
) -> None:
    writer.emit_comment(
        f"lazy native constraint translation from {constraints_path.name}"
    )

    for index, raw_constraint in enumerate(constraints):
        if not isinstance(raw_constraint, str):
            writer.emit_comment(
                f"skipping {constraints_path.name} constraint {index}: expected atom string"
            )
            continue
        try:
            constraint_name, args = _parse_constraint_atom(raw_constraint)
        except ValueError as exc:
            writer.emit_comment(
                f"skipping {constraints_path.name} constraint {index}: invalid atom ({exc})"
            )
            continue
        if constraint_name not in _LAZY_NATIVE_CONSTRAINTS:
            writer.emit_comment(
                f"skipping {constraints_path.name} constraint {index}: unsupported native atom {constraint_name}"
            )
            continue
        _emit_lazy_constraint(
            writer,
            config=config,
            constraint_name=constraint_name,
            args=args,
            allowed_selectors=allowed_selectors,
            allowed_data_selectors=allowed_data_selectors,
            selector_ids=selector_ids,
            data_selector_ids=data_selector_ids,
            tool_ids=tool_ids,
            operation_ids=operation_ids,
            source_name=constraints_path.name,
            index=index,
        )


def _load_lazy_constraints(config: SnakeConfig) -> tuple[object, list[Any], str] | None:
    constraints_path = config.constraints_path
    if constraints_path is None or not constraints_path.exists():
        return None

    with constraints_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    constraints = raw.get("constraints", [])
    if not constraints:
        return None

    has_strings = any(isinstance(entry, str) for entry in constraints)
    has_mappings = any(isinstance(entry, Mapping) for entry in constraints)
    if has_strings and has_mappings:
        raise ValueError(
            f"{constraints_path} mixes native atom strings and APE-style constraint objects"
        )
    if has_strings:
        if not all(isinstance(entry, str) for entry in constraints):
            raise ValueError(
                f"{constraints_path} contains unsupported native constraint entries"
            )
        return constraints_path, list(constraints), "native"
    if has_mappings:
        if not all(isinstance(entry, Mapping) for entry in constraints):
            raise ValueError(
                f"{constraints_path} contains unsupported APE-style constraint entries"
            )
        return constraints_path, list(constraints), "template"
    raise ValueError(
        f"{constraints_path} must contain either native atom strings or APE-style constraint objects"
    )


def _tool_selector_ancestors(
    config: SnakeConfig,
    ontology: Ontology,
    tools: tuple[ToolMode, ...],
) -> dict[str, frozenset[str]]:
    tool_taxonomy_nodes = set(ontology.descendants_of(config.tools_taxonomy_root))
    tool_taxonomy_nodes.add(config.tools_taxonomy_root)
    ancestors_by_tool: dict[str, frozenset[str]] = {}
    for tool in tools:
        ancestors = {tool.mode_id}
        for tax_op in tool.taxonomy_operations:
            ancestors.add(tax_op)
            if tax_op in ontology.nodes:
                ancestors.update(
                    ancestor
                    for ancestor in ontology.ancestors_of(tax_op)
                    if ancestor in tool_taxonomy_nodes
                )
        ancestors_by_tool[tool.mode_id] = frozenset(ancestors)
    return ancestors_by_tool


def _collect_lazy_forbidden_tool_ids(
    config: SnakeConfig,
    ontology: Ontology,
    tools: tuple[ToolMode, ...],
) -> set[str]:
    loaded_constraints = _load_lazy_constraints(config)
    if loaded_constraints is None:
        return set()

    allowed_selectors = _lazy_allowed_selectors(config, ontology, tools)
    tool_ids = {tool.mode_id for tool in tools}
    operation_ids = {tax_op for tool in tools for tax_op in tool.taxonomy_operations}
    ancestors_by_tool = _tool_selector_ancestors(config, ontology, tools)
    forbidden_tool_ids: set[str] = set()

    def _mark_forbidden(constraint_name: str, args: tuple[str | int, ...]) -> None:
        base_constraint_name, selector_policies = _resolve_constraint_template_name(constraint_name)
        if base_constraint_name != "nuse_m" or len(args) != 1:
            return
        raw_value = args[0]
        if not isinstance(raw_value, str):
            return
        selector = _strip_constraint_value(raw_value, prefix=config.ontology_prefix).strip()
        if not selector or selector not in allowed_selectors:
            return
        selector_kind = _constraint_selector_kind(
            selector,
            tool_ids=tool_ids,
            operation_ids=operation_ids,
        )
        selector_policy = selector_policies[0] if selector_policies else "auto"
        if selector_policy == "class_transitive" and selector_kind != "class":
            return
        if selector_policy == "tool_exact" and selector_kind != "tool":
            return
        selector_mode = _constraint_selector_mode(
            base_constraint_name,
            selector_kind=selector_kind,
            selector_policy=selector_policy,
        )
        for tool in tools:
            if selector_mode == "exact":
                if selector == tool.mode_id or selector in tool.taxonomy_operations:
                    forbidden_tool_ids.add(tool.mode_id)
            elif selector in ancestors_by_tool.get(tool.mode_id, frozenset()):
                forbidden_tool_ids.add(tool.mode_id)

    _constraints_path, constraints, constraint_kind = loaded_constraints
    if constraint_kind == "template":
        for raw_constraint in constraints:
            constraint_id = str(raw_constraint.get("constraintid", "")).strip()
            if not constraint_id:
                continue
            raw_parameters = raw_constraint.get("parameters") or []
            try:
                selectors = tuple(
                    _extract_constraint_selector(parameter, prefix=config.ontology_prefix)
                    for parameter in raw_parameters
                )
            except ValueError:
                continue
            _mark_forbidden(constraint_id, selectors)
    else:
        for raw_constraint in constraints:
            if not isinstance(raw_constraint, str):
                continue
            try:
                constraint_name, args = _parse_constraint_atom(raw_constraint)
            except ValueError:
                continue
            _mark_forbidden(constraint_name, args)

    return forbidden_tool_ids


def _collect_lazy_selector_lower_bounds(
    config: SnakeConfig,
    ontology: Ontology,
    tools: tuple[ToolMode, ...],
    *,
    tool_min_steps: Mapping[str, int],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    loaded_constraints = _load_lazy_constraints(config)
    if loaded_constraints is None:
        return (), ()

    allowed_selectors = _lazy_allowed_selectors(config, ontology, tools)
    tool_ids = {tool.mode_id for tool in tools}
    operation_ids = {tax_op for tool in tools for tax_op in tool.taxonomy_operations}
    ancestors_by_tool = _tool_selector_ancestors(config, ontology, tools)
    tools_by_id = {tool.mode_id: tool for tool in tools}
    must_use_steps: list[int] = []
    at_step_steps: list[int] = []

    def _matching_tool_steps(constraint_name: str, raw_value: str, position: int) -> list[int]:
        base_constraint_name, selector_policies = _resolve_constraint_template_name(constraint_name)
        selector = _strip_constraint_value(raw_value, prefix=config.ontology_prefix).strip()
        if not selector or selector not in allowed_selectors:
            return []
        selector_kind = _constraint_selector_kind(
            selector,
            tool_ids=tool_ids,
            operation_ids=operation_ids,
        )
        selector_policy = selector_policies[position] if position < len(selector_policies) else "auto"
        if selector_policy == "class_transitive" and selector_kind != "class":
            return []
        if selector_policy == "tool_exact" and selector_kind != "tool":
            return []
        selector_mode = _constraint_selector_mode(
            base_constraint_name,
            selector_kind=selector_kind,
            selector_policy=selector_policy,
        )
        matching_steps: list[int] = []
        for tool_id, min_step in tool_min_steps.items():
            tool = tools_by_id[tool_id]
            if selector_mode == "exact":
                matches_tool = selector == tool.mode_id or selector in tool.taxonomy_operations
            else:
                matches_tool = selector in ancestors_by_tool.get(tool.mode_id, frozenset())
            if matches_tool:
                matching_steps.append(min_step)
        return matching_steps

    _constraints_path, constraints, constraint_kind = loaded_constraints
    if constraint_kind == "template":
        parsed_constraints: list[tuple[str, tuple[str | int, ...]]] = []
        for raw_constraint in constraints:
            constraint_id = str(raw_constraint.get("constraintid", "")).strip()
            if not constraint_id:
                continue
            raw_parameters = raw_constraint.get("parameters") or []
            try:
                selectors = tuple(
                    _extract_constraint_selector(parameter, prefix=config.ontology_prefix)
                    for parameter in raw_parameters
                )
            except ValueError:
                continue
            parsed_constraints.append((constraint_id, selectors))
    else:
        parsed_constraints = []
        for raw_constraint in constraints:
            if not isinstance(raw_constraint, str):
                continue
            try:
                parsed_constraints.append(_parse_constraint_atom(raw_constraint))
            except ValueError:
                continue

    for constraint_name, args in parsed_constraints:
        base_constraint_name, _selector_policies = _resolve_constraint_template_name(constraint_name)
        if base_constraint_name == "use_m" and len(args) == 1 and isinstance(args[0], str):
            matching_steps = _matching_tool_steps(constraint_name, args[0], 0)
            if matching_steps:
                must_use_steps.append(min(matching_steps))
        elif (
            base_constraint_name == "at_step"
            and len(args) == 2
            and isinstance(args[0], str)
            and isinstance(args[1], int)
        ):
            matching_steps = _matching_tool_steps(constraint_name, args[0], 0)
            if matching_steps:
                at_step_steps.append(max(min(matching_steps), args[1]))

    return tuple(must_use_steps), tuple(at_step_steps)


def _candidate_outputs_match_data_selector(
    ontology: Ontology,
    candidate_record: Mapping[str, object],
    data_selector: str,
) -> bool:
    for output_port in tuple(candidate_record["output_ports"]):
        for values in output_port["port_values_by_dimension"].values():
            for actual_value in values:
                if actual_value == data_selector or data_selector in ontology.ancestors_of(actual_value):
                    return True
    return False


def _collect_lazy_backward_relevant_candidates(
    config: SnakeConfig,
    ontology: Ontology,
    tools: tuple[ToolMode, ...],
    *,
    candidate_records: Iterable[Mapping[str, object]],
    reverse_edges: Mapping[str, set[str]],
    direct_goal_candidates: set[str],
) -> tuple[set[str], dict[str, int]]:
    """Collect exact backward-relevant candidates for use_all_generated_data=ALL.

    The anchors are the candidates that can directly satisfy a terminal
    requirement:
    - goal-producing candidates,
    - selector-matching candidates for positive tool-use constraints,
    - candidates that can produce required data-selector artifacts,
    - candidates whose tools can witness operation_input constraints.
    """

    loaded_constraints = _load_lazy_constraints(config)
    if loaded_constraints is None:
        return set(direct_goal_candidates), {
            candidate_id: 0 for candidate_id in direct_goal_candidates
        }

    allowed_selectors = _lazy_allowed_selectors(config, ontology, tools)
    allowed_data_selectors = _lazy_allowed_data_selectors(config, ontology, tools)
    tool_ids = {tool.mode_id for tool in tools}
    operation_ids = {tax_op for tool in tools for tax_op in tool.taxonomy_operations}
    ancestors_by_tool = _tool_selector_ancestors(config, ontology, tools)
    tools_by_id = {tool.mode_id: tool for tool in tools}
    candidate_records_by_id = {
        str(record["candidate_id"]): record
        for record in candidate_records
    }
    candidate_ids_by_tool_id: dict[str, set[str]] = defaultdict(set)
    for candidate_id, record in candidate_records_by_id.items():
        tool_id = str(record["tool"].mode_id)
        candidate_ids_by_tool_id[tool_id].add(candidate_id)

    def _matching_tool_ids(
        constraint_name: str,
        raw_value: str,
        position: int,
    ) -> frozenset[str]:
        base_constraint_name, selector_policies = _resolve_constraint_template_name(constraint_name)
        selector = _strip_constraint_value(raw_value, prefix=config.ontology_prefix).strip()
        if not selector or selector not in allowed_selectors:
            return frozenset()
        selector_kind = _constraint_selector_kind(
            selector,
            tool_ids=tool_ids,
            operation_ids=operation_ids,
        )
        selector_policy = selector_policies[position] if position < len(selector_policies) else "auto"
        if selector_policy == "class_transitive" and selector_kind != "class":
            return frozenset()
        if selector_policy == "tool_exact" and selector_kind != "tool":
            return frozenset()
        selector_mode = _constraint_selector_mode(
            base_constraint_name,
            selector_kind=selector_kind,
            selector_policy=selector_policy,
        )
        matches: set[str] = set()
        for tool_id in tools_by_id:
            tool = tools_by_id[tool_id]
            if selector_mode == "exact":
                matches_tool = selector == tool.mode_id or selector in tool.taxonomy_operations
            else:
                matches_tool = selector in ancestors_by_tool.get(tool.mode_id, frozenset())
            if matches_tool:
                matches.add(tool_id)
        return frozenset(matches)

    def _matching_data_selector(raw_value: str) -> str | None:
        selector = raw_value.strip()
        if not selector:
            return None
        aliases = _data_selector_aliases(selector, prefix=config.ontology_prefix)
        for alias in aliases:
            if alias in allowed_data_selectors:
                return alias
        return None

    _constraints_path, constraints, constraint_kind = loaded_constraints
    if constraint_kind == "template":
        parsed_constraints: list[tuple[str, tuple[str | int, ...]]] = []
        for raw_constraint in constraints:
            constraint_id = str(raw_constraint.get("constraintid", "")).strip()
            if not constraint_id:
                continue
            raw_parameters = raw_constraint.get("parameters") or []
            try:
                selectors = tuple(
                    _extract_constraint_selector(parameter, prefix=config.ontology_prefix)
                    for parameter in raw_parameters
                )
            except ValueError:
                continue
            parsed_constraints.append((constraint_id, selectors))
    else:
        parsed_constraints = []
        for raw_constraint in constraints:
            if not isinstance(raw_constraint, str):
                continue
            try:
                parsed_constraints.append(_parse_constraint_atom(raw_constraint))
            except ValueError:
                continue

    anchor_candidate_ids: set[str] = set(direct_goal_candidates)

    positive_tool_constraints = {
        "use_m",
        "at_step",
        "first_m",
        "unique_inputs",
        "max_uses",
        "used_iff_used",
        "connected_op",
        "ite_m",
        "depend_m",
        "itn_m",
        "next_m",
    }

    for constraint_name, args in parsed_constraints:
        base_constraint_name, _selector_policies = _resolve_constraint_template_name(constraint_name)
        if base_constraint_name in positive_tool_constraints:
            for position, arg in enumerate(args):
                if not isinstance(arg, str):
                    continue
                for tool_id in _matching_tool_ids(constraint_name, arg, position):
                    anchor_candidate_ids.update(candidate_ids_by_tool_id.get(tool_id, set()))
        elif base_constraint_name == "operation_input":
            if len(args) == 2 and isinstance(args[0], str):
                for tool_id in _matching_tool_ids(constraint_name, args[0], 0):
                    anchor_candidate_ids.update(candidate_ids_by_tool_id.get(tool_id, set()))
            if len(args) == 2 and isinstance(args[1], str):
                data_selector = _matching_data_selector(args[1])
                if data_selector is not None:
                    for candidate_id, record in candidate_records_by_id.items():
                        if _candidate_outputs_match_data_selector(ontology, record, data_selector):
                            anchor_candidate_ids.add(candidate_id)
        elif base_constraint_name == "use_t":
            if len(args) == 1 and isinstance(args[0], str):
                data_selector = _matching_data_selector(args[0])
                if data_selector is not None:
                    for candidate_id, record in candidate_records_by_id.items():
                        if _candidate_outputs_match_data_selector(ontology, record, data_selector):
                            anchor_candidate_ids.add(candidate_id)

    backward_relevant_candidates: set[str] = set(anchor_candidate_ids)
    min_anchor_distance_by_candidate: dict[str, int] = {
        candidate_id: 0 for candidate_id in anchor_candidate_ids
    }
    frontier: deque[str] = deque(sorted(anchor_candidate_ids))
    while frontier:
        consumer_candidate = frontier.popleft()
        next_distance = min_anchor_distance_by_candidate[consumer_candidate] + 1
        for producer_candidate in sorted(reverse_edges.get(consumer_candidate, set())):
            if producer_candidate in backward_relevant_candidates:
                continue
            backward_relevant_candidates.add(producer_candidate)
            min_anchor_distance_by_candidate[producer_candidate] = next_distance
            frontier.append(producer_candidate)

    return backward_relevant_candidates, min_anchor_distance_by_candidate


def _collect_lazy_exact_prefix_lower_bound(
    config: SnakeConfig,
    ontology: Ontology,
    tools: tuple[ToolMode, ...],
    *,
    candidate_records: Iterable[Mapping[str, object]],
    workflow_bindable_ports: Mapping[str, set[int]],
    produced_bindable_ports: Mapping[str, Mapping[int, set[str]]],
    query_goal_candidates: set[str],
    max_exact_horizon: int = 2,
) -> int:
    """Compute a small exact lower bound for early lazy horizons.

    This is intentionally limited to the first two horizons. It performs an
    exact candidate-level feasibility check for 1-step and 2-step workflows
    using the already-computed workflow/producers bindability surface. If no
    such workflow can satisfy the must-use selectors plus the goal within the
    tested horizons, later solving can safely skip them entirely.
    """

    if config.solution_length_max <= 1 or max_exact_horizon < 1:
        return 1

    loaded_constraints = _load_lazy_constraints(config)
    if loaded_constraints is None:
        return 1

    allowed_selectors = _lazy_allowed_selectors(config, ontology, tools)
    tool_ids = {tool.mode_id for tool in tools}
    operation_ids = {tax_op for tool in tools for tax_op in tool.taxonomy_operations}
    ancestors_by_tool = _tool_selector_ancestors(config, ontology, tools)
    tools_by_id = {tool.mode_id: tool for tool in tools}

    def _matching_tool_ids(
        constraint_name: str,
        raw_value: str,
        position: int,
    ) -> frozenset[str]:
        base_constraint_name, selector_policies = _resolve_constraint_template_name(constraint_name)
        selector = _strip_constraint_value(raw_value, prefix=config.ontology_prefix).strip()
        if not selector or selector not in allowed_selectors:
            return frozenset()
        selector_kind = _constraint_selector_kind(
            selector,
            tool_ids=tool_ids,
            operation_ids=operation_ids,
        )
        selector_policy = selector_policies[position] if position < len(selector_policies) else "auto"
        if selector_policy == "class_transitive" and selector_kind != "class":
            return frozenset()
        if selector_policy == "tool_exact" and selector_kind != "tool":
            return frozenset()
        selector_mode = _constraint_selector_mode(
            base_constraint_name,
            selector_kind=selector_kind,
            selector_policy=selector_policy,
        )
        matches: set[str] = set()
        for tool_id in tools_by_id:
            tool = tools_by_id[tool_id]
            if selector_mode == "exact":
                matches_tool = selector == tool.mode_id or selector in tool.taxonomy_operations
            else:
                matches_tool = selector in ancestors_by_tool.get(tool.mode_id, frozenset())
            if matches_tool:
                matches.add(tool_id)
        return frozenset(matches)

    _constraints_path, constraints, constraint_kind = loaded_constraints
    if constraint_kind == "template":
        parsed_constraints: list[tuple[str, tuple[str | int, ...]]] = []
        for raw_constraint in constraints:
            constraint_id = str(raw_constraint.get("constraintid", "")).strip()
            if not constraint_id:
                continue
            raw_parameters = raw_constraint.get("parameters") or []
            try:
                selectors = tuple(
                    _extract_constraint_selector(parameter, prefix=config.ontology_prefix)
                    for parameter in raw_parameters
                )
            except ValueError:
                continue
            parsed_constraints.append((constraint_id, selectors))
    else:
        parsed_constraints = []
        for raw_constraint in constraints:
            if not isinstance(raw_constraint, str):
                continue
            try:
                parsed_constraints.append(_parse_constraint_atom(raw_constraint))
            except ValueError:
                continue

    must_use_selector_tools: list[frozenset[str]] = []
    at_step_requirements: dict[int, list[frozenset[str]]] = defaultdict(list)
    for constraint_name, args in parsed_constraints:
        base_constraint_name, _selector_policies = _resolve_constraint_template_name(constraint_name)
        if base_constraint_name == "use_m" and len(args) == 1 and isinstance(args[0], str):
            matching_tools = _matching_tool_ids(constraint_name, args[0], 0)
            if matching_tools:
                must_use_selector_tools.append(matching_tools)
        elif (
            base_constraint_name == "at_step"
            and len(args) == 2
            and isinstance(args[0], str)
            and isinstance(args[1], int)
        ):
            matching_tools = _matching_tool_ids(constraint_name, args[0], 0)
            if matching_tools:
                at_step_requirements[int(args[1])].append(matching_tools)

    candidate_tool_ids = {
        str(record["candidate_id"]): str(record["tool"].mode_id)
        for record in candidate_records
    }
    candidate_input_ports = {
        str(record["candidate_id"]): tuple(int(port["port_idx"]) for port in tuple(record["input_ports"]))
        for record in candidate_records
    }
    candidate_must_use_mask: dict[str, int] = {}
    for candidate_id, tool_id in candidate_tool_ids.items():
        mask = 0
        for index, matching_tools in enumerate(must_use_selector_tools):
            if tool_id in matching_tools:
                mask |= 1 << index
        candidate_must_use_mask[candidate_id] = mask
    full_must_use_mask = (1 << len(must_use_selector_tools)) - 1

    def _matches_step_constraints(candidate_id: str, step_index: int) -> bool:
        tool_id = candidate_tool_ids[candidate_id]
        return all(
            tool_id in matching_tools
            for matching_tools in at_step_requirements.get(step_index, ())
        )

    def _workflow_feasible(candidate_id: str) -> bool:
        return all(
            port_idx in workflow_bindable_ports.get(candidate_id, set())
            for port_idx in candidate_input_ports[candidate_id]
        )

    def _sequence_step_two_feasible(first_candidate: str, second_candidate: str) -> bool:
        for port_idx in candidate_input_ports[second_candidate]:
            if port_idx in workflow_bindable_ports.get(second_candidate, set()):
                continue
            if first_candidate not in produced_bindable_ports.get(second_candidate, {}).get(port_idx, set()):
                return False
        return True

    feasible_step_one_candidates = [
        candidate_id
        for candidate_id in candidate_tool_ids
        if _workflow_feasible(candidate_id)
    ]

    if 1 not in at_step_requirements or feasible_step_one_candidates:
        for candidate_id in feasible_step_one_candidates:
            if not _matches_step_constraints(candidate_id, 1):
                continue
            if full_must_use_mask and candidate_must_use_mask[candidate_id] != full_must_use_mask:
                continue
            if candidate_id in query_goal_candidates:
                return 1

    if config.solution_length_max <= 2 or max_exact_horizon < 2:
        return 2

    if any(required_step > 2 for required_step in at_step_requirements):
        return 3

    for first_candidate in feasible_step_one_candidates:
        if not _matches_step_constraints(first_candidate, 1):
            continue
        first_mask = candidate_must_use_mask[first_candidate]
        first_has_goal = first_candidate in query_goal_candidates
        for second_candidate in candidate_tool_ids:
            if not _matches_step_constraints(second_candidate, 2):
                continue
            if not _sequence_step_two_feasible(first_candidate, second_candidate):
                continue
            combined_mask = first_mask | candidate_must_use_mask[second_candidate]
            if combined_mask != full_must_use_mask:
                continue
            if first_has_goal or second_candidate in query_goal_candidates:
                return 2

    return 3


def _emit_lazy_constraints(
    writer: _FactWriter,
    config: SnakeConfig,
    ontology: Ontology,
    tools: tuple[ToolMode, ...],
) -> None:
    allowed_selectors = _lazy_allowed_selectors(config, ontology, tools)
    allowed_data_selectors = _lazy_allowed_data_selectors(config, ontology, tools)
    selector_ids: dict[tuple[str, str], str] = {}
    data_selector_ids: dict[str, str] = {}
    tool_ids = {tool.mode_id for tool in tools}
    operation_ids = {tax_op for tool in tools for tax_op in tool.taxonomy_operations}
    loaded_constraints = _load_lazy_constraints(config)
    if loaded_constraints is None:
        return

    constraints_path, constraints, constraint_kind = loaded_constraints
    if constraint_kind == "template":
        _emit_lazy_template_constraints(
            writer,
            config=config,
            constraints_path=constraints_path,
            constraints=constraints,
            allowed_selectors=allowed_selectors,
            allowed_data_selectors=allowed_data_selectors,
            selector_ids=selector_ids,
            data_selector_ids=data_selector_ids,
            tool_ids=tool_ids,
            operation_ids=operation_ids,
        )
    else:
        _emit_lazy_native_constraints(
            writer,
            config=config,
            constraints_path=constraints_path,
            constraints=constraints,
            allowed_selectors=allowed_selectors,
            allowed_data_selectors=allowed_data_selectors,
            selector_ids=selector_ids,
            data_selector_ids=data_selector_ids,
            tool_ids=tool_ids,
            operation_ids=operation_ids,
        )

    ancestors_by_tool = _tool_selector_ancestors(config, ontology, tools)
    for (selector, selector_mode), selector_id in sorted(selector_ids.items()):
        for tool in tools:
            matches_tool = False
            if selector_mode == "exact":
                matches_tool = selector == tool.mode_id or selector in tool.taxonomy_operations
            else:
                matches_tool = selector in ancestors_by_tool.get(tool.mode_id, frozenset())
            if matches_tool:
                writer.emit_fact(
                    "lazy_constraint_selector_matches_tool",
                    _quote(selector_id),
                    _quote(tool.mode_id),
                )


def _lazy_port_expansion(
    resolver: _ExpansionResolver,
    dims: Mapping[str, tuple[str, ...]],
    *,
    expand_outputs: bool,
) -> tuple[tuple[tuple[str, str], ...], int]:
    """Expand one port without materializing cross-products across ports."""

    port_values: list[tuple[str, str]] = []
    variant_cardinality = 1

    for dim, values in sorted(dims.items()):
        expanded_values = _dedupe_stable(
            expanded_value
            for value in values
            for expanded_value in resolver.expanded_values(
                dim,
                value,
                expand_outputs=expand_outputs,
            )
        )
        variant_cardinality *= max(len(expanded_values), 1)
        port_values.extend((dim, expanded_value) for expanded_value in expanded_values)

    return tuple(port_values), variant_cardinality


def _group_port_values_by_dimension(
    port_values: tuple[tuple[str, str], ...],
) -> dict[str, tuple[str, ...]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for dim, value in port_values:
        grouped[dim].append(value)
    return {
        dim: tuple(values)
        for dim, values in grouped.items()
    }


def _output_port_group_key(
    output_port: Mapping[str, object],
) -> tuple[tuple[tuple[str, tuple[str, ...]], ...], tuple[tuple[str, tuple[str, ...]], ...]]:
    declared_dims = output_port["declared_dims"]
    port_values_by_dimension = output_port["port_values_by_dimension"]
    assert isinstance(declared_dims, Mapping)
    assert isinstance(port_values_by_dimension, Mapping)
    return (
        tuple((str(dim), tuple(str(value) for value in values)) for dim, values in sorted(declared_dims.items())),
        tuple(
            (str(dim), tuple(str(value) for value in values))
            for dim, values in sorted(port_values_by_dimension.items())
        ),
    )


def _compress_duplicate_output_ports(
    output_ports: tuple[dict[str, object], ...],
) -> tuple[dict[str, object], ...]:
    grouped: list[dict[str, object]] = []
    grouped_by_key: dict[
        tuple[tuple[tuple[str, tuple[str, ...]], ...], tuple[tuple[str, tuple[str, ...]], ...]],
        dict[str, object],
    ] = {}

    for output_port in output_ports:
        group_key = _output_port_group_key(output_port)
        existing = grouped_by_key.get(group_key)
        multiplicity = int(output_port.get("multiplicity", 1))
        if existing is None:
            grouped_port = {
                **output_port,
                "multiplicity": multiplicity,
            }
            grouped.append(grouped_port)
            grouped_by_key[group_key] = grouped_port
        else:
            existing["multiplicity"] = int(existing.get("multiplicity", 1)) + multiplicity

    return tuple(grouped)


def _workflow_input_matches_lazy_port(
    ontology: Ontology,
    workflow_input: Mapping[str, tuple[str, ...]],
    port_values_by_dimension: Mapping[str, tuple[str, ...]],
) -> bool:
    for dim, accepted_values in port_values_by_dimension.items():
        actual_values = workflow_input.get(dim, ())
        # A workflow input that omits a dimension leaves that dimension
        # unconstrained rather than making the port incompatible.
        if not actual_values:
            continue
        if not any(
            actual_value in ontology.ancestors_of(required_value)
            for actual_value in actual_values
            for required_value in accepted_values
        ):
            return False
    return True


def _compute_lazy_candidate_min_steps(
    candidate_records: Iterable[Mapping[str, object]],
    workflow_bindable_ports: Mapping[str, set[int]],
    produced_bindable_ports: Mapping[str, Mapping[int, set[str]]],
) -> dict[str, int]:
    """Compute the earliest feasible step for each lazy candidate.

    A candidate can run at step 1 if every input port can be bound from a
    workflow input. Otherwise, each non-workflow-bound port must be fed by a
    producer candidate from a strictly earlier step.
    """
    candidate_input_ports: dict[str, tuple[int, ...]] = {}
    for record in candidate_records:
        candidate_id = str(record["candidate_id"])
        candidate_input_ports[candidate_id] = tuple(
            int(port["port_idx"])
            for port in tuple(record["input_ports"])
        )

    min_required_producers_by_candidate: dict[str, int] = {}
    for candidate_id, input_ports in candidate_input_ports.items():
        uncovered_ports = tuple(
            port_idx
            for port_idx in input_ports
            if port_idx not in workflow_bindable_ports.get(candidate_id, set())
        )
        if not uncovered_ports:
            min_required_producers_by_candidate[candidate_id] = 0
            continue

        port_bit_index = {port_idx: bit for bit, port_idx in enumerate(uncovered_ports)}
        full_mask = (1 << len(uncovered_ports)) - 1
        producer_to_mask: dict[str, int] = {}
        for port_idx in uncovered_ports:
            for producer_candidate in produced_bindable_ports.get(candidate_id, {}).get(port_idx, set()):
                producer_to_mask[producer_candidate] = (
                    producer_to_mask.get(producer_candidate, 0)
                    | (1 << port_bit_index[port_idx])
                )
        producer_masks = {mask for mask in producer_to_mask.values() if mask}
        best_cover = len(uncovered_ports)
        frontier = {0: 0}
        for producer_mask in sorted(producer_masks):
            next_frontier = dict(frontier)
            for covered_mask, used_count in frontier.items():
                new_mask = covered_mask | producer_mask
                new_count = used_count + 1
                old_count = next_frontier.get(new_mask)
                if old_count is None or new_count < old_count:
                    next_frontier[new_mask] = new_count
            frontier = next_frontier
        best_cover = frontier.get(full_mask, best_cover)
        min_required_producers_by_candidate[candidate_id] = best_cover

    min_step_by_candidate: dict[str, int] = {}
    changed = True
    while changed:
        changed = False
        for candidate_id, input_ports in candidate_input_ports.items():
            if not input_ports:
                candidate_step = 1
            else:
                port_steps: list[int] = []
                for port_idx in input_ports:
                    feasible_steps: list[int] = []
                    if port_idx in workflow_bindable_ports.get(candidate_id, set()):
                        feasible_steps.append(1)
                    for producer_candidate in produced_bindable_ports.get(candidate_id, {}).get(port_idx, set()):
                        producer_step = min_step_by_candidate.get(producer_candidate)
                        if producer_step is not None:
                            feasible_steps.append(producer_step + 1)
                    if not feasible_steps:
                        port_steps = []
                        break
                    port_steps.append(min(feasible_steps))
                if not port_steps:
                    continue
                candidate_step = max(
                    max(port_steps),
                    1 + min_required_producers_by_candidate.get(candidate_id, 0),
                )

            current_step = min_step_by_candidate.get(candidate_id)
            if current_step is None or candidate_step < current_step:
                min_step_by_candidate[candidate_id] = candidate_step
                changed = True

    return min_step_by_candidate


def _lazy_output_matches_lazy_input(
    output_values_by_dimension: Mapping[str, tuple[str, ...]],
    input_values_by_dimension: Mapping[str, tuple[str, ...]],
) -> bool:
    for dim, required_values in input_values_by_dimension.items():
        produced_values = output_values_by_dimension.get(dim, ())
        if not produced_values:
            return False
        if not set(required_values).intersection(produced_values):
            return False
    return True


def _lazy_output_matches_lazy_input_fset(
    output_fsets: Mapping[str, frozenset[str]],
    input_fsets: Mapping[str, frozenset[str]],
) -> bool:
    for dim, required_fset in input_fsets.items():
        produced_fset = output_fsets.get(dim)
        if not produced_fset:
            return False
        if required_fset.isdisjoint(produced_fset):
            return False
    return True


def _compress_lazy_output_choice_values(
    ontology: Ontology,
    output_values_by_dimension: Mapping[str, tuple[str, ...]],
    output_fsets: Mapping[str, frozenset[str]],
    bindable_input_ports: tuple[Mapping[str, object], ...],
    goal_port_values: tuple[Mapping[str, tuple[str, ...]], ...],
    goal_fsets: tuple[Mapping[str, frozenset[str]], ...],
    *,
    preserve_goal_profiles: bool,
) -> dict[str, tuple[str, ...]]:
    globally_bindable_input_ports = tuple(
        input_port
        for input_port in bindable_input_ports
        if _lazy_output_matches_lazy_input_fset(
            output_fsets,
            input_port["port_values_fset"],
        )
    )
    globally_bindable_goal_ids = tuple(
        goal_id
        for goal_id, goal_fset in enumerate(goal_fsets)
        if _lazy_output_matches_lazy_input_fset(output_fsets, goal_fset)
    )

    default_consumer_signatures_by_dimension: dict[str, set[int]] = defaultdict(set)
    consumer_signatures_by_dimension_value: dict[str, dict[str, set[int]]] = defaultdict(
        lambda: defaultdict(set)
    )
    for input_port in globally_bindable_input_ports:
        signature_id = int(input_port["signature_id"])
        input_dims = input_port.get("signature_requirements", input_port["port_values_by_dimension"])
        assert isinstance(input_dims, Mapping)
        for dim in output_values_by_dimension:
            required_values = input_dims.get(dim)
            if required_values is None:
                default_consumer_signatures_by_dimension[dim].add(signature_id)
                continue
            for value in required_values:
                consumer_signatures_by_dimension_value[dim][value].add(signature_id)

    default_goal_ids_by_dimension: dict[str, set[int]] = defaultdict(set)
    goal_ids_by_dimension_value: dict[str, dict[str, set[int]]] = defaultdict(
        lambda: defaultdict(set)
    )
    for goal_id in globally_bindable_goal_ids:
        goal_dims = goal_port_values[goal_id]
        for dim in output_values_by_dimension:
            required_values = goal_dims.get(dim)
            if required_values is None:
                default_goal_ids_by_dimension[dim].add(goal_id)
                continue
            for value in required_values:
                goal_ids_by_dimension_value[dim][value].add(goal_id)

    compressed: dict[str, tuple[str, ...]] = {}
    for dim, values in output_values_by_dimension.items():
        if len(values) <= 1:
            compressed[dim] = values
            continue

        representatives: dict[tuple[tuple[int, ...], tuple[int, ...]], str] = {}
        for value in values:
            consumer_profile = tuple(
                sorted(
                    default_consumer_signatures_by_dimension.get(dim, set())
                    | consumer_signatures_by_dimension_value[dim].get(value, set())
                )
            )
            goal_profile = tuple(
                sorted(
                    default_goal_ids_by_dimension.get(dim, set())
                    | goal_ids_by_dimension_value[dim].get(value, set())
                )
            )
            profile_key = (
                consumer_profile,
                goal_profile if preserve_goal_profiles else (),
            )
            if profile_key in representatives:
                representatives[profile_key] = _prefer_less_specific_value(
                    ontology,
                    representatives[profile_key],
                    value,
                )
            else:
                representatives[profile_key] = value

        compressed[dim] = tuple(representatives.values())

    return compressed


def _lazy_dim_values_cache_key(
    dim_values: Mapping[str, Iterable[str]],
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    return tuple(
        sorted(
            (str(dim), tuple(str(value) for value in values))
            for dim, values in dim_values.items()
        )
    )



def build_lazy_fact_bundle(
    config: SnakeConfig,
    ontology: Ontology,
    tools: tuple[ToolMode, ...],
) -> FactBundle:
    """Build a diagnostic lazy candidate bundle without full variant expansion."""
    roots = _build_roots(config, ontology)
    resolver = _ExpansionResolver(ontology, roots, "python")
    tool_stats: list[ToolExpansionStat] = []
    candidate_records: list[dict[str, object]] = []

    lazy_offset: dict[str, int] = defaultdict(int)

    forbidden_tool_ids = _collect_lazy_forbidden_tool_ids(config, ontology, tools)
    candidate_source_tools = tuple(
        tool
        for tool in tools
        if tool.mode_id not in forbidden_tool_ids
    )

    for tool in candidate_source_tools:
        candidate_index = lazy_offset[tool.mode_id]
        candidate_id = f"{tool.mode_id}_lc{candidate_index}"
        lazy_offset[tool.mode_id] += 1

        input_port_value_counts: list[int] = []
        output_port_value_counts: list[int] = []
        input_variant_count = 1
        output_variant_count = 1
        lazy_input_value_count = 0
        lazy_output_value_count = 0
        input_ports: list[dict[str, object]] = []
        output_ports: list[dict[str, object]] = []

        for port_idx, port in enumerate(tool.inputs):
            port_values, port_variant_count = _lazy_port_expansion(
                resolver,
                _normalize_dim_map(port.dimensions),
                expand_outputs=False,
            )
            port_values_by_dimension = _group_port_values_by_dimension(port_values)
            workflow_input_matches = [
                wf_index
                for wf_index, workflow_input in enumerate(config.inputs)
                if _workflow_input_matches_lazy_port(ontology, workflow_input, port_values_by_dimension)
            ]
            input_ports.append(
                {
                    "port_idx": port_idx,
                    "port_values": port_values,
                    "port_values_by_dimension": port_values_by_dimension,
                    "port_values_fset": {dim: frozenset(vals) for dim, vals in port_values_by_dimension.items()},
                    "workflow_input_matches": workflow_input_matches,
                }
            )
            emitted_count = len(port_values)
            input_port_value_counts.append(emitted_count)
            lazy_input_value_count += emitted_count
            input_variant_count *= port_variant_count

        for port_idx, port in enumerate(tool.outputs):
            declared_dims = _normalize_dim_map(port.dimensions)
            port_values, port_variant_count = _lazy_port_expansion(
                resolver,
                declared_dims,
                expand_outputs=True,
            )
            port_values_by_dimension = _group_port_values_by_dimension(port_values)
            output_ports.append(
                {
                    "port_idx": port_idx,
                    "declared_dims": declared_dims,
                    "port_values_by_dimension": port_values_by_dimension,
                    "port_values_fset": {dim: frozenset(vals) for dim, vals in port_values_by_dimension.items()},
                }
            )
            emitted_count = sum(len(values) for values in port_values_by_dimension.values())
            output_port_value_counts.append(emitted_count)
            lazy_output_value_count += emitted_count
            output_variant_count *= port_variant_count

        tool_stats.append(
            ToolExpansionStat(
                tool_id=tool.mode_id,
                tool_label=tool.label,
                input_ports=len(tool.inputs),
                output_ports=len(tool.outputs),
                input_variant_count=input_variant_count,
                output_variant_count=output_variant_count,
                lazy_input_value_count=lazy_input_value_count,
                lazy_output_value_count=lazy_output_value_count,
                lazy_input_port_value_counts=tuple(input_port_value_counts),
                lazy_output_port_value_counts=tuple(output_port_value_counts),
                lazy_cross_product_estimate=input_variant_count * output_variant_count,
            )
        )
        candidate_records.append(
            {
                "tool": tool,
                "candidate_id": candidate_id,
                "input_ports": tuple(input_ports),
                "output_ports": _compress_duplicate_output_ports(tuple(output_ports)),
            }
        )

    goal_port_values: list[dict[str, tuple[str, ...]]] = []
    for goal_item in config.outputs:
        goal_dims: dict[str, tuple[str, ...]] = {}
        for dim, values in sorted(goal_item.items()):
            goal_dims[dim] = _dedupe_stable(
                expanded_value
                for value in values
                for expanded_value in resolver.expanded_values(
                    dim,
                    value,
                    expand_outputs=True,
                )
                )
        goal_port_values.append(goal_dims)
    goal_port_values_tuple = tuple(goal_port_values)
    goal_fsets: tuple[dict[str, frozenset[str]], ...] = tuple(
        {dim: frozenset(vals) for dim, vals in g.items()} for g in goal_port_values
    )

    _t0 = perf_counter()

    bindable_pairs: set[tuple[str, int, str, int]] = set()
    reverse_edges: dict[str, set[str]] = defaultdict(set)
    candidate_records_by_id = {
        str(record["candidate_id"]): record
        for record in candidate_records
    }
    workflow_inputs = tuple(
        {
            str(dim): tuple(str(value) for value in values)
            for dim, values in item.items()
        }
        for item in config.inputs
    )
    direct_goal_candidates: set[str] = set()

    for record in candidate_records:
        candidate_id = str(record["candidate_id"])
        output_ports = tuple(record["output_ports"])
        for output_port in output_ports:
            output_fset = output_port["port_values_fset"]
            if any(_lazy_output_matches_lazy_input_fset(output_fset, gf) for gf in goal_fsets):
                direct_goal_candidates.add(candidate_id)

    _t1 = perf_counter()

    for producer_record in candidate_records:
        producer_candidate = str(producer_record["candidate_id"])
        for output_port in tuple(producer_record["output_ports"]):
            producer_port = int(output_port["port_idx"])
            output_fset = output_port["port_values_fset"]
            for consumer_record in candidate_records:
                consumer_candidate = str(consumer_record["candidate_id"])
                for input_port in tuple(consumer_record["input_ports"]):
                    consumer_port = int(input_port["port_idx"])
                    if _lazy_output_matches_lazy_input_fset(output_fset, input_port["port_values_fset"]):
                        bindable_pairs.add(
                            (producer_candidate, producer_port, consumer_candidate, consumer_port)
                        )
                        reverse_edges[consumer_candidate].add(producer_candidate)

    _t2 = perf_counter()

    workflow_bindable_ports: dict[str, set[int]] = defaultdict(set)
    produced_bindable_ports: dict[str, dict[int, set[str]]] = defaultdict(lambda: defaultdict(set))
    for record in candidate_records:
        candidate_id = str(record["candidate_id"])
        for input_port in tuple(record["input_ports"]):
            port_idx = int(input_port["port_idx"])
            if any(
                _workflow_input_matches_lazy_port(
                    ontology,
                    workflow_input,
                    input_port["port_values_by_dimension"],
                )
                for workflow_input in workflow_inputs
            ):
                workflow_bindable_ports[candidate_id].add(port_idx)

    for producer_candidate, _, consumer_candidate, consumer_port in bindable_pairs:
        produced_bindable_ports[consumer_candidate][consumer_port].add(producer_candidate)

    relevant_candidates: set[str] = set()
    frontier: deque[str] = deque()
    for record in candidate_records:
        candidate_id = str(record["candidate_id"])
        input_ports = tuple(record["input_ports"])
        if all(int(port["port_idx"]) in workflow_bindable_ports[candidate_id] for port in input_ports):
            relevant_candidates.add(candidate_id)
            frontier.append(candidate_id)

    while frontier:
        producer_candidate = frontier.popleft()
        for consumer_candidate, ports_by_source in produced_bindable_ports.items():
            if consumer_candidate in relevant_candidates:
                continue
            consumer_record = candidate_records_by_id[consumer_candidate]
            input_ports = tuple(consumer_record["input_ports"])
            if all(
                int(port["port_idx"]) in workflow_bindable_ports[consumer_candidate]
                or any(
                    source_candidate in relevant_candidates
                    for source_candidate in produced_bindable_ports[consumer_candidate].get(int(port["port_idx"]), set())
                )
                for port in input_ports
            ):
                relevant_candidates.add(consumer_candidate)
                frontier.append(consumer_candidate)

    _t3 = perf_counter()

    min_anchor_distance_by_candidate: dict[str, int] = {}
    if config.use_all_generated_data == "ALL":
        backward_relevant_candidates, min_anchor_distance_by_candidate = _collect_lazy_backward_relevant_candidates(
            config,
            ontology,
            tools,
            candidate_records=(
                record
                for record in candidate_records
                if str(record["candidate_id"]) in relevant_candidates
            ),
            reverse_edges=reverse_edges,
            direct_goal_candidates=direct_goal_candidates,
        )
        relevant_candidates &= backward_relevant_candidates

    relevant_records = [
        record
        for record in candidate_records
        if str(record["candidate_id"]) in relevant_candidates
    ]
    output_compression_cache: dict[
        tuple[tuple[tuple[str, tuple[str, ...]], ...], bool],
        dict[str, tuple[str, ...]],
    ] = {}
    relevant_input_ports = tuple(
        input_port
        for record in relevant_records
        for input_port in tuple(record["input_ports"])
    )
    signature_profiles_by_id, profile_values_by_id, profile_accepts_by_id, lazy_schema_stats = _assign_lazy_signature_profiles(
        ontology,
        roots,
        relevant_input_ports,
    )
    for record in relevant_records:
        compressed_output_ports: list[dict[str, object]] = []
        for output_port in tuple(record["output_ports"]):
            preserve_goal_profiles = str(record["candidate_id"]) in direct_goal_candidates
            compression_cache_key = (
                _lazy_dim_values_cache_key(output_port["port_values_by_dimension"]),
                preserve_goal_profiles,
            )
            compressed_vals = output_compression_cache.get(compression_cache_key)
            if compressed_vals is None:
                compressed_vals = _compress_lazy_output_choice_values(
                    ontology,
                    output_port["port_values_by_dimension"],
                    output_port["port_values_fset"],
                    relevant_input_ports,
                    goal_port_values_tuple,
                    goal_fsets,
                    preserve_goal_profiles=preserve_goal_profiles,
                )
                output_compression_cache[compression_cache_key] = compressed_vals
            compressed_fset = {dim: frozenset(vals) for dim, vals in compressed_vals.items()}
            compressed_output_ports.append(
                {
                    **output_port,
                    "port_values_by_dimension": compressed_vals,
                    "port_values_fset": compressed_fset,
                }
            )
        record["output_ports"] = _compress_duplicate_output_ports(tuple(compressed_output_ports))

    _t4 = perf_counter()
    relevant_tools = tuple(record["tool"] for record in relevant_records)
    relevant_records_by_candidate = {
        str(record["candidate_id"]): record
        for record in relevant_records
    }
    query_goal_candidates: set[str] = set()
    for record in relevant_records:
        candidate_id = str(record["candidate_id"])
        output_ports = tuple(record["output_ports"])
        goals_satisfied = all(
            any(
                _lazy_output_matches_lazy_input_fset(
                    output_port["port_values_fset"],
                    gf,
                )
                for output_port in output_ports
            )
            for gf in goal_fsets
        )
        if not goals_satisfied:
            continue
        total_output_multiplicity = sum(
            int(output_port.get("multiplicity", 1))
            for output_port in output_ports
        )
        if config.use_all_generated_data == "ALL" and any(
            not any(
                _lazy_output_matches_lazy_input_fset(
                    output_port["port_values_fset"],
                    gf,
                )
                for gf in goal_fsets
            )
            for output_port in output_ports
        ):
            continue
        if config.use_all_generated_data == "ALL" and total_output_multiplicity != len(goal_fsets):
            continue
        query_goal_candidates.add(candidate_id)
    query_goal_tools = {
        str(record["tool"].mode_id)
        for record in relevant_records
        if str(record["candidate_id"]) in query_goal_candidates
    }
    min_goal_distance_by_candidate: dict[str, int] = {}
    frontier: deque[str] = deque(sorted(query_goal_candidates))
    for candidate_id in frontier:
        min_goal_distance_by_candidate[candidate_id] = 0
    while frontier:
        consumer_candidate = frontier.popleft()
        next_distance = min_goal_distance_by_candidate[consumer_candidate] + 1
        for producer_candidate in sorted(reverse_edges.get(consumer_candidate, set())):
            if producer_candidate in min_goal_distance_by_candidate:
                continue
            if producer_candidate not in relevant_candidates:
                continue
            min_goal_distance_by_candidate[producer_candidate] = next_distance
            frontier.append(producer_candidate)
    min_step_by_candidate = _compute_lazy_candidate_min_steps(
        relevant_records,
        workflow_bindable_ports,
        produced_bindable_ports,
    )
    tool_min_steps: dict[str, int] = {}
    for record in relevant_records:
        candidate_id = str(record["candidate_id"])
        tool_id = str(record["tool"].mode_id)
        min_step = min_step_by_candidate.get(candidate_id)
        if min_step is None:
            continue
        existing_step = tool_min_steps.get(tool_id)
        if existing_step is None or min_step < existing_step:
            tool_min_steps[tool_id] = min_step
    must_use_min_steps, at_step_lower_bounds = _collect_lazy_selector_lower_bounds(
        config,
        ontology,
        tools,
        tool_min_steps=tool_min_steps,
    )
    exact_prefix_lower_bound = _collect_lazy_exact_prefix_lower_bound(
        config,
        ontology,
        tools,
        candidate_records=relevant_records,
        workflow_bindable_ports=workflow_bindable_ports,
        produced_bindable_ports=produced_bindable_ports,
        query_goal_candidates=query_goal_candidates,
    )
    allowed_candidates_by_step: dict[int, set[str]] = defaultdict(set)
    allowed_tools_by_step: dict[int, set[str]] = defaultdict(set)
    for record in relevant_records:
        candidate_id = str(record["candidate_id"])
        tool_id = str(record["tool"].mode_id)
        min_step = min_step_by_candidate.get(candidate_id)
        if min_step is None:
            continue
        max_step = config.solution_length_max
        if config.use_all_generated_data == "ALL":
            anchor_distance = min_anchor_distance_by_candidate.get(candidate_id)
            if anchor_distance is not None:
                max_step = min(
                    max_step,
                    config.solution_length_max - anchor_distance,
                )
        if max_step < min_step:
            continue
        for step_index in range(min_step, max_step + 1):
            allowed_candidates_by_step[step_index].add(candidate_id)
            allowed_tools_by_step[step_index].add(tool_id)
    writer = _FactWriter()
    _build_common_facts(writer, config, ontology, relevant_tools)
    _emit_lazy_constraints(writer, config, ontology, tools)

    for candidate_id in sorted(query_goal_candidates):
        writer.emit_fact("lazy_query_goal_candidate", _quote(candidate_id))
    for tool_id in sorted(query_goal_tools):
        writer.emit_fact("lazy_query_goal_tool", _quote(tool_id))
    for candidate_id, goal_distance in sorted(min_goal_distance_by_candidate.items()):
        writer.emit_fact("lazy_candidate_goal_distance", _quote(candidate_id), str(goal_distance))
    for step_index, candidate_ids in sorted(allowed_candidates_by_step.items()):
        for candidate_id in sorted(candidate_ids):
            writer.emit_fact(
                "lazy_candidate_allowed_at_step",
                _quote(candidate_id),
                str(step_index),
            )
    for step_index, tool_ids in sorted(allowed_tools_by_step.items()):
        for tool_id in sorted(tool_ids):
            writer.emit_fact(
                "lazy_step_allowed_tool",
                _quote(tool_id),
                str(step_index),
            )
    _t5 = perf_counter()

    for record in relevant_records:
        tool = record["tool"]
        candidate_id = str(record["candidate_id"])
        writer.emit_fact("lazy_tool_candidate", _quote(candidate_id), _quote(tool.mode_id))

        for input_port in tuple(record["input_ports"]):
            port_idx = int(input_port["port_idx"])
            writer.emit_fact(
                "lazy_candidate_input_port",
                _quote(candidate_id),
                str(port_idx),
            )
            writer.emit_fact(
                "lazy_candidate_input_signature_id",
                _quote(candidate_id),
                str(port_idx),
                str(input_port["signature_id"]),
            )
            for wf_index in input_port["workflow_input_matches"]:
                writer.emit_fact(
                    "lazy_initial_bindable",
                    _quote(candidate_id),
                    str(port_idx),
                    _quote(f"wf_input_{wf_index}"),
                )

        for output_port in tuple(record["output_ports"]):
            port_idx = int(output_port["port_idx"])
            writer.emit_fact(
                "lazy_candidate_output_port",
                _quote(candidate_id),
                str(port_idx),
            )
            writer.emit_fact(
                "lazy_candidate_output_multiplicity",
                _quote(candidate_id),
                str(port_idx),
                str(int(output_port.get("multiplicity", 1))),
            )
            for dim, declared_values in output_port["declared_dims"].items():
                for declared_value in declared_values:
                    writer.emit_fact(
                        "lazy_candidate_output_declared_type",
                        _quote(candidate_id),
                        str(port_idx),
                        _quote(declared_value),
                        _quote(dim),
                    )
            for dim, values in sorted(output_port["port_values_by_dimension"].items()):
                if len(values) == 1:
                    writer.emit_fact(
                        "lazy_candidate_output_singleton",
                        _quote(candidate_id),
                        str(port_idx),
                        _quote(values[0]),
                        _quote(dim),
                    )
                else:
                    for value in values:
                        writer.emit_fact(
                            "lazy_candidate_output_choice_value",
                            _quote(candidate_id),
                            str(port_idx),
                            _quote(value),
                            _quote(dim),
                        )
        writer.emit_fact(
            "lazy_candidate_total_output_multiplicity",
            _quote(candidate_id),
            str(sum(int(output_port.get("multiplicity", 1)) for output_port in tuple(record["output_ports"]))),
        )

    for producer_candidate, producer_port, consumer_candidate, consumer_port in sorted(bindable_pairs):
        if producer_candidate not in relevant_candidates or consumer_candidate not in relevant_candidates:
            continue
        writer.emit_fact(
            "lazy_candidate_port_bindable",
            _quote(producer_candidate),
            str(producer_port),
            _quote(consumer_candidate),
            str(consumer_port),
        )
    for consumer_candidate, ports_by_source in sorted(produced_bindable_ports.items()):
        if consumer_candidate not in relevant_candidates:
            continue
        for consumer_port, producer_candidates in sorted(ports_by_source.items()):
            for producer_candidate in sorted(producer_candidates):
                if producer_candidate not in relevant_candidates:
                    continue
                writer.emit_fact(
                    "lazy_candidate_bindable_producer",
                    _quote(consumer_candidate),
                    str(consumer_port),
                    _quote(producer_candidate),
                )
    for signature_id, category_profiles in sorted(signature_profiles_by_id.items()):
        for dim, (profile_id, _values) in sorted(category_profiles.items()):
            writer.emit_fact(
                "lazy_signature_profile",
                str(signature_id),
                _quote(dim),
                str(profile_id),
            )
    for profile_id, values in sorted(profile_values_by_id.items()):
        for value in values:
            writer.emit_fact(
                "lazy_profile_value",
                str(profile_id),
                _quote(value),
            )
    for profile_id, values in sorted(profile_accepts_by_id.items()):
        for value in values:
            writer.emit_fact(
                "lazy_profile_accepts",
                str(profile_id),
                _quote(value),
            )
    _t6 = perf_counter()
    print(
        f"  lazy builder phases: "
        f"goal_check={_t1-_t0:.2f}s "
        f"bindable_pairs={_t2-_t1:.2f}s "
        f"bfs_pruning={_t3-_t2:.2f}s "
        f"compression={_t4-_t3:.2f}s "
        f"step_indexing={_t5-_t4:.2f}s "
        f"fact_emission={_t6-_t5:.2f}s"
    )

    earliest_goal_step = min(
        (
            min_step_by_candidate[candidate_id]
            for candidate_id in query_goal_candidates
            if candidate_id in min_step_by_candidate
        ),
        default=config.solution_length_max + 1,
    )
    earliest_solution_step = max(
        1,
        earliest_goal_step,
        exact_prefix_lower_bound,
        *must_use_min_steps,
        *at_step_lower_bounds,
    )

    return _finalize_fact_bundle(
        writer,
        config=config,
        tools=relevant_tools,
        tool_stats=tool_stats,
        cache_stats={**ontology.cache_stats(), **resolver.stats(), **lazy_schema_stats},
        earliest_solution_step=earliest_solution_step,
    )


def build_fact_bundle(
    config: SnakeConfig,
    ontology: Ontology,
    tools: tuple[ToolMode, ...],
    strategy: str,
) -> FactBundle:
    """Build a solver-ready fact bundle."""

    writer = _FactWriter()
    roots = _build_common_facts(writer, config, ontology, tools)
    resolver = _ExpansionResolver(ontology, roots, strategy)
    tool_stats: list[ToolExpansionStat] = []

    # Per-mode_id offsets to avoid ID collisions when multiple ToolMode objects
    # share the same mode_id (e.g. relax_structure with Bulk vs Slab variants).
    _variant_offset: dict[str, int] = defaultdict(int)
    _output_offset: dict[str, int] = defaultdict(int)

    for tool in tools:
        input_port_variants = tuple(
            tuple(
                resolver.iter_dimension_maps(
                    _normalize_dim_map(port.dimensions),
                    expand_outputs=False,
                )
            )
            for port in tool.inputs
        )
        input_variant_count = _product(len(variants) for variants in input_port_variants) if input_port_variants else 1
        output_variant_count = len(tool.outputs)
        tool_stats.append(
            ToolExpansionStat(
                tool_id=tool.mode_id,
                tool_label=tool.label,
                input_ports=len(tool.inputs),
                output_ports=len(tool.outputs),
                input_variant_count=input_variant_count,
                output_variant_count=output_variant_count,
            )
        )
        if tool.inputs:
            v_base = _variant_offset[tool.mode_id]
            for variant_index, port_specs in enumerate(product(*input_port_variants)):
                variant_id = f"{tool.mode_id}_v{v_base + variant_index}"
                writer.emit_fact("tool_input", _quote(tool.mode_id), _quote(variant_id))
                for port_index, dims in enumerate(port_specs):
                    port_id = f"{variant_id}_p{port_index}"
                    writer.emit_fact("input_port", _quote(variant_id), _quote(port_id))
                    for dim, value in sorted(dims.items()):
                        writer.emit_fact(
                            "dimension",
                            _quote(port_id),
                            f"({_quote(value)}, {_quote(dim)})",
                        )
            _variant_offset[tool.mode_id] += input_variant_count

        o_base = _output_offset[tool.mode_id]
        for output_index, port in enumerate(tool.outputs):
            output_id = f"{tool.mode_id}_out_{o_base + output_index}"
            port_id = f"{output_id}_port_0"
            writer.emit_fact("tool_output", _quote(tool.mode_id), _quote(output_id))
            writer.emit_fact("output_port", _quote(output_id), _quote(port_id))
            for dim, values in sorted(_normalize_dim_map(port.dimensions).items()):
                for value in values:
                    writer.emit_fact(
                        "dimension",
                        _quote(port_id),
                        f"({_quote(value)}, {_quote(dim)})",
                    )
        _output_offset[tool.mode_id] += output_variant_count

    _emit_lazy_constraints(writer, config, ontology, tools)
    return _finalize_fact_bundle(
        writer,
        config=config,
        tools=tools,
        tool_stats=tool_stats,
        cache_stats={**ontology.cache_stats(), **resolver.stats()},
    )


def build_fact_bundle_ape_multi_shot(
    config: SnakeConfig,
    ontology: Ontology,
    tools: tuple[ToolMode, ...],
) -> FactBundle:
    """Build facts matching APE's Java ClingoSynthesisEngine contract.

    This intentionally avoids any legacy variant expansion:
    - one `tool_input(..., "<tool>_v0")` per tool with inputs
    - one `tool_output(..., "<tool>_out_<i>")` per declared output port
    - raw declared dimension values only
    """

    writer = _FactWriter()
    _build_common_facts(writer, config, ontology, tools)
    tool_stats: list[ToolExpansionStat] = []

    for tool in tools:
        tool_stats.append(
            ToolExpansionStat(
                tool_id=tool.mode_id,
                tool_label=tool.label,
                input_ports=len(tool.inputs),
                output_ports=len(tool.outputs),
                input_variant_count=1 if tool.inputs else 0,
                output_variant_count=len(tool.outputs),
            )
        )

        if tool.inputs:
            variant_id = f"{tool.mode_id}_v0"
            writer.emit_fact("tool_input", _quote(tool.mode_id), _quote(variant_id))
            for port_index, port in enumerate(tool.inputs):
                port_id = f"{variant_id}_p{port_index}"
                writer.emit_fact("input_port", _quote(variant_id), _quote(port_id))
                for dim, values in sorted(_normalize_dim_map(port.dimensions).items()):
                    for value in values:
                        writer.emit_fact(
                            "dimension",
                            _quote(port_id),
                            f"({_quote(value)}, {_quote(dim)})",
                        )

        for output_index, port in enumerate(tool.outputs):
            output_id = f"{tool.mode_id}_out_{output_index}"
            port_id = f"{output_id}_port_0"
            writer.emit_fact("tool_output", _quote(tool.mode_id), _quote(output_id))
            writer.emit_fact("output_port", _quote(output_id), _quote(port_id))
            for dim, values in sorted(_normalize_dim_map(port.dimensions).items()):
                for value in values:
                    writer.emit_fact(
                        "dimension",
                        _quote(port_id),
                        f"({_quote(value)}, {_quote(dim)})",
                    )

    _emit_lazy_constraints(writer, config, ontology, tools)
    return _finalize_fact_bundle(
        writer,
        config=config,
        tools=tools,
        tool_stats=tool_stats,
        cache_stats=dict(ontology.cache_stats()),
    )
