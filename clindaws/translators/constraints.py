"""Constraint parsing, selector resolution, and fact emission.

This module bridges user constraint files and the internal facts consumed by
the ASP encodings. It parses APE-style template objects and native atom
strings, resolves whether selectors should match tools transitively or exactly,
and emits both runtime constraint facts and translation-time pruning helpers.
"""

from __future__ import annotations
import json
import re
from collections import defaultdict, deque
from collections.abc import Iterable, Mapping
from typing import Any

from clindaws.core.models import SnakeConfig, ToolMode
from clindaws.core.ontology import Ontology

from clindaws.translators.utils import _dedupe_stable, _quote
from clindaws.translators.fact_writer import _FactWriter



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
    "prev_m",
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
_DYNAMIC_SUPPORTED_CONSTRAINTS = (
    set(_SELECTOR_TEMPLATE_ALIASES)
    | {
        "use_t",
        "operationInput",
    }
)
_DYNAMIC_NATIVE_CONSTRAINTS = _DYNAMIC_SUPPORTED_CONSTRAINTS | {
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
    "prev_m": "transitive",
    "used_iff_used": "transitive",
    "max_uses": "transitive",
    "mutex_tools": "transitive",
    "connected_op": "transitive",
    "operationInput": "transitive",
    "operation_input": "transitive",
}
# The helpers below normalize prefixed ontology/tool selectors from the config
# into the compact ids used internally by the translator and encodings.
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
def _extract_constraint_data_selector_spec(
    parameter: Mapping[str, Any],
    *,
    prefix: str,
) -> ConstraintDataSelectorSpec:
    if not parameter:
        raise ValueError("empty parameter")

    dims: list[tuple[str, tuple[str, ...]]] = []
    for raw_category, raw_values in parameter.items():
        category = _strip_constraint_value(str(raw_category), prefix).strip()
        if not category:
            raise ValueError("parameter contained an empty category")
        if isinstance(raw_values, str):
            values_iter: Iterable[str] = (raw_values,)
        elif isinstance(raw_values, Iterable):
            values_iter = (str(raw_value) for raw_value in raw_values)
        else:
            raise ValueError("parameter category did not contain selector values")
        values = _dedupe_stable(
            normalized
            for raw_value in values_iter
            if (normalized := _strip_constraint_value(str(raw_value), prefix).strip())
        )
        if not values:
            raise ValueError(f"parameter category {category} did not contain selector values")
        dims.append((category, values))
    if not dims:
        raise ValueError("parameter did not contain any categories")
    return tuple(dims)
def _is_constraint_data_selector_spec(value: object) -> bool:
    if not isinstance(value, tuple) or not value:
        return False
    for item in value:
        if (
            not isinstance(item, tuple)
            or len(item) != 2
            or not isinstance(item[0], str)
            or not isinstance(item[1], tuple)
            or not all(isinstance(selector_value, str) for selector_value in item[1])
        ):
            return False
    return True
def _parse_template_constraint_args(
    config: SnakeConfig,
    constraint_id: str,
    raw_parameters: Iterable[Mapping[str, Any]],
) -> tuple[Any, ...]:
    base_constraint_name, _selector_policies = _resolve_constraint_template_name(constraint_id)
    parsed_args: list[Any] = []
    for index, parameter in enumerate(raw_parameters):
        if base_constraint_name == "use_t":
            parsed_args.append(
                _extract_constraint_data_selector_spec(parameter, prefix=config.ontology_prefix)
            )
            continue
        if base_constraint_name in {"operation_input", "operationInput"} and index == 1:
            parsed_args.append(
                _extract_constraint_data_selector_spec(parameter, prefix=config.ontology_prefix)
            )
            continue
        parsed_args.append(
            _extract_constraint_selector(parameter, prefix=config.ontology_prefix)
        )
    return tuple(parsed_args)
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
    """Resolve whether a selector should match by hierarchy or exact identity."""
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
def _dynamic_allowed_selectors(
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
def _dynamic_allowed_data_selectors(
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
def _infer_constraint_data_selector_category(
    config: SnakeConfig,
    ontology: Ontology,
    raw_selector: str,
) -> str | None:
    selector = _strip_constraint_value(raw_selector, prefix=config.ontology_prefix).strip()
    if not selector:
        return None
    matching_roots = tuple(
        root
        for root in config.data_dimensions_taxonomy_roots
        if selector == root or root in ontology.ancestors_of(selector)
    )
    if len(matching_roots) != 1:
        return None
    return matching_roots[0]
def _emit_dynamic_constraint(
    writer: _FactWriter,
    *,
    config: SnakeConfig,
    constraint_name: str,
    args: tuple[Any, ...],
    allowed_selectors: set[str],
    allowed_data_selectors: set[str],
    selector_ids: dict[tuple[str, str], str],
    data_selector_ids: dict[object, str],
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

    def _data_selector_id_for(raw_selector: str | ConstraintDataSelectorSpec) -> str:
        selector_key: object
        if isinstance(raw_selector, str):
            selector_key = ("flat", raw_selector)
        else:
            selector_key = ("composite", raw_selector)
        selector_id = data_selector_ids.get(selector_key)
        if selector_id is None:
            selector_id = f"constraint_data_selector_{len(data_selector_ids)}"
            data_selector_ids[selector_key] = selector_id
            if isinstance(raw_selector, str):
                for alias in _data_selector_aliases(raw_selector, prefix=config.ontology_prefix):
                    writer.emit_fact("constraint_data_selector", _quote(selector_id), _quote(alias))
            else:
                for category, values in raw_selector:
                    writer.emit_fact(
                        "constraint_data_selector_category",
                        _quote(selector_id),
                        _quote(category),
                    )
                    for value in values:
                        for alias in _data_selector_aliases(value, prefix=config.ontology_prefix):
                            writer.emit_fact(
                                "constraint_data_selector_dim",
                                _quote(selector_id),
                                _quote(category),
                                _quote(alias),
                            )
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

    def _data_selector_arg(position: int) -> str | ConstraintDataSelectorSpec | None:
        if position >= len(args):
            _skip(f"{constraint_name} is missing data selector argument {position + 1}")
            return None
        raw_value = args[position]
        if isinstance(raw_value, str):
            selector = raw_value.strip()
            if not selector:
                _skip(f"{constraint_name} has an empty data selector argument")
                return None
            return selector
        if not _is_constraint_data_selector_spec(raw_value):
            _skip(f"{constraint_name} expects data selector argument {position + 1}")
            return None
        return raw_value

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
    elif base_constraint_name == "prev_m":
        writer.emit_fact("constraint_prev", selector_a_id, selector_b_id)
    elif base_constraint_name == "used_iff_used":
        writer.emit_fact("constraint_used_iff_used", selector_a_id, selector_b_id)
    elif base_constraint_name == "mutex_tools":
        writer.emit_fact("constraint_mutex", selector_a_id, selector_b_id)
    elif base_constraint_name == "connected_op":
        writer.emit_fact("constraint_connected", selector_a_id, selector_b_id)
    else:
        _skip(f"unsupported constraint {constraint_name}")
def _emit_dynamic_template_constraints(
    writer: _FactWriter,
    *,
    config: SnakeConfig,
    constraints_path,
    constraints: list[Mapping[str, Any]],
    allowed_selectors: set[str],
    allowed_data_selectors: set[str],
    selector_ids: dict[tuple[str, str], str],
    data_selector_ids: dict[object, str],
    tool_ids: set[str],
    operation_ids: set[str],
) -> None:
    writer.emit_comment(
        f"dynamic constraint translation from {constraints_path.name}"
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
        if constraint_id not in _DYNAMIC_SUPPORTED_CONSTRAINTS:
            writer.emit_comment(
                f"skipping {constraints_path.name} constraint {index}: unsupported template {constraint_id}"
            )
            continue

        raw_parameters = raw_constraint.get("parameters") or []
        try:
            selectors = _parse_template_constraint_args(config, constraint_id, raw_parameters)
        except ValueError as exc:
            writer.emit_comment(
                f"skipping {constraints_path.name} constraint {index}: invalid parameters ({exc})"
            )
            continue

        _emit_dynamic_constraint(
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
def _emit_dynamic_native_constraints(
    writer: _FactWriter,
    *,
    config: SnakeConfig,
    ontology: Ontology,
    constraints_path,
    constraints: list[str],
    allowed_selectors: set[str],
    allowed_data_selectors: set[str],
    selector_ids: dict[tuple[str, str], str],
    data_selector_ids: dict[object, str],
    tool_ids: set[str],
    operation_ids: set[str],
) -> None:
    writer.emit_comment(
        f"dynamic native constraint translation from {constraints_path.name}"
    )

    parsed_constraints: list[tuple[int, str, tuple[str | int, ...]]] = []
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
        if constraint_name not in _DYNAMIC_NATIVE_CONSTRAINTS:
            writer.emit_comment(
                f"skipping {constraints_path.name} constraint {index}: unsupported native atom {constraint_name}"
            )
            continue
        parsed_constraints.append((index, constraint_name, args))

    grouped_use_t_indices: set[int] = set()
    grouped_use_t_selector: ConstraintDataSelectorSpec | None = None
    native_use_t_entries = [
        (index, args[0])
        for index, constraint_name, args in parsed_constraints
        if constraint_name == "use_t" and len(args) == 1 and isinstance(args[0], str)
    ]
    if len(native_use_t_entries) == len(config.data_dimensions_taxonomy_roots):
        grouped_values_by_category: dict[str, str] = {}
        for index, raw_value in native_use_t_entries:
            category = _infer_constraint_data_selector_category(config, ontology, raw_value)
            normalized_value = _strip_constraint_value(raw_value, prefix=config.ontology_prefix).strip()
            if category is None or not normalized_value or category in grouped_values_by_category:
                grouped_values_by_category = {}
                break
            grouped_values_by_category[category] = normalized_value
        if set(grouped_values_by_category) == set(config.data_dimensions_taxonomy_roots):
            grouped_use_t_selector = tuple(
                (category, (grouped_values_by_category[category],))
                for category in config.data_dimensions_taxonomy_roots
            )
            grouped_use_t_indices = {index for index, _ in native_use_t_entries}

    if grouped_use_t_selector is not None:
        _emit_dynamic_constraint(
            writer,
            config=config,
            constraint_name="use_t",
            args=(grouped_use_t_selector,),
            allowed_selectors=allowed_selectors,
            allowed_data_selectors=allowed_data_selectors,
            selector_ids=selector_ids,
            data_selector_ids=data_selector_ids,
            tool_ids=tool_ids,
            operation_ids=operation_ids,
            source_name=constraints_path.name,
            index=min(grouped_use_t_indices),
        )

    for index, constraint_name, args in parsed_constraints:
        if index in grouped_use_t_indices:
            continue
        _emit_dynamic_constraint(
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
def _load_dynamic_constraints(config: SnakeConfig) -> tuple[object, list[Any], str] | None:
    """Load the configured constraint file and classify its representation."""
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
def _collect_dynamic_forbidden_tool_ids(
    config: SnakeConfig,
    ontology: Ontology,
    tools: tuple[ToolMode, ...],
) -> set[str]:
    """Collect tools that can be removed before candidate expansion starts."""
    loaded_constraints = _load_dynamic_constraints(config)
    if loaded_constraints is None:
        return set()

    allowed_selectors = _dynamic_allowed_selectors(config, ontology, tools)
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
                selectors = _parse_template_constraint_args(config, constraint_id, raw_parameters)
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
def _collect_dynamic_selector_lower_bounds(
    config: SnakeConfig,
    ontology: Ontology,
    tools: tuple[ToolMode, ...],
    *,
    tool_min_steps: Mapping[str, int],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    loaded_constraints = _load_dynamic_constraints(config)
    if loaded_constraints is None:
        return (), ()

    allowed_selectors = _dynamic_allowed_selectors(config, ontology, tools)
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
        parsed_constraints: list[tuple[str, tuple[Any, ...]]] = []
        for raw_constraint in constraints:
            constraint_id = str(raw_constraint.get("constraintid", "")).strip()
            if not constraint_id:
                continue
            raw_parameters = raw_constraint.get("parameters") or []
            try:
                selectors = _parse_template_constraint_args(config, constraint_id, raw_parameters)
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
    data_selector: str | ConstraintDataSelectorSpec,
) -> bool:
    output_ports = tuple(candidate_record["output_ports"])
    if isinstance(data_selector, str):
        for output_port in output_ports:
            port_values_by_dimension = output_port["port_values_by_dimension"]
            assert isinstance(port_values_by_dimension, Mapping)
            for values in port_values_by_dimension.values():
                for actual_value in values:
                    if actual_value == data_selector or data_selector in ontology.ancestors_of(actual_value):
                        return True
        return False

    for output_port in output_ports:
        port_values_by_dimension = output_port["port_values_by_dimension"]
        assert isinstance(port_values_by_dimension, Mapping)
        if all(
            any(
                actual_value == wanted_value or wanted_value in ontology.ancestors_of(actual_value)
                for wanted_value in wanted_values
                for actual_value in tuple(port_values_by_dimension.get(category, ()))
            )
            for category, wanted_values in data_selector
        ):
            return True
    return False
def _collect_dynamic_backward_relevant_candidates(
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

    loaded_constraints = _load_dynamic_constraints(config)
    if loaded_constraints is None:
        return set(direct_goal_candidates), {
            candidate_id: 0 for candidate_id in direct_goal_candidates
        }

    allowed_selectors = _dynamic_allowed_selectors(config, ontology, tools)
    allowed_data_selectors = _dynamic_allowed_data_selectors(config, ontology, tools)
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

    def _matching_data_selector(
        raw_value: str | ConstraintDataSelectorSpec,
    ) -> str | ConstraintDataSelectorSpec | None:
        if isinstance(raw_value, str):
            selector = raw_value.strip()
            if not selector:
                return None
            aliases = _data_selector_aliases(selector, prefix=config.ontology_prefix)
            for alias in aliases:
                if alias in allowed_data_selectors:
                    return alias
            return None
        matched_dims: list[tuple[str, tuple[str, ...]]] = []
        for category, values in raw_value:
            matched_values = tuple(
                alias
                for value in values
                for alias in _data_selector_aliases(value, prefix=config.ontology_prefix)
                if alias in allowed_data_selectors
            )
            if not matched_values:
                return None
            matched_dims.append((category, _dedupe_stable(matched_values)))
        return tuple(matched_dims)

    _constraints_path, constraints, constraint_kind = loaded_constraints
    if constraint_kind == "template":
        parsed_constraints: list[tuple[str, tuple[str | int, ...]]] = []
        for raw_constraint in constraints:
            constraint_id = str(raw_constraint.get("constraintid", "")).strip()
            if not constraint_id:
                continue
            raw_parameters = raw_constraint.get("parameters") or []
            try:
                selectors = _parse_template_constraint_args(config, constraint_id, raw_parameters)
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
        "prev_m",
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
            if len(args) == 2 and (
                isinstance(args[1], str) or _is_constraint_data_selector_spec(args[1])
            ):
                data_selector = _matching_data_selector(args[1])
                if data_selector is not None:
                    for candidate_id, record in candidate_records_by_id.items():
                        if _candidate_outputs_match_data_selector(ontology, record, data_selector):
                            anchor_candidate_ids.add(candidate_id)
        elif base_constraint_name == "use_t":
            if len(args) == 1 and (
                isinstance(args[0], str) or _is_constraint_data_selector_spec(args[0])
            ):
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
def _collect_dynamic_exact_prefix_lower_bound(
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
    """Compute a small exact lower bound for early dynamic horizons.

    This is intentionally limited to the first two horizons. It performs an
    exact candidate-level feasibility check for 1-step and 2-step workflows
    using the already-computed workflow/producers bindability surface. If no
    such workflow can satisfy the must-use selectors plus the goal within the
    tested horizons, later solving can safely skip them entirely.
    """

    if config.solution_length_max <= 1 or max_exact_horizon < 1:
        return 1

    loaded_constraints = _load_dynamic_constraints(config)
    if loaded_constraints is None:
        return 1

    allowed_selectors = _dynamic_allowed_selectors(config, ontology, tools)
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
                selectors = _parse_template_constraint_args(config, constraint_id, raw_parameters)
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
def _emit_dynamic_constraints(
    writer: _FactWriter,
    config: SnakeConfig,
    ontology: Ontology,
    tools: tuple[ToolMode, ...],
) -> None:
    """Emit runtime constraint facts plus selector-match metadata."""
    allowed_selectors = _dynamic_allowed_selectors(config, ontology, tools)
    allowed_data_selectors = _dynamic_allowed_data_selectors(config, ontology, tools)
    selector_ids: dict[tuple[str, str], str] = {}
    data_selector_ids: dict[str, str] = {}
    tool_ids = {tool.mode_id for tool in tools}
    operation_ids = {tax_op for tool in tools for tax_op in tool.taxonomy_operations}
    loaded_constraints = _load_dynamic_constraints(config)
    if loaded_constraints is None:
        return

    constraints_path, constraints, constraint_kind = loaded_constraints
    if constraint_kind == "template":
        _emit_dynamic_template_constraints(
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
        _emit_dynamic_native_constraints(
            writer,
            config=config,
            ontology=ontology,
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
                    "dynamic_constraint_selector_matches_tool",
                    _quote(selector_id),
                    _quote(tool.mode_id),
                )
