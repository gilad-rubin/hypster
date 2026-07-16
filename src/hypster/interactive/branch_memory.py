from __future__ import annotations

import json
from typing import Any, Dict, Iterable, Mapping

from hypster.explore import ParameterInfo


class BranchChoiceMemory:
    """Session-local memory for values that may become reachable again."""

    def __init__(self) -> None:
        self._history: Dict[str, list[Any]] = {}

    def remember(self, parameter: ParameterInfo, value: Any, context: Mapping[str, Any]) -> None:
        history = self._history.setdefault(_memory_key(parameter, context), [])
        history[:] = [item for item in history if item != value]
        history.append(value)

    def remember_many(self, parameters: list[ParameterInfo], values: Dict[str, Any]) -> None:
        for index, parameter in enumerate(parameters):
            if parameter.path not in values:
                continue
            self.remember(parameter, values[parameter.path], _context_for(parameters, values, index))

    def latest_compatible(self, parameter: ParameterInfo, context: Mapping[str, Any]) -> tuple[bool, Any]:
        for value in reversed(self._history.get(_memory_key(parameter, context), [])):
            if is_compatible_value(parameter, value):
                return True, value
        return False, None


def _memory_key(parameter: ParameterInfo, context: Mapping[str, Any]) -> str:
    signature = {
        "path": parameter.path,
        "name": parameter.name,
        "kind": parameter.kind,
        "default": parameter.default_value,
        "options": parameter.options,
        "minimum": parameter.minimum,
        "maximum": parameter.maximum,
    }
    return _stable_json({"parameter": signature, "context": dict(context)})


def _context_for(parameters: list[ParameterInfo], values: Mapping[str, Any], index: int) -> Dict[str, Any]:
    context: Dict[str, Any] = {}
    for previous in parameters[:index]:
        if previous.path in values:
            context[previous.path] = values[previous.path]
    return context


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=repr, separators=(",", ":"))


def is_compatible_value(parameter: ParameterInfo, value: Any) -> bool:
    if parameter.options is not None:
        return _matches_options(parameter.kind, value, parameter.options)
    if parameter.kind == "bool":
        return isinstance(value, bool)
    if parameter.kind == "int":
        return isinstance(value, int) and _matches_bounds(value, parameter)
    if parameter.kind == "float":
        return isinstance(value, float) and _matches_bounds(value, parameter)
    if parameter.kind == "text":
        return isinstance(value, str)
    if parameter.kind == "multi_int":
        return isinstance(value, list) and all(
            isinstance(item, int) and _matches_bounds(item, parameter) for item in value
        )
    if parameter.kind == "multi_float":
        return isinstance(value, list) and all(
            isinstance(item, float) and _matches_bounds(item, parameter) for item in value
        )
    if parameter.kind == "multi_text":
        return isinstance(value, list) and all(isinstance(item, str) for item in value)
    if parameter.kind == "multi_bool":
        return isinstance(value, list) and all(isinstance(item, bool) for item in value)
    return True


def _matches_options(kind: str, value: Any, options: Iterable[Any]) -> bool:
    option_set = set(options)
    if kind == "multi_select":
        if not isinstance(value, list):
            return False
        try:
            return set(value).issubset(option_set)
        except TypeError:
            return False
    return value in option_set


def _matches_bounds(value: int | float, parameter: ParameterInfo) -> bool:
    if parameter.minimum is not None and value < parameter.minimum:
        return False
    if parameter.maximum is not None and value > parameter.maximum:
        return False
    return True
