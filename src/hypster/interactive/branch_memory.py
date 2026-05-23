from __future__ import annotations

from typing import Any, Dict, Iterable

from hypster.explore import ParameterInfo


class BranchChoiceMemory:
    """Session-local memory for values that may become reachable again."""

    def __init__(self) -> None:
        self._history: Dict[str, list[Any]] = {}

    def remember(self, path: str, value: Any) -> None:
        history = self._history.setdefault(path, [])
        history[:] = [item for item in history if item != value]
        history.append(value)

    def remember_many(self, values: Dict[str, Any]) -> None:
        for path, value in values.items():
            self.remember(path, value)

    def latest_compatible(self, parameter: ParameterInfo) -> tuple[bool, Any]:
        for value in reversed(self._history.get(parameter.path, [])):
            if _is_compatible(parameter, value):
                return True, value
        return False, None


def _is_compatible(parameter: ParameterInfo, value: Any) -> bool:
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
    return True


def _matches_options(kind: str, value: Any, options: Iterable[Any]) -> bool:
    option_set = set(options)
    if kind == "multi_select":
        return isinstance(value, list) and set(value).issubset(option_set)
    return value in option_set


def _matches_bounds(value: int | float, parameter: ParameterInfo) -> bool:
    if parameter.minimum is not None and value < parameter.minimum:
        return False
    if parameter.maximum is not None and value > parameter.maximum:
        return False
    return True
