from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .core import ConfigFunc
from .hp import HP
from .hp_calls import HPCallError
from .utils import validate_config_func_signature


def _format_value(value: Any) -> str:
    if isinstance(value, str):
        return json.dumps(value)
    if value is None:
        return "None"
    if isinstance(value, list):
        return "[" + ", ".join(_format_value(item) for item in value) + "]"
    return repr(value)


@dataclass
class ParameterInfo:
    name: str
    path: str
    kind: str
    default_value: Any = None
    selected_value: Any = None
    options: Optional[List[Any]] = None
    minimum: Optional[int | float] = None
    maximum: Optional[int | float] = None
    children: List["ParameterInfo"] = field(default_factory=list)

    def is_group(self) -> bool:
        return self.kind == "group"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "path": self.path,
            "kind": self.kind,
            "default_value": self.default_value,
            "selected_value": self.selected_value,
            "options": self.options,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "children": [child.to_dict() for child in self.children],
        }

    def defaults(self) -> Dict[str, Any]:
        if self.is_group():
            defaults: Dict[str, Any] = {}
            for child in self.children:
                defaults.update(child.defaults())
            return defaults
        return {self.path: self.default_value}

    def format_label(self) -> str:
        if self.is_group():
            return self.name

        details: List[str] = []
        if self.options is not None:
            details.append(f"options: {_format_value(self.options)}")
        if self.minimum is not None and self.maximum is not None:
            details.append(f"{self.minimum}-{self.maximum}")
        elif self.minimum is not None:
            details.append(f"min: {self.minimum}")
        elif self.maximum is not None:
            details.append(f"max: {self.maximum}")

        suffix = f"  ({'; '.join(details)})" if details else ""
        return f"{self.name}: {self.kind} = {_format_value(self.selected_value)}{suffix}"


@dataclass
class ConfigSchema:
    name: str
    parameters: List[ParameterInfo] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "parameters": [parameter.to_dict() for parameter in self.parameters],
        }

    def defaults(self) -> Dict[str, Any]:
        defaults: Dict[str, Any] = {}
        for parameter in self.parameters:
            defaults.update(parameter.defaults())
        return defaults

    def format_tree(self) -> str:
        lines = [self.name]
        lines.extend(self._format_parameters(self.parameters))
        return "\n".join(lines)

    def _format_parameters(self, parameters: List[ParameterInfo], prefix: str = "") -> List[str]:
        lines: List[str] = []
        for index, parameter in enumerate(parameters):
            is_last = index == len(parameters) - 1
            connector = "└──" if is_last else "├──"
            lines.append(f"{prefix}{connector} {parameter.format_label()}")
            if parameter.children:
                extension = "    " if is_last else "│   "
                lines.extend(self._format_parameters(parameter.children, prefix + extension))
        return lines

    def __str__(self) -> str:
        return self.format_tree()


class SchemaTracer(HP):
    def __init__(self, values: Dict[str, Any], exploration_tracker: Optional[Any] = None):
        tracker = exploration_tracker or self
        super().__init__(values, exploration_tracker=tracker)
        if exploration_tracker is None:
            self._schema = ConfigSchema(name="")
            self._nodes_by_path: Dict[str, ParameterInfo] = {}

    def record_nest(self, *, path: str, name: str) -> None:
        self._ensure_group(path, name)

    def record_parameter(
        self,
        *,
        path: str,
        name: str,
        kind: str,
        default_value: Any,
        selected_value: Any,
        options: Optional[List[Any]] = None,
        minimum: Optional[int | float] = None,
        maximum: Optional[int | float] = None,
    ) -> None:
        parent_path = path.rpartition(".")[0]
        container = self._schema.parameters if not parent_path else self._ensure_group_path(parent_path).children

        parameter = ParameterInfo(
            name=name,
            path=path,
            kind=kind,
            default_value=default_value,
            selected_value=selected_value,
            options=options,
            minimum=minimum,
            maximum=maximum,
        )
        existing = self._nodes_by_path.get(path)
        if existing is None:
            container.append(parameter)
            self._nodes_by_path[path] = parameter
            return

        existing.kind = kind
        existing.default_value = default_value
        existing.selected_value = selected_value
        existing.options = options
        existing.minimum = minimum
        existing.maximum = maximum

    def build_schema(self, root_name: str) -> ConfigSchema:
        self._schema.name = root_name
        return self._schema

    def _ensure_group_path(self, path: str) -> ParameterInfo:
        existing = self._nodes_by_path.get(path)
        if existing is not None:
            return existing

        parent_path, _, name = path.rpartition(".")
        return self._ensure_group(path, name, parent_path=parent_path or None)

    def _ensure_group(self, path: str, name: str, parent_path: Optional[str] = None) -> ParameterInfo:
        existing = self._nodes_by_path.get(path)
        if existing is not None:
            return existing

        if parent_path is None:
            parent_path = path.rpartition(".")[0]

        container = self._schema.parameters if not parent_path else self._ensure_group_path(parent_path).children
        group = ParameterInfo(name=name, path=path, kind="group")
        container.append(group)
        self._nodes_by_path[path] = group
        return group


def explore(
    func: ConfigFunc[Any],
    *,
    values: Optional[Dict[str, Any]] = None,
    args: Tuple[Any, ...] = (),
    kwargs: Optional[Dict[str, Any]] = None,
    return_info: bool = False,
) -> Optional[ConfigSchema]:
    validate_config_func_signature(func)

    tracer = SchemaTracer(values or {})
    kwargs = kwargs or {}

    try:
        func(tracer, *args, **kwargs)
    except HPCallError as e:
        raise ValueError(str(e)) from e

    schema = tracer.build_schema(getattr(func, "__name__", func.__class__.__name__))

    if return_info:
        return schema

    print(schema.format_tree())
    return None
