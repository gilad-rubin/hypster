from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Tuple

from .core import ConfigFunc, _handle_unknown_parameters, _validate_on_unknown
from .hp import HP
from .hp_calls import HPCallError
from .utils import normalize_values, validate_config_func_signature


def _format_value(value: Any) -> str:
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if value is None:
        return "None"
    if isinstance(value, list):
        return "[" + ", ".join(_format_value(item) for item in value) + "]"
    return repr(value)


def _to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Enum):
        return _to_jsonable(value.value)
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    return repr(value)


def _humanize_name(name: str) -> str:
    acronyms = {
        "api": "API",
        "hp": "HP",
        "hpo": "HPO",
        "id": "ID",
        "k": "K",
        "llm": "LLM",
        "mlflow": "MLflow",
        "openai": "OpenAI",
        "ui": "UI",
        "url": "URL",
    }
    return " ".join(acronyms.get(part, part.capitalize()) for part in name.split("_"))


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
    description: Optional[str] = None
    children: List["ParameterInfo"] = field(default_factory=list)

    def is_group(self) -> bool:
        return self.kind == "group"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "path": self.path,
            "kind": self.kind,
            "default_value": _to_jsonable(self.default_value),
            "selected_value": _to_jsonable(self.selected_value),
            "options": _to_jsonable(self.options),
            "minimum": _to_jsonable(self.minimum),
            "maximum": _to_jsonable(self.maximum),
            "description": self.description,
            "display_label": _humanize_name(self.name),
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
    def __init__(self, values: Dict[str, Any], parameter_tracker: Optional[Any] = None):
        tracker = parameter_tracker or self
        super().__init__(values, parameter_tracker=tracker)
        if parameter_tracker is None:
            self._schema = ConfigSchema(name="")
            self._nodes_by_path: Dict[str, ParameterInfo] = {}
        else:
            self._schema = tracker._schema
            self._nodes_by_path = tracker._nodes_by_path

    def record_nest(self, *, path: str, name: str, description: Optional[str] = None) -> None:
        group = self._ensure_group(path, name)
        group.description = description

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
        description: Optional[str] = None,
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
            description=description,
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
        existing.description = description

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
    on_unknown: Literal["warn", "raise", "ignore"] = "raise",
    return_info: bool = False,
) -> Optional[ConfigSchema]:
    validate_config_func_signature(func)
    _validate_on_unknown(on_unknown)

    normalized_values: Dict[str, Any] = normalize_values(values)
    tracer = SchemaTracer(normalized_values)
    kwargs = kwargs or {}
    original_called_params = tracer.called_params.copy()

    try:
        func(tracer, *args, **kwargs)
    except HPCallError as e:
        raise ValueError(str(e)) from e

    called_params = tracer.called_params - original_called_params
    _handle_unknown_parameters(normalized_values, called_params, on_unknown)

    schema = tracer.build_schema(getattr(func, "__name__", func.__class__.__name__))

    if return_info:
        return schema

    print(schema.format_tree())
    return None
