from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Generic, Mapping, Optional, Tuple, TypeVar

from hypster.core import ConfigFunc, UnknownPolicy, instantiate_with_params
from hypster.explore import ConfigSchema, ParameterInfo, explore
from hypster.utils import normalize_values

from .branch_memory import BranchChoiceMemory

T = TypeVar("T")


def _parameters(schema: ConfigSchema) -> list[ParameterInfo]:
    parameters: list[ParameterInfo] = []
    for parameter in schema.parameters:
        parameters.extend(_flatten_parameter(parameter))
    return parameters


def _flatten_parameter(parameter: ParameterInfo) -> list[ParameterInfo]:
    if not parameter.is_group():
        return [parameter]

    parameters: list[ParameterInfo] = []
    for child in parameter.children:
        parameters.extend(_flatten_parameter(child))
    return parameters


@dataclass
class InteractiveError:
    kind: str
    message: str

    def to_dict(self) -> Dict[str, str]:
        return {"kind": self.kind, "message": self.message}


@dataclass
class InteractiveSession(Generic[T]):
    func: ConfigFunc[T]
    values: Optional[Mapping[str, Any]] = None
    args: Tuple[Any, ...] = ()
    kwargs: Optional[Dict[str, Any]] = None
    on_unknown: UnknownPolicy = "raise"
    auto_apply: bool = True

    def __post_init__(self) -> None:
        self._kwargs = dict(self.kwargs or {})
        self._draft_values: Dict[str, Any] = {}
        self._applied_values: Dict[str, Any] = {}
        self._params: Dict[str, Any] = {}
        self._schema: ConfigSchema | None = None
        self._baseline_values: Dict[str, Any] = {}
        self._value: T
        self._draft_error: InteractiveError | None = None
        self._applied_error: InteractiveError | None = None
        self._memory = BranchChoiceMemory()
        self._initialize(normalize_values(dict(self.values or {})))

    @property
    def value(self) -> T:
        if self._applied_error is not None:
            raise RuntimeError(self._applied_error.message)
        return self._value

    @property
    def params(self) -> Dict[str, Any]:
        if self._applied_error is not None:
            raise RuntimeError(self._applied_error.message)
        return dict(self._params)

    @property
    def snapshot(self) -> Dict[str, Any]:
        return {
            "schema": self._schema.to_dict() if self._schema is not None else None,
            "draft_values": dict(self._draft_values),
            "applied_values": dict(self._applied_values),
            "selected_params": dict(self._params),
            "mode": {"auto_apply": self.auto_apply},
            "status": self._status(),
            "error": self._snapshot_error(),
        }

    def dispatch(self, action: Mapping[str, Any]) -> Dict[str, Any]:
        action_type = action.get("type")
        if action_type == "set_value":
            self._set_value(str(action["path"]), action.get("value"))
            return self.snapshot
        if action_type == "reset":
            self._reset()
            return self.snapshot
        if action_type == "apply":
            self._apply_current_draft()
            return self.snapshot
        raise ValueError(f"Unknown interactive action type: {action_type!r}")

    def _status(self) -> str:
        if self._applied_error is not None:
            return "error"
        if self._draft_error is not None:
            return "draft_error"
        if not self.auto_apply and self._draft_values != self._applied_values:
            return "pending"
        return "applied"

    def _snapshot_error(self) -> Dict[str, str] | None:
        error = self._applied_error or self._draft_error
        return error.to_dict() if error is not None else None

    def _initialize(self, values: Dict[str, Any]) -> None:
        schema, selected_values = self._build_values(values)
        self._baseline_values = dict(selected_values)
        self._memory.remember_many(selected_values)
        self._apply(schema, selected_values)

    def _set_value(self, path: str, value: Any) -> None:
        if self._schema is None:
            raise RuntimeError("Interactive session has not been initialized")

        current_parameters = _parameters(self._schema)
        current_paths = [parameter.path for parameter in current_parameters]
        if path not in current_paths:
            raise ValueError(f"Cannot set unreachable parameter: {path}")

        self._memory.remember_many(self._draft_values)
        self._memory.remember(path, value)

        prefix_values: Dict[str, Any] = {}
        for parameter in current_parameters:
            if parameter.path == path:
                prefix_values[path] = value
                break
            if parameter.path in self._draft_values:
                prefix_values[parameter.path] = self._draft_values[parameter.path]

        try:
            schema, selected_values = self._build_values(prefix_values)
        except Exception as exc:
            self._draft_values = dict(prefix_values)
            self._draft_error = InteractiveError(kind="exploration", message=str(exc))
            if self.auto_apply:
                self._applied_error = self._draft_error
            return

        self._draft_error = None
        if self.auto_apply:
            try:
                self._apply(schema, selected_values)
            except Exception as exc:
                self._schema = schema
                self._draft_values = dict(selected_values)
                self._applied_values = dict(selected_values)
                self._applied_error = InteractiveError(kind="instantiation", message=str(exc))
            return

        self._schema = schema
        self._draft_values = dict(selected_values)

    def _reset(self) -> None:
        self._memory = BranchChoiceMemory()
        schema, selected_values = self._build_values(dict(self._baseline_values))
        self._memory.remember_many(selected_values)
        self._draft_error = None
        self._applied_error = None
        self._apply(schema, selected_values)

    def _apply_current_draft(self) -> None:
        if self._schema is None:
            raise RuntimeError("Interactive session has not been initialized")
        if self._draft_error is not None:
            return
        try:
            self._apply(self._schema, self._draft_values)
        except Exception as exc:
            self._applied_error = InteractiveError(kind="instantiation", message=str(exc))

    def _build_values(self, seed_values: Dict[str, Any]) -> tuple[ConfigSchema, Dict[str, Any]]:
        values = dict(seed_values)
        while True:
            schema = self._explore(values)
            missing = next((parameter for parameter in _parameters(schema) if parameter.path not in values), None)
            if missing is None:
                return schema, values

            found, value = self._memory.latest_compatible(missing)
            values[missing.path] = value if found else missing.selected_value

    def _explore(self, values: Dict[str, Any]) -> ConfigSchema:
        schema = explore(
            self.func,
            values=values,
            args=self.args,
            kwargs=self._kwargs,
            on_unknown=self.on_unknown,
            return_info=True,
        )
        if schema is None:
            raise RuntimeError("explore(..., return_info=True) did not return a schema")
        return schema

    def _apply(self, schema: ConfigSchema, selected_values: Dict[str, Any]) -> None:
        output = instantiate_with_params(
            self.func,
            values=selected_values,
            args=self.args,
            kwargs=self._kwargs,
            on_unknown=self.on_unknown,
        )

        self._schema = schema
        self._draft_values = dict(selected_values)
        self._applied_values = dict(selected_values)
        self._value = output.value
        self._params = dict(output.params)
        self._draft_error = None
        self._applied_error = None


class InteractiveResult(Generic[T]):
    def __init__(self, session: InteractiveSession[T]):
        self._session = session
        self._views: list[Any] = []

    @property
    def value(self) -> T:
        return self._session.value

    @property
    def params(self) -> Dict[str, Any]:
        return self._session.params

    @property
    def snapshot(self) -> Dict[str, Any]:
        return self._session.snapshot

    def dispatch(self, action: Mapping[str, Any]) -> Dict[str, Any]:
        snapshot = self._session.dispatch(action)
        self._sync_views(snapshot)
        return snapshot

    def interact(self) -> Any:
        from .widget import HypsterInteractWidget

        widget = HypsterInteractWidget(self)
        self._views.append(widget)
        return widget

    def _sync_views(self, snapshot: Dict[str, Any]) -> None:
        for view in self._views:
            view.snapshot = snapshot

    def _ipython_display_(self) -> None:
        from IPython.display import display

        display(self.interact())


def interact(
    func: ConfigFunc[T],
    *,
    values: Optional[Mapping[str, Any]] = None,
    args: Tuple[Any, ...] = (),
    kwargs: Optional[Dict[str, Any]] = None,
    on_unknown: UnknownPolicy = "raise",
    auto_apply: bool = True,
) -> InteractiveResult[T]:
    session: InteractiveSession[T] = InteractiveSession(
        func=func,
        values=values,
        args=args,
        kwargs=kwargs,
        on_unknown=on_unknown,
        auto_apply=auto_apply,
    )
    return InteractiveResult(session)
