"""The Core API for instantiating configurations."""

from dataclasses import dataclass, field
from typing import Any, Dict, Generic, Literal, Optional, Protocol, TypeVar

from ._execution import (
    handle_unknown_parameters,
    reject_removed_execution_argument_containers,
    validate_on_unknown,
)
from .hp import HP
from .utils import normalize_values, validate_config_func_signature

T = TypeVar("T", covariant=True)
UnknownPolicy = Literal["warn", "raise", "ignore"]


class ConfigFunc(Protocol[T]):
    def __call__(self, hp: HP, *args: Any, **kwargs: Any) -> T: ...


@dataclass(frozen=True)
class InstantiationOutput(Generic[T]):
    """Output of a config execution with its selected params sidecar."""

    value: T
    params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "params", dict(self.params))


class ParamsTracker:
    """Collect selected params from HP's parameter-recording hook."""

    def __init__(self) -> None:
        self.params: Dict[str, Any] = {}

    def record_parameter(self, *, path: str, selected_value: Any, **event: Any) -> None:
        self.params[path] = selected_value

    def record_nest(self, **event: Any) -> None:
        pass


class _CompositeTracker:
    """Send HP recording events to the params collector and a caller tracker.

    Caller-supplied trackers may implement only a subset of the tracker
    contract, so events are forwarded tolerantly.
    """

    def __init__(self, *trackers: Any) -> None:
        self.trackers = trackers

    def record_parameter(self, **event: Any) -> None:
        for tracker in self.trackers:
            record = getattr(tracker, "record_parameter", None)
            if callable(record):
                record(**event)

    def record_nest(self, **event: Any) -> None:
        for tracker in self.trackers:
            record = getattr(tracker, "record_nest", None)
            if callable(record):
                record(**event)


def instantiate(
    func: ConfigFunc[T],
    *,
    values: Optional[Dict[str, Any]] = None,
    on_unknown: UnknownPolicy = "raise",
    **kwargs: Any,
) -> T:
    """
    Execute a config function with the given values.

    Args:
        func: Config function with first param hp: HP
        values: Parameter values by name
        **kwargs: Execution arguments forwarded directly to func
        on_unknown: How to handle unknown/unreachable parameters:
            - 'raise': Raise ValueError (default)
            - 'warn': Issue warning and continue
            - 'ignore': Silently ignore unknown parameters

    Returns:
        Whatever the config function returns

    Raises:
        ValueError: If func doesn't have hp: HP as first parameter
        ValueError: If unknown parameters in values and on_unknown='raise'
    """
    return _run_config(func, values=values, kwargs=kwargs, on_unknown=on_unknown)


def _run_config(
    func: ConfigFunc[T],
    *,
    values: Optional[Dict[str, Any]] = None,
    kwargs: Optional[Dict[str, Any]] = None,
    on_unknown: UnknownPolicy = "raise",
    parameter_tracker: Optional[Any] = None,
    value_provider: Optional[Any] = None,
) -> T:
    """Execute a config function through the shared instantiation path."""
    validate_config_func_signature(func)
    validate_on_unknown(on_unknown)

    normalized_values = normalize_values(values)
    kwargs = kwargs or {}
    reject_removed_execution_argument_containers(kwargs)
    hp = HP(normalized_values, parameter_tracker=parameter_tracker, value_provider=value_provider)

    result = func(hp, **kwargs)
    leaf_params = hp.called_params - hp.nested_scope_paths
    handle_unknown_parameters(normalized_values, leaf_params, on_unknown)
    return result


def instantiate_with_params(
    func: ConfigFunc[T],
    *,
    values: Optional[Dict[str, Any]] = None,
    on_unknown: UnknownPolicy = "raise",
    tracker: Optional[Any] = None,
    **kwargs: Any,
) -> InstantiationOutput[T]:
    """
    Execute a config function and return its value with selected params.

    Args mirror instantiate(). ``tracker`` optionally observes the same rich
    parameter and nest events used internally while selected params continue to
    be collected unchanged.
    """
    params_tracker = ParamsTracker()
    parameter_tracker = params_tracker if tracker is None else _CompositeTracker(params_tracker, tracker)
    result = _run_config(
        func,
        values=values,
        kwargs=kwargs,
        on_unknown=on_unknown,
        parameter_tracker=parameter_tracker,
    )
    return InstantiationOutput(value=result, params=params_tracker.params)
