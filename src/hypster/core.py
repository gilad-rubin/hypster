"""The Core API for instantiating configurations."""

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, Generic, Iterable, Literal, Optional, Protocol, TypeVar

from .hp import HP
from .hp_calls import HPCallError
from .utils import normalize_values, suggest_similar_names, validate_config_func_signature

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

    def record_parameter(
        self,
        *,
        path: str,
        name: str,
        kind: str,
        default_value: Any,
        selected_value: Any,
        options: Optional[list[Any]] = None,
        minimum: Optional[int | float] = None,
        maximum: Optional[int | float] = None,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.params[path] = selected_value


class _CompositeTracker:
    """Send HP recording events to the params collector and a caller tracker."""

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


def _reject_removed_execution_argument_containers(execution_kwargs: Dict[str, Any]) -> None:
    if "args" in execution_kwargs or "kwargs" in execution_kwargs:
        raise TypeError(
            "This Hypster execution API no longer accepts args= or kwargs=. "
            "Pass execution arguments as direct keyword arguments."
        )


def _reject_reserved_execution_arguments(
    api_name: str,
    execution_kwargs: Dict[str, Any],
    reserved_names: Iterable[str],
    guidance: str,
) -> None:
    reserved = sorted(name for name in reserved_names if name in execution_kwargs)
    if not reserved:
        return

    names = ", ".join(f"{name}=" for name in reserved)
    raise TypeError(f"{api_name} reserves {names} for Hypster execution controls. {guidance}")


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


def _validate_on_unknown(on_unknown: str) -> None:
    """Validate unknown-parameter policy at the API boundary."""
    if on_unknown not in {"warn", "raise", "ignore"}:
        raise ValueError("on_unknown must be one of 'raise', 'warn', or 'ignore'.")


def _run_config(
    func: ConfigFunc[T],
    *,
    values: Optional[Dict[str, Any]] = None,
    kwargs: Optional[Dict[str, Any]] = None,
    on_unknown: UnknownPolicy = "raise",
    parameter_tracker: Optional[Any] = None,
) -> T:
    """Execute a config function through the shared instantiation path."""
    validate_config_func_signature(func)
    _validate_on_unknown(on_unknown)

    normalized_values = normalize_values(values)
    kwargs = kwargs or {}
    _reject_removed_execution_argument_containers(kwargs)
    hp = HP(normalized_values, parameter_tracker=parameter_tracker)
    original_called_params = hp.called_params.copy()

    try:
        result = func(hp, **kwargs)
        called_params = hp.called_params - original_called_params
        leaf_params = called_params - hp.nested_scope_paths
        _handle_unknown_parameters(normalized_values, leaf_params, on_unknown)
        return result
    except HPCallError as e:
        raise ValueError(str(e)) from e


def _handle_unknown_parameters(provided_values: Dict[str, Any], called_params: set[str], on_unknown: str) -> None:
    """Handle unknown or unreachable parameters based on on_unknown setting."""
    if on_unknown == "ignore":
        return

    # Find parameters that were provided but never called
    unknown_params = set(provided_values.keys()) - called_params

    if not unknown_params:
        return

    # Generate suggestions for typos
    suggestions = {}
    for unknown in unknown_params:
        similar = suggest_similar_names(unknown, list(called_params), threshold=0.6)
        if similar:
            suggestions[unknown] = [name for name, _ in similar[:3]]  # Top 3 suggestions

    # Format error message
    error_lines = ["Unknown or unreachable parameters:"]
    for param in sorted(unknown_params):
        if param in suggestions and suggestions[param]:
            best_suggestion = suggestions[param][0]
            similarity = suggest_similar_names(param, [best_suggestion], threshold=0.0)[0][1]
            error_lines.append(f"  - '{param}': Did you mean '{best_suggestion}'? (similarity: {similarity:.0%})")
        else:
            error_lines.append(f"  - '{param}': Unknown parameter")

    error_lines.append("")
    error_lines.append("Run explore(config, values=...) to inspect the active branch.")
    error_lines.append(
        "Nested dict values are interpreted as parameter paths; use dict-backed select keys for objects."
    )
    error_message = "\n".join(error_lines)

    if on_unknown == "raise":
        raise ValueError(error_message)
    elif on_unknown == "warn":
        warnings.warn(error_message, UserWarning, stacklevel=3)


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
