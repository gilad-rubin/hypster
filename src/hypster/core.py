"""The Core API for instantiating configurations."""

import warnings
from typing import Any, Dict, Literal, Optional, Protocol, Tuple, TypeVar

from .hp import HP
from .hp_calls import HPCallError
from .utils import suggest_similar_names, validate_config_func_signature

T = TypeVar("T", covariant=True)


class ConfigFunc(Protocol[T]):
    def __call__(self, hp: HP, *args: Any, **kwargs: Any) -> T: ...


def instantiate(
    func: ConfigFunc[T],
    *,
    values: Optional[Dict[str, Any]] = None,
    args: Tuple[Any, ...] = (),
    kwargs: Optional[Dict[str, Any]] = None,
    on_unknown: Literal["warn", "raise", "ignore"] = "warn",
) -> T:
    """
    Execute a config function with the given values.

    Args:
        func: Config function with first param hp: HP
        values: Parameter values by name
        args: Additional positional arguments for func
        kwargs: Additional keyword arguments for func
        on_unknown: How to handle unknown/unreachable parameters:
            - 'warn': Issue warning and continue (default)
            - 'raise': Raise ValueError
            - 'ignore': Silently ignore unknown parameters

    Returns:
        Whatever the config function returns

    Raises:
        ValueError: If func doesn't have hp: HP as first parameter
        ValueError: If unknown parameters in values and on_unknown='raise'
    """
    # Validate function signature
    validate_config_func_signature(func)

    # Prepare values
    values = values or {}
    kwargs = kwargs or {}

    # Create HP instance
    hp = HP(values)

    # Track called parameters during execution
    original_called_params = hp.called_params.copy()

    try:
        # Execute the function
        result = func(hp, *args, **kwargs)

        # Check for unknown/unreachable parameters
        called_params = hp.called_params - original_called_params
        _handle_unknown_parameters(values, called_params, on_unknown)

        return result

    except HPCallError as e:
        # Re-raise HP errors as ValueError for cleaner API
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

    error_message = "\n".join(error_lines)

    if on_unknown == "raise":
        raise ValueError(error_message)
    elif on_unknown == "warn":
        warnings.warn(error_message, UserWarning, stacklevel=3)
