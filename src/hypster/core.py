"""The Core API for instantiating configurations."""

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, Literal, Optional, Protocol, Tuple, TypeVar

from .hp import HP
from .hp_calls import HPCallError
from .utils import suggest_similar_names, unflatten_dict, validate_config_func_signature

T = TypeVar("T", covariant=True)


class ConfigFunc(Protocol[T]):
    """Protocol for configuration functions that accept HP as first parameter."""

    def __call__(self, hp: HP, *args: Any, **kwargs: Any) -> T:
        """Execute the configuration function.

        Args:
            hp: The HP parameter interface for defining hyperparameters
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments

        Returns:
            The configuration result of type T
        """
        ...


@dataclass
class SelectedParams:
    """Container for selected hyperparameter values.

    Provides access to the parameters that were selected during instantiation,
    with both flat (dot-separated keys) and nested dictionary formats.

    Attributes:
        _values: Internal flat dictionary with dot-separated keys and sanitized values
    """

    _values: Dict[str, Any] = field(default_factory=dict)

    def get_flat(self) -> Dict[str, Any]:
        """Return parameters as flat dictionary with dot-separated keys.

        Example:
            >>> params.get_flat()
            {'batch_size': 32, 'optimizer.lr': 0.001, 'optimizer.momentum': 0.9}
        """
        return self._values.copy()

    def get_nested(self) -> Dict[str, Any]:
        """Return parameters as nested dictionary structure.

        Example:
            >>> params.get_nested()
            {'batch_size': 32, 'optimizer': {'lr': 0.001, 'momentum': 0.9}}

        Raises:
            ValueError: If there are prefix collisions (e.g., both 'model' and 'model.param' exist)
        """
        # Check for prefix collisions before unflattening
        keys = list(self._values.keys())
        for i, key1 in enumerate(keys):
            for key2 in keys[i + 1 :]:
                # Check if one key is a prefix of another
                if key1.startswith(key2 + ".") or key2.startswith(key1 + "."):
                    raise ValueError(
                        f"Cannot create nested structure: parameter name collision detected. "
                        f"Both '{key1}' and '{key2}' exist, where one is a prefix of the other. "
                        f"This creates ambiguity in the nested structure."
                    )
        return unflatten_dict(self._values)

    def __repr__(self) -> str:
        """Return string representation of SelectedParams."""
        return f"SelectedParams({self._values})"


@dataclass
class InstantiateResult:
    """Result of a configuration instantiation.

    Provides access to both the function's return value and the selected parameters.
    Supports dictionary-like access to values for convenience.

    Attributes:
        values: The return value from the config function (can be any type)
        params: Container for the selected hyperparameter values

    Example:
        >>> result = instantiate(my_config)
        >>> result.values  # The config function's return value
        {'model': <Model>, 'optimizer': <Optimizer>}
        >>> result['model']  # Dict-like access to values
        <Model>
        >>> result.params.get_flat()  # Selected parameter values
        {'learning_rate': 0.001, 'batch_size': 32}
    """

    values: Any
    params: SelectedParams = field(default_factory=SelectedParams)

    def __getitem__(self, key: str) -> Any:
        """Dict-like access to values.

        Raises:
            TypeError: If values is not subscriptable
            KeyError: If key is not found in values
        """
        return self.values[key]

    def __contains__(self, key: str) -> bool:
        """Check if key exists in values."""
        try:
            return key in self.values
        except TypeError:
            return False

    def get(self, key: str, default: Any = None) -> Any:
        """Get value with default.

        Args:
            key: The key to look up in values
            default: Value to return if key is not found

        Returns:
            The value for key if found, otherwise default
        """
        try:
            return self.values.get(key, default)
        except AttributeError:
            # values doesn't have .get() method
            try:
                return self.values[key]
            except (KeyError, TypeError, IndexError):
                return default

    def keys(self) -> Any:
        """Return keys of values if it's a dict-like object."""
        return self.values.keys()

    def items(self) -> Any:
        """Return items of values if it's a dict-like object."""
        return self.values.items()

    def __iter__(self) -> Iterator:
        """Iterate over values keys if it's a dict-like object."""
        return iter(self.values)

    def __len__(self) -> int:
        """Return length of values."""
        return len(self.values)

    def __repr__(self) -> str:
        """Return string representation of InstantiateResult."""
        return f"InstantiateResult(values={self.values!r}, params={self.params!r})"


def instantiate(
    func: ConfigFunc[T],
    *,
    values: Optional[Dict[str, Any]] = None,
    args: Tuple[Any, ...] = (),
    kwargs: Optional[Dict[str, Any]] = None,
    on_unknown: Literal["warn", "raise", "ignore"] = "warn",
) -> InstantiateResult:
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
        InstantiateResult containing:
            - values: Whatever the config function returns
            - params: SelectedParams with the parameter values used

    Raises:
        ValueError: If func doesn't have hp: HP as first parameter
        ValueError: If unknown parameters in values and on_unknown='raise'

    Example:
        >>> @config
        ... def my_config(hp: HP):
        ...     lr = hp.float(0.001, name="learning_rate")
        ...     batch_size = hp.int(32, name="batch_size")
        ...     return {"lr": lr, "batch_size": batch_size}
        ...
        >>> result = instantiate(my_config, values={"learning_rate": 0.01})
        >>> result.values
        {'lr': 0.01, 'batch_size': 32}
        >>> result["lr"]  # Dict-like access
        0.01
        >>> result.params.get_flat()
        {'learning_rate': 0.01, 'batch_size': 32}
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

        # Get sanitized values and complex params
        sanitized_values, complex_params = hp.get_selected_values()

        # Issue consolidated warning for complex params
        if complex_params:
            _issue_complex_params_warning(complex_params)

        # Create SelectedParams and InstantiateResult
        selected_params = SelectedParams(sanitized_values)
        return InstantiateResult(values=result, params=selected_params)

    except HPCallError as e:
        # Re-raise HP errors as ValueError for cleaner API
        raise ValueError(str(e)) from e


def _handle_unknown_parameters(provided_values: Dict[str, Any], called_params: set[str], on_unknown: str) -> None:
    """Handle unknown or unreachable parameters based on on_unknown setting.

    Args:
        provided_values: Dictionary of parameter values provided by the user
        called_params: Set of parameter names that were actually called during execution
        on_unknown: Strategy for handling unknown parameters ('warn', 'raise', or 'ignore')

    Raises:
        ValueError: If on_unknown='raise' and unknown parameters are found
    """
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


def _issue_complex_params_warning(complex_params: list[tuple[str, str]]) -> None:
    """Issue a consolidated warning for all complex parameter values.

    Args:
        complex_params: List of tuples containing (param_name, reason) for each
                       parameter with a complex (non-primitive) value that was converted
    """
    if not complex_params:
        return

    lines = ["The following parameter values are not primitives and were converted:"]
    for param_name, reason in complex_params:
        lines.append(f"  - {param_name}: {reason}")

    warnings.warn("\n".join(lines), UserWarning, stacklevel=3)
