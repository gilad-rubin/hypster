"""Utilities for error messages, similarity matching, and validation."""

import inspect
import keyword
from collections.abc import Mapping
from difflib import SequenceMatcher
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Tuple


def suggest_similar_names(unknown: str, known: List[str], threshold: float = 0.6) -> List[Tuple[str, float]]:
    """Find similar parameter names with similarity scores."""
    suggestions = []
    for name in known:
        similarity = SequenceMatcher(None, unknown.lower(), name.lower()).ratio()
        if similarity >= threshold:
            suggestions.append((name, similarity))

    # Sort by similarity score (highest first)
    suggestions.sort(key=lambda x: x[1], reverse=True)
    return suggestions


def validate_config_func_signature(func: Callable) -> None:
    """
    Validate that func has hp: HP as first parameter.
    Raises ValueError with helpful message if not.
    """
    try:
        hash(func)
    except TypeError:
        _validate_config_func_signature_uncached(func)
        return
    _validate_config_func_signature_cached(func)


@lru_cache(maxsize=1024)
def _validate_config_func_signature_cached(func: Callable) -> None:
    """Cached signature validation for repeat executions of the same config."""
    _validate_config_func_signature_uncached(func)


def _validate_config_func_signature_uncached(func: Callable) -> None:
    """Validate a config function signature without assuming hashability."""
    try:
        sig = inspect.signature(func)
        params = list(sig.parameters.values())

        if not params:
            raise ValueError(
                f"Configuration function '{func.__name__}' must have 'hp: HP' as first parameter. "
                f"Got: {func.__name__}() - no parameters"
            )

        first_param = params[0]
        if first_param.name != "hp":
            raise ValueError(
                f"Configuration function '{func.__name__}' first param must be named 'hp'. Got: {first_param.name}"
            )

        # Check type annotation if present
        if first_param.annotation != inspect.Parameter.empty:
            # Get the type annotation string representation
            annotation_str = str(first_param.annotation)
            if "HP" not in annotation_str:
                raise ValueError(
                    f"Configuration function '{func.__name__}' first parameter must be typed as 'hp: HP'. "
                    f"Got: hp: {annotation_str}"
                )

    except Exception as e:
        if isinstance(e, ValueError):
            raise
        raise ValueError(f"Error validating configuration function '{func.__name__}': {str(e)}")


def validate_identifier_name(name: Any, *, kind: str = "name") -> str:
    """Validate a user-facing Hypster name segment."""
    if not isinstance(name, str):
        raise ValueError(
            f"Invalid {kind}: {name!r}\n\n"
            "  -> Hypster names must be strings\n\n"
            "How to fix: Use a Python identifier-style string such as 'learning_rate'."
        )
    return _validate_identifier_name_cached(name, kind)


@lru_cache(maxsize=2048)
def _validate_identifier_name_cached(name: str, kind: str) -> str:
    """Cached validation for string identifier segments."""
    if not name.isidentifier():
        raise ValueError(
            f"Invalid {kind}: {name!r}\n\n"
            "  -> Hypster names must be valid Python identifiers\n\n"
            "How to fix: Use letters, numbers, and underscores, and do not include dots, spaces, or hyphens."
        )
    if keyword.iskeyword(name):
        raise ValueError(
            f"Invalid {kind}: {name!r}\n\n"
            f"  -> {name!r} is a Python keyword and cannot be used as a Hypster name\n\n"
            f"How to fix: Use a different name, such as '{name}_value'."
        )
    return name


def validate_parameter_path(path: Any, *, kind: str = "values key") -> str:
    """Validate a dotted Hypster parameter path."""
    if not isinstance(path, str):
        raise ValueError(
            f"Invalid {kind}: {path!r}\n\n"
            "  -> Values keys must be strings\n\n"
            "How to fix: Use a dotted parameter path such as 'model.learning_rate'."
        )
    return _validate_parameter_path_cached(path, kind)


@lru_cache(maxsize=4096)
def _validate_parameter_path_cached(path: str, kind: str) -> str:
    """Cached validation for string dotted parameter paths."""
    for segment in path.split("."):
        validate_identifier_name(segment, kind=f"{kind} segment")
    return path


def validate_select_choice(value: Any, *, param_path: str, allow_none: bool = False) -> Any:
    """Validate a select key/custom value is safe to log and replay."""
    try:
        return _validate_select_choice_cached(value, param_path, allow_none)
    except TypeError:
        return _validate_select_choice_uncached(value, param_path=param_path, allow_none=allow_none)


@lru_cache(maxsize=4096, typed=True)
def _validate_select_choice_cached(value: Any, param_path: str, allow_none: bool) -> Any:
    """Cached validation for hashable select choices."""
    return _validate_select_choice_uncached(value, param_path=param_path, allow_none=allow_none)


def _validate_select_choice_uncached(value: Any, *, param_path: str, allow_none: bool = False) -> Any:
    """Validate a select choice without assuming hashability."""
    if value is None:
        if allow_none:
            return value
        raise ValueError(
            f"Parameter '{param_path}': None is only allowed when allow_none=True.\n\n"
            "How to fix: pass allow_none=True, or use a string key such as 'none' in a dict-backed select."
        )

    if isinstance(value, (bool, int, float, str)):
        return value

    raise ValueError(
        f"Parameter '{param_path}': Select choices must be logging-safe scalar values "
        f"(None, bool, int, float, or str), got {type(value).__name__} ({value!r}).\n\n"
        "How to fix: Use dict-backed select so the key is simple and the mapped value can be complex. "
        "Example: hp.select({'small': {'layers': 2}}, name='model')."
    )


def normalize_values(values: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Flatten nested values into dotted paths and reject duplicate paths."""
    if values is None:
        return {}
    if not isinstance(values, Mapping):
        raise ValueError(
            f"Invalid values: expected a dictionary of parameter paths to values, got {type(values).__name__}.\n\n"
            "How to fix: pass values={'learning_rate': 0.1} or values={'model': {'depth': 3}}."
        )
    if not values:
        return {}

    flat_values: Dict[str, Any] = {}
    for key, value in values.items():
        if isinstance(value, Mapping):
            break
        flat_values[validate_parameter_path(key)] = value
    else:
        return flat_values

    normalized: Dict[str, Any] = {}
    sources: Dict[str, str] = {}

    def add_validated(path: str, value: Any, source: str) -> None:
        if path in normalized:
            raise ValueError(
                f"Duplicate value for '{path}': provided via both {sources[path]} and {source}.\n\n"
                "How to fix: Use only one form for each parameter path."
            )
        normalized[path] = value
        sources[path] = source

    def nested_source(parts: tuple[str, ...]) -> str:
        return "nested dict " + " -> ".join(repr(part) for part in parts)

    def visit_nested(prefix: str, mapping: Mapping[str, Any], source_parts: tuple[str, ...]) -> None:
        for key, value in mapping.items():
            segment = validate_identifier_name(key, kind="nested values key")
            full_path = f"{prefix}.{segment}" if prefix else segment
            parts = (*source_parts, segment)
            if isinstance(value, Mapping):
                visit_nested(full_path, value, parts)
            else:
                add_validated(full_path, value, nested_source(parts))

    for key, value in values.items():
        key_path = validate_parameter_path(key)
        if isinstance(value, Mapping):
            visit_nested(key_path, value, (key,))
        else:
            source = f"dotted key {key!r}" if "." in key else f"key {key!r}"
            add_validated(key_path, value, source)
    return normalized
