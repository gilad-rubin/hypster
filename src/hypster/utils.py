"""Utilities for error messages, similarity matching, and validation."""

import inspect
from difflib import SequenceMatcher
from typing import Any, Callable, Dict, List, Tuple


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


def format_error_with_suggestions(
    unknown_params: Dict[str, Any], suggestions: Dict[str, List[str]], reachability: Dict[str, str]
) -> str:
    """
    Format helpful error messages distinguishing between:
    - Typos (similar names exist)
    - Unreachable parameters (exist but not in current conditional path)
    - Truly unknown parameters
    """
    lines = ["Unknown or unreachable parameters:"]

    for param, value in unknown_params.items():
        if param in suggestions and suggestions[param]:
            # Typo suggestion
            similar = suggestions[param][0]  # Take the best suggestion
            lines.append(f"  - '{param}': Did you mean '{similar}'?")
        elif param in reachability:
            # Unreachable parameter
            lines.append(f"  - '{param}': This parameter exists but is only reachable when {reachability[param]}")
        else:
            # Truly unknown
            lines.append(f"  - '{param}': Unknown parameter")

    return "\n".join(lines)


def merge_nested_dicts(dotted: Dict[str, Any], nested: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """
    Merge dotted keys and nested dicts (nested takes precedence).
    Returns merged dict and list of conflict warnings.
    """
    result = dotted.copy()
    warnings = []

    # Convert dotted notation to nested structure
    expanded: Dict[str, Any] = {}
    for key, value in dotted.items():
        parts = key.split(".")
        current = expanded
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[parts[-1]] = value

    # Merge with nested dict (nested takes precedence)
    def merge_recursive(base: Dict[str, Any], override: Dict[str, Any], path: str = "") -> None:
        for key, value in override.items():
            full_key = f"{path}.{key}" if path else key
            if key in base:
                if isinstance(base[key], dict) and isinstance(value, dict):
                    merge_recursive(base[key], value, full_key)
                else:
                    warnings.append(
                        f"Parameter '{full_key}' specified in both dotted and nested format, using nested value"
                    )
                    base[key] = value
            else:
                base[key] = value

    merge_recursive(expanded, nested)
    return expanded, warnings


def validate_config_func_signature(func: Callable) -> None:
    """
    Validate that func has hp: HP as first parameter.
    Raises ValueError with helpful message if not.
    """
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


def flatten_dict(d: Dict[str, Any], parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
    """Flatten a nested dictionary into dotted notation."""
    items: List[Tuple[str, Any]] = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def unflatten_dict(d: Dict[str, Any], sep: str = ".") -> Dict[str, Any]:
    """Convert dotted notation dictionary to nested structure."""
    result: Dict[str, Any] = {}
    for key, value in d.items():
        parts = key.split(sep)
        current = result
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[parts[-1]] = value
    return result
