"""Shared execution-policy helpers used by both HP and the run APIs.

This module sits below hp.py and core.py so neither has to import the other.
"""

import warnings
from typing import Any, Dict, Iterable

from .utils import suggest_similar_names

UNKNOWN_POLICIES = {"warn", "raise", "ignore"}


def validate_on_unknown(on_unknown: str) -> None:
    """Validate unknown-parameter policy at the API boundary."""
    if on_unknown not in UNKNOWN_POLICIES:
        raise ValueError("on_unknown must be one of 'raise', 'warn', or 'ignore'.")


def reject_removed_execution_argument_containers(execution_kwargs: Dict[str, Any]) -> None:
    if "args" in execution_kwargs or "kwargs" in execution_kwargs:
        raise TypeError(
            "This Hypster execution API no longer accepts args= or kwargs=. "
            "Pass execution arguments as direct keyword arguments."
        )


def reject_reserved_execution_arguments(
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


def handle_unknown_parameters(provided_values: Dict[str, Any], called_params: set[str], on_unknown: str) -> None:
    """Handle unknown or unreachable parameters based on on_unknown setting."""
    if on_unknown == "ignore":
        return

    unknown_params = set(provided_values.keys()) - called_params
    if not unknown_params:
        return

    known = list(called_params)
    error_lines = ["Unknown or unreachable parameters:"]
    for param in sorted(unknown_params):
        similar = suggest_similar_names(param, known, threshold=0.6)
        if similar:
            best_name, similarity = similar[0]
            error_lines.append(f"  - '{param}': Did you mean '{best_name}'? (similarity: {similarity:.0%})")
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
