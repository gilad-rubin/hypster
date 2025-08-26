"""
Minimal registry for alias-based retrieval of Hypster configs.
Optimized for notebooks and ad-hoc workflows.
"""

from typing import TYPE_CHECKING, Dict

if TYPE_CHECKING:
    from .core import Hypster

# Global registry storage
_registry: Dict[str, "Hypster"] = {}


def register(obj: "Hypster", name: str) -> str:
    """
    Register a Hypster object under an alias.

    Args:
        obj: The Hypster object to register
        name: The alias name (e.g., "retriever/tfidf")

    Returns:
        The registered name
    """
    _registry[name] = obj
    return name


def get(name: str) -> "Hypster":
    """
    Retrieve a previously registered Hypster object.

    Args:
        name: The alias name

    Returns:
        The registered Hypster object

    Raises:
        KeyError: If the name is not found
    """
    if name not in _registry:
        raise KeyError(f"No config registered under alias '{name}'. Available: {list(_registry.keys())}")

    return _registry[name]


def list() -> Dict[str, str]:
    """
    List all registered configs.

    Returns:
        Dict mapping alias names to string representations
    """
    return {name: f"<Hypster {getattr(obj, 'name', 'unnamed')}>" for name, obj in _registry.items()}


def clear():
    """Clear all registered configs. Useful for testing."""
    global _registry
    _registry = {}
