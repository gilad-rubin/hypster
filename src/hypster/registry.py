"""
Hypster Configuration Registry

Provides a global registry system for configuration management with namespace support.
"""

import logging
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class Registry:
    """
    A global registry for Hypster configurations with namespace support.

    The registry organizes configurations in a hierarchical namespace structure
    using dot notation (e.g., "models.transformers.bert").
    """

    def __init__(self):
        """Initialize an empty registry."""
        self._configs: Dict[str, Any] = {}
        self._namespaces: Set[str] = set()

    def register(self, key: str, config: Any, *, override: bool = False) -> None:
        """
        Register a configuration with a namespaced key.

        Args:
            key: Namespaced key (e.g., "models.bert_base")
            config: The configuration object to register
            override: If True, allow overriding existing registrations

        Raises:
            ValueError: If key already exists and override=False
        """
        if key in self._configs and not override:
            raise ValueError(f"Configuration '{key}' already registered. Use override=True to replace.")

        self._configs[key] = config

        # Register all parent namespaces
        parts = key.split(".")
        for i in range(1, len(parts)):
            namespace = ".".join(parts[:i])
            self._namespaces.add(namespace)

        logger.debug(f"Registered config '{key}' in registry")

    def get(self, key: str) -> Any:
        """
        Retrieve a configuration by its namespaced key.

        Args:
            key: The namespaced key to retrieve

        Returns:
            The registered configuration object

        Raises:
            KeyError: If the key is not found in the registry
        """
        if key not in self._configs:
            raise KeyError(f"Configuration '{key}' not found in registry")

        return self._configs[key]

    def list(self, namespace: Optional[str] = None) -> List[str]:
        """
        List all configuration keys, optionally filtered by namespace.

        Args:
            namespace: If provided, only return keys within this namespace

        Returns:
            List of configuration keys
        """
        if namespace is None:
            return list(self._configs.keys())

        # Return keys that start with the namespace prefix
        prefix = f"{namespace}."
        return [key for key in self._configs.keys() if key.startswith(prefix)]

    def contains(self, key: str) -> bool:
        """
        Check if a configuration key exists in the registry.

        Args:
            key: The namespaced key to check

        Returns:
            True if the key exists, False otherwise
        """
        return key in self._configs

    def clear(self, namespace: Optional[str] = None) -> None:
        """
        Clear configurations from the registry.

        Args:
            namespace: If provided, only clear configurations within this namespace.
                      If None, clear the entire registry.
        """
        if namespace is None:
            self._configs.clear()
            self._namespaces.clear()
            logger.debug("Cleared entire registry")
        else:
            # Remove all keys that start with the namespace prefix
            prefix = f"{namespace}."
            keys_to_remove = [key for key in self._configs.keys() if key.startswith(prefix) or key == namespace]

            for key in keys_to_remove:
                del self._configs[key]

            # Clean up empty namespaces
            self._rebuild_namespaces()

            logger.debug(f"Cleared namespace '{namespace}' from registry")

    def _rebuild_namespaces(self) -> None:
        """Rebuild the namespace set based on current keys."""
        self._namespaces.clear()
        for key in self._configs.keys():
            parts = key.split(".")
            for i in range(1, len(parts)):
                namespace = ".".join(parts[:i])
                self._namespaces.add(namespace)

    def __contains__(self, key: str) -> bool:
        """Support 'in' operator for key checking."""
        return self.contains(key)

    def __len__(self) -> int:
        """Return the number of registered configurations."""
        return len(self._configs)

    def get_namespaces(self) -> List[str]:
        """
        Get all registered namespaces.

        Returns:
            List of all namespace prefixes
        """
        return sorted(self._namespaces)


# Global registry instance
registry = Registry()
