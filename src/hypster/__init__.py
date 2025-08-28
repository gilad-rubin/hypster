"""Hypster v2 - Explicit, typed Python configuration management."""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

from .core import ConfigFunc, instantiate
from .hp import HP

try:
    # Resolve version from installed package metadata; __name__ == "hypster" at package top-level
    __version__ = _pkg_version(__name__)
except PackageNotFoundError:  # During editable/source runs before install
    __version__ = "0.0.0"


__all__ = ["HP", "instantiate", "ConfigFunc", "__version__"]
