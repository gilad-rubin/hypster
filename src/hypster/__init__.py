"""Hypster v2 - Explicit, typed Python configuration management."""

from .core import ConfigFunc, instantiate
from .hp import HP

__all__ = ["HP", "instantiate", "ConfigFunc"]
