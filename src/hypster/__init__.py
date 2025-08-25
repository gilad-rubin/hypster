try:
    from importlib.metadata import version

    __version__ = version(__name__)
except Exception:
    __version__ = "dev"

from .config import config
from .core import Hypster, load, save
from .hp import HP
from .registry import registry

__all__ = ["config", "save", "load", "HP", "registry"]
