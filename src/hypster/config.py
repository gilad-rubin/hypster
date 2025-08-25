from typing import Callable

from .core import Hypster
from .hp import HP

HPFunc = Callable[[HP], None]


def create_hypster_instance(func: HPFunc) -> Hypster:
    """
    Create a Hypster instance from a configuration function.

    Args:
        func: The configuration function to convert.

    Returns:
        Hypster: An instance of the Hypster class.
    """
    # Use the function directly instead of extracting source code
    return Hypster(func.__name__, func, func.__globals__)


def config(func: HPFunc) -> Hypster:
    """
    Decorator to create a Hypster instance from a configuration function.

    Args:
        func: The configuration function to decorate.

    Returns:
        Hypster: An instance of the Hypster class.
    """
    return create_hypster_instance(func)
