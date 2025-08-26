from typing import Any, Callable, Union, overload

from .core import Hypster
from .hp import HP

HPFunc = Callable[[HP], Any]


def create_hypster_instance(func: HPFunc) -> Hypster:
    """
    Create a Hypster instance from a configuration function.

    Args:
        func: The configuration function.

    Returns:
        Hypster: An instance of the Hypster class.
    """
    return Hypster(func)


@overload
def config(func: HPFunc) -> Hypster: ...


@overload
def config() -> Callable[[HPFunc], Hypster]: ...


def config(func: Union[HPFunc, None] = None) -> Union[Hypster, Callable[[HPFunc], Hypster]]:
    """
    Decorator to create a Hypster instance from a configuration function.

    This decorator can be used in two ways:
    1. As a simple decorator: @config
    2. With parentheses: @config()

    Args:
        func: Internal parameter to enable both decorator usage patterns.
              Users should not pass this argument directly.

    Returns:
        Hypster: An instance of the Hypster class.

    Note:
        The inject_names parameter has been removed as automatic naming is no longer supported.
        All HP calls that need to be overridable must specify explicit names.
    """

    def decorator(func: HPFunc) -> Hypster:
        return create_hypster_instance(func)

    if func is None:
        return decorator
    return decorator(func)
