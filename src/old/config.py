import inspect
from typing import Callable, Union, overload

from .core import Hypster
from .hp import HP
from .utils import find_hp_function_body_and_name

HPFunc = Callable[[HP], None]


def create_hypster_instance(func: HPFunc, inject_names: bool) -> Hypster:
    """
    Create a Hypster instance from a configuration function.

    Args:
        func (HPFunc): The configuration function.
        inject_names (bool): Whether to inject names into the source code.

    Returns:
        Hypster: An instance of the Hypster class.

    Raises:
        ValueError: If no configuration function is found in the module.
    """
    source_code = inspect.getsource(func)
    result = find_hp_function_body_and_name(source_code)
    if result is None:
        raise ValueError("No configuration function found in the module")
    config_name, config_body = result
    namespace = {"HP": HP}
    return Hypster(config_name, config_body, namespace, inject_names=inject_names)


@overload
def config(func: HPFunc) -> Hypster: ...


@overload
def config(*, inject_names: bool = True) -> Callable[[HPFunc], Hypster]: ...


def config(
    func: Union[HPFunc, None] = None, *, inject_names: bool = True
) -> Union[Hypster, Callable[[HPFunc], Hypster]]:
    """
    Decorator to create a Hypster instance from a configuration function.

    This decorator can be used in two ways:
    1. As a simple decorator: @config
    2. With arguments: @config(inject_names=False)

    Args:
        func: Internal parameter to enable both decorator usage patterns.
              Users should not pass this argument directly.
        inject_names (bool, optional): Whether to automatically infer and inject
            parameter names into the source code. Defaults to True.

    Returns:
        Hypster: An instance of the Hypster class.

    Note:
        Although the return type is annotated as Union[Hypster, Callable[[HPFunc], Hypster]]
        for technical reasons, this function always returns a Hypster instance when used as
        a decorator.
    """

    def decorator(func: HPFunc) -> Hypster:
        return create_hypster_instance(func, inject_names)

    if func is None:
        return decorator
    return decorator(func)
