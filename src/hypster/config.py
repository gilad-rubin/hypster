from typing import Callable, Optional, Union

from .core import Hypster
from .hp import HP
from .registry import registry

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


def config(
    func: Optional[HPFunc] = None,
    *,
    register: Optional[Union[str, bool]] = None,
    name: Optional[str] = None,
    namespace: Optional[str] = None,
    override: bool = False,
) -> Union[Hypster, Callable[[HPFunc], Hypster]]:
    """
    Decorator to create a Hypster instance from a configuration function.

    Can be used as @config or @config(...) with registration options.

    Args:
        func: The configuration function to decorate (when used as @config)
        register: Registration key (str) or True for auto-registration, None for no registration
        name: Custom name for registration (used with register=True)
        namespace: Namespace for registration (used with register=True)
        override: Allow overriding existing registrations

    Returns:
        Hypster instance or decorator function
    """

    def decorator(f: HPFunc) -> Hypster:
        hypster_instance = create_hypster_instance(f)

        # Handle registration
        if register is not None:
            if isinstance(register, str):
                # Direct string registration key
                registry.register(register, hypster_instance, override=override)
            elif register is True:
                # Auto-registration with optional name/namespace
                reg_name = name or f.__name__
                if namespace:
                    reg_key = f"{namespace}.{reg_name}"
                else:
                    reg_key = reg_name
                registry.register(reg_key, hypster_instance, override=override)

        return hypster_instance

    # If called with function directly (@config)
    if func is not None:
        return decorator(func)

    # If called with parameters (@config(...))
    return decorator
