# core.py

import ast
import inspect
from typing import Any, Callable, Dict, List, Union

from .logging_utils import configure_logging

logger = configure_logging()


class HP:
    def __init__(self, selections: Dict[str, Any], overrides: Dict[str, Any]):
        self.selections = selections
        self.overrides = overrides
        self.config_dict = {}
        logger.info("Initialized HP with selections: %s and overrides: %s", self.selections, self.overrides)

    def select(self, options: Union[Dict[str, Any], List[Any]], name: str = None, default: Any = None):
        if name is None:
            raise ValueError("Name must be provided explicitly or automatically inferred.")

        if isinstance(options, dict):
            if not all(isinstance(k, str) for k in options.keys()):
                bad_keys = [key for key in options.keys() if not isinstance(key, str)]
                raise ValueError(f"Dictionary keys must be strings. got {bad_keys} instead.")
        elif isinstance(options, list):
            if not all(isinstance(v, (str, int, bool, float)) for v in options):
                raise ValueError("List values must be one of: str, int, bool, float.")
            options = {v: v for v in options}
        else:
            raise ValueError("Options must be a dictionary or a list.")

        if default is not None and default not in options:
            raise ValueError("Default value must be one of the options.")

        logger.debug("Select called with options: %s, name: %s, default: %s", options, name, default)

        result = None
        if name in self.overrides:
            override_value = self.overrides[name]
            logger.debug("Found override for %s: %s", name, override_value)
            if override_value in options:
                result = options[override_value]
            else:
                result = override_value
            logger.info("Applied override for %s: %s", name, result)
        elif name in self.selections:
            selected_value = self.selections[name]
            logger.debug("Found selection for %s: %s", name, selected_value)
            if selected_value in options:
                result = options[selected_value]
                logger.info("Applied selection for %s: %s", name, result)
            else:
                raise InvalidSelectionError(
                    f"Invalid selection '{selected_value}' for '{name}'. Not in options: {list(options.keys())}"
                )
        elif default is not None:
            result = options[default]
        else:
            raise ValueError(f"No selection or override found for {name} and no default provided.")

        self.config_dict[name] = result
        return result


class Hypster:
    def __init__(self, func: Callable, source_code: str = None):
        self.func = func
        self.source_code = source_code or inspect.getsource(func)

    def __call__(self, final_vars: List[str] = [], selections: Dict[str, Any] = {}, overrides: Dict[str, Any] = {}):
        logger.info(
            "Hypster called with final_vars: %s, selections: %s, overrides: %s", final_vars, selections, overrides
        )
        hp = HP(selections, overrides)
        self.func(hp)  # Execute the function without expecting a return value
        result = hp.config_dict  # Use the config_dict from HP instance

        if not final_vars:
            return result
        else:
            return {k: result.get(k, None) for k in final_vars}


def config(func: Callable) -> Hypster:
    return Hypster(func)


def save(hypster_instance: Hypster, path: str = None):
    if not isinstance(hypster_instance, Hypster):
        raise ValueError("The provided object is not a Hypster instance")

    if path is None:
        path = f"{hypster_instance.func.__name__}.py"

    # Parse the source code into an AST
    tree = ast.parse(hypster_instance.source_code)

    # Find the function definition and remove decorators
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            node.decorator_list = []
            break

    # Convert the modified AST back to source code
    modified_source = ast.unparse(tree)

    with open(path, "w") as f:
        f.write(modified_source)

    logger.info("Configuration saved to %s", path)


def load(path: str) -> Hypster:
    with open(path, "r") as f:
        source = f.read()

    # Execute the source code to define the function
    namespace = {}
    exec(source, namespace)

    # Find the function in the namespace
    for name, obj in namespace.items():
        if callable(obj) and not name.startswith("__"):
            # Create and return a Hypster instance with the source code
            return Hypster(obj, source_code=source)

    raise ValueError("No suitable function found in the source code")


class InvalidSelectionError(Exception):
    pass


# Example usage (can be commented out in the actual module)
"""
@config
def my_config(hp):
    hp.select(["a", "b", "c"], name="a", default="a")
    hp.select({"x": 1, "y": 2}, name="b", default="x")

# Save the configuration
save(my_config, "my_config.py")

# Load the configuration
loaded_config = load("my_config.py")

# Use the loaded configuration
result = loaded_config(final_vars=["a"], selections={"b": "y"}, overrides={"a": "c"})
print(result)
"""
