import importlib.util
import inspect
import logging
import os
import uuid
from typing import Any, Callable, Dict, List, Optional

from .hp import HP
from .run_history import HistoryDatabase, InMemoryHistory

# Correct logging configuration
# logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class Hypster:
    def __init__(self, func: Callable[[HP], Any], name: Optional[str] = None):
        """
        Initialize a Hypster instance.

        Args:
            func: The configuration function that takes an HP instance and returns a value.
            name: Optional name for the Hypster instance.
        """
        self.func = func
        self.name = name or getattr(func, "__name__", "unnamed")
        self.run_history: HistoryDatabase = InMemoryHistory()

    def __call__(
        self,
        values: Dict[str, Any] = None,
        explore_mode: bool = False,
    ) -> Any:
        """
        Execute the configuration function.

        Args:
            values: Override values for named parameters.
            explore_mode: Whether to run in explore mode.

        Returns:
            Whatever the configuration function returns (no filtering).
        """
        if values is None:
            values = {}

        # Create HP instance with minimal signature
        hp = HP(
            values=values,
            run_history=self.run_history,
            run_id=uuid.uuid4(),
            explore_mode=explore_mode,
        )

        # Execute the original function directly
        try:
            result = self.func(hp)
        except TypeError as e:
            if "return" in str(e).lower():
                raise ValueError(
                    "Configuration function must include an explicit 'return' statement. "
                    "Add 'return hp.collect(locals())' or 'return {object}' at the end of your config function."
                ) from e
            raise

        # Check if function returned None (missing return)
        if result is None:
            raise ValueError(
                "Configuration function must include an explicit 'return' statement. "
                "Add 'return hp.collect(locals())' or 'return {object}' at the end of your config function."
            )

        # Validate that all values correspond to named parameters
        hp._validate_values()

        # Return result as-is (no filtering)
        return result

    def save(self, path: Optional[str] = None):
        """
        Save the current object to a file.

        Parameters:
        path (Optional[str]): The file path where the object should be saved.
                              If None, the object will be saved with its name as the filename.

        Returns:
        None
        """
        if path is None:
            path = f"{self.name}.py"
        save(self, path)

    def get_last_snapshot(self) -> Dict[str, Any]:
        return self.run_history.get_latest_run_records(flattened=True)

    def get_snapshots(self) -> List[Dict[str, Any]]:
        return self.run_history.get_run_records(flattened=True)


def save(hypster_instance: Hypster, path: Optional[str] = None):
    """
    Save the configuration function to a file as an importable Python module.

    Args:
        hypster_instance: The Hypster instance to save.
        path: The file path where the configuration should be saved.

    Raises:
        ValueError: If the provided object is not a Hypster instance.

    Returns:
        None
    """
    if not isinstance(hypster_instance, Hypster):
        raise ValueError("The provided object is not a Hypster instance")

    if path is None:
        path = f"{hypster_instance.name}.py"

    try:
        # Get the original function source
        func_source = inspect.getsource(hypster_instance.func)

        # Check if the function already has @config decorator
        has_config_decorator = "@config" in func_source

        # Build the module content
        module_content = "from hypster import config, HP\n\n"

        if has_config_decorator:
            # Function already has decorator, just add it as-is
            module_content += func_source
        else:
            # Need to add the decorator
            # Extract the function definition
            lines = func_source.strip().split("\n")
            # Find the def line
            def_line_idx = 0
            for i, line in enumerate(lines):
                if line.strip().startswith("def "):
                    def_line_idx = i
                    break

            # Add decorator before the def line
            indented_lines = []
            for i, line in enumerate(lines):
                if i == def_line_idx:
                    # Add decorator with same indentation as def
                    indent = len(line) - len(line.lstrip())
                    indented_lines.append(" " * indent + "@config")
                indented_lines.append(line)

            module_content += "\n".join(indented_lines)

        # Only add bound instance if function doesn't already have @config decorator
        if not has_config_decorator:
            func_name = hypster_instance.func.__name__
            module_content += f"\n\n# Bound Hypster instance\n{func_name}_config = {func_name}\n"

        # Ensure directory exists
        directory = os.path.dirname(path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

        # Write the module
        with open(path, "w") as f:
            f.write(module_content)

        logger.info("Configuration saved to %s", path)

    except OSError as e:
        raise ValueError(f"Could not get source code for function {hypster_instance.func.__name__}: {e}")


def load(path: str, config_name: Optional[str] = None) -> Hypster:
    """
    Load a Python module and retrieve a Hypster configuration object.

    Args:
        path: The file path to the Python module to be loaded.
        config_name: Optional name of the config object to retrieve. If not provided,
                    will try to find a Hypster object in the module.

    Returns:
        Hypster: The loaded Hypster instance.

    Raises:
        ValueError: If no Hypster object is found or if multiple are found without specifying config_name.
    """
    # Import the module using importlib
    spec = importlib.util.spec_from_file_location("loaded_config", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Could not load module from {path}")

    module = importlib.util.module_from_spec(spec)

    try:
        spec.loader.exec_module(module)
    except Exception as e:
        raise ValueError(f"Error executing module {path}: {e}")

    # Find Hypster objects in the module
    hypster_objects = {}
    for name in dir(module):
        if not name.startswith("_"):
            obj = getattr(module, name)
            if isinstance(obj, Hypster):
                hypster_objects[name] = obj

    if not hypster_objects:
        raise ValueError(f"No Hypster objects found in module {path}")

    if config_name:
        if config_name not in hypster_objects:
            available = ", ".join(hypster_objects.keys())
            raise ValueError(f"Config '{config_name}' not found in module. Available: {available}")
        return hypster_objects[config_name]

    if len(hypster_objects) == 1:
        return list(hypster_objects.values())[0]

    # Multiple Hypster objects found
    available = ", ".join(hypster_objects.keys())
    raise ValueError(
        f"Multiple Hypster objects found in module: {available}. "
        f"Please specify config_name parameter to choose which one to load."
    )
