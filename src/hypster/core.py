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
    Save the configuration of a Hypster instance to a file.
    This function extracts the configuration function from the provided
    Hypster instance's source code and writes it to a specified file. If no
    path is provided, the configuration is saved to a file named after the
    Hypster instance's name with a .py extension.
    Args:
        hypster_instance (Hypster): The Hypster instance whose configuration
            is to be saved.
        path (Optional[str]): The file path where the configuration should be
            saved. If None, the configuration is saved to a file named after
            the Hypster instance's name.
    Raises:
        ValueError: If the provided object is not a Hypster instance or if no
            configuration function is found in the module.
    Returns:
        None
    """
    if not isinstance(hypster_instance, Hypster):
        raise ValueError("The provided object is not a Hypster instance")

    if path is None:
        path = f"{hypster_instance.name}.py"

    result = find_hp_function_body_and_name(hypster_instance.source_code)

    if result is None:
        raise ValueError("No configuration function found in the module")

    func_name, hp_func_source = result

    modified_source = "from hypster import HP\n\n\n" + hp_func_source
    modified_source = modified_source.rstrip("\n")
    modified_source = modified_source + "\n"

    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    with open(path, "w") as f:
        f.write(modified_source)

    logger.info("Configuration saved to %s", path)


def load(path: str, inject_names=True) -> Hypster:
    """
    Loads a Python module from the specified file path, executes it, and retrieves a configuration function.
    Args:
        path (str): The file path to the Python module to be loaded.
        inject_names (bool, optional): If True, injects names into the namespace. Defaults to True.

    Returns:
        Hypster: An instance of the Hypster class containing the configuration function and its context.

    Raises:
        ValueError: If no configuration function is found in the module or
                    if the function cannot be retrieved from the namespace.
    """
    with open(path, "r") as f:
        module_source = f.read()

    # Create a namespace with HP already imported
    namespace = {"HP": HP}

    # Execute the entire module
    exec(module_source, namespace)

    # Find the configuration function
    result = find_hp_function_body_and_name(module_source)

    if result is None:
        raise ValueError("No configuration function found in the module")

    # Unpack the function name and body from the result
    func_name, config_body = result

    # Retrieve the function object from the namespace
    func = namespace.get(func_name)

    if func is None:
        raise ValueError(f"Could not find the function {func_name} in the loaded module")

    return Hypster(func_name, config_body, namespace, inject_names)
