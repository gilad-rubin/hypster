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
    def __init__(self, name: str, func: Callable, namespace: Dict[str, Any]):
        """
        Initialize a Hypster instance.

        Args:
            name (str): The name of the Hypster instance.
            func (Callable): The configuration function.
            namespace (Dict[str, Any]): The namespace for execution.
        """
        self.name = name
        self.func = func
        self.namespace = namespace
        self.run_history: HistoryDatabase = InMemoryHistory()

    def __call__(
        self,
        final_vars: List[str] = [],
        exclude_vars: List[str] = [],
        values: Dict[str, Any] = {},
        explore_mode: bool = False,
    ) -> Dict[str, Any]:
        hp = HP(
            final_vars,
            exclude_vars,
            values,
            run_history=self.run_history,
            run_id=uuid.uuid4(),
            explore_mode=explore_mode,
        )
        result = self._execute_function(hp)
        return result

    def _execute_function(self, hp: HP) -> Dict[str, Any]:
        """
        Execute the function directly with the given HP instance.

        Args:
            hp (HP): The HP instance for values management.

        Returns:
            Dict[str, Any]: The instantiated config.
        """
        # Execute the function directly - the HP object will collect the parameter values
        self.func(hp)

        # Get the collected values from the HP object
        collected_values = hp.get_collected_values()

        # Process and filter the results
        return self._process_collected_values(collected_values, hp.final_vars, hp.exclude_vars)

    def _process_collected_values(
        self, collected_values: Dict[str, Any], final_vars: List[str], exclude_vars: List[str]
    ) -> Dict[str, Any]:
        """
        Process and filter the collected values from HP execution.

        Args:
            collected_values (Dict[str, Any]): Values collected from HP parameter calls.
            final_vars (List[str]): List of variables to include in the final result.
            exclude_vars (List[str]): List of variables to exclude from the final result.

        Returns:
            Dict[str, Any]: The processed and filtered results.

        Raises:
            ValueError: If any variable in final_vars does not exist in the collected values.
        """
        nested_vars = self.find_nested_vars(final_vars, self.run_history)
        final_vars = [var for var in final_vars if var not in nested_vars]

        if not final_vars:
            final_result = collected_values.copy()
        else:
            non_existent_vars = set(final_vars) - set(collected_values.keys())
            if non_existent_vars:
                raise ValueError(
                    "The following variables specified in final_vars "
                    "do not exist in the configuration: "
                    f"{', '.join(non_existent_vars)}"
                )
            final_result = {k: collected_values[k] for k in final_vars}

        # Apply exclude_vars after final_vars filtering
        if exclude_vars:
            final_result = {k: v for k, v in final_result.items() if k not in exclude_vars}

        logger.debug("Collected values: %s", collected_values)
        logger.debug("Final result after filtering: %s", final_result)

        return final_result

    def find_nested_vars(self, vars: List[str], run_history: HistoryDatabase) -> List[str]:
        """Find variables that reference nested configurations.

        Args:
            vars: List of variable names to check
            run_history: Database containing parameter records

        Returns:
            List of variable names that reference nested configs
        """
        nested_vars = []
        for var in vars:
            # Get the top-level variable name before any dot notation
            prefix = var.split(".")[0] if "." in var else var

            # Check if this variable is a nested config
            for record in run_history.get_latest_run_records().values():
                if record.name == prefix and record.parameter_type == "nest":
                    nested_vars.append(var)
                    break

        return nested_vars

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
        # Access the concrete InMemoryHistory method directly to use the Optional[str] signature
        from .run_history import InMemoryHistory

        if isinstance(self.run_history, InMemoryHistory):
            return self.run_history.get_run_records(run_id=None, flattened=True)
        else:
            # Fallback for other implementations
            return []


def save(hypster_instance: Hypster, path: Optional[str] = None):
    """
    Save the configuration function of a Hypster instance to a file.

    Args:
        hypster_instance (Hypster): The Hypster instance whose configuration
            function is to be saved.
        path (Optional[str]): The file path where the configuration should be
            saved. If None, the configuration is saved to a file named after
            the Hypster instance's name.
    Raises:
        ValueError: If the provided object is not a Hypster instance.
    Returns:
        None
    """
    if not isinstance(hypster_instance, Hypster):
        raise ValueError("The provided object is not a Hypster instance")

    if path is None:
        path = f"{hypster_instance.name}.py"

    # Get the source code of the function
    func_source = inspect.getsource(hypster_instance.func)

    # Remove the @config decorator if present
    lines = func_source.split("\n")
    filtered_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("@config") or stripped.startswith("@"):
            continue  # Skip decorator lines
        filtered_lines.append(line)

    func_source = "\n".join(filtered_lines)

    # Add the import statement
    modified_source = "from hypster import HP\n\n" + func_source
    modified_source = modified_source.rstrip("\n") + "\n"

    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    with open(path, "w") as f:
        f.write(modified_source)

    logger.info("Configuration saved to %s", path)


def load(path: str) -> Hypster:
    """
    Loads a Python module from the specified file path and creates a Hypster instance.

    Args:
        path (str): The file path to the Python module to be loaded.

    Returns:
        Hypster: An instance of the Hypster class containing the configuration function.

    Raises:
        ValueError: If no configuration function is found in the module.
    """
    with open(path, "r") as f:
        module_source = f.read()

    # Create a namespace with HP already imported
    namespace = {"HP": HP}

    # Execute the entire module
    exec(module_source, namespace)

    # Find functions that take HP as a parameter
    config_funcs = []
    for name, obj in namespace.items():
        if callable(obj) and hasattr(obj, "__annotations__"):
            # Check if the function has an 'hp' parameter with HP type annotation
            if "hp" in obj.__annotations__ and obj.__annotations__["hp"] == HP:
                config_funcs.append((name, obj))
        elif callable(obj) and not name.startswith("_"):
            # Fallback: check the parameter name in the function signature
            try:
                sig = inspect.signature(obj)
                if "hp" in sig.parameters:
                    config_funcs.append((name, obj))
            except (ValueError, TypeError):
                continue

    if not config_funcs:
        raise ValueError("No configuration function found in the module")

    # Use the first function found
    func_name, func = config_funcs[0]

    return Hypster(func_name, func, namespace)
