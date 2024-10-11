import logging
import os
import textwrap
import types
from typing import Any, Dict, List, Optional, Tuple, Union

from .ast_analyzer import (
    collect_hp_calls,
    find_independent_select_calls,
    find_referenced_vars,
    inject_names_to_source_code,
)
from .hp import HP
from .utils import find_hp_function_body_and_name, remove_function_signature

# Correct logging configuration
# logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

HypsterReturn = Union[Dict[str, Any], Tuple[Dict[str, Any], Dict[str, Any]]]


class Hypster:
    def __init__(self, name, source_code: str, namespace: Dict[str, Any], inject_names: bool = True):
        """
        Initialize a Hypster instance.

        Args:
            name (str): The name of the Hypster instance.
            source_code (str): The source code to be executed.
            namespace (Dict[str, Any]): The namespace for execution.
            inject_names (bool, optional): Whether to inject names into the source code. Defaults to True.
        """
        self.name = name
        self.source_code = source_code
        self.namespace = namespace
        self.hp_calls = collect_hp_calls(self.source_code)

        self.modified_source = (
            inject_names_to_source_code(self.source_code, self.hp_calls) if inject_names else self.source_code
        )

        self.referenced_vars = find_referenced_vars(self.source_code)
        self.independent_select_calls = find_independent_select_calls(self.referenced_vars, self.hp_calls)
        self.combinations = []
        self.defaults = {}
        self.snapshot_history = []

    def __call__(
        self,
        final_vars: Optional[List[str]] = None,
        selections: Optional[Dict[str, Any]] = None,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> HypsterReturn:
        """
        Execute the Hypster instance with given parameters.

        Args:
            final_vars (Optional[List[str]], optional): List of variables to include in the final result.
            selections (Optional[Dict[str, Any]], optional): Predefined selections for hyperparameters.
            overrides (Optional[Dict[str, Any]], optional): Overrides for hyperparameters.

        Returns:
            Dict[str, Any]: The execution result.
        """
        hp = HP(final_vars or [], selections or {}, overrides or {})
        result = self._execute_function(hp, self.modified_source)
        self.snapshot_history.append(hp.snapshot)
        return result

    def _execute_function(self, hp: HP, modified_source: str) -> Dict[str, Any]:
        """
        Execute the modified source code with the given HP instance.

        Args:
            hp (HP): The HP instance for selections & overrides management.
            modified_source (str): The modified source code to execute.

        Returns:
            Dict[str, Any]: The instantiated config.
        """

        body_wo_signature = remove_function_signature(modified_source)
        function_body = textwrap.dedent(body_wo_signature)

        # Create a new namespace with the original namespace and add the 'hp' object to it
        exec_namespace = self.namespace.copy()
        exec_namespace["hp"] = hp

        # Execute the modified function body in this namespace
        exec(function_body, exec_namespace)

        # Process and filter the results
        return self._process_results(exec_namespace, hp.final_vars)

    def _process_results(self, namespace: Dict[str, Any], final_vars: List[str]) -> Dict[str, Any]:
        """
        Process and filter the execution results.

        Args:
            namespace (Dict[str, Any]): The namespace after execution.
            final_vars (List[str]): List of variables to include in the final result.

        Returns:
            Dict[str, Any]: The processed and filtered results.

        Raises:
            ValueError: If any variable in final_vars does not exist in the configuration.
        """

        filtered_locals = {
            k: v
            for k, v in namespace.items()
            if k != "hp" and not k.startswith("__") and not isinstance(v, (types.ModuleType, types.FunctionType, type))
        }

        if not final_vars:
            final_result = {k: v for k, v in filtered_locals.items() if not k.startswith("_")}
        else:
            non_existent_vars = set(final_vars) - set(filtered_locals.keys())
            if non_existent_vars:
                raise ValueError(
                    "The following variables specified in final_vars "
                    "do not exist in the configuration: "
                    f"{', '.join(non_existent_vars)}"
                )
            final_result = {k: filtered_locals[k] for k in final_vars}

        logger.debug("Captured locals: %s", filtered_locals)
        logger.debug("Final result after filtering: %s", final_result)

        return final_result

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

    def get_combinations(self):
        if self.combinations:
            return self.combinations

        hp = HP([], {}, {}, explore_mode=True)
        combinations = []

        while True:
            self._execute_function(hp, self.modified_source)
            combinations.append(hp.current_combination.copy())
            if not hp.get_next_combination(combinations):
                break
        self.combinations = combinations
        self.defaults = hp.defaults  # TODO: improve this
        return combinations

    def get_defaults(self):
        if not self.combinations:
            self.get_combinations()
        return self.defaults

    def get_snapshot_history(self):
        return self.snapshot_history

    def get_last_snapshot(self):
        if not self.snapshot_history:
            return {}
        return self.snapshot_history[-1]


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

    modified_source = "from hypster import HP\n\n" + hp_func_source
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
