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

        # Remove excluded vars from final_vars to avoid checking for their existence
        if exclude_vars:
            final_vars = [var for var in final_vars if var not in exclude_vars]

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
            final_result = self._apply_exclude_vars(final_result, exclude_vars)

        # Convert dotted keys to nested dictionaries
        final_result = self._create_nested_structure(final_result)

        logger.debug("Collected values: %s", collected_values)
        logger.debug("Final result after filtering: %s", final_result)

        return final_result

    def _apply_exclude_vars(self, data: Dict[str, Any], exclude_vars: List[str]) -> Dict[str, Any]:
        """Apply exclude_vars filtering with support for dotted notation."""
        result = {}

        for key, value in data.items():
            should_exclude = False

            # Check if this key should be excluded
            for exclude_pattern in exclude_vars:
                if exclude_pattern == key:
                    # Direct match
                    should_exclude = True
                    break
                elif exclude_pattern.startswith(f"{key}."):
                    # This is a nested exclusion for this key
                    if isinstance(value, dict):
                        # Apply nested exclusion
                        nested_exclude_pattern = exclude_pattern[len(key) + 1 :]  # Remove "key." prefix
                        value = self._apply_exclude_vars(value, [nested_exclude_pattern])
                    # Don't exclude the key itself, but continue with modified value
                    break

            if not should_exclude:
                result[key] = value

        return result

    def _create_nested_structure(self, flat_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Convert flat dictionary with dotted keys to nested structure."""
        result = {}

        for key, value in flat_dict.items():
            if "." in key:
                # Split the key into parts
                parts = key.split(".")
                current = result

                # Navigate/create the nested structure
                for part in parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]

                # Set the final value
                current[parts[-1]] = value
            else:
                # Simple key, just set directly
                result[key] = value

        return result

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
    Save a Hypster configuration to a Python file, preserving external imports.

    Args:
        hypster_instance (Hypster): The instance to save.
        path (str, optional): The file path where the configuration will be saved.
                              If None, uses the instance name with .py extension.

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

    # Remove leading indentation to make the function start at the left margin
    import textwrap

    func_source = textwrap.dedent(func_source)

    # Extract external imports from the function's module
    external_imports = _extract_external_imports(hypster_instance.func)

    # Build the complete source
    import_lines = ["from hypster import HP"]
    if external_imports:
        import_lines.extend(external_imports)

    imports_section = "\n".join(import_lines)
    modified_source = f"{imports_section}\n\n{func_source}"
    modified_source = modified_source.rstrip("\n") + "\n"

    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    with open(path, "w") as f:
        f.write(modified_source)

    logger.info("Configuration saved to %s", path)


def _extract_external_imports(func: Callable) -> List[str]:
    """
    Extract external imports from a function's module, excluding Hypster-related imports.

    Args:
        func: The function to analyze

    Returns:
        List of import statements as strings
    """
    import ast

    try:
        # Get the module where the function is defined
        module = inspect.getmodule(func)
        if module is None:
            return []

        # Get the source of the entire module
        try:
            module_source = inspect.getsource(module)
        except OSError:
            # Module source not available (built-in, etc.)
            return []

        # Parse the module to extract imports
        tree = ast.parse(module_source)
        imports = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if not _is_hypster_related(alias.name):
                        if alias.asname:
                            imports.append(f"import {alias.name} as {alias.asname}")
                        else:
                            imports.append(f"import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                if node.module and not _is_hypster_related(node.module):
                    module_name = node.module
                    names = []
                    for alias in node.names:
                        if alias.asname:
                            names.append(f"{alias.name} as {alias.asname}")
                        else:
                            names.append(alias.name)

                    if names:
                        level_prefix = "." * (node.level or 0)
                        imports.append(f"from {level_prefix}{module_name} import {', '.join(names)}")

        return imports

    except Exception as e:
        logger.warning(f"Could not extract imports from function module: {e}")
        return []


def _is_hypster_related(module_name: str) -> bool:
    """Check if a module name is related to Hypster and should be excluded."""
    hypster_modules = {"hypster", "hypster.core", "hypster.hp", "hypster.config", "hypster.registry"}
    return module_name in hypster_modules or module_name.startswith("hypster.")


def load(path: str) -> Hypster:
    """
    Loads a Python module from the specified path and creates a Hypster instance.

    Supports multiple formats:
    - "path/to/file.py" - loads first config function from file
    - "path/to/file.py:function_name" - loads specific function from file
    - "module.submodule" - loads first config function from module
    - "module.submodule:function_name" - loads specific function from module

    Args:
        path (str): The path to load, supporting various formats

    Returns:
        Hypster: An instance of the Hypster class containing the configuration function.

    Raises:
        ValueError: If no configuration function is found
        ImportError: If module cannot be imported
    """
    # Parse the path to check for specific object notation
    if ":" in path:
        module_path, object_name = path.rsplit(":", 1)
    else:
        module_path, object_name = path, None

    # Determine if this is a file path or module path
    if module_path.endswith(".py") or "/" in module_path or "\\" in module_path:
        # File path
        return _load_from_file(module_path, object_name)
    else:
        # Module path
        return _load_from_module(module_path, object_name)


def _load_from_file(file_path: str, object_name: Optional[str] = None) -> Hypster:
    """Load configuration from a file path."""
    with open(file_path, "r") as f:
        module_source = f.read()

    # Create a namespace with HP already imported
    namespace = {"HP": HP}

    # Execute the entire module
    exec(module_source, namespace)

    return _extract_config_function(namespace, object_name, file_path)


def _load_from_module(module_path: str, object_name: Optional[str] = None) -> Hypster:
    """Load configuration from a module path."""
    try:
        import importlib

        module = importlib.import_module(module_path)
        namespace = vars(module)
        return _extract_config_function(namespace, object_name, module_path)
    except ImportError as e:
        raise ImportError(f"Cannot import module '{module_path}': {e}")


def _extract_config_function(namespace: Dict[str, Any], object_name: Optional[str], source_path: str) -> Hypster:
    """Extract configuration function from namespace."""
    if object_name:
        # Look for specific object
        if object_name not in namespace:
            raise ValueError(f"Object '{object_name}' not found in {source_path}")

        obj = namespace[object_name]
        if not callable(obj):
            raise ValueError(f"Object '{object_name}' is not callable")

        # Verify it's a config function
        if not _is_config_function(obj):
            raise ValueError(f"Object '{object_name}' is not a valid configuration function (must have 'hp' parameter)")

        return Hypster(object_name, obj, namespace)

    # Find any config function
    config_funcs = []
    for name, obj in namespace.items():
        if callable(obj) and not name.startswith("_") and _is_config_function(obj):
            config_funcs.append((name, obj))

    if not config_funcs:
        raise ValueError(f"No configuration function found in {source_path}")

    # Use the first function found
    func_name, func = config_funcs[0]
    return Hypster(func_name, func, namespace)


def _is_config_function(obj: Any) -> bool:
    """Check if an object is a valid configuration function."""
    if not callable(obj):
        return False

    # Check for HP type annotation
    if hasattr(obj, "__annotations__") and "hp" in obj.__annotations__:
        return obj.__annotations__["hp"] == HP

    # Fallback: check parameter name in signature
    try:
        sig = inspect.signature(obj)
        return "hp" in sig.parameters
    except (ValueError, TypeError):
        return False
