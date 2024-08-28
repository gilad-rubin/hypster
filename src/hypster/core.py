import ast
import inspect
import textwrap
import types
from typing import Any, Callable, Dict, List, Optional, Union

from .ast_analyzer import analyze_hp_calls, inject_names
from .logging_utils import configure_logging

logger = configure_logging()


class HP:
    def __init__(
        self,
        final_vars: List[str],
        selections: Dict[str, Any],
        overrides: Dict[str, Any],
    ):
        self.final_vars = final_vars
        self.selections = selections
        self.overrides = overrides
        self.config_dict = {}  # Stores only HP call results
        self.function_results = {}  # Stores full function results
        self.current_namespace = []
        logger.info(
            "Initialized HP with final_vars: %s, selections: %s, and overrides: %s",
            self.final_vars,
            self.selections,
            self.overrides,
        )

    def _get_full_name(self, name: str) -> str:
        return ".".join(self.current_namespace + [name])

    def _store_value(self, name: str, value: Any):
        full_name = self._get_full_name(name)
        self.config_dict[full_name] = value

    def select(
        self,
        options: Union[Dict[str, Any], List[Any]],
        name: str = None,
        default: Any = None,
    ):
        if name is None:
            raise ValueError("Name must be provided explicitly or automatically inferred.")

        full_name = self._get_full_name(name)
        if options is None or len(options) == 0:
            raise ValueError("Options must be a non-empty list or dictionary.")

        if isinstance(options, dict):
            if not all(isinstance(k, (str, int, bool, float)) for k in options.keys()):
                bad_keys = [key for key in options.keys() if not isinstance(key, (str, int, bool, float))]
                raise ValueError(f"Dictionary keys must be str, int, bool, float. got {bad_keys} instead.")
        elif isinstance(options, list):
            if not all(isinstance(v, (str, int, bool, float)) for v in options):
                raise ValueError("List values must be one of: str, int, bool, float.")
            options = {v: v for v in options}
        else:
            raise ValueError("Options must be a dictionary or a list.")

        if default is not None and default not in options:
            raise ValueError("Default value must be one of the options.")

        logger.debug(
            "Select called with options: %s, name: %s, default: %s",
            options,
            name,
            default,
        )

        result = None
        if full_name in self.overrides:
            override_value = self.overrides[full_name]
            logger.debug("Found override for %s: %s", full_name, override_value)
            if override_value in options:
                result = options[override_value]
            else:
                result = override_value
            logger.info("Applied override for %s: %s", full_name, result)
        elif full_name in self.selections:
            selected_value = self.selections[full_name]
            logger.debug("Found selection for %s: %s", full_name, selected_value)
            if selected_value in options:
                result = options[selected_value]
                logger.info("Applied selection for %s: %s", full_name, result)
            else:
                raise ValueError(
                    f"Invalid selection '{selected_value}' for '{full_name}'. Not in options: {list(options.keys())}"
                )
        elif default is not None:
            result = options[default]
        else:
            raise ValueError(f"No selection or override found for {full_name} and no default provided.")

        self._store_value(name, result)
        return result

    def text_input(self, default: Optional[str] = None, name: Optional[str] = None) -> str:
        if name is None:
            raise ValueError("Name must be provided explicitly or automatically inferred.")

        full_name = self._get_full_name(name)
        logger.debug("Text input called with default: %s, name: %s", default, full_name)

        if full_name in self.overrides:
            result = self.overrides[full_name]
        elif default is None:
            raise ValueError(f"No default value or override provided for text input {full_name}.")
        else:
            result = default

        logger.info("Text input for %s: %s", full_name, result)

        self._store_value(name, result)
        return result

    def number_input(
        self,
        default: Optional[Union[int, float]] = None,
        name: Optional[str] = None,
    ) -> Union[int, float]:
        if name is None:
            raise ValueError("Name must be provided explicitly or automatically inferred.")

        full_name = self._get_full_name(name)
        logger.debug(
            "Number input called with default: %s, name: %s",
            default,
            full_name,
        )

        if full_name in self.overrides:
            result = self.overrides[full_name]
        elif default is None:
            raise ValueError(f"No default value or override provided for number input {full_name}.")
        else:
            result = default

        # Ensure the result is of the same type as the default
        if default is not None:
            if isinstance(default, int):
                result = int(result)
            elif isinstance(default, float):
                result = float(result)

        logger.info("Number input for %s: %s", full_name, result)

        self._store_value(name, result)
        return result

    def propagate(self, config_func: Callable, name: str = None) -> Dict[str, Any]:
        logger.info(f"Propagating configuration for {name}")

        if name is None:
            name = config_func.__name__

        self.current_namespace.append(name)

        # Create dictionaries for the nested configuration
        nested_selections = {k[len(name) + 1 :]: v for k, v in self.selections.items() if k.startswith(f"{name}.")}
        nested_overrides = {k[len(name) + 1 :]: v for k, v in self.overrides.items() if k.startswith(f"{name}.")}

        # Automatically propagate final_vars
        nested_final_vars = [var[len(name) + 1 :] for var in self.final_vars if var.startswith(f"{name}.")]

        logger.debug(
            f"Propagated configuration for {name} with Selections:\n{nested_selections}\
                \n& Overrides:\n{nested_overrides}\nAuto-propagated final vars: {nested_final_vars}"
        )

        # Run the nested configuration
        result, nested_snapshot = config_func(
            final_vars=nested_final_vars,
            selections=nested_selections,
            overrides=nested_overrides,
            return_config_snapshot=True,
        )

        # Store the full result
        self.function_results[name] = result

        # Merge only the HP call results into the parent configuration
        for key, value in nested_snapshot.items():
            full_key = f"{name}.{key}"
            self.config_dict[full_key] = value

        self.current_namespace.pop()

        # Return the full result of the propagated function
        return result

    def get_config_snapshot(self) -> Dict[str, Any]:
        return self.config_dict

    def get_function_results(self) -> Dict[str, Any]:
        return self.function_results


class Hypster:
    def __init__(self, func: Callable, source_code: str = None):
        self.func = func
        self.source_code = source_code or inspect.getsource(func)

    def __call__(
        self,
        final_vars: List[str] = [],
        selections: Dict[str, Any] = {},
        overrides: Dict[str, Any] = {},
        return_config_snapshot: bool = False,
    ):
        logger.info(
            "Hypster called with final_vars: %s, selections: %s, overrides: %s, return_config_snapshot: %s",
            final_vars,
            selections,
            overrides,
            return_config_snapshot,
        )
        try:
            hp = HP(final_vars, selections, overrides)

            # Dedent the source code if necessary
            self.source_code = dedent_source(self.source_code)

            # Analyze and modify the source code
            results, hp_calls = analyze_hp_calls(self.source_code)
            modified_source = inject_names(self.source_code, hp_calls)

            # Execute the function
            result = self._execute_function(hp, modified_source)

            if return_config_snapshot:
                config_snapshot = hp.get_config_snapshot()
                return result, config_snapshot
            else:
                return result

        except Exception as e:
            logger.error("An error occurred: %s", str(e))
            raise

    def _execute_function(self, hp: HP, modified_source: str) -> Dict[str, Any]:
        # Extract the function body
        function_body = self._extract_function_body(modified_source)

        # Create a new namespace and add the 'hp' object to it
        namespace = {"hp": hp}

        # Execute the modified function body in this namespace
        exec(function_body, globals(), namespace)

        # Process and filter the results
        return self._process_results(namespace, hp.final_vars)

    def _process_results(self, namespace: Dict[str, Any], final_vars: List[str]) -> Dict[str, Any]:
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
                    f"The following variables specified in final_vars\
                          do not exist in the configuration: {', '.join(non_existent_vars)}"
                )
            final_result = {k: filtered_locals[k] for k in final_vars}

        logger.debug("Captured locals: %s", filtered_locals)
        logger.debug("Final result after filtering: %s", final_result)

        return final_result

    def save(self, path: str):
        save(self, path)

    def _extract_function_body(self, source: str) -> str:
        lines = source.split("\n")
        body_start = next(i for i, line in enumerate(lines) if line.strip().endswith(":"))
        body_lines = lines[body_start + 1 :]
        min_indent = min(len(line) - len(line.lstrip()) for line in body_lines if line.strip())
        return "\n".join(line[min_indent:] for line in body_lines)


def dedent_source(source: str) -> str:
    """
    Dedent the source code if it's unexpectedly indented.
    """
    lines = source.splitlines()
    if not lines:
        return source

    # Check if the first non-empty line is indented
    first_non_empty = next((line for line in lines if line.strip()), "")
    if not first_non_empty or not first_non_empty[0].isspace():
        # No dedenting needed
        return source

    # Check if this is a function or class definition (which should start at column 0)
    if first_non_empty.lstrip().startswith(("def ", "class ", "@")):
        # Unexpected indentation, dedent needed
        dedented = textwrap.dedent(source)
        if dedented != source:
            logger.debug("Source code was indented and has been dedented")
            return dedented

    return source


def config(func: Callable) -> Hypster:
    return Hypster(func)


def save(hypster_instance: Hypster, path: Optional[str] = None):
    if not isinstance(hypster_instance, Hypster):
        raise ValueError("The provided object is not a Hypster instance")

    if path is None:
        path = f"{hypster_instance.func.__name__}.py"

    # Dedent the source code if necessary
    source_code = dedent_source(hypster_instance.source_code)

    # Parse the dedented source code into an AST
    tree = ast.parse(source_code)

    # Find the function definition and remove decorators
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            node.decorator_list = []
            break

    # Check if the function signature contains 'HP'
    contains_hp = False
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            for arg in node.args.args:
                if arg.annotation and isinstance(arg.annotation, ast.Name) and arg.annotation.id == "HP":
                    contains_hp = True
                    break
            break

    # Convert the modified AST back to source code
    modified_source = ast.unparse(tree)

    # Add "from hypster import HP" if needed
    if contains_hp:
        modified_source = "from hypster import HP\n\n" + modified_source

    with open(path, "w") as f:
        f.write(modified_source)

    logger.info("Configuration saved to %s", path)


def load(path: str) -> Hypster:
    with open(path, "r") as f:
        source = f.read()

    # Execute the source code to define the function
    namespace = {"HP": HP}
    exec(source, namespace)

    # Find the function in the namespace
    for name, obj in namespace.items():
        if callable(obj) and not name.startswith("__"):
            # Create and return a Hypster instance with the source code
            return Hypster(obj, source_code=source)

    raise ValueError("No suitable function found in the source code")


class InvalidSelectionError(Exception):
    pass
