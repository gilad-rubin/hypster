import ast
import functools
import inspect
import itertools

# from .logging_utils import configure_logging
import logging
import textwrap
import types
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from .ast_analyzer import HPCall, analyze_hp_calls, collect_hp_calls, find_referenced_vars, inject_names_to_source_code
from .hp import HP, PropagatedConfig

# Correct logging configuration
logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class Hypster:
    def __init__(self, name, source_code: str, namespace: Dict[str, Any], inject_names: bool = True):
        self.name = name
        self.source_code = source_code
        self.namespace = namespace
        self.hp_calls = collect_hp_calls(self.source_code)
        self.referenced_vars = find_referenced_vars(self.source_code)

        self.modified_source = (
            inject_names_to_source_code(self.source_code, self.hp_calls) if inject_names else self.source_code
        )

        self.independent_select_calls = self.find_independent_select_calls()

    def get_combinations(self):
        hp = HP([], {}, {}, explore_mode=True)
        return self._get_combinations_recursive(hp)

    def _get_combinations_recursive(self, hp):
        combinations = []
        
        while True:
            try:
                self._execute_function(hp, self.modified_source)
                combination = hp.get_current_combination()
                print(f"Generated combination: {combination}")
                combinations.append(combination)

                if not hp.increment_last_select():
                    print("No more combinations to generate, breaking the loop")
                    break
            except Exception as e:
                print(f"Error in _get_combinations_recursive: {str(e)}")
                break

        print(f"Total combinations generated: {len(combinations)}")
        return combinations

    def find_independent_select_calls(self) -> List[str]:
        independent_vars = {
            call.implicit_name.split(".")[0]
            for call in self.hp_calls
            if call.implicit_name
            and call.method_name == "select"
            and call.implicit_name.split(".")[0] not in self.referenced_vars
        }

        return list(independent_vars)

    def __call__(
        self,
        final_vars: List[str] = [],
        selections: Dict[str, Any] = {},
        overrides: Dict[str, Any] = {},
        return_config_snapshot: bool = False,
    ):
        hp = HP(final_vars, selections, overrides)
        result = self._execute_function(hp, self.modified_source)

        if return_config_snapshot:
            config_snapshot = hp.get_config_snapshot()
            return result, config_snapshot
        else:
            return result

    def _execute_function(self, hp: HP, modified_source: str) -> Dict[str, Any]:
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
        if path is None:
            path = f"{self.name}.py"
        save(self, path)


def config(arg: Union[Callable, None] = None, *, inject_names: bool = True):
    def decorator(func: Callable) -> Hypster:
        @functools.wraps(func)
        def wrapper():
            source_code = inspect.getsource(func)
            result = find_hp_function_body_and_name(source_code)

            if result is None:
                raise ValueError("No configuration function found in the module")

            config_name, config_body = result
            namespace = {"HP": HP}
            return Hypster(config_name, config_body, namespace, inject_names=inject_names)

        return wrapper()

    if callable(arg):
        # @config used without arguments
        return decorator(arg)
    else:
        # @config(inject_names=True/False)
        return decorator


def save(hypster_instance: Hypster, path: Optional[str] = None):
    if not isinstance(hypster_instance, Hypster):
        raise ValueError("The provided object is not a Hypster instance")

    if path is None:
        path = f"{hypster_instance.name}.py"

    result = find_hp_function_body_and_name(hypster_instance.source_code)

    if result is None:
        raise ValueError("No configuration function found in the module")

    func_name, hp_func_source = result

    modified_source = "from hypster import HP\n\n" + hp_func_source

    with open(path, "w") as f:
        f.write(modified_source)

    logger.info("Configuration saved to %s", path)


def load(path: str, inject_names=True) -> Hypster:
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


# TODO: consider moving these functions to ast_analyzer
def get_hp_function_node(tree: ast.Module) -> Optional[ast.FunctionDef]:
    """
    Finds the first function with 'hp' in its signature in the given abstract syntax tree.

    :param tree: The abstract syntax tree to search in
    :return: The function definition node with 'hp' in its signature
    :raises ValueError: If no function with 'hp' is found, or if multiple functions \
    with 'hp' are found or if there's a function with 'hp' and another argument
    """
    hp_functions = []

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            has_hp = False
            has_other_args = False
            for arg in node.args.args:
                if arg.arg == "hp":
                    has_hp = True
                elif arg.arg != "self":  # Ignore 'self' for methods
                    has_other_args = True

            if has_hp:
                if has_other_args:
                    raise ValueError(f"Function '{node.name}' has 'hp' and other arguments in its signature.")
                hp_functions.append(node)

    if len(hp_functions) > 1:
        raise ValueError("Multiple functions with 'hp' in their signatures found.")
    elif len(hp_functions) == 0:
        raise ValueError("No function with 'hp' in its signature found.")

    return hp_functions[0]


def find_hp_function_body_and_name(source_code: str) -> Optional[Tuple[str, str]]:
    dedented_source = textwrap.dedent(source_code)
    tree = ast.parse(dedented_source)

    function_node = get_hp_function_node(tree)

    if function_node is None:
        return None

    # Ensure that both function_name and function_body are strings
    function_name = function_node.name if isinstance(function_node.name, str) else None
    function_body = ast.get_source_segment(dedented_source, function_node)

    # If function_body or function_name is None, return None
    if function_name is None or function_body is None:
        return None

    return function_name, function_body


def remove_function_signature(source: str) -> str:
    lines = source.split("\n")
    for i, line in enumerate(lines):
        if line.strip().endswith(":"):
            return "\n".join(lines[i + 1 :])
    raise ValueError("Could not find function signature")
