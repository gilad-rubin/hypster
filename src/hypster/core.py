import ast
import functools
import inspect
import textwrap
import types
from typing import Any, Callable, Dict, List, Optional, Union

from .ast_analyzer import analyze_hp_calls, inject_names
from .hp import HP
from .logging_utils import configure_logging

logger = configure_logging()


class Hypster:
    def __init__(self, source_code: str, namespace: Dict[str, Any], inject_names=True):
        self.source_code = source_code
        self.namespace = namespace
        self.inject_names = inject_names
        self._prepare_source_code()

    def _prepare_source_code(self):
        if self.inject_names:
            results, hp_calls = analyze_hp_calls(self.source_code)
            self.modified_source = inject_names(self.source_code, hp_calls)
        else:
            self.modified_source = self.source_code

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

    def save(self, path: str):
        save(self, path)


def config(arg: Union[Callable, None] = None, *, inject_names: bool = True):
    def decorator(func: Callable) -> Hypster:
        @functools.wraps(func)
        def wrapper():
            source_code = inspect.getsource(func)
            config_body = find_hp_function_body(source_code)
            namespace = {"HP": HP}
            return Hypster(config_body, namespace, inject_names=inject_names)

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
        path = f"{hypster_instance.func.__name__}.py"

    hp_func_source = find_hp_function_body(hypster_instance.source_code)

    # Adding "from hypster import HP" to enable importing the module
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
    config_body = find_hp_function_body(module_source)
    if config_body is None:
        raise ValueError("No configuration function found in the module")

    return Hypster(config_body, namespace, inject_names)


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


def find_hp_function_body(source_code: str) -> Optional[str]:
    dedented_source = textwrap.dedent(source_code)
    tree = ast.parse(dedented_source)
    function_node = get_hp_function_node(tree)
    function_body = ast.get_source_segment(dedented_source, function_node)
    return function_body


def remove_function_signature(source: str) -> str:
    lines = source.split("\n")
    for i, line in enumerate(lines):
        if line.strip().endswith(":"):
            return "\n".join(lines[i + 1 :])
    raise ValueError("Could not find function signature")
