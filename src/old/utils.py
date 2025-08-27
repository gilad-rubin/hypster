import ast
import logging
import textwrap
from typing import Optional, Tuple

# logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


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


def query_combinations(combinations, query):
    """
    Filter combinations based on the provided query.

    Args:
    combinations (list): List of dictionaries, each representing a combination of hyperparameters.
    query (dict): Dictionary of key-value pairs to filter the combinations.

    Returns:
    list: Filtered list of combinations that match the query criteria.
    """

    def match_combination(combination, query):
        for key, value in query.items():
            if key not in combination or combination[key] != value:
                return False
        return True

    return [comb for comb in combinations if match_combination(comb, query)]
