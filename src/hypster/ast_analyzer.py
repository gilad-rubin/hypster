import ast
import logging
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class HPCall:
    def __init__(
        self,
        lineno: int,
        col_offset: int,
        method_name: str,
        implicit_name: Optional[str] = None,
        has_explicit_name: bool = False,
    ):
        self.lineno = lineno
        self.col_offset = col_offset
        self.method_name = method_name
        self.implicit_name = implicit_name
        self.has_explicit_name = has_explicit_name
        self.call_index = 0  # Will be set later

    def __repr__(self):
        return (
            f"HPCall(lineno={self.lineno}, col_offset={self.col_offset}, "
            f"method_name='{self.method_name}', implicit_name='{self.implicit_name}', "
            f"has_explicit_name={self.has_explicit_name}, call_index={self.call_index})"
        )


def build_parent_map(node: ast.AST) -> Dict[ast.AST, ast.AST]:
    """
    Build a parent map for all nodes in the AST.

    Args:
        node (ast.AST): The root of the AST.

    Returns:
        Dict[ast.AST, ast.AST]: A mapping from each node to its parent.
    """
    parent_map = {}
    stack = [(node, None)]  # (child, parent)

    while stack:
        current, parent = stack.pop()
        if parent is not None:
            parent_map[current] = parent
        for child in ast.iter_child_nodes(current):
            stack.append((child, current))

    logger.debug("Parent map built with %d entries", len(parent_map))
    return parent_map


class HPCallVisitor(ast.NodeVisitor):
    def __init__(self, parent_map: Dict[ast.AST, ast.AST]):
        self.parent_map = parent_map
        self.hp_calls: List[HPCall] = []
        logger.debug("Initialized HPCallVisitor")

    def visit_Call(self, node: ast.Call):
        if self.is_hp_call(node):
            logger.debug(f"Detected hp call at line {node.lineno}, column {node.col_offset}")
            implicit_name = self.infer_implicit_name(node)
            has_explicit_name = self.has_explicit_name(node)
            method_name = node.func.attr

            hp_call = HPCall(
                lineno=node.lineno,
                col_offset=node.col_offset,
                method_name=method_name,
                implicit_name=implicit_name,
                has_explicit_name=has_explicit_name,
            )
            logger.debug(f"Created HPCall instance: {hp_call}")
            self.hp_calls.append(hp_call)

        self.generic_visit(node)

    def is_hp_call(self, node: ast.Call) -> bool:
        is_hp = (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "hp"
        )
        logger.debug(f"Checking if node is HP call: {is_hp}")
        return is_hp

    def has_explicit_name(self, node: ast.Call) -> bool:
        for kw in node.keywords:
            if kw.arg == "name":
                logger.debug(f"HP call at line {node.lineno} has explicit name: {ast.dump(kw.value)}")
                return True
        logger.debug(f"HP call at line {node.lineno} does not have an explicit name")
        return False

    def infer_implicit_name(self, node: ast.Call) -> Optional[str]:
        """
        Infer the implicit name for an HP call by traversing its context.

        Args:
            node (ast.Call): The HP call node.

        Returns:
            Optional[str]: The inferred implicit name or None if unsupported.
        """
        context = []
        current_node = node

        while True:
            parent = self.parent_map.get(current_node)
            if parent is None:
                logger.debug(f"No parent found for node at line {current_node.lineno}")
                break

            # Handle different parent types
            if isinstance(parent, ast.Assign):
                target = parent.targets[0]
                target_name = self.get_target_name(target)
                context.append(("assignment", target_name))
                logger.debug(f"Found assignment to variable '{target_name}'")
                current_node = parent

            elif isinstance(parent, ast.keyword):
                arg_name = parent.arg
                context.append(("keyword", arg_name))
                logger.debug(f"Found keyword argument '{arg_name}'")
                current_node = self.parent_map.get(parent)

            elif isinstance(parent, ast.Call):
                if self.is_method_call(parent):
                    # Stop inference as it's within a method call
                    logger.debug(f"Found method call '{parent.func.attr}', stopping inference")
                    return None
                elif self.is_class_call(parent):
                    class_name = self.get_target_name(parent.func)
                    context.append(("class", class_name))
                    logger.debug(f"Found class call to '{class_name}'")
                    current_node = parent
                else:
                    # It's a function call
                    func_name = self.get_target_name(parent.func)
                    context.append(("function", func_name))
                    logger.debug(f"Found function call to '{func_name}'")
                    current_node = parent

            elif isinstance(parent, ast.Dict):
                key = self.get_dict_key(parent, current_node)
                if key:
                    context.append(("dct", key))
                    logger.debug(f"Found dictionary key '{key}'")
                    current_node = parent
                else:
                    logger.debug("Dictionary key not found for current node")
                    break

            elif isinstance(parent, ast.Attribute):
                attr_name = parent.attr
                context.append(("attribute", attr_name))
                logger.debug(f"Found attribute '{attr_name}'")
                current_node = parent

            elif isinstance(parent, ast.Subscript):
                # Extract the subscript key and include it in the context
                key = self.get_subscript_key(parent)
                if key:
                    context.append(("subscript", key))
                    logger.debug(f"Found subscript key '{key}'")
                    current_node = parent
                else:
                    logger.debug("Subscript key not found for current node")
                    break

            else:
                logger.debug(f"Encountered unsupported parent type '{type(parent).__name__}'")
                break

        if context:
            reversed_context = list(reversed([name for _, name in context]))
            implicit_name = ".".join(reversed_context)
            logger.debug(f"Inferred implicit name: '{implicit_name}'")
            return implicit_name
        logger.debug("Could not infer implicit name for hp call")
        return None

    def get_subscript_key(self, node: ast.Subscript) -> Optional[str]:
        """
        Extract the key from a subscript node.

        Args:
            node (ast.Subscript): The subscript node.

        Returns:
            Optional[str]: The key as a string or None.
        """
        if isinstance(node.slice, ast.Constant):
            return str(node.slice.value)
        elif isinstance(node.slice, ast.Index):  # For Python <3.9
            return self.get_node_value(node.slice.value)
        return None

    def is_class_call(self, node: ast.Call) -> bool:
        """
        Determine if a call node is a class instantiation.

        Args:
            node (ast.Call): The call node.

        Returns:
            bool: True if it's a class instantiation, False otherwise.
        """
        if isinstance(node.func, ast.Name):
            # Simple class name (conventionally starts with uppercase)
            is_class = node.func.id[0].isupper()
            logger.debug(f"Checking if call to '{node.func.id}' is a class: {is_class}")
            return is_class
        elif isinstance(node.func, ast.Attribute):
            # Possibly a class from a module (e.g., module.ClassName)
            is_class = node.func.attr[0].isupper()
            logger.debug(f"Checking if call to attribute '{node.func.attr}' is a class: {is_class}")
            return is_class
        return False

    def is_method_call(self, node: ast.Call) -> bool:
        """
        Determine if a call node is a method call.

        Args:
            node (ast.Call): The call node.

        Returns:
            bool: True if it's a method call, False otherwise.
        """
        return isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Call)

    def get_target_name(self, node: ast.AST) -> str:
        """
        Extract the target name from assignment targets.

        Args:
            node (ast.AST): The target node.

        Returns:
            str: The name of the target.
        """
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self.get_target_name(node.value)}.{node.attr}"
        else:
            return "Unknown"

    def get_keyword_arg(self, call_node: ast.Call, child_node: ast.AST) -> Optional[str]:
        """
        Get the keyword argument name for a given child node.

        Args:
            call_node (ast.Call): The call node containing the keyword.
            child_node (ast.AST): The child node corresponding to the keyword's value.

        Returns:
            Optional[str]: The keyword argument name or None.
        """
        for kw in call_node.keywords:
            if kw.value == child_node:
                return kw.arg
        return None

    def get_dict_key(self, dict_node: ast.Dict, child_node: ast.AST) -> Optional[str]:
        """
        Get the dictionary key for a given child node.

        Args:
            dict_node (ast.Dict): The dictionary node.
            child_node (ast.AST): The value node within the dictionary.

        Returns:
            Optional[str]: The key as a string or None.
        """
        for key, value in zip(dict_node.keys, dict_node.values):
            if value == child_node:
                return self.get_node_value(key)
        return None

    def get_node_value(self, node: ast.AST) -> str:
        """
        Safely extract the value from a node.

        Args:
            node (ast.AST): The node to extract the value from.

        Returns:
            str: The extracted value as a string.
        """
        if isinstance(node, ast.Constant):
            return str(node.value)
        elif isinstance(node, ast.Str):  # For Python <3.8
            return node.s
        else:
            try:
                return ast.unparse(node)
            except AttributeError:
                return "Unknown"


class VariableReferenceCollector(ast.NodeVisitor):
    def __init__(self):
        self.referenced_vars: Set[str] = set()
        logger.debug("Initialized VariableReferenceCollector")

    def visit_Name(self, node: ast.Name):
        if isinstance(node.ctx, ast.Load):
            self.referenced_vars.add(node.id)
            logger.debug(f"Referenced variable found: '{node.id}'")
        self.generic_visit(node)


def collect_hp_calls(code: str) -> List[HPCall]:
    """
    Collect all HP calls from the given source code.

    Args:
        code (str): The source code to analyze.

    Returns:
        List[HPCall]: A list of HPCall instances found in the code.
    """
    logger.info("Starting HP calls collection")
    try:
        tree = ast.parse(code)
        logger.debug("AST parsing successful")
    except SyntaxError as e:
        logger.error(f"Syntax error while parsing code: {e}")
        return []

    # Build parent map
    parent_map = build_parent_map(tree)

    # Initialize visitor with parent map
    visitor = HPCallVisitor(parent_map)
    visitor.visit(tree)

    logger.info(f"Collected {len(visitor.hp_calls)} HP calls")
    for call in visitor.hp_calls:
        logger.debug(f"HP Call: {call}")

    return visitor.hp_calls


def inject_names_to_source_code(code: str, hp_calls: List[HPCall]) -> str:
    """
    Inject implicit names into HP calls that lack an explicit name.

    Args:
        code (str): The original source code.
        hp_calls (List[HPCall]): The list of HPCall instances to process.

    Returns:
        str: The modified source code with injected names.
    """
    logger.info("Starting name injection into source code")
    tree = ast.parse(code)
    injector = NameInjector(hp_calls)
    modified_tree = injector.visit(tree)
    ast.fix_missing_locations(modified_tree)
    try:
        modified_code = ast.unparse(modified_tree)
        logger.debug("AST unparsed successfully using ast.unparse")
    except AttributeError:
        # For Python versions < 3.9 where ast.unparse is not available
        import astor

        modified_code = astor.to_source(modified_tree)
        logger.debug("AST unparsed successfully using astor.to_source")
    logger.info("Name injection completed")
    return modified_code


class NameInjector(ast.NodeTransformer):
    def __init__(self, hp_calls: List[HPCall]):
        self.hp_calls = hp_calls
        self.call_index = 0
        logger.debug("Initialized NameInjector")

    def visit_Call(self, node: ast.Call):
        if (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "hp"
        ):
            if self.call_index < len(self.hp_calls):
                hp_call = self.hp_calls[self.call_index]
                if hp_call.implicit_name and not hp_call.has_explicit_name:
                    logger.debug(f"Injecting name '{hp_call.implicit_name}' into hp call on line {hp_call.lineno}")
                    # Avoid duplicate 'name' arguments
                    if not any(kw.arg == "name" for kw in node.keywords):
                        node.keywords.append(
                            ast.keyword(
                                arg="name",
                                value=ast.Constant(value=hp_call.implicit_name),
                            )
                        )
                else:
                    logger.debug(f"No injection needed for hp call on line {hp_call.lineno}")
                self.call_index += 1
            else:
                logger.warning("More hp_calls than hp.Call nodes encountered")
        return self.generic_visit(node)


def find_referenced_vars(code: str) -> Set[str]:
    """
    Find all variables that are referenced (read) in the given source code.

    Args:
        code (str): The source code to analyze.

    Returns:
        Set[str]: A set of variable names that are referenced.
    """
    logger.info("Starting referenced variables collection")
    tree = ast.parse(code)
    collector = VariableReferenceCollector()
    collector.visit(tree)
    logger.info(f"Referenced variables collected: {collector.referenced_vars}")
    return collector.referenced_vars


def find_independent_select_calls(referenced_vars: Set[str], hp_calls: List[HPCall]) -> List[str]:
    """
    Identify independent hp.select calls not influenced by referenced variables.

    Args:
        referenced_vars (Set[str]): Set of variables that are referenced in the code.
        hp_calls (List[HPCall]): List of collected HPCall instances.

    Returns:
        List[str]: List of independent hp.select implicit names.
    """
    independent_vars = {
        call.implicit_name.split(".")[0]
        for call in hp_calls
        if call.implicit_name
        and call.method_name == "select"
        and call.implicit_name.split(".")[0] not in referenced_vars
    }
    logger.debug(f"Independent select calls identified: {independent_vars}")
    return list(independent_vars)
