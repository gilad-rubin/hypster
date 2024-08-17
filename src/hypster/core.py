import inspect
import logging
from functools import wraps
from typing import Any, Dict, List, Optional, Union

class CustomFormatter(logging.Formatter):
    BLACK = "\033[0;30m"
    RED = "\033[0;31m"
    GREEN = "\033[0;32m"
    BROWN = "\033[0;33m"
    BLUE = "\033[0;34m"
    PURPLE = "\033[0;35m"
    CYAN = "\033[0;36m"
    LIGHT_GRAY = "\033[0;37m"
    DARK_GRAY = "\033[1;30m"
    LIGHT_RED = "\033[1;31m"
    LIGHT_GREEN = "\033[1;32m"
    YELLOW = "\033[1;33m"
    LIGHT_BLUE = "\033[1;34m"
    LIGHT_PURPLE = "\033[1;35m"
    LIGHT_CYAN = "\033[1;36m"
    LIGHT_WHITE = "\033[1;37m"
    BOLD = "\033[1m"
    FAINT = "\033[2m"
    ITALIC = "\033[3m"
    UNDERLINE = "\033[4m"
    BLINK = "\033[5m"
    NEGATIVE = "\033[7m"
    CROSSED = "\033[9m"
    END = "\033[0m"
    reset = "\x1b[0m"
    format = "%(message)s"

    FORMATS = {
        logging.DEBUG: CYAN + "%(levelname)s" + reset + " - " + format,
        logging.INFO: GREEN + "%(levelname)s" + reset + " - " + format,
        logging.WARNING: YELLOW + "%(levelname)s" + reset + " - " + format,
        logging.ERROR: RED + "%(levelname)s" + reset + " - " + format,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


# Logging Configuration Function
def configure_logging():
    # Get the root logger
    logger = logging.getLogger()

    # Remove all existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    # Initialize the handler with the custom formatter
    handler = logging.StreamHandler()
    handler.setFormatter(CustomFormatter())

    # Set the handler for the logger
    logger.addHandler(handler)
    logger.setLevel(logging.WARNING)

    return logger


logger = configure_logging()

import ast
from typing import List, Dict, Union, Any, Tuple
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class Select:
    def __init__(self, lineno: int, select_index: int, explicit_name: str = None, implicit_name: str = None):
        self.lineno = lineno
        self.select_index = select_index
        self.explicit_name = explicit_name
        self.implicit_name = implicit_name

class VariableGraphBuilder(ast.NodeVisitor):
    def __init__(self):
        self.graph = {}
        self.current_path = []

    def visit_Assign(self, node):
        target = self.get_target_name(node.targets[0])
        logger.debug(f"VariableGraphBuilder: Visiting assignment to {target}")
        self.current_path = [target]
        self.graph[target] = self.build_subgraph(node.value)

    def build_subgraph(self, node):
        logger.debug(f"VariableGraphBuilder: Building subgraph for node type {type(node).__name__}")
        if isinstance(node, ast.Dict):
            return {self.get_key_name(k): self.build_subgraph(v) for k, v in zip(node.keys, node.values)}
        elif isinstance(node, ast.Call):
            return {
                '__call__': self.get_target_name(node.func),
                'args': [self.get_node_value(arg) for arg in node.args],
                'kwargs': {kw.arg: self.get_node_value(kw.value) for kw in node.keywords}
            }
        elif isinstance(node, (ast.Name, ast.Attribute)):
            return self.get_target_name(node)
        elif isinstance(node, ast.Constant):
            return node.value
        else:
            return {'__unknown__': ast.unparse(node)}

    def get_node_value(self, node):
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, (ast.Name, ast.Attribute)):
            return self.get_target_name(node)
        else:
            return ast.unparse(node)

    def get_target_name(self, node):
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self.get_target_name(node.value)}.{node.attr}"
        else:
            return "Unknown"

    def get_key_name(self, node):
        if isinstance(node, ast.Str):
            return node.s
        return self.get_target_name(node)

class HPSelectAnalyzer(ast.NodeVisitor):
    def __init__(self, variable_graph):
        self.variable_graph = variable_graph
        self.results = {}
        self.current_assignment = None
        self.current_complex_assignment = None
        self.selects = []

    def visit_Assign(self, node):
        logger.debug(f"HPSelectAnalyzer: Visiting assignment on line {node.lineno}")
        self.current_assignment = node
        start_line = node.lineno
        end_line = self.get_last_line(node)
        
        if isinstance(node.value, (ast.Dict, ast.Call)):
            self.current_complex_assignment = node
        
        self.results[start_line] = {'code': ast.unparse(node), 'selects': [], 'end_line': end_line}
        self.generic_visit(node)
        
        self.current_complex_assignment = None
        self.current_assignment = None

    def visit_Call(self, node):
        if self.is_hp_select(node):
            logger.debug(f"HPSelectAnalyzer: hp.select call detected on line {node.lineno}")
            line_number = node.lineno
            select_index = len([s for s in self.selects if s.lineno == line_number])
            
            explicit_name = self.get_hp_select_name(node)
            inferred_name = self.infer_name(node)
            
            if not explicit_name and inferred_name and self.is_valid_inference(inferred_name):
                implicit_name = inferred_name
            else:
                implicit_name = None
            
            select = Select(line_number, select_index, explicit_name, implicit_name)
            self.selects.append(select)
            
            if self.current_complex_assignment and self.current_complex_assignment.lineno <= line_number <= self.get_last_line(self.current_complex_assignment):
                logger.debug(f"HPSelectAnalyzer: hp.select is part of a complex assignment starting on line {self.current_complex_assignment.lineno}")
                line_number = self.current_complex_assignment.lineno
            
            if line_number in self.results:
                self.results[line_number]['selects'].append(select)
            else:
                self.results[line_number] = {
                    'code': ast.unparse(self.current_complex_assignment or node),
                    'selects': [select],
                    'end_line': self.get_last_line(self.current_complex_assignment or node)
                }
        
        self.generic_visit(node)

    def is_valid_inference(self, inferred_name):
        logger.debug(f"HPSelectAnalyzer: Checking validity of inferred name: {inferred_name}")
        parts = inferred_name.split('.')
        valid = len(parts) <= 4 and all(not part.startswith('arg') for part in parts)
        logger.debug(f"HPSelectAnalyzer: Inferred name {'is' if valid else 'is not'} valid")
        return valid

    def get_last_line(self, node):
        return max(getattr(node, 'lineno', 0), getattr(node, 'end_lineno', 0))

    def is_hp_select(self, node):
        return (isinstance(node.func, ast.Attribute) and
                isinstance(node.func.value, ast.Name) and
                node.func.value.id == 'hp' and
                node.func.attr == 'select')

    def get_hp_select_name(self, node):
        logger.debug(f"HPSelectAnalyzer: Attempting to get hp.select name for node on line {node.lineno}")
        if len(node.args) >= 2:
            name = self.get_node_value(node.args[1])
            logger.debug(f"HPSelectAnalyzer: Found name in second argument: {name}")
            return name
        for keyword in node.keywords:
            if keyword.arg == 'name':
                name = self.get_node_value(keyword.value)
                logger.debug(f"HPSelectAnalyzer: Found name in keyword argument: {name}")
                return name
        logger.debug("HPSelectAnalyzer: No explicit name found for hp.select")
        return None

    def get_node_value(self, node):
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.Name):
            return node.id
        else:
            return ast.unparse(node)

    def infer_name(self, node):
        logger.debug(f"HPSelectAnalyzer: Attempting to infer name for node on line {node.lineno}")
        if self.current_assignment:
            target = self.get_target_name(self.current_assignment.targets[0])
            if isinstance(self.current_assignment.value, ast.Call) and self.is_hp_select(self.current_assignment.value):
                logger.debug(f"HPSelectAnalyzer: Direct assignment to hp.select detected: {target}")
                return target
            inferred = self.find_node_in_assignment(self.current_assignment.value, node, [target])
            logger.debug(f"HPSelectAnalyzer: Inferred name from current assignment: {inferred}")
            return inferred
        
        for var, subgraph in self.variable_graph.items():
            path = self.find_node_in_graph(subgraph, node)
            if path:
                inferred = '.'.join([var] + [p for p in path if p != 'kwargs'])
                logger.debug(f"HPSelectAnalyzer: Inferred name from variable graph: {inferred}")
                return inferred
        logger.debug("HPSelectAnalyzer: Unable to infer name")
        return None

    def find_node_in_assignment(self, value_node, target_node, path):
        logger.debug(f"HPSelectAnalyzer: Searching for node in assignment, current path: {'.'.join(path)}")
        if isinstance(value_node, ast.Dict):
            for key, value in zip(value_node.keys, value_node.values):
                if value == target_node:
                    result = '.'.join(path + [self.get_node_value(key)])
                    logger.debug(f"HPSelectAnalyzer: Found node in dictionary: {result}")
                    return result
                result = self.find_node_in_assignment(value, target_node, path + [self.get_node_value(key)])
                if result:
                    return result
        elif isinstance(value_node, ast.Call):
            for idx, arg in enumerate(value_node.args):
                if arg == target_node:
                    result = '.'.join(path + [f'arg{idx}'])
                    logger.debug(f"HPSelectAnalyzer: Found node in function argument: {result}")
                    return result
                result = self.find_node_in_assignment(arg, target_node, path + [f'arg{idx}'])
                if result:
                    return result
            for keyword in value_node.keywords:
                if keyword.value == target_node:
                    result = '.'.join(path + [keyword.arg])
                    logger.debug(f"HPSelectAnalyzer: Found node in keyword argument: {result}")
                    return result
                result = self.find_node_in_assignment(keyword.value, target_node, path + [keyword.arg])
                if result:
                    return result
        logger.debug("HPSelectAnalyzer: Node not found in assignment")
        return None

    def find_node_in_graph(self, graph, node, path=None):
        if path is None:
            path = []
        
        logger.debug(f"HPSelectAnalyzer: Searching for node in graph, current path: {'.'.join(path)}")
        if isinstance(graph, dict):
            if graph.get('__call__') == 'hp.select' and graph['args'] and self.get_node_value(node.args[0]) == graph['args'][0]:
                logger.debug(f"HPSelectAnalyzer: Found matching hp.select in graph at path: {'.'.join(path)}")
                return path
            for key, value in graph.items():
                new_path = self.find_node_in_graph(value, node, path + [key])
                if new_path:
                    return new_path
        elif isinstance(graph, list):
            for i, item in enumerate(graph):
                new_path = self.find_node_in_graph(item, node, path + [str(i)])
                if new_path:
                    return new_path
        logger.debug("HPSelectAnalyzer: Node not found in graph")
        return None

    def get_target_name(self, node):
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self.get_target_name(node.value)}.{node.attr}"
        else:
            return "Unknown"

def analyze_hp_select(code: str) -> Tuple[Dict[int, Dict[str, Any]], List[Select]]:
    logger.info("Starting hp.select analysis")
    tree = ast.parse(code)
    
    logger.debug("Building variable graph")
    graph_builder = VariableGraphBuilder()
    graph_builder.visit(tree)
    
    logger.debug("Analyzing hp.select calls")
    analyzer = HPSelectAnalyzer(graph_builder.graph)
    analyzer.visit(tree)
    
    logger.info("hp.select analysis complete")
    return analyzer.results, analyzer.selects

def inject_names(source_code: str, selects: List[Select]) -> str:
    tree = ast.parse(source_code)
    
    class NameInjector(ast.NodeTransformer):
        def __init__(self, selects):
            self.selects = selects
            self.select_index = {}

        def visit_Call(self, node):
            if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == 'hp' and node.func.attr == 'select':
                lineno = node.lineno
                if lineno not in self.select_index:
                    self.select_index[lineno] = 0
                else:
                    self.select_index[lineno] += 1

                select = next((s for s in self.selects if s.lineno == lineno and s.select_index == self.select_index[lineno]), None)
                
                if select and not select.explicit_name and select.implicit_name:
                    # Inject the implicit name as a keyword argument
                    node.keywords.append(ast.keyword(arg='name', value=ast.Constant(value=select.implicit_name)))

            return self.generic_visit(node)

    injector = NameInjector(selects)
    modified_tree = injector.visit(tree)
    return ast.unparse(modified_tree)

import inspect
import logging
import re
import textwrap
import types
from functools import wraps
from typing import Any, Dict, List, Union

import re

def add_name_to_selects(source: str) -> str:
    def process_select_call(line: str) -> str:
        # Identify the start of the select call
        select_start = line.find(".select(")
        if select_start == -1:
            return line  # No select call found

        # Find the start and end of the argument list
        start = select_start + len(".select(")
        end = start
        open_braces = 0

        while end < len(line):
            if line[end] == '{':
                open_braces += 1
            elif line[end] == '}':
                open_braces -= 1
            elif line[end] == ')' and open_braces == 0:
                break
            end += 1

        # Extract the arguments inside the select call
        select_args = line[start:end]
        rest_of_line = line[end + 1:]

        # Check if 'name=' is already present
        if "name=" not in select_args:
            # Find the variable name before the select call
            var_name_end = line.find("=", 0, select_start)
            var_name = line[:var_name_end].strip()

            # Add the name argument before default
            if "default=" in select_args:
                parts = select_args.split("default=", 1)
                select_args = f"{parts[0]}name='{var_name}', default={parts[1]}"
            else:
                select_args += f", name='{var_name}'"

        # Rebuild the line with the updated select call, ensuring the closing ")" is preserved
        return line[:start] + select_args + ")" + rest_of_line

    # Process each line in the source code
    lines = source.splitlines()
    modified_lines = [process_select_call(line) for line in lines]

    # Rejoin the modified lines into the final source code
    return "\n".join(modified_lines)

def extract_function_body(source: str) -> str:
    lines = source.split('\n')
    # Find the line where the function body starts
    body_start = next(i for i, line in enumerate(lines) if line.strip().endswith(':'))
    # Extract the function body, including nested functions if any
    body_lines = lines[body_start + 1:]
    min_indent = min(len(line) - len(line.lstrip()) for line in body_lines if line.strip())
    return '\n'.join(line[min_indent:] for line in body_lines)

def hypster(func):
    @wraps(func)
    def wrapper(final_vars: List[str]=[], selections: Dict[str, Any]={}, overrides: Dict[str, Any]={}):
        logging.info(
            f"Wrapper called with final_vars: {final_vars}, selections: {selections}, overrides: {overrides}"
        )
        try:
            hp = HP(selections, overrides)

            # Get the source code of the function
            source = inspect.getsource(func)

            results, selects = analyze_hp_select(source)
            # Modify the source to add 'name' to .select calls
            modified_source = inject_names(source, selects)
            
            modified_source.replace("@hypster", "")
            # Extract the function body
            function_body = extract_function_body(modified_source)

            # Create a new namespace and add the 'hp' object to it
            namespace = {'hp': hp}

            # Execute the modified function body in this namespace
            exec(function_body, globals(), namespace)

            # Remove 'hp' and filter out imports, modules, and built-ins
            filtered_locals = {
                k: v for k, v in namespace.items()
                if k != 'hp' and not k.startswith('__') and not isinstance(v, (types.ModuleType, types.FunctionType, type))
            }

            # Filter out variables starting with '_'
            final_result = {
                k: v for k, v in filtered_locals.items() if not k.startswith('_')
            }

            logging.debug(f"Captured locals: {filtered_locals}")
            logging.debug(f"Final result after filtering: {final_result}")

            # If final_vars is empty, return all variables except those starting with '_'
            if not final_vars:
                return final_result
            else:
                result = {k: final_result.get(k, None) for k in final_vars}
                logging.debug(f"Final result after filtering by final_vars: {result}")
                return result

        except Exception as e:
            logging.error(f"An error occurred: {str(e)}")
            raise  # Re-raise the exception for debugging

    return wrapper

class HP:
    def __init__(self, selections: Dict[str, Any], overrides: Dict[str, Any]):
        self.selections = selections
        self.overrides = overrides
        self.config_dict = {}
        logging.info(f"Initialized HP with selections: {self.selections} and overrides: {self.overrides}")

    def select(self, options: Union[Dict[str, Any], List[Any]], name=None, default=None, hide=False):
        if name is None:
            raise ValueError("Name must be provided explicitly or automatically inferred.")
        
        if isinstance(options, dict):
            if not all(isinstance(k, str) for k in options.keys()):
                bad_keys = [key for key in options.keys() if not isinstance(key, str)]
                raise ValueError(f"Dictionary keys must be strings. got {bad_keys} instead.")
        elif isinstance(options, list):
            if not all(isinstance(v, (str, int, bool, float)) for v in options):
                raise ValueError("List values must be one of: str, int, bool, float.")
            options = {v: v for v in options}
        else:
            raise ValueError("Options must be a dictionary or a list.")
                
        if default is not None and default not in options:
            raise ValueError("Default value must be one of the options.")
        
        logging.debug(f"Select called with options: {options}, name: {name}, default: {default}")

        result = None
        # Apply override if it exists
        if name in self.overrides:
            override_value = self.overrides[name]
            logging.debug(f"Found override for {name}: {override_value}")
            if override_value in options:
                result = options[override_value]
            else:
                result = override_value
            logging.info(f"Applied override for {name}: {result}")
        elif name in self.selections:
            selected_value = self.selections[name]
            logging.debug(f"Found selection for {name}: {selected_value}")
            if selected_value in options:
                result = options[selected_value]
                logging.info(f"Applied selection for {name}: {result}")
            else:
                raise InvalidSelectionError(
                    f"Invalid selection '{selected_value}' for '{name}'. Not in options: {list(options.keys())}")
        elif default is not None:
            result = default
        else:
            raise ValueError(f"No selection or override found for {name} and no default provided.")

        self.config_dict[name] = result
        return result

class InvalidSelectionError(Exception):
    pass
