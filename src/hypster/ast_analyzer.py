import ast
from typing import Any, Dict, List, Tuple

from .logging_utils import configure_logging

logger = configure_logging()

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

class HPCall:
    def __init__(self, lineno: int, call_index: int, method_name: str, explicit_name: str = None, implicit_name: str = None):
        self.lineno = lineno
        self.call_index = call_index
        self.method_name = method_name
        self.explicit_name = explicit_name
        self.implicit_name = implicit_name
        
class HPCallAnalyzer(ast.NodeVisitor):
    def __init__(self, variable_graph):
        self.variable_graph = variable_graph
        self.results = {}
        self.current_assignment = None
        self.current_complex_assignment = None
        self.hp_calls = []

    def visit_Assign(self, node):
        logger.debug(f"HPCallAnalyzer: Visiting assignment on line {node.lineno}")
        self.current_assignment = node
        start_line = node.lineno
        end_line = self.get_last_line(node)
        
        if isinstance(node.value, (ast.Dict, ast.Call)):
            self.current_complex_assignment = node
        
        self.results[start_line] = {'code': ast.unparse(node), 'hp_calls': [], 'end_line': end_line}
        self.generic_visit(node)
        
        self.current_complex_assignment = None
        self.current_assignment = None

    def visit_Call(self, node):
        if self.is_hp_call(node):
            logger.debug(f"HPCallAnalyzer: hp call detected on line {node.lineno}")
            line_number = node.lineno
            call_index = len([c for c in self.hp_calls if c.lineno == line_number])
            
            method_name = node.func.attr
            explicit_name = self.get_hp_call_name(node)
            inferred_name = self.infer_name(node)
            
            if not explicit_name and inferred_name and self.is_valid_inference(inferred_name):
                implicit_name = inferred_name
            else:
                implicit_name = None
            
            hp_call = HPCall(line_number, call_index, method_name, explicit_name, implicit_name)
            self.hp_calls.append(hp_call)
            
            if self.current_complex_assignment and self.current_complex_assignment.lineno <= line_number <= self.get_last_line(self.current_complex_assignment):
                logger.debug(f"HPCallAnalyzer: hp call is part of a complex assignment starting on line {self.current_complex_assignment.lineno}")
                line_number = self.current_complex_assignment.lineno
            
            if line_number in self.results:
                self.results[line_number]['hp_calls'].append(hp_call)
            else:
                self.results[line_number] = {
                    'code': ast.unparse(self.current_complex_assignment or node),
                    'hp_calls': [hp_call],
                    'end_line': self.get_last_line(self.current_complex_assignment or node)
                }
        
        self.generic_visit(node)

    def is_valid_inference(self, inferred_name):
        logger.debug(f"HPCallAnalyzer: Checking validity of inferred name: {inferred_name}")
        parts = inferred_name.split('.')
        valid = len(parts) <= 4 and all(not part.startswith('arg') for part in parts)
        logger.debug(f"HPCallAnalyzer: Inferred name {'is' if valid else 'is not'} valid")
        return valid

    def get_last_line(self, node):
        return max(getattr(node, 'lineno', 0), getattr(node, 'end_lineno', 0))

    def is_hp_call(self, node):
        return (isinstance(node.func, ast.Attribute) and
                isinstance(node.func.value, ast.Name) and
                node.func.value.id == 'hp')

    def get_hp_call_name(self, node):
        logger.debug(f"HPCallAnalyzer: Attempting to get hp call name for node on line {node.lineno}")
        if len(node.args) >= 2:
            name = self.get_node_value(node.args[1])
            logger.debug(f"HPCallAnalyzer: Found name in second argument: {name}")
            return name
        for keyword in node.keywords:
            if keyword.arg == 'name':
                name = self.get_node_value(keyword.value)
                logger.debug(f"HPCallAnalyzer: Found name in keyword argument: {name}")
                return name
        logger.debug("HPCallAnalyzer: No explicit name found for hp call")
        return None

    def get_node_value(self, node):
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.Name):
            return node.id
        else:
            return ast.unparse(node)

    def infer_name(self, node):
        logger.debug(f"HPCallAnalyzer: Attempting to infer name for node on line {node.lineno}")
        if self.current_assignment:
            target = self.get_target_name(self.current_assignment.targets[0])
            if isinstance(self.current_assignment.value, ast.Call) and self.is_hp_call(self.current_assignment.value):
                logger.debug(f"HPCallAnalyzer: Direct assignment to hp call detected: {target}")
                return target
            inferred = self.find_node_in_assignment(self.current_assignment.value, node, [target])
            logger.debug(f"HPCallAnalyzer: Inferred name from current assignment: {inferred}")
            return inferred
        
        for var, subgraph in self.variable_graph.items():
            path = self.find_node_in_graph(subgraph, node)
            if path:
                inferred = '.'.join([var] + [p for p in path if p != 'kwargs'])
                logger.debug(f"HPCallAnalyzer: Inferred name from variable graph: {inferred}")
                return inferred
        logger.debug("HPCallAnalyzer: Unable to infer name")
        return None

    def find_node_in_assignment(self, value_node, target_node, path):
        logger.debug(f"HPCallAnalyzer: Searching for node in assignment, current path: {'.'.join(path)}")
        if isinstance(value_node, ast.Dict):
            for key, value in zip(value_node.keys, value_node.values):
                if value == target_node:
                    result = '.'.join(path + [self.get_node_value(key)])
                    logger.debug(f"HPCallAnalyzer: Found node in dictionary: {result}")
                    return result
                result = self.find_node_in_assignment(value, target_node, path + [self.get_node_value(key)])
                if result:
                    return result
        elif isinstance(value_node, ast.Call):
            for idx, arg in enumerate(value_node.args):
                if arg == target_node:
                    result = '.'.join(path + [f'arg{idx}'])
                    logger.debug(f"HPCallAnalyzer: Found node in function argument: {result}")
                    return result
                result = self.find_node_in_assignment(arg, target_node, path + [f'arg{idx}'])
                if result:
                    return result
            for keyword in value_node.keywords:
                if keyword.value == target_node:
                    result = '.'.join(path + [keyword.arg])
                    logger.debug(f"HPCallAnalyzer: Found node in keyword argument: {result}")
                    return result
                result = self.find_node_in_assignment(keyword.value, target_node, path + [keyword.arg])
                if result:
                    return result
        logger.debug("HPCallAnalyzer: Node not found in assignment")
        return None

    def find_node_in_graph(self, graph, node, path=None):
        if path is None:
            path = []
        
        logger.debug(f"HPCallAnalyzer: Searching for node in graph, current path: {'.'.join(path)}")
        if isinstance(graph, dict):
            if graph.get('__call__', '').startswith('hp.') and graph['args'] and self.get_node_value(node.args[0]) == graph['args'][0]:
                logger.debug(f"HPCallAnalyzer: Found matching hp call in graph at path: {'.'.join(path)}")
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
        logger.debug("HPCallAnalyzer: Node not found in graph")
        return None

    def get_target_name(self, node):
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self.get_target_name(node.value)}.{node.attr}"
        else:
            return "Unknown"
        
def analyze_hp_calls(code: str) -> Tuple[Dict[int, Dict[str, Any]], List[HPCall]]:
    logger.info("Starting HP calls analysis")
    tree = ast.parse(code)
    
    logger.debug("Building variable graph")
    graph_builder = VariableGraphBuilder()
    graph_builder.visit(tree)
    
    logger.debug("Analyzing HP calls")
    analyzer = HPCallAnalyzer(graph_builder.graph)
    analyzer.visit(tree)
    
    logger.info("HP calls analysis complete")
    return analyzer.results, analyzer.hp_calls

def inject_names(source_code: str, hp_calls: List[HPCall]) -> str:
    tree = ast.parse(source_code)
    
    class NameInjector(ast.NodeTransformer):
        def __init__(self, hp_calls):
            self.hp_calls = hp_calls
            self.call_index = {}

        def visit_Call(self, node):
            if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == 'hp':
                lineno = node.lineno
                method_name = node.func.attr
                
                if lineno not in self.call_index:
                    self.call_index[lineno] = {}
                if method_name not in self.call_index[lineno]:
                    self.call_index[lineno][method_name] = 0
                else:
                    self.call_index[lineno][method_name] += 1

                hp_call = next((c for c in self.hp_calls if c.lineno == lineno and 
                                c.method_name == method_name and 
                                c.call_index == self.call_index[lineno][method_name]), None)
                
                if hp_call and not hp_call.explicit_name and hp_call.implicit_name:
                    # Inject the implicit name as a keyword argument
                    node.keywords.append(ast.keyword(arg='name', value=ast.Constant(value=hp_call.implicit_name)))

            return self.generic_visit(node)

    injector = NameInjector(hp_calls)
    modified_tree = injector.visit(tree)
    return ast.unparse(modified_tree)