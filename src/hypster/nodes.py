import ast
import importlib
import inspect
from dataclasses import dataclass, field
from types import ModuleType
from typing import Any, Dict, List, Set

from .core import Select


@dataclass
class ConfigNode:
    name: str
    type: str
    value: Any = None
    children: Dict[str, "ConfigNode"] = field(default_factory=dict)
    parents: Set[int] = field(
        default_factory=set
    )  # Store parent IDs instead of objects
    is_shared: bool = False

    def add_child(self, child: "ConfigNode"):
        self.children[child.name] = child
        child.parents.add(id(self))

    def __hash__(self):
        return hash((self.name, self.type))

    def __eq__(self, other):
        if not isinstance(other, ConfigNode):
            return NotImplemented
        return self.name == other.name and self.type == other.type


class NodeFactory:
    def __init__(self, module_vars):
        self.module_vars = module_vars
        self.all_nodes: Dict[str, ConfigNode] = {}
        self.value_references: Dict[str, List[ConfigNode]] = {}

    def create_node(
        self, name: str, node_type: str, value: Any = None, namespace: str = ""
    ) -> ConfigNode:
        full_name = f"{namespace}.{name}" if namespace else name
        if full_name not in self.all_nodes:
            node = ConfigNode(name, node_type, value)
            self.all_nodes[full_name] = node
            if value is not None:
                self.value_references.setdefault(full_name, []).append(node)
        return self.all_nodes[full_name]

    def create_node_from_prep(self, name: str, prep_obj: Any) -> ConfigNode:
        if isinstance(prep_obj, Select):
            return self.create_node(name, "Select", prep_obj.name)
        elif isinstance(prep_obj, ast.Call):
            if prep_obj.func.id == "prep":
                return self._create_node_from_prep_call(name, prep_obj.args[0])
            elif prep_obj.func.id == "Select":
                return self.create_node(name, "Select", prep_obj.args[0].s)
        return self.create_node(name, "value", prep_obj)

    def _create_node_from_prep_call(self, name: str, arg: ast.AST) -> ConfigNode:
        if isinstance(arg, ast.Call):
            node = self.create_node(name, arg.func.id)
            for kw in arg.keywords:
                child_name = kw.arg
                child_node = self._create_child_node(
                    child_name, kw.value, namespace=name
                )
                node.add_child(child_node)
            return node
        else:
            return self.create_node(name, "value", arg)

    def _create_child_node(
        self, name: str, value: ast.AST, namespace: str
    ) -> ConfigNode:
        if isinstance(value, ast.Name):
            child_value = self.module_vars.get(value.id)
            return self.create_node(name, "reference", child_value, namespace=namespace)
        elif isinstance(value, ast.Call) and value.func.id == "Select":
            select_name = value.args[0].s
            return self.create_node(name, "Select", select_name, namespace=namespace)
        else:
            try:
                child_value = ast.literal_eval(value)
                return self.create_node(name, "value", child_value, namespace=namespace)
            except ValueError:
                return self.create_node(name, "value", value, namespace=namespace)

    def mark_shared_values(self):
        for nodes in self.value_references.values():
            if len(nodes) > 1:
                for node in nodes:
                    node.is_shared = True


class TreeBuilder:
    def __init__(self, node_factory: NodeFactory):
        self.node_factory = node_factory
        self.root = ConfigNode("root", "root")
        self.select_nodes = {}
        self.option_nodes = {}

    def build_tree(self, prepped_objects, module_vars):
        self._create_nodes_from_prepped_objects(prepped_objects)
        self._create_nodes_from_module_vars(module_vars)
        self._connect_select_nodes()
        self._add_remaining_top_level_nodes()
        self.node_factory.mark_shared_values()
        return self.root

    def _create_nodes_from_prepped_objects(self, prepped_objects):
        for name, obj in prepped_objects:
            node = self.node_factory.create_node_from_prep(name, obj)
            if isinstance(obj, Select) or (isinstance(obj, ast.Call) and getattr(obj.func, 'id', None) == 'Select'):
                select_name = obj.name if isinstance(obj, Select) else obj.args[0].s
                self.select_nodes[select_name] = node
            if '__' in name:
                prefix, option = name.split('__', 1)
                self.option_nodes.setdefault(prefix, []).append((option, node))
            else:
                self.root.add_child(node)

    def _create_nodes_from_module_vars(self, module_vars):
        for name, value in module_vars.items():
            if isinstance(value, Select):
                node = self.node_factory.create_node(name, "Select", value.name)
                self.select_nodes[value.name] = node
            elif name not in self.root.children:
                node = self.node_factory.create_node(name, "value", value)
                self.root.add_child(node)

    def _connect_select_nodes(self):
        for select_name, select_node in self.select_nodes.items():
            options = self.option_nodes.get(select_name, [])
            for option, option_node in options:
                select_node.add_child(option_node)
            
            # Find the parent node for this select node
            parent_found = False
            for node in self.root.children.values():
                if isinstance(node.value, dict) and select_name in node.value:
                    node.add_child(select_node)
                    parent_found = True
                    break
            
            if not parent_found:
                self.root.add_child(select_node)

    def _add_remaining_top_level_nodes(self):
        for options in self.option_nodes.values():
            for _, option in options:
                if option.name not in self.root.children and not any(option in node.children.values() for node in self.select_nodes.values()):
                    self.root.add_child(option)

class ModuleAnalyzer:
    def __init__(self, module):
        self.module = module
        self.prepped_objects = []
        self.module_vars = {}

    def analyze(self):
        self._extract_module_variables()
        self._extract_prepped_objects()
        return self.prepped_objects, self.module_vars

    def _extract_module_variables(self):
        for name, value in vars(self.module).items():
            if not name.startswith("__") and not callable(value):
                self.module_vars[name] = value

    def _extract_prepped_objects(self):
        if isinstance(self.module, str):
            # If module is a string (module name), import it first
            import importlib

            self.module = importlib.import_module(self.module)

        source = ast.unparse(ast.parse(inspect.getsource(self.module)))
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        if isinstance(node.value, ast.Call):
                            if isinstance(
                                node.value.func, ast.Name
                            ) and node.value.func.id in ["prep", "Select"]:
                                self.prepped_objects.append((target.id, node.value))
                        elif isinstance(node.value, Select):
                            self.prepped_objects.append((target.id, node.value))