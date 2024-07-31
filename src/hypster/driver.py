from types import ModuleType
from typing import Any, Dict, List, Optional, Union

from .core import Select
from .nodes import ConfigNode, ModuleAnalyzer, NodeFactory, TreeBuilder


class Builder:
    def __init__(self):
        self.modules: List[Union[str, ModuleType]] = []

    def with_modules(self, *modules: Union[str, ModuleType]):
        self.modules.extend(modules)
        return self

    def build(self):
        root = ConfigNode("root", "root")
        module_vars = {}
        prepped_objects = []
        for module in self.modules:
            analyzer = ModuleAnalyzer(module)
            module_prepped_objects, module_vars_update = analyzer.analyze()
            prepped_objects.extend(module_prepped_objects)
            module_vars.update(module_vars_update)

        node_factory = NodeFactory(module_vars)
        tree_builder = TreeBuilder(node_factory)
        root = tree_builder.build_tree(prepped_objects, module_vars)

        return HypsterDriver(root, self.modules)

class HypsterDriver:
    def __init__(self, root: ConfigNode, modules: List[Any]):
        self.root = root
        self.classes = {}
        for module in modules:
            for name, obj in vars(module).items():
                if isinstance(obj, type):
                    self.classes[name] = obj

    def instantiate(self, final_vars: List[str], selections: Dict[str, str] = None, overrides: Dict[str, Any] = None):
        filtered_root = self.filter_config(final_vars)
        if selections:
            self.apply_selections(filtered_root, selections)
        if overrides:
            self.apply_overrides(filtered_root, overrides)
        return self._instantiate_node(filtered_root, {})

    def filter_config(self, final_vars: List[str]) -> ConfigNode:
        filtered_root = ConfigNode("root", "root")
        for var in final_vars:
            node = self._find_node(self.root, var)
            if node:
                filtered_root.add_child(node)
        return filtered_root

    def apply_selections(self, node: ConfigNode, selections: Dict[str, str]):
        for path, selection in selections.items():
            select_node = self._find_node(node, path)
            if select_node:
                if select_node.type == "Select":
                    selected_node = next((child for child in self.root.children.values() if child.name.endswith(f"__{selection}")), None)
                    if selected_node:
                        select_node.children = {selection: selected_node}
                else:
                    select_node.value = selection

    def _instantiate_node(self, node: ConfigNode, config: Dict[str, Any]) -> Any:
        if node.type == "root":
            return {child.name: self._instantiate_node(child, config.get(child.name, {})) for child in node.children.values()}
        elif node.type == "Select":
            if not node.children:
                # If no selection was made, return None or a default value
                return None
            selected_option = next(iter(node.children.keys()))
            selected_node = node.children[selected_option]
            return self._instantiate_node(selected_node, config.get(selected_option, {}))
        elif node.type == "value":
            return node.value
        elif not node.children:
            return node.value
        else:
            class_name = node.type
            class_ = self._get_class(class_name)
            if class_ is None:
                raise ValueError(f"Class {class_name} not found in the modules")
            args = {}
            for child_name, child_node in node.children.items():
                args[child_name] = self._instantiate_node(child_node, config.get(child_name, {}))
            return class_(**args)

    def apply_overrides(self, node: ConfigNode, overrides: Dict[str, Any]):
        for path, value in overrides.items():
            override_node = self._find_node(node, path)
            if override_node:
                override_node.value = value

    def _get_class(self, class_name: str) -> Optional[type]:
        return self.classes.get(class_name)

    def _find_node(self, node: ConfigNode, path: str) -> Optional[ConfigNode]:
        parts = path.split('.')
        current = node
        for part in parts:
            if part in current.children:
                current = current.children[part]
            else:
                return None
        return current