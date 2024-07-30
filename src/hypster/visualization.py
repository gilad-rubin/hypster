from typing import Set
from .nodes import ConfigNode

def visualize_config_tree(root: ConfigNode) -> str:
    def _visualize(node: ConfigNode, prefix: str = "", is_last: bool = True, visited: Set[str] = set()) -> str:
        node_name = node.name.split('__')[-1]
        if node_name in visited:
            return f"{prefix}{'└── ' if is_last else '├── '}{node_name} ({node.type}) [SHARED]\n"
        
        visited.add(node_name)
        result = f"{prefix}{'└── ' if is_last else '├── '}{node_name} ({node.type})"
        if node.value is not None:
            result += f": {node.value}"
        if node.is_shared:
            result += " [SHARED]"
        result += "\n"

        children = list(node.children.values())
        for i, child in enumerate(children):
            is_last_child = i == len(children) - 1
            result += _visualize(
                child,
                prefix + ("    " if is_last else "│   "),
                is_last_child,
                visited
            )
        return result

    return _visualize(root)