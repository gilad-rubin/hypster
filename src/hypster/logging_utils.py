import logging


class CustomFormatter(logging.Formatter):
    def format(self, record):
        return record.getMessage()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(CustomFormatter())
logger.addHandler(handler)

class ConfigLogger:
    @staticmethod
    def log_variable_hierarchy(variable, indent="", is_last=False, parent_name=""):
        prefix = "└── " if is_last else "├── "
        name_without_prefix = variable.name[len(parent_name):].lstrip('.')
        logger.debug(f"{indent}{prefix}{variable.format_for_log(name_without_prefix)}")

        children = variable.get_children_for_log()
        child_indent = indent + ("    " if is_last else "│   ")
        for i, child in enumerate(children):
            is_last_child = i == len(children) - 1
            ConfigLogger.log_variable_hierarchy(child, child_indent, is_last_child, variable.name)

    @staticmethod
    def log_graph(graph, stage):
        logger.debug(f"--- {stage} ---")
        roots = [n for n in graph.nodes() if graph.in_degree(n) == 0]
        for i, root in enumerate(sorted(roots)):
            is_last = i == len(roots) - 1
            ConfigLogger.log_variable_hierarchy(graph.nodes[root]['variable'], is_last=is_last)
        logger.debug("-----------------------------------")

    @staticmethod
    def log_selected_and_overridden_values(graph, selections, overrides):
        logger.debug("--- Selected and Overridden Values ---")
        for node in sorted(graph.nodes):
            if node in selections:
                logger.debug(f"{node}: Selected as {selections[node]}")
            if node in overrides:
                logger.debug(f"{node}: Overridden to {overrides[node]}")
        logger.debug("-----------------------------------")

    @staticmethod
    def log_instantiated_config(config):
        logger.debug("--- Instantiated Config ---")
        for name, value in sorted(config.items()):
            logger.debug(f"{name}: {value}")
        logger.debug("-----------------------------------")

def set_debug_level(level):
    logger.setLevel(level)
    logger.debug(f"Debug level set to {level}")