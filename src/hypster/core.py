import inspect
import logging
from typing import Any

import networkx as nx

from .logging_utils import ConfigLogger, logger, set_debug_level
from .variables import LazyClass, Options, Variable, wrap_variable


class Composer:
    def __init__(self):
        self.modules = []
        self.object_references = {}

    def with_modules(self, *modules):
        self.modules.extend(modules)
        return self

    def compose(self):
        self._gather_object_references()
        dependency_graph = nx.DiGraph()
        self._build_dependency_graph(dependency_graph)
        config = Config(dependency_graph)
        ConfigLogger.log_graph(dependency_graph, "Full Configuration Graph")
        return config

    def _build_dependency_graph(self, graph: nx.DiGraph):
        for module in self.modules:
            for name, obj in inspect.getmembers(module):
                if self._should_include_variable(name, obj):
                    var = wrap_variable(name, obj, self.object_references)
                    graph.add_node(var.name, variable=var)
                    self._add_dependencies(graph, var)

    def _gather_object_references(self):
        for module in self.modules:
            for name, obj in inspect.getmembers(module):
                if not name.startswith("__"):
                    obj_id = id(obj)
                    self.object_references[obj_id] = name

    def _should_include_variable(self, name: str, obj: Any) -> bool:
        return (
            not name.startswith("__")
            and not inspect.ismodule(obj)
            and not inspect.isfunction(obj)
            and not inspect.isclass(obj)
            and not isinstance(obj, type)
            and (isinstance(obj, (Options, LazyClass)) or not callable(obj))
        )

    def _add_dependencies(self, graph: nx.DiGraph, var: Variable):
        for dep in var.get_dependencies():
            graph.add_node(dep.name, variable=dep)
            graph.add_edge(var.name, dep.name)
            self._add_dependencies(graph, dep)


class Config:
    def __init__(self, dependency_graph: nx.DiGraph):
        self.dependency_graph = dependency_graph

    def instantiate(self, final_vars, selections={}, overrides={}):
        logger.debug("Starting instantiation process")

        filtered_graph = self._filter_graph(final_vars)
        ConfigLogger.log_graph(filtered_graph, "Filtered Graph")

        ConfigLogger.log_selected_and_overridden_values(
            filtered_graph, selections, overrides
        )

        instantiated_config = {}
        for var_name in final_vars:
            instantiated_config[var_name] = self._resolve_variable(
                var_name, selections, overrides
            )

        ConfigLogger.log_instantiated_config(instantiated_config)

        logger.debug("Instantiation process completed")
        return instantiated_config

    def _filter_graph(self, final_vars):
        filtered = nx.DiGraph()
        for var_name in final_vars:
            self._add_subgraph(filtered, var_name)
        return filtered

    def _add_subgraph(self, filtered, node):
        if node not in filtered:
            filtered.add_node(node, **self.dependency_graph.nodes[node])
            for predecessor in self.dependency_graph.predecessors(node):
                filtered.add_edge(predecessor, node)
                self._add_subgraph(filtered, predecessor)

    def _resolve_variable(self, var_name, selections, overrides):
        variable = self.dependency_graph.nodes[var_name]["variable"]
        return variable.resolve(self, selections, overrides)


def run_configuration(
    config_module, final_vars, selections={}, overrides={}, log_level=logging.INFO
):
    set_debug_level(log_level)
    logger.info("Starting configuration process")
    try:
        composer = Composer().with_modules(config_module)
        config = composer.compose()
        logger.info("Configuration composed successfully")

        logger.info("Instantiating configuration")
        instantiated_config = config.instantiate(final_vars, selections, overrides)

        logger.info("Configuration instantiated successfully")
        return instantiated_config
    except Exception as e:
        logger.error(f"Error during configuration process: {str(e)}")
        raise


__all__ = ["Composer", "Config"]
