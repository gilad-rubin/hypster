import inspect
import logging
from typing import Any, Dict, List, Union
import networkx as nx

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


class Options:
    def __init__(self, options: Union[List, Dict], default: Any = None):
        self.options = options
        self.default = default
        
        if isinstance(self.options, list):
            self.options = {val: val for val in self.options}
        elif not isinstance(self.options, dict):
            raise ValueError("Options must be a list or a dictionary")

class LazyClass:
    def __init__(self, class_type: type, *args, **kwargs):
        self.class_type = class_type
        self.args = args
        self.kwargs = kwargs
        self.arg_names = self._get_arg_names()

    def _get_arg_names(self):
        params = self._get_init_params()
        return [name for name in params.keys() if name != 'self']

    def _get_init_params(self):
        params = {}
        for cls in self.class_type.__mro__:
            if cls == object:
                break
            if hasattr(cls, '__init__'):
                sig = inspect.signature(cls.__init__)
                params.update(sig.parameters)
        return params

class Variable:
    def __init__(self, name: str):
        self.name = name

    def get_dependencies(self):
        return []

    def resolve(self, config, selections, overrides):
        if self.name in overrides:
            return overrides[self.name]
        if self.name in selections:
            return selections[self.name]
        return self.get_value()

    def get_value(self):
        raise NotImplementedError("Subclasses must implement get_value()")

    def __repr__(self):
        return f"{self.__class__.__name__}('{self.name}')"

class Value(Variable):
    def __init__(self, name: str, value: Any):
        super().__init__(name)
        self.value = value

    def get_value(self):
        return self.value

    def __repr__(self):
        return f"Value('{self.name}', {repr(self.value)})"

class Reference(Variable):
    def __init__(self, name: str, referred_var: Union[str, Variable]):
        super().__init__(name)
        self.referred_var = referred_var

    def get_dependencies(self):
        return [self.referred_var] if isinstance(self.referred_var, Variable) else []

    def resolve(self, config, selections, overrides):
        if isinstance(self.referred_var, str):
            referred = config.get_variable(self.referred_var)
        else:
            referred = self.referred_var
        return referred.resolve(config, selections, overrides)

    def __repr__(self):
        referred_name = self.referred_var if isinstance(self.referred_var, str) else self.referred_var.name
        return f"Reference('{self.name}' -> {referred_name})"

class OptionsVariable(Variable):
    def __init__(self, name: str, options: 'Options'):
        super().__init__(name)
        self.options = options
        self._wrapped_options = self._wrap_options()

    def _wrap_options(self):
        wrapped = {}
        for k, v in self.options.options.items():
            wrapped[k] = wrap_variable(f"{self.name}.{k}", v)
        return wrapped

    def get_dependencies(self):
        return list(self._wrapped_options.values())

    def resolve(self, config, selections, overrides):
        if self.name in overrides:
            selected_key = overrides[self.name]
            if selected_key not in self._wrapped_options:
                return selected_key #override that sets the value
        
        elif self.name in selections:
            selected_key = selections[self.name]
        else:
            selected_key = self.options.default

        if selected_key not in self._wrapped_options:
            raise ValueError(f"Invalid selection '{selected_key}' for {self.name}")
        selected = self._wrapped_options[selected_key]
        
        return selected.resolve(config, selections, overrides)

    def __repr__(self):
        options_repr = ", ".join(f"{k}: {v}" for k, v in self._wrapped_options.items())
        return f"OptionsVariable('{self.name}', options={{{options_repr}}}, default='{self.options.default}')"

class LazyClassVariable(Variable):
    def __init__(self, name: str, lazy_class: 'LazyClass'):
        super().__init__(name)
        self.lazy_class = lazy_class
        self._wrapped_args = [wrap_variable(f"{name}.arg{i}", arg) 
                              for i, arg in enumerate(lazy_class.args)]
        self._wrapped_kwargs = {k: wrap_variable(f"{name}.{k}", v) 
                                for k, v in lazy_class.kwargs.items()}

    def get_dependencies(self):
        return self._wrapped_args + list(self._wrapped_kwargs.values())

    def resolve(self, config, selections, overrides):
        resolved_args = [arg.resolve(config, selections, overrides) for arg in self._wrapped_args]
        resolved_kwargs = {k: v.resolve(config, selections, overrides) for k, v in self._wrapped_kwargs.items()}
        return self.lazy_class.class_type(*resolved_args, **resolved_kwargs)

    def __repr__(self):
        args_repr = ", ".join(repr(arg) for arg in self._wrapped_args)
        kwargs_repr = ", ".join(f"{k}={repr(v)}" for k, v in self._wrapped_kwargs.items())
        return f"LazyClassVariable('{self.name}', {self.lazy_class.class_type.__name__}, args=[{args_repr}], kwargs={{{kwargs_repr}}})"

def wrap_variable(name: str, obj: Any) -> Variable:
    if isinstance(obj, Variable):
        obj.name = name
        return obj
    elif isinstance(obj, Options):
        return OptionsVariable(name, obj)
    elif isinstance(obj, LazyClass):
        return LazyClassVariable(name, obj)
    elif isinstance(obj, (int, float, str, bool)):
        return Value(name, obj)
    else:
        return Reference(name, obj)

class Composer:
    def __init__(self):
        self.modules = []

    def with_modules(self, *modules):
        self.modules.extend(modules)
        return self

    def compose(self):
        dependency_graph = nx.DiGraph()
        self._build_dependency_graph(dependency_graph)
        config = Config(dependency_graph)
        logger.debug("Configuration composed:")
        self._log_dependency_graph(dependency_graph)
        return config

    def _build_dependency_graph(self, graph: nx.DiGraph):
        for module in self.modules:
            for name, obj in inspect.getmembers(module):
                if self._should_include_variable(name, obj):
                    var = wrap_variable(name, obj)
                    graph.add_node(var.name, variable=var)
                    self._add_dependencies(graph, var)

    def _should_include_variable(self, name: str, obj: Any) -> bool:
        return (not name.startswith('__') and 
                not inspect.ismodule(obj) and 
                not inspect.isfunction(obj) and
                not inspect.isclass(obj) and
                not isinstance(obj, type) and
                (isinstance(obj, (Options, LazyClass)) or not callable(obj)))

    def _add_dependencies(self, graph: nx.DiGraph, var: Variable):
        for dep in var.get_dependencies():
            graph.add_node(dep.name, variable=dep)
            graph.add_edge(var.name, dep.name)
            self._add_dependencies(graph, dep)

    def _log_dependency_graph(self, graph: nx.DiGraph, node=None, indent=""):
        if node is None:
            roots = [n for n in graph.nodes() if graph.in_degree(n) == 0]
            for root in roots:
                self._log_dependency_graph(graph, root)
        else:
            variable = graph.nodes[node]['variable']
            logger.debug(f"{indent}{variable}")
            for child in graph.successors(node):
                self._log_dependency_graph(graph, child, indent + "  ")

class Config:
    def __init__(self, dependency_graph: nx.DiGraph):
        self.dependency_graph = dependency_graph

    def instantiate(self, final_vars, selections={}, overrides={}):
        logger.debug("Starting instantiation process")
        
        filtered_graph = self._filter_graph(final_vars)
        self._log_graph(filtered_graph, "Filtered Graph")
        
        self._log_selected_and_overridden_values(filtered_graph, selections, overrides)
        
        instantiated_config = {}
        for var_name in final_vars:
            instantiated_config[var_name] = self._resolve_variable(var_name, selections, overrides)
        
        self._log_instantiated_config(instantiated_config)
        
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
        variable = self.dependency_graph.nodes[var_name]['variable']
        return variable.resolve(self, selections, overrides)

    def _log_graph(self, graph, stage):
        logger.debug(f"--- {stage} ---")
        self._log_graph_hierarchy(graph)
        logger.debug("-----------------------------------")

    def _log_graph_hierarchy(self, graph, node=None, indent=""):
        if node is None:
            roots = [n for n in graph.nodes() if graph.in_degree(n) == 0]
            for root in roots:
                self._log_graph_hierarchy(graph, root)
        else:
            variable = graph.nodes[node]['variable']
            logger.debug(f"{indent}{variable}")
            for child in graph.successors(node):
                self._log_graph_hierarchy(graph, child, indent + "  ")

    def _log_selected_and_overridden_values(self, graph, selections, overrides):
        logger.debug("--- Selected and Overridden Values ---")
        for node in graph.nodes:
            if node in selections:
                logger.debug(f"{node}: Selected as {selections[node]}")
            if node in overrides:
                logger.debug(f"{node}: Overridden to {overrides[node]}")
        logger.debug("-----------------------------------")

    def _log_instantiated_config(self, config):
        logger.debug("--- Instantiated Config ---")
        for name, value in config.items():
            logger.debug(f"{name}: {value}")
        logger.debug("-----------------------------------")

def lazy(cls):
    def wrapper(*args, **kwargs):
        return LazyClass(cls, *args, **kwargs)
    return wrapper

def run_configuration(config_module, final_vars, selections={}, overrides={}, log_level=logging.INFO):
    logger.setLevel(log_level)
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

__all__ = ['Options', 'lazy', 'Composer', 'run_configuration']

def set_debug_level(level):
    logger.setLevel(level)
    logger.debug(f"Debug level set to {level}")