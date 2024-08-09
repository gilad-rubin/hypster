import inspect
from typing import Any, Dict, List, Union


class Options:
    def __init__(self, options: Union[List, Dict], default: Any = None):
        self.options = options
        self.default = default
        self.original_type = type(options)

        if isinstance(self.options, list):  # TODO: allow only strings and references
            self.options = {str(val): val for val in self.options}
        elif not isinstance(self.options, dict):
            raise ValueError("Options must be a list or a dictionary")


import sys
import inspect
from typing import Union, List, Type, Dict, Optional, Callable
from functools import wraps

class LazyClass:
    def __init__(self, class_type: type, *args, **kwargs):
        self.class_type = class_type
        self.args = args
        self.kwargs = kwargs
        self.instance = None

    def __getattr__(self, name):
        if self.instance is None:
            self.instance = self.class_type(*self.args, **self.kwargs)
        return getattr(self.instance, name)

def is_class_like(obj):
    return isinstance(obj, type) or (callable(obj) and hasattr(obj, '__class__'))

def lazy(arg: Union[Type, List[Type], Callable], update_globals: bool = False, namespace: Optional[Dict] = None) -> Union[Callable, Dict[str, Callable], None]:
    """
    Make a class or multiple classes lazy.

    Args:
        arg: A class, or a list of classes to be made lazy.
        update_globals: If True, modifies the classes in the global namespace. Default is False.
        namespace: The namespace to update. If None, uses the global namespace of the caller.

    Returns:
        - If arg is a single class and update_globals is False: returns a wrapper function.
        - If arg is a list of classes and update_globals is False: returns a dictionary of wrapper functions.
        - If update_globals is True: returns None (classes are modified in-place).

    Raises:
        TypeError: If the argument is neither a class nor a list of classes.
    """
    def make_lazy_wrapper(cls):
        if hasattr(cls, '__lazy_wrapped__'):
            return cls  # If it's already lazy, just return it
        
        @wraps(cls)
        def wrapper(*args, **kwargs):
            return LazyClass(cls, *args, **kwargs)
        
        # Preserve class attributes
        for attr in dir(cls):
            if not hasattr(wrapper, attr):
                setattr(wrapper, attr, getattr(cls, attr))
        
        # Preserve type hints
        wrapper.__annotations__ = cls.__annotations__

        # Mark as lazy wrapped
        wrapper.__lazy_wrapped__ = True
        
        return wrapper

    if namespace is None:
        namespace = sys._getframe(1).f_globals

    if isinstance(arg, list):
        lazy_wrappers = {}
        for cls in arg:
            if not is_class_like(cls):
                raise TypeError(f"All elements in the list must be classes. Found: {type(cls)}")
            lazy_wrapper = make_lazy_wrapper(cls)
            lazy_wrappers[cls.__name__] = lazy_wrapper
            if update_globals:
                namespace[cls.__name__] = lazy_wrapper
        return None if update_globals else lazy_wrappers
    elif is_class_like(arg):
        lazy_wrapper = make_lazy_wrapper(arg)
        if update_globals:
            namespace[arg.__name__] = lazy_wrapper
            return None
        return lazy_wrapper
    else:
        raise TypeError("lazy() argument must be a class or a list of classes")


class Variable:
    def __init__(self, name: str):
        self.name = name

    def get_dependencies(self):
        return []

    def resolve(self, config, selections, overrides, reference=None):
        if self.name in overrides:
            return overrides[self.name]
        if self.name in selections:
            return selections[self.name]
        return self.get_value()

    def get_value(self):
        raise NotImplementedError("Subclasses must implement get_value()")

    def __repr__(self):
        return f"{self.__class__.__name__}('{self.name}')"

    def format_for_log(self, name):
        return f"{name}: {self.__class__.__name__}"

    def get_children_for_log(self):
        return []


class Value(Variable):
    def __init__(self, name: str, value: Any):
        super().__init__(name)
        self.value = value

    def get_value(self):
        return self.value

    def __repr__(self):
        return f"Value('{self.name}', {repr(self.value)})"

    def format_for_log(self, name):
        return f"{name}: {self.value}"


class Reference(Variable):
    def __init__(self, name: str, referred_var: Variable):
        super().__init__(name)
        self.referred_var = referred_var

    def get_dependencies(self):
        return [self.referred_var]

    def resolve(self, config, selections, overrides):
        return self.referred_var.resolve(
            config, selections, overrides, reference=self.name
        )

    def __repr__(self):
        referred_name = (
            self.referred_var
            if isinstance(self.referred_var, str)
            else self.referred_var.name
        )
        return f"Reference('{self.name}' -> {referred_name})"

    def format_for_log(self, name):
        referred_name = (
            self.referred_var
            if isinstance(self.referred_var, str)
            else self.referred_var.name
        )
        referred_value = (
            self.referred_var.get_value()
            if hasattr(self.referred_var, "get_value")
            else str(self.referred_var)
        )
        return f"{name} (Reference -> {referred_name}: {referred_value})"

    def get_children_for_log(self):
        return []

    def get_value(self):
        return self.referred_var.get_value()


class OptionsVariable(Variable):
    def __init__(self, name: str, options: "Options", object_references: dict):
        super().__init__(name)
        self.options = options
        self._wrapped_options = self._wrap_options(object_references)

    def _wrap_options(self, object_references):
        wrapped = {}
        for k, v in self.options.options.items():
            wrapped[k] = wrap_variable(f"{self.name}.{k}", v, object_references)
        
        options_are_list = self.options.original_type == list
        all_references = all(isinstance(w, Reference) for w in wrapped.values())
        if options_are_list and all_references:  # Special case where there's a list of references and we need to change their dict keys
            options_with_reference_names = {}
            for k, v in wrapped.items():
                reference_name = v.referred_var.name
                options_with_reference_names[reference_name] = v
            return options_with_reference_names

        return wrapped

    def get_value(self):
        return self.options.options

    def get_dependencies(self):
        return list(self._wrapped_options.values())

    def resolve(self, config, selections, overrides, reference=None):
        if self.name in overrides:
            selected_key = overrides[self.name]
            if selected_key not in self._wrapped_options:
                return selected_key  # override that sets the value

        elif self.name in selections:
            selected_key = selections[self.name]
        else:
            selected_key = self.options.default

        if reference:  # TODO: avoid DRY
            if reference in overrides:
                selected_key = overrides[reference]
                if selected_key not in self._wrapped_options:
                    return selected_key
            elif reference in selections:
                selected_key = selections[reference]

        if selected_key not in self._wrapped_options:
            raise ValueError(f"Invalid selection '{selected_key}' for {self.name}")
        selected = self._wrapped_options[selected_key]

        return selected.resolve(config, selections, overrides)

    def __repr__(self):
        options_repr = ", ".join(f"{k}: {v}" for k, v in self._wrapped_options.items())
        return f"OptionsVariable('{self.name}', options={{{options_repr}}}, default='{self.options.default}')"

    def format_for_log(self, name):
        options_str = ", ".join(f"{k}: {v}" for k, v in self.options.options.items())
        return f"{name} (OptionsVariable, options={{{options_str}}}, default='{self.options.default}')"

    def get_children_for_log(self):
        return list(self._wrapped_options.values())


class LazyClassVariable(Variable):
    def __init__(self, name: str, lazy_class: "LazyClass", object_references: dict):
        super().__init__(name)
        self.lazy_class = lazy_class
        self._wrapped_args = [
            wrap_variable(f"{name}.arg{i}", arg, object_references)
            for i, arg in enumerate(lazy_class.args)
        ]
        self._wrapped_kwargs = {
            k: wrap_variable(f"{name}.{k}", v, object_references)
            for k, v in lazy_class.kwargs.items()
        }

    def get_dependencies(self):
        return self._wrapped_args + list(self._wrapped_kwargs.values())

    def resolve(self, config, selections, overrides, reference=None):
        resolved_args = [
            arg.resolve(config, selections, overrides) for arg in self._wrapped_args
        ]
        resolved_kwargs = {
            k: v.resolve(config, selections, overrides)
            for k, v in self._wrapped_kwargs.items()
        }
        return self.lazy_class.class_type(*resolved_args, **resolved_kwargs)

    def __repr__(self):
        args_repr = ", ".join(repr(arg) for arg in self._wrapped_args)
        kwargs_repr = ", ".join(
            f"{k}={repr(v)}" for k, v in self._wrapped_kwargs.items()
        )
        return f"LazyClassVariable('{self.name}', {self.lazy_class.class_type.__name__}, args=[{args_repr}], kwargs={{{kwargs_repr}}})"

    def format_for_log(self, name):
        return f"{name} (LazyClassVariable: {self.lazy_class.class_type.__name__})"

    def get_children_for_log(self):
        return list(self._wrapped_kwargs.values())

    def get_value(self):
        args_repr = ", ".join(arg.get_value() for arg in self._wrapped_args)
        kwargs_repr = ", ".join(
            f"{k}={v.get_value()}" for k, v in self._wrapped_kwargs.items()
        )
        return f"{self.lazy_class.class_type.__name__}({args_repr}, {kwargs_repr})"


def wrap_variable(name: str, obj: Any, object_references: dict) -> Variable:
    reference_name = object_references.get(id(obj), None)
    if reference_name and reference_name != name:
        referred_wrapped = wrap_variable(reference_name, obj, object_references)
        return Reference(name, referred_wrapped)
    elif isinstance(obj, Options):
        return OptionsVariable(name, obj, object_references)
    elif isinstance(obj, LazyClass):
        return LazyClassVariable(name, obj, object_references)
    else:
        return Value(name, obj)
