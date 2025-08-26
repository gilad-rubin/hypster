from contextlib import contextmanager
from typing import Any, Dict, Optional

# Global state for the current instantiation
_current_instantiation = None


class Config:
    """The configuration result object"""

    def __init__(self, values: Dict[str, Any]):
        self._values = values

    def __getattr__(self, key: str):
        if key.startswith("_"):
            return super().__getattribute__(key)
        if key in self._values:
            return self._values[key]
        raise AttributeError(f"Config has no attribute '{key}'")

    def __repr__(self):
        return f"Config({self._values})"


class Space:
    """Collector object inside 'with hp.space()' block"""

    def __init__(self, values: Dict[str, Any]):
        self._values = values  # Values provided for instantiation
        self._defined = {}  # Values being defined in this space
        self._var_counter = 0  # To track which variable is being set
        self._var_names = []  # Track order of variable names

    def __setattr__(self, key: str, value: Any):
        if key.startswith("_"):
            super().__setattr__(key, value)
        else:
            # Track the variable name
            if hasattr(self, "_var_names"):
                self._var_names.append(key)

            # If value is a placeholder from select(), resolve it
            if isinstance(value, SelectPlaceholder):
                actual_value = value.resolve(key, self._values)
            else:
                actual_value = value

            # Store both for access during definition and for final result
            if hasattr(self, "_defined"):
                self._defined[key] = actual_value
            super().__setattr__(key, actual_value)

    def _to_config(self) -> Config:
        return Config(self._defined)


class SelectPlaceholder:
    """Placeholder that gets resolved when assigned to a variable"""

    def __init__(self, options, default):
        self.options = options
        self.default = default

    def resolve(self, var_name: str, provided_values: Dict[str, Any]):
        """Resolve to actual value based on variable name and provided values"""
        if var_name in provided_values:
            value = provided_values[var_name]
            if value in self.options:
                return value
            else:
                raise ValueError(f"{var_name}={value} not in options {self.options}")
        return self.default if self.default is not None else self.options[0]


class ConfigSpace:
    """The decorated configuration space object"""

    def __init__(self, func):
        self.func = func
        self.name = func.__name__

    def __call__(self, values: Optional[Dict[str, Any]] = None, **kwargs) -> Config:
        # Merge dict values and kwargs
        all_values = values or {}
        all_values.update(kwargs)

        # Set global state for this instantiation
        global _current_instantiation
        _current_instantiation = all_values

        try:
            # Execute the configuration function
            result = self.func()
            return result
        finally:
            # Clear global state
            _current_instantiation = None

    def save(self, path: str):
        print(f"Would save {self.name} to {path}")

    def get_parameter_space(self):
        print(f"Would return parameter space for {self.name}")


@contextmanager
def space():
    """Context manager for configuration definition"""
    global _current_instantiation
    s = Space(_current_instantiation or {})
    yield s


def select(options, default=None):
    """Select parameter from options"""
    global _current_instantiation

    # If we're in an instantiation, return a placeholder
    # that will be resolved when assigned
    if _current_instantiation is not None:
        return SelectPlaceholder(options, default)

    # If not in instantiation (shouldn't happen), return default
    return default if default is not None else options[0]


def config(func):
    """Decorator that creates a ConfigSpace"""
    return ConfigSpace(func)
