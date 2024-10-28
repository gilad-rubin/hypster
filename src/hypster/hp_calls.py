import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union

ValidKeyType = Union[str, int, float, bool]
OptionsType = Union[Dict[ValidKeyType, Any], List[ValidKeyType]]

logger = logging.getLogger(__name__)


class HPCall(ABC):
    def __init__(self, name: str, default: Any = None):
        if not name:
            raise ValueError("`name` argument is required.")
        self.name = name
        self.default = default

    @abstractmethod
    def execute(self, selections: Dict[str, Any], overrides: Dict[str, Any]) -> Any:
        pass


def validate_and_process_options(name: str, options: OptionsType) -> Dict[ValidKeyType, Any]:
    if not options:
        raise ValueError(f"Options must be provided and cannot be empty for '{name}'.")

    if isinstance(options, list):
        validate_items(name, options)
        options = {v: v for v in options}
    elif isinstance(options, dict):
        validate_items(name, options.keys())
    else:
        raise ValueError(f"Options for '{name}' must be a dictionary or a list.")

    return options


def validate_items(name: str, items: List[ValidKeyType]):
    if not all(isinstance(item, ValidKeyType) for item in items):
        invalid_items = [item for item in items if not isinstance(item, ValidKeyType)]
        raise ValueError(f"Items for '{name}' must be one of: str, int, bool, float. Invalid items: {invalid_items}")


class SelectCall(HPCall):
    def __init__(self, name: str, options: OptionsType, default: Any = None, disable_overrides: bool = False):
        super().__init__(name, default)
        self.disable_overrides = disable_overrides
        self.options = validate_and_process_options(name, options)
        self.validate_default(default)

    def validate_default(self, default: ValidKeyType):
        if default is None:
            return
        if isinstance(default, list):
            raise ValueError(f"Default for '{self.name}' must not be a list.")
        if default not in self.options:
            raise ValueError(f"Default value '{default}' for '{self.name}' must be one of the options.")

    def execute(self, selections: Dict[str, Any], overrides: Dict[str, Any]) -> Any:
        if self.disable_overrides and self.name in overrides:
            raise ValueError(f"Overrides are disabled for '{self.name}'.")

        if self.name in overrides:
            override_value = overrides[self.name]

            # this handles the case where the override is a list and one or more values\
            # are in the options keys - this is probably a mistake, since we're using hp.select()
            # which expects a single value
            if isinstance(override_value, list) and any(v in self.options for v in override_value):
                raise ValueError(
                    f"Override values {override_value} for '{self.name}' are not all valid options."
                )  # TODO: improve message and comment

            return self.options.get(override_value, override_value)

        if self.name in selections:
            selected_value = selections[self.name]
            if isinstance(selected_value, list):
                raise TypeError(f"Selection for '{self.name}' must not be a list.")
            if selected_value not in self.options:
                raise ValueError(
                    f"Invalid selection '{selected_value}' for parameter '{self.name}'. "
                    f"Not in options: {list(self.options.keys())}"
                )
            return self.options.get(selected_value, selected_value)
        elif self.default:
            return self.options.get(self.default)
        else:
            raise ValueError(f"No overrides, selections or default provided for '{self.name}'.")


class MultiSelectCall(HPCall):
    def __init__(self, name: str, options: OptionsType, default=None, disable_overrides=False):
        super().__init__(name, default)
        self.disable_overrides = disable_overrides
        self.options = validate_and_process_options(name, options)
        self.validate_default(default)

    def validate_default(self, default: List[ValidKeyType]):
        if not isinstance(default, list):
            raise ValueError(f"Default for '{self.name}' must be a list.")
        invalid_defaults = [d for d in default if d not in self.options]
        if invalid_defaults:
            raise ValueError(f"Default values {invalid_defaults} for '{self.name}' must be one of the options.")

    def execute(self, selections: Dict[str, Any], overrides: Dict[str, Any]) -> List[Any]:
        if self.disable_overrides and self.name in overrides:
            raise ValueError(f"Overrides are disabled for '{self.name}'.")

        if self.name in overrides:
            override_values = overrides[self.name]
            if not isinstance(override_values, list):
                raise TypeError(f"Override for '{self.name}' must be a list.")
            return [self.options.get(v, v) for v in override_values]

        elif self.name in selections:
            selected_values = selections[self.name]
            if not isinstance(selected_values, list):
                raise TypeError(f"Selection for '{self.name}' must be a list.")

            invalid_selections = [v for v in selected_values if v not in self.options]
            if invalid_selections:
                raise ValueError(
                    f"Selection values {invalid_selections} for parameter '{self.name}'. "
                    f"Not in options: {list(self.options.keys())}."
                )

            return [self.options.get(v) for v in selected_values]
        elif self.default is not None:
            return [self.options.get(d) for d in self.default]
        else:
            raise ValueError(f"No overrides, selections or default provided for '{self.name}'.")


class SingleValueCall(HPCall):
    """Base class for all single-value hyperparameter calls."""

    def __init__(self, name: str, default: Any, expected_type: Union[type, tuple]):
        super().__init__(name, default)
        self.expected_type = expected_type
        if not isinstance(default, expected_type):
            type_name = self._get_type_name(expected_type)
            raise TypeError(f"Default value for '{name}' must be of type {type_name}")

    def execute(self, overrides: Dict[str, Any]) -> Any:
        if self.name in overrides:
            override_value = overrides[self.name]
            if isinstance(override_value, list):
                raise TypeError(f"Override for '{self.name}' must not be a list")
            if not isinstance(override_value, self.expected_type):
                type_name = self._get_type_name(self.expected_type)
                raise TypeError(f"Override for '{self.name}' must be of type {type_name}")
            return override_value
        return self.default

    @staticmethod
    def _get_type_name(type_obj: Union[type, tuple]) -> str:
        if isinstance(type_obj, tuple):
            return " or ".join(t.__name__ for t in type_obj)
        return type_obj.__name__


class MultiValueCall(HPCall):
    """Base class for all multi-value hyperparameter calls."""

    def __init__(self, name: str, default: List[Any], expected_type: Union[type, tuple]):
        super().__init__(name, default)
        self.expected_type = expected_type
        if not isinstance(default, list) or not all(isinstance(x, expected_type) for x in default):
            type_name = self._get_type_name(expected_type)
            raise TypeError(f"Default value for '{name}' must be a list of {type_name}")

    def execute(self, overrides: Dict[str, Any]) -> List[Any]:
        if self.name in overrides:
            override_values = overrides[self.name]
            if not isinstance(override_values, list):
                raise TypeError(f"Override for '{self.name}' must be a list")
            if not all(isinstance(x, self.expected_type) for x in override_values):
                type_name = self._get_type_name(self.expected_type)
                raise TypeError(f"All override values for '{self.name}' must be of type {type_name}")
            return override_values
        return self.default

    @staticmethod
    def _get_type_name(type_obj: Union[type, tuple]) -> str:
        if isinstance(type_obj, tuple):
            return " or ".join(t.__name__ for t in type_obj)
        return type_obj.__name__


class TextInputCall(SingleValueCall):
    def __init__(self, name: str, default: str):
        super().__init__(name, default, str)


class MultiTextCall(MultiValueCall):
    def __init__(self, name: str, default: List[str]):
        super().__init__(name, default, str)


class NumberInputCall(SingleValueCall):
    def __init__(self, name: str, default: Union[int, float]):
        super().__init__(name, default, (int, float))


class MultiNumberCall(MultiValueCall):
    def __init__(self, name: str, default: List[Union[int, float]]):
        super().__init__(name, default, (int, float))


class IntInputCall(SingleValueCall):
    def __init__(self, name: str, default: int):
        super().__init__(name, default, int)


class MultiIntCall(MultiValueCall):
    def __init__(self, name: str, default: List[int]):
        super().__init__(name, default, int)


class BoolInputCall(SingleValueCall):
    def __init__(self, name: str, default: bool):
        super().__init__(name, default, bool)


class MultiBoolCall(MultiValueCall):
    def __init__(self, name: str, default: List[bool]):
        super().__init__(name, default, bool)
