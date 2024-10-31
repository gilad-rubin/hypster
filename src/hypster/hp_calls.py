from abc import abstractmethod
from typing import Any, Dict, Generic, List, Optional, Type, TypeVar, Union

from pydantic import (
    BaseModel,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    ValidationError,
    field_validator,
    model_validator,
)

ValidKeyType = Union[str, int, float, bool]
OptionsType = Union[Dict[ValidKeyType, Any], List[ValidKeyType]]
NumericType = Union[StrictInt, StrictFloat]


class NumberInputCall(BaseModel):
    single_value: bool = True
    default: NumericType


T = TypeVar("T", int, str, bool, NumericType)


class HPCallError(ValueError):
    def __init__(self, name: str, message: str):
        super().__init__(f"{message} for '{name}'")


class BaseHPCall(BaseModel):
    """Base class for all HP calls."""

    name: str
    single_value: bool

    def execute(self, values: Dict[str, Any], potential_values: List[Any], explore_mode: bool) -> Any:
        """Execute the HP call."""
        if self.name in values:
            return self.process_value(values[self.name])

        if explore_mode:
            for value in potential_values:
                try:
                    return self.process_value(value)
                except (ValueError, HPCallError, ValidationError):
                    continue

        return self.get_fallback_value(explore_mode)

    @abstractmethod
    def process_value(self, value: Any) -> Any:
        """Process and validate input value."""
        pass

    @abstractmethod
    def get_fallback_value(self, explore_mode: bool) -> Any:
        """Get fallback value when no input is provided."""
        pass


class NumericBounds(BaseModel):
    """Numeric bounds configuration with validation."""

    min_val: Optional[NumericType] = None
    max_val: Optional[NumericType] = None

    @field_validator("max_val")
    def validate_bounds(cls, v: Optional[NumericType], info) -> Optional[NumericType]:
        min_val = info.data.get("min_val")
        if v is not None and min_val is not None and v < min_val:
            raise ValueError("max_val must be greater than min_val")
        return v

    def validate_value(self, name: str, value: NumericType) -> None:
        """Validate a numeric value against bounds."""
        if self.min_val is not None and value < self.min_val:
            raise HPCallError(name, f"Value must be >= {self.min_val}")
        if self.max_val is not None and value > self.max_val:
            raise HPCallError(name, f"Value must be <= {self.max_val}")


class OptionsHPCall(BaseHPCall):
    """Base class for HP calls with options."""

    options: OptionsType
    options_only: bool = False
    default: Optional[Union[ValidKeyType, List[ValidKeyType]]] = None

    @property
    def processed_options(self) -> Dict[ValidKeyType, Any]:
        if isinstance(self.options, list):
            return {item: item for item in self.options}
        return self.options

    @model_validator(mode="after")
    def validate_defaults(self) -> "OptionsHPCall":
        """Validate that default value(s) exist in processed options if options_only is True."""
        if self.default is None:
            return self
        options = self.processed_options
        if self.single_value:
            if self.default not in options:
                raise HPCallError(self.name, f"Default value '{self.default}' must be one of the options")
        else:
            for value in self.default:
                if value not in options:
                    raise HPCallError(self.name, f"Default value '{value}' must be one of the options")
        return self

    def process_value(self, value: Any) -> Any:
        if self.single_value:
            if isinstance(value, list):
                raise HPCallError(self.name, "Expected single value, got a list")
            return self._process_single_value(value)

        if not isinstance(value, list):
            raise HPCallError(self.name, "Expected a list of values, got a single value")
        return [self._process_single_value(v) for v in value]

    def _process_single_value(self, value: Any) -> Any:
        if value in self.processed_options:
            return self.processed_options[value]
        if self.options_only:
            raise HPCallError(self.name, f"Value '{value}' must be one of the options")
        return value

    def get_fallback_value(self, explore_mode: bool) -> Any:
        if self.default is not None:
            return self.process_value(self.default)

        if explore_mode:
            return next(iter(self.processed_options.values()))

        raise HPCallError(self.name, "No default or value defined")


class ValueHPCall(BaseHPCall, Generic[T]):
    """Base class for value-based HP calls."""

    default: T | List[T]
    bounds: Optional[NumericBounds] = None

    def process_value(self, value: T | List[T]) -> T | List[T]:
        if self.single_value:
            if isinstance(value, list):
                raise HPCallError(self.name, "Expected single value, got a list")
            self._validate_single_value(value)
            return value

        if not isinstance(value, list):
            raise HPCallError(self.name, "Expected a list of values, got a single value")
        for v in value:
            self._validate_single_value(v)
        return value

    def get_expected_type(self, value: T | List[T]) -> Type[T]:
        return type(value[0] if isinstance(value, list) else value)

    def _validate_single_value(self, value: T) -> None:
        expected_type = self.get_expected_type(self.default)
        if not isinstance(value, expected_type):
            raise HPCallError(self.name, f"Expected value of type {expected_type}, got {type(value)}")

        if self.bounds and isinstance(value, (int, float)):
            self.bounds.validate_value(self.name, value)

    def get_fallback_value(self, explore_mode: bool) -> Any:
        return self.process_value(self.default)


class SelectCall(OptionsHPCall):
    single_value: bool = True
    default: Optional[ValidKeyType] = None


class MultiSelectCall(OptionsHPCall):
    single_value: bool = False
    default: List[ValidKeyType] = Field(default_factory=list)


class NumberInputCall(ValueHPCall[NumericType]):
    single_value: bool = True
    default: NumericType


class MultiNumberCall(ValueHPCall[NumericType]):
    single_value: bool = False
    default: List[NumericType] = Field(default_factory=list)


class IntInputCall(ValueHPCall[int]):
    single_value: bool = True
    default: StrictInt


class MultiIntCall(ValueHPCall[int]):
    single_value: bool = False
    default: List[StrictInt] = Field(default_factory=list)


class TextInputCall(ValueHPCall[str]):
    single_value: bool = True
    default: StrictStr


class MultiTextCall(ValueHPCall[str]):
    single_value: bool = False
    default: List[StrictStr] = Field(default_factory=list)


class BoolInputCall(ValueHPCall[bool]):
    single_value: bool = True
    default: StrictBool


class MultiBoolCall(ValueHPCall[bool]):
    single_value: bool = False
    default: List[StrictBool] = Field(default_factory=list)


class PropagateCall(BaseHPCall):
    """Handles nested configuration propagation."""

    single_value: bool = True

    def process_value(self, value: Any) -> Any:
        return value

    def get_fallback_value(self, explore_mode: bool) -> Any:
        return None
