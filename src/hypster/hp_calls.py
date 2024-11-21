from abc import abstractmethod
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Dict, Generic, List, Optional, Tuple, TypeVar, Union

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

if TYPE_CHECKING:
    from .core import Hypster
    from .run_history import HistoryDatabase

BasicType = Union[str, int, float, bool]
OptionsType = Union[Dict[BasicType, Any], List[BasicType]]
NumericType = Union[StrictInt, StrictFloat]


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


class StoredValue(BaseModel):
    """Value stored in run history"""

    value: BasicType
    reproducible: bool


class MultiStoredValue(BaseModel):
    """Multiple values stored in run history"""

    value: List[BasicType]
    reproducible: List[bool]


class BaseOptionsHPCall(BaseHPCall):
    """Abstract base class for options-based HP calls"""

    options: OptionsType
    options_only: bool = False
    stored_value: Optional[StoredValue | MultiStoredValue] = None

    @property
    def processed_options(self) -> Dict[BasicType, Any]:
        """Convert options to a dictionary if they are a list"""
        if isinstance(self.options, list):
            return {item: item for item in self.options}
        return self.options

    def validate_and_transform_value(self, value: Any) -> Tuple[Any, bool]:
        """Validate value and return transformed value with reproducibility flag"""
        is_reproducible = isinstance(value, BasicType)

        if value in self.processed_options.keys() or value in self.processed_options.values():
            return self.processed_options[value], is_reproducible

        if self.options_only:
            raise HPCallError(self.name, f"Value '{value}' must be one of the options")

        return value, is_reproducible

    def get_fallback_value(self, explore_mode: bool) -> Any:
        """Get fallback value when no input is provided"""
        if self.default is not None:
            return self.process_value(self.default)

        if explore_mode:
            value = next(iter(self.processed_options.keys()))
            return self.process_value(value)
        raise HPCallError(self.name, "No default or value defined")


class SelectCall(BaseOptionsHPCall):
    """Single-value selection call"""

    default: Optional[BasicType] = None
    single_value: bool = True

    @model_validator(mode="after")
    def validate_default(self) -> "SelectCall":
        """Validate that default value exists in processed options"""
        if self.default is not None and self.default not in self.processed_options:
            raise HPCallError(self.name, f"Default value '{self.default}' must be one of the options")
        return self

    def process_value(self, value: Any) -> Any:
        if isinstance(value, list):
            raise HPCallError(self.name, "Expected single value, got a list")

        processed_value, is_reproducible = self.validate_and_transform_value(value)
        self.stored_value = StoredValue(value=value if is_reproducible else str(value), reproducible=is_reproducible)
        return processed_value


class MultiSelectCall(BaseOptionsHPCall):
    """Multi-value selection call"""

    default: List[BasicType] = Field(default_factory=list)
    single_value: bool = False

    @model_validator(mode="after")
    def validate_defaults(self) -> "MultiSelectCall":
        """Validate that all default values exist in processed options"""
        for value in self.default:
            if value not in self.processed_options:
                raise HPCallError(self.name, f"Default value '{value}' must be one of the options")
        return self

    def process_value(self, value: Any) -> List[Any]:
        if not isinstance(value, list):
            raise HPCallError(self.name, "Expected a list of values, got a single value")

        results = []
        stored_values = []
        reproducible = []

        for item in value:
            processed_value, is_reproducible = self.validate_and_transform_value(item)
            results.append(processed_value)
            stored_values.append(item if is_reproducible else str(item))
            reproducible.append(is_reproducible)

        self.stored_value = MultiStoredValue(value=stored_values, reproducible=reproducible)
        return results


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
        if self.min_val is not None and value < self.min_val:
            raise HPCallError(name, f"Value must be >= {self.min_val}")
        if self.max_val is not None and value > self.max_val:
            raise HPCallError(name, f"Value must be <= {self.max_val}")


class ValueHPCall(BaseHPCall, Generic[T]):
    """Base class for value-based HP calls."""

    default: T | List[T]

    def process_value(self, value: T | List[T]) -> T | List[T]:
        if self.single_value:
            if isinstance(value, list):
                raise HPCallError(self.name, "Expected single value, got a list")
            self.validate_single_value(value)
            return value

        if not isinstance(value, list):
            raise HPCallError(self.name, "Expected a list of values, got a single value")
        for v in value:
            self.validate_single_value(v)
        return value

    def get_fallback_value(self, explore_mode: bool) -> Any:
        return self.process_value(self.default)


class NumberBaseCall(ValueHPCall[NumericType]):
    allow_int: bool = True
    allow_float: bool = True
    bounds: Optional[NumericBounds] = None

    def validate_single_value(self, value: NumericType) -> None:
        if not isinstance(value, (int, float)):
            raise HPCallError(self.name, f"Expected a number, got a non-number value: {value}")

        if not self.allow_int and isinstance(value, int):
            raise HPCallError(self.name, f"Integer values are not allowed: {value}")
        if not self.allow_float and isinstance(value, float):
            raise HPCallError(self.name, f"Float values are not allowed: {value}")

        if self.bounds:
            self.bounds.validate_value(self.name, value)


class NumberInputCall(NumberBaseCall):
    single_value: bool = True
    default: NumericType


class MultiNumberCall(NumberBaseCall):
    single_value: bool = False
    default: List[NumericType] = Field(default_factory=list)


class IntInputCall(NumberBaseCall):
    single_value: bool = True
    default: StrictInt
    allow_float: bool = False


class MultiIntCall(NumberBaseCall):
    single_value: bool = False
    default: List[StrictInt] = Field(default_factory=list)
    allow_float: bool = False


class TextInputCall(ValueHPCall[str]):
    single_value: bool = True
    default: StrictStr

    def validate_single_value(self, value: str) -> None:
        if not isinstance(value, str):
            raise HPCallError(self.name, f"Expected a string, got a non-string value: {value}")


class MultiTextCall(ValueHPCall[str]):
    single_value: bool = False
    default: List[StrictStr] = Field(default_factory=list)

    def validate_single_value(self, value: str) -> None:
        if not isinstance(value, str):
            raise HPCallError(self.name, f"Expected a string, got a non-string value: {value}")


class BoolInputCall(ValueHPCall[bool]):
    single_value: bool = True
    default: StrictBool

    def validate_single_value(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise HPCallError(self.name, f"Expected a boolean, got a non-boolean value: {value}")


class MultiBoolCall(ValueHPCall[bool]):
    single_value: bool = False
    default: List[StrictBool] = Field(default_factory=list)

    def validate_single_value(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise HPCallError(self.name, f"Expected a boolean, got a non-boolean value: {value}")


class NestedCall(BaseModel):
    """Handles nested configuration nesting."""

    name: str

    def execute(
        self,
        config_func: "Hypster",
        final_vars: List[str] = [],
        original_final_vars: List[str] = [],
        exclude_vars: List[str] = [],
        original_exclude_vars: List[str] = [],
        values: Dict[str, Any] = {},
        original_values: Dict[str, Any] = {},
        explore_mode: bool = False,
        run_history: Optional["HistoryDatabase"] = None,
    ) -> Dict[str, Any]:
        """Execute the nest call with nested configuration handling."""
        if len(original_final_vars) > 0:
            nested_final_vars = self._process_final_vars(original_final_vars)
        else:
            nested_final_vars = final_vars

        if len(original_exclude_vars) > 0:
            nested_exclude_vars = self._process_final_vars(original_exclude_vars)
        else:
            nested_exclude_vars = exclude_vars

        nested_values = values.copy()
        nested_values.update(self._extract_nested_dict(original_values))

        # Extract and add historical records with prefix before executing
        if run_history:
            nested_records = run_history.get_param_records(self.name)
            for record in nested_records.values():
                for run_records in record.run_history.get_run_records().values():
                    for nested_record in run_records.values():
                        config_func.run_history.add_record(nested_record)

        result = config_func(
            final_vars=nested_final_vars,
            exclude_vars=nested_exclude_vars,
            values=nested_values,
            explore_mode=explore_mode,
        )

        return result

    def _extract_nested_dict(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Extract nested configuration using both dot notation and direct dict values."""
        if not config:
            return {}

        result = defaultdict(dict)
        prefix_dot = f"{self.name}."

        # Process dot notation entries
        dot_entries = {k[len(prefix_dot) :]: v for k, v in config.items() if k.startswith(prefix_dot)}

        # Process direct dict entry
        direct_entry = config.get(self.name, {})
        if not isinstance(direct_entry, dict):
            direct_entry = {}

        # Merge both sources
        result.update(direct_entry)
        result.update(dot_entries)

        # Validate no conflicts exist
        self._validate_no_conflicts(direct_entry, dot_entries)

        return dict(result)

    def _validate_no_conflicts(self, direct_dict: Dict[str, Any], dot_dict: Dict[str, Any]) -> None:
        """Validate that there are no conflicts between direct and dot notation values."""
        common_keys = set(direct_dict.keys()) & set(dot_dict.keys())
        conflicts = {k: (direct_dict[k], dot_dict[k]) for k in common_keys if direct_dict[k] != dot_dict[k]}

        if conflicts:
            raise ValueError(f"Conflicting values found in nested configuration for '{self.name}': {conflicts}")

    def _process_final_vars(self, final_vars: List[str]) -> List[str]:
        """Process final variables for nested scope."""
        prefix_dot = f"{self.name}."
        return [var[len(prefix_dot) :] for var in final_vars if var.startswith(prefix_dot)]
