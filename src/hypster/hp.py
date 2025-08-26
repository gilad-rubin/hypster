import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union
from uuid import UUID

from .hp_calls import (
    BaseHPCall,
    BasicType,
    BoolInputCall,
    IntInputCall,
    MultiBoolCall,
    MultiIntCall,
    MultiNumberCall,
    MultiSelectCall,
    MultiTextCall,
    NumberInputCall,
    NumericBounds,
    NumericType,
    SelectCall,
    TextInputCall,
)
from .run_history import HistoryDatabase, NestedHistoryRecord, ParameterRecord, ParameterSource

if TYPE_CHECKING:
    from .core import Hypster

logger = logging.getLogger(__name__)
MAX_POTENTIAL_VALUES = 5


class HP:
    def __init__(
        self,
        values: Dict[str, Any],
        run_history: HistoryDatabase,
        run_id: UUID,
        explore_mode: bool = False,
    ):
        self.values = values
        self.run_history = run_history
        self.run_id = run_id
        self.explore_mode = explore_mode
        self.source = ParameterSource.UI if explore_mode else ParameterSource.USER
        self._named_parameters = set()  # Track all named parameters that are called
        self._validate_values_on_completion = True
        logger.info(f"Initialized HP with explore_mode: {explore_mode}")

    def _validate_values(self):
        """Validate that all values keys correspond to actual named parameters"""
        if not self._validate_values_on_completion:
            return

        # Check for values that don't correspond to any named parameter
        unmatched_values = set(self.values.keys()) - self._named_parameters
        if unmatched_values:
            available = ", ".join(sorted(self._named_parameters)) if self._named_parameters else "none"
            raise ValueError(
                f"Values provided for parameters that don't exist or aren't named: {sorted(unmatched_values)}. "
                f"Available named parameters: {available}. "
                f"Make sure all HP calls that need to be overridden have explicit 'name' parameters."
            )

    def select(
        self,
        options: Union[Dict[BasicType, Any], List[BasicType]],
        *,
        name: Optional[str] = None,
        default: Optional[BasicType] = None,
        options_only: bool = False,
    ) -> Any:
        call = SelectCall(name=name, options=options, default=default, options_only=options_only)
        options_keys = list(call.processed_options.keys())
        return self._execute_call(call=call, parameter_type="select", options=options_keys)

    def multi_select(
        self,
        options: Union[Dict[BasicType, Any], List[BasicType]],
        *,
        name: Optional[str] = None,
        default: Optional[List[BasicType]] = [],
        options_only: bool = False,
    ) -> List[Any]:
        call = MultiSelectCall(name=name, options=options, default=default, options_only=options_only)
        options_keys = list(call.processed_options.keys())
        return self._execute_call(call=call, parameter_type="multi_select", options=options_keys)

    def number(
        self,
        default: NumericType,
        *,
        name: Optional[str] = None,
        min: Optional[NumericType] = None,
        max: Optional[NumericType] = None,
    ) -> NumericType:
        bounds = NumericBounds(min_val=min, max_val=max) if (min is not None or max is not None) else None
        call = NumberInputCall(name=name, default=default, bounds=bounds)
        return self._execute_call(call=call, parameter_type="number", numeric_bounds=bounds)

    def multi_number(
        self,
        default: List[NumericType] = [],
        *,
        name: Optional[str] = None,
        min: Optional[NumericType] = None,
        max: Optional[NumericType] = None,
    ) -> List[NumericType]:
        bounds = NumericBounds(min_val=min, max_val=max) if (min is not None or max is not None) else None
        call = MultiNumberCall(name=name, default=default, bounds=bounds)
        return self._execute_call(call=call, parameter_type="multi_number", numeric_bounds=bounds)

    def int(
        self,
        default: int,
        *,
        name: Optional[str] = None,
        min: Optional[int] = None,
        max: Optional[int] = None,
    ) -> int:
        bounds = NumericBounds(min_val=min, max_val=max) if (min is not None or max is not None) else None
        call = IntInputCall(name=name, default=default, bounds=bounds)
        return self._execute_call(call=call, parameter_type="int", numeric_bounds=bounds)

    def multi_int(
        self,
        default: List[int] = [],
        *,
        name: Optional[str] = None,
        min: Optional[int] = None,
        max: Optional[int] = None,
    ) -> List[int]:
        bounds = NumericBounds(min_val=min, max_val=max) if (min is not None or max is not None) else None
        call = MultiIntCall(name=name, default=default, bounds=bounds)
        return self._execute_call(call=call, parameter_type="multi_int", numeric_bounds=bounds)

    def text(self, default: str, *, name: Optional[str] = None) -> str:
        call = TextInputCall(name=name, default=default)
        return self._execute_call(call=call, parameter_type="text")

    def multi_text(self, default: List[str] = [], *, name: Optional[str] = None) -> List[str]:
        call = MultiTextCall(name=name, default=default)
        return self._execute_call(call=call, parameter_type="multi_text")

    def bool(self, default: bool, *, name: Optional[str] = None) -> bool:
        call = BoolInputCall(name=name, default=default)
        return self._execute_call(call=call, parameter_type="bool")

    def multi_bool(self, default: List[bool] = [], *, name: Optional[str] = None) -> List[bool]:
        call = MultiBoolCall(name=name, default=default)
        return self._execute_call(call=call, parameter_type="multi_bool")

    def collect(
        self, vars_dict: Dict[str, Any], include: Optional[List[str]] = None, exclude: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Sanitize locals() and return a filtered dict.

        Args:
            vars_dict: Dictionary of variables (typically from locals())
            include: Optional list of variable names to include (if provided, only these are included)
            exclude: Optional list of variable names to exclude

        Returns:
            Filtered dictionary with noise removed
        """
        import types

        # Start with all variables
        filtered = vars_dict.copy()

        # Remove noise: hp, dunder/private names, modules, functions, classes
        noise_keys = []
        for key, value in filtered.items():
            if (
                key == "hp"
                or key.startswith("__")
                or key.startswith("_")
                or isinstance(value, (types.ModuleType, types.FunctionType, type))
            ):
                noise_keys.append(key)

        for key in noise_keys:
            filtered.pop(key, None)

        # Apply include filter if provided
        if include is not None:
            # Only keep keys that are in the include list and exist in filtered
            filtered = {k: v for k, v in filtered.items() if k in include}

        # Apply exclude filter if provided
        if exclude is not None:
            for key in exclude:
                filtered.pop(key, None)

        return filtered

    def nest(
        self,
        config_func: Union[str, Path, "Hypster"],
        *,
        name: Optional[str] = None,
        values: Dict[str, Any] = None,
    ) -> Any:
        """
        Nest another configuration.

        Args:
            config_func: Can be:
                - A Hypster object directly
                - A registry alias string (e.g., "retriever/tfidf")
                - A file path to import
                - An import path (module:attr)
            name: Name for this nested config in the run history
            values: Override values for the nested config

        Returns:
            Whatever the nested config returns (pass-through)
        """
        if values is None:
            values = {}

        # Resolve the config_func to a Hypster object
        if isinstance(config_func, str):
            # Check if it's a registry alias first
            try:
                from . import registry

                config_func = registry.get(config_func)
            except (KeyError, ImportError):
                # Not a registry alias, try as file path or import path
                if ":" in config_func:
                    # Import path format "module:attr"
                    module_name, attr_name = config_func.split(":", 1)
                    import importlib

                    module = importlib.import_module(module_name)
                    config_func = getattr(module, attr_name)
                else:
                    # File path
                    from .core import load

                    config_func = load(str(config_func))
        elif isinstance(config_func, Path):
            # File path
            from .core import load

            config_func = load(str(config_func))

        # Build values specifically for the nested config
        nested_values = {}

        # Extract relevant values for this nested config
        if name:
            # Look for values prefixed with the nested config name
            prefix = f"{name}."
            for key, value in self.values.items():
                if key.startswith(prefix):
                    # Remove the prefix to get the nested parameter name
                    nested_key = key[len(prefix) :]
                    nested_values[nested_key] = value
                    # Track that we're using this parameter
                    self._named_parameters.add(key)

        # Merge with any direct values passed to nest()
        nested_values.update(values)

        # Execute the nested config
        result = config_func(values=nested_values, explore_mode=self.explore_mode)

        # Record the nested call
        record = NestedHistoryRecord(
            name=name,
            parameter_type="nest",
            run_history=config_func.run_history,
            run_id=self.run_id,
            source=self.source,
        )
        self.run_history.add_record(record)

        return result

    def _execute_call(
        self,
        call: BaseHPCall,
        parameter_type: str,
        options: Optional[List[BasicType]] = None,
        numeric_bounds: Optional[NumericBounds] = None,
    ) -> Any:
        """Execute HP call and record its result"""
        logger.debug(f"Added {parameter_type}Call: {call.name}")

        # Track named parameters
        if call.name is not None:
            self._named_parameters.add(call.name)

        potential_values = self._get_potential_values(call.name) if self.explore_mode and call.name else []

        result = call.execute(values=self.values, potential_values=potential_values, explore_mode=self.explore_mode)

        if parameter_type in ("select", "multi_select"):
            value = call.stored_value.value
            is_reproducible = call.stored_value.reproducible
        else:
            value = result
            is_reproducible = True

        record = ParameterRecord(
            name=call.name,
            parameter_type=parameter_type,
            single_value=call.single_value,
            default=call.default,
            value=value,
            is_reproducible=is_reproducible,
            options=options,
            numeric_bounds=numeric_bounds,
            run_id=self.run_id,
            source=self.source,
        )
        self.run_history.add_record(record)
        return result

    def _get_potential_values(self, name: str) -> List[Any]:
        records_dict = self.run_history.get_param_records(name)
        # Extract just the records, ignoring the run_ids
        records = list(records_dict.values())
        potential_values = list(
            dict.fromkeys(  # remove duplicates
                record.value
                for record in reversed(records)  # LIFO
                if isinstance(record, ParameterRecord)
                and (
                    record.is_reproducible if isinstance(record.is_reproducible, bool) else all(record.is_reproducible)
                )
            )
        )[:MAX_POTENTIAL_VALUES]
        return potential_values
