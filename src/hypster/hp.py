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
    NestedCall,
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
        final_vars: List[str],
        exclude_vars: List[str],
        values: Dict[str, Any],
        run_history: HistoryDatabase,
        run_id: UUID,
        explore_mode: bool = False,
    ):
        self.final_vars = final_vars
        self.exclude_vars = exclude_vars
        self.values = values
        self.run_history = run_history
        self.run_id = run_id
        self.explore_mode = explore_mode
        self.source = ParameterSource.UI if explore_mode else ParameterSource.USER
        logger.info(f"Initialized HP with explore_mode: {explore_mode}")

    def select(
        self,
        options: Union[Dict[BasicType, Any], List[BasicType]],
        *,
        name: str,
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
        name: str,
        default: Optional[List[BasicType]] = None,
        options_only: bool = False,
    ) -> List[Any]:
        if default is None:
            default = []
        call = MultiSelectCall(name=name, options=options, default=default, options_only=options_only)
        options_keys = list(call.processed_options.keys())
        return self._execute_call(call=call, parameter_type="multi_select", options=options_keys)

    def number(
        self,
        default: NumericType,
        *,
        name: str,
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
        name: str,
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
        name: str,
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
        name: str,
        min: Optional[int] = None,
        max: Optional[int] = None,
    ) -> List[int]:
        bounds = NumericBounds(min_val=min, max_val=max) if (min is not None or max is not None) else None
        call = MultiIntCall(name=name, default=default, bounds=bounds)
        return self._execute_call(call=call, parameter_type="multi_int", numeric_bounds=bounds)

    def text(self, default: str, *, name: str) -> str:
        call = TextInputCall(name=name, default=default)
        return self._execute_call(call=call, parameter_type="text")

    def multi_text(self, default: List[str] = [], *, name: str) -> List[str]:
        call = MultiTextCall(name=name, default=default)
        return self._execute_call(call=call, parameter_type="multi_text")

    def bool(self, default: bool, *, name: str) -> bool:
        call = BoolInputCall(name=name, default=default)
        return self._execute_call(call=call, parameter_type="bool")

    def multi_bool(self, default: List[bool] = [], *, name: str) -> List[bool]:
        call = MultiBoolCall(name=name, default=default)
        return self._execute_call(call=call, parameter_type="multi_bool")

    def nest(
        self,
        config_func: Union[str, Path, "Hypster"],
        *,
        name: str,
        final_vars: List[str] = [],
        exclude_vars: List[str] = [],
        values: Dict[str, Any] = {},
    ) -> Dict[str, Any]:
        if isinstance(config_func, (str, Path)):
            config_func = self._resolve_config_target(str(config_func))

        call = NestedCall(name=name)
        result = call.execute(
            config_func,
            final_vars=final_vars,
            original_final_vars=self.final_vars,
            exclude_vars=exclude_vars,
            original_exclude_vars=self.exclude_vars,
            values=values,
            original_values=self.values,
            explore_mode=self.explore_mode,
            run_history=self.run_history,
        )

        record = NestedHistoryRecord(
            name=name,
            parameter_type="nest",
            run_history=config_func.run_history,
            run_id=self.run_id,
            source=self.source,
            final_vars=call.processed_final_vars,
            exclude_vars=call.processed_exclude_vars,
        )
        self.run_history.add_record(record)
        return result

    def _resolve_config_target(self, target: str) -> "Hypster":
        """
        Resolve a string target to a Hypster configuration.

        Resolution order:
        1. Check registry for exact match
        2. If contains ":", treat as file/module path with specific object
        3. If ends with ".py", treat as file path
        4. Otherwise, attempt to load as module path

        Args:
            target: The target string to resolve

        Returns:
            Hypster instance

        Raises:
            ValueError: If target cannot be resolved
        """
        from .core import load
        from .registry import registry

        # 1. Check registry for exact match
        if registry.contains(target):
            return registry.get(target)

        # 2. Check if it's a path/module with specific object (contains ":")
        # 3. Check if it's a file path (ends with ".py")
        # 4. Otherwise try as module path
        try:
            return load(target)
        except (ValueError, ImportError, FileNotFoundError) as e:
            # Provide helpful error message with resolution attempts
            error_msg = f"Could not resolve configuration target '{target}'. "

            if registry.contains(target):
                error_msg += "Found in registry but failed to retrieve."
            else:
                available_keys = registry.list()
                if available_keys:
                    closest_matches = [key for key in available_keys if target.lower() in key.lower()]
                    if closest_matches:
                        error_msg += f"Not found in registry. Did you mean one of: {closest_matches}?"
                    else:
                        error_msg += (
                            f"Not found in registry. Available keys: "
                            f"{available_keys[:5]}{'...' if len(available_keys) > 5 else ''}"
                        )
                else:
                    error_msg += "Not found in registry (registry is empty)."

            error_msg += f" Also failed to load as file/module: {e}"
            raise ValueError(error_msg)

    def _execute_call(
        self,
        call: BaseHPCall,
        parameter_type: str,
        options: Optional[List[BasicType]] = None,
        numeric_bounds: Optional[NumericBounds] = None,
    ) -> Any:
        """Execute HP call and record its result"""
        logger.debug(f"Added {parameter_type}Call: {call.name}")

        potential_values = self._get_potential_values(call.name) if self.explore_mode else []

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

    def get_collected_values(self) -> Dict[str, Any]:
        """Get all parameter values that were collected during execution"""
        from .run_history import NestedHistoryRecord, ParameterRecord

        result = {}
        latest_records = self.run_history.get_latest_run_records()

        for name, record in latest_records.items():
            if isinstance(record, ParameterRecord):
                # Apply final_vars and exclude_vars filtering for regular parameters
                if name in self.exclude_vars:
                    continue  # Skip excluded variables
                if not self.final_vars or name in self.final_vars:
                    result[name] = record.value
            elif isinstance(record, NestedHistoryRecord):
                # Get the collected values from the nested configuration
                # Use the final_vars and exclude_vars that were stored during execution
                nested_hp = HP(record.final_vars, record.exclude_vars, {}, record.run_history, record.run_id, False)
                nested_result = nested_hp.get_collected_values()
                result[name] = nested_result
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
