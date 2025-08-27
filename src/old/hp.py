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

    def nest(
        self,
        config_func: Union[str, Path, "Hypster"],
        *,
        name: Optional[str] = None,
        final_vars: List[str] = [],
        exclude_vars: List[str] = [],
        values: Dict[str, Any] = {},
    ) -> Dict[str, Any]:
        if isinstance(config_func, (str, Path)):
            from .core import load

            config_func = load(str(config_func))

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
