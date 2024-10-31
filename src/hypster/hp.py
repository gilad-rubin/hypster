import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from .db import DatabaseInterface, NestedDBRecord, NumericOptions, ParameterRecord, ParameterSource
from .hp_calls import (
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
    OptionsType,
    PropagateCall,
    SelectCall,
    TextInputCall,
    ValidKeyType,
)

logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class HP:
    def __init__(
        self,
        final_vars: List[str],
        exclude_vars: List[str],
        values: Dict[str, Any],
        db: DatabaseInterface,
        run_id: str,
        explore_mode: bool = False,
    ):
        self.final_vars = final_vars
        self.exclude_vars = exclude_vars
        self.values = values
        self.db = db
        self.run_id = run_id
        self.explore_mode = explore_mode
        self.source = ParameterSource.UI if explore_mode else ParameterSource.USER
        logger.info(f"Initialized HP with explore_mode: {explore_mode}")

    def select(
        self,
        options: OptionsType,
        *,
        name: Optional[str] = None,
        default: Optional[ValidKeyType] = None,
        options_only: bool = False,
    ) -> Any:
        select_call = SelectCall(name=name, options=options, default=default, options_only=options_only)

        potential_values = []
        if self.explore_mode:
            potential_values = self.db.get_potential_values(name)

        result = select_call.execute(
            values=self.values, potential_values=potential_values, explore_mode=self.explore_mode
        )

        self._record_parameter(
            name=name,
            parameter_type="select",
            default=default,
            value=result,
            options=list(select_call.processed_options.keys()),
        )
        return result

    def _record_parameter(
        self,
        name: str,
        parameter_type: str,
        default: Any,
        value: Any,
        options: Optional[List[Any]] = None,
        numeric_options: Optional[NumericOptions] = None,
    ) -> None:
        """Record parameter to database."""
        record = ParameterRecord(
            name=name,
            parameter_type=parameter_type,
            default=default,
            value=value,
            options=options,
            numeric_options=numeric_options,
            run_id=self.run_id,
            source=self.source,
        )
        self.db.add_record(record, run_id=self.run_id)

    def multi_select(
        self,
        options: OptionsType,
        *,
        name: Optional[str] = None,
        default: Optional[List[ValidKeyType]] = None,
        options_only: bool = False,
    ) -> List[Any]:
        multi_select_call = MultiSelectCall(name=name, options=options, default=default, options_only=options_only)
        logger.debug(f"Added MultiSelectCall: {name}")
        potential_values = []
        if self.explore_mode:
            potential_values = self.db.get_potential_values(name)

        result = multi_select_call.execute(
            values=self.values, potential_values=potential_values, explore_mode=self.explore_mode
        )
        options_keys = list(multi_select_call.processed_options.keys())

        record = ParameterRecord(
            name=name,
            parameter_type="multi_select",
            default=default,
            value=result,
            options=options_keys,
            run_id=self.run_id,
            source=self.source,
        )
        self.db.add_record(record, run_id=self.run_id)
        return result

    def text_input(self, default: str, *, name: Optional[str] = None) -> str:
        text_input_call = TextInputCall(name=name, default=default)
        logger.debug(f"Added TextInputCall: {name}")
        result = text_input_call.execute(self.values, self.db, self.explore_mode)

        record = ParameterRecord(
            name=name, parameter_type="text", default=default, value=result, run_id=self.run_id, source=self.source
        )
        self.db.add_record(record, run_id=self.run_id)
        return result

    def multi_text(self, default: List[str] = [], *, name: Optional[str] = None) -> List[str]:
        multi_text_call = MultiTextCall(name=name, default=default)
        logger.debug(f"Added MultiTextCall: {name}")
        result = multi_text_call.execute(self.values, self.db, self.explore_mode)

        record = ParameterRecord(
            name=name,
            parameter_type="multi_text",
            default=default,
            value=result,
            run_id=self.run_id,
            source=self.source,
        )
        self.db.add_record(record, run_id=self.run_id)
        return result

    def number_input(
        self,
        default: NumericType,
        *,
        name: Optional[str] = None,
        min: Optional[NumericType] = None,
        max: Optional[NumericType] = None,
    ) -> NumericType:
        bounds = NumericBounds(min_val=min, max_val=max) if (min is not None or max is not None) else None
        number_input_call = NumberInputCall(name=name, default=default, bounds=bounds)
        logger.debug(f"Added NumberInputCall: {name}")

        potential_values = []
        if self.explore_mode:
            potential_values = self.db.get_potential_values(name)

        result = number_input_call.execute(
            values=self.values, potential_values=potential_values, explore_mode=self.explore_mode
        )

        self._record_parameter(
            name=name,
            parameter_type="number",
            default=default,
            value=result,
            numeric_options=NumericOptions(min=min, max=max),
        )
        return result

    def multi_number(
        self,
        default: List[NumericType] = [],
        *,
        name: Optional[str] = None,
        min: Optional[NumericType] = None,
        max: Optional[NumericType] = None,
    ) -> List[NumericType]:
        bounds = NumericBounds(min_val=min, max_val=max) if (min is not None or max is not None) else None
        multi_number_call = MultiNumberCall(name=name, default=default, bounds=bounds)
        logger.debug(f"Added MultiNumberCall: {name}")

        potential_values = []
        if self.explore_mode:
            potential_values = self.db.get_potential_values(name)

        result = multi_number_call.execute(
            values=self.values, potential_values=potential_values, explore_mode=self.explore_mode
        )

        self._record_parameter(
            name=name,
            parameter_type="multi_number",
            default=default,
            value=result,
            numeric_options=NumericOptions(min=min, max=max),
        )
        return result

    def int_input(
        self,
        default: int,
        *,
        name: Optional[str] = None,
        min: Optional[int] = None,
        max: Optional[int] = None,
    ) -> int:
        bounds = NumericBounds(min_val=min, max_val=max) if (min is not None or max is not None) else None
        int_input_call = IntInputCall(name=name, default=default, bounds=bounds)
        logger.debug(f"Added IntInputCall: {name}")

        potential_values = []
        if self.explore_mode:
            potential_values = self.db.get_potential_values(name)

        result = int_input_call.execute(
            values=self.values, potential_values=potential_values, explore_mode=self.explore_mode
        )

        self._record_parameter(
            name=name,
            parameter_type="int",
            default=default,
            value=result,
            numeric_options=NumericOptions(min=min, max=max, allow_float=False),
        )
        return result

    def multi_int(
        self,
        default: List[int] = [],
        *,
        name: Optional[str] = None,
        min: Optional[int] = None,
        max: Optional[int] = None,
    ) -> List[int]:
        bounds = NumericBounds(min_val=min, max_val=max) if (min is not None or max is not None) else None
        multi_int_call = MultiIntCall(name=name, default=default, bounds=bounds)
        logger.debug(f"Added MultiIntCall: {name}")

        potential_values = []
        if self.explore_mode:
            potential_values = self.db.get_potential_values(name)

        result = multi_int_call.execute(
            values=self.values, potential_values=potential_values, explore_mode=self.explore_mode
        )

        self._record_parameter(
            name=name,
            parameter_type="multi_int",
            default=default,
            value=result,
            numeric_options=NumericOptions(min=min, max=max, allow_float=False),
        )
        return result

    def bool_input(self, default: bool, *, name: Optional[str] = None) -> bool:
        bool_input_call = BoolInputCall(name=name, default=default)
        logger.debug(f"Added BoolInputCall: {name}")
        result = bool_input_call.execute(self.values, self.db, self.explore_mode)

        record = ParameterRecord(
            name=name, parameter_type="bool", default=default, value=result, run_id=self.run_id, source=self.source
        )
        self.db.add_record(record, run_id=self.run_id)
        return result

    def multi_bool(self, default: List[bool] = [], *, name: Optional[str] = None) -> List[bool]:
        multi_bool_call = MultiBoolCall(name=name, default=default)
        logger.debug(f"Added MultiBoolCall: {name}")
        result = multi_bool_call.execute(self.values, self.db, self.explore_mode)

        record = ParameterRecord(
            name=name,
            parameter_type="multi_bool",
            default=default,
            value=result,
            run_id=self.run_id,
            source=self.source,
        )
        self.db.add_record(record, run_id=self.run_id)
        return result

    def propagate(
        self,
        config_func: Union[str, Path, Callable],
        *,
        name: Optional[str] = None,
        final_vars: List[str] = [],
        exclude_vars: List[str] = [],
        values: Dict[str, Any] = {},
    ) -> Dict[str, Any]:
        propagate_call = PropagateCall(name=name)
        logger.debug(f"Added PropagateCall: {name}")
        if isinstance(config_func, (str, Path)):
            from .core import load

            config_func = load(str(config_func))
        result = propagate_call.execute(
            config_func,
            final_vars=final_vars,
            original_final_vars=self.final_vars,
            exclude_vars=exclude_vars,
            original_exclude_vars=self.exclude_vars,
            values=values,
            original_values=self.values,
        )
        record = NestedDBRecord(
            name=name, parameter_type="propagate", db=config_func.db, run_id=self.run_id, source=self.source
        )
        self.db.add_record(record, run_id=self.run_id)
        return result
