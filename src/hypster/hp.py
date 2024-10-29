import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from .db import DatabaseInterface, NestedDBRecord, NumericOptions, ParameterRecord
from .hp_calls import (
    BoolInputCall,
    IntInputCall,
    MultiBoolCall,
    MultiIntCall,
    MultiNumberCall,
    MultiSelectCall,
    MultiTextCall,
    NumberInputCall,
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
    ):
        self.final_vars = final_vars
        self.exclude_vars = exclude_vars
        self.values = values
        self.db = db
        logger.info("Initialized HP")

    def select(
        self,
        options: OptionsType,
        *,
        name: Optional[str] = None,
        default: Optional[ValidKeyType] = None,
        options_only: bool = False,
    ) -> Any:
        select_call = SelectCall(name=name, options=options, default=default, options_only=options_only)
        logger.debug(f"Added SelectCall: {name}")

        result = select_call.execute(self.values)
        options_keys = list(select_call.options.keys())

        record = ParameterRecord(
            name=name,
            parameter_type="select",
            default=default,
            value=result,
            options=options_keys,
        )
        self.db.add_record(record)
        return result

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
        result = multi_select_call.execute(self.values)
        options_keys = list(multi_select_call.options.keys())

        record = ParameterRecord(
            name=name, parameter_type="multi_select", default=default, value=result, options=options_keys
        )
        self.db.add_record(record)
        return result

    def text_input(self, default: str, *, name: Optional[str] = None) -> str:
        text_input_call = TextInputCall(name=name, default=default)
        logger.debug(f"Added TextInputCall: {name}")
        result = text_input_call.execute(self.values)

        record = ParameterRecord(name=name, parameter_type="text", default=default, value=result)
        self.db.add_record(record)
        return result

    def multi_text(self, default: List[str] = [], *, name: Optional[str] = None) -> List[str]:
        multi_text_call = MultiTextCall(name=name, default=default)
        logger.debug(f"Added MultiTextCall: {name}")
        result = multi_text_call.execute(self.values)

        record = ParameterRecord(name=name, parameter_type="multi_text", default=default, value=result)
        self.db.add_record(record)
        return result

    def number_input(
        self,
        default: Union[int, float],
        *,
        name: Optional[str] = None,
        min: Optional[Union[int, float]] = None,
        max: Optional[Union[int, float]] = None,
    ) -> Union[int, float]:
        number_input_call = NumberInputCall(name=name, default=default, min=min, max=max)
        logger.debug(f"Added NumberInputCall: {name}")
        result = number_input_call.execute(self.values)

        record = ParameterRecord(
            name=name,
            parameter_type="number",
            default=default,
            value=result,
            options=NumericOptions(min=min, max=max),
        )
        self.db.add_record(record)
        return result

    def multi_number(
        self,
        default: List[Union[int, float]] = [],
        *,
        name: Optional[str] = None,
        min: Optional[Union[int, float]] = None,
        max: Optional[Union[int, float]] = None,
    ) -> List[Union[int, float]]:
        multi_number_call = MultiNumberCall(name=name, default=default, min=min, max=max)
        logger.debug(f"Added MultiNumberCall: {name}")
        result = multi_number_call.execute(self.values)

        record = ParameterRecord(
            name=name,
            parameter_type="multi_number",
            default=default,
            value=result,
            options=NumericOptions(min=min, max=max),
        )
        self.db.add_record(record)
        return result

    def int_input(
        self,
        default: int,
        *,
        name: Optional[str] = None,
        min: Optional[int] = None,
        max: Optional[int] = None,
    ) -> int:
        int_input_call = IntInputCall(name=name, default=default, min=min, max=max)
        logger.debug(f"Added IntInputCall: {name}")
        result = int_input_call.execute(self.values)

        record = ParameterRecord(
            name=name,
            parameter_type="int",
            default=default,
            value=result,
            options=NumericOptions(min=min, max=max, allow_float=False),
        )
        self.db.add_record(record)
        return result

    def multi_int(
        self,
        default: List[int] = [],
        *,
        name: Optional[str] = None,
        min: Optional[int] = None,
        max: Optional[int] = None,
    ) -> List[int]:
        multi_int_call = MultiIntCall(name=name, default=default, min=min, max=max)
        logger.debug(f"Added MultiIntCall: {name}")
        result = multi_int_call.execute(self.values)

        record = ParameterRecord(
            name=name,
            parameter_type="multi_int",
            default=default,
            value=result,
            options=NumericOptions(min=min, max=max, allow_float=False),
        )
        self.db.add_record(record)
        return result

    def bool_input(self, default: bool, *, name: Optional[str] = None) -> bool:
        bool_input_call = BoolInputCall(name=name, default=default)
        logger.debug(f"Added BoolInputCall: {name}")
        result = bool_input_call.execute(self.values)

        record = ParameterRecord(name=name, parameter_type="bool", default=default, value=result)
        self.db.add_record(record)
        return result

    def multi_bool(self, default: List[bool] = [], *, name: Optional[str] = None) -> List[bool]:
        multi_bool_call = MultiBoolCall(name=name, default=default)
        logger.debug(f"Added MultiBoolCall: {name}")
        result = multi_bool_call.execute(self.values)

        record = ParameterRecord(name=name, parameter_type="multi_bool", default=default, value=result)
        self.db.add_record(record)
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
        record = NestedDBRecord(config_func.db)
        self.db.add_record(record)
        return result
