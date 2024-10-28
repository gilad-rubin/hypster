import logging
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Union

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
    SelectCall,
    TextInputCall,
    ValidKeyType,
)

logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class HP:
    def __init__(self, final_vars: List[str], selections: Dict[str, Any], overrides: Dict[str, Any]):
        self.final_vars = final_vars
        self.selections = selections
        self.overrides = overrides
        self.name_prefix = None
        self.current_space: OrderedDict[str, Any] = OrderedDict()
        self.current_combination: OrderedDict[str, Any] = OrderedDict()
        self.defaults: Dict[str, Any] = {}
        self.snapshot: Dict[str, Any] = {}
        logger.info("Initialized HP")

    def select(
        self,
        options: OptionsType,
        *,
        name: Optional[str] = None,
        default: Optional[ValidKeyType] = None,
        disable_overrides: bool = False,
    ) -> Any:
        select_call = SelectCall(name=name, options=options, default=default, disable_overrides=disable_overrides)
        logger.debug(f"Added SelectCall: {name}")
        return select_call.execute(self.selections, self.overrides)

    def multi_select(
        self,
        options: OptionsType,
        *,
        name: Optional[str] = None,
        default: Optional[List[ValidKeyType]] = None,
        disable_overrides: bool = False,
    ) -> List[Any]:
        multi_select_call = MultiSelectCall(
            name=name, options=options, default=default, disable_overrides=disable_overrides
        )
        logger.debug(f"Added MultiSelectCall: {name}")
        return multi_select_call.execute(self.selections, self.overrides)

    def text_input(self, default: str, *, name: Optional[str] = None) -> str:
        text_input_call = TextInputCall(name=name, default=default)
        logger.debug(f"Added TextInputCall: {name}")
        return text_input_call.execute(self.overrides)

    def multi_text(self, default: List[str] = [], *, name: Optional[str] = None) -> List[str]:
        multi_text_call = MultiTextCall(name=name, default=default)
        logger.debug(f"Added MultiTextCall: {name}")
        return multi_text_call.execute(self.overrides)

    def number_input(self, default: Union[int, float], *, name: Optional[str] = None) -> Union[int, float]:
        number_input_call = NumberInputCall(name=name, default=default)
        logger.debug(f"Added NumberInputCall: {name}")
        return number_input_call.execute(self.overrides)

    def multi_number(
        self, default: List[Union[int, float]] = [], *, name: Optional[str] = None
    ) -> List[Union[int, float]]:
        multi_number_call = MultiNumberCall(name=name, default=default)
        logger.debug(f"Added MultiNumberCall: {name}")
        return multi_number_call.execute(self.overrides)

    def int_input(self, default: int, *, name: Optional[str] = None) -> int:
        int_input_call = IntInputCall(name=name, default=default)
        logger.debug(f"Added IntInputCall: {name}")
        return int_input_call.execute(self.overrides)

    def multi_int(self, default: List[int] = [], *, name: Optional[str] = None) -> List[int]:
        multi_int_call = MultiIntCall(name=name, default=default)
        logger.debug(f"Added MultiIntCall: {name}")
        return multi_int_call.execute(self.overrides)

    def bool_input(self, default: bool, *, name: Optional[str] = None) -> bool:
        bool_input_call = BoolInputCall(name=name, default=default)
        logger.debug(f"Added BoolInputCall: {name}")
        return bool_input_call.execute(self.overrides)

    def multi_bool(self, default: List[bool] = [], *, name: Optional[str] = None) -> List[bool]:
        multi_bool_call = MultiBoolCall(name=name, default=default)
        logger.debug(f"Added MultiBoolCall: {name}")
        return multi_bool_call.execute(self.overrides)
