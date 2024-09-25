import logging
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from .hp_calls import (
    MultiNumberCall,
    MultiSelectCall,
    MultiTextCall,  # New imports
    NumberInputCall,
    PropagateCall,
    SelectCall,
    TextInputCall,
)

logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class HP:
    def __init__(
        self, final_vars: List[str], selections: Dict[str, Any], overrides: Dict[str, Any], explore_mode: bool = False
    ):
        self.final_vars = final_vars
        self.selections = selections
        self.overrides = overrides
        self.name_prefix = None
        self.explore_mode = explore_mode
        self.current_space = OrderedDict()  # TODO: consider turning these 3 items to a dataclass
        self.current_combination = OrderedDict()
        self.defaults = {}
        self.snapshot = {}
        logger.info(f"Initialized HP with explore_mode: {explore_mode}")

    def select(self, options: Union[Dict[str, Any], List[Any]], *, name: Optional[str] = None, default: Any = None):
        select_call = SelectCall(self, name=name, default=default)
        logger.debug(f"Added SelectCall: {name}")
        return select_call.execute(options)

    # TODO: add "min_items" to multi_select
    def multi_select(
        self, options: Union[Dict[str, Any], List[Any]], *, name: Optional[str] = None, default: Any = None
    ) -> List[Any]:
        multi_select_call = MultiSelectCall(self, name=name, default=default)
        logger.debug(f"Added MultiSelectCall: {name}")
        return multi_select_call.execute(options)

    def text_input(self, default: str, *, name: Optional[str] = None) -> str:
        text_input_call = TextInputCall(self, name=name, default=default)
        logger.debug(f"Added TextInputCall: {name}")
        return text_input_call.execute()

    def number_input(self, default: Union[int, float], *, name: Optional[str] = None) -> Union[int, float]:
        number_input_call = NumberInputCall(self, name=name, default=default)
        logger.debug(f"Added NumberInputCall: {name}")
        return number_input_call.execute()

    def multi_text(self, default: List[str] = [], *, name: Optional[str] = None) -> List[str]:
        multi_text_call = MultiTextCall(self, name=name, default=default)
        logger.debug(f"Added MultiTextCall: {name}")
        return multi_text_call.execute()

    def multi_number(
        self, default: List[Union[int, float]] = [], *, name: Optional[str] = None
    ) -> List[Union[int, float]]:
        multi_number_call = MultiNumberCall(self, name=name, default=default)
        logger.debug(f"Added MultiNumberCall: {name}")
        return multi_number_call.execute()

    def propagate(
        self, config_func: Callable, *, name: Optional[str] = None
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        propagate_call = PropagateCall(self, name=name)
        logger.debug(f"Added PropagateCall: {name}")
        return propagate_call.execute(config_func)

    def get_next_combination(self, explored_combinations: List[Dict[str, Any]]) -> bool:
        for param in reversed(self.current_combination.keys()):
            options = self.current_space[param]
            current_value = self.current_combination[param]

            next_option = self._get_next_option(param, options, current_value, explored_combinations)
            if next_option is not None:
                self.current_combination[param] = next_option
                self._reset_subsequent_params(param)
                return True

        return False

    def _flatten_combination(self, combination: OrderedDict) -> OrderedDict:
        flattened = OrderedDict()
        for key, value in combination.items():
            if isinstance(value, dict):
                for sub_key, sub_value in self._flatten_combination(value).items():
                    flattened[f"{key}.{sub_key}"] = sub_value
            else:
                flattened[key] = value
        return flattened

    def _get_next_option(
        self, param: str, options: List[Any], current_value: Any, explored_combinations: List[Dict[str, Any]]
    ) -> Optional[Any]:
        param_index = list(self.current_combination.keys()).index(param)
        previous_selections = {
            k: v
            for k, v in self.current_combination.items()
            if list(self.current_combination.keys()).index(k) < param_index
        }
        used_options = set(
            _hashable_value(comb[param])
            for comb in explored_combinations
            if all(_values_equal(comb.get(k), v) for k, v in previous_selections.items())
        )

        for option in options:
            if _hashable_value(option) not in used_options:
                return option

        return None

    def _reset_subsequent_params(self, param: str):
        reset = False
        for name in list(self.current_combination.keys()):
            if reset:
                self.current_combination.pop(name)
            if name == param:
                reset = True


class InvalidSelectionError(Exception):
    pass


def _hashable_value(value: Any) -> Tuple:
    if isinstance(value, dict):
        return tuple(sorted((k, _hashable_value(v)) for k, v in value.items()))
    elif isinstance(value, list):
        return tuple(_hashable_value(item) for item in value)
    return value


def _values_equal(a: Any, b: Any) -> bool:
    if isinstance(a, dict) and isinstance(b, dict):
        return all(_values_equal(a.get(k), b.get(k)) for k in set(a.keys()) | set(b.keys()))
    elif isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_values_equal(x, y) for x, y in zip(a, b))
    return a == b
