import itertools
import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

if TYPE_CHECKING:
    from .hp import HP  # Adjust the import path as needed

logger = logging.getLogger(__name__)


class HPCall(ABC):
    def __init__(self, hp_instance: "HP", name: Optional[str] = None, default: Any = None):
        if name is None:
            raise ValueError("`name` argument is missing and must be provided explicitly.")
        self.hp = hp_instance
        self.name = self._get_full_name(name)
        self.default = default
        # TODO: improve this
        if default is not None:
            if self.name not in self.hp.defaults:
                self.hp.defaults[self.name] = [default]
            elif default not in self.hp.defaults[self.name]:
                self.hp.defaults[self.name].append(default)
        self.options = None

    def _get_full_name(self, name: str) -> str:
        if self.hp.name_prefix:
            return f"{self.hp.name_prefix}.{name}"
        return name

    def validate_and_process_options(
        self, options: Optional[Union[Dict[str, Any], List[Any]]]
    ) -> Optional[Dict[Any, Any]]:
        if options is not None:
            self._check_options_exists(options)
            if isinstance(options, dict):
                self._validate_dict_keys(options)
            elif isinstance(options, list):
                self._validate_list_values(options)
                options = {v: v for v in options}
            else:
                raise ValueError("Options must be a dictionary or a list.")

            if self.default is not None:
                self._validate_default(self.default, options)
            self.options = options
            return options
        return None

    def _check_options_exists(self, options: Union[Dict[str, Any], List[Any]]):
        if not isinstance(options, (list, dict)) or len(options) == 0:
            raise ValueError("Options must be a non-empty list or dictionary.")

    def _validate_dict_keys(self, options: Dict[str, Any]):
        if not all(isinstance(k, (str, int, bool, float)) for k in options.keys()):
            bad_keys = [key for key in options.keys() if not isinstance(key, (str, int, bool, float))]
            raise ValueError(f"Dictionary keys must be str, int, bool, float. Got {bad_keys} instead.")

    def _validate_list_values(self, options: List[Any]):
        if not all(isinstance(v, (str, int, bool, float)) for v in options):
            raise ValueError(
                "List values must be one of: str, int, bool, float. For complex types - use a dictionary instead"
            )

    def _validate_default(self, default: Any, options: Dict[str, Any]):
        if default is not None and isinstance(default, list):
            for key in default:
                if key not in options:
                    raise ValueError("Default values must be one of the options.")
        elif default is not None and default not in options:  # TODO: add var/param names to error
            raise ValueError("Default value must be one of the options.")

    def handle_overrides_selections(self) -> Any:
        if self.name in self.hp.overrides:
            result = self._get_result_from_override()
        elif self.name in self.hp.selections:
            result = self._get_result_from_selection()
        elif self.default is not None:
            if isinstance(self.default, list):
                result = self.default
            else:
                # TODO: check why this fails tests when I drop that last if part
                result = self.options[self.default] if self.options else self.default
        else:
            raise ValueError(f"`{self.name}` has no selections, overrides or defaults provided.")

        return result

    def _get_result_from_override(self):
        override_value = self.hp.overrides[self.name]
        logger.debug("Found override for %s: %s", self.name, override_value)
        if isinstance(override_value, list):
            result = []
            for value in override_value:
                if self.options and value in self.options:
                    result.append(self.options[value])
                else:
                    result.append(value)
        elif self.options and override_value in self.options:
            result = self.options[override_value]
        else:
            result = override_value
        logger.info("Applied override for %s: %s", self.name, result)
        return result

    def _get_result_from_selection(self):
        # TODO: add error if there's a selection for something that only works with overrides,
        # like text, number input (including multi)
        selected_value = self.hp.selections[self.name]
        logger.debug("Found selection for %s: %s", self.name, selected_value)
        if isinstance(selected_value, list):
            result = []
            for key in selected_value:
                if self.options and key in self.options:
                    result.append(self.options[key])
                else:
                    raise ValueError(
                        f"Invalid selection '{key}' for '{self.name}'. Not in options: "
                        f"{list(self.options.keys() if self.options else [])}"
                    )
            return result
        elif self.options and selected_value in self.options:
            result = self.options[selected_value]
            logger.info("Applied selection for %s: %s", self.name, result)
            return result
        else:
            raise ValueError(
                f"Invalid selection '{selected_value}' for '{self.name}'. "
                f"Not in options: {list(self.options.keys() if self.options else [])}"
            )

    # TODO: handle different cases where options is a list, one item, or empty.

    @abstractmethod
    def explore(self) -> Any:
        pass

    @abstractmethod
    def explore(self) -> bool:
        pass


class SelectCall(HPCall):
    def execute(self, options: Optional[Union[Dict[str, Any], List[Any]]] = None) -> Any:
        self.options = self.validate_and_process_options(options)
        if self.hp.explore_mode:
            self.hp.current_space[self.name] = list(self.options.keys())
            return self.explore()
        return self.handle_overrides_selections()

    def explore(self) -> Any:
        if self.name in self.hp.current_combination:
            selected_key = self.hp.current_combination[self.name]
        else:
            selected_key = list(self.options.keys())[0]
            self.hp.current_combination[self.name] = selected_key

        return selected_key


class MultiSelectCall(HPCall):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.all_combinations = None

    def execute(self, options: Union[Dict[str, Any], List[Any]]) -> List[Any]:
        processed_options = self.validate_and_process_options(options)
        self.options = processed_options

        if self.hp.explore_mode:
            if self.name in self.hp.current_combination:
                chosen_keys = self.hp.current_combination[self.name]
                return chosen_keys

            combinations = self._generate_all_combinations()
            self.hp.current_space[self.name] = combinations
            return self.explore(combinations)

        result = self.handle_overrides_selections()

        logger.info(f"MultiSelect call executed for {self.name}: {result}")
        return result

    def explore(self, combinations) -> List[Any]:
        selected_combination = combinations[0]
        self.hp.current_combination[self.name] = selected_combination
        return selected_combination

    def _generate_all_combinations(self) -> List[List[str]]:
        keys = list(self.options.keys())
        all_combinations = [[]]  # Include empty list for none selected
        for r in range(1, len(keys) + 1):
            all_combinations.extend(itertools.combinations(keys, r))
        return [list(combo) for combo in all_combinations]


class TextInputCall(HPCall):
    def execute(self, options: None = None) -> str:
        result = self.handle_overrides_selections()
        logger.info(f"TextInput call executed for {self.name}: {result}")
        return result

    def explore(self) -> str:
        return self.default if self.default is not None else ""


class NumberInputCall(HPCall):
    def execute(self, options: None = None) -> Union[int, float]:
        result = self.handle_overrides_selections()
        logger.info(f"NumberInput call executed for {self.name}: {result}")
        return result

    def explore(self) -> Union[int, float]:
        # TODO: handle no defaults in combinations
        return self.default


class MultiTextCall(HPCall):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_values = self.default or []

    def execute(self, options: None = None) -> List[str]:
        result = self.handle_overrides_selections()

        logger.info(f"MultiText call executed for {self.name}: {result}")
        return result

    def explore(self) -> List[str]:
        pass


class MultiNumberCall(HPCall):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_values = self.default or []

    def execute(self, options: None = None) -> List[Union[int, float]]:
        result = self.handle_overrides_selections()

        logger.info(f"MultiNumber call executed for {self.name}: {result}")
        return result

    def explore(self) -> List[Union[int, float]]:
        pass


class PropagateCall(HPCall):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def execute(self, config_func: Callable) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        if self.hp.explore_mode:
            # TODO: make sure that this condition is valid and this doesn't need to be recalculated every time
            if self.name in self.hp.current_combination:
                selected_dct = self.hp.current_combination[self.name]
                # TODO: add default statuses
                return selected_dct

            combinations = config_func.get_combinations()
            defaults = config_func.get_defaults()
            self.hp.defaults[self.name] = defaults
            self.hp.current_space[self.name] = combinations
            return self.explore(combinations)

        original_name_prefix = self.hp.name_prefix
        self.hp.name_prefix = self.name if original_name_prefix is None else f"{original_name_prefix}.{self.name}"
        nested_config = self._prepare_nested_config()
        result = self._run_nested_config(config_func, nested_config)

        self.hp.name_prefix = original_name_prefix
        return result

    def explore(self, combinations: List[Dict[str, Any]]) -> Dict[str, Any]:
        chosen_combination = combinations[0]
        self.hp.current_combination[self.name] = chosen_combination
        return chosen_combination

    def get_current_combination(self) -> Dict[str, Any]:
        return {f"{self.name}.{k}": v for k, v in self.combinations[self.current_index].items()}

    def _add_prefix_to_keys(self, combinations: List[Dict[str, Any]]):
        return [{f"{self.name}.{k}": v for k, v in combination.items()} for combination in combinations]

    def _prepare_nested_config(self) -> Dict[str, Any]:
        return {
            "selections": {
                k[len(self.name) + 1 :]: v for k, v in self.hp.selections.items() if k.startswith(f"{self.name}.")
            },
            "overrides": {
                k[len(self.name) + 1 :]: v for k, v in self.hp.overrides.items() if k.startswith(f"{self.name}.")
            },
            "final_vars": [var[len(self.name) + 1 :] for var in self.hp.final_vars if var.startswith(f"{self.name}.")],
        }

    def _run_nested_config(self, config_func: Callable, nested_config: Dict[str, Any]) -> Dict[str, Any]:
        logger.debug(f"Running nested configuration with: {nested_config}")
        return config_func(
            final_vars=nested_config["final_vars"],
            selections=nested_config["selections"],
            overrides=nested_config["overrides"],
        )
