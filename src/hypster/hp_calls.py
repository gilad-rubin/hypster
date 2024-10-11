import collections
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
                result = []
                for key in self.default:
                    if self.options:
                        if key in self.options:
                            result.append(self.options[key])
                        else:  # TODO: fix
                            raise ValueError("Invalid default not in options.")
                    else:
                        result.append(key)
                snapshot = self.default
            else:
                result = self.options[self.default] if self.options else self.default
                snapshot = self.default
            self.hp.snapshot[self.name] = snapshot
        else:
            raise ValueError(f"`{self.name}` has no selections, overrides or defaults provided.")

        return result

    def _get_result_from_override(self):
        override_value = self.hp.overrides[self.name]
        logger.debug("Found override for %s: %s", self.name, override_value)
        snapshot = None
        if isinstance(override_value, list):
            result = []
            snapshot = []
            for value in override_value:
                snapshot.append(value)
                if self.options and value in self.options:
                    result.append(self.options[value])
                else:
                    result.append(value)
        elif self.options and override_value in self.options:
            result = self.options[override_value]
        else:
            result = override_value

        if snapshot is None:
            snapshot = override_value
        self.hp.snapshot[self.name] = snapshot
        logger.info("Applied override for %s: %s", self.name, result)
        return result

    def _get_result_from_selection(self):
        # TODO: add error if there's a selection for something that only works with overrides,
        # like text, number input (including multi)
        selected_value = self.hp.selections[self.name]
        logger.debug("Found selection for %s: %s", self.name, selected_value)
        snapshot = None
        if isinstance(selected_value, list):
            result = []
            snapshot = []
            for key in selected_value:
                if self.options and key in self.options:
                    result.append(self.options[key])
                    snapshot.append(key)
                else:
                    raise ValueError(
                        f"Invalid selection '{key}' for '{self.name}'. Not in options: "
                        f"{list(self.options.keys() if self.options else [])}"
                    )
        elif self.options and selected_value in self.options:
            result = self.options[selected_value]
            logger.info("Applied selection for %s: %s", self.name, result)
        else:
            raise ValueError(
                f"Invalid selection '{selected_value}' for '{self.name}'. "
                f"Not in options: {list(self.options.keys() if self.options else [])}"
            )

        if snapshot is None:  # not a list
            snapshot = selected_value
        if self.name not in self.hp.snapshot:
            self.hp.snapshot[self.name] = snapshot
        return result

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

        return self.options[selected_key]


class MultiSelectCall(HPCall):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.all_combinations = None

    def execute(self, options: Union[Dict[str, Any], List[Any]]) -> List[Any]:
        processed_options = self.validate_and_process_options(options)
        self.options = processed_options

        if self.hp.explore_mode:
            return self.explore()

        result = self.handle_overrides_selections()

        logger.info(f"MultiSelect call executed for {self.name}: {result}")
        return result

    def explore(self) -> List[Any]:
        if self.name in self.hp.current_combination:
            chosen_keys = self.hp.current_combination[self.name]
        else:
            combinations = self._generate_all_combinations()
            self.hp.current_space[self.name] = combinations
            chosen_keys = combinations[0]
            self.hp.current_combination[self.name] = chosen_keys

        result = []
        for key in chosen_keys:
            result.append(self.options[key])
        return result

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


class IntInputCall(HPCall):
    def execute(self, options: None = None) -> int:
        result = self.handle_overrides_selections()
        logger.info(f"IntInput call executed for {self.name}: {result}")
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

    def execute(
        self,
        config_func: Callable,
        selections: Optional[Dict[str, Any]] = None,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if self.hp.explore_mode:
            return self.explore(config_func)

        original_name_prefix = self.hp.name_prefix
        self.hp.name_prefix = self.name if original_name_prefix is None else f"{original_name_prefix}.{self.name}"
        nested_config = self._prepare_nested_config()

        result = self._run_nested_config(config_func, nested_config)
        nested_snapshot = config_func.get_last_snapshot()
        self.hp.snapshot[self.name] = nested_snapshot
        self.hp.name_prefix = original_name_prefix
        return result

    def explore(self, config_func: Callable) -> Dict[str, Any]:
        if self.name in self.hp.current_combination:
            chosen_combination = self.hp.current_combination[self.name]
        else:
            combinations = config_func.get_combinations()
            defaults = config_func.get_defaults()
            self.hp.defaults[self.name] = defaults
            self.hp.current_space[self.name] = combinations
            chosen_combination = combinations[0]
            self.hp.current_combination[self.name] = chosen_combination

        nested_config = {}
        nested_config["selections"] = chosen_combination.copy()
        nested_config["final_vars"] = []
        nested_config["overrides"] = {}
        result = self._run_nested_config(config_func, nested_config)
        return result

    def _prepare_nested_config(self) -> Dict[str, Any]:
        return {
            "selections": self.extract_nested_config(self.hp.selections, self.name),
            "overrides": self.extract_nested_config(self.hp.overrides, self.name),
            "final_vars": self.process_final_vars(self.hp.final_vars, self.name),
        }

    def _run_nested_config(self, config_func: Callable, nested_config: Dict[str, Any]) -> Dict[str, Any]:
        logger.debug(f"Running nested configuration with: {nested_config}")
        return config_func(
            final_vars=nested_config["final_vars"],
            selections=nested_config["selections"],
            overrides=nested_config["overrides"],
        )

    @staticmethod
    def extract_nested_config(config: Dict[str, Any], prefix: str) -> Dict[str, Any]:
        result = collections.defaultdict(dict)
        prefix_dot = f"{prefix}."

        def process_item(key: str, value: Any, target: Dict[str, Any]):
            if key.startswith(prefix_dot):
                nested_key = key[len(prefix_dot) :]
                target[nested_key] = value
            elif key == prefix and isinstance(value, dict):
                target.update(value)

        for key, value in config.items():
            process_item(key, value, result)

        # Check for discrepancies
        flat_result = {k: v for k, v in PropagateCall._flatten_dict(result).items()}
        flat_dot_notation = {k[len(prefix_dot) :]: v for k, v in config.items() if k.startswith(prefix_dot)}

        # Check for keys with different values
        discrepancies = {
            k: (flat_result[k], flat_dot_notation[k])
            for k in set(flat_result) & set(flat_dot_notation)
            if flat_result[k] != flat_dot_notation[k]
        }

        if discrepancies:
            raise ValueError(f"Discrepancies found in nested configuration for '{prefix}': {discrepancies}")

        return dict(result)

    @staticmethod
    def _flatten_dict(d: Dict[str, Any], parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(PropagateCall._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    @staticmethod
    def process_final_vars(final_vars: List[str], name: str) -> List[str]:
        dot_notation_vars = [var[len(name) + 1 :] for var in final_vars if var.startswith(f"{name}.")]
        dict_style_vars = list(PropagateCall.extract_nested_config(dict.fromkeys(final_vars, None), name).keys())
        dict_style_vars = [var for var in dict_style_vars if var not in dot_notation_vars]  # remove dups
        return dot_notation_vars + dict_style_vars
