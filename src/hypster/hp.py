#from .logging_utils import configure_logging
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# Correct logging configuration
logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class PropagatedConfig:
    def __init__(self, config_func, name):
        self.config_func = config_func
        self.name = name
        self.combinations = self.config_func.get_combinations()  # Pre-generate all combinations
        self.current_index = 0

    def increment_last_select(self):
        if self.current_index < len(self.combinations) - 1:
            self.current_index += 1
            return True
        return False

    def get_current_combination(self):
        if self.current_index < len(self.combinations):
            return self.combinations[self.current_index]
        return {}

class HP:
    def __init__(
        self, final_vars: List[str], selections: Dict[str, Any], overrides: Dict[str, Any], explore_mode: bool = False
    ):
        self.final_vars = final_vars
        self.selections = selections
        self.overrides = overrides
        self.config_dict = {}
        self.function_results = {}
        self.current_namespace = []
        
        self.explore_mode = explore_mode
        self.exploration_state = {}
        self.options_for_name = {}
        self.propagated_configs = {}

        self._log_initialization()

    def _log_initialization(self):
        logger.info(
            "Initialized HP with final_vars: %s, selections: %s, and overrides: %s",
            self.final_vars,
            self.selections,
            self.overrides,
        )

    def _explore_select(self, options: Union[Dict[str, Any], List[Any]], full_name: str):
        if isinstance(options, list):
            options = {str(v): v for v in options}

        self.options_for_name[full_name] = options

        if full_name not in self.exploration_state:
            self.exploration_state[full_name] = 0

        return list(options.values())[self.exploration_state[full_name]]

    def increment_last_select(self):
        print("Starting increment_last_select")
        print(f"Current exploration_state: {self.exploration_state}")
        print(f"Current options_for_name: {self.options_for_name}")
        
        # First, try to increment the last local select
        for name in reversed(list(self.exploration_state.keys())):
            print(f"Checking {name}")
            if self.exploration_state[name] < len(self.options_for_name[name]) - 1:
                self.exploration_state[name] += 1
                print(f"Incremented {name} to {self.exploration_state[name]}")
                return True
            else:
                self.exploration_state[name] = 0
                print(f"Reset {name} to 0")
        
        # If no local select could be incremented, try to increment nested configs
        for name in reversed(list(self.propagated_configs.keys())):
            prop_config = self.propagated_configs[name]
            print(f"Checking propagated config: {name}")
            if prop_config.increment_last_select():
                print(f"Incremented nested config {name}")
                return True
        
        print("No more combinations to generate")
        return False

    def get_current_combination(self):
        combination = {name: list(self.options_for_name[name].values())[index] 
                       for name, index in self.exploration_state.items()}
        
        for name, prop_config in self.propagated_configs.items():
            nested_combination = prop_config.get_current_combination()
            for nested_name, nested_value in nested_combination.items():
                combination[f"{name}.{nested_name}"] = nested_value
        
        return combination

    def select(self, options: Union[Dict[str, Any], List[Any]], name: Optional[str] = None, default: Any = None):
        if name is None:
            raise ValueError("`name` argument is missing and must be provided explicitly.")

        self._check_options_exists(options)

        full_name = self._get_full_name(name)

        if self.explore_mode:
            return self._explore_select(options, full_name)

        if isinstance(options, dict):
            self._validate_dict_keys(options)
        elif isinstance(options, list):
            self._validate_list_values(options)
            options = {v: v for v in options}
        else:
            raise ValueError("Options must be a dictionary or a list.")

        self._validate_default(default, options)

        print("Select called with options: %s, name: %s, default: %s", options, name, default)

        if full_name in self.overrides:
            result = self._get_result_from_override(full_name, options)
        elif full_name in self.selections:
            result = self._get_result_from_selection(full_name, options)
        elif default is not None:
            result = options[default]
        else:
            raise ValueError(f"`{full_name}` has no selections, overrides or defaults provided.")

        self._store_value(full_name, result)
        return result

    def text_input(self, default: Optional[str] = None, name: Optional[str] = None) -> str:
        if name is None:
            raise ValueError("`name` argument is missing and must be provided explicitly.")

        full_name = self._get_full_name(name)
        print("Text input called with default: %s, name: %s", default, full_name)

        if full_name in self.overrides:
            result = self.overrides[full_name]
        elif default is None:
            raise ValueError(f"`{full_name}` has no default value or overrides provided.")
        else:
            result = default

        logger.info("Text input for %s: %s", full_name, result)
        self._store_value(full_name, result)
        return result

    def number_input(
        self, default: Optional[Union[int, float]] = None, name: Optional[str] = None
    ) -> Union[int, float]:
        if name is None:
            raise ValueError("`name` argument is missing and must be provided explicitly.")

        full_name = self._get_full_name(name)
        logger.debug("Number input called with default: %s, name: %s", default, full_name)

        if full_name in self.overrides:
            result = self.overrides[full_name]
        elif default is None:
            raise ValueError(f"`{full_name}` has no default value or overrides provided.")
        else:
            result = default

        logger.info("Number input for %s: %s", full_name, result)
        self._store_value(full_name, result)
        return result

    def propagate(self, config_func: Callable, name: Optional[str] = None) -> Union[Dict[str, Any], PropagatedConfig]:
        if name is None:
            raise ValueError("`name` argument is missing and must be provided explicitly.")

        logger.info(f"Propagating configuration for {name}")
        self.current_namespace.append(name)

        nested_config = self._prepare_nested_config(name)

        if self.explore_mode:
            propagated_config = PropagatedConfig(config_func, name)
            self.propagated_configs[name] = propagated_config
            return propagated_config

        # Existing logic for non-explore mode
        result, nested_snapshot = self._run_nested_config(config_func, nested_config)
        self._store_nested_results(name, result, nested_snapshot)
        self.current_namespace.pop()
        return result

    def _get_full_name(self, name: str) -> str:
        return ".".join(self.current_namespace + [name])

    def _store_value(self, full_name: str, value: Any):
        self.config_dict[full_name] = value

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
        if default is not None and default not in options:
            raise ValueError("Default value must be one of the options.")

    def _get_result_from_override(self, full_name: str, options: Dict[str, Any]):
        override_value = self.overrides[full_name]
        logger.debug("Found override for %s: %s", full_name, override_value)
        if override_value in options:
            result = options[override_value]
        else:
            result = override_value
        logger.info("Applied override for %s: %s", full_name, result)
        return result

    def _get_result_from_selection(self, full_name: str, options: Dict[str, Any]):
        selected_value = self.selections[full_name]
        logger.debug("Found selection for %s: %s", full_name, selected_value)
        if selected_value in options:
            result = options[selected_value]
            logger.info("Applied selection for %s: %s", full_name, result)
            return result
        else:
            raise ValueError(
                f"Invalid selection '{selected_value}' for '{full_name}'. Not in options: {list(options.keys())}"
            )

    def _prepare_nested_config(self, name: str) -> Dict[str, Any]:
        return {
            "selections": {k[len(name) + 1 :]: v for k, v in self.selections.items() if k.startswith(f"{name}.")},
            "overrides": {k[len(name) + 1 :]: v for k, v in self.overrides.items() if k.startswith(f"{name}.")},
            "final_vars": [var[len(name) + 1 :] for var in self.final_vars if var.startswith(f"{name}.")],
        }

    def _run_nested_config(
        self, config_func: Callable, nested_config: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        logger.debug(f"Running nested configuration with: {nested_config}")
        return config_func(
            final_vars=nested_config["final_vars"],
            selections=nested_config["selections"],
            overrides=nested_config["overrides"],
            return_config_snapshot=True,
        )

    def _store_nested_results(self, name: str, result: Dict[str, Any], nested_snapshot: Dict[str, Any]):
        self.function_results[name] = result
        for key, value in nested_snapshot.items():
            full_key = f"{name}.{key}"
            self.config_dict[full_key] = value

    def get_config_snapshot(self) -> Dict[str, Any]:
        return self.config_dict

    def get_function_results(self) -> Dict[str, Any]:
        return self.function_results


class InvalidSelectionError(Exception):
    pass
