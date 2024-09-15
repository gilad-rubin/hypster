# from .logging_utils import configure_logging
import itertools
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union

# Define a type variable for the allowed key types
KeyType = TypeVar("KeyType", str, int, float, bool)

# Define a type alias for the options dictionary
OptionsDict = Dict[KeyType, Any]

# Define a type alias for the options list
OptionsList = List[Union[str, int, float, bool]]
# Correct logging configuration
logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def cross_join_nested_dict(nested_dict):
    # Extract the lists of dictionaries
    dict_lists = list(nested_dict.values())

    # Generate all combinations
    result = []
    for combination in itertools.product(*dict_lists):
        combined = {}
        for d in combination:
            combined.update(d)
        result.append(combined)

    return result


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
        self.propagated_combinations = {}

        self._log_initialization()

    def _log_initialization(self):
        logger.info(
            "Initialized HP with final_vars: %s, selections: %s, and overrides: %s",
            self.final_vars,
            self.selections,
            self.overrides,
        )

    def _explore_select(self, options: dict, full_name: str):
        self.options_for_name[full_name] = options

        if full_name not in self.exploration_state:
            self.exploration_state[full_name] = 0

        return list(options.values())[self.exploration_state[full_name]]

    def increment_last_select(self):
        # First, try to increment the last local select
        for name in reversed(list(self.options_for_name.keys())):
            if self.exploration_state[name] < len(self.options_for_name[name]) - 1:
                self.exploration_state[name] += 1
                return True
            else:
                self.exploration_state[name] = 0

        return False

    def get_current_combinations(self):
        select_combination = {
            name: list(self.options_for_name[name].keys())[index]
            for name, index in self.exploration_state.items()
            if name in self.options_for_name
        }
        dcts = {}
        if len(select_combination) > 0:
            dcts["hp_select"] = [select_combination]
        if len(self.propagated_combinations) > 0:
            dcts.update(self.propagated_combinations.copy())

        return cross_join_nested_dict(dcts)

    def select(self, options: Union[Dict[str, Any], List[Any]], *, name: Optional[str] = None, default: Any = None):
        if name is None:
            raise ValueError("`name` argument is missing and must be provided explicitly.")

        self._check_options_exists(options)

        full_name = self._get_full_name(name)

        if isinstance(options, dict):
            self._validate_dict_keys(options)
        elif isinstance(options, list):
            self._validate_list_values(options)
            options = {v: v for v in options}
        else:
            raise ValueError("Options must be a dictionary or a list.")

        self._validate_default(default, options)

        if self.explore_mode:
            return self._explore_select(options, full_name)

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

    def text_input(self, default: Optional[str] = None, *, name: Optional[str] = None) -> str:
        if name is None:
            raise ValueError("`name` argument is missing and must be provided explicitly.")

        full_name = self._get_full_name(name)

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
        self, default: Optional[Union[int, float]] = None, *, name: Optional[str] = None
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

    def _explore_propagate(self, name: str, config_func: Callable):
        combinations = config_func.get_combinations()
        renamed_combinations = []
        for combination in combinations:
            renamed_combination = {}
            for key, value in combination.items():
                renamed_combination[f"{name}.{key}"] = value
            renamed_combinations.append(renamed_combination)
        self.propagated_combinations[name] = renamed_combinations
        self.current_namespace.pop()
        return

    def propagate(
        self, config_func: Callable, *, name: Optional[str] = None
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        if name is None:
            raise ValueError("`name` argument is missing and must be provided explicitly.")

        logger.info(f"Propagating configuration for {name}")
        self.current_namespace.append(name)

        if self.explore_mode:
            return self._explore_propagate(name, config_func)

        nested_config = self._prepare_nested_config(name)
        result, nested_snapshot = self._run_nested_config(config_func, nested_config)
        self._store_nested_results(name, result, nested_snapshot)
        self.current_namespace.pop()
        return result

    def _get_full_name(self, name: str) -> str:
        return ".".join(self.current_namespace + [name])

    def _store_value(self, full_name: str, value: Any):
        self.config_dict[full_name] = value

    def _check_options_exists(self, options: Union[OptionsDict, OptionsList]):
        if not isinstance(options, (list, dict)) or len(options) == 0:
            raise ValueError("Options must be a non-empty list or dictionary.")

    def _validate_dict_keys(self, options: OptionsDict):
        if not all(isinstance(k, (str, int, bool, float)) for k in options.keys()):
            bad_keys = [key for key in options.keys() if not isinstance(key, (str, int, bool, float))]
            raise ValueError(f"Dictionary keys must be str, int, bool, float. Got {bad_keys} instead.")

    def _validate_list_values(self, options: OptionsList):
        if not all(isinstance(v, (str, int, bool, float)) for v in options):
            raise ValueError(
                "List values must be one of: str, int, bool, float. For complex types - use a dictionary instead"
            )

    def _validate_default(self, default: Any, options: OptionsDict):
        if default is not None and default not in options:
            raise ValueError("Default value must be one of the options.")

    def _get_result_from_override(self, full_name: str, options: OptionsDict):
        override_value = self.overrides[full_name]
        logger.debug("Found override for %s: %s", full_name, override_value)
        if override_value in options:
            result = options[override_value]
        else:
            result = override_value
        logger.info("Applied override for %s: %s", full_name, result)
        return result

    def _get_result_from_selection(self, full_name: str, options: OptionsDict):
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
