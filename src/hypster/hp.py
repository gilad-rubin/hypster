from typing import Any, Callable, Dict, List, Optional, Union

from .logging_utils import configure_logging

logger = configure_logging()


class HP:
    def __init__(
        self,
        final_vars: List[str],
        selections: Dict[str, Any],
        overrides: Dict[str, Any],
    ):
        self.final_vars = final_vars
        self.selections = selections
        self.overrides = overrides
        self.config_dict = {}  # Stores only HP call results
        self.function_results = {}  # Stores full function results
        self.current_namespace = []
        logger.info(
            "Initialized HP with final_vars: %s, selections: %s, and overrides: %s",
            self.final_vars,
            self.selections,
            self.overrides,
        )

    def _get_full_name(self, name: str) -> str:
        return ".".join(self.current_namespace + [name])

    def _store_value(self, name: str, value: Any):
        full_name = self._get_full_name(name)
        self.config_dict[full_name] = value

    def select(
        self,
        options: Union[Dict[str, Any], List[Any]],
        name: str = None,
        default: Any = None,
    ):
        if name is None:
            raise ValueError("Name must be provided explicitly or automatically inferred.")

        full_name = self._get_full_name(name)
        if options is None or len(options) == 0:
            raise ValueError("Options must be a non-empty list or dictionary.")

        if isinstance(options, dict):
            if not all(isinstance(k, (str, int, bool, float)) for k in options.keys()):
                bad_keys = [key for key in options.keys() if not isinstance(key, (str, int, bool, float))]
                raise ValueError(f"Dictionary keys must be str, int, bool, float. got {bad_keys} instead.")
        elif isinstance(options, list):
            if not all(isinstance(v, (str, int, bool, float)) for v in options):
                raise ValueError(
                    "List values must be one of: str, int, bool, float. For complex types - use a dictionary instead"
                )
            options = {v: v for v in options}
        else:
            raise ValueError("Options must be a dictionary or a list.")

        if default is not None and default not in options:
            raise ValueError("Default value must be one of the options.")

        logger.debug(
            "Select called with options: %s, name: %s, default: %s",
            options,
            name,
            default,
        )

        result = None
        if full_name in self.overrides:
            override_value = self.overrides[full_name]
            logger.debug("Found override for %s: %s", full_name, override_value)
            if override_value in options:
                result = options[override_value]
            else:
                result = override_value
            logger.info("Applied override for %s: %s", full_name, result)
        elif full_name in self.selections:
            selected_value = self.selections[full_name]
            logger.debug("Found selection for %s: %s", full_name, selected_value)
            if selected_value in options:
                result = options[selected_value]
                logger.info("Applied selection for %s: %s", full_name, result)
            else:
                raise ValueError(
                    f"Invalid selection '{selected_value}' for '{full_name}'. Not in options: {list(options.keys())}"
                )
        elif default is not None:
            result = options[default]
        else:
            raise ValueError(f"No selection or override found for {full_name} and no default provided.")

        self._store_value(name, result)
        return result

    def text_input(self, default: Optional[str] = None, name: Optional[str] = None) -> str:
        if name is None:
            raise ValueError("Name must be provided explicitly or automatically inferred.")

        full_name = self._get_full_name(name)
        logger.debug("Text input called with default: %s, name: %s", default, full_name)

        if full_name in self.overrides:
            result = self.overrides[full_name]
        elif default is None:
            raise ValueError(f"No default value or override provided for text input {full_name}.")
        else:
            result = default

        logger.info("Text input for %s: %s", full_name, result)

        self._store_value(name, result)
        return result

    def number_input(
        self,
        default: Optional[Union[int, float]] = None,
        name: Optional[str] = None,
    ) -> Union[int, float]:
        if name is None:
            raise ValueError("Name must be provided explicitly or automatically inferred.")

        full_name = self._get_full_name(name)
        logger.debug(
            "Number input called with default: %s, name: %s",
            default,
            full_name,
        )

        if full_name in self.overrides:
            result = self.overrides[full_name]
        elif default is None:
            raise ValueError(f"No default value or override provided for number input {full_name}.")
        else:
            result = default

        # Ensure the result is of the same type as the default
        if default is not None:
            if isinstance(default, int):
                result = int(result)
            elif isinstance(default, float):
                result = float(result)

        logger.info("Number input for %s: %s", full_name, result)

        self._store_value(name, result)
        return result

    def propagate(self, config_func: Callable, name: str = None) -> Dict[str, Any]:
        logger.info(f"Propagating configuration for {name}")

        if name is None:
            name = config_func.__name__

        self.current_namespace.append(name)

        # Create dictionaries for the nested configuration
        nested_selections = {k[len(name) + 1 :]: v for k, v in self.selections.items() if k.startswith(f"{name}.")}
        nested_overrides = {k[len(name) + 1 :]: v for k, v in self.overrides.items() if k.startswith(f"{name}.")}

        # Automatically propagate final_vars
        nested_final_vars = [var[len(name) + 1 :] for var in self.final_vars if var.startswith(f"{name}.")]

        logger.debug(
            f"Propagated configuration for {name} with Selections:\n{nested_selections}\
                \n& Overrides:\n{nested_overrides}\nAuto-propagated final vars: {nested_final_vars}"
        )

        # Run the nested configuration
        result, nested_snapshot = config_func(
            final_vars=nested_final_vars,
            selections=nested_selections,
            overrides=nested_overrides,
            return_config_snapshot=True,
        )

        # Store the full result
        self.function_results[name] = result

        # Merge only the HP call results into the parent configuration
        for key, value in nested_snapshot.items():
            full_key = f"{name}.{key}"
            self.config_dict[full_key] = value

        self.current_namespace.pop()

        # Return the full result of the propagated function
        return result

    def get_config_snapshot(self) -> Dict[str, Any]:
        return self.config_dict

    def get_function_results(self) -> Dict[str, Any]:
        return self.function_results


class InvalidSelectionError(Exception):
    pass
