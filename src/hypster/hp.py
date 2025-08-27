"""The HP Parameter Interface."""

from typing import Any, Callable, Dict, List, Optional, Union

from .hp_calls import (
    BoolValidator,
    FloatValidator,
    HPCallError,
    IntValidator,
    MultiValidator,
    SelectValidator,
    TextValidator,
)
from .utils import unflatten_dict


class HP:
    """The HP parameter interface for configuration functions."""

    def __init__(self, values: Dict[str, Any], exploration_tracker: Optional[Any] = None):
        """HP is created by instantiate() - users don't instantiate directly."""
        self.values = values or {}
        self.exploration_tracker = exploration_tracker  # Optional, only for explore mode
        self.namespace_stack: List[str] = []  # For nested name prefixes
        self.called_params: set[str] = set()  # Track called parameter names

    def _get_full_param_path(self, name: str) -> str:
        """Get full parameter path including namespace stack."""
        if self.namespace_stack:
            return ".".join(self.namespace_stack + [name])
        return name

    def _make_full_path(self, name: str) -> str:
        """Make full path for parameter name validation."""
        return self._get_full_param_path(name)

    def _get_value_for_param(self, name: str) -> tuple[Any, bool]:
        """Get value for parameter, returns (value, found)."""
        # Check for exact match with just the name first (for nested contexts)
        if name in self.values:
            return self.values[name], True

        # Then check with full path
        full_path = self._get_full_param_path(name)
        if full_path in self.values:
            return self.values[full_path], True

        # Check for nested structure
        nested_values = unflatten_dict(self.values)
        try:
            current = nested_values
            for part in full_path.split("."):
                current = current[part]
            return current, True
        except (KeyError, TypeError):
            pass

        # Try with just the name in nested structure
        try:
            current = nested_values
            for part in name.split("."):
                current = current[part]
            return current, True
        except (KeyError, TypeError):
            return None, False

    def _validate_name_not_called(self, name: str) -> None:
        """Validate that this parameter name hasn't been called before."""
        full_path = self._make_full_path(name)
        if full_path in self.called_params:
            raise HPCallError(full_path, "has already been defined")

    # Single-value parameters (all require name for overrides)
    def int(
        self, default: int, *, name: str, min: Optional[int] = None, max: Optional[int] = None, strict: bool = False
    ) -> int:
        """Integer parameter with optional bounds validation."""
        validator = IntValidator()
        full_path = self._get_full_param_path(name)

        validator.validate_name(name, full_path)
        self._validate_name_not_called(name)

        # Track this parameter as called
        self.called_params.add(full_path)

        value, found = self._get_value_for_param(name)
        if found:
            validated_value = validator.validate_value(value, full_path, strict=strict)
        else:
            validated_value = validator.validate_value(default, full_path, strict=strict)

        # Validate bounds
        if min is not None or max is not None:
            validator.validate_bounds(validated_value, min, max, full_path)

        return validated_value

    def float(
        self,
        default: float,
        *,
        name: str,
        min: Optional[float] = None,
        max: Optional[float] = None,
        strict: bool = False,
    ) -> float:
        """Float parameter with optional bounds validation."""
        validator = FloatValidator()
        full_path = self._get_full_param_path(name)

        validator.validate_name(name, full_path)
        self._validate_name_not_called(name)

        # Track this parameter as called
        self.called_params.add(full_path)

        value, found = self._get_value_for_param(name)
        if found:
            validated_value = validator.validate_value(value, full_path, strict=strict)
        else:
            validated_value = validator.validate_value(default, full_path, strict=strict)

        # Validate bounds
        if min is not None or max is not None:
            validator.validate_bounds(validated_value, min, max, full_path)

        return validated_value

    def text(self, default: str, *, name: str) -> str:
        """Text parameter."""
        validator = TextValidator()
        full_path = self._get_full_param_path(name)

        validator.validate_name(name, full_path)
        self._validate_name_not_called(name)

        # Track this parameter as called
        self.called_params.add(full_path)

        value, found = self._get_value_for_param(name)
        if found:
            return validator.validate_value(value, full_path)
        else:
            return validator.validate_value(default, full_path)

    def bool(self, default: bool, *, name: str) -> bool:
        """Boolean parameter."""
        validator = BoolValidator()
        full_path = self._get_full_param_path(name)

        validator.validate_name(name, full_path)
        self._validate_name_not_called(name)

        value, found = self._get_value_for_param(name)
        if found:
            return validator.validate_value(value, full_path)
        else:
            return validator.validate_value(default, full_path)

    def select(
        self,
        options: Union[List[Any], Dict[Any, Any]],
        *,
        name: str,
        default: Optional[Any] = None,
        options_only: bool = False,
    ) -> Any:
        """Selection parameter from options."""
        validator = SelectValidator()
        full_path = self._get_full_param_path(name)

        validator.validate_name(name, full_path)
        self._validate_name_not_called(name)

        # Handle dict options - convert to list of keys for validation
        if isinstance(options, dict):
            option_keys = list(options.keys())
            option_map = options
        else:
            option_keys = options
            option_map = {k: k for k in options}

        # Determine the actual default value
        if default is None:
            if isinstance(options, dict):
                actual_default = list(options.keys())[0] if options else None
            else:
                actual_default = options[0] if options else None
        else:
            actual_default = default

        value, found = self._get_value_for_param(name)
        if found:
            validated_key = validator.validate_value(value, option_keys, options_only, full_path)
            # Return mapped value for dict options
            if isinstance(options, dict):
                return option_map.get(validated_key, validated_key)
            return validated_key
        else:
            # Use default
            if actual_default is not None:
                validated_key = validator.validate_value(actual_default, option_keys, options_only, full_path)
                # Return mapped value for dict options
                if isinstance(options, dict):
                    return option_map.get(validated_key, validated_key)
                return validated_key
            return None

    # Multi-value parameters
    def multi_int(
        self,
        default: List[int],
        *,
        name: str,
        min: Optional[int] = None,
        max: Optional[int] = None,
        strict: bool = False,
    ) -> List[int]:
        """Multi-integer parameter with optional bounds validation."""
        validator = MultiValidator(IntValidator())
        full_path = self._get_full_param_path(name)

        # Validate name
        if name is None:
            raise HPCallError(full_path, "requires 'name' for overrides")
        self._validate_name_not_called(name)

        value, found = self._get_value_for_param(name)
        if found:
            validated_values = validator.validate_value(value, full_path, strict=strict)
        else:
            validated_values = validator.validate_value(default, full_path, strict=strict)

        # Validate bounds for each element
        if min is not None or max is not None:
            validator.validate_bounds(validated_values, min, max, full_path)

        return validated_values

    def multi_float(
        self,
        default: List[float],
        *,
        name: str,
        min: Optional[float] = None,
        max: Optional[float] = None,
        strict: bool = False,
    ) -> List[float]:
        """Multi-float parameter with optional bounds validation."""
        validator = MultiValidator(FloatValidator())
        full_path = self._get_full_param_path(name)

        # Validate name
        if name is None:
            raise HPCallError(full_path, "requires 'name' for overrides")
        self._validate_name_not_called(name)

        value, found = self._get_value_for_param(name)
        if found:
            validated_values = validator.validate_value(value, full_path, strict=strict)
        else:
            validated_values = validator.validate_value(default, full_path, strict=strict)

        # Validate bounds for each element
        if min is not None or max is not None:
            validator.validate_bounds(validated_values, min, max, full_path)

        return validated_values

    def multi_text(self, default: List[str], *, name: str) -> List[str]:
        """Multi-text parameter."""
        validator = MultiValidator(TextValidator())
        full_path = self._get_full_param_path(name)

        # Validate name
        if name is None:
            raise HPCallError(full_path, "requires 'name' for overrides")
        self._validate_name_not_called(name)

        value, found = self._get_value_for_param(name)
        if found:
            return validator.validate_value(value, full_path)
        else:
            return validator.validate_value(default, full_path)

    def multi_bool(self, default: List[bool], *, name: str) -> List[bool]:
        """Multi-boolean parameter."""
        validator = MultiValidator(BoolValidator())
        full_path = self._get_full_param_path(name)

        # Validate name
        if name is None:
            raise HPCallError(full_path, "requires 'name' for overrides")
        self._validate_name_not_called(name)

        value, found = self._get_value_for_param(name)
        if found:
            return validator.validate_value(value, full_path)
        else:
            return validator.validate_value(default, full_path)

    def multi_select(
        self,
        options: Union[List[Any], Dict[Any, Any]],
        *,
        name: str,
        default: Optional[List[Any]] = None,
        options_only: bool = False,
    ) -> List[Any]:
        """Multi-selection parameter from options."""
        validator = SelectValidator()
        full_path = self._get_full_param_path(name)

        validator.validate_name(name, full_path)
        self._validate_name_not_called(name)

        # Handle dict options
        if isinstance(options, dict):
            option_keys = list(options.keys())
            option_map = options
        else:
            option_keys = options
            option_map = {k: k for k in options}

        # Determine the actual default value
        actual_default = default or []

        value, found = self._get_value_for_param(name)
        if found:
            if not isinstance(value, list):
                raise HPCallError(full_path, f"expected list but got {type(value).__name__} ({value})")

            validated_keys = []
            for i, item in enumerate(value):
                validated_key = validator.validate_value(item, option_keys, options_only, f"{full_path}[{i}]")
                validated_keys.append(validated_key)

            # Return mapped values for dict options
            if isinstance(options, dict):
                return [option_map.get(k, k) for k in validated_keys]
            return validated_keys
        else:
            # Use default
            validated_keys = []
            for i, item in enumerate(actual_default):
                validated_key = validator.validate_value(item, option_keys, options_only, f"{full_path}[{i}]")
                validated_keys.append(validated_key)

            # Return mapped values for dict options
            if isinstance(options, dict):
                return [option_map.get(k, k) for k in validated_keys]
            return validated_keys

    # Composition
    def nest(
        self,
        child: Callable,
        *,
        name: str,
        values: Optional[Dict[str, Any]] = None,
        args: tuple = (),
        kwargs: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Nest another configuration function."""
        from .utils import validate_config_func_signature

        full_path = self._get_full_param_path(name)

        # Validate name
        if name is None:
            raise HPCallError(full_path, "requires 'name' for nesting")

        # Validate function signature
        validate_config_func_signature(child)

        # Prepare nested values by filtering with name prefix
        nested_values = {}
        prefix = name + "."  # Use just the name, not the full path

        # Extract nested values from main values dict
        for key, value in self.values.items():
            if key.startswith(prefix):
                nested_key = key[len(prefix) :]
                nested_values[nested_key] = value

        # Merge with explicit values if provided
        if values:
            nested_values.update(values)

        # Create new HP instance for nested call with namespace
        nested_hp = HP(nested_values)
        nested_hp.namespace_stack = self.namespace_stack + [name]
        nested_hp.called_params = self.called_params  # Share called_params tracking

        # Prepare arguments
        kwargs = kwargs or {}

        # Call the nested function
        result = child(nested_hp, *args, **kwargs)

        # Track nested parameters in parent's called_params
        nested_params = list(nested_hp.called_params)  # Create a copy to avoid iteration issues
        for param in nested_params:
            # Check if the parameter is already a fully qualified path or just a local name
            if "." in param:
                # It's already a fully qualified nested parameter, add it as-is
                self.called_params.add(param)
            else:
                # It's a local parameter from the nested function, prefix it
                full_nested_param = f"{full_path}.{param}"
                self.called_params.add(full_nested_param)

        return result

    # Helpers
    def collect(
        self, locals_dict: Dict[str, Any], include: Optional[List[str]] = None, exclude: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Helper to collect local variables, excluding HP-related ones."""
        result = {}

        # Default exclusions
        default_exclude = {"hp", "self", "__builtins__", "__name__", "__doc__"}
        exclude_set = set(exclude or []) | default_exclude

        for key, value in locals_dict.items():
            # Skip private/dunder variables
            if key.startswith("_"):
                continue

            # Apply include filter if specified
            if include is not None and key not in include:
                continue

            # Apply exclude filter
            if key in exclude_set:
                continue

            result[key] = value

        return result
