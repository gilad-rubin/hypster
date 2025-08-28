"""The HP Parameter Interface."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

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

if TYPE_CHECKING:  # only for type hints; avoid runtime imports
    from .hpo.types import HpoCategorical, HpoFloat, HpoInt


@dataclass(frozen=True)
class OptionsAdapter:
    """Helper to normalize options and defaults for select/multi_select."""

    options: Union[List[Any], Dict[Any, Any]]
    options_only: bool

    def keys_and_map(self) -> Tuple[List[Any], Dict[Any, Any]]:
        if isinstance(self.options, dict):
            option_keys = list(self.options.keys())
            option_map = self.options
        else:
            option_keys = list(self.options)
            option_map = {k: k for k in option_keys}
        return option_keys, option_map

    def resolve_default(self, explicit_default: Optional[Any]) -> Optional[Any]:
        if explicit_default is not None:
            return explicit_default
        if isinstance(self.options, dict):
            # First key if available
            return next(iter(self.options.keys()), None)
        # First item if available
        return self.options[0] if self.options else None


class HP:
    """The HP parameter interface for configuration functions."""

    def __init__(self, values: Dict[str, Any], exploration_tracker: Optional[Any] = None):
        """HP is created by instantiate() - users don't instantiate directly."""
        self.values = values or {}
        self.exploration_tracker = exploration_tracker  # Optional, only for explore mode
        self.namespace_stack: List[str] = []  # For nested name prefixes
        self.called_params: set[str] = set()  # Track called parameter names
        self.nested_scopes: set[str] = set()  # Track names that have been nested

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

    # --- Common handlers ---
    def _validate_name(self, name: Optional[str], full_path: str) -> None:
        """Manual name validation for calls that don't use a validator for naming."""
        if name is None:
            raise HPCallError(full_path, "requires 'name' for overrides")

    def _handle_single_value(
        self,
        *,
        default: Any,
        name: str,
        validator: Any,
        supports_strict: bool = False,
        strict: bool = False,
        min: Optional[Union[int, float]] = None,
        max: Optional[Union[int, float]] = None,
        track_called: bool = False,
        use_validator_name: bool = True,
    ) -> Any:
        full_path = self._get_full_param_path(name)

        # Validate name (either via validator or manual message)
        if use_validator_name:
            validator.validate_name(name, full_path)
        else:
            self._validate_name(name, full_path)

        self._validate_name_not_called(name)
        if track_called:
            self.called_params.add(full_path)

        value, found = self._get_value_for_param(name)
        if found:
            if supports_strict:
                validated_value = validator.validate_value(value, full_path, strict=strict)
            else:
                validated_value = validator.validate_value(value, full_path)
        else:
            if supports_strict:
                validated_value = validator.validate_value(default, full_path, strict=strict)
            else:
                validated_value = validator.validate_value(default, full_path)

        # Bounds validation if applicable
        if (min is not None or max is not None) and hasattr(validator, "validate_bounds"):
            validator.validate_bounds(validated_value, min, max, full_path)

        return validated_value

    def _handle_multi_value(
        self,
        *,
        default: List[Any],
        name: str,
        element_validator: Any,
        supports_strict: bool = False,
        strict: bool = False,
        min: Optional[Union[int, float]] = None,
        max: Optional[Union[int, float]] = None,
    ) -> List[Any]:
        full_path = self._get_full_param_path(name)

        # Multi-value calls use manual name validation (to preserve message)
        self._validate_name(name, full_path)
        self._validate_name_not_called(name)
        self.called_params.add(full_path)

        multi_validator = MultiValidator(element_validator)

        value, found = self._get_value_for_param(name)
        if found:
            if supports_strict:
                validated_values = multi_validator.validate_value(value, full_path, strict=strict)
            else:
                validated_values = multi_validator.validate_value(value, full_path)
        else:
            if supports_strict:
                validated_values = multi_validator.validate_value(default, full_path, strict=strict)
            else:
                validated_values = multi_validator.validate_value(default, full_path)

        if min is not None or max is not None:
            multi_validator.validate_bounds(validated_values, min, max, full_path)

        return validated_values

    def _handle_select_single(
        self,
        *,
        options: Union[List[Any], Dict[Any, Any]],
        name: str,
        default: Optional[Any] = None,
        options_only: bool = False,
    ) -> Any:
        validator = SelectValidator()
        full_path = self._get_full_param_path(name)

        validator.validate_name(name, full_path)
        self._validate_name_not_called(name)
        self.called_params.add(full_path)

        adapter = OptionsAdapter(options=options, options_only=options_only)
        option_keys, option_map = adapter.keys_and_map()
        actual_default = adapter.resolve_default(default)

        value, found = self._get_value_for_param(name)
        if found:
            validated_key = validator.validate_value(value, option_keys, options_only, full_path)
            return option_map.get(validated_key, validated_key)
        else:
            if actual_default is not None:
                validated_key = validator.validate_value(actual_default, option_keys, options_only, full_path)
                return option_map.get(validated_key, validated_key)
            return None

    def _handle_select_multi(
        self,
        *,
        options: Union[List[Any], Dict[Any, Any]],
        name: str,
        default: Optional[List[Any]] = None,
        options_only: bool = False,
    ) -> List[Any]:
        validator = SelectValidator()
        full_path = self._get_full_param_path(name)

        validator.validate_name(name, full_path)
        self._validate_name_not_called(name)
        self.called_params.add(full_path)

        adapter = OptionsAdapter(options=options, options_only=options_only)
        option_keys, option_map = adapter.keys_and_map()
        actual_default = list(default or [])

        value, found = self._get_value_for_param(name)
        if found:
            if not isinstance(value, list):
                raise HPCallError(full_path, f"expected list but got {type(value).__name__} ({value})")

            validated_keys = []
            for i, item in enumerate(value):
                validated_key = validator.validate_value(item, option_keys, options_only, f"{full_path}[{i}]")
                validated_keys.append(validated_key)

            return [option_map.get(k, k) for k in validated_keys]
        else:
            validated_keys = []
            for i, item in enumerate(actual_default):
                validated_key = validator.validate_value(item, option_keys, options_only, f"{full_path}[{i}]")
                validated_keys.append(validated_key)

            return [option_map.get(k, k) for k in validated_keys]

    # --- Executor structures ---
    @dataclass(frozen=True)
    class SingleValueSpec:
        name: str
        default: Any
        validator: Any
        supports_strict: bool = False
        strict: bool = False
        min: Optional[Union[int, float]] = None
        max: Optional[Union[int, float]] = None
        track_called: bool = False
        use_validator_name: bool = True

    @dataclass(frozen=True)
    class MultiValueSpec:
        name: str
        default: List[Any]
        element_validator: Any
        supports_strict: bool = False
        strict: bool = False
        min: Optional[Union[int, float]] = None
        max: Optional[Union[int, float]] = None

    @dataclass(frozen=True)
    class SelectSingleSpec:
        name: str
        options: Union[List[Any], Dict[Any, Any]]
        default: Optional[Any] = None
        options_only: bool = False

    @dataclass(frozen=True)
    class SelectMultiSpec:
        name: str
        options: Union[List[Any], Dict[Any, Any]]
        default: Optional[List[Any]] = None
        options_only: bool = False

    # --- Unified executors ---
    def _execute_single(self, spec: "HP.SingleValueSpec") -> Any:
        return self._handle_single_value(
            default=spec.default,
            name=spec.name,
            validator=spec.validator,
            supports_strict=spec.supports_strict,
            strict=spec.strict,
            min=spec.min,
            max=spec.max,
            track_called=spec.track_called,
            use_validator_name=spec.use_validator_name,
        )

    def _execute_multi(self, spec: "HP.MultiValueSpec") -> List[Any]:
        return self._handle_multi_value(
            default=spec.default,
            name=spec.name,
            element_validator=spec.element_validator,
            supports_strict=spec.supports_strict,
            strict=spec.strict,
            min=spec.min,
            max=spec.max,
        )

    def _execute_select_single(self, spec: "HP.SelectSingleSpec") -> Any:
        return self._handle_select_single(
            options=spec.options,
            name=spec.name,
            default=spec.default,
            options_only=spec.options_only,
        )

    def _execute_select_multi(self, spec: "HP.SelectMultiSpec") -> List[Any]:
        return self._handle_select_multi(
            options=spec.options,
            name=spec.name,
            default=spec.default,
            options_only=spec.options_only,
        )

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

        # Disallow nesting under a prefix that already has parameters defined by the parent
        # but allow repeated nesting with the same name to surface duplicate param errors later
        if name not in self.nested_scopes:
            for existing in self.called_params:
                if existing.startswith(full_path + ".") or existing == full_path:
                    raise HPCallError(full_path, f"prefix '{name}' reserved")

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

        # If an explicit nested dict is provided under the name key, it overrides dotted keys
        if name in self.values and isinstance(self.values[name], dict):
            nested_values.update(self.values[name])

        # Merge with explicit values if provided
        if values:
            nested_values.update(values)

        # Create new HP instance for nested call with namespace
        nested_hp = HP(nested_values)
        nested_hp.namespace_stack = self.namespace_stack + [name]
        nested_hp.called_params = self.called_params  # Share called_params tracking

        # Mark that we've nested under this name
        self.nested_scopes.add(name)
        self.called_params.add(full_path)

        # Prepare arguments
        kwargs = kwargs or {}

        # Call the nested function
        result = child(nested_hp, *args, **kwargs)

        # Track nested parameters in parent's called_params
        nested_params = list(nested_hp.called_params)  # Create a copy to avoid iteration issues
        for param in nested_params:
            if "." in param:
                self.called_params.add(param)
            else:
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

    # --- Public API methods ---
    # Use underscore prefix to avoid conflicts with built-in names

    def _int(
        self,
        default: int,
        *,
        name: str,
        min: Optional[int] = None,
        max: Optional[int] = None,
        strict: bool = False,
        hpo_spec: "HpoInt | None" = None,
    ) -> int:
        """Integer parameter with optional bounds validation."""
        spec = HP.SingleValueSpec(
            name=name,
            default=default,
            validator=IntValidator(),
            supports_strict=True,
            strict=strict,
            min=min,
            max=max,
            track_called=True,
            use_validator_name=True,
        )
        return self._execute_single(spec)

    def _float(
        self,
        default: float,
        *,
        name: str,
        min: Optional[float] = None,
        max: Optional[float] = None,
        strict: bool = False,
        hpo_spec: "HpoFloat | None" = None,
    ) -> float:
        """Float parameter with optional bounds validation."""
        spec = HP.SingleValueSpec(
            name=name,
            default=default,
            validator=FloatValidator(),
            supports_strict=True,
            strict=strict,
            min=min,
            max=max,
            track_called=True,
            use_validator_name=True,
        )
        return self._execute_single(spec)

    def _text(self, default: str, *, name: str) -> str:
        """Text parameter."""
        spec = HP.SingleValueSpec(
            name=name,
            default=default,
            validator=TextValidator(),
            supports_strict=False,
            track_called=True,
            use_validator_name=True,
        )
        return self._execute_single(spec)

    def _bool(self, default: bool, *, name: str) -> bool:
        """Boolean parameter."""
        spec = HP.SingleValueSpec(
            name=name,
            default=default,
            validator=BoolValidator(),
            supports_strict=False,
            track_called=True,
            use_validator_name=True,
        )
        return self._execute_single(spec)

    def _select(
        self,
        options: Union[List[Any], Dict[Any, Any]],
        *,
        name: str,
        default: Optional[Any] = None,
        options_only: bool = False,
        hpo_spec: "HpoCategorical | None" = None,
    ) -> Any:
        """Selection parameter from options."""
        spec = HP.SelectSingleSpec(name=name, options=options, default=default, options_only=options_only)
        return self._execute_select_single(spec)

    def _multi_int(
        self,
        default: List[int],
        *,
        name: str,
        min: Optional[int] = None,
        max: Optional[int] = None,
        strict: bool = False,
    ) -> List[int]:
        """Multi-integer parameter with optional bounds validation."""
        spec = HP.MultiValueSpec(
            name=name,
            default=default,
            element_validator=IntValidator(),
            supports_strict=True,
            strict=strict,
            min=min,
            max=max,
        )
        return self._execute_multi(spec)

    def _multi_float(
        self,
        default: List[float],
        *,
        name: str,
        min: Optional[float] = None,
        max: Optional[float] = None,
        strict: bool = False,
    ) -> List[float]:
        """Multi-float parameter with optional bounds validation."""
        spec = HP.MultiValueSpec(
            name=name,
            default=default,
            element_validator=FloatValidator(),
            supports_strict=True,
            strict=strict,
            min=min,
            max=max,
        )
        return self._execute_multi(spec)

    def _multi_text(self, default: List[str], *, name: str) -> List[str]:
        """Multi-text parameter."""
        spec = HP.MultiValueSpec(
            name=name,
            default=default,
            element_validator=TextValidator(),
            supports_strict=False,
        )
        return self._execute_multi(spec)

    def _multi_bool(self, default: List[bool], *, name: str) -> List[bool]:
        """Multi-boolean parameter."""
        spec = HP.MultiValueSpec(
            name=name,
            default=default,
            element_validator=BoolValidator(),
            supports_strict=False,
        )
        return self._execute_multi(spec)

    def _multi_select(
        self,
        options: Union[List[Any], Dict[Any, Any]],
        *,
        name: str,
        default: Optional[List[Any]] = None,
        options_only: bool = False,
    ) -> List[Any]:
        """Multi-selection parameter from options."""
        spec = HP.SelectMultiSpec(name=name, options=options, default=default, options_only=options_only)
        return self._execute_select_multi(spec)

    # Map public API names to internal methods
    def __getattr__(self, name: str) -> Any:
        """Route public API names to internal methods."""
        method_map = {
            "int": self._int,
            "float": self._float,
            "text": self._text,
            "bool": self._bool,
            "select": self._select,
            "multi_int": self._multi_int,
            "multi_float": self._multi_float,
            "multi_text": self._multi_text,
            "multi_bool": self._multi_bool,
            "multi_select": self._multi_select,
        }

        if name in method_map:
            return method_map[name]

        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    # Type stubs for IDE/type checker support
    if TYPE_CHECKING:
        # These declarations exist only for type checking
        int = _int
        float = _float
        text = _text
        bool = _bool
        select = _select
        multi_int = _multi_int
        multi_float = _multi_float
        multi_text = _multi_text
        multi_bool = _multi_bool
        multi_select = _multi_select
