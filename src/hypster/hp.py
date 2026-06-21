"""The HP Parameter Interface."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Literal, Optional, Protocol, Tuple, Union, overload

from ._sentinels import NO_DEFAULT as _NO_DEFAULT
from .hp_calls import (
    BoolValidator,
    FloatValidator,
    HPCallError,
    IntValidator,
    MultiValidator,
    SelectValidator,
    TextValidator,
)
from .utils import normalize_values, validate_identifier_name, validate_metadata, validate_select_choice

if TYPE_CHECKING:  # only for type hints; avoid runtime imports
    from .hpo.types import HpoCategorical, HpoFloat, HpoInt


class ParameterTracker(Protocol):
    """Receives parameter-recording events from HP execution."""

    def record_parameter(
        self,
        *,
        path: str,
        name: str,
        kind: str,
        default_value: Any,
        selected_value: Any,
        options: Optional[List[Any]] = None,
        minimum: Optional[Union[int, float]] = None,
        maximum: Optional[Union[int, float]] = None,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None: ...


@dataclass(frozen=True)
class OptionsAdapter:
    """Helper to normalize options and defaults for select/multi_select."""

    options: Union[List[Any], Mapping[Any, Any]]

    def keys_and_map(self) -> Tuple[List[Any], Mapping[Any, Any]]:
        if isinstance(self.options, Mapping):
            option_keys = list(self.options.keys())
            option_map = self.options
        else:
            option_keys = list(self.options)
            option_map = {}
        return option_keys, option_map

    def resolve_default(self, explicit_default: Any) -> Any:
        if explicit_default is not _NO_DEFAULT:
            return explicit_default
        if isinstance(self.options, Mapping):
            # First key if available
            return next(iter(self.options.keys()), None)
        # First item if available
        return self.options[0] if self.options else None


class HP:
    """The HP parameter interface for configuration functions."""

    def __init__(self, values: Dict[str, Any], parameter_tracker: Optional[ParameterTracker] = None):
        """HP is created by instantiate() - users don't instantiate directly."""
        self.values = values or {}
        self.parameter_tracker = parameter_tracker
        self.namespace_stack: List[str] = []  # For nested name prefixes
        self.called_params: set[str] = set()  # Track called parameter names
        self.nested_scopes: set[str] = set()  # Track names that have been nested
        self.nested_scope_paths: set[str] = set()  # Track nest/group paths separately from parameter leaves

    def _get_full_param_path(self, name: str) -> str:
        """Get full parameter path including namespace stack."""
        if self.namespace_stack:
            return ".".join(self.namespace_stack + [name])
        return name

    def _make_full_path(self, name: str) -> str:
        """Make full path for parameter name validation."""
        return self._get_full_param_path(name)

    def _record_parameter(
        self,
        *,
        path: str,
        name: str,
        kind: str,
        default_value: Any,
        selected_value: Any,
        options: Optional[List[Any]] = None,
        minimum: Optional[Union[int, float]] = None,
        maximum: Optional[Union[int, float]] = None,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if self.parameter_tracker is None:
            return

        record = getattr(self.parameter_tracker, "record_parameter", None)
        if callable(record):
            record(
                path=path,
                name=name,
                kind=kind,
                default_value=default_value,
                selected_value=selected_value,
                options=options,
                minimum=minimum,
                maximum=maximum,
                description=description,
                metadata=metadata,
            )

    def _record_nest(self, *, path: str, name: str, description: Optional[str] = None) -> None:
        if self.parameter_tracker is None:
            return

        record = getattr(self.parameter_tracker, "record_nest", None)
        if callable(record):
            record(path=path, name=name, description=description)

    def _get_value_for_param(self, name: str) -> tuple[Any, bool]:
        """Get value for parameter, returns (value, found)."""
        # Check for exact match with just the name first (for nested contexts)
        if name in self.values:
            return self.values[name], True

        # Then check with full path
        full_path = self._get_full_param_path(name)
        if full_path in self.values:
            return self.values[full_path], True

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
        validate_identifier_name(name, kind="parameter name")

    def _handle_single_value(
        self,
        *,
        default: Any,
        name: str,
        param_type: str,
        validator: Any,
        supports_strict: bool = False,
        strict: bool = False,
        min: Optional[Union[int, float]] = None,
        max: Optional[Union[int, float]] = None,
        allow_none: bool = False,
        track_called: bool = False,
        use_validator_name: bool = True,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Any:
        full_path = self._get_full_param_path(name)

        # Validate name (either via validator or manual message)
        if use_validator_name:
            validator.validate_name(name, full_path)
            validate_identifier_name(name, kind="parameter name")
        else:
            self._validate_name(name, full_path)

        self._validate_name_not_called(name)
        if track_called:
            self.called_params.add(full_path)

        if default is None and not allow_none:
            raise HPCallError(
                full_path,
                "default=None requires allow_none=True. How to fix: pass allow_none=True, or use a non-None default.",
            )

        value, found = self._get_value_for_param(name)
        raw_value = value if found else default
        if raw_value is None:
            if not allow_none:
                raise HPCallError(
                    full_path,
                    "None is only allowed when allow_none=True. "
                    "How to fix: pass allow_none=True, or provide a non-None value.",
                )
            validated_value = None
        else:
            if supports_strict:
                validated_value = validator.validate_value(raw_value, full_path, strict=strict)
            else:
                validated_value = validator.validate_value(raw_value, full_path)

        # Bounds validation if applicable
        if (
            validated_value is not None
            and (min is not None or max is not None)
            and hasattr(validator, "validate_bounds")
        ):
            validator.validate_bounds(validated_value, min, max, full_path)

        metadata = validate_metadata(metadata, param_path=full_path)
        self._record_parameter(
            path=full_path,
            name=name,
            kind=param_type,
            default_value=default,
            selected_value=validated_value,
            minimum=min,
            maximum=max,
            description=description,
            metadata=metadata,
        )

        return validated_value

    def _handle_multi_value(
        self,
        *,
        default: List[Any],
        name: str,
        param_type: str,
        element_validator: Any,
        supports_strict: bool = False,
        strict: bool = False,
        min: Optional[Union[int, float]] = None,
        max: Optional[Union[int, float]] = None,
        allow_none: bool = False,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Any]:
        full_path = self._get_full_param_path(name)

        # Multi-value calls use manual name validation (to preserve message)
        self._validate_name(name, full_path)
        self._validate_name_not_called(name)
        self.called_params.add(full_path)

        if allow_none:
            raise HPCallError(
                full_path,
                f"allow_none=True is not supported for hp.{param_type}() yet. "
                "How to fix: remove allow_none=True, or use hp.multi_select([...], allow_none=True) "
                "when you need nullable categorical choices.",
            )

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

        metadata = validate_metadata(metadata, param_path=full_path)
        self._record_parameter(
            path=full_path,
            name=name,
            kind=param_type,
            default_value=default,
            selected_value=validated_values,
            minimum=min,
            maximum=max,
            description=description,
            metadata=metadata,
        )

        return validated_values

    def _handle_select_single(
        self,
        *,
        options: Union[List[Any], Mapping[Any, Any]],
        name: str,
        default: Any = _NO_DEFAULT,
        options_only: bool = False,
        allow_none: bool = False,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Any:
        validator = SelectValidator()
        full_path = self._get_full_param_path(name)

        validator.validate_name(name, full_path)
        validate_identifier_name(name, kind="parameter name")
        self._validate_name_not_called(name)
        self.called_params.add(full_path)

        is_mapping = isinstance(options, Mapping)
        adapter = OptionsAdapter(options=options)
        option_keys, option_map = adapter.keys_and_map()
        for index, option in enumerate(option_keys):
            validate_select_choice(option, param_path=f"{full_path} option #{index}", allow_none=allow_none)
        actual_default = adapter.resolve_default(default)

        value, found = self._get_value_for_param(name)
        if found:
            validate_select_choice(value, param_path=full_path, allow_none=allow_none)
            validated_key = validator.validate_value(value, option_keys, options_only, full_path)
            metadata = validate_metadata(metadata, param_path=full_path)
            self._record_parameter(
                path=full_path,
                name=name,
                kind="select",
                default_value=actual_default,
                selected_value=validated_key,
                options=option_keys,
                description=description,
                metadata=metadata,
            )
            return option_map.get(validated_key, validated_key) if is_mapping else validated_key
        else:
            if actual_default is None and default is _NO_DEFAULT and not option_keys and not allow_none:
                raise HPCallError(
                    full_path,
                    "select has no options and no default. "
                    "How to fix: provide at least one option, pass default=..., or pass allow_none=True.",
                )
            validate_select_choice(actual_default, param_path=full_path, allow_none=allow_none)
            validated_key = validator.validate_value(actual_default, option_keys, options_only, full_path)
            metadata = validate_metadata(metadata, param_path=full_path)
            self._record_parameter(
                path=full_path,
                name=name,
                kind="select",
                default_value=actual_default,
                selected_value=validated_key,
                options=option_keys,
                description=description,
                metadata=metadata,
            )
            return option_map.get(validated_key, validated_key) if is_mapping else validated_key

    def _handle_select_multi(
        self,
        *,
        options: Union[List[Any], Mapping[Any, Any]],
        name: str,
        default: Optional[List[Any]] = None,
        options_only: bool = False,
        allow_none: bool = False,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Any]:
        validator = SelectValidator()
        full_path = self._get_full_param_path(name)

        validator.validate_name(name, full_path)
        validate_identifier_name(name, kind="parameter name")
        self._validate_name_not_called(name)
        self.called_params.add(full_path)

        is_mapping = isinstance(options, Mapping)
        adapter = OptionsAdapter(options=options)
        option_keys, option_map = adapter.keys_and_map()
        for index, option in enumerate(option_keys):
            validate_select_choice(option, param_path=f"{full_path} option #{index}", allow_none=allow_none)
        actual_default = list(default or [])

        value, found = self._get_value_for_param(name)
        if found:
            if not isinstance(value, list):
                raise HPCallError(full_path, f"expected list but got {type(value).__name__} ({value})")

            validated_keys = []
            for i, item in enumerate(value):
                validate_select_choice(item, param_path=f"{full_path}[{i}]", allow_none=allow_none)
                validated_key = validator.validate_value(item, option_keys, options_only, f"{full_path}[{i}]")
                validated_keys.append(validated_key)

            metadata = validate_metadata(metadata, param_path=full_path)
            self._record_parameter(
                path=full_path,
                name=name,
                kind="multi_select",
                default_value=actual_default,
                selected_value=validated_keys,
                options=option_keys,
                description=description,
                metadata=metadata,
            )
            return [option_map.get(k, k) for k in validated_keys] if is_mapping else validated_keys
        else:
            validated_keys = []
            for i, item in enumerate(actual_default):
                validate_select_choice(item, param_path=f"{full_path}[{i}]", allow_none=allow_none)
                validated_key = validator.validate_value(item, option_keys, options_only, f"{full_path}[{i}]")
                validated_keys.append(validated_key)

            metadata = validate_metadata(metadata, param_path=full_path)
            self._record_parameter(
                path=full_path,
                name=name,
                kind="multi_select",
                default_value=actual_default,
                selected_value=validated_keys,
                options=option_keys,
                description=description,
                metadata=metadata,
            )
            return [option_map.get(k, k) for k in validated_keys] if is_mapping else validated_keys

    # --- Executor structures ---
    @dataclass(frozen=True)
    class SingleValueSpec:
        name: str
        default: Any
        param_type: str
        validator: Any
        supports_strict: bool = False
        strict: bool = False
        min: Optional[Union[int, float]] = None
        max: Optional[Union[int, float]] = None
        track_called: bool = False
        use_validator_name: bool = True
        allow_none: bool = False
        description: Optional[str] = None
        metadata: Optional[Dict[str, Any]] = None

    @dataclass(frozen=True)
    class MultiValueSpec:
        name: str
        default: List[Any]
        param_type: str
        element_validator: Any
        supports_strict: bool = False
        strict: bool = False
        min: Optional[Union[int, float]] = None
        max: Optional[Union[int, float]] = None
        allow_none: bool = False
        description: Optional[str] = None
        metadata: Optional[Dict[str, Any]] = None

    @dataclass(frozen=True)
    class SelectSingleSpec:
        name: str
        options: Union[List[Any], Mapping[Any, Any]]
        default: Any = _NO_DEFAULT
        options_only: bool = False
        allow_none: bool = False
        description: Optional[str] = None
        metadata: Optional[Dict[str, Any]] = None

    @dataclass(frozen=True)
    class SelectMultiSpec:
        name: str
        options: Union[List[Any], Mapping[Any, Any]]
        default: Optional[List[Any]] = None
        options_only: bool = False
        allow_none: bool = False
        description: Optional[str] = None
        metadata: Optional[Dict[str, Any]] = None

    # --- Unified executors ---
    def _execute_single(self, spec: "HP.SingleValueSpec") -> Any:
        return self._handle_single_value(
            default=spec.default,
            name=spec.name,
            param_type=spec.param_type,
            validator=spec.validator,
            supports_strict=spec.supports_strict,
            strict=spec.strict,
            min=spec.min,
            max=spec.max,
            allow_none=spec.allow_none,
            track_called=spec.track_called,
            use_validator_name=spec.use_validator_name,
            description=spec.description,
            metadata=spec.metadata,
        )

    def _execute_multi(self, spec: "HP.MultiValueSpec") -> List[Any]:
        return self._handle_multi_value(
            default=spec.default,
            name=spec.name,
            param_type=spec.param_type,
            element_validator=spec.element_validator,
            supports_strict=spec.supports_strict,
            strict=spec.strict,
            min=spec.min,
            max=spec.max,
            allow_none=spec.allow_none,
            description=spec.description,
            metadata=spec.metadata,
        )

    def _execute_select_single(self, spec: "HP.SelectSingleSpec") -> Any:
        return self._handle_select_single(
            options=spec.options,
            name=spec.name,
            default=spec.default,
            options_only=spec.options_only,
            allow_none=spec.allow_none,
            description=spec.description,
            metadata=spec.metadata,
        )

    def _execute_select_multi(self, spec: "HP.SelectMultiSpec") -> List[Any]:
        return self._handle_select_multi(
            options=spec.options,
            name=spec.name,
            default=spec.default,
            options_only=spec.options_only,
            allow_none=spec.allow_none,
            description=spec.description,
            metadata=spec.metadata,
        )

    # Composition
    def nest(
        self,
        child: Callable,
        *,
        name: str,
        values: Optional[Dict[str, Any]] = None,
        description: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """Nest another configuration function."""
        from .utils import validate_config_func_signature

        full_path = self._get_full_param_path(name)

        # Validate name
        if name is None:
            raise HPCallError(full_path, "requires 'name' for nesting")
        validate_identifier_name(name, kind="nest name")

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

        explicit_values = normalize_values(values) if values else {}

        # Merge with explicit values if provided
        if explicit_values:
            nested_values.update(explicit_values)

        # Create new HP instance for nested call with namespace
        self._record_nest(path=full_path, name=name, description=description)

        nested_hp = self.__class__(nested_values, parameter_tracker=self.parameter_tracker)
        nested_hp.namespace_stack = self.namespace_stack + [name]
        nested_hp.called_params = self.called_params  # Share called_params tracking
        nested_hp.nested_scope_paths = self.nested_scope_paths

        # Mark that we've nested under this name
        self.nested_scopes.add(name)
        self.called_params.add(full_path)
        self.nested_scope_paths.add(full_path)

        from .core import _reject_removed_execution_argument_containers

        _reject_removed_execution_argument_containers(kwargs)

        # Call the nested function
        result = child(nested_hp, **kwargs)

        if explicit_values:
            from .core import _handle_unknown_parameters

            prefixed_explicit_values = {f"{full_path}.{key}": value for key, value in explicit_values.items()}
            leaf_params = self.called_params - self.nested_scope_paths
            _handle_unknown_parameters(prefixed_explicit_values, leaf_params, "raise")

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

    @overload
    def _int(
        self,
        default: int,
        *,
        name: str,
        min: Optional[int] = None,
        max: Optional[int] = None,
        strict: bool = False,
        allow_none: Literal[False] = False,
        hpo_spec: "HpoInt | None" = None,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int: ...

    @overload
    def _int(
        self,
        default: Optional[int],
        *,
        name: str,
        min: Optional[int] = None,
        max: Optional[int] = None,
        strict: bool = False,
        allow_none: Literal[True],
        hpo_spec: "HpoInt | None" = None,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[int]: ...

    def _int(
        self,
        default: Optional[int],
        *,
        name: str,
        min: Optional[int] = None,
        max: Optional[int] = None,
        strict: bool = False,
        allow_none: bool = False,
        hpo_spec: "HpoInt | None" = None,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[int]:
        """Integer parameter with optional bounds validation."""
        spec = HP.SingleValueSpec(
            name=name,
            default=default,
            param_type="int",
            validator=IntValidator(),
            supports_strict=True,
            strict=strict,
            min=min,
            max=max,
            allow_none=allow_none,
            description=description,
            metadata=metadata,
            track_called=True,
            use_validator_name=True,
        )
        return self._execute_single(spec)

    @overload
    def _float(
        self,
        default: float,
        *,
        name: str,
        min: Optional[float] = None,
        max: Optional[float] = None,
        strict: bool = False,
        allow_none: Literal[False] = False,
        hpo_spec: "HpoFloat | None" = None,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> float: ...

    @overload
    def _float(
        self,
        default: Optional[float],
        *,
        name: str,
        min: Optional[float] = None,
        max: Optional[float] = None,
        strict: bool = False,
        allow_none: Literal[True],
        hpo_spec: "HpoFloat | None" = None,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[float]: ...

    def _float(
        self,
        default: Optional[float],
        *,
        name: str,
        min: Optional[float] = None,
        max: Optional[float] = None,
        strict: bool = False,
        allow_none: bool = False,
        hpo_spec: "HpoFloat | None" = None,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[float]:
        """Float parameter with optional bounds validation."""
        spec = HP.SingleValueSpec(
            name=name,
            default=default,
            param_type="float",
            validator=FloatValidator(),
            supports_strict=True,
            strict=strict,
            min=min,
            max=max,
            allow_none=allow_none,
            description=description,
            metadata=metadata,
            track_called=True,
            use_validator_name=True,
        )
        return self._execute_single(spec)

    @overload
    def _text(
        self,
        default: str,
        *,
        name: str,
        multiline: bool = False,
        allow_none: Literal[False] = False,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str: ...

    @overload
    def _text(
        self,
        default: Optional[str],
        *,
        name: str,
        multiline: bool = False,
        allow_none: Literal[True],
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]: ...

    def _text(
        self,
        default: Optional[str],
        *,
        name: str,
        multiline: bool = False,
        allow_none: bool = False,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Text parameter."""
        combined_metadata = dict(metadata or {})
        if multiline:
            combined_metadata["multiline"] = True
        spec = HP.SingleValueSpec(
            name=name,
            default=default,
            param_type="text",
            validator=TextValidator(),
            supports_strict=False,
            allow_none=allow_none,
            description=description,
            metadata=combined_metadata if combined_metadata else None,
            track_called=True,
            use_validator_name=True,
        )
        return self._execute_single(spec)

    @overload
    def _bool(
        self,
        default: bool,
        *,
        name: str,
        allow_none: Literal[False] = False,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool: ...

    @overload
    def _bool(
        self,
        default: Optional[bool],
        *,
        name: str,
        allow_none: Literal[True],
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[bool]: ...

    def _bool(
        self,
        default: Optional[bool],
        *,
        name: str,
        allow_none: bool = False,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[bool]:
        """Boolean parameter."""
        spec = HP.SingleValueSpec(
            name=name,
            default=default,
            param_type="bool",
            validator=BoolValidator(),
            supports_strict=False,
            allow_none=allow_none,
            description=description,
            metadata=metadata,
            track_called=True,
            use_validator_name=True,
        )
        return self._execute_single(spec)

    def _select(
        self,
        options: Union[List[Any], Mapping[Any, Any]],
        *,
        name: str,
        default: Any = _NO_DEFAULT,
        options_only: bool = False,
        allow_none: bool = False,
        hpo_spec: "HpoCategorical | None" = None,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Selection parameter from options."""
        spec = HP.SelectSingleSpec(
            name=name,
            options=options,
            default=default,
            options_only=options_only,
            allow_none=allow_none,
            description=description,
            metadata=metadata,
        )
        return self._execute_select_single(spec)

    def _multi_int(
        self,
        default: List[int],
        *,
        name: str,
        min: Optional[int] = None,
        max: Optional[int] = None,
        strict: bool = False,
        allow_none: bool = False,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[int]:
        """Multi-integer parameter with optional bounds validation."""
        spec = HP.MultiValueSpec(
            name=name,
            default=default,
            param_type="multi_int",
            element_validator=IntValidator(),
            supports_strict=True,
            strict=strict,
            min=min,
            max=max,
            allow_none=allow_none,
            description=description,
            metadata=metadata,
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
        allow_none: bool = False,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[float]:
        """Multi-float parameter with optional bounds validation."""
        spec = HP.MultiValueSpec(
            name=name,
            default=default,
            param_type="multi_float",
            element_validator=FloatValidator(),
            supports_strict=True,
            strict=strict,
            min=min,
            max=max,
            allow_none=allow_none,
            description=description,
            metadata=metadata,
        )
        return self._execute_multi(spec)

    def _multi_text(
        self,
        default: List[str],
        *,
        name: str,
        allow_none: bool = False,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """Multi-text parameter."""
        spec = HP.MultiValueSpec(
            name=name,
            default=default,
            param_type="multi_text",
            element_validator=TextValidator(),
            supports_strict=False,
            allow_none=allow_none,
            description=description,
            metadata=metadata,
        )
        return self._execute_multi(spec)

    def _multi_bool(
        self,
        default: List[bool],
        *,
        name: str,
        allow_none: bool = False,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[bool]:
        """Multi-boolean parameter."""
        spec = HP.MultiValueSpec(
            name=name,
            default=default,
            param_type="multi_bool",
            element_validator=BoolValidator(),
            supports_strict=False,
            allow_none=allow_none,
            description=description,
            metadata=metadata,
        )
        return self._execute_multi(spec)

    def _multi_select(
        self,
        options: Union[List[Any], Mapping[Any, Any]],
        *,
        name: str,
        default: Optional[List[Any]] = None,
        options_only: bool = False,
        allow_none: bool = False,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Any]:
        """Multi-selection parameter from options."""
        spec = HP.SelectMultiSpec(
            name=name,
            options=options,
            default=default,
            options_only=options_only,
            allow_none=allow_none,
            description=description,
            metadata=metadata,
        )
        return self._execute_select_multi(spec)

    def _rules(
        self,
        *,
        when: list,
        name: str,
        then: Any,
        combinators: Optional[List[str]] = None,
        default: Optional[list] = None,
        description: Optional[str] = None,
    ) -> list:
        """Rules parameter — a list of WHEN/THEN rules as a config value.

        ``when`` is a list of FieldSpec objects declaring the condition vocabulary.
        ``then`` is a FieldSpec (or type shorthand) for the payload widget.
        ``combinators`` controls the tier: ["and"] for simple, ["and","or","not"] for full.
        """
        full_path = self._get_full_param_path(name)
        validate_identifier_name(name, kind="parameter name")
        self._validate_name_not_called(name)
        self.called_params.add(full_path)

        if combinators is None:
            combinators = ["and"]

        then_spec = self._resolve_then_spec(then)
        field_specs = [fs.to_dict() for fs in when]
        then_specs = [then_spec.to_dict()] if not isinstance(then_spec, list) else [s.to_dict() for s in then_spec]

        value, found = self._get_value_for_param(name)
        rules_value = self._coerce_rules(value if found else (default or []))

        rules_metadata: Dict[str, Any] = {
            "field_specs": field_specs,
            "then_specs": then_specs,
            "combinators": combinators,
        }

        self._record_parameter(
            path=full_path,
            name=name,
            kind="rules",
            default_value=self._rules_to_jsonable(default or []),
            selected_value=self._rules_to_jsonable(rules_value),
            description=description,
            metadata=rules_metadata,
        )

        return rules_value

    @staticmethod
    def _resolve_then_spec(then: Any) -> Any:
        from hypster.field_spec import FieldSpec

        if isinstance(then, FieldSpec):
            if then.name is None:
                raise ValueError("then FieldSpec must have a name, e.g. field.text(name='prompt', multiline=True)")
            return then
        if isinstance(then, list):
            for i, item in enumerate(then):
                if not isinstance(item, FieldSpec):
                    raise TypeError(f"then[{i}]: expected a FieldSpec, got {type(item).__name__}")
                if item.name is None:
                    raise ValueError(f"then[{i}]: FieldSpec must have a name")
            return then
        raise TypeError(
            f"then must be a FieldSpec (e.g. field.text(name='prompt', multiline=True)), got {type(then).__name__}"
        )

    @staticmethod
    def _coerce_rules(raw: list) -> list:
        from hypster.rules import Rule

        result = []
        for i, item in enumerate(raw):
            if isinstance(item, Rule):
                result.append(item)
            elif isinstance(item, dict):
                result.append(Rule.from_dict(item))
            else:
                raise ValueError(f"rules[{i}]: expected a Rule or dict, got {type(item).__name__}")
        return result

    @staticmethod
    def _rules_to_jsonable(rules: list) -> list:
        from hypster.rules import Rule

        return [r.to_dict() if isinstance(r, Rule) else r for r in rules]

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
            "rules": self._rules,
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
        rules = _rules
