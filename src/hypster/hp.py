"""The HP Parameter Interface."""

import builtins
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Literal, Optional, Protocol, Tuple, Union, overload

from ._execution import handle_unknown_parameters, reject_removed_execution_argument_containers
from ._sentinels import NO_DEFAULT as _NO_DEFAULT
from ._sentinels import NOT_PROVIDED
from .field_spec import FieldSpec, resolve_then_spec
from .hp_calls import (
    BoolValidator,
    FloatValidator,
    HPCallError,
    IntValidator,
    MultiValidator,
    SelectValidator,
    TextValidator,
)
from .rules import coerce_rules, rules_to_jsonable
from .schema_field import coerce_schema_fields, schema_fields_to_jsonable
from .utils import (
    normalize_values,
    validate_config_func_signature,
    validate_identifier_name,
    validate_metadata,
    validate_select_choice,
)

if TYPE_CHECKING:  # only for type hints; avoid runtime imports
    from .hpo.types import HpoCategorical, HpoFloat, HpoInt

Number = Union[int, float]


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
        minimum: Optional[Number] = None,
        maximum: Optional[Number] = None,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None: ...

    def record_nest(
        self,
        *,
        path: str,
        name: str,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None: ...


class ValueProvider(Protocol):
    """Supplies a value for a parameter that has no explicit value.

    Consulted for single-value kinds (int/float/text/bool) and select before
    falling back to the parameter's default. Return NOT_PROVIDED to decline;
    the parameter then uses its default. Provided values go through the same
    validation as explicit values.
    """

    def provide_value(
        self,
        *,
        path: str,
        kind: str,
        default: Any,
        allow_none: bool,
        strict: bool,
        options: Optional[List[Any]],
        min: Optional["Number"],
        max: Optional["Number"],
        hpo_spec: Any,
    ) -> Any: ...


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

    def __init__(
        self,
        values: Dict[str, Any],
        parameter_tracker: Optional[ParameterTracker] = None,
        value_provider: Optional[ValueProvider] = None,
    ):
        """HP is created by instantiate() - users don't instantiate directly."""
        self.values = values or {}
        self.parameter_tracker = parameter_tracker
        self.value_provider = value_provider
        self.namespace_stack: List[str] = []  # For nested name prefixes
        self.called_params: set[str] = set()  # Track called parameter names
        self.nested_scopes: set[str] = set()  # Track names that have been nested
        self.nested_scope_paths: set[str] = set()  # Track nest/group paths separately from parameter leaves

    def _get_full_param_path(self, name: str) -> str:
        """Get full parameter path including namespace stack."""
        if self.namespace_stack:
            return ".".join(self.namespace_stack + [name])
        return name

    def _record_parameter(self, **event: Any) -> None:
        if self.parameter_tracker is not None:
            self.parameter_tracker.record_parameter(**event)

    def _record_nest(self, **event: Any) -> None:
        if self.parameter_tracker is not None:
            self.parameter_tracker.record_nest(**event)

    def _get_value_for_param(self, name: str) -> tuple[Any, builtins.bool]:
        """Get value for parameter, returns (value, found)."""
        # Check for exact match with just the name first (for nested contexts)
        if name in self.values:
            return self.values[name], True

        # Then check with full path
        full_path = self._get_full_param_path(name)
        if full_path in self.values:
            return self.values[full_path], True

        return None, False

    def _provide_value(
        self,
        *,
        path: str,
        kind: str,
        default: Any,
        allow_none: builtins.bool = False,
        strict: builtins.bool = False,
        options: Optional[List[Any]] = None,
        min: Optional[Number] = None,
        max: Optional[Number] = None,
        hpo_spec: Any = None,
    ) -> Any:
        if self.value_provider is None:
            return NOT_PROVIDED
        return self.value_provider.provide_value(
            path=path,
            kind=kind,
            default=default,
            allow_none=allow_none,
            strict=strict,
            options=options,
            min=min,
            max=max,
            hpo_spec=hpo_spec,
        )

    def _register_param(self, name: Optional[str]) -> str:
        """Validate a parameter name, register it as called, and return its full path.

        Name validation must run before path construction: joining the
        namespace stack with a non-string name would raise a raw TypeError.
        """
        if name is None:
            raise HPCallError("<unnamed>", "requires 'name' for overrides. How to fix: pass name='...'")
        validate_identifier_name(name, kind="parameter name")
        full_path = self._get_full_param_path(name)
        if full_path in self.called_params:
            raise HPCallError(full_path, "has already been defined")
        self.called_params.add(full_path)
        return full_path

    def _require_default(self, default: Any, full_path: str, param_type: str, name: str) -> None:
        if default is _NO_DEFAULT:
            raise HPCallError(
                full_path,
                f"requires a default value as its first argument. "
                f"How to fix: hp.{param_type}(<default>, name={name!r})",
            )

    # --- Common handlers ---
    def _handle_single_value(
        self,
        *,
        default: Any,
        name: str,
        param_type: str,
        validator: Any,
        strict: builtins.bool = False,
        min: Optional[Number] = None,
        max: Optional[Number] = None,
        allow_none: builtins.bool = False,
        hpo_spec: Any = None,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Any:
        full_path = self._register_param(name)
        self._require_default(default, full_path, param_type, name)

        if default is None and not allow_none:
            raise HPCallError(
                full_path,
                "default=None requires allow_none=True. How to fix: pass allow_none=True, or use a non-None default.",
            )

        value, found = self._get_value_for_param(name)
        if not found:
            provided = self._provide_value(
                path=full_path,
                kind=param_type,
                default=default,
                allow_none=allow_none,
                strict=strict,
                min=min,
                max=max,
                hpo_spec=hpo_spec,
            )
            if provided is not NOT_PROVIDED:
                value, found = provided, True
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
            validated_value = validator.validate_value(raw_value, full_path, strict=strict)

        if validated_value is not None and (min is not None or max is not None):
            validator.validate_bounds(validated_value, min, max, full_path)

        metadata = validate_metadata(metadata, param_path=full_path)
        self._record_parameter(
            path=full_path,
            name=name,
            kind=param_type,
            default_value=default,
            selected_value=validated_value,
            options=None,
            minimum=min,
            maximum=max,
            description=description,
            metadata=metadata,
        )

        return validated_value

    def _handle_multi_value(
        self,
        *,
        default: Any,
        name: str,
        param_type: str,
        element_validator: Any,
        strict: builtins.bool = False,
        min: Optional[Number] = None,
        max: Optional[Number] = None,
        allow_none: builtins.bool = False,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Any]:
        full_path = self._register_param(name)
        self._require_default(default, full_path, param_type, name)

        if allow_none:
            raise HPCallError(
                full_path,
                f"allow_none=True is not supported for hp.{param_type}() yet. "
                "How to fix: remove allow_none=True, or use hp.multi_select([...], allow_none=True) "
                "when you need nullable categorical choices.",
            )

        multi_validator = MultiValidator(element_validator)
        value, found = self._get_value_for_param(name)
        validated_values = multi_validator.validate_value(value if found else default, full_path, strict=strict)

        if min is not None or max is not None:
            multi_validator.validate_bounds(validated_values, min, max, full_path)

        metadata = validate_metadata(metadata, param_path=full_path)
        self._record_parameter(
            path=full_path,
            name=name,
            kind=param_type,
            default_value=default,
            selected_value=validated_values,
            options=None,
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
        options_only: builtins.bool = False,
        allow_none: builtins.bool = False,
        hpo_spec: Any = None,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Any:
        full_path = self._register_param(name)

        is_mapping = isinstance(options, Mapping)
        adapter = OptionsAdapter(options=options)
        option_keys, option_map = adapter.keys_and_map()
        for index, option in enumerate(option_keys):
            validate_select_choice(option, param_path=f"{full_path} option #{index}", allow_none=allow_none)
        actual_default = adapter.resolve_default(default)

        value, found = self._get_value_for_param(name)
        if not found:
            provided = self._provide_value(
                path=full_path,
                kind="select",
                default=actual_default,
                allow_none=allow_none,
                options=option_keys,
                hpo_spec=hpo_spec,
            )
            if provided is not NOT_PROVIDED:
                value, found = provided, True
        if found:
            chosen = value
        else:
            if actual_default is None and default is _NO_DEFAULT and not option_keys and not allow_none:
                raise HPCallError(
                    full_path,
                    "select has no options and no default. "
                    "How to fix: provide at least one option, pass default=..., or pass allow_none=True.",
                )
            chosen = actual_default

        validate_select_choice(chosen, param_path=full_path, allow_none=allow_none)
        validated_key = SelectValidator().validate_value(chosen, option_keys, options_only, full_path)
        metadata = validate_metadata(metadata, param_path=full_path)
        self._record_parameter(
            path=full_path,
            name=name,
            kind="select",
            default_value=actual_default,
            selected_value=validated_key,
            options=option_keys,
            minimum=None,
            maximum=None,
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
        options_only: builtins.bool = False,
        allow_none: builtins.bool = False,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Any]:
        full_path = self._register_param(name)

        is_mapping = isinstance(options, Mapping)
        adapter = OptionsAdapter(options=options)
        option_keys, option_map = adapter.keys_and_map()
        for index, option in enumerate(option_keys):
            validate_select_choice(option, param_path=f"{full_path} option #{index}", allow_none=allow_none)
        actual_default = list(default or [])

        value, found = self._get_value_for_param(name)
        chosen_list = value if found else actual_default
        if not isinstance(chosen_list, list):
            raise HPCallError(full_path, f"expected list but got {type(chosen_list).__name__} ({chosen_list})")

        validator = SelectValidator()
        validated_keys = []
        for i, item in enumerate(chosen_list):
            validate_select_choice(item, param_path=f"{full_path}[{i}]", allow_none=allow_none)
            validated_keys.append(validator.validate_value(item, option_keys, options_only, f"{full_path}[{i}]"))

        metadata = validate_metadata(metadata, param_path=full_path)
        self._record_parameter(
            path=full_path,
            name=name,
            kind="multi_select",
            default_value=actual_default,
            selected_value=validated_keys,
            options=option_keys,
            minimum=None,
            maximum=None,
            description=description,
            metadata=metadata,
        )
        return [option_map.get(k, k) for k in validated_keys] if is_mapping else validated_keys

    # Composition
    def nest(
        self,
        child: Callable,
        *,
        name: str,
        values: Optional[Dict[str, Any]] = None,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """Nest another configuration function."""
        # Validate the name before building the path: joining the namespace
        # stack with a non-string name would raise a raw TypeError.
        if name is None:
            raise HPCallError("<unnamed>", "requires 'name' for nesting")
        validate_identifier_name(name, kind="nest name")
        full_path = self._get_full_param_path(name)

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

        metadata = validate_metadata(metadata, param_path=full_path)

        # Create new HP instance for nested call with namespace
        self._record_nest(path=full_path, name=name, description=description, metadata=metadata)

        nested_hp = self.__class__(
            nested_values,
            parameter_tracker=self.parameter_tracker,
            value_provider=self.value_provider,
        )
        nested_hp.namespace_stack = self.namespace_stack + [name]
        nested_hp.called_params = self.called_params  # Share called_params tracking
        nested_hp.nested_scope_paths = self.nested_scope_paths

        # Mark that we've nested under this name
        self.nested_scopes.add(name)
        self.called_params.add(full_path)
        self.nested_scope_paths.add(full_path)

        reject_removed_execution_argument_containers(kwargs)

        # Call the nested function
        result = child(nested_hp, **kwargs)

        if explicit_values:
            prefixed_explicit_values = {f"{full_path}.{key}": value for key, value in explicit_values.items()}
            leaf_params = self.called_params - self.nested_scope_paths
            handle_unknown_parameters(prefixed_explicit_values, leaf_params, "raise")

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
    # Method names deliberately shadow builtins inside this class body, so
    # annotations below must spell builtins.int / builtins.float / builtins.bool.

    @overload
    def int(
        self,
        default: builtins.int,
        *,
        name: str,
        min: Optional[builtins.int] = None,
        max: Optional[builtins.int] = None,
        strict: builtins.bool = False,
        allow_none: Literal[False] = False,
        hpo_spec: "HpoInt | None" = None,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> builtins.int: ...

    @overload
    def int(
        self,
        default: Optional[builtins.int],
        *,
        name: str,
        min: Optional[builtins.int] = None,
        max: Optional[builtins.int] = None,
        strict: builtins.bool = False,
        allow_none: Literal[True],
        hpo_spec: "HpoInt | None" = None,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[builtins.int]: ...

    def int(
        self,
        default: Any = _NO_DEFAULT,
        *,
        name: str,
        min: Optional[builtins.int] = None,
        max: Optional[builtins.int] = None,
        strict: builtins.bool = False,
        allow_none: builtins.bool = False,
        hpo_spec: "HpoInt | None" = None,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[builtins.int]:
        """Integer parameter with optional bounds validation."""
        return self._handle_single_value(
            default=default,
            name=name,
            param_type="int",
            validator=IntValidator(),
            strict=strict,
            min=min,
            max=max,
            allow_none=allow_none,
            hpo_spec=hpo_spec,
            description=description,
            metadata=metadata,
        )

    @overload
    def float(
        self,
        default: builtins.float,
        *,
        name: str,
        min: Optional[builtins.float] = None,
        max: Optional[builtins.float] = None,
        strict: builtins.bool = False,
        allow_none: Literal[False] = False,
        hpo_spec: "HpoFloat | None" = None,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> builtins.float: ...

    @overload
    def float(
        self,
        default: Optional[builtins.float],
        *,
        name: str,
        min: Optional[builtins.float] = None,
        max: Optional[builtins.float] = None,
        strict: builtins.bool = False,
        allow_none: Literal[True],
        hpo_spec: "HpoFloat | None" = None,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[builtins.float]: ...

    def float(
        self,
        default: Any = _NO_DEFAULT,
        *,
        name: str,
        min: Optional[builtins.float] = None,
        max: Optional[builtins.float] = None,
        strict: builtins.bool = False,
        allow_none: builtins.bool = False,
        hpo_spec: "HpoFloat | None" = None,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[builtins.float]:
        """Float parameter with optional bounds validation."""
        return self._handle_single_value(
            default=default,
            name=name,
            param_type="float",
            validator=FloatValidator(),
            strict=strict,
            min=min,
            max=max,
            allow_none=allow_none,
            hpo_spec=hpo_spec,
            description=description,
            metadata=metadata,
        )

    @overload
    def text(
        self,
        default: str,
        *,
        name: str,
        multiline: builtins.bool = False,
        allow_none: Literal[False] = False,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str: ...

    @overload
    def text(
        self,
        default: Optional[str],
        *,
        name: str,
        multiline: builtins.bool = False,
        allow_none: Literal[True],
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]: ...

    def text(
        self,
        default: Any = _NO_DEFAULT,
        *,
        name: str,
        multiline: builtins.bool = False,
        allow_none: builtins.bool = False,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Text parameter."""
        if metadata is not None and not isinstance(metadata, Mapping):
            raise ValueError(
                f"Parameter '{name}': metadata must be a dictionary with string keys and JSON-compatible values."
            )
        combined_metadata = dict(metadata or {})
        if multiline:
            combined_metadata["multiline"] = True
        return self._handle_single_value(
            default=default,
            name=name,
            param_type="text",
            validator=TextValidator(),
            allow_none=allow_none,
            description=description,
            metadata=combined_metadata if combined_metadata else None,
        )

    @overload
    def bool(
        self,
        default: builtins.bool,
        *,
        name: str,
        allow_none: Literal[False] = False,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> builtins.bool: ...

    @overload
    def bool(
        self,
        default: Optional[builtins.bool],
        *,
        name: str,
        allow_none: Literal[True],
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[builtins.bool]: ...

    def bool(
        self,
        default: Any = _NO_DEFAULT,
        *,
        name: str,
        allow_none: builtins.bool = False,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[builtins.bool]:
        """Boolean parameter."""
        return self._handle_single_value(
            default=default,
            name=name,
            param_type="bool",
            validator=BoolValidator(),
            allow_none=allow_none,
            description=description,
            metadata=metadata,
        )

    def select(
        self,
        options: Union[List[Any], Mapping[Any, Any]],
        *,
        name: str,
        default: Any = _NO_DEFAULT,
        options_only: builtins.bool = False,
        allow_none: builtins.bool = False,
        hpo_spec: "HpoCategorical | None" = None,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Selection parameter from options."""
        return self._handle_select_single(
            options=options,
            name=name,
            default=default,
            options_only=options_only,
            allow_none=allow_none,
            hpo_spec=hpo_spec,
            description=description,
            metadata=metadata,
        )

    def multi_int(
        self,
        default: Any = _NO_DEFAULT,
        *,
        name: str,
        min: Optional[builtins.int] = None,
        max: Optional[builtins.int] = None,
        strict: builtins.bool = False,
        allow_none: builtins.bool = False,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[builtins.int]:
        """Multi-integer parameter with optional bounds validation."""
        return self._handle_multi_value(
            default=default,
            name=name,
            param_type="multi_int",
            element_validator=IntValidator(),
            strict=strict,
            min=min,
            max=max,
            allow_none=allow_none,
            description=description,
            metadata=metadata,
        )

    def multi_float(
        self,
        default: Any = _NO_DEFAULT,
        *,
        name: str,
        min: Optional[builtins.float] = None,
        max: Optional[builtins.float] = None,
        strict: builtins.bool = False,
        allow_none: builtins.bool = False,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[builtins.float]:
        """Multi-float parameter with optional bounds validation."""
        return self._handle_multi_value(
            default=default,
            name=name,
            param_type="multi_float",
            element_validator=FloatValidator(),
            strict=strict,
            min=min,
            max=max,
            allow_none=allow_none,
            description=description,
            metadata=metadata,
        )

    def multi_text(
        self,
        default: Any = _NO_DEFAULT,
        *,
        name: str,
        allow_none: builtins.bool = False,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """Multi-text parameter."""
        return self._handle_multi_value(
            default=default,
            name=name,
            param_type="multi_text",
            element_validator=TextValidator(),
            allow_none=allow_none,
            description=description,
            metadata=metadata,
        )

    def multi_bool(
        self,
        default: Any = _NO_DEFAULT,
        *,
        name: str,
        allow_none: builtins.bool = False,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[builtins.bool]:
        """Multi-boolean parameter."""
        return self._handle_multi_value(
            default=default,
            name=name,
            param_type="multi_bool",
            element_validator=BoolValidator(),
            allow_none=allow_none,
            description=description,
            metadata=metadata,
        )

    def multi_select(
        self,
        options: Union[List[Any], Mapping[Any, Any]],
        *,
        name: str,
        default: Optional[List[Any]] = None,
        options_only: builtins.bool = False,
        allow_none: builtins.bool = False,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Any]:
        """Multi-selection parameter from options."""
        return self._handle_select_multi(
            options=options,
            name=name,
            default=default,
            options_only=options_only,
            allow_none=allow_none,
            description=description,
            metadata=metadata,
        )

    def rules(
        self,
        *,
        when: list,
        name: str,
        then: Any,
        combinators: Optional[List[str]] = None,
        default: Optional[list] = None,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> list:
        """Rules parameter — a list of WHEN/THEN rules as a config value.

        ``when`` is a list of FieldSpec objects declaring the condition vocabulary.
        ``then`` is a FieldSpec (or type shorthand) for the payload widget.
        ``combinators`` controls the tier: ["and"] for simple, ["and","or","not"] for full.
        """
        full_path = self._register_param(name)

        if combinators is None:
            combinators = ["and"]

        if not isinstance(when, list):
            raise TypeError(f"when must be a list, got {type(when).__name__}")
        for i, fs in enumerate(when):
            if not isinstance(fs, FieldSpec):
                raise TypeError(f"when[{i}]: expected a FieldSpec, got {type(fs).__name__}")

        then_spec = resolve_then_spec(then)
        field_specs = [fs.to_dict() for fs in when]
        then_specs = [then_spec.to_dict()] if not isinstance(then_spec, list) else [s.to_dict() for s in then_spec]

        value, found = self._get_value_for_param(name)
        rules_value = coerce_rules(value if found else (default or []))

        metadata = validate_metadata(metadata, param_path=full_path)
        rules_metadata: Dict[str, Any] = {
            "field_specs": field_specs,
            "then_specs": then_specs,
            "combinators": combinators,
        }
        if metadata:
            rules_metadata.update(metadata)

        self._record_parameter(
            path=full_path,
            name=name,
            kind="rules",
            default_value=rules_to_jsonable(default or []),
            selected_value=rules_to_jsonable(rules_value),
            options=None,
            minimum=None,
            maximum=None,
            description=description,
            metadata=rules_metadata,
        )

        return rules_value

    def schema(
        self,
        *,
        name: str,
        default: Optional[list] = None,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> list:
        """Schema parameter — a list of SchemaField extraction field definitions."""
        full_path = self._register_param(name)

        value, found = self._get_value_for_param(name)
        schema_value = coerce_schema_fields(value if found else (default or []))

        metadata = validate_metadata(metadata, param_path=full_path)
        # "schema_fields", not "field_specs": CONTEXT.md reserves "Field Spec"
        # for the rules condition/payload vocabulary from hypster.field.
        schema_metadata: Dict[str, Any] = {
            "schema_fields": [f.to_dict() for f in schema_value],
        }
        if metadata:
            schema_metadata.update(metadata)

        self._record_parameter(
            path=full_path,
            name=name,
            kind="schema",
            default_value=schema_fields_to_jsonable(default or []),
            selected_value=schema_fields_to_jsonable(schema_value),
            options=None,
            minimum=None,
            maximum=None,
            description=description,
            metadata=schema_metadata,
        )

        return schema_value
