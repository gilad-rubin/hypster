"""Parameter validators and processors for HP calls."""

from typing import Any, List, Optional, Union


class HPCallError(ValueError):
    """Base exception for parameter validation errors with context."""

    def __init__(self, param_path: str, message: str):
        # param_path includes nesting: "model.optimizer.lr"
        super().__init__(f"Parameter '{param_path}': {message}")


class ParameterValidator:
    """Base class for parameter validation."""

    def validate_name(self, name: Optional[str], param_path: str) -> None:
        """Validate that name is provided for overrides."""
        if name is None:
            raise HPCallError(param_path, "requires 'name' for overrides. Example: hp.int(10, name='batch_size')")

    def validate_value(self, value: Any, param_path: str) -> Any:
        """Process and validate input value."""
        raise NotImplementedError

    def validate_bounds(
        self,
        value: Union[int, float],
        min_val: Optional[Union[int, float]],
        max_val: Optional[Union[int, float]],
        param_path: str,
    ) -> None:
        """Validate numeric bounds."""
        if min_val is not None and value < min_val:
            if max_val is not None:
                raise HPCallError(
                    param_path,
                    f"value {value} is below minimum bound {min_val}. Value must be in range [{min_val}, {max_val}]",
                )
            else:
                raise HPCallError(param_path, f"value {value} is below minimum bound {min_val}")

        if max_val is not None and value > max_val:
            if min_val is not None:
                raise HPCallError(
                    param_path,
                    f"value {value} exceeds maximum bound {max_val}. Value must be in range [{min_val}, {max_val}]",
                )
            else:
                raise HPCallError(param_path, f"value {value} exceeds maximum bound {max_val}")


class IntValidator(ParameterValidator):
    """Validates int parameters with optional type conversion."""

    def validate_value(self, value: Any, param_path: str, strict: bool = False) -> int:
        if isinstance(value, float):
            if strict:
                raise HPCallError(param_path, f"expected int but got float ({value}). Use an integer value.")
            if value != int(value):
                raise HPCallError(
                    param_path,
                    f"float {value} would lose precision when converted to int. "
                    f"Use {int(value)} or allow precision loss explicitly.",
                )
            return int(value)
        if not isinstance(value, int):
            raise HPCallError(param_path, f"expected int but got {type(value).__name__} ({value})")
        return value


class FloatValidator(ParameterValidator):
    """Validates float parameters with optional type conversion."""

    def validate_value(self, value: Any, param_path: str, strict: bool = False) -> float:
        if isinstance(value, int):
            if strict or "." in param_path:
                raise HPCallError(
                    param_path,
                    f"expected float but got int ({value}). Please provide a float value like {float(value)}",
                )
            return float(value)
        if not isinstance(value, float):
            raise HPCallError(param_path, f"expected float but got {type(value).__name__} ({value})")
        return value


class TextValidator(ParameterValidator):
    """Validates text parameters."""

    def validate_value(self, value: Any, param_path: str) -> str:
        if not isinstance(value, str):
            raise HPCallError(param_path, f"expected string but got {type(value).__name__} ({value})")
        return value


class BoolValidator(ParameterValidator):
    """Validates boolean parameters."""

    def validate_value(self, value: Any, param_path: str) -> bool:
        if not isinstance(value, bool):
            raise HPCallError(param_path, f"expected boolean but got {type(value).__name__} ({value})")
        return value


class SelectValidator:
    """Validates selection from options."""

    def validate_name(self, name: Optional[str], param_path: str) -> None:
        """Validate that name is provided for overrides."""
        if name is None:
            raise HPCallError(
                param_path, "requires 'name' for overrides. Example: hp.select(['a', 'b'], name='choice')"
            )

    def validate_value(self, value: Any, options: List[Any], options_only: bool, param_path: str) -> Any:
        if options_only and value not in options:
            # Show available options
            options_str = ", ".join(repr(o) for o in options[:5])
            if len(options) > 5:
                options_str += f", ... ({len(options) - 5} more)"
            raise HPCallError(param_path, f"'{value}' not in allowed options. Available: [{options_str}]")
        # Note: if options_only=False, any value is allowed
        return value


class MultiValidator:
    """Validates multi-value parameters."""

    def __init__(self, element_validator: ParameterValidator):
        self.element_validator = element_validator

    def validate_value(self, value: Any, param_path: str, **kwargs: Any) -> List[Any]:
        if not isinstance(value, list):
            raise HPCallError(param_path, f"expected list but got {type(value).__name__} ({value})")

        result = []
        for i, item in enumerate(value):
            try:
                validated_item = self.element_validator.validate_value(item, f"{param_path}[{i}]", **kwargs)
                result.append(validated_item)
            except HPCallError as e:
                # Re-raise with list context
                raise HPCallError(param_path, f"invalid item at index {i}: {str(e).split(': ', 1)[1]}")

        return result

    def validate_bounds(
        self,
        values: List[Union[int, float]],
        min_val: Optional[Union[int, float]],
        max_val: Optional[Union[int, float]],
        param_path: str,
    ) -> None:
        """Validate bounds for each element in the list."""
        for i, value in enumerate(values):
            try:
                self.element_validator.validate_bounds(value, min_val, max_val, f"{param_path}[{i}]")
            except HPCallError as e:
                # Re-raise with list context
                raise HPCallError(param_path, f"invalid item at index {i}: {str(e).split(': ', 1)[1]}")
