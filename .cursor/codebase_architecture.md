# Hypster v2 Codebase Architecture

## Overview

Hypster v2 is a complete rewrite that removes all AST manipulation, automatic naming, registries, and implicit behaviors in favor of explicit, typed Python functions. The new architecture is built around three core principles:

1. **Pure functions**: Config functions are plain Python functions with `hp: HP` as the first parameter
2. **Explicit naming**: All overrideable parameters must have explicit `name` arguments
3. **Pass-through semantics**: What you return is what you get - no runtime filtering

## Core Modules

### 1. `hp.py` - The HP Parameter Interface

The `HP` class is instantiated by the framework and passed to config functions. It provides typed parameter methods that record calls and apply overrides.

**Key responsibilities:**
- Provide typed parameter methods (`int`, `float`, `text`, `bool`, `select`, `multi_*` variants)
- Apply value overrides from the `values` dict based on explicit names
- Handle nested configuration composition via `hp.nest()`
- Provide `hp.collect()` helper for ergonomic locals collection
- Track parameter calls for exploration (delegated to separate ExplorationTracker)

**Core methods:**
```python
class HP:
    def __init__(self, values: dict, exploration_tracker: Optional[ExplorationTracker] = None):
        """HP is created by instantiate() - users don't instantiate directly"""
        self.values = values
        self.exploration_tracker = exploration_tracker  # Optional, only for explore mode
        self.namespace_stack: list[str] = []  # For nested name prefixes
        self.called_params: set[str] = set()  # Track called parameter names

    # Single-value parameters (all require name for overrides)
    def int(self, default: int, *, name: str, min: int = None, max: int = None, strict: bool = False) -> int
    def float(self, default: float, *, name: str, min: float = None, max: float = None, strict: bool = False) -> float
    def text(self, default: str, *, name: str) -> str
    def bool(self, default: bool, *, name: str) -> bool
    def select(self, options: list|dict, *, name: str, default: Any = None, options_only: bool = False) -> Any

    # Multi-value parameters
    def multi_int(self, default: list[int], *, name: str, min: int = None, max: int = None, strict: bool = False) -> list[int]
    def multi_float(self, default: list[float], *, name: str, min: float = None, max: float = None, strict: bool = False) -> list[float]
    def multi_text(self, default: list[str], *, name: str) -> list[str]
    def multi_bool(self, default: list[bool], *, name: str) -> list[bool]
    def multi_select(self, options: list|dict, *, name: str, default: list = None, options_only: bool = False) -> list

    # Composition
    def nest(self, child: Callable, *, name: str, values: dict = None, args: tuple = (), kwargs: dict = None) -> Any

    # Helpers
    def collect(self, locals_dict: dict, include: list[str] = None, exclude: list[str] = None) -> dict
```

**Design decisions:**
- `name` is mandatory - raises `HPCallError` if missing
- **No `hp.number()`** - use `hp.int()` or `hp.float()` for clear type expectations
- **`strict` parameter** for `int`/`float` controls type conversion:
  - `strict=False` (default): Allow conversion if no precision loss (`64.0 → 64`, `1 → 1.0`)
  - `strict=True`: Reject any type mismatch
- **Name collision detection**: HP tracks parameter names called during execution
  - Same name in different conditional branches: ✅ Allowed (only one branch executes)
  - Same name in same execution path: ❌ Error (duplicate definition)
  - Example:
    ```python
    # This is OK - only one branch executes
    if condition:
        value = hp.int(10, name='value')
    else:
        value = hp.float(10.0, name='value')

    # This is NOT OK - both calls execute
    x = hp.int(1, name='param')
    y = hp.int(2, name='param')  # Error: Parameter 'param' already defined
    ```
- All validation happens at call time with human-readable errors:
  ```python
  # Example error messages:
  "Parameter 'n_estimators': default value 3 is outside the defined range [10, 100]. Consider changing the default to be within [10, 100] or adjusting the range."
  "Parameter 'learning_rate': expected float but got int (1). Please provide a float value like 1.0"
  ```
- `hp.nest()` validates that child is callable with correct signature:
  ```python
  # Validates first parameter is named 'hp' and has HP type hint
  "hp.nest() requires a config function with signature (hp: HP, ...). Got: (config: Config)"
  ```
- Exploration tracking is separate from core HP functionality

### 2. `core.py` - The Core API

Provides the main entry points for instantiating configurations.

**Key functions:**

```python
from typing import TypeVar, Callable, Any

T = TypeVar('T')
ConfigFunc = Callable[[HP, ...], T]

def instantiate(
    func: ConfigFunc[T],
    *,
    values: dict[str, Any] | None = None,
    args: tuple[Any, ...] = (),
    kwargs: dict[str, Any] | None = None,
    on_unknown: str = 'warn'
) -> T:
    """
    Execute a config function with the given values.

    Args:
        func: Config function with first param hp: HP
        values: Parameter values by name
        args: Additional positional arguments for func
        kwargs: Additional keyword arguments for func
        on_unknown: How to handle unknown/unreachable parameters:
            - 'warn': Issue warning and continue (default)
            - 'raise': Raise ValueError
            - 'ignore': Silently ignore unknown parameters

    Returns:
        Whatever the config function returns

    Raises:
        ValueError: If func doesn't have hp: HP as first parameter
        ValueError: If unknown parameters in values and on_unknown='raise'
    """
```

**Implementation notes:**
- `instantiate()` creates an HP instance, calls the function, returns result as-is
- No AST manipulation - functions execute normally
- No automatic filtering of return values
- Validates function signature before execution

### 3. `hp_calls.py` - Parameter Validators and Processors

Contains the validation logic for each parameter type. Uses manual validation for better error messages.

**Key classes:**

```python
class HPCallError(ValueError):
    """Base exception for parameter validation errors with context"""
    def __init__(self, param_path: str, message: str):
        # param_path includes nesting: "model.optimizer.lr"
        super().__init__(f"Parameter '{param_path}': {message}")

class ParameterValidator:
    """Base class for parameter validation"""
    def validate_name(self, name: str | None, param_path: str) -> None
    def validate_value(self, value: Any, param_path: str) -> Any
    def validate_bounds(self, value: NumericType, min: NumericType, max: NumericType, param_path: str) -> None

class IntValidator(ParameterValidator):
    """Validates int parameters with optional type conversion"""
    def validate_value(self, value: Any, param_path: str, strict: bool = False) -> int:
        if isinstance(value, float):
            if strict:
                raise HPCallError(param_path, f"expected int but got float ({value}). Use an integer value.")
            if value != int(value):
                raise HPCallError(param_path, f"float {value} would lose precision when converted to int. Use {int(value)} or allow precision loss explicitly.")
            return int(value)
        if not isinstance(value, int):
            raise HPCallError(param_path, f"expected int but got {type(value).__name__} ({value})")
        return value

class FloatValidator(ParameterValidator):
    """Validates float parameters with optional type conversion"""
    def validate_value(self, value: Any, param_path: str, strict: bool = False) -> float:
        if isinstance(value, int):
            if strict:
                raise HPCallError(param_path, f"expected float but got int ({value}). Please provide a float value like {float(value)}")
            return float(value)
        if not isinstance(value, float):
            raise HPCallError(param_path, f"expected float but got {type(value).__name__} ({value})")
        return value

class SelectValidator(ParameterValidator):
    """Validates selection from options"""
    def validate_value(self, value: Any, options: list, options_only: bool, param_path: str) -> Any:
        if options_only and value not in options:
            # Show available options
            options_str = ", ".join(repr(o) for o in options[:5])
            if len(options) > 5:
                options_str += f", ... ({len(options)-5} more)"
            raise HPCallError(param_path, f"'{value}' not in allowed options. Available: [{options_str}]")
        # Note: if options_only=False, any value is allowed
        ...

# etc for other types...
```

**Validation rules with full path context:**

- Missing `name`:
  ```python
  HPCallError("Parameter requires 'name' for overrides. Example: hp.int(10, name='batch_size')")
  ```

- Duplicate parameter names in same execution path:
  ```python
  # When hp.int(1, name='param') is called twice
  HPCallError("Parameter 'param' has already been defined in this execution path")
  ```

- Type mismatches with nested context:
  ```python
  # For hp.float(0.5, name='lr') with nested path 'optimizer.lr'
  instantiate(config, values={'optimizer.lr': 1})
  # Raises: HPCallError("Parameter 'optimizer.lr': expected float but got int (1). Please provide a float value like 1.0")
  ```

- Unknown or unreachable parameters:
  ```python
  instantiate(config, values={'tmep': 0.5, 'rf.n_trees': 100})
  # When model_type='logistic' is selected:
  # Raises: ValueError("Unknown or unreachable parameters:
  #   - 'tmep': Did you mean 'temp'? (similarity: 75%)
  #   - 'rf.n_trees': This parameter exists but is only reachable when model_type='rf' (current: 'logistic')")
  ```

- Bounds violations with context:
  ```python
  # For hp.float(0.5, name='lr', min=0.0, max=1.0)
  instantiate(config, values={'lr': 1.5})
  # Raises: HPCallError("Parameter 'lr': value 1.5 exceeds maximum bound 1.0. Value must be in range [0.0, 1.0]")
  ```

- Select with options_only:
  ```python
  # For hp.select(['gpt-4o', 'haiku'], name='model', options_only=True)
  instantiate(config, values={'model': 'gpt-5'})
  # Raises: HPCallError("Parameter 'model': 'gpt-5' not in allowed options. Available: ['gpt-4o', 'haiku']")

  # Note: options_only=False allows any value
  hp.select(['gpt-4o', 'haiku'], name='model', options_only=False)
  instantiate(config, values={'model': 'custom-model'})  # OK - returns 'custom-model'
  ```

### 4. `exploration.py` - Parameter Discovery and Tracking

Separate module for exploration/discovery functionality, keeping core HP clean.

```python
class ExplorationTracker:
    """Tracks parameter calls during exploration mode"""
    def __init__(self):
        self.discovered_params: OrderedDict[str, ParameterInfo] = OrderedDict()
        self.call_order: list[str] = []
        self.potential_values: dict[str, list[Any]] = {}

    def record_call(self, name: str, param_info: ParameterInfo) -> None:
        """Record a parameter call during exploration"""
        self.discovered_params[name] = param_info
        self.call_order.append(name)

    def get_next_unknown(self, known_values: dict[str, Any]) -> str | None:
        """Get the next parameter that hasn't been set yet"""
        for name in self.call_order:
            if name not in known_values:
                return name
        return None
```

### 5. `utils.py` - Utilities

Helper functions for error messages, similarity matching, etc.

```python
def suggest_similar_names(unknown: str, known: list[str], threshold: float = 0.6) -> list[tuple[str, float]]:
    """Find similar parameter names with similarity scores"""

def format_error_with_suggestions(unknown_params: dict[str, Any], suggestions: dict[str, list[str]], reachability: dict[str, str]) -> str:
    """
    Format helpful error messages distinguishing between:
    - Typos (similar names exist)
    - Unreachable parameters (exist but not in current conditional path)
    - Truly unknown parameters
    """

def merge_nested_dicts(dotted: dict, nested: dict) -> tuple[dict, list[str]]:
    """
    Merge dotted keys and nested dicts (nested takes precedence).
    Returns merged dict and list of conflict warnings.
    """

def validate_config_func_signature(func: Callable) -> None:
    """
    Validate that func has hp: HP as first parameter.
    Raises ValueError with helpful message if not.
    """
```

## Removed Components

The following v1 components are completely removed:

- **AST analyzer** - No AST manipulation, functions execute normally
- **Auto-naming injection** - Names must be explicit
- **Registry** - No string resolution, use callables directly
- **Save/load** - Users handle code organization with normal Python
- **Runtime filtering** - Return values pass through as-is
- **exec() execution** - Functions run directly
- **`hp.number()`** - Use `hp.int()` or `hp.float()` for clear type expectations

## Key Behavioral Changes

### 1. Explicit Returns Required

```python
# v1 - implicit locals collection
def config_v1(hp):
    model = hp.select(['a', 'b'])
    lr = hp.float(0.1)
    # No return - Hypster collected locals

# v2 - explicit return required
def config_v2(hp: HP) -> dict:
    model = hp.select(['a', 'b'], name='model')
    lr = hp.float(0.1, name='lr')
    return {'model': model, 'lr': lr}  # Explicit return
```

### 2. Explicit Naming Required

```python
# v1 - auto-naming from assignment
def config_v1(hp):
    model = hp.select(['a', 'b'])  # Auto-named as 'model'

# v2 - explicit name required
def config_v2(hp: HP):
    model = hp.select(['a', 'b'], name='model')  # Must specify name
    temp = hp.float(0.7, name='temperature')  # Variable name ≠ parameter name
```

### 3. Callable-Only Nesting

```python
# v1 - string/path resolution
def parent_v1(hp):
    child = hp.nest('path/to/child.py')
    child = hp.nest('registry_alias')

# v2 - callables only + imports are allowed outside of the function
from configs import child_config

def parent_v2(hp: HP):
    result = hp.nest(child_config, name='child')  # Direct callable
```

### 4. Pass-Through Returns

```python
# v2 examples of different return types

# Single object (best IDE support)
def model_config(hp: HP) -> RandomForestClassifier:
    n = hp.int(100, name='n_estimators')
    return RandomForestClassifier(n_estimators=n)

# Typed container
@dataclass
class TrainerConfig:
    lr: float
    epochs: int

def trainer_config(hp: HP) -> TrainerConfig:
    return TrainerConfig(
        lr=hp.float(0.1, name='lr'),
        epochs=hp.int(10, name='epochs')
    )

# Plain dict
def dict_config(hp: HP) -> dict:
    return {
        'model': hp.select(['a', 'b'], name='model'),
        'lr': hp.float(0.1, name='lr')
    }
```

## Error Handling Strategy

The `on_unknown` parameter in `instantiate()` provides flexible error handling:

```python
# Warning mode (default) - log and continue
import warnings
with warnings.catch_warnings(record=True) as w:
    result = instantiate(config, values={'unknown_param': 1})  # on_unknown='warn' by default
    # Result uses default values, warning issued
    assert len(w) == 1
    assert "unknown_param" in str(w[0].message)

# Strict mode - fail fast
try:
    result = instantiate(config, values={'typo_param': 1, 'unreachable_param': 2}, on_unknown='raise')
except ValueError as e:
    print(e)
    # "Unknown or unreachable parameters:
    #   - 'typo_param': Did you mean 'temp_param'? (similarity: 80%)
    #   - 'unreachable_param': This parameter exists but is only reachable when model_type='other'"

# Ignore mode - silent ignore for dynamic/experimental use
result = instantiate(config, values={'maybe_param': 1, 'experimental': 2}, on_unknown='ignore')
# Continues silently, uses default values
```

**Error message improvements:**

```python
# Type conversion with strict parameter
hp.int(32, name='batch_size', strict=True)
instantiate(config, values={'batch_size': 64.0})
# Raises: HPCallError("Parameter 'batch_size': expected int but got float (64.0). Use an integer value.")

hp.float(0.5, name='lr')  # strict=False by default
instantiate(config, values={'lr': 1})
# Returns: 1.0 (converted)

hp.int(32, name='batch_size')  # strict=False by default
instantiate(config, values={'batch_size': 64.5})
# Raises: HPCallError("Parameter 'batch_size': float 64.5 would lose precision when converted to int. Use 64 or allow precision loss explicitly.")
```
