# Selectable Types

The `select` and `multi_select` methods enable categorical parameter configuration. These methods support both single and multiple value selection with flexible validation options.

I'll help improve the documentation for the select and multi\_select parameters by adding the requested information. Here's the enhanced version:

## Select & Multi\_select Parameters

The `select` and `multi_select` methods enable categorical parameter configuration using either lists or dictionaries. These methods support both single and multiple value selection with flexible validation options.

### Function Signatures

#### select

```python
def select(
    options: Union[Dict[ValidKeyType, Any], List[ValidKeyType]],
    *,
    name: str,
    default: Optional[ValidKeyType] = None,
    options_only: bool = False
) -> Any
```

#### multi\_select

```python
def multi_select(
    options: Union[Dict[ValidKeyType, Any], List[ValidKeyType]],
    *,
    name: str,
    default: List[ValidKeyType] = None,
    options_only: bool = False
) -> List[Any]
```

### Parameters

* `options`: Either a list of valid values or a dictionary mapping keys to values
* `name`: Required name for the parameter (used for identification and access)
* `default`: Default value(s) if none provided (single value for select, list for multi\_select)
* `options_only`: When True, only allows values from the predefined options

## Pre-defined Parameter Forms

### List Form

Use when the parameter keys and values are identical:

```python
# Single selection
model_type = hp.select(["haiku", "sonnet"], default="haiku")

# Multiple selection
features = hp.multi_select(["price", "size", "color"], default=["price", "size"])
```

### Dictionary Form

Use when parameter keys need to map to different values:

```python
# Single selection with simple values
model = hp.select({
    "haiku": "claude-3-haiku-20240307",
    "sonnet": "claude-3-sonnet-20240229"
}, default="haiku")

# Multiple selection with complex values
callbacks = hp.multi_select({
    "cost": cost_callback,
    "runtime": runtime_callback
}, default=["cost"])
```

### Value Resolution

When using dictionary form, the configuration system maps input keys to their corresponding values:

```python
# Configuration definition
model = hp.select({
    "haiku": "claude-3-haiku-20240307",
    "sonnet": "claude-3-sonnet-20240229"
}, default="haiku")

# Usage
config = my_config(values={"model": "haiku"})
# Returns: "claude-3-haiku-20240307"

config = my_config(values={"model": "sonnet"})
# Returns: "claude-3-sonnet-20240229"
```

#### When to Use Dictionary Form?

Dictionary form is recommended when working with:

* Long string values: `{"haiku": "claude-3-haiku-20240307"}`
* Precise numeric values: `{"small": 1.524322}`
* Object references: `{"rf": RandomForest(n_estimators=100)}`

## Default Values

The `default` parameter must be a valid option from the predefined choices. For dictionary form, the default must be one of the keys (not values).

### List Form Defaults

When using list form, the default must be one of the items in the list:

```python
# Valid defaults
model = hp.select(["haiku", "sonnet"], default="haiku")  # OK: "haiku" is in list
features = hp.multi_select(["price", "size"], default=["price", "size"])  # OK: "price" and "size" are in list

# Invalid defaults - will raise errors
model = hp.select(["haiku", "sonnet"], default="opus")  # Error: "opus" not in list
features = hp.multi_select(["price", "size"], default=["color"])  # Error: "color" not in list
```

### Dictionary Form Defaults

When using dictionary form, the default must be one of the dictionary keys:

```python
# Valid defaults
model = hp.select({
    "haiku": "claude-3-haiku-20240307",
    "sonnet": "claude-3-sonnet-20240229"
}, default="haiku")  # OK: "haiku" is a key

callbacks = hp.multi_select({
    "cost": cost_callback,
    "runtime": runtime_callback
}, default=["cost"])  # OK: "cost" is a key

# Invalid defaults - will raise errors
model = hp.select({
    "haiku": "claude-3-haiku-20240307",
    "sonnet": "claude-3-sonnet-20240229"
}, default="claude-3-haiku-20240307")  # Error: using value instead of key

callbacks = hp.multi_select({
    "cost": cost_callback,
    "runtime": runtime_callback
}, default=["timing"])  # Error: "timing" is not a key
```

### Instantiating With Missing Default Values

If no default is provided, a value must be specified during configuration:

```python
# No default provided
model = hp.select(["haiku", "sonnet"])

# Must provide value during configuration
config = my_config(values={"model": "haiku"})  # OK
config = my_config()  # Error: no default and no value provided
```

## Value Validation

The `options_only` parameter determines how strictly values are validated:

```python
# Flexible validation - allows any value (default)
model = hp.select(["haiku", "sonnet"], options_only=False)

# Strict validation - only predefined options allowed
model = hp.select(["haiku", "sonnet"], options_only=True)
```

### Valid Instantiation Examples

```python
# Using predefined values
my_config(values={"model_type": "haiku"})

# Using custom values (when options_only=False)
my_config(values={"model_type": "claude-3-opus-20240229"})
```

### Invalid Instantiation Examples

```python
# Using a value not in the options list with options_only=True
my_config(values={"model_type": "claude-3-opus-20240229"})
```

I'll improve the Reproducibility and Value History section by adding clear examples:

## Reproducibility and Value History

Hypster maintains a historical record of parameter values to ensure configuration reproducibility across different runs. This history can be accessed using `my_config.get_last_snapshot()`, allowing you to view and reuse previous configurations.

### Value Serialization

When instantiating parameters with values outside the predefined options, Hypster handles serialization in two ways:

#### Simple types (str, int, float, bool)&#x20;

are properly logged and reproducible, regardless of if they were originally in the pre-defined options or not.

#### Complex objects

{% hint style="warning" %}
Complex object (classes, functions, anything outside of str, int, float, bool) will be serialized as strings using `str(value)` and will **not be reproducible for future runs**.
{% endhint %}

### Examples

```python
# Define configuration with options
model = hp.select({
    "haiku": "claude-3-haiku-20240307",
    "sonnet": "claude-3-sonnet-20240229"
})

# Example 1: Serializable value not in options
my_config(values={"model": "claude-3-opus"})
# Stored in history as "claude-3-opus"
# Fully reproducible since it's a simple string

# Example 2: Non-serializable value
class ModelClass:
    def __init__(self, model: str):
        self.model = model

    def __str__(self):
        return f"ModelClass(model={self.model})"

my_config(values={"model": ModelClass(model="haiku")})
# Stored in history as a string: "ModelClass(model=haiku)"
# Not reproducible since it's converted to string representation
```
