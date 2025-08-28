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
* `default`: Default value(s) if none provided (single value for select, list for multi\_select)
* `options_only`: When True, only allows values from the predefined options

## Pre-defined Parameter Forms

### List Form

Use when the parameter keys and values are identical:

```python
from hypster import HP, instantiate

def model_config(hp: HP):
    # Single selection from list
    model_type = hp.select(["haiku", "sonnet"], name="model_type", default="haiku")

    # Multiple selection from list
    features = hp.multi_select(["price", "size", "color"], name="features", default=["price", "size"])

    return {
        "model_type": model_type,
        "features": features
    }

# Usage
cfg = instantiate(model_config, values={"model_type": "sonnet", "features": ["price", "color"]})
```

### Dictionary Form

Use when parameter keys need to map to different values:

```python
from hypster import HP, instantiate

def llm_config(hp: HP):
    # Single selection with value mapping
    model = hp.select({
        "haiku": "claude-3-haiku-20240307",
        "sonnet": "claude-3-sonnet-20240229"
    }, name="model", default="haiku")

    # Multiple selection with object mapping
    callbacks = hp.multi_select({
        "cost": cost_callback,
        "runtime": runtime_callback
    }, name="callbacks", default=["cost"])

    return {
        "model": model,
        "callbacks": callbacks
    }

# Usage
cfg = instantiate(llm_config, values={"model": "sonnet", "callbacks": ["cost", "runtime"]})
```

### Value Resolution

When using dictionary form, the configuration system maps input keys to their corresponding values:

```python
from hypster import HP, instantiate

def my_config(hp: HP):
    # Configuration definition with dictionary mapping
    model = hp.select({
        "haiku": "claude-3-haiku-20240307",
        "sonnet": "claude-3-sonnet-20240229"
    }, name="model", default="haiku")

    return {"model": model}

# Usage - keys are mapped to their values
config1 = instantiate(my_config, values={"model": "haiku"})
# config1 -> {"model": "claude-3-haiku-20240307"}

config2 = instantiate(my_config, values={"model": "sonnet"})
# config2 -> {"model": "claude-3-sonnet-20240229"}
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
from hypster import HP, instantiate

def my_config(hp: HP):
    model_type = hp.select(["haiku", "sonnet"], name="model_type", options_only=False)
    return {"model_type": model_type}

# Using predefined values
cfg1 = instantiate(my_config, values={"model_type": "haiku"})

# Using custom values (when options_only=False)
cfg2 = instantiate(my_config, values={"model_type": "claude-3-opus-20240229"})
```

### Invalid Instantiation Examples

```python
def strict_config(hp: HP):
    model_type = hp.select(["haiku", "sonnet"], name="model_type", options_only=True)
    return {"model_type": model_type}

# This will raise an error - value not in options list when options_only=True
cfg = instantiate(strict_config, values={"model_type": "claude-3-opus-20240229"})
```

## Required Name Parameter

{% hint style="warning" %}
All `hp.*` calls that you want to be overrideable must include an explicit `name="..."` argument.
{% endhint %}

```python
# Correct usage - explicit names
model_type = hp.select(["haiku", "sonnet"], name="model_type")
features = hp.multi_select(["price", "size", "color"], name="features")

# Incorrect usage - missing names (will raise error)
model_type = hp.select(["haiku", "sonnet"])  # Error: missing name
features = hp.multi_select(["price", "size", "color"])  # Error: missing name
```
