# Boolean Types

Hypster provides boolean parameter configuration through `bool` and `multi_bool` methods. These methods handle boolean values without additional validation.

## Function Signatures

```python
def bool(
    default: bool,
    *,
    name: str
) -> bool

def multi_bool(
    default: List[bool] = [],
    *,
    name: str
) -> List[bool]
```

## Usage Examples

### Single Boolean Values

```python
from hypster import HP, instantiate

def stream_config(hp: HP):
    # Single boolean parameters with defaults
    stream = hp.bool(True, name="stream")
    use_cache = hp.bool(False, name="use_cache")
    verbose = hp.bool(True, name="verbose")

    return {
        "stream": stream,
        "use_cache": use_cache,
        "verbose": verbose
    }

# Usage with overrides
cfg = instantiate(stream_config, values={"stream": False, "use_cache": True})
# cfg -> {"stream": False, "use_cache": True, "verbose": True}
```

### Multiple Boolean Values

```python
from hypster import HP, instantiate

def training_config(hp: HP):
    # Multiple boolean parameters for layer configurations
    layer_trainable = hp.multi_bool([True, True, False], name="layer_trainable")
    feature_flags = hp.multi_bool([False, False], name="feature_flags")

    return {
        "layer_trainable": layer_trainable,
        "feature_flags": feature_flags
    }

# Usage with overrides
cfg = instantiate(training_config, values={
    "layer_trainable": [False, True, True],
    "feature_flags": [True, False, True]
})
```

### Invalid Values

```python
# These will raise errors during instantiation
instantiate(stream_config, values={"stream": "true"})  # String instead of boolean
instantiate(training_config, values={"layer_trainable": [1, 0]})  # Numbers instead of booleans
```

## Required Name Parameter

{% hint style="warning" %}
All `hp.*` calls that you want to be overrideable must include an explicit `name="..."` argument.
{% endhint %}

```python
# Correct usage - explicit names
stream = hp.bool(True, name="stream")
use_cache = hp.bool(False, name="use_cache")

# Incorrect usage - missing names (will raise error)
stream = hp.bool(True)        # Error: missing name
use_cache = hp.bool(False)    # Error: missing name
```
