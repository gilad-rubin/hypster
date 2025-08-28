# Numeric Types

Hypster provides flexible numeric parameter configuration through `int`, `multi_int`, `float`, and `multi_float` methods. These methods support automatic validation with optional bounds checking.

## Numeric Parameters

### Function Signatures

#### Integer Methods

```python
def int(
    default: int,
    *,
    name: str,
    min: Optional[int] = None,
    max: Optional[int] = None
) -> int

def multi_int(
    default: List[int] = [],
    *,
    name: str,
    min: Optional[int] = None,
    max: Optional[int] = None
) -> List[int]
```

#### Float Methods

```python
def float(
    default: float,
    *,
    name: str,
    min: Optional[float] = None,
    max: Optional[float] = None
) -> float

def multi_float(
    default: List[float] = [],
    *,
    name: str,
    min: Optional[float] = None,
    max: Optional[float] = None
) -> List[float]
```

## Type Specificity

### Float vs Integer

* `float`/`multi_float`: Accepts floating-point values only
* `int`/`multi_int`: Accepts integer values only

The first argument in both methods is the default value:

```python
from hypster import HP, instantiate

def model_config(hp: HP):
    # Float parameters accept floating-point values
    temperature = hp.float(0.7, name="temperature", min=0, max=1)

    # Integer parameters only accept integers
    max_tokens = hp.int(256, name="max_tokens")

    return {
        "temperature": temperature,
        "max_tokens": max_tokens
    }

# Usage
cfg = instantiate(model_config, values={"temperature": 0.5})  # Float OK
cfg = instantiate(model_config, values={"max_tokens": 1024})   # Integer OK
cfg = instantiate(model_config, values={"max_tokens": 512.5}) # Error: Float not allowed
```

## Bounds Validation

All numeric parameters support optional minimum and maximum bounds:

```python
from hypster import HP, instantiate

def training_config(hp: HP):
    # Single value with bounds
    learning_rate = hp.float(0.001, name="learning_rate", min=0.0001)
    batch_size = hp.int(32, name="batch_size", max=256)

    # Multiple values with bounds
    layer_sizes = hp.multi_int(
        default=[128, 64, 32],
        name="layer_sizes",
        min=16,
        max=512
    )

    learning_rates = hp.multi_float(
        default=[0.1, 0.01, 0.001],
        name="learning_rates",
        min=0.0001,
        max=1.0
    )

    return {
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "layer_sizes": layer_sizes,
        "learning_rates": learning_rates
    }
```

### Valid Examples

```python
# Single values within bounds
cfg1 = instantiate(training_config, values={"learning_rate": 0.01})    # Within bounds
cfg2 = instantiate(training_config, values={"batch_size": 128})        # Within bounds

# Multiple values within bounds
cfg3 = instantiate(training_config, values={"layer_sizes": [256, 128, 64]})           # All within bounds
cfg4 = instantiate(training_config, values={"learning_rates": [0.1, 0.05, 0.001]})    # All within bounds
```

### Invalid Examples

```python
# Out of bounds errors
cfg = instantiate(training_config, values={"learning_rate": 0.2})     # Exceeds max bound
cfg = instantiate(training_config, values={"batch_size": 0})          # Below min bound

# Type errors
cfg = instantiate(training_config, values={"batch_size": 64.5})       # Float not allowed for int
cfg = instantiate(training_config, values={"layer_sizes": [32.5]})    # Float not allowed for multi_int

# Multiple values with invalid elements
cfg = instantiate(training_config, values={"layer_sizes": [1024, 8, 64]})  # 1024 exceeds max, 8 below min
```


```python
# Correct usage - explicit names
learning_rate = hp.float(0.001, name="learning_rate")
batch_size = hp.int(32, name="batch_size")

# Incorrect usage - missing names (will raise error)
learning_rate = hp.float(0.001)  # Error: missing name
batch_size = hp.int(32)          # Error: missing name
```

## Complete Example

```python
from hypster import HP, instantiate

def model_config(hp: HP):
    # Integer parameters
    epochs = hp.int(100, name="epochs", min=1)
    batch_size = hp.int(32, name="batch_size", min=1, max=512)

    # Float parameters
    learning_rate = hp.float(0.001, name="learning_rate", max=1.0)
    dropout = hp.float(0.1, name="dropout", min=0.0, max=0.9)

    # Multiple values
    layer_sizes = hp.multi_int([128, 64], name="layer_sizes", min=8, max=1024)

    return {
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "dropout": dropout,
        "layer_sizes": layer_sizes
    }

# Instantiate with custom values
cfg = instantiate(
    model_config,
    values={
        "epochs": 200,
        "learning_rate": 0.01,
        "layer_sizes": [256, 128, 64]
    }
)
```
