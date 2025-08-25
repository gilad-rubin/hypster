# Numeric Types

Hypster provides flexible numeric parameter configuration through four methods: `number`, `multi_number`, `int`, and `multi_int`. These methods support automatic validation with optional bounds checking.

I'll create a comprehensive documentation for the numeric parameters:

## Numeric Parameters

Hypster provides flexible numeric parameter configuration through four methods: `number`, `multi_number`, `int`, and `multi_int`. These methods support automatic validation with optional bounds checking.

### Function Signatures

#### Single Value Methods

```python
NumericType = Union[StrictInt, StrictFloat] #pydantic types

def number(
    default: NumericType,
    *,
    name: str,
    min: Optional[NumericType] = None,
    max: Optional[NumericType] = None
) -> NumericType

def int(
    default: int,
    *,
    name: str,
    min: Optional[int] = None,
    max: Optional[int] = None
) -> int
```

#### Multiple Value Methods

```python
def multi_number(
    default: List[NumericType] = [],
    *,
    name: str,
    min: Optional[NumericType] = None,
    max: Optional[NumericType] = None
) -> List[NumericType]

def multi_int(
    default: List[int] = [],
    *,
    name: str,
    min: Optional[int] = None,
    max: Optional[int] = None
) -> List[int]
```

### Parameters

* `default`: Default value (single value) or list of default values (multi methods)
* `name`: Required name for the parameter (used for identification and access)
* `min`: Optional minimum value constraint
* `max`: Optional maximum value constraint

## Type Flexibility

### Number vs Integer

* `number`/`multi_number`: Accepts both integers and floating-point values
* `int`/`multi_int`: Accepts only integer values

The first argument in both methods is the default value:

```python
# Number parameters accept both types
temperature = hp.number(0.7, name="temperature", min=0, max=1) # min and max are optional
config(values={"temperature": 0.5})  # Float OK
config(values={"temperature": 1})    # Integer OK

# Integer parameters only accept integers
max_tokens = hp.int(256)
config(values={"max_tokens": 1024})  # Integer OK
config(values={"max_tokens": 512.5}) # Error: Float not allowed
```

## Bounds Validation

All numeric parameters support optional minimum and maximum bounds:

```python
# Single value with bounds
learning_rate = hp.number(0.001, min=0.0001)
batch_size = hp.int(32, max=256)

# Multiple values with bounds
layer_sizes = hp.multi_int(
    default=[128, 64, 32],
    min=16,
    max=512
)

learning_rates = hp.multi_number(
    default=[0.1, 0.01, 0.001],
    min=0.0001,
    max=1.0
)
```

### Valid Examples

```python
# Single values
config(values={"learning_rate": 0.01})    # Within bounds
config(values={"batch_size": 128})        # Within bounds

# Multiple values
config(values={"layer_sizes": [256, 128, 64]})           # All within bounds
config(values={"learning_rates": [0.1, 0.05, 0.001]})    # All within bounds
```

### Invalid Examples

```python
# Out of bounds errors
config(values={"learning_rate": 0.2})     # Exceeds max
config(values={"batch_size": 0})          # Below min

# Type errors
config(values={"batch_size": 64.5})       # Float not allowed for int
config(values={"layer_sizes": [32.5]})    # Float not allowed for multi_int

# Multiple values with invalid elements
config(values={"layer_sizes": [1024, 8, 64]})  # 1024 exceeds max, 8 below min
```

## Reproducibility

All numeric parameters are fully serializable and reproducible:

```python
# Configuration will be exactly reproduced
snapshot = config.get_last_snapshot()
restored_config = config(values=snapshot)
```
