# Numeric Types

Use `hp.int`, `hp.float`, `hp.multi_int`, and `hp.multi_float` for numeric parameters. By default, Hypster uses safe numeric coercion. Use `strict=True` when callers must provide the exact numeric type.

## Signatures

{% code overflow="wrap" %}
```python
hp.int(default, *, name, min=None, max=None, strict=False, allow_none=False, hpo_spec=None)
hp.float(default, *, name, min=None, max=None, strict=False, allow_none=False, hpo_spec=None)
hp.multi_int(default, *, name, min=None, max=None, strict=False, allow_none=False)
hp.multi_float(default, *, name, min=None, max=None, strict=False, allow_none=False)
```
{% endcode %}

`hpo_spec` is ignored by normal instantiation and consumed by the Optuna adapter.

## Bounds

{% code overflow="wrap" %}
```python
from hypster import HP, instantiate

def training_config(hp: HP):
    return {
        "learning_rate": hp.float(0.001, name="learning_rate", min=1e-6, max=1.0),
        "batch_size": hp.int(64, name="batch_size", min=1, max=2048),
        "layers": hp.multi_int([256, 128], name="layers", min=1, max=4096),
    }

cfg = instantiate(
    training_config,
    values={"learning_rate": 0.01, "batch_size": 128, "layers": [512, 256]},
)
```
{% endcode %}

Values outside bounds raise.

## Safe Numeric Coercion

With `strict=False`, the default:

* `hp.int` and `hp.multi_int` accept real integers and integral floats such as `3.0`, which become `3`.
* `hp.float` and `hp.multi_float` accept real floats and integers such as `1`, which become `1.0`.
* `True` and `False` are never accepted as numeric values, even though `bool` is a subclass of `int` in Python.

This behavior is the same for top-level parameters and nested paths.

The policy is defined for Python numeric scalars. If a data library gives you NumPy, pandas, or framework-specific scalar objects, convert them to plain `int` or `float` values before passing them through `values=`.

{% code overflow="wrap" %}
```python
from hypster import HP, instantiate

def config(hp: HP):
    return {
        "epochs": hp.int(10, name="epochs"),
        "lr": hp.float(0.1, name="lr"),
    }

assert instantiate(config, values={"epochs": 20.0, "lr": 1}) == {
    "epochs": 20,
    "lr": 1.0,
}
```
{% endcode %}

Precision-losing integer conversions are rejected:

{% code overflow="wrap" %}
```python
def config(hp: HP):
    return hp.int(10, name="epochs")

instantiate(config, values={"epochs": 20.5})
# ValueError: float 20.5 would lose precision when converted to int
```
{% endcode %}

Bool values are rejected for numeric parameters:

{% code overflow="wrap" %}
```python
def config(hp: HP):
    return {
        "epochs": hp.int(10, name="epochs"),
        "lr": hp.float(0.1, name="lr"),
    }

instantiate(config, values={"epochs": True})
# ValueError: expected int but got bool
```
{% endcode %}

## Strict Numeric Types

With `strict=True`:

* `hp.int` and `hp.multi_int` accept real integers only, excluding bool.
* `hp.float` and `hp.multi_float` accept real floats only, excluding integers and bool.

{% code overflow="wrap" %}
```python
def strict_config(hp: HP):
    return {
        "epochs": hp.int(10, name="epochs", strict=True),
        "lr": hp.float(0.1, name="lr", strict=True),
    }

instantiate(strict_config, values={"epochs": 20.0})
# ValueError: expected int but got float

instantiate(strict_config, values={"lr": 1})
# ValueError: expected float but got int
```
{% endcode %}

## Nullable Numeric Values

Use `allow_none=True` when `None` is an intentional scalar value:

{% code overflow="wrap" %}
```python
def tree_config(hp: HP):
    return {
        "max_depth": hp.int(None, name="max_depth", allow_none=True),
        "dropout": hp.float(0.1, name="dropout", min=0.0, max=1.0, allow_none=True),
    }

assert instantiate(tree_config, values={"dropout": None}) == {
    "max_depth": None,
    "dropout": None,
}
```
{% endcode %}

Nullable elements are not supported for `multi_int` or `multi_float`. Use `multi_select(..., allow_none=True)` for nullable categorical lists.

## HPO Specs

{% code overflow="wrap" %}
```python
from hypster.hpo.types import HpoFloat, HpoInt

def search_config(hp: HP):
    return {
        "learning_rate": hp.float(
            0.001,
            name="learning_rate",
            min=1e-6,
            max=1.0,
            hpo_spec=HpoFloat(scale="log"),
        ),
        "batch_size": hp.int(
            64,
            name="batch_size",
            min=16,
            max=512,
            hpo_spec=HpoInt(step=16),
        ),
    }
```
{% endcode %}
