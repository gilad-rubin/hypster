# Select & Multi-Select

Use `hp.select()` and `hp.multi_select()` for categorical choices.

Selected choices are part of Hypster's reproducibility surface. They are what `instantiate_with_params(...).params` records and what you pass back through `values=...` to replay a run.

## Signatures

```python
hp.select(options, *, name, default=NO_DEFAULT, options_only=False, allow_none=False, hpo_spec=None)
hp.multi_select(options, *, name, default=None, options_only=False, allow_none=False)
```

## List Form

Use list form when the logged choice and returned value are the same simple value:

```python
from hypster import HP, instantiate

def config(hp: HP):
    model = hp.select(["haiku", "sonnet"], name="model", default="haiku")
    features = hp.multi_select(["cache", "trace"], name="features", default=["cache"])
    return {"model": model, "features": features}

instantiate(config, values={"model": "sonnet", "features": ["cache", "trace"]})
# => {"model": "sonnet", "features": ["cache", "trace"]}
```

List-form choices must be logging-safe scalar values: `None`, `bool`, `int`, `float`, or `str`. If you need a complex object, use dictionary form.

## Dictionary Form

Use dictionary form when a simple logged key should return a different value. The key is logged and replayed; the mapped value is returned from the config.

```python
from hypster import HP, instantiate_with_params

def config(hp: HP):
    model = hp.select(
        {
            "small": {"layers": 2, "units": [64, 32]},
            "large": {"layers": 4, "units": [256, 128]},
        },
        name="model",
        default="small",
    )
    return {"model": model}

run = instantiate_with_params(config, values={"model": "large"})

assert run.value == {"model": {"layers": 4, "units": [256, 128]}}
assert run.params == {"model": "large"}
```

Use `options_only=True` with dictionary form when the logged keys are a closed enum:

```python
def strict_config(hp: HP):
    model = hp.select(
        {
            "small": {"layers": 2},
            "large": {"layers": 4},
        },
        name="model",
        default="small",
        options_only=True,
    )
    return {"model": model}

run = instantiate_with_params(strict_config, values={"model": "large"})

assert run.value == {"model": {"layers": 4}}
assert run.params == {"model": "large"}
```

Dictionary form is the recommended way to return:

* objects or callables
* dictionaries, lists, tuples, or dataclasses
* long provider/model IDs behind short aliases

```python
architecture = hp.select(
    {
        "small": {"layers": 2, "units": [64, 32]},
        "large": {"layers": 4, "units": [256, 128]},
    },
    name="architecture",
    default="small",
)
```

For nullable choices, you can use `None` directly in list-form options with `allow_none=True`.

## Explicit None

If `None` itself is a selectable choice or override, mark the parameter as nullable with `allow_none=True`:

```python
def config(hp: HP):
    thinking_level = hp.select(
        [None, "low", "medium", "high"],
        name="thinking_level",
        default=None,
        allow_none=True,
    )
    features = hp.multi_select(
        [None, "cache", "trace"],
        name="features",
        default=[None],
        allow_none=True,
    )
    return {"thinking_level": thinking_level, "features": features}
```

Without `allow_none=True`, `None` defaults, choices, and overrides raise with guidance.

## Empty Nullable Selects

An empty option list can default to `None` when the parameter is explicitly nullable:

```python
def config(hp: HP):
    return hp.select([], name="choice", allow_none=True)

assert instantiate(config) is None
```

Without `allow_none=True`, an empty option list with no explicit default raises because Hypster has no safe value to select.

## Custom Choices

By default, `options_only=False`, so callers may provide a custom choice outside the declared options:

```python
def config(hp: HP):
    return hp.select(["haiku", "sonnet"], name="model")

assert instantiate(config, values={"model": "opus"}) == "opus"
```

Custom choices must still be logging-safe scalar values. Use `options_only=True` to reject anything outside the declared options:

```python
def config(hp: HP):
    return hp.select(["haiku", "sonnet"], name="model", options_only=True)

instantiate(config, values={"model": "opus"})
# ValueError: 'opus' not in allowed options
```

## Names

`name=` must be a valid Python identifier and cannot be a Python keyword. Hypster composes dotted parameter paths from nested names, so literal dots, spaces, and hyphens are not allowed in individual names.
