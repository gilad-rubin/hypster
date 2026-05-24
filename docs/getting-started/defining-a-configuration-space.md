# Define A Config Function

A Hypster configuration space is an ordinary Python function whose first argument is named `hp`. It is not a DSL or a separate config file format: use normal Python control flow, lists, helper functions, imports, dataclasses, and object construction.

```python
from hypster import HP

def config(hp: HP):
    ...
```

`hp` must be the first positional parameter; keyword-only `hp` is rejected before the config executes. The `hp: HP` annotation is recommended, but unannotated config functions are still valid.

The function can return anything your application needs: a dictionary, dataclass, model instance, tuple, or fully constructed workflow object. Prefer a return type annotation when the output is a meaningful object.

Because Hypster discovers available parameters by running this function, keep config bodies fast and predictable. Avoid side effects such as writes, training, paid API calls, network calls, database access, or heavyweight initialization inside the config.

## Add Parameters

Use `hp.*` calls for values that should be visible, overrideable, replayable, searchable, or rendered in a UI.

```python
from hypster import HP

def llm_config(hp: HP):
    provider = hp.select(["openai", "gemini"], name="provider", default="openai", options_only=True)
    temperature = hp.float(0.2, name="temperature", min=0.0, max=2.0)
    max_tokens = hp.int(1024, name="max_tokens", min=1, max=16_384)

    return {
        "provider": provider,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
```

Every public parameter needs an explicit `name=...`. Names must be valid Python identifiers, so use `batch_size`, not `batch-size` or `model.learning_rate`.

## Use Branches

Hypster is define-by-run. Only parameters touched by the active branch are selected for that run.

```python
def model_config(hp: HP):
    family = hp.select(["linear", "forest"], name="family", default="forest", options_only=True)

    if family == "linear":
        return {
            "family": family,
            "alpha": hp.float(0.1, name="alpha", min=0.0, max=10.0),
        }

    return {
        "family": family,
        "n_estimators": hp.int(200, name="n_estimators", min=10, max=1000),
        "max_depth": hp.int(None, name="max_depth", min=1, max=100, allow_none=True),
    }
```

If `family="linear"`, `n_estimators` and `max_depth` are unreachable. Hypster raises if you pass values for inactive branches.

This define-by-run model is what makes normal Python branches work: Hypster runs the function with the selected values and records the `hp.*` calls it reaches.

## Compose With Nesting

Split large configs into smaller functions and compose them with `hp.nest()`.

```python
def optimizer_config(hp: HP):
    return {
        "learning_rate": hp.float(0.001, name="learning_rate", min=1e-6, max=1.0),
        "weight_decay": hp.float(0.0, name="weight_decay", min=0.0, max=1.0),
    }

def training_config(hp: HP):
    return {
        "epochs": hp.int(10, name="epochs", min=1),
        "optimizer": hp.nest(optimizer_config, name="optimizer"),
    }
```

Nested parameters use dotted paths in `values=`:

```python
{"optimizer.learning_rate": 0.01}
```

## Use Dict-Backed Selects For Complex Values

Select choices must be logging-safe scalars. If the runtime value is complex, map a simple key to it:

```python
def architecture_config(hp: HP):
    return hp.select(
        {
            "small": {"layers": 2, "units": [64, 32]},
            "large": {"layers": 4, "units": [256, 128]},
        },
        name="architecture",
        default="small",
        options_only=True,
    )
```

Your config receives the mapped dictionary, while `instantiate_with_params()` records the key. Add `options_only=True` when values outside the mapping should be rejected.

## Return Initialized Objects

It is often best to let a config return the initialized object your application will use:

```python
from sklearn.ensemble import RandomForestClassifier
from hypster import HP

def classifier_config(hp: HP) -> RandomForestClassifier:
    n_estimators = hp.int(200, name="n_estimators", min=10, max=1000)
    max_depth = hp.int(None, name="max_depth", allow_none=True)

    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
    )
```

{% hint style="info" %}
Initialized in-memory objects are a good fit when construction is cheap. For SDK clients, remote retrievers, loaded indexes, database handles, training jobs, or writes, return lightweight settings and build the side-effectful object after `instantiate()`. See [Best Practices](../in-depth/basic-best-practices.md).
{% endhint %}

## Next Step

Use [Explore a Configuration Space](exploring-a-configuration-space.md) to inspect the parameter tree before running it.
