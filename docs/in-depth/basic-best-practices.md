# Best Practices

These practices keep Hypster configs easy to explore, optimize, log, and replay.

## Embrace Pure Python

Hypster configs are ordinary Python functions rather than a DSL. Use `if` statements, loops, local variables, lists, helpers, dataclasses, imports, and typed return values when they make the config clearer.

The implication is that Hypster discovers the available parameters by running your function. Design config functions so they can be run repeatedly by `explore()`, HPO, and interactive UIs without causing side effects or surprising costs.

## Return Typed Runtime Objects

A strong Hypster pattern is to make each config function a typed factory for the object the caller needs:

```python
from hypster import HP
from sklearn.ensemble import RandomForestClassifier

def classifier_config(hp: HP) -> RandomForestClassifier:
    n_estimators = hp.int(200, name="n_estimators", min=10, max=1000)
    max_depth = hp.int(None, name="max_depth", allow_none=True)

    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
    )
```

Use a return type annotation for config functions whenever the output is a meaningful object. It makes the config easier to read, test, and compose with `hp.nest()`.

## Keep Config Functions Side-Effect-Light

`explore()`, HPO, and UI builders execute your config function to discover parameters. Interactive UIs may rerun it on every value change. Initializing cheap in-memory runtime objects is a good fit for config functions; effects and expensive work should stay outside the config body:

* train the model after `instantiate()`
* make paid API or network calls after `instantiate()`
* write files or database rows after `instantiate()`
* load indexes, large datasets, or heavyweight clients after `instantiate()`
* defer costly resource construction when exploratory safety matters

Use this boundary when deciding what a config should return:

| Return from the config | Usually safe during `explore()`? | Notes |
| --- | --- | --- |
| Dataclasses, dictionaries, enums, small Python objects | Yes | Good for UI generation, experiment tracking, and replay. |
| In-memory model estimators or pipeline objects | Usually | Good when construction is cheap and does not open files, sockets, or remote handles. |
| SDK clients, database handles, loaded indexes, network retrievers | Usually no | Return lightweight settings or factories, then build these after `instantiate()`. |
| Training jobs, writes, API calls, migrations | No | Run these outside the config function. |

## Name Everything Explicitly

Every `hp.*` call needs a stable `name=`. Names become the keys in `values=`, `explore()` output, and `instantiate_with_params().params`.

```python
hp.float(0.001, name="learning_rate")
```

Use Python identifier-style names:

* Good: `learning_rate`, `max_depth`, `retriever_kind`
* Avoid: `learning-rate`, `model.lr`, `max depth`

Let `hp.nest()` create dotted paths.

## Use Branches For Real Runtime Decisions

Branch when downstream structure changes:

```python
def model_config(hp: HP):
    family = hp.select(["linear", "forest"], name="family", default="forest", options_only=True)

    if family == "linear":
        return {"alpha": hp.float(0.1, name="alpha", min=0.0, max=10.0)}

    return {"n_estimators": hp.int(200, name="n_estimators", min=10)}
```

Avoid carrying irrelevant parameters for inactive branches. Branch-aware configs make experiment logs cleaner and HPO search spaces smaller.

## Use Dict-Backed Selects For Complex Runtime Values

Select keys should be simple and replayable. Map those keys to complex runtime values:

```python
tokenizer = hp.select(
    {
        "simple": "basic_tokenizer",
        "wordpiece": {"kind": "wordpiece", "vocab": "vocab.txt"},
    },
    name="tokenizer",
    default="wordpiece",
)
```

This keeps `params={"tokenizer": "wordpiece"}` while your app receives the mapped dictionary when that branch is selected.

## Turn On `options_only=True` For Enums

By default, `select` allows custom scalar values outside the listed options. Use `options_only=True` when the option list is closed:

```python
provider = hp.select(["openai", "gemini"], name="provider", default="openai", options_only=True)
```

## Use `allow_none=True` Deliberately

`None` is a real value, not an unspecified value. Mark it explicitly:

```python
max_depth = hp.int(None, name="max_depth", allow_none=True)
```

For nullable choices, you can put `None` directly in the options:

```python
tokenizer = hp.select([None, "basic"], name="tokenizer", default=None, allow_none=True)
```

## Use Numeric Coercion Deliberately

Hypster safely coerces common numeric inputs by default. Integral floats can be used for integer parameters, and integers can be used for float parameters:

```python
def config(hp: HP):
    return {
        "epochs": hp.int(10, name="epochs"),
        "lr": hp.float(0.1, name="lr"),
    }

instantiate(config, values={"epochs": 20.0, "lr": 1})
# => {"epochs": 20, "lr": 1.0}
```

Use `strict=True` when the input type itself matters:

```python
def strict_config(hp: HP):
    return {
        "epochs": hp.int(10, name="epochs", strict=True),
        "lr": hp.float(0.1, name="lr", strict=True),
    }
```

`True` and `False` are rejected by numeric parameters. Use `hp.bool()` for boolean choices.

## Capture Params For Anything You May Replay

Use `instantiate_with_params()` for experiments, UI submissions, scheduled jobs, and production runs:

```python
run = instantiate_with_params(config, values={"learning_rate": 0.01})
# tracker.log_params(run.params)
```

The params include defaults as well as explicit overrides, so later replay does not depend on changing defaults.

## Explore Before Instantiating Conditional Values

When overriding a branch, inspect it first:

```python
explore(config, values={"provider": "gemini"})
```

This prevents stale values from inactive branches from leaking into logs.

## Keep Return Values Narrow

Return what the caller needs. A small return surface makes configs easier to test and less likely to couple unrelated workflow stages.

```python
return {
    "model": model,
    "optimizer": optimizer,
}
```

Use `hp.collect(locals(), include=[...])` when that makes the return explicit and concise.
