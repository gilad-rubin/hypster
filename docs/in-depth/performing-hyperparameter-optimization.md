# Performing Hyperparameter Optimization

Hypster's Optuna integration lets the same config function define both runtime configuration and a branch-aware search space.

For the task recipe, see [Perform Hyperparameter Optimization](../how-to/perform-hyperparameter-optimization.md). This page explains the model.

## Mental Model

1. A config function calls `hp.int`, `hp.float`, and `hp.select`.
2. `suggest_values(trial, ...)` runs that config with a trial-backed `HP` proxy.
3. The proxy asks Optuna for values and returns a `values` dictionary.
4. You pass that dictionary to `instantiate(config, values=values)`.
5. Your objective evaluates the instantiated runtime object.

Because the config is executed normally, conditionals work naturally. If Optuna samples `family="linear"`, only the linear parameters are suggested.

## Search Semantics Live Next To Parameters

Use `hpo_spec=` to document search semantics where the parameter is defined:

{% code overflow="wrap" %}
```python
from hypster import HP
from hypster.hpo.types import HpoFloat, HpoInt
from my_app.training import Trainer

def config(hp: HP) -> Trainer:
    learning_rate = hp.float(
        0.001,
        name="learning_rate",
        min=1e-6,
        max=1.0,
        hpo_spec=HpoFloat(scale="log"),
    )
    batch_size = hp.int(
        64,
        name="batch_size",
        min=16,
        max=512,
        hpo_spec=HpoInt(step=16),
    )
    return Trainer(learning_rate=learning_rate, batch_size=batch_size)
```
{% endcode %}

Normal `instantiate()` ignores `hpo_spec`; the Optuna adapter consumes it.

## Supported Surface

The current Optuna adapter supports:

* `hp.int`
* `hp.float`
* `hp.select`
* `hp.nest`

It does not expand multi-value parameters such as `hp.multi_select`. Model those choices as individual scalar or categorical parameters when they need to be optimized.

The adapter rejects `hpo_spec` fields that Optuna cannot honor rather than silently ignoring them:

* `HpoInt(step=..., scale="linear"|"log", include_max=...)` is supported; custom `base=...` is rejected.
* `HpoFloat(step=..., scale="linear"|"log")` is supported.
* `HpoFloat(distribution="uniform")` maps to non-log `suggest_float()`.
* `HpoFloat(distribution="loguniform")` maps to `suggest_float(..., log=True)`.
* `HpoFloat(distribution="normal"|"lognormal")`, `center=...`, `spread=...`, and custom `base=...` are rejected.
* `HpoCategorical(ordered=False, weights=None)` is supported; `ordered=True` and `weights=...` are rejected.

Nested HPO paths behave like normal nested values. Explicit child-local overrides passed with `hp.nest(..., values=...)` are validated before `suggest_values()` returns.

## Nullable Values

Nullable numeric HPO parameters are not supported directly. Model the nullable choice as a categorical branch:

{% code overflow="wrap" %}
```python
from sklearn.ensemble import RandomForestClassifier

def tree_config(hp: HP) -> RandomForestClassifier:
    depth_mode = hp.select(["unlimited", "bounded"], name="depth_mode", default="bounded", options_only=True)

    if depth_mode == "unlimited":
        max_depth = None
    else:
        max_depth = hp.int(12, name="max_depth", min=1, max=64)

    return RandomForestClassifier(max_depth=max_depth, random_state=42)
```
{% endcode %}

## Exact API

See [Optuna HPO API](../reference/optuna-hpo.md) for signatures and supported spec fields.
