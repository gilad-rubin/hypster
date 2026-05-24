# Optuna

Optuna is the first supported HPO backend for Hypster. The integration lives in `hypster.hpo.optuna`.

## Install

{% code overflow="wrap" %}
```bash
uv add 'hypster[optuna]'
```
{% endcode %}

or:

{% code overflow="wrap" %}
```bash
pip install 'hypster[optuna]'
```
{% endcode %}

## Basic Pattern

{% code overflow="wrap" %}
```python
import optuna
from hypster import HP, instantiate
from hypster.hpo.optuna import suggest_values
from hypster.hpo.types import HpoFloat, HpoInt

def model_config(hp: HP):
    family = hp.select(["linear", "forest"], name="family", default="forest", options_only=True)

    if family == "linear":
        alpha = hp.float(
            0.1,
            name="alpha",
            min=1e-4,
            max=10.0,
            hpo_spec=HpoFloat(scale="log"),
        )
        return {"family": family, "alpha": alpha}

    n_estimators = hp.int(
        200,
        name="n_estimators",
        min=50,
        max=1000,
        hpo_spec=HpoInt(step=50),
    )
    return {"family": family, "n_estimators": n_estimators}

def objective(trial: optuna.Trial) -> float:
    values = suggest_values(trial, config=model_config)
    cfg = instantiate(model_config, values=values)
    return train_and_score(cfg)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)
```
{% endcode %}

## What Is Supported

* `hp.int`, backed by `trial.suggest_int`
* `hp.float`, backed by `trial.suggest_float`
* `hp.select`, backed by `trial.suggest_categorical`
* `hp.nest`, which prefixes nested parameter paths

Multi-value HP calls are not expanded by the current adapter.

The adapter only accepts HPO spec fields that Optuna can represent. Supported fields include `HpoInt(step=..., scale=..., include_max=...)`, `HpoFloat(step=..., scale=...)`, `HpoFloat(distribution="uniform"|"loguniform")`, and `HpoCategorical(ordered=False, weights=None)`. Unsupported fields such as custom `base=...`, normal/lognormal float distributions, `center=...`, `spread=...`, ordered categoricals, and categorical weights raise instead of being ignored.

Nested explicit overrides passed through `hp.nest(..., values=...)` are validated before `suggest_values()` returns.

## More

* [Perform Hyperparameter Optimization](../how-to/perform-hyperparameter-optimization.md)
* [Optuna HPO API](../reference/optuna-hpo.md)
