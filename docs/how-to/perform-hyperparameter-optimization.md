# Perform Hyperparameter Optimization

Use this guide when you want Optuna to sample values for the active branch of a Hypster config.

## Install Optuna Support

```bash
uv add 'hypster[optuna]'
```

or:

```bash
pip install 'hypster[optuna]'
```

## Define A Searchable Config

```python
from dataclasses import dataclass
from hypster import HP, instantiate_with_params
from hypster.hpo.types import HpoFloat, HpoInt

@dataclass
class ModelConfig:
    family: str
    params: dict

def model_config(hp: HP) -> ModelConfig:
    family = hp.select(["linear", "forest"], name="family", default="forest", options_only=True)

    if family == "linear":
        alpha = hp.float(
            0.1,
            name="alpha",
            min=1e-4,
            max=10.0,
            hpo_spec=HpoFloat(scale="log"),
        )
        return ModelConfig(family=family, params={"alpha": alpha})

    n_estimators = hp.int(
        200,
        name="n_estimators",
        min=50,
        max=1000,
        hpo_spec=HpoInt(step=50),
    )
    max_depth = hp.int(12, name="max_depth", min=2, max=64, hpo_spec=HpoInt(scale="log"))
    return ModelConfig(family=family, params={"n_estimators": n_estimators, "max_depth": max_depth})
```

## Use It In An Objective

```python
import optuna
from hypster.hpo.optuna import suggest_values

def train_and_score(cfg: ModelConfig) -> float:
    # Replace with your training loop.
    if cfg.family == "linear":
        return 0.7
    return 0.8

def objective(trial: optuna.Trial) -> float:
    values = suggest_values(trial, config=model_config)
    run = instantiate_with_params(model_config, values=values)
    trial.set_user_attr("hypster_params", run.params)
    return train_and_score(run.value)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)
```

## Fix Part Of The Search

Wrap the config when you want a fixed branch. This keeps Optuna from sampling parameters that will later become unreachable.

```python
def forest_only_config(hp: HP) -> ModelConfig:
    n_estimators = hp.int(
        200,
        name="n_estimators",
        min=50,
        max=1000,
        hpo_spec=HpoInt(step=50),
    )
    max_depth = hp.int(12, name="max_depth", min=2, max=64, hpo_spec=HpoInt(scale="log"))
    return ModelConfig(family="forest", params={"n_estimators": n_estimators, "max_depth": max_depth})

def objective(trial: optuna.Trial) -> float:
    values = suggest_values(trial, config=forest_only_config)
    run = instantiate_with_params(forest_only_config, values=values)
    return train_and_score(run.value)
```

Prefer encoding fixed branches in the config itself when possible. It keeps the search space smaller and the replay payload cleaner.

## Supported HPO Calls

| Surface | Supported | Unsupported | Workaround |
| --- | --- | --- | --- |
| `hp.int(...)` | `HpoInt(step=..., scale="linear"|"log", include_max=...)` | custom `base=...`, nullable numeric suggestions | Use default `base=10.0`; model nullable choices as categorical branches. |
| `hp.float(...)` | `HpoFloat(step=..., scale=...)`, `distribution="uniform"|"loguniform"` | custom `base=...`, `distribution="normal"|"lognormal"`, `center=...`, `spread=...`, nullable numeric suggestions | Use Optuna-compatible float ranges or write a custom objective branch. |
| `hp.select(...)` | `HpoCategorical(ordered=False, weights=None)` | `ordered=True`, `weights=...` | Encode ordering/weights in your objective or sampler setup. |
| `hp.nest(...)` | Nested paths are prefixed and branch-aware. | Unknown child-local overrides | Keep child-local `values=` reachable for the selected branch. |
| `multi_*` calls | Not expanded by the adapter. | `multi_int`, `multi_float`, `multi_text`, `multi_bool`, `multi_select` search spaces | Model each optimized choice as scalar or categorical parameters. |

If an HPO numeric call omits `min` or `max`, the adapter uses the parameter default for the missing bound. Omitting both bounds collapses that parameter to the default value, so set explicit `min` and `max` for real search ranges.

## Multi-Value Search Choices

The Optuna adapter does not expand `multi_*` calls into a search space. Keep `multi_*` for runtime lists you want to log but not optimize, and model optimized list-like choices with scalar or categorical parameters.

Use categorical booleans for per-feature include/exclude decisions:

```python
def feature_config(hp: HP) -> list[str]:
    features = []
    if hp.select([False, True], name="include_age", default=True, options_only=True):
        features.append("age")
    if hp.select([False, True], name="include_income", default=True, options_only=True):
        features.append("income")
    if hp.select([False, True], name="include_days_active", default=False, options_only=True):
        features.append("days_active")
    return features
```

Use `hp.select([False, True], ...)` here rather than `hp.bool(...)` because the Optuna adapter samples categorical `select` calls, not boolean HP calls.

Use fixed slots when position matters:

```python
def top_features_config(hp: HP) -> list[str]:
    first = hp.select(["age", "income", "days_active"], name="feature_1", options_only=True)
    second = hp.select(["age", "income", "days_active"], name="feature_2", options_only=True)
    return [first, second]
```

Use dict-backed categorical presets for finite feature subsets:

```python
def preset_features_config(hp: HP) -> list[str]:
    return hp.select(
        {
            "core": ["age", "income"],
            "engagement": ["days_active", "sessions"],
            "full": ["age", "income", "days_active", "sessions"],
        },
        name="feature_preset",
        default="core",
        options_only=True,
    )
```
