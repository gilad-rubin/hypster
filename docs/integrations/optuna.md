# Optuna

Optuna is a popular black-box optimizer. This page shows how to use Optuna with Hypster.

## Installation

```bash
uv add 'hypster[optuna]'
# or
uv add optuna
```

## Usage

See Advanced → [Performing Hyperparameter Optimization](../in-depth/performing-hyperparameter-optimization.md) for a full example. Summary:

1. Define your Hypster config (hpo\_spec is optional; defaults are sensible).
2. In your objective, call `hypster.hpo.optuna.suggest_values(trial, config=...)` to get a values dict.
3. Instantiate your config with those values and evaluate.

```python
import optuna
from hypster import HP, instantiate
from hypster.hpo.optuna import suggest_values


def model_cfg(hp: HP):
    kind = hp.select(["rf", "lr"], name="kind")  # hpo_spec omitted: linear/unordered defaults
    # ... conditional params ...
    return {"model": ("rf", 100, 10.0)}


def objective(trial: optuna.Trial) -> float:
    values = suggest_values(trial, config=model_cfg)
    cfg = instantiate(model_cfg, values=values)
    # ... train/evaluate with cfg["model"] ...
    return 0.0
```

## Notes

* Supports scalar parameters: `hp.int`, `hp.float`, `hp.select`.
* Multi-\* parameters are not expanded in the Optuna adapter (use booleans or explicit scalars if needed).
* It’s easy to add other backends (e.g., Ray Tune). Please open an issue or PR if you’re interested.
