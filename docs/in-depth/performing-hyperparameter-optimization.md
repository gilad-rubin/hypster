# 🧪 Performing Hyperparameter Optimization

This page shows how to run HPO using Hypster’s define-by-run configs and typed HPO specs. The examples use Optuna.

## Concepts

* Define-by-run: your config executes conditionally; only touched parameters are suggested.
* Inline HPO specs (optional): pass `hpo_spec=` to encode search semantics (log/step/ordered) next to each parameter. If you omit it, sensible defaults are used: linear scale, no quantization (step=None), and unordered categoricals (ordered=False).

HPO spec classes (backend-agnostic):

* `HpoInt(step=None, scale="linear"|"log", base=10.0, include_max=True)`
* `HpoFloat(step=None, scale="linear"|"log", base=10.0, distribution=None|"uniform"|"loguniform"|"normal"|"lognormal", center=None, spread=None)`
* `HpoCategorical(ordered=False, weights=None)`

## Installation

```bash
uv add 'hypster[optuna]'
# or
uv add optuna
```

## Minimal example (Optuna)

```python
import optuna
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier, LogisticRegression

from hypster import HP, instantiate
from hypster.hpo.types import HpoInt, HpoFloat, HpoCategorical
from hypster.hpo.optuna import suggest_values


def model_cfg(hp: HP):
    # hpo_spec is optional; defaults are linear scale and unordered categoricals
    kind = hp.select(["rf", "lr"], name="kind")

    if kind == "rf":
        # with explicit hpo_spec (quantized int and 0.5 step float)
        n = hp.int(100, name="n_estimators", min=50, max=300, hpo_spec=HpoInt(step=50))
        d = hp.float(10.0, name="max_depth", min=2.0, max=30.0, hpo_spec=HpoFloat(step=0.5))
        return RandomForestClassifier(n_estimators=n, max_depth=d, random_state=42)

    else:
        # without hpo_spec: float uses linear scale by default
        C = hp.float(1.0, name="C", min=1e-5, max=10.0)
        # with explicit hpo_spec (categorical ordered flag)
        solver = hp.select(["lbfgs", "saga"], name="solver", hpo_spec=HpoCategorical(ordered=False))
        return LogisticRegression(C=C, solver=solver, max_iter=2000, random_state=42)


def objective(trial: optuna.Trial) -> float:
    values = suggest_values(trial, config=model_cfg)
    model = instantiate(model_cfg, values=values)
    X, y = make_classification(n_samples=400, n_features=20, n_informative=10, random_state=42)
    return cross_val_score(model, X, y, cv=3, n_jobs=-1).mean()


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)
print(study.best_value, study.best_params)
```

## Notes

* Supported params: `hp.int`, `hp.float`, `hp.select`.
* Multi-value params exist in Hypster but are not expanded for HPO in the Optuna integration yet.
* It’s straightforward to add other backends (e.g., Ray Tune). If you’re interested, please open an issue or PR.
* For Optuna-specific details and ask-and-tell examples, see the Integrations section.
