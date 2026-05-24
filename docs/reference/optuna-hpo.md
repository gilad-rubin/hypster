# Optuna HPO API

Install Optuna support:

```bash
uv add 'hypster[optuna]'
```

Public imports:

```python
from hypster.hpo.optuna import suggest_values
from hypster.hpo.types import HpoCategorical, HpoFloat, HpoInt
```

## suggest_values

```python
suggest_values(trial, *, config, args=(), kwargs=None) -> dict
```

Runs `config` with a trial-backed `HP` proxy and returns a `values` dictionary that can be passed to `instantiate()`.

```python
values = suggest_values(trial, config=model_config)
cfg = instantiate(model_config, values=values)
```

The adapter is branch-aware. It only suggests parameters reached by the sampled execution path.

For numeric suggestions, `min` and `max` on the `hp.int` or `hp.float` call define the Optuna search range. If either bound is omitted, the parameter default is used for that missing bound. If both are omitted, the Optuna suggestion collapses to the default value.

## HpoInt

```python
HpoInt(
    step=None,
    scale="linear",
    base=10.0,
    include_max=True,
)
```

| Field | Meaning |
| --- | --- |
| `step` | Quantization step passed to `trial.suggest_int`. |
| `scale` | `"linear"` or `"log"`, passed as Optuna's `log` flag. |
| `base` | Must remain `10.0`; custom bases are rejected because Optuna `suggest_int()` has no base parameter. |
| `include_max` | When `False`, the Optuna high bound is reduced by `step` or `1`. |

## HpoFloat

```python
HpoFloat(
    step=None,
    scale="linear",
    base=10.0,
    distribution=None,
    center=None,
    spread=None,
)
```

| Field | Meaning |
| --- | --- |
| `step` | Quantization step passed to `trial.suggest_float`. |
| `scale` | `"linear"` or `"log"`, passed as Optuna's `log` flag when `distribution` is not set. |
| `base` | Must remain `10.0`; custom bases are rejected because Optuna `suggest_float()` has no base parameter. |
| `distribution` | `None` and `"uniform"` use non-log `suggest_float()`. `"loguniform"` uses `suggest_float(..., log=True)`. `"normal"` and `"lognormal"` are rejected. |
| `center`, `spread` | Rejected by the Optuna adapter because they only make sense for normal/lognormal distributions. |

If `distribution="loguniform"`, the adapter uses Optuna's log sampling even if `scale` is left at its default.

## HpoCategorical

```python
HpoCategorical(ordered=False, weights=None)
```

The current Optuna adapter uses `trial.suggest_categorical()` for `hp.select`. `ordered=True` and `weights=...` are rejected because Optuna categorical suggestions cannot express ordered or weighted categorical semantics.

## Supported HP Calls

| HP call | Optuna behavior |
| --- | --- |
| `hp.int` | `trial.suggest_int(path, low, high, step=..., log=...)` |
| `hp.float` | `trial.suggest_float(path, low, high, step=..., log=...)` |
| `hp.select` | `trial.suggest_categorical(path, keys)` |
| `hp.nest` | Prefixes child parameter paths. |

`multi_int`, `multi_float`, `multi_text`, `multi_bool`, and `multi_select` are not expanded by the Optuna adapter.

Explicit child-local overrides passed with `hp.nest(child, name="child", values=...)` are validated before `suggest_values()` returns. Unknown or unreachable child keys raise instead of being silently ignored.

## Nullable Numeric Values

`allow_none=True` is not supported for HPO numeric suggestions. Model nullable search choices as categoricals, then branch to numeric parameters when needed.

## Invalid HPO Specs

`suggest_values()` raises `ValueError` when a backend-agnostic HPO spec asks for semantics that Optuna cannot represent.

```python
from hypster import HP
from hypster.hpo.optuna import suggest_values
from hypster.hpo.types import HpoCategorical, HpoFloat


def normal_float_config(hp: HP):
    return hp.float(
        0.1,
        name="dropout",
        min=0.0,
        max=1.0,
        hpo_spec=HpoFloat(distribution="normal", center=0.2, spread=0.05),
    )


def weighted_choice_config(hp: HP):
    return hp.select(
        ["small", "large"],
        name="model",
        hpo_spec=HpoCategorical(weights=[0.8, 0.2]),
    )
```

Both fail during `suggest_values(trial, config=...)` before a values dictionary is returned. Show the error as configuration feedback:

```text
This HPO spec cannot be represented by the Optuna adapter. Use uniform/loguniform float sampling, remove categorical weights or ordering, or implement custom sampling inside the objective.
```
