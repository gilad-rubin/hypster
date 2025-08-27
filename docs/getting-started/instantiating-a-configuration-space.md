# ⚡ Instantiating a Config Function

Use `instantiate(func, values=...)` to execute your configuration function with optional overrides.

```python
from hypster import instantiate

result = instantiate(model_cfg, values={
    "kind": "rf",
    "n_estimators": 200,
    "max_depth": 12.5,
})
# => {"model": ("rf", 200, 12.5)}
```

## Unknown parameters

Control how unknown or conditionally unreachable values are handled via `on_unknown`:
- `on_unknown="warn"` (default): issue a warning and continue
- `on_unknown="raise"`: raise a `ValueError`
- `on_unknown="ignore"`: silently ignore

```python
instantiate(model_cfg, values={"n_trees": 200}, on_unknown="warn")
```

## Dotted keys vs nested dicts

See In Depth → Values & Overrides for how to pass nested overrides and precedence.

## Passing args/kwargs to nested configs

When using `hp.nest`, you can pass `args=`/`kwargs=` to the child function; these are forwarded at call time.

```python

def child(hp: HP, multiplier: int, offset: int = 0) -> int:
    base = hp.int(5, name="base")
    return base * multiplier + offset


def parent(hp: HP):
    calc1 = hp.nest(child, name="calc1", args=(2,))
    calc2 = hp.nest(child, name="calc2", args=(3,), kwargs={"offset": 10})
    return {"calc1": calc1, "calc2": calc2}
