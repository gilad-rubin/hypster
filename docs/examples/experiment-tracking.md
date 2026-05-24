# Experiment Tracking

Use `instantiate_with_params()` when you need both the runtime object and the exact selected parameters to log to MLflow, Weights & Biases, a database, or a JSON file.

## Capture The Value And Params

```python
from hypster import HP, instantiate, instantiate_with_params

def feature_config(hp: HP):
    return {
        "numeric": hp.multi_text(["age", "income"], name="numeric"),
        "categorical": hp.multi_text(["country"], name="categorical"),
        "scale": hp.bool(True, name="scale"),
    }

def model_config(hp: HP):
    model = hp.select(["linear", "tree"], name="model", default="linear", options_only=True)
    features = hp.nest(feature_config, name="features")

    if model == "tree":
        depth = hp.int(6, name="depth", min=1, max=64)
        return {"model": model, "features": features, "depth": depth}

    alpha = hp.float(0.1, name="alpha", min=0.0, max=10.0)
    return {"model": model, "features": features, "alpha": alpha}

run = instantiate_with_params(
    model_config,
    values={
        "model": "tree",
        "depth": 12,
        "features.numeric": ["age", "income", "days_active"],
    },
)

assert run.value["depth"] == 12
assert run.params == {
    "model": "tree",
    "features.numeric": ["age", "income", "days_active"],
    "features.categorical": ["country"],
    "features.scale": True,
    "depth": 12,
}
assert instantiate(model_config, values=run.params) == run.value
```

## Log Params In Your Tracker

```python
def log_to_tracker(tracker, run):
    for path, value in run.params.items():
        tracker.log_param(path, value)
```

The important detail is that `run.params` contains defaulted values as well as user overrides. That makes it suitable for exact replay.
