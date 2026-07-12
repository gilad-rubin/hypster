# Experiment Tracking

Use `instantiate_with_params()` when you need both the runtime object and the exact selected parameters to log to MLflow, Weights & Biases, a database, or a JSON file.

## Capture The Value And Params

{% code overflow="wrap" %}
```python
from hypster import HP, instantiate, instantiate_with_params
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from my_app.experiments import TrainingJob
from my_app.features import FeatureSelector

def feature_config(hp: HP) -> FeatureSelector:
    numeric = hp.multi_text(["age", "income"], name="numeric")
    categorical = hp.multi_text(["country"], name="categorical")
    scale = hp.bool(True, name="scale")
    return FeatureSelector(numeric=numeric, categorical=categorical, scale=scale)

def linear_model(hp: HP) -> LogisticRegression:
    C = hp.float(1.0, name="C", min=1e-4, max=100.0)
    return LogisticRegression(C=C, max_iter=1000)

def tree_model(hp: HP) -> RandomForestClassifier:
    max_depth = hp.int(6, name="max_depth", min=1, max=64)
    return RandomForestClassifier(max_depth=max_depth, random_state=42)

model_options = {
    "linear": linear_model,
    "tree": tree_model,
}

def training_job_config(hp: HP) -> TrainingJob:
    selected_model = hp.select(model_options, name="model_family", default="linear", options_only=True)
    features = hp.nest(feature_config, name="features")
    model = hp.nest(selected_model, name="model")
    return TrainingJob(features=features, model=model)

run = instantiate_with_params(
    training_job_config,
    values={
        "model_family": "tree",
        "model.max_depth": 12,
        "features.numeric": ["age", "income", "days_active"],
    },
)

assert run.value.model.max_depth == 12
assert run.params == {
    "model_family": "tree",
    "features.numeric": ["age", "income", "days_active"],
    "features.categorical": ["country"],
    "features.scale": True,
    "model.max_depth": 12,
}
replayed = instantiate(training_job_config, values=run.params)
assert replayed.model.max_depth == run.value.model.max_depth
```
{% endcode %}

## Log Params In Your Tracker

{% code overflow="wrap" %}
```python
def log_to_tracker(tracker, run):
    for path, value in run.params.items():
        tracker.log_param(path, value)
```
{% endcode %}

The important detail is that `run.params` contains defaulted values as well as user overrides. That makes it suitable for exact replay.

For run-record structure, versioning metadata, and replay-recovery guidance, see [Experiment Tracking (Reproducibility)](../reproducibility/experiment-tracking.md).
