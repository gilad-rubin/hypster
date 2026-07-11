# Experiment Tracking

Use `instantiate_with_params()` to log the exact parameters selected by a run.

{% code overflow="wrap" %}
```python
from hypster import HP, instantiate_with_params
from my_app.training import TrainingJob

def training_config(hp: HP) -> TrainingJob:
    model_family = hp.select(["linear", "forest"], name="model_family", default="forest", options_only=True)
    seed = hp.int(42, name="seed", min=0)
    batch_size = hp.int(64, name="batch_size", min=1)
    return TrainingJob(model_family=model_family, seed=seed, batch_size=batch_size)

run = instantiate_with_params(
    training_config,
    values={"model_family": "linear", "batch_size": 128},
)

assert run.params == {
    "model_family": "linear",
    "seed": 42,
    "batch_size": 128,
}
```
{% endcode %}

## Log To A Tracker

{% code overflow="wrap" %}
```python
def log_hypster_run(tracker, run):
    for path, value in run.params.items():
        tracker.log_param(path, value)
```
{% endcode %}

The params include defaults as well as explicit overrides. That matters because defaults may change between versions.

## Recommended Run Record

Use `run.params` for replay, then store adjacent metadata that explains the code, data, and outputs that produced the run:

{% code overflow="wrap" %}
```python
import hypster

record = {
    "params": run.params,
    "metrics": {"accuracy": 0.91, "loss": 0.24},
    "outputs": {"model_uri": "models:/churn/17"},
    "artifacts": {"confusion_matrix": "artifacts/confusion_matrix.png"},
    "hypster_version": hypster.__version__,
    "app_version": "2026.05.24",
    "git_commit": "abc1234",
    "dataset_id": "warehouse/churn/2026-05-01",
}
```
{% endcode %}

For trackers with separate concepts, log `params` as parameters, versions and dataset IDs as tags, metrics as metrics, and large files as artifacts.

## Record Package And Code Versions

At minimum, log:

* `hypster.__version__`
* your package or application version
* the git commit or build ID
* `run.params`
* the metric values produced by the run

{% code overflow="wrap" %}
```python
import hypster

metadata = {
    "hypster_version": hypster.__version__,
    "params": run.params,
}
```
{% endcode %}

## Replay

{% code overflow="wrap" %}
```python
from hypster import instantiate

replayed = instantiate(training_config, values=run.params)
assert replayed.model_family == run.value.model_family
assert replayed.batch_size == run.value.batch_size
```
{% endcode %}

If replay fails because a parameter is now unknown, inspect the old payload with `explore(training_config, values=old_params, on_unknown="warn")` and migrate it deliberately.

## Streaming Events With a Tracker

Instead of iterating `run.params` after the fact, pass `tracker=` to observe every parameter event live — including kind, default, options, bounds, and nesting — as the config executes:

{% code overflow="wrap" %}
```python
class SchemaLogger:
    def __init__(self):
        self.events = []

    def record_parameter(self, *, path, kind, default_value, selected_value, **event):
        self.events.append({"path": path, "kind": kind, "default": default_value, "selected": selected_value})

    def record_nest(self, *, path, name, **event):
        self.events.append({"path": path, "kind": "nest"})

logger = SchemaLogger()
run = instantiate_with_params(training_config, values={"batch_size": 64}, tracker=logger)
```
{% endcode %}

A tracker may implement `record_parameter`, `record_nest`, or both. `run.params` is unaffected.

For a compact worked example, see [Experiment Tracking (Examples)](../examples/experiment-tracking.md).
