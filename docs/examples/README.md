# Examples Overview

Use these examples as copyable shapes for real projects. Each page focuses on one workflow and only uses public Hypster APIs.

Hypster config functions are plain Python functions:

{% code overflow="wrap" %}
```python
from hypster import HP

def config(hp: HP):
    mode = hp.select(["fast", "accurate"], name="mode", default="fast")
    return {"mode": mode}
```
{% endcode %}

The same function can serve several jobs:

* `explore(config)` prints the active parameter tree.
* `explore(config, return_info=True)` returns a schema object for tools and UIs.
* `instantiate(config, values={...})` returns the runtime object.
* `instantiate_with_params(config, values={...})` returns the runtime object plus replayable selected params.
* `suggest_values(trial, config=config)` lets Optuna choose values for the reachable parameters.

## Example Map

| Page | Use it when you need |
| --- | --- |
| [Machine Learning](machine-learning.md) | Model-family branches, numeric bounds, nullable values, and HPO-ready configs. |
| [Data Processing](data-processing.md) | Ingestion, cleaning, feature flags, and export options in one pipeline config. |
| [AI Workflows](ai-workflows.md) | Provider selection, RAG knobs, prompt settings, and dict-backed complex choices. |
| [Nested Workflows](nested-workflows.md) | Reusable child configs, conditional nesting, deep value paths, and branch exploration. |
| [Interactive UI From Schema](interactive-ui.md) | Generate form state from `explore(..., return_info=True)` and feed it back to `instantiate`. |
| [Experiment Tracking](experiment-tracking.md) | Capture selected params for logs, cards, and replay. |

## A Small End-to-End Pattern

{% code overflow="wrap" %}
```python
from hypster import HP, explore, instantiate_with_params

def pipeline_config(hp: HP):
    stage = hp.select(["debug", "full"], name="stage", default="debug")

    if stage == "full":
        sample_rows = hp.int(1_000_000, name="sample_rows", min=100, max=10_000_000)
    else:
        sample_rows = hp.int(1000, name="sample_rows", min=100, max=1_000_000)

    return {"stage": stage, "sample_rows": sample_rows}

explore(pipeline_config)

run = instantiate_with_params(
    pipeline_config,
    values={"stage": "full", "sample_rows": 250_000},
)

assert run.value == {"stage": "full", "sample_rows": 250_000}
assert run.params == {"stage": "full", "sample_rows": 250_000}
```
{% endcode %}

{% hint style="info" %}
Hypster raises when `values=` includes unknown parameters or parameters from a branch that was not reached. That default protects replayability. Use `explore(config, values=...)` to inspect the branch you are about to run.
{% endhint %}
