# Hamilton

Hamilton and Hypster can be used together by keeping graph construction and parameter selection separate:

* Use Hypster to select a small, replayable set of workflow parameters.
* Pass the instantiated value into your Hamilton driver, module, or execution wrapper.
* Log `instantiate_with_params(...).params` next to Hamilton run metadata.

## Shape

{% code overflow="wrap" %}
```python
from hypster import HP, instantiate_with_params

def hamilton_run_config(hp: HP):
    dataset = hp.select(["sample", "warehouse"], name="dataset", default="sample", options_only=True)
    feature_set = hp.select(["basic", "extended"], name="feature_set", default="basic", options_only=True)
    model_family = hp.select(["linear", "forest"], name="model_family", default="linear", options_only=True)

    return {
        "dataset": dataset,
        "feature_set": feature_set,
        "model_family": model_family,
    }

run = instantiate_with_params(
    hamilton_run_config,
    values={"dataset": "warehouse", "feature_set": "extended"},
)

# driver.execute(..., inputs=run.value)
# tracker.log_params(run.params)
```
{% endcode %}

## Execution Boundary

Use `run.value` at the Hamilton execution boundary, and log `run.params` beside Hamilton metadata:

{% code overflow="wrap" %}
```python
# from hamilton import driver
# import my_hamilton_nodes

run = instantiate_with_params(
    hamilton_run_config,
    values={"dataset": "warehouse", "feature_set": "extended"},
)

# dr = driver.Builder().with_modules(my_hamilton_nodes).build()
# result = dr.execute(
#     ["trained_model", "validation_metrics"],
#     inputs=run.value,
# )
# tracker.log_params(run.params)
# tracker.log_metrics(result["validation_metrics"])
```
{% endcode %}

Use Hamilton `inputs=` for runtime choices such as dataset, feature set, and model family. Use Hamilton `config=` only for static graph-construction choices in your Hamilton project. Hypster stays outside Hamilton's graph; it selects and records the values that you pass into the graph.

Hypster does not ship a Hamilton-specific adapter today. This page documents the integration pattern for projects that already use Hamilton.
