---
layout:
  width: default
  title:
    visible: true
  description:
    visible: true
  tableOfContents:
    visible: true
  outline:
    visible: true
  pagination:
    visible: true
  metadata:
    visible: true
---

# 👋 Welcome

<div data-full-width="false"><figure><picture><source srcset=".gitbook/assets/hypster_text_white_text.png" media="(prefers-color-scheme: dark)"><img src=".gitbook/assets/hypster_with_text (1).png" alt=""></picture><figcaption></figcaption></figure></div>

## Hypster is a lightweight configuration framework for building, exploring, replaying, and optimizing Python workflows.

> Hypster is in preview and is not ready for production use.
>
> We're working hard to make Hypster stable and feature-complete, but until then, expect to encounter bugs, missing features, and occasional breaking changes.

Hypster is designed for projects where a "configuration" is not just a flat settings file. It works well when you need conditional branches, nested components, ML and AI experiments, data pipelines, or a UI/search system that needs to discover the active parameter space before running code.

Hypster configs are pure Python functions, not a separate DSL. You can use normal `if` statements, loops, lists, helper functions, imports, dataclasses, and initialized objects. The tradeoff is important: to discover which parameters exist for a branch, Hypster runs the config function. Keep that function fast and free of side effects or surprise costs, especially when using `explore()` or `interact()`.

## First Example

{% code overflow="wrap" %}
```python
from dataclasses import dataclass
from hypster import HP, explore, instantiate_with_params

@dataclass
class ForestTrainingConfig:
    seed: int
    n_estimators: int
    max_depth: int | None

@dataclass
class LinearTrainingConfig:
    seed: int
    alpha: float

def train_config(hp: HP) -> ForestTrainingConfig | LinearTrainingConfig:
    model = hp.select(["linear", "forest"], name="model", default="forest")
    seed = hp.int(42, name="seed", min=0)

    if model == "forest":
        n_estimators = hp.int(200, name="n_estimators", min=10, max=1000)
        max_depth = hp.int(None, name="max_depth", allow_none=True)
        return ForestTrainingConfig(seed=seed, n_estimators=n_estimators, max_depth=max_depth)

    alpha = hp.float(0.1, name="alpha", min=0.0, max=10.0)
    return LinearTrainingConfig(seed=seed, alpha=alpha)

explore(train_config)

run = instantiate_with_params(
    train_config,
    values={"model": "forest", "n_estimators": 500},
)

print(run.value)
print(run.params)
```
{% endcode %}

The output value is the typed runtime object you use in your application. The `params` sidecar is the replayable record of every parameter Hypster selected on that run, including defaults.

## Why Use Hypster?

* **Plain Python configs**: a config function is an ordinary function whose first argument is `hp: HP`.
* **Typed runtime outputs**: return initialized objects directly, with a return type annotation when the output type matters.
* **Branch-aware exploration**: `explore()` shows the active parameter tree, including nested branches selected by `values=`.
* **Replayable runs**: `instantiate_with_params()` records the selected parameter paths for experiment tracking and production replay.
* **Nested composition**: use `hp.nest()` to assemble workflows from smaller config functions.
* **Typed validation**: numeric bounds, scalar types, nullable values, select options, and unknown overrides are checked at the API boundary.
* **HPO bridge**: `hypster.hpo.optuna.suggest_values()` converts the same config function into an Optuna search space.

## Install

{% code overflow="wrap" %}
```bash
uv add hypster
```
{% endcode %}

For Optuna support:

{% code overflow="wrap" %}
```bash
uv add 'hypster[optuna]'
```
{% endcode %}

For the notebook visualization UI:

{% code overflow="wrap" %}
```bash
uv add 'hypster[viz]'
```
{% endcode %}

## Where To Go Next

<table data-view="cards"><thead><tr><th></th><th></th><th data-hidden data-card-target data-type="content-ref"></th></tr></thead><tbody><tr><td><strong>Getting Started</strong></td><td>Install Hypster, define your first config, explore branches, and instantiate values.</td><td><a href="getting-started/installation.md">installation.md</a></td></tr><tr><td><strong>Examples</strong></td><td>Copy patterns for ML, data processing, AI workflows, nested configs, UI generation, and experiment tracking.</td><td><a href="examples/">examples</a></td></tr><tr><td><strong>Reference</strong></td><td>Exact public API signatures, parameter behavior, error handling, and Optuna integration facts.</td><td><a href="reference/api.md">api.md</a></td></tr></tbody></table>

## Core Workflow

{% stepper %}
{% step %}
**Define a config function**

Use `hp.*` calls for values that should be visible, overrideable, replayable, or searchable.
{% endstep %}

{% step %}
**Explore the active schema**

Run `explore(config)` for a tree, or `explore(config, return_info=True)` for JSON-serializable metadata.
{% endstep %}

{% step %}
**Instantiate the runtime object**

Call `instantiate(config, values={...})` when you only need the returned value.
{% endstep %}

{% step %}
**Log the replayable params**

Call `instantiate_with_params(config, values={...})` when you need a stable record for experiments, UI state, or production replay.
{% endstep %}
{% endstepper %}

## Design Notes

Hypster treats `values=` as a reproducibility surface. Unknown values and values for inactive branches raise by default, because silently accepting them can make an experiment impossible to replay. Use `explore(config, values=...)` to inspect a branch before instantiating it.

Because exploration and interactive controls execute the config function to discover the current branch, avoid doing work there that should happen only once or only after the user confirms a run. Build expensive clients, load indexes, write files, call paid APIs, and train models after `instantiate()` returns.

## Additional Reading

* [Introducing Hypster](https://medium.com/@giladrubin/introducing-hypster-a-pythonic-framework-for-managing-configurations-to-build-highly-optimized-ai-5ee004dbd6a5)
* [Implementing Modular RAG With Haystack & Hypster](https://towardsdatascience.com/implementing-modular-rag-with-haystack-and-hypster-d2f0ecc88b8f)
* [5 Pillars for Hyper-Optimized AI Workflows](https://medium.com/@giladrubin/5-pillars-for-a-hyper-optimized-ai-workflow-21fcaefe48ca)
