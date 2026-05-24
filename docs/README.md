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

## Hypster is a lightweight configuration framework for managing and **optimizing AI & ML workflows**

> Hypster is in preview and is not ready for production use.
>
> We're working hard to make Hypster stable and feature-complete, but until then, expect to encounter bugs, missing features, and occasional breaking changes.

## Key Features

* :snake: **Pure Python, Not a DSL**: Use normal functions, `if` statements, loops, helper functions, imports, and real object construction
* :nesting\_dolls: **Hierarchical, Conditional Configurations**: Support for nested and swappable runtime components
* :triangular\_ruler: **Type Safety**: Built-in type hints and validation
* :mag: **Schema Exploration**: Inspect parameters, defaults, and active branches with `explore()`
* :test_tube: **Hyperparameter Optimization Built-In**: Native, first-class optuna support

Hypster configs are ordinary Python functions rather than a separate configuration language. That keeps them flexible and readable, but it also means Hypster discovers the available parameters by executing the function. Keep config functions fast and side-effect-free: avoid paid API calls, network calls, file writes, training, database access, and costly initialization in code paths used by `explore()`, HPO, or interactive UIs.

> Show your support by giving us a [star](https://github.com/gilad-rubin/hypster)! ⭐

## How Does it work?

{% stepper %}
{% step %}
**Install Hypster**

{% code overflow="wrap" %}
```bash
uv add hypster
```
{% endcode %}

Or using pip:

{% code overflow="wrap" %}
```bash
pip install hypster
```
{% endcode %}

{% code overflow="wrap" %}
```bash
# optional notebook visualization UI
uv add 'hypster[viz]'
# optional HPO backend
uv add 'hypster[optuna]'
```
{% endcode %}

{% code overflow="wrap" %}
```bash
# optional notebook visualization UI
pip install 'hypster[viz]'
# optional HPO backend
pip install 'hypster[optuna]'
```
{% endcode %}
{% endstep %}

{% step %}
**Define a configuration space**

{% code overflow="wrap" %}
```python
from hypster import HP
from my_app.llms import LLMClient


def llm_config(hp: HP) -> LLMClient:
    model_name = hp.select(
        ["gpt-5", "claude-sonnet-4-6", "gemini-3.5-flash"],
        name="model_name",
        default="gpt-5",
        options_only=True,
    )
    temperature = hp.float(0.0, name="temperature", min=0.0, max=1.0)
    return LLMClient(model_name=model_name, temperature=temperature)
```
{% endcode %}
{% endstep %}

{% step %}
**Explore your configuration**

{% code overflow="wrap" %}
```python
from hypster import explore

explore(llm_config)
```
{% endcode %}
{% endstep %}

{% step %}
**Instantiate your runtime object**

{% code overflow="wrap" %}
```python
from hypster import instantiate

llm = instantiate(llm_config, values={"model_name": "gpt-5", "temperature": 0.7})
```
{% endcode %}
{% endstep %}

{% step %}
**Execute!**

{% code overflow="wrap" %}
```python
llm.invoke("What is Hypster?")
```
{% endcode %}
{% endstep %}
{% endstepper %}

## Discover Hypster

<table data-view="cards"><thead><tr><th></th><th></th><th></th><th data-hidden data-card-cover data-type="files"></th><th data-hidden data-card-target data-type="content-ref"></th></tr></thead><tbody><tr><td><strong>Getting Started</strong></td><td>How to create &#x26; instantiate Hypster configs</td><td></td><td><a href=".gitbook/assets/Group 4 (5).png">Group 4 (5).png</a></td><td><a href="getting-started/installation.md">installation.md</a></td></tr><tr><td><strong>Tutorials</strong></td><td>Step-by-step guides for ML &#x26; Generative AI use-cases</td><td></td><td><a href=".gitbook/assets/Group 53.png">Group 53.png</a></td><td><a href="examples/">examples</a></td></tr><tr><td><strong>Best Practices</strong></td><td>How to make the most out of Hypster</td><td></td><td><a href=".gitbook/assets/Group 26.png">Group 26.png</a></td><td><a href="in-depth/basic-best-practices.md">basic-best-practices.md</a></td></tr></tbody></table>

## Why Use Hypster?

In modern AI/ML development, we often need to handle **multiple configurations across different scenarios**. This is essential because:

1. We don't know in advance which **hyperparameters** will best optimize our performance metrics and satisfy our constraints.
2. We need to support multiple **"modes"** for different scenarios. For example:
   1. Local vs. Remote Environments, Development vs. Production Settings
   2. Different App Configurations for specific use-cases and populations

Hypster takes care of these challenges by providing a simple way to define configuration spaces and instantiate them into concrete workflows. This enables you to manage and optimize swappable runtime components in your codebase.

## Core Workflow

* **Define** ordinary Python config functions whose first argument is `hp: HP`.
* **Return** the initialized object your application will use whenever that object is cheap and safe to construct.
* **Choose** swappable components with named option dictionaries that map simple keys to config functions.
* **Explore** the active parameter tree with `explore(config)` before running a branch.
* **Instantiate** with `instantiate(config, values={...})` when you only need the returned object.
* **Log params** with `instantiate_with_params(config, values={...})` when you need a stable replay record.


## Design Notes

Hypster treats `values=` as a reproducibility surface. Unknown values and values for inactive branches raise by default, because silently accepting them can make an experiment impossible to replay. Use `explore(config, values=...)` to inspect a branch before instantiating it.

Because exploration and interactive controls execute the config function to discover the current branch, avoid doing work there that should happen only once or only after the user confirms a run. Build expensive clients, load indexes, write files, call paid APIs, and train models after `instantiate()` returns.

## Additional Reading

* [Introducing Hypster](https://medium.com/@giladrubin/introducing-hypster-a-pythonic-framework-for-managing-configurations-to-build-highly-optimized-ai-5ee004dbd6a5)
* [Implementing Modular RAG With Haystack & Hypster](https://towardsdatascience.com/implementing-modular-rag-with-haystack-and-hypster-d2f0ecc88b8f)
* [5 Pillars for Hyper-Optimized AI Workflows](https://medium.com/@giladrubin/5-pillars-for-a-hyper-optimized-ai-workflow-21fcaefe48ca)
