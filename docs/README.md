---
icon: hand-wave
---

# Welcome

<div data-full-width="false">

<figure><img src=".gitbook/assets/hypster_with_text.png" alt=""><figcaption></figcaption></figure>

</div>

### **Hypster is a lightweight framework for defining and optimizing ML & AI Workflows.**

### Key Features

* :snake: **Pythonic API**: Intuitive & minimal syntax that feels natural to Python developers
* :nesting\_dolls: **Hierarchical Configurations**: Support for nested and swappable configurations
* :triangular\_ruler: **Type Safety**: Built-in type hints and validation using [`Pydantic`](https://github.com/pydantic/pydantic)
* :package: **Portability**: Easy serialization and loading of configurations
* :test\_tube: **Experiment Ready**: Built-in support for hyperparameter optimization
* :video\_game: **Interactive UI**: Jupyter widgets integration for interactive parameter selection

> Show your support by giving us a [star](https://github.com/gilad-rubin/hypster)! ⭐&#x20;

### How Does it work?

{% stepper %}
{% step %}
#### Install Hypster

```bash
pip install hypster
```
{% endstep %}

{% step %}
#### Define a configuration space

```python
from hypster import config, HP


@config
def my_config(hp: HP):
    model = hp.select(["gpt-4o", "claude-3-5-sonnet"], default="gpt-4o")
    temperature = hp.number(0.5)
    llm = LLM(model=model, temperature=temperature)
```
{% endstep %}

{% step %}
#### Instantiate your configuration

```python
results = my_config(values={"model" : "claude-3-5-sonnet"})
```
{% endstep %}

{% step %}
#### Define an execution function

```python
def generate(llm: LLM, prompt: str):
    return llm.invoke(prompt)
```
{% endstep %}

{% step %}
#### Execute!

```python
generate(results["llm"], prompt="What is Hypster?")
```
{% endstep %}
{% endstepper %}

## Discover Hypster

<table data-view="cards"><thead><tr><th></th><th></th><th></th><th data-hidden data-card-cover data-type="files"></th><th data-hidden data-card-target data-type="content-ref"></th></tr></thead><tbody><tr><td><strong>Getting Started</strong></td><td>How to create &#x26; instantiate Hypster configs</td><td></td><td><a href=".gitbook/assets/Group 4 (5).png">Group 4 (5).png</a></td><td><a href="getting-started/installation.md">installation.md</a></td></tr><tr><td><strong>Best Practices</strong></td><td>How to make the most out of Hypster in your AI/ML project</td><td></td><td><a href=".gitbook/assets/Group 26.png">Group 26.png</a></td><td><a href="in-depth/basic-best-practices.md">basic-best-practices.md</a></td></tr><tr><td><strong>Tutorials</strong></td><td>Learn step-by-step tutorials for ML &#x26; Generative AI use-cases </td><td></td><td><a href=".gitbook/assets/Group 53.png">Group 53.png</a></td><td><a href="getting-started/tutorials/">tutorials</a></td></tr></tbody></table>

## Why Use Hypster?

In modern AI/ML development, we often need to handle **multiple configurations across different scenarios**. This is essential because:

1. We don't know in advance **which** **hyperparameters** will best optimize our performance metrics and satisfy our constraints.
2. We need to support **multiple "modes"** for different scenarios. For example:
   1. Local vs. Remote Environments, Development vs. Production Settings
   2. Different App Configurations for specific use-cases and populations

Hypster takes care of these challenges by providing a simple way to define configuration spaces and instantiate them into concrete workflows, turning your codebase into a "superposition" of workflows.
