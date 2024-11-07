---
layout:
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
---

# 👋 Welcome

<div data-full-width="false">

<figure><picture><source srcset=".gitbook/assets/hypster_text_white_text.png" media="(prefers-color-scheme: dark)"><img src=".gitbook/assets/hypster_with_text (1).png" alt=""></picture><figcaption></figcaption></figure>

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
def llm_config(hp: HP):
    model_name = hp.select(["gpt-4o-mini", "gpt-4o"])
    temperature = hp.number(0.0, min=0.0, max=1.0)
    max_tokens = hp.int(256, max=2048)
```
{% endstep %}

{% step %}
#### Instantiate your configuration

```python
results = my_config(values={"model" : "gpt-4o"})
```
{% endstep %}

{% step %}
#### Define an execution function

```python
def generate(prompt: str, 
             model_name: str, 
             temperature: float, 
             max_tokens: int) -> str:
    model = llm.get_model(model_name)
    response = model.prompt(prompt, 
                            temperature=temperature, 
                            max_tokens=max_tokens)
    return response
```
{% endstep %}

{% step %}
#### Execute!

```python
generate(prompt="What is Hypster?", **results)
```
{% endstep %}
{% endstepper %}

## Discover Hypster

<table data-view="cards"><thead><tr><th></th><th></th><th></th><th data-hidden data-card-cover data-type="files"></th><th data-hidden data-card-target data-type="content-ref"></th></tr></thead><tbody><tr><td><strong>Getting Started</strong></td><td>How to create &#x26; instantiate Hypster configs</td><td></td><td><a href=".gitbook/assets/Group 4 (5).png">Group 4 (5).png</a></td><td><a href="getting-started/installation.md">installation.md</a></td></tr><tr><td><strong>Tutorials</strong></td><td>Step-by-step guides for ML &#x26; Generative AI use-cases </td><td></td><td><a href=".gitbook/assets/Group 53.png">Group 53.png</a></td><td><a href="getting-started/usage-examples/">usage-examples</a></td></tr><tr><td><strong>Best Practices</strong></td><td>How to make the most out of Hypster</td><td></td><td><a href=".gitbook/assets/Group 26.png">Group 26.png</a></td><td><a href="in-depth/basic-best-practices.md">basic-best-practices.md</a></td></tr></tbody></table>

## Why Use Hypster?

In modern AI/ML development, we often need to handle **multiple configurations across different scenarios**. This is essential because:

1. We don't know in advance which **hyperparameters** will best optimize our performance metrics and satisfy our constraints.
2. We need to support multiple **"modes"** for different scenarios. For example:
   1. Local vs. Remote Environments, Development vs. Production Settings
   2. Different App Configurations for specific use-cases and populations

Hypster takes care of these challenges by providing a simple way to define configuration spaces and instantiate them into concrete workflows. This enables you to easily manage and optimize multiple configurations in your codebase.

## Additional Reading

Explore these articles to deepen your understanding of Hypster and its applications:

### Core Concepts

* [Introducing Hypster: A Pythonic Framework for Managing Configurations](https://medium.com/@giladrubin/introducing-hypster-a-pythonic-framework-for-managing-configurations-to-build-highly-optimized-ai-5ee004dbd6a5)
  * Overview of Hypster's core concepts
  * Understanding configuration spaces
  * Basic usage patterns and examples

### Practical Applications - Modular RAG

* [Implementing Modular RAG with Haystack and Hypster](https://towardsdatascience.com/implementing-modular-rag-with-haystack-and-hypster-d2f0ecc88b8f)
  * Real-world example of building modular RAG systems
  * Integration with Haystack
  * Advanced configuration patterns

### The Philosophy Behind Hypster

* [5 Pillars for a Hyper-Optimized AI Workflow](https://medium.com/@giladrubin/5-pillars-for-a-hyper-optimized-ai-workflow-21fcaefe48ca)
  * Design principles for AI workflows
  * Optimization strategies
  * System architecture considerations
