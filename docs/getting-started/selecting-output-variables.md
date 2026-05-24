---
coverY: 0
layout:
  cover:
    visible: false
    size: full
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

# Select Return Values

Hypster config functions return exactly what you decide to return. There is no framework-specific config object to build. In most application code, the cleanest return value is the initialized object that the caller will actually use.

## Return Initialized Runtime Objects

The common Hypster style is to make the config a typed factory for a runtime object.

{% code overflow="wrap" %}
```python
from sklearn.ensemble import RandomForestClassifier
from hypster import HP, instantiate

def classifier_config(hp: HP) -> RandomForestClassifier:
    n_estimators = hp.int(200, name="n_estimators", min=10, max=1000)
    max_depth = hp.int(None, name="max_depth", min=1, max=100, allow_none=True)
    min_samples_leaf = hp.int(2, name="min_samples_leaf", min=1, max=50)
    random_state = hp.int(42, name="random_state", min=0)

    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
    )

model = instantiate(classifier_config, values={"n_estimators": 500})
assert isinstance(model, RandomForestClassifier)
```
{% endcode %}

The return type annotation helps readers, IDEs, and downstream code understand what instantiation produces.

## Return A Dict When The Runtime Shape Is A Dict

Returning a dict is fine when the object your application needs is actually a mapping. Avoid using dicts as a generic bag of settings that must be assembled somewhere else.

{% code overflow="wrap" %}
```python
from hypster import HP, instantiate

def metric_tags_config(hp: HP) -> dict[str, str]:
    environment = hp.select(["dev", "staging", "prod"], name="environment", default="dev", options_only=True)
    owner = hp.text("ml-platform", name="owner")
    return {"environment": environment, "owner": owner}

tags = instantiate(metric_tags_config, values={"environment": "prod"})
assert tags == {"environment": "prod", "owner": "ml-platform"}
```
{% endcode %}

## Use Dict-Backed Selects For Swappable Objects

Use dict-backed `select` when a parameter should log a simple key but return a more complex runtime value. For swappable configs, map keys to config functions and nest the selected function.

{% code overflow="wrap" %}
```python
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from hypster import HP

def logistic_model(hp: HP) -> LogisticRegression:
    C = hp.float(1.0, name="C", min=1e-4, max=100.0)
    return LogisticRegression(C=C, max_iter=1000)

def forest_model(hp: HP) -> RandomForestClassifier:
    n_estimators = hp.int(200, name="n_estimators", min=10, max=1000)
    max_depth = hp.int(None, name="max_depth", min=1, max=100, allow_none=True)
    min_samples_leaf = hp.int(2, name="min_samples_leaf", min=1, max=50)
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
    )

model_options = {
    "logistic": logistic_model,
    "forest": forest_model,
}

def classifier_config(hp: HP) -> ClassifierMixin:
    selected_config = hp.select(model_options, name="model_family", default="forest", options_only=True)
    return hp.nest(selected_config, name="model")
```
{% endcode %}

The named `model_options` dict is part of the pattern. If the option set is long, keeping it separate from the parent config is usually clearer than embedding a large dictionary inside `hp.select(...)` or growing an `if`/`elif` chain. Nesting also gives UI renderers a natural group for the selected child's controls.

## Use hp.collect

`hp.collect()` helps gather local variables while excluding `hp`, private names, and anything you explicitly exclude.

{% code overflow="wrap" %}
```python
from hypster import HP

def config(hp: HP):
    batch_size = hp.int(32, name="batch_size")
    learning_rate = hp.float(0.001, name="learning_rate")
    helper = "not returned"
    return hp.collect(locals(), exclude=["helper"])
```
{% endcode %}

Use `include=[...]` when you want to whitelist outputs:

{% code overflow="wrap" %}
```python
def config(hp: HP):
    batch_size = hp.int(32, name="batch_size")
    learning_rate = hp.float(0.001, name="learning_rate")
    debug_label = "local"
    return hp.collect(locals(), include=["batch_size", "learning_rate"])
```
{% endcode %}

## Keep Expensive Side Effects Outside Configs

Initializing in-memory objects is a good fit for Hypster configs. Keep expensive effects such as training, network calls, or database writes outside the config function when possible. That keeps `explore()`, HPO, UI generation, and replay fast and predictable.
