# Define A Config Function

A Hypster configuration space is an ordinary Python function whose first argument is named `hp`. It is not a DSL or a separate config file format: use normal Python control flow, lists, helper functions, imports, and object construction.

{% code overflow="wrap" %}
```python
from hypster import HP

def config(hp: HP):
    ...
```
{% endcode %}

`hp` must be the first positional parameter; keyword-only `hp` is rejected before the config executes. The `hp: HP` annotation is recommended, but unannotated config functions are still valid.

The function can return anything your application needs, but the most common pattern is to return the initialized runtime object your application will use. Prefer a return type annotation when the output is a meaningful object.

Because Hypster discovers available parameters by running this function, keep config bodies fast and predictable. Avoid side effects such as writes, training, paid API calls, network calls, database access, or heavyweight initialization inside the config.

## Add Parameters

Use `hp.*` calls for values that should be visible, overrideable, replayable, searchable, or rendered in a UI.

{% code overflow="wrap" %}
```python
from hypster import HP
from my_app.llms import LLMClient

def llm_config(hp: HP) -> LLMClient:
    model_name = hp.select(["gpt-5.5", "gpt-5.5-mini"], name="model_name")
    temperature = hp.float(0.2, name="temperature", min=0.0, max=2.0)
    max_tokens = hp.int(1024, name="max_tokens", min=1, max=16_384)

    return LLMClient(
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
    )
```
{% endcode %}

Every public parameter needs an explicit `name=...`. Names must be valid Python identifiers, so use `batch_size`, not `batch-size` or `model.learning_rate`.

## Use Branches

Hypster is define-by-run. Only parameters touched by the active branch are selected for that run.

{% code overflow="wrap" %}
```python
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from hypster import HP

def model_config(hp: HP) -> ClassifierMixin:
    family = hp.select(["linear", "forest"], name="family", default="forest", options_only=True)

    if family == "linear":
        C = hp.float(1.0, name="C", min=1e-4, max=100.0)
        return LogisticRegression(C=C, max_iter=1000)

    n_estimators = hp.int(200, name="n_estimators", min=10, max=1000)
    max_depth = hp.int(None, name="max_depth", min=1, max=100, allow_none=True)
    return RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
```
{% endcode %}

If `family="linear"`, `n_estimators` and `max_depth` are unreachable. Hypster raises if you pass values for inactive branches.

This define-by-run model is what makes normal Python branches work: Hypster runs the function with the selected values and records the `hp.*` calls it reaches.

## Compose With Nesting

The branch example above works, but for reusable components we recommend composition with `hp.nest()`. Each model gets its own config function, the parent config only chooses which child to run, and interactive UIs can render the nested child parameters as a contained group.

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
    return RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)

model_options = {
    "linear": logistic_model,
    "forest": forest_model,
}

def model_config(hp: HP) -> ClassifierMixin:
    selected_config = hp.select(model_options, name="family", default="forest", options_only=True)
    return hp.nest(selected_config, name="model")
```
{% endcode %}

Nested parameters use dotted paths in `values=`:

{% code overflow="wrap" %}
```python
{"family": "forest", "model.n_estimators": 500}
```
{% endcode %}

Compared with the inline branch version, this keeps the forest parameters under the `model` group and the linear parameters under the same reusable child scope. That makes larger configs easier to scan, test, reuse, and render in interactive UIs.

## Use Dict-Backed Selects For Swappable Components

The nested model example uses a dict-backed `select`: `values=` records the simple key such as `"forest"`, while the config receives the mapped function `forest_model`.

Keep the options mapping in a named variable such as `model_options`, `optimizer_options`, or `retriever_options`. That makes long option sets easier to scan, test, and reuse, while the parent config stays focused on the runtime decision. Add `options_only=True` when values outside the mapping should be rejected.

## Return Initialized Objects

It is often best to let a config return the initialized object your application will use:

{% code overflow="wrap" %}
```python
from sklearn.ensemble import RandomForestClassifier
from hypster import HP

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
```
{% endcode %}

{% hint style="info" %}
Initialized in-memory objects are a good fit when construction is cheap. For SDK clients, remote retrievers, loaded indexes, database handles, training jobs, or writes, return lightweight settings and build the side-effectful object after `instantiate()`. See [Best Practices](../in-depth/basic-best-practices.md).
{% endhint %}

## Next Step

Use [Explore a Configuration Space](exploring-a-configuration-space.md) to inspect the parameter tree before running it.
