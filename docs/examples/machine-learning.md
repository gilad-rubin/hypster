# Machine Learning

This example shows the recommended ML shape: each Hypster config returns an initialized model object with a precise return type annotation. The snippets assume your project has scikit-learn installed.

## Define Model Factories

{% code overflow="wrap" %}
```python
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from hypster import HP, explore, instantiate_with_params

def logistic_model(hp: HP) -> LogisticRegression:
    C = hp.float(1.0, name="C", min=1e-4, max=100.0)
    solver = hp.select(["lbfgs", "liblinear"], name="solver", default="lbfgs", options_only=True)

    return LogisticRegression(
        C=C,
        solver=solver,
        max_iter=1000,
    )

def forest_model(hp: HP) -> RandomForestClassifier:
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

## Choose The Active Model

For model families, keep the selectable options in a dict from replayable keys to child config functions. This scales better than a long conditional, keeps each model's hyperparameters close to the object it initializes, and lets UIs show the selected model parameters as one nested group.

{% code overflow="wrap" %}
```python
model_options = {
    "logistic": logistic_model,
    "forest": forest_model,
}

def classifier_config(hp: HP) -> ClassifierMixin:
    selected_config = hp.select(model_options, name="model_family", default="forest", options_only=True)
    return hp.nest(selected_config, name="model")
```
{% endcode %}

The config returns a ready-to-use classifier, not a dictionary that must be assembled later. The selected params record `"forest"` or `"logistic"`, while the application receives the initialized estimator.

## Explore And Instantiate

{% code overflow="wrap" %}
```python
explore(classifier_config)

run = instantiate_with_params(
    classifier_config,
    values={
        "model_family": "forest",
        "model.n_estimators": 500,
        "model.max_depth": 12,
    },
)

model = run.value
params = run.params
```
{% endcode %}

Expected selected params:

{% code overflow="wrap" %}
```python
{
    "model_family": "forest",
    "model.n_estimators": 500,
    "model.max_depth": 12,
    "model.min_samples_leaf": 2,
    "model.random_state": 42,
}
```
{% endcode %}

Use `model.fit(X_train, y_train)` in your training code after instantiation. Keep training itself outside the config function so `explore()`, HPO, and UI generation stay fast.

## Build A Full Pipeline Config

For larger experiments, nest preprocessing and model configs under one parent. Each nested config owns one part of the workflow, while the selected params remain a flat replayable dictionary.

{% code overflow="wrap" %}
```python
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler


def preprocessing_config(hp: HP) -> Pipeline:
    scaler = hp.select(
        [None, "standard", "robust"],
        name="scaler",
        default="standard",
        allow_none=True,
        options_only=True,
    )
    impute_strategy = hp.select(["median", "mean"], name="impute_strategy", default="median", options_only=True)

    steps = [("imputer", SimpleImputer(strategy=impute_strategy))]
    if scaler == "standard":
        steps.append(("scaler", StandardScaler()))
    elif scaler == "robust":
        steps.append(("scaler", RobustScaler()))

    return Pipeline(steps)


def experiment_config(hp: HP) -> Pipeline:
    preprocessing = hp.nest(preprocessing_config, name="preprocessing")
    classifier = hp.nest(classifier_config, name="classifier")
    return Pipeline([("preprocessing", preprocessing), ("classifier", classifier)])


run = instantiate_with_params(
    experiment_config,
    values={
        "preprocessing.scaler": "robust",
        "classifier.model_family": "forest",
        "classifier.model.n_estimators": 500,
    },
)

assert isinstance(run.value.named_steps["classifier"], RandomForestClassifier)
assert run.params["preprocessing.scaler"] == "robust"
assert run.params["classifier.model.n_estimators"] == 500
```
{% endcode %}

Use `run.value.fit(X_train, y_train)` in your training code, and log `run.params` as the exact experiment record.

## Make It HPO-Ready

Add `hpo_spec=` where search semantics matter. Normal instantiation ignores these specs; the Optuna adapter consumes them.

{% code overflow="wrap" %}
```python
from hypster.hpo.types import HpoFloat, HpoInt

def tunable_forest(hp: HP) -> RandomForestClassifier:
    n_estimators = hp.int(
        200,
        name="n_estimators",
        min=50,
        max=1000,
        hpo_spec=HpoInt(step=50),
    )
    max_depth = hp.int(
        12,
        name="max_depth",
        min=2,
        max=64,
        hpo_spec=HpoInt(scale="log"),
    )

    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
    )

def regularized_logistic(hp: HP) -> LogisticRegression:
    C = hp.float(
        1.0,
        name="C",
        min=1e-4,
        max=100.0,
        hpo_spec=HpoFloat(scale="log"),
    )

    return LogisticRegression(C=C, max_iter=1000)
```
{% endcode %}

See [Perform Hyperparameter Optimization](../how-to/perform-hyperparameter-optimization.md) for the Optuna objective pattern.
