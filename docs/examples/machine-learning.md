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

{% code overflow="wrap" %}
```python
def classifier_config(hp: HP) -> ClassifierMixin:
    model_family = hp.select(["logistic", "forest"], name="model_family", default="forest", options_only=True)

    if model_family == "logistic":
        return hp.nest(logistic_model, name="logistic")

    return hp.nest(forest_model, name="forest")
```
{% endcode %}

The config returns a ready-to-use classifier, not a dictionary that must be assembled later.

## Explore And Instantiate

{% code overflow="wrap" %}
```python
explore(classifier_config)

run = instantiate_with_params(
    classifier_config,
    values={
        "model_family": "forest",
        "forest.n_estimators": 500,
        "forest.max_depth": 12,
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
    "forest.n_estimators": 500,
    "forest.max_depth": 12,
    "forest.min_samples_leaf": 2,
    "forest.random_state": 42,
}
```
{% endcode %}

Use `model.fit(X_train, y_train)` in your training code after instantiation. Keep training itself outside the config function so `explore()`, HPO, and UI generation stay fast.

## Build A Full Pipeline Config

For larger experiments, nest preprocessing, training, and model configs under one parent. Each nested config owns one part of the workflow, while the selected params remain a flat replayable dictionary.

{% code overflow="wrap" %}
```python
from dataclasses import dataclass


@dataclass(frozen=True)
class PreprocessingSettings:
    scaler: str | None
    impute_strategy: str


@dataclass(frozen=True)
class TrainingSettings:
    cv_folds: int
    scoring: str


@dataclass(frozen=True)
class MLExperiment:
    preprocessing: PreprocessingSettings
    training: TrainingSettings
    model: ClassifierMixin


def preprocessing_config(hp: HP) -> PreprocessingSettings:
    return PreprocessingSettings(
        scaler=hp.select(
            [None, "standard", "robust"],
            name="scaler",
            default="standard",
            allow_none=True,
            options_only=True,
        ),
        impute_strategy=hp.select(["median", "mean"], name="impute_strategy", default="median", options_only=True),
    )


def training_settings(hp: HP) -> TrainingSettings:
    return TrainingSettings(
        cv_folds=hp.int(5, name="cv_folds", min=2, max=10),
        scoring=hp.select(["accuracy", "f1", "roc_auc"], name="scoring", default="f1", options_only=True),
    )


def experiment_config(hp: HP) -> MLExperiment:
    return MLExperiment(
        preprocessing=hp.nest(preprocessing_config, name="preprocessing"),
        training=hp.nest(training_settings, name="training"),
        model=hp.nest(classifier_config, name="model"),
    )


run = instantiate_with_params(
    experiment_config,
    values={
        "preprocessing.scaler": "robust",
        "training.cv_folds": 3,
        "model.model_family": "forest",
        "model.forest.n_estimators": 500,
    },
)

assert isinstance(run.value.model, RandomForestClassifier)
assert run.params["preprocessing.scaler"] == "robust"
assert run.params["model.forest.n_estimators"] == 500
```
{% endcode %}

Use `run.value.preprocessing` to build your feature transformer, `run.value.model` as the initialized estimator, and `run.params` as the exact experiment record.

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
