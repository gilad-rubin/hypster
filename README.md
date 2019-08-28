# HyPSTER - HyperParameter optimization on STERoids

### You have just found HyPSTER.
HyPSTER is a brand new Python package built on top of Optuna.
 
It helps you find compact and accurate ML Pipelines while staying light and efficient.

HyPSTER uses state of the art algorithms for sampling hyperparameters (e.g. TPE, CMA-ES) and pruning unpromising trials (e.g. Asynchronous Successive Halving), combined with cross-validated early stopping and adaptive learning rates, all packed up in a simple sklearn API that allows for automatic Preprocessing pipeline selection and supports your favorite ML packages (e.g. XGBoost, LightGBM, CatBoost, SGDClassifier) out of the box.
 
And yes, it supports multi CPU/GPU training.

### Guiding principles
User friendliness

Modularity

Easy extensibility

Work with Python

## Getting started
an XGBoost example:

```python
from hypster import HyPSTERClassifier
from hypster.hypster_xgboost import XGBClassifierHypster

xgb_tree = XGBClassifierHypster(booster_list = ["dart", "gbtree"])
xgb_linear = XGBClassifierHypster(booster_list = ["gblinear"])
estimators = [xgb_tree] #xgb_linear
clf = HyPSTERClassifier(estimators, scoring="roc_auc", max_iter=30, n_jobs=-1, random_state=SEED)
clf.fit(X_train, y_train, cat_cols=cat_cols, n_trials_per_estimator=30)

clf.predict_proba(X_test)

```

## Installation
```bash
> pip install hypster
```

## Contributors

Gilad Rubin

[Tal Peretz](https://www.linkedin.com/in/talper/)
