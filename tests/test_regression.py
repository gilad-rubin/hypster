import sklearn
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from hypster import HyPSTERRegressor
from hypster.estimators.regression.xgboost import XGBRegressorHypster

SEED = 42

boston = pd.read_csv("../data/boston.csv")
cat_cols = ["CHAS", "RAD"]

X = boston.drop("target", axis=1).copy()
y = boston["target"].copy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=SEED)

def test_regression():
    xgb = XGBRegressorHypster(n_iter_per_round=2)#booster_list=["dart", "gbtree"])
    #xgb_linear = XGBRegressorHypster(booster_list=["gblinear"])
    #estimators = [xgb_tree, xgb_linear]
    estimators = [xgb]

    reg = HyPSTERRegressor(estimators, scoring="neg_mean_squared_error",
                           max_iter=3, cv=3, n_jobs=-1, random_state=SEED)
    reg.fit(X_train, y_train, cat_cols=cat_cols, n_trials=10)

    print(np.sqrt(reg.best_score_))
    preds = reg.predict(X_test)
    score = np.sqrt(sklearn.metrics.mean_squared_error(y_test, preds))
    print(score)
    assert score < 20