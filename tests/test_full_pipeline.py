import sklearn
import pandas as pd

from sklearn.model_selection import train_test_split

from hypster import HyPSTERClassifier
from hypster.classification.xgb_hyptser import XGBClassifierHypster

SEED = 42

adult = pd.read_csv("../data/adult.csv")

cat_cols = ["workclass", "education", "marital-status", "occupation", "relationship", "race", "sex", "native-country"]

X = adult.drop("class", axis=1).copy()
y = adult["class"].copy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=SEED)

def test_classification():
    xgb_tree = XGBClassifierHypster(booster_list=["dart", "gbtree"])
    xgb_linear = XGBClassifierHypster(booster_list=["gblinear"])
    estimators = [xgb_tree, xgb_linear]
    clf = HyPSTERClassifier(estimators, scoring="roc_auc", max_iter=5, cv=3, n_jobs=-1, random_state=SEED)
    clf.fit(X_train, y_train, cat_cols=cat_cols, n_trials_per_estimator=2)
    preds = clf.predict_proba(X_test)
    roc_score = sklearn.metrics.roc_auc_score(y_test, preds[:, 1])
    assert roc_score > 0.85