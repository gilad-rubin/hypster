import sklearn
import scipy
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

from hypster import HyPSTERClassifier

SEED = 50

def test_flow():
    X, y = make_classification(n_samples=300, n_features=40, n_informative=30, random_state=SEED)
    X = pd.DataFrame(X)

    # TODO add categorical
    n = X.shape[0]
    X['A'] = pd.Series(['alpha', 'beta', 'gamma'] * int(n / 4)).head(n)
    X['B'] = pd.Series(np.random.randint(0, 20, n)).astype(str)
    X['C'] = pd.Series(np.random.randint(0, 15, n)).astype(str)
    cat_cols = ["A", "B", "C"]
    # cat_cols = None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=SEED)

    frameworks = ["xgboost", "lightgbm", "sklearn"]
    model_types = ["linear", "tree_based"]

    n_trials = 20

    pre_proc = []
    proc_dict = [("categorical", ["A", "B"], ["encode", "normalize"]),
                 ("cat_other", ["C"], ["encode"])]
    post_proc = ["feature selection", "scale"]

    clf = HyPSTERClassifier(frameworks=frameworks,
                            model_types=model_types,
                            pre_proc=pre_proc,
                            proc_dict=proc_dict,
                            post_proc=post_proc,
                            scoring="roc_auc",
                            cv=3,
                            max_iter=10,
                            tol=1e-5,
                            max_fails=0,
                            n_jobs=-1,
                            random_state=SEED)

    clf.fit(X_train, y_train, cat_cols=cat_cols, n_trials=n_trials)
    preds = clf.predict_proba(X_test)
    roc_score = sklearn.metrics.roc_auc_score(y_test, preds[:, 1])
    assert roc_score > 0.5