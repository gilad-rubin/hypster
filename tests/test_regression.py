import numpy as np
import sklearn
import scipy
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression

from hypster import HyPSTERRegressor

SEED = 50

def test_melbourne():
    data = pd.read_csv("Melbourne_Data.csv")
    data.head()

    X = data
    X = X.drop(["Sale.price *TARGET*", "Suburb", "Address", "Type", "Method", "Agent.name", "Sale.date", "CouncilArea",
                "Regionname", "latLongTitude", "last_8_weeks_by_Geography.zipcode"], axis=1)
    X.head()

    y = data["Sale.price *TARGET*"]
    y.head()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=SEED)

    # Fit HyPSTER On the Data

    frameworks = ["sklearn"]#, "xgboost"]  # , "lightgbm"]
    model_types = ["tree_based", "linear"]

    reg = HyPSTERRegressor(frameworks=frameworks,
                           model_types=model_types,
                           scoring="neg_mean_squared_error",
                           cv=10,
                           max_iter=20,
                           n_jobs=1,
                           random_state=SEED)

    reg.fit(X_train, y_train, n_trials=5)
    print(reg.best_score_)

def test_dense():
    X, y = make_regression(n_samples=300, n_features=40, n_informative=10, random_state=SEED)
    X = pd.DataFrame(X)

    n = X.shape[0]
    X['A'] = pd.Series(['alpha', 'beta', 'gamma'] * int(n / 4)).head(n)
    X['B'] = pd.Series(np.random.randint(0, 20, n)).astype(str)
    cat_cols = ["A", "B"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=SEED)

    frameworks = ["sklearn", "lightgbm", "xgboost", "sklearn"]
    model_types = ["linear", "tree_based"]

    n_trials = 20

    reg = HyPSTERRegressor(frameworks=frameworks,
                            model_types=model_types,
                            scoring="neg_mean_squared_error",
                            cv=3,
                            max_iter=10,
                            tol=1e-7,
                            max_fails=0,
                            n_jobs=-1,
                            random_state=SEED)

    reg.fit(X_train, y_train, cat_cols=cat_cols, n_trials=n_trials)
    preds = reg.predict(X_test)
    score = np.sqrt(sklearn.metrics.mean_squared_error(y_test, preds))
    assert score < 100

def test_sparse():
    X, y = make_regression(n_samples=500, n_features=40, n_informative=10, random_state=SEED)
    X = scipy.sparse.csr_matrix(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=SEED)
    cat_cols = None

    frameworks = ["sklearn", "lightgbm", "xgboost", "sklearn"]
    model_types = ["linear", "tree_based"]

    n_trials = 20

    reg = HyPSTERRegressor(frameworks=frameworks,
                            model_types=model_types,
                            scoring="neg_mean_squared_error",
                            cv=3,
                            max_iter=10,
                            tol=1e-7,
                            max_fails=0,
                            n_jobs=-1,
                            random_state=SEED)
    reg.fit(X_train, y_train, cat_cols=cat_cols, n_trials=n_trials)
    preds = reg.predict(X_test)
    score = np.sqrt(sklearn.metrics.mean_squared_error(y_test, preds))
    assert score < 100

