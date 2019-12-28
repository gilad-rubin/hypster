import sklearn
import pandas as pd
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import train_test_split

from hypster import HyPSTERClassifier
from hypster.estimators.classification.xgboost import XGBTreeClassifierHypster, XGBLinearClassifierHypster
from hypster.estimators.classification.lightgbm import LGBClassifierHypster
from hypster.estimators.classification.sgd import SGDClassifierHypster

SEED = 50

dataset="newsgroup" #adult
if dataset == "adult":
    adult = pd.read_csv("../data/adult.csv")
    cat_cols = ["workclass", "education", "marital-status", "occupation", "relationship", "race", "sex", "native-country"]
    X = adult.drop("class", axis=1).copy()
    y = adult["class"].copy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=SEED)
elif dataset=="newsgroup":
    from scipy.sparse import load_npz
    X_train = load_npz("../data/X_train.npz")
    y_train = pd.read_pickle("../data/y_train.pkl")
    X_test = load_npz("../data/X_test.npz")
    y_test = pd.read_pickle("../data/y_test.pkl")
    cat_cols=None

def test_classification():
    n_iters = 1
    xgb_tree = XGBTreeClassifierHypster(n_iter_per_round=n_iters)
    xgb_linear = XGBLinearClassifierHypster(n_iter_per_round=n_iters)
    lgb = LGBClassifierHypster(n_iter_per_round=n_iters)
    sgd = SGDClassifierHypster(n_iter_per_round=n_iters)
    estimators = [sgd, lgb, xgb_tree, xgb_linear]

    n_trials = 30
    # sampler = optuna.integration.CmaEsSampler(n_startup_trials=int(n_trials / 3),
    #                                           independent_sampler=TPESampler(**TPESampler.hyperopt_parameters()),
    #                                           warn_independent_sampling=False, seed=SEED)
    sampler = TPESampler(**TPESampler.hyperopt_parameters())
    from hypster.linear_extrapolation_pruner import LinearExtrapolationPruner
    pruner = LinearExtrapolationPruner(n_steps_back=2, n_steps_forward=10, percentage_from_best=90)

    clf = HyPSTERClassifier(estimators,
                            scoring="roc_auc",
                            sampler=sampler,
                            pruner=pruner,
                            max_iter=10,
                            cv=3,
                            n_jobs=-1,
                            tol=1e-5,
                            max_fails=0,
                            random_state=SEED)

    clf.fit(X_train, y_train, cat_cols=cat_cols, n_trials=n_trials)
    preds = clf.predict_proba(X_test)
    roc_score = sklearn.metrics.roc_auc_score(y_test, preds[:, 1])
    print(roc_score)
    print(clf.best_model_.get_params())

    assert roc_score > 0.8