import pandas as pd
import numpy as np
import scipy
import sklearn

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel, SelectPercentile, chi2, VarianceThreshold
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn import metrics

import optuna
from optuna.visualization import plot_intermediate_values

import numpy as np
import xgboost as xgb
#import lightgbm as lgb

SEED = 85

from hypster_xgboost import *
from hypster import *

# Get Dataset

from scipy.sparse import csr_matrix, save_npz, load_npz

dataset = "newsgroup" #adult, boston

if dataset=="adult":
    X_train = pd.read_pickle("./data/adult_X_train.pkl")
    y_train = pd.read_pickle("./data/adult_y_train.pkl")
    X_test = pd.read_pickle("./data/adult_X_test.pkl")
    y_test = pd.read_pickle("./data/adult_y_test.pkl")
    cat_columns = X_train.select_dtypes(include="object").columns
elif dataset=="newsgroup":
    X_train = load_npz("./data/X_train.npz")
    y_train = pd.read_pickle("./data/y_train.pkl")
    X_test = load_npz("./data/X_test.npz")
    y_test = pd.read_pickle("./data/y_test.pkl")
    cat_columns=None
else:
    X_train = pd.read_pickle("./data/boston_X_train.pkl")
    y_train = pd.read_pickle("./data/boston_y_train.pkl")
    X_test = pd.read_pickle("./data/boston_X_test.pkl")
    y_test = pd.read_pickle("./data/boston_y_test.pkl")
    cat_columns = None

#X_train = X_train.sample(n=10000, random_state=SEED, axis=0)

#y_train = y_train.iloc[X_train.index].reset_index(drop=True)
#X_train.reset_index(inplace=True, drop=True)

#pipeline - pipeline_objective OR regular pipeline
#consider making pre-made steps with best practices (FS, scaling, etc...) then add option to concat to make one pipeline

#pipeline = Pipeline([("sel", SelectPercentile(chi2))])
#pipe_params = {"sel__percentile" : optuna.distributions.IntUniformDistribution(1,100)}

pipeline = None
pipe_params = None

from optuna.samplers import TPESampler

sampler = TPESampler(**TPESampler.hyperopt_parameters(), seed=SEED)

# sampler = optuna.integration.CmaEsSampler(n_startup_trials=40,
#                       independent_sampler=TPESampler(**TPESampler.hyperopt_parameters()),
#                       warn_independent_sampling=False, seed=SEED)

#TODO: make sure uninstalled estimators don't show up

xgb_linear = XGBClassifierHypster(booster_list=['gblinear'], lr_decay=0.1, n_iter_per_round=2
                                 ,param_dict={#'subsample' : 0.9,
                                              #'eta' : optuna.distributions.LogUniformDistribution(0.8, 1.0)
                                             }
                                 )
#gb_dart = XGBClassifierHypster(booster_list=['dart'])
#xgb_tree = XGBClassifierHypster(booster_list=['gbtree', 'dart'], user_param_dict={'max_depth' : 2})
xgb_tree = XGBClassifierHypster(booster_list=['gbtree', 'dart'],
                                n_iter_per_round=3
                                )
#lgb_estimator = LGBClassifierOptuna()
#sgd_estimator = SGDClassifierOptuna()
#rf_estimator  = RFClassifierOptuna()

estimators = [xgb_linear, xgb_tree]#, sgd|_estimator]

clf = HyPSTERClassifier(estimators, pipeline, pipe_params, save_cv_preds=True,
                        scoring="roc_auc", cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED), tol=1e-5,
                        sampler=sampler, refit=False, random_state=SEED, n_jobs=-1, max_iter=2)
import time
start_time = time.time()

clf.fit(X_train, y_train, cat_columns=cat_columns, n_trials_per_estimator=2)

print("time elapsed: {:.2f}s".format(time.time() - start_time))
print(clf.best_score_)

clf.best_params_

clf.refit(X_train, y_train)

test_preds = clf.predict(X_test)

sklearn.metrics.accuracy_score(y_test, test_preds)

test_probs = clf.predict_proba(X_test)
test_probs = test_probs[:,1]

print(sklearn.metrics.roc_auc_score(y_test, test_probs))

print(clf.best_estimator_.named_steps["model"].get_params()['learning_rates'])