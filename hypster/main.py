import pandas as pd
import sklearn

from sklearn.preprocessing import LabelEncoder

SEED = 85

from hypster import HyPSTERClassifier
from hypster.estimators.classification.xgboost import XGBClassifierHypster
# Get Dataset

from scipy.sparse import load_npz

dataset = "adult" #adult, boston

if dataset=="adult":
    X_train = pd.read_pickle("../data/adult_X_train.pkl")
    y_train = pd.read_pickle("../data/adult_y_train.pkl")
    X_test = pd.read_pickle("../data/adult_X_test.pkl")
    y_test = pd.read_pickle("../data/adult_y_test.pkl")
    cat_columns = X_train.select_dtypes(include="object").columns
elif dataset=="newsgroup":
    X_train = load_npz("../data/X_train.npz")
    y_train = pd.read_pickle("../data/y_train.pkl")
    X_test = load_npz("../data/X_test.npz")
    y_test = pd.read_pickle("../data/y_test.pkl")
    cat_columns=None
else:
    X_train = pd.read_pickle("../data/boston_X_train.pkl")
    y_train = pd.read_pickle("../data/boston_y_train.pkl")
    X_test = pd.read_pickle("../data/boston_X_test.pkl")
    y_test = pd.read_pickle("../data/boston_y_test.pkl")
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

#sampler = TPESampler(**TPESampler.hyperopt_parameters(), seed=SEED)

# sampler = optuna.integration.CmaEsSampler(n_startup_trials=40,
#                       independent_sampler=TPESampler(**TPESampler.hyperopt_parameters()),
#                       warn_independent_sampling=False, seed=SEED)

#TODO: make sure uninstalled estimators don't show up

xgb_linear = XGBClassifierHypster(booster_list=['gblinear'], lr_decay=0.1, n_iter_per_round=2
                                 #,param_dict={#'subsample' : 0.9,
                                              #'eta' : optuna.distributions.LogUniformDistribution(0.8, 1.0)
                                 #            }
                                 )
#gb_dart = XGBClassifierHypster(booster_list=['dart'])
#xgb_tree = XGBClassifierHypster(booster_list=['gbtree', 'dart'], user_param_dict={'max_depth' : 2})
xgb_tree = XGBClassifierHypster(booster_list=['gbtree', 'dart'],
                                n_iter_per_round=3
                                )
#
# xgb_linear = XGBRegressorHypster(booster_list=['gblinear'], lr_decay=0.1, n_iter_per_round=2
#                                  #,param_dict={#'subsample' : 0.9,
#                                               #'eta' : optuna.distributions.LogUniformDistribution(0.8, 1.0)
#                                  #            }
#                                  )
#
# xgb_tree = XGBRegressorHypster(booster_list=['gbtree', 'dart'],
#                                 n_iter_per_round=2
#                                 )

#lgb_estimator = LGBClassifierOptuna()
#sgd_estimator = SGDClassifierOptuna()
#rf_estimator  = RFClassifierOptuna()

#estimators = [xgb_linear, xgb_tree]#, sgd|_estimator]
estimators = [xgb_tree, xgb_linear]#, sgd|_estimator]

model = HyPSTERClassifier(estimators, pipeline, pipe_params, save_cv_preds=True,
                          scoring="roc_auc",
                          cv=3, tol=1e-5,
                          refit=False, random_state=SEED, n_jobs=-1, max_iter=5)

# model = HyPSTERRegressor(estimators, pipeline, pipe_params, save_cv_preds=True,
#                         scoring="neg_mean_squared_error", cv=KFold(n_splits=5, random_state=SEED), tol=1e-5,
#                         sampler=sampler, refit=False, random_state=SEED, n_jobs=1, max_iter=30)

import time
start_time = time.time()

model.fit(X_train, y_train, cat_cols=cat_columns, n_trials=10)

print("time elapsed: {:.2f}s".format(time.time() - start_time))
print(model.best_score_)

#model.best_params_

model.refit(X_train, y_train)

test_preds = model.predict(X_test)
sklearn.metrics.accuracy_score(LabelEncoder().fit_transform(y_test), test_preds)
print(model.best_pipeline_.named_steps["model"].get_params()['learning_rates'])

# test_preds = model.predict(X_test)
# print(sklearn.metrics.mean_absolute_error(y_test, test_preds))
# print(np.sqrt(sklearn.metrics.mean_squared_error(y_test, test_preds)))

test_probs = model.predict_proba(X_test)
test_probs = test_probs[:,1]

print(sklearn.metrics.roc_auc_score(y_test, test_probs))