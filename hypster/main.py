import sklearn
import pandas as pd

from sklearn.model_selection import train_test_split

from hypster import HyPSTERClassifier
from hypster.estimators.classification.xgboost import XGBTreeClassifierHypster, XGBLinearClassifierHypster
from hypster.estimators.classification.lightgbm import LGBClassifierHypster
from hypster.estimators.classification.sgd import SGDClassifierHypster

SEED = 50

dataset="newsgroup" #"newsgroup"
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

n_iters = 2
xgb_tree = XGBTreeClassifierHypster(n_iter_per_round=n_iters, n_jobs=-1)
xgb_linear = XGBLinearClassifierHypster(n_iter_per_round=n_iters, n_jobs=-1)
lgb = LGBClassifierHypster(n_iter_per_round=n_iters, n_jobs=-1)
sgd = SGDClassifierHypster(n_iter_per_round=n_iters, n_jobs=-1)
estimators = [xgb_linear, sgd, lgb, xgb_tree]#, lgb]# xgb_tree, xgb_linear, lgb]


import optuna
from optuna.samplers import TPESampler

n_trials = 30

sampler = optuna.integration.CmaEsSampler(n_startup_trials=int(n_trials/3),
                       independent_sampler=TPESampler(**TPESampler.hyperopt_parameters()),
                       warn_independent_sampling=False, seed=SEED)

sampler = TPESampler(**TPESampler.hyperopt_parameters())
from hypster.linear_extrapolation_pruner import LinearExtrapolationPruner
pruner = LinearExtrapolationPruner(n_steps_back=2, n_steps_forward=10, percentage_from_best=90)

clf = HyPSTERClassifier(estimators,
                        scoring="roc_auc",
                        sampler=sampler,
                        pruner=pruner,
                        max_iter=30,
                        cv=5,
                        n_jobs=-1,
                        tol=1e-7,
                        max_fails=0,
                        random_state=SEED)

clf.fit(X_train, y_train, cat_cols=cat_cols, n_trials=n_trials)
print(clf.best_score_)
preds = clf.predict_proba(X_test)
roc_score = sklearn.metrics.roc_auc_score(y_test, preds[:, 1])
print(roc_score)
print(clf.best_model_.get_params())

# import pandas as pd
# import sklearn
#
# from sklearn.preprocessing import LabelEncoder
#
# SEED = 85
#
# from hypster import HyPSTERClassifier
# from hypster.estimators.classification.xgboost import XGBClassifierHypster
# # Get Dataset
#
# from scipy.sparse import load_npz
#
# dataset = "adult" #adult, boston
#
# if dataset=="adult":
#     X_train = pd.read_pickle("../data/adult_X_train.pkl")
#     y_train = pd.read_pickle("../data/adult_y_train.pkl")
#     X_test = pd.read_pickle("../data/adult_X_test.pkl")
#     y_test = pd.read_pickle("../data/adult_y_test.pkl")
#     cat_columns = X_train.select_dtypes(include="object").columns
# elif dataset=="newsgroup":
#     X_train = load_npz("../data/X_train.npz")
#     y_train = pd.read_pickle("../data/y_train.pkl")
#     X_test = load_npz("../data/X_test.npz")
#     y_test = pd.read_pickle("../data/y_test.pkl")
#     cat_columns=None
# else:
#     X_train = pd.read_pickle("../data/boston_X_train.pkl")
#     y_train = pd.read_pickle("../data/boston_y_train.pkl")
#     X_test = pd.read_pickle("../data/boston_X_test.pkl")
#     y_test = pd.read_pickle("../data/boston_y_test.pkl")
#     cat_columns = None
#
# #X_train = X_train.sample(n=10000, random_state=SEED, axis=0)
#
# #y_train = y_train.iloc[X_train.index].reset_index(drop=True)
# #X_train.reset_index(inplace=True, drop=True)
#
# #pipeline - pipeline_objective OR regular pipeline
# #consider making pre-made steps with best practices (FS, scaling, etc...) then add option to concat to make one pipeline
#
# #pipeline = Pipeline([("sel", SelectPercentile(chi2))])
# #pipe_params = {"sel__percentile" : optuna.distributions.IntUniformDistribution(1,100)}
#
# pipeline = None
# pipe_params = None
#
# #sampler = TPESampler(**TPESampler.hyperopt_parameters(), seed=SEED)
#
# # sampler = optuna.integration.CmaEsSampler(n_startup_trials=40,
# #                       independent_sampler=TPESampler(**TPESampler.hyperopt_parameters()),
# #                       warn_independent_sampling=False, seed=SEED)
#
# #TODO: make sure uninstalled estimators don't show up
#
# xgb_linear = XGBClassifierHypster(booster_list=['gblinear'], lr_decay=0.1, n_iter_per_round=2
#                                  #,param_dict={#'subsample' : 0.9,
#                                               #'eta' : optuna.distributions.LogUniformDistribution(0.8, 1.0)
#                                  #            }
#                                  )
# #gb_dart = XGBClassifierHypster(booster_list=['dart'])
# #xgb_tree = XGBClassifierHypster(booster_list=['gbtree', 'dart'], user_param_dict={'max_depth' : 2})
# xgb_tree = XGBClassifierHypster(booster_list=['gbtree', 'dart'],
#                                 n_iter_per_round=3
#                                 )
# #
# # xgb_linear = XGBRegressorHypster(booster_list=['gblinear'], lr_decay=0.1, n_iter_per_round=2
# #                                  #,param_dict={#'subsample' : 0.9,
# #                                               #'eta' : optuna.distributions.LogUniformDistribution(0.8, 1.0)
# #                                  #            }
# #                                  )
# #
# # xgb_tree = XGBRegressorHypster(booster_list=['gbtree', 'dart'],
# #                                 n_iter_per_round=2
# #                                 )
#
# #lgb_estimator = LGBClassifierOptuna()
# #sgd_estimator = SGDClassifierOptuna()
# #rf_estimator  = RFClassifierOptuna()
#
# #estimators = [xgb_linear, xgb_tree]#, sgd|_estimator]
# estimators = [xgb_tree, xgb_linear]#, sgd|_estimator]
#
# model = HyPSTERClassifier(estimators, pipeline, pipe_params, save_cv_preds=True,
#                           scoring="roc_auc",
#                           cv=3, tol=1e-5,
#                           refit=False, random_state=SEED, n_jobs=-1, max_iter=5)
#
# # model = HyPSTERRegressor(estimators, pipeline, pipe_params, save_cv_preds=True,
# #                         scoring="neg_mean_squared_error", cv=KFold(n_splits=5, random_state=SEED), tol=1e-5,
# #                         sampler=sampler, refit=False, random_state=SEED, n_jobs=1, max_iter=30)
#
# import time
# start_time = time.time()
#
# model.fit(X_train, y_train, cat_cols=cat_columns, n_trials=10)
#
# print("time elapsed: {:.2f}s".format(time.time() - start_time))
# print(model.best_score_)
#
# #model.best_params_
#
# model.refit(X_train, y_train)
#
# test_preds = model.predict(X_test)
# sklearn.metrics.accuracy_score(LabelEncoder().fit_transform(y_test), test_preds)
# print(model.best_pipeline_.named_steps["model"].get_params()['learning_rates'])
#
# # test_preds = model.predict(X_test)
# # print(sklearn.metrics.mean_absolute_error(y_test, test_preds))
# # print(np.sqrt(sklearn.metrics.mean_squared_error(y_test, test_preds)))
#
# test_probs = model.predict_proba(X_test)
# test_probs = test_probs[:,1]
#
# print(sklearn.metrics.roc_auc_score(y_test, test_probs))

requirements = {"include model frameworks" : ["xgboost", "lightgbm", "sklearn",
                                              "catboost", "tf", "keras", "fastai", "all"],
                "exclude model frameworks" : ["xgboost", "lightgbm", "sklearn",
                                              "catboost", "tf", "keras", "fastai", "none"],
                "model types" : ["tree_based", "linear", "deep_learning"],
                "use_cpu" : [True, False],
                "use_gpu" : [True, False],
                "create new features" : [True, False],
                "create uninterpretable features" : [True, False],
                #"pipeline complexity" : ["low", "medium", "high"],
                #"training_speed" : ["fast", "medium", "slow"],
                #"inference speed" : ["fast", "medium", "slow"],
                }

requirements = {"include model frameworks" : ["xgboost", "lightgbm", "sklearn"],
                "exclude model frameworks" : "none",
                "model types" : "all",
                "use_cpu" : True,
                "use_gpu" : False,
                "create new features" : False,
                "create uninterpretable features" : False,
                #"pipeline complexity" : "low",
                # "training_speed" : "medium",
                # "inference speed" : "medium",
                }

# get requested frameworks + check versions (by requirements)
# filter relevant models (types) + initialize if needed by type (e.g. gblinear)
# check if system gpu is compatible. create a list of models with gpu
# initialize based on gpu/cpu

# filter based on speed + cpu/gpu (?)