from lightgbm.sklearn import LGBMClassifier
import numpy as np
from ..lightgbm import LGBModelHypster

class LGBClassifierHypster(LGBModelHypster):
    def get_tags(self):
        self.tags = {'name' : "LightGBM Classifier",
                    'model type': "tree",
                    'supports regression': False,
                    'supports ranking': False,
                    'supports classification': True,
                    'supports multiclass': True,
                    'supports multilabel': False,
                    'handles categorical' : False, #change to True
                    'handles categorical nan': False, #TODO: test
                    'handles sparse': True,
                    'handles numeric nan': True,
                    'nan value when sparse': 0, #TODO: test
                    'sensitive to feature scaling': False,
                    'has predict_proba' : True,
                    'has model embeddings': False,
                    'adjustable model complexity' : False #TODO: change to True
                    }
        return self.tags

    def choose_and_set_params(self, trial, class_counts, missing):
        self.trial = trial
        n_classes = len(class_counts)

        #TODO change according to Laurae
        model_params = {'n_jobs': -1,
                        'verbose' : -1,
                        'random_state': self.random_state,
                        #'use_missing' 'zero_as_missing'
                        'learning_rate': self.sample_hp('learning_rate', "log-uniform", [1e-3, 1.0]),
                        'boosting_type': self.sample_hp('boosting', "categorical", ['gbdt', 'goss', 'dart']), #'rf'
                        'max_depth': self.sample_hp('max_depth', "int", [2, 20]),#TODO: maybe change to higher range? (for competition or production mode or bigger datasets)
                        'min_child_samples': self.sample_hp('min_child_samples', "int", [1, 30]),
                        'min_child_weight': self.sample_hp('lgb_min_child_weight', "log-uniform", [1e-3, 30.0]),
                        'colsample_bytree': self.sample_hp('colsample_bytree', "uniform", [0.1, 1.0]),
                        #TODO check if it works on sklearn classifier
                        #'feature_fraction_bynode': self.sample_hp('feature_fraction_bynode', "uniform", [0.1, 1.0]),
                        #max_delta_step?
                        # min_split_gain
                        'reg_alpha': self.sample_hp('reg_alpha', "log-uniform", [1e-10, 1.0]),
                        'reg_lambda': self.sample_hp('reg_lambda', "log-uniform", [1e-10, 1.0])
                 }
        if model_params["max_depth"] > 12:
            model_params['num_leaves'] = 100
        else:
            model_params['num_leaves'] = 40 #TODO: change
            #max_leaves = np.power(2, model_params["max_depth"])
            #model_params['num_leaves'] = int(self.sample_hp('num_leaves', "log-uniform",
            #                                        [max_leaves/2, max_leaves]))  # TODO: change?
        if n_classes == 2:
            binary_objectives = ['binary', 'cross_entropy', 'cross_entropy_lambda']
            model_params['objective'] = self.sample_hp('binary objective', 'categorical', binary_objectives)
            if model_params["objective"] == "binary": #TODO: check if works with other objectives

                model_params['is_unbalance'] = self.sample_hp("is_unbalance", "categorical", [False, True])
                # TODO change to "categorical" [1,pos_weight] ?
                # pos_weight = class_counts[0] / class_counts[1]
                #model_params['scale_pos_weight'] = self.sample_hp("scale_pos_weight", "uniform", [1, pos_weight])

        else:  # multiclass
            multiclass_objectives = ["multiclass", "multiclassova"]
            model_params['objective'] = self.sample_hp('multiclass objective', 'categorical', multiclass_objectives)
            model_params["num_class"] = n_classes
            model_params['is_unbalance'] = self.sample_hp("is_unbalance", "categorical", [False, True])
            #TODO change base and sample weight on DMatrix
            #change base score to class priors (https://github.com/dmlc/xgboost/issues/1380)
            #change sample weight by multiplying class_weight and sample weight

        #TODO: check out pos_bagging and neg_bagging
        if model_params["boosting_type"] != "goss":
            model_params['subsample_freq'] = 1
            model_params['subsample'] = self.sample_hp("subsample", "uniform", [0.4, 1.0])

        #if model_params["objective"] in ["binary", "multiclassova", "cross_entropy"]:
        model_params['boost_from_average'] = True

        if model_params['boosting_type'] == 'dart':
            model_params['xgboost_dart_mode'] = self.sample_hp('xgboost_dart_mode', "categorical", [False, True])
            model_params['drop_rate'] = self.sample_hp('drop_rate', "log_uniform", [1e-8, 1.0])
            model_params['skip_drop'] = self.sample_hp('skip_drop', "log_uniform", [1e-8, 1.0])
        if model_params['boosting_type'] == 'goss':
            model_params['top_rate'] = self.sample_hp('top_rate', "uniform", [0.0, 1.0])
            model_params['other_rate'] = self.sample_hp('other_rate', "uniform", [0.0, 1.0 - model_params['top_rate']])
        # if model_params['boosting_type'] == 'rf':
        #     model_params['bagging_freq'] = 2

        #TODO: add categorical hps. starting at "min_data_per_group"
        # https://lightgbm.readthedocs.io/en/latest/Parameters.html

        self.model_params = model_params

    def predict_proba(self):
        class_probs = self.current_model.predict(self.dtest)
        if self.model_params['objective'] in ["multiclass", "multiclassova"]:
            return class_probs

        classone_probs = class_probs
        classzero_probs = 1.0 - classone_probs
        return np.vstack((classzero_probs, classone_probs)).transpose()

    def create_model(self):
        #TODO: if learning rates are identical throughout - create a regular Classifier
        if "is_unbalance" in self.model_params:
            is_unbalance = self.model_params.pop("is_unbalance")
            self.model_params["class_weight"] = "balanced" if is_unbalance else None

        self.model_params['n_estimators'] = self.best_n_iterations
        self.model_params["learning_rate"] = self.learning_rates[0] #TODO change
        
        final_model = LGBMClassifier(**self.model_params)
        return final_model

# class XGBClassifierLR(XGBClassifier):
#     def __init__(self, learning_rates = None,
#                  max_depth=3, learning_rate=0.1, n_estimators=100,
#                  verbosity=1,
#                  objective="binary:logistic", booster='gbtree',
#                  n_jobs=1, nthread=None, gamma=0, min_child_weight=1, max_delta_step=0,
#                  subsample=1, colsample_bytree=1, colsample_bylevel=1,
#                  colsample_bynode=1, reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
#                  base_score=0.5, random_state=0, seed=None, missing=None, **kwargs):
#
#         if 'learning_rates' in kwargs:
#             self.learning_rates = kwargs.pop('learning_rates')
#         else:
#             self.learning_rates = learning_rates
#
#         super(XGBClassifierLR, self).__init__(
#             max_depth=max_depth, learning_rate=learning_rate, n_estimators=n_estimators,
#             verbosity=verbosity, objective=objective, booster=booster,
#             n_jobs=n_jobs, nthread=nthread, gamma=gamma,
#             min_child_weight=min_child_weight, max_delta_step=max_delta_step,
#             subsample=subsample, colsample_bytree=colsample_bytree,
#             colsample_bylevel=colsample_bylevel, colsample_bynode=colsample_bynode,
#             reg_alpha=reg_alpha, reg_lambda=reg_lambda, scale_pos_weight=scale_pos_weight,
#             base_score=base_score, random_state=random_state, seed=seed, missing=missing,
#             **kwargs)
#
#     def fit(self, X, y, sample_weight=None, eval_set=None, eval_metric=None,
#             early_stopping_rounds=None, verbose=True, xgb_model=None,
#             sample_weight_eval_set=None, callbacks=None):
#
#         # TODO add support for class and sample weight for multilabel
#         if self.learning_rates is not None:
#             lr_callback = [xgb.callback.reset_learning_rate(self.learning_rates)]
#         else:
#             lr_callback = None
#
#         if callbacks is not None:
#             callbacks = [callback for callback in callbacks if 'reset_learning_rate' not in str(callback)]
#             callbacks = callbacks + lr_callback
#         else:
#             callbacks = lr_callback
#
#         return super(XGBClassifierLR, self).fit(X, y, callbacks = callbacks)

# class LGBClassifierLR(ClassifierMixin):
#     def __init__(self, model_params=None, n_estimators=None, learning_rates=None):
#         self.model_params = model_params
#         self.n_estimators = n_estimators
#         self.learning_rates = learning_rates
#
#     def fit(self, X, y, sample_weight=None):
#         dtrain = lgb.Dataset(X, label=y)
#         model = lgb.train(self.model_params
#                           , dtrain
#                           , num_boost_round=self.n_estimators
#                           , learning_rates=self.learning_rates
#                           )
#         self.model = model
#
#     def predict(self, X):
#         return self.model.predict(X)
#         # TODO Fix
#
#     def predict_proba(self, X):
#         return self.model.predict(X)
#
#     def get_params(self):
#         return self.learning_rates