import xgboost as xgb
from xgboost import XGBClassifier, XGBRegressor
import numpy as np
from sklearn.base import clone
from copy import deepcopy

class HypsterEstimator():
    def __init__(self, n_iter_per_round=1, n_jobs=1, random_state=1, param_dict={}):
        self.n_iter_per_round = n_iter_per_round
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.param_dict = param_dict
        self.best_model = None
        self.current_model = None
        self.tags = {}

    def get_name(self):
        raise NotImplementedError

    def get_seed(self):
        return self.random_state

    def set_seed(self, seed):
        self.random_state = seed

    def get_n_jobs(self):
        return self.n_jobs

    def set_n_jobs(self, n_jobs):
        self.n_jobs = n_jobs

    def get_tags(self):
        return self.tags

    def set_current_model(self, model):
        self.current_model = model

    def get_current_model(self):
        return self.current_model

    def get_best_model(self):
        return self.best_model

    def set_best_model(self, best_model):
        self.best_model = best_model

    def set_n_iter_per_round(self, n_iter):
        self.n_iter_per_round = n_iter

    def get_n_iter_per_round(self):
        return self.n_iter_per_round

    def choose_and_set_params(self, trial, weights, y_stats):
        raise NotImplementedError()

    def fit(self, X, y, warm_start):
        raise NotImplementedError()

    def predict(self, X):
        raise NotImplementedError()

    def create_model(self):
        raise NotImplementedError()

class XGBModelHypster(HypsterEstimator):
    def __init__(self, lr_decay=0.2, booster_list=['gblinear', 'gbtree', 'dart'],
                 n_iter_per_round=1, random_state=1, n_jobs=1, param_dict={}):
        self.lr_decay = lr_decay
        self.booster_list = booster_list
        self.best_n_iterations = 0
        self.learning_rates = []

        super(XGBModelHypster, self).__init__(n_iter_per_round=n_iter_per_round, n_jobs=n_jobs,
                                              random_state=random_state, param_dict=param_dict)

    def update_tags(self):
        self.tags['sensitive to feature scaling'] = self.model_params["booster"] == "gblinear"
        self.tags['tree based'] = self.model_params["booster"] in ["gbtree", "dart"]

    def set_train(self, X, y, sample_weight=None, missing=None):
        # TODO make sample_weight works with class_weight
        self.dtrain = xgb.DMatrix(X, y, weight=sample_weight, missing=missing, nthread=self.n_jobs)

    def set_test(self, X, y, sample_weight=None, missing=None):
        # TODO make sample_weight works with class_weight
        # TODO make sure sample_weight passes to XGBClassifierLR
        self.dtest = xgb.DMatrix(X, y, weight=sample_weight, missing=missing, nthread=self.n_jobs)

    def fit(self, sample_weight=None, warm_start=False):
        if warm_start==True:
            xgb_model = self.current_model
        else:
            xgb_model = None

        learning_rates = [self.model_params['eta']] * self.n_iter_per_round

        self.current_model = xgb.train(self.model_params
                               , self.dtrain
                               , xgb_model=xgb_model
                               , num_boost_round=self.n_iter_per_round
                               , callbacks=[xgb.callback.reset_learning_rate(learning_rates)]
                               )

    def set_learning_rates(self, learning_rates):
        self.learning_rates = learning_rates

    def get_learning_rates(self):
        return self.learning_rates

    def lower_complexity(self):
        self.model_params['eta'] = self.model_params['eta'] * self.lr_decay

    def save_best(self):
        self.learning_rates += [self.model_params['eta']] * self.n_iter_per_round
        self.best_n_iterations = len(self.learning_rates)
        self.set_best_model(self.get_current_model().copy())

class XGBClassifierHypster(XGBModelHypster):
    @staticmethod
    def get_name():
        return 'XGBoost Classifier'

    def set_default_tags(self):
        self.tags = {'alias' : ['xgb', 'xgboost'],
                    'supports regression': False,
                    'supports ranking': False,
                    'supports classification': True,
                    'supports multiclass': True,
                    'supports multilabel': False,
                    'handles categorical' : False,
                    'handles categorical nan': False,
                    'handles sparse': True,
                    'handles numeric nan': True,
                    'nan value when sparse': 0,
                    'sensitive to feature scaling': False,
                    'has predict_proba' : True,
                    'has model embeddings': True,
                    'adjustable model complexity' : True,
                    'tree based': True
                    }

    def choose_and_set_params(self, trial, class_counts, missing):
        n_classes = len(class_counts)

        #TODO change according to Laurae
        model_params = {'seed': self.random_state
            , 'verbosity': 1
            , 'nthread': self.n_jobs
            , 'missing' : missing
            , 'eta': trial.suggest_loguniform('eta', 1e-3, 1.0)
            , 'booster': trial.suggest_categorical('booster', self.booster_list)
            , 'lambda': trial.suggest_loguniform('lambda', 1e-10, 1.0)
            , 'alpha': trial.suggest_loguniform('alpha', 1e-10, 1.0)
            }

        if n_classes == 2:
            model_params['objective'] = 'binary:logistic'

            pos_weight = class_counts[0] / class_counts[1]
            model_params['scale_pos_weight'] = trial.suggest_categorical("scale_pos_weight", [1.0, pos_weight])

            base_score = class_counts[1] / (class_counts[0] + class_counts[1])  # equivalent to np.mean(y)
            model_params['base_score'] = base_score
        else:  # multiclass
            model_params['objective'] = 'multi:softprob'
            #TODO change base and sample weight on DMatrix
            #change base score to class priors (https://github.com/dmlc/xgboost/issues/1380)
            #change sample weight by multiplying class_weight and sample weight

        if model_params['booster'] in ['gbtree', 'dart']:
            tree_dict = {'max_depth': trial.suggest_int('max_depth', 2, 20) #TODO: maybe change to higher range?
                , 'min_child_weight': trial.suggest_int('min_child_weight', 1, 20)
                , 'gamma': trial.suggest_loguniform('gamma', 1e-10, 5.0)
                , 'grow_policy': trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide'])
                , 'subsample': trial.suggest_uniform('subsample', 0.5, 1.0)
                , 'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.1, 1.0)
                , 'colsample_bynode': trial.suggest_uniform('colsample_bynode', 0.1, 1.0)
                }

            forest_boosting = trial.suggest_categorical('forest_boosting', [True, False])
            if forest_boosting:
                model_params['num_parallel_tree'] = trial.suggest_int('num_parallel_tree', 2, 10)
            else:
                model_params['num_parallel_tree'] = 1

            model_params.update(tree_dict)

        else:  # gblinear
            model_params['feature_selector'] = trial.suggest_categorical('shotgun_feature_selector',
                                                                         ['cyclic', 'shuffle'])

        if model_params['booster'] == 'dart':
            dart_dict = {'sample_type': trial.suggest_categorical('sample_type', ['uniform', 'weighted'])
                , 'normalize_type': trial.suggest_categorical('normalize_type', ['tree', 'forest'])
                , 'rate_drop': trial.suggest_loguniform('rate_drop', 1e-8, 1.0)
                , 'skip_drop': trial.suggest_loguniform('skip_drop', 1e-8, 1.0)
                }

            model_params.update(dart_dict)

        self.model_params = model_params

    def predict_proba(self):
        if self.model_params["booster"] == "dart":
            class_probs = self.current_model.predict(self.dtest, output_margin=False, ntree_limit=0)
        else:
            class_probs = self.current_model.predict(self.dtest, output_margin=False)

        if self.model_params['objective'] == "multi:softprob":
            return class_probs

        classone_probs = class_probs
        classzero_probs = 1.0 - classone_probs
        return np.vstack((classzero_probs, classone_probs)).transpose()

    def create_model(self):
        #TODO: if learning rates are identical throughout - create a regular Classifier

        self.model_params['n_estimators'] = self.best_n_iterations
        self.model_params['learning_rate'] = self.model_params["eta"]

        self.model_params['n_jobs'] = self.model_params.pop('nthread')
        self.model_params['random_state'] = self.model_params.pop('seed')
        self.model_params['reg_lambda'] = self.model_params.pop('lambda')
        self.model_params['reg_alpha'] = self.model_params.pop('alpha')

        final_model = XGBClassifierLR(learning_rates=self.learning_rates, **self.model_params)
        return final_model

class XGBClassifierLR(XGBClassifier):
    def __init__(self, learning_rates = None,
                 max_depth=3, learning_rate=0.1, n_estimators=100,
                 verbosity=1, silent=True,
                 objective="binary:logistic", booster='gbtree',
                 n_jobs=1, nthread=None, gamma=0, min_child_weight=1, max_delta_step=0,
                 subsample=1, colsample_bytree=1, colsample_bylevel=1,
                 colsample_bynode=1, reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
                 base_score=0.5, random_state=0, seed=None, missing=None, **kwargs):

        if 'learning_rates' in kwargs:
            self.learning_rates = kwargs.pop('learning_rates')
        else:
            self.learning_rates = learning_rates

        super(XGBClassifierLR, self).__init__(
            max_depth=max_depth, learning_rate=learning_rate, n_estimators=n_estimators,
            verbosity=verbosity, silent=silent, objective=objective, booster=booster,
            n_jobs=n_jobs, nthread=nthread, gamma=gamma,
            min_child_weight=min_child_weight, max_delta_step=max_delta_step,
            subsample=subsample, colsample_bytree=colsample_bytree,
            colsample_bylevel=colsample_bylevel, colsample_bynode=colsample_bynode,
            reg_alpha=reg_alpha, reg_lambda=reg_lambda, scale_pos_weight=scale_pos_weight,
            base_score=base_score, random_state=random_state, seed=seed, missing=missing,
            **kwargs)

    def fit(self, X, y, sample_weight=None, eval_set=None, eval_metric=None,
            early_stopping_rounds=None, verbose=True, xgb_model=None,
            sample_weight_eval_set=None, callbacks=None):

        # TODO add support for class and sample weight for multilabel
        if self.learning_rates is not None:
            lr_callback = [xgb.callback.reset_learning_rate(self.learning_rates)]
        else:
            lr_callback = None

        if callbacks is not None:
            callbacks = [callback for callback in callbacks if 'reset_learning_rate' not in str(callback)]
            callbacks = callbacks + lr_callback
        else:
            callbacks = lr_callback

        return super(XGBClassifierLR, self).fit(X, y, callbacks = callbacks)

class XGBRegressorHypster(XGBModelHypster):
    @staticmethod
    def get_name():
        return 'XGBoost Regressor'

    def set_default_tags(self):
        self.tags = {'alias' : ['xgb', 'xgboost'],
                    'supports regression': True,
                    'supports ranking': False,
                    'supports classification': False,
                    'supports multiclass': False,
                    'supports multilabel': False,
                    'handles categorical' : False,
                    'handles sparse': True,
                    'handles nan': True,
                    'default nan value when sparse': 0,
                    'sensitive to feature scaling': False,
                    'has predict_proba' : False,
                    'has model embeddings': True,
                    'adjustable model complexity' : True,
                    'tree based': True
                    }

    def choose_and_set_params(self, trial, y_mean, missing):
        model_params = {'seed': self.random_state
                        , 'verbosity': 1
                        , 'nthread': self.n_jobs
                        , 'objective' : 'reg:squarederror'
                        , 'base_score' : y_mean
                        , 'missing' : missing
                        , 'eta': trial.suggest_loguniform('eta', 1e-3, 1.0)
                        , 'booster': trial.suggest_categorical('booster', self.booster_list)
                        , 'lambda': trial.suggest_loguniform('lambda', 1e-10, 1.0)
                        , 'alpha': trial.suggest_loguniform('alpha', 1e-10, 1.0)
                        }

        if model_params['booster'] in ['gbtree', 'dart']:
            tree_dict = {'max_depth': trial.suggest_int('max_depth', 2, 20) #TODO: maybe change to higher range?
                , 'min_child_weight': trial.suggest_int('min_child_weight', 1, 20)
                , 'gamma': trial.suggest_loguniform('gamma', 1e-10, 5.0)
                , 'grow_policy': trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide'])
                , 'subsample': trial.suggest_uniform('subsample', 0.5, 1.0)
                , 'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.1, 1.0)
                , 'colsample_bynode': trial.suggest_uniform('colsample_bynode', 0.1, 1.0)
                }

            forest_boosting = trial.suggest_categorical('forest_boosting', [True, False])
            if forest_boosting:
                model_params['num_parallel_tree'] = trial.suggest_int('num_parallel_tree', 2, 10)
            else:
                model_params['num_parallel_tree'] = 1

            model_params.update(tree_dict)

        else:  # gblinear
            model_params['feature_selector'] = trial.suggest_categorical('shotgun_feature_selector',
                                                                         ['cyclic', 'shuffle'])

        if model_params['booster'] == 'dart':
            dart_dict = {'sample_type': trial.suggest_categorical('sample_type', ['uniform', 'weighted'])
                , 'normalize_type': trial.suggest_categorical('normalize_type', ['tree', 'forest'])
                , 'rate_drop': trial.suggest_loguniform('rate_drop', 1e-8, 1.0)
                , 'skip_drop': trial.suggest_loguniform('skip_drop', 1e-8, 1.0)
                }

            model_params.update(dart_dict)

        self.model_params = model_params

    def predict(self):
        if self.model_params["booster"] == "dart":
            preds = self.current_model.predict(self.dtest, output_margin=False, ntree_limit=0)
        else:
            preds= self.current_model.predict(self.dtest, output_margin=False)

        return preds

    def create_model(self):
        # TODO: if learning rates are identical throughout - create a regular Classifier

        self.model_params['n_estimators'] = self.best_n_iterations
        self.model_params['learning_rate'] = self.model_params["eta"]

        self.model_params['n_jobs'] = self.model_params.pop('nthread')
        self.model_params['random_state'] = self.model_params.pop('seed')
        self.model_params['reg_lambda'] = self.model_params.pop('lambda')
        self.model_params['reg_alpha'] = self.model_params.pop('alpha')

        final_model = XGBRegressorLR(learning_rates=self.learning_rates, **self.model_params)
        return final_model

class XGBRegressorLR(XGBRegressor):
    def __init__(self, learning_rates = None,
                 max_depth=3, learning_rate=1, n_estimators=100,
                 verbosity=1, silent=True,
                 objective="reg:squarederror", booster="gbtree", n_jobs=1, nthread=None, gamma=0,
                 min_child_weight=1, max_delta_step=0, subsample=0.8, colsample_bytree=1,
                 colsample_bylevel=1, colsample_bynode=0.8, reg_alpha=0, reg_lambda=1,
                 scale_pos_weight=1, base_score=0.5, random_state=0, seed=None,
                 missing=None, **kwargs):

        if 'learning_rates' in kwargs:
            self.learning_rates = kwargs.pop('learning_rates')
        else:
            self.learning_rates = learning_rates

        super(XGBRegressorLR, self).__init__(
            max_depth=max_depth, learning_rate=learning_rate, n_estimators=n_estimators,
            verbosity=verbosity, silent=silent, objective=objective, booster=booster,
            n_jobs=n_jobs, nthread=nthread, gamma=gamma,
            min_child_weight=min_child_weight, max_delta_step=max_delta_step,
            subsample=subsample, colsample_bytree=colsample_bytree,
            colsample_bylevel=colsample_bylevel, colsample_bynode=colsample_bynode,
            reg_alpha=reg_alpha, reg_lambda=reg_lambda, scale_pos_weight=scale_pos_weight,
            base_score=base_score, random_state=random_state, seed=seed, missing=missing,
            **kwargs)


    def fit(self, X, y, sample_weight=None, eval_set=None, eval_metric=None,
            early_stopping_rounds=None, verbose=True, xgb_model=None,
            sample_weight_eval_set=None, callbacks=None):

        y = np.array(y)

        if self.learning_rates is not None:
            lr_callback = [xgb.callback.reset_learning_rate(self.learning_rates)]
        else:
            lr_callback = None

        if callbacks is not None:
            callbacks = [callback for callback in callbacks if 'reset_learning_rate' not in str(callback)]
            callbacks = callbacks + lr_callback
        else:
            callbacks = lr_callback

        return super(XGBRegressorLR, self).fit(X, y, callbacks = callbacks)