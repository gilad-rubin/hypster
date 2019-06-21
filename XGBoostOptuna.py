import xgboost as xgb
from xgboost import XGBClassifier
import numpy as np

class XGBClassifierOptuna(object):
    def __init__(self, lr_decay=0.5, booster_list=['gblinear', 'gbtree', 'dart'],
                 seed=None, user_param_dict={}):

        #TODO: add "verbose" and other parameters (user_param_dict)
        #TODO: add support for argument "missing"

        self.random_state = seed
        self.lr_decay = lr_decay
        self.booster_list = booster_list
        self.user_param_dict = user_param_dict
        self.n_estimators = 0
        self.learning_rates = []

    def get_properties(self):
        return {'shortname': 'XGBoost Classifier',
                'name': 'XGBoost Classifier',
                'handles_regression': False,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': False,
                'is_deterministic': False,
                # Both input and output must be tuple(iterable)
                #'input': [DENSE, SIGNED_DATA, UNSIGNED_DATA],
                #'output': [PREDICTIONS]
                }

    def set_train(self, X_train, y_train):
        #TODO: use XGBLabelEncoder
        self.dtrain = xgb.DMatrix(X_train, label=np.array(y_train), )

    def set_test(self, X_test, y_test):
        self.dtest = xgb.DMatrix(X_test, label=np.array(y_test))

    def choose_and_set_params(self, trial, weights):
        pos_weight = weights[1]
        model_params = {'seed': self.random_state
            , 'verbosity': 0
            , 'nthread': -1
            , 'scale_pos_weight': pos_weight
            , 'objective': 'binary:logistic'
            , 'eta': trial.suggest_loguniform('init_eta', 1e-3, 1.0)
            , 'booster': trial.suggest_categorical('booster', self.booster_list)
            , 'lambda': trial.suggest_loguniform('lambda', 1e-10, 1.0)
            , 'alpha': trial.suggest_loguniform('alpha', 1e-10, 1.0)
                        }

        if model_params['booster'] in ['gbtree', 'dart']:
            tree_dict = {'max_depth': trial.suggest_int('max_depth', 2, 20)
                , 'min_child_weight': trial.suggest_int('min_child_weight', 1, 20)
                , 'gamma': trial.suggest_loguniform('gamma', 1e-10, 1.0)
                , 'grow_policy': trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide'])
                , 'subsample': trial.suggest_uniform('subsample', 0.5, 1.0)
                , 'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.1, 1.0)
                , 'colsample_bynode': trial.suggest_uniform('colsample_bynode', 0.1, 1.0)
                , 'num_parallel_tree': trial.suggest_int('num_parallel_tree', 1, 20)
                }

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

        #TODO: add the option to run over default HP
        # if self.user_param_dict is not None:
        #     for (key, value) in self.user_param_dict:
        #         self.model_params[key] = self.user_param_dict[key]

        self.model_params = model_params

    def train_one_iteration(self):
        if self.n_estimators == 0:
            init_model = None
            self.init_eta = self.model_params["eta"]
        else:
            init_model = self.best_model
            init_model.set_param({"eta": self.model_params['eta']})

        self.model = xgb.train(self.model_params
                               , self.dtrain
                               , xgb_model=init_model
                               , num_boost_round=1
                               , learning_rates=[self.model_params['eta']]
                               )

    def score_test(self, scorer):
        preds = self.model.predict(self.dtest)
        return scorer(self.dtest.get_label(), preds)

    def lower_complexity(self):
        self.model_params['eta'] = self.model_params['eta'] * self.lr_decay
        return False

    def save_best(self):
        self.best_model = self.model.copy()
        self.learning_rates.append(self.model_params['eta'])
        self.n_estimators += 1

    def create_model(self):
        self.model_params['n_estimators'] = self.n_estimators
        self.model_params['learning_rate'] = self.init_eta

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

        lr_callback = [xgb.callback.reset_learning_rate(self.learning_rates)]

        if callbacks is not None:
            callbacks = callbacks + lr_callback
        else:
            callbacks = lr_callback

        #TODO: check if there's already a "reset_learning_rate" callback and decide what to do

        return super(XGBClassifierLR, self).fit(X, y, callbacks = callbacks)