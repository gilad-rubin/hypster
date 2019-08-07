import xgboost as xgb
from xgboost import XGBClassifier, XGBRegressor
import numpy as np

class HypsterEstimator:
    def __init__(self, random_state=1, n_jobs=1, param_dict={}):
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.param_dict = param_dict

    def get_seed(self):
        return self.random_state

    def set_seed(self, seed):
        self.random_state = seed

    def get_properties(self):
        raise NotImplementedError()

    def set_train_test(self, X_train, y_train, X_test, y_test, cat_columns=None):
        raise NotImplementedError()

    def choose_and_set_params(self, trial, weights, n_classes):
        raise NotImplementedError()

    def train_one_iteration(self):
        raise NotImplementedError()

    def predict(self):
        raise NotImplementedError()

    def lower_complexity(self):
        raise NotImplementedError()

    def save_best_model(self):
        raise NotImplementedError()

    def create_model(self):
        raise NotImplementedError()

class XGBModelHypster(HypsterEstimator):
    def __init__(self, lr_decay=0.5, booster_list=['gblinear', 'gbtree', 'dart'],
                 random_state=1, n_jobs=1, param_dict=None):

        #TODO: add "verbose" and other parameters (user_param_dict)

        self.random_state = random_state
        self.n_jobs = n_jobs
        self.lr_decay = lr_decay
        self.booster_list = booster_list
        self.param_dict = param_dict
        self.n_estimators = 0
        self.learning_rates = []

    def set_train_test(self, X_train, y_train, X_test, y_test, cat_columns=None): #TODO: move cat_columns to choose_set_params?
        self.dtrain = xgb.DMatrix(X_train, label=y_train)
        self.dtest = xgb.DMatrix(X_test, label=y_test)

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

    def lower_complexity(self):
        self.model_params['eta'] = self.model_params['eta'] * self.lr_decay
        return True

    def save_best(self):
        self.best_model = self.model.copy() #TODO check if needed to use copy or clone?
        self.learning_rates.append(self.model_params['eta'])
        self.n_estimators += 1

class XGBClassifierHypster(XGBModelHypster):
    def get_properties(self):
        #TODO: follow sklearn's new API
        return {'alias' : ['xgb', 'xgboost'],
                'name': 'XGBoost Classifier',
                'handles_regression': False,
                'handles_classification': True,
                'handles_categorical' : False,
                'handles_multiclass': True,
                'handles_multilabel': False,
                'is_deterministic': False,
                'can_lower_complexity': True,
                # Both input and output must be tuple(iterable)
                #'input': [DENSE, SIGNED_DATA, UNSIGNED_DATA],
                #'output': [PREDICTIONS]
                }

    def choose_and_set_params(self, trial, class_counts):
        #included on autosklearn
        # learning_rate, n_estimators, subsample, booster, max_depth,
        # colsample_bylevel, colsample_bytree, reg_alpha, reg_lambda,

        #not included:
        # gamma, min_child_weight, max_delta_step,
        # base_score, scale_pos_weight, n_jobs = 1, init = None,
        # random_state = None, verbose = 0,
        n_classes = len(class_counts)
        
        base_score = 0.5 #TODO fix?
        if n_classes==2:
            objective = 'binary:logistic'
            pos_weight = class_counts[0] / class_counts[1]
            base_score = class_counts[1] / (class_counts[0] + class_counts[1]) #equivalent to np.mean(y)
        else: #multiclass
            objective = 'multi:softprob'
            #pos_weight = #TODO
            base_score = 0.5 #TODO change to class priors (https://github.com/dmlc/xgboost/issues/1380)

        model_params = {'seed': self.random_state
            , 'verbosity': 1
            , 'nthread': self.n_jobs
            , 'scale_pos_weight': trial.suggest_categorical('scale_pos_weight', [1.0, pos_weight])
            , 'objective': objective
            , 'base_score' : base_score
            , 'eta': trial.suggest_loguniform('init_eta', 1e-4, 1.0)
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
                model_params['num_parallel_tree'] = trial.suggest_int('num_parallel_tree', 2, 20)
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

    ##TODO: change to predict, predict proba
    def score_test(self, scorer, scorer_type):
        if self.model_params["booster"] == "dart":
            preds = self.model.predict(self.dtest, output_margin=False, ntree_limit=self.n_estimators + 1)
        else:
            preds = self.model.predict(self.dtest, output_margin=False)

        if (scorer_type == "predict"): #TODO find best threshold w.r.t scoring
            preds = preds > 0.5
            preds = preds.astype(int)

        return scorer(self.dtest.get_label(), preds)

    def create_model(self):
        self.model_params['n_estimators'] = self.n_estimators
        self.model_params['learning_rate'] = self.init_eta

        self.model_params['n_jobs'] = self.model_params.pop('nthread')
        self.model_params['random_state'] = self.model_params.pop('seed')
        self.model_params['reg_lambda'] = self.model_params.pop('lambda')
        self.model_params['reg_alpha'] = self.model_params.pop('alpha')

        final_model = XGBClassifierLR(learning_rates=self.learning_rates, **self.model_params)

        return final_model

class XGBRegressorHypster(XGBModelHypster):
    def get_properties(self):
        return {'alias' : ['xgb', 'xgboost', 'xgbregressor'],
                'name': 'XGBoost Regressor',
                'handles_regression': True,
                'handles_classification': False,
                'handles_categorical' : False,
                'handles_multiclass': False,
                'handles_multilabel': False,
                'is_deterministic': False,
                'can_lower_complexity' : True
                }

    def choose_and_set_params(self, trial):
        #base_score = np.mean(self.dtrain.get_label()) #TODO get it before train/test

        model_params = {'seed': self.random_state
            , 'verbosity': 1
            , 'nthread': self.n_jobs
            , 'objective': 'reg:squarederror'
            #, 'base_score' : base_score
            , 'eta': trial.suggest_loguniform('init_eta', 1e-4, 1.0)
            , 'booster': trial.suggest_categorical('booster', self.booster_list)
            , 'lambda': trial.suggest_loguniform('lambda', 1e-10, 1.0)
            , 'alpha': trial.suggest_loguniform('alpha', 1e-10, 1.0)
            }

        if model_params['booster'] in ['gbtree', 'dart']:
            tree_dict = {'max_depth': trial.suggest_int('max_depth', 2, 20) #TODO: maybe change to higher range?
                , 'min_child_weight': trial.suggest_int('min_child_weight', 1, 20)
                , 'gamma': trial.suggest_loguniform('gamma', 1e-10, 10.0)
                , 'grow_policy': trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide'])
                , 'subsample': trial.suggest_uniform('subsample', 0.5, 1.0)
                , 'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.1, 1.0)
                , 'colsample_bynode': trial.suggest_uniform('colsample_bynode', 0.1, 1.0)
                }

            forest_boosting = trial.suggest_categorical('forest_boosting', [True, False])
            if forest_boosting:
                model_params['num_parallel_tree'] = trial.suggest_int('num_parallel_tree', 2, 20)
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
    
    def score_test(self, scorer, scorer_type):
        if self.model_params["booster"] == "dart":
            preds = self.model.predict(self.dtest, output_margin=False, ntree_limit=self.n_estimators + 1)
        else:
            preds = self.model.predict(self.dtest, output_margin=False)

        return scorer(self.dtest.get_label(), preds)

    def create_model(self):
        self.model_params['n_estimators'] = self.n_estimators
        self.model_params['learning_rate'] = self.init_eta

        self.model_params['n_jobs'] = self.model_params.pop('nthread')
        self.model_params['random_state'] = self.model_params.pop('seed')
        self.model_params['reg_lambda'] = self.model_params.pop('lambda')
        self.model_params['reg_alpha'] = self.model_params.pop('alpha')

        final_model = XGBRegressorLR(learning_rates=self.learning_rates, **self.model_params)

        return final_model

class XGBClassifierLR(XGBClassifier):
    #TODO add get_params with learning_rates
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

        if self.learning_rates is not None: #TODO check that it works when initializing without learning rates
            lr_callback = [xgb.callback.reset_learning_rate(self.learning_rates)]
        else:
            lr_callback = None

        if callbacks is not None:
            callbacks = callbacks + lr_callback
        else:
            callbacks = lr_callback

        #TODO: check if there's already a "reset_learning_rate" callback and decide what to do

        return super(XGBClassifierLR, self).fit(X, y, callbacks = callbacks)
    #TODO implement get_params,set_params, score & decision_function?
    #from sklearn.utils.estimator_checks import check_estimator
    #check_estimator(LinearSVC)

class XGBRegressorLR(XGBRegressor):
    #TODO add get_params with learning_rates
    def __init__(self, learning_rates = None,
                 max_depth=3, learning_rate=1, n_estimators=100,
                 verbosity=1, silent=None,
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

        y = np.array(y) #TODO fix
        if self.learning_rates is not None:
            lr_callback = [xgb.callback.reset_learning_rate(self.learning_rates)]
        else:
            lr_callback = None

        if callbacks is not None:
            callbacks = callbacks + lr_callback
        else:
            callbacks = lr_callback

        #TODO: check if there's already a "reset_learning_rate" callback and decide what to do

        return super(XGBRegressorLR, self).fit(X, y, callbacks = callbacks)