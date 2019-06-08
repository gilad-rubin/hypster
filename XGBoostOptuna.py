from sklearn.base import ClassifierMixin
# %%
from sklearn.base import clone
from copy import deepcopy
from sklearn.utils import safe_indexing


class XGBClassifierOptuna(object):
    def __init__(self, lr_decay=0.5, seed=42):
        self.random_state = seed
        self.lr_decay = lr_decay
        self.n_estimators = 0
        self.learning_rates = []

    def set_train(self, X_train, y_train):
        self.dtrain = xgb.DMatrix(X_train, label=np.array(y_train))

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
            , 'booster': trial.suggest_categorical('booster', ['gblinear'])  # , 'gbtree', 'dart'])
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

        self.model_params = model_params

    def train_one_iteration(self):
        if self.n_estimators == 0:
            init_model = None
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

    def save_best(self):
        self.best_model = self.model.copy()
        self.learning_rates.append(self.model_params['eta'])
        self.n_estimators += 1

    def create_model(self):
        final_model = XGBClassifierLR(self.model_params, self.n_estimators, self.learning_rates)
        return final_model

# %%
class XGBClassifierLR(ClassifierMixin):
    def __init__(self, model_params=None, n_estimators=None, learning_rates=None):
        self.learning_rates = learning_rates
        self.model_params = model_params
        self.n_estimators = n_estimators

    def fit(self, X, y, sample_weight=None):
        dtrain = xgb.DMatrix(X, label=y)
        model = xgb.train(self.model_params
                          , dtrain
                          , num_boost_round=self.n_estimators
                          , learning_rates=self.learning_rates
                          )
        self.model = model

    def predict(self, X):
        test_dmatrix = xgb.DMatrix(X)
        return self.model.predict(test_dmatrix, output_margin=False)

    def predict_proba(self, X):
        test_dmatrix = xgb.DMatrix(X)
        return self.model.predict(test_dmatrix, output_margin=True)

    def get_params(self):
        return self.learning_rates