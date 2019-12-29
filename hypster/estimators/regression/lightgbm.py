from lightgbm.sklearn import LGBMRegressor
from ..lightgbm import LGBModelHypster


class LGBRegressorHypster(LGBModelHypster):
    def get_tags(self):
        self.tags = {'name': "LightGBM Regressor",
                     'model type': "tree",
                     'supports regression': True,
                     'supports ranking': False,
                     'supports classification': False,
                     'supports multiclass': False,
                     'supports multilabel': False,
                     'handles categorical': False,  #TODO: change to True
                     'handles categorical nan': False,  # TODO: test
                     'handles sparse': True,
                     'handles numeric nan': True,
                     'nan value when sparse': 0,  # TODO: test
                     'sensitive to feature scaling': False,
                     'has predict_proba': False,
                     'has model embeddings': False,
                     'adjustable model complexity': False  # TODO: change to True
                     }
        return self.tags

    def choose_and_set_params(self, trial, y_mean, missing):
        self.trial = trial
        # TODO change according to Laurae
        model_params = {'n_jobs': -1,
                        'verbose': -1,
                        'random_state': self.random_state,
                        'objective' : "regression", #TODO: check "regression_l1"
                        # 'use_missing' 'zero_as_missing'
                        'learning_rate': self.sample_hp('learning_rate', "log-uniform", [1e-3, 1.0]),
                        'boosting_type': self.sample_hp('boosting', "categorical", ['gbdt', 'goss', 'dart']),  # 'rf'
                        'max_depth': self.sample_hp('max_depth', "int", [2, 20]),
                        # TODO: maybe change to higher range? (for competition or production mode or bigger datasets)
                        'min_child_samples': self.sample_hp('min_child_samples', "int", [1, 30]),
                        'min_child_weight': self.sample_hp('lgb_min_child_weight', "log-uniform", [1e-3, 30.0]),
                        'colsample_bytree': self.sample_hp('colsample_bytree', "uniform", [0.1, 1.0]),
                        # TODO check if it works on sklearn classifier
                        # 'feature_fraction_bynode': self.sample_hp('feature_fraction_bynode', "uniform", [0.1, 1.0]),
                        # max_delta_step?
                        # min_split_gain
                        'reg_alpha': self.sample_hp('reg_alpha', "log-uniform", [1e-10, 1.0]),
                        'reg_lambda': self.sample_hp('reg_lambda', "log-uniform", [1e-10, 1.0])
                        }

        if model_params["max_depth"] > 12:
            model_params['num_leaves'] = 100
        else:
            model_params['num_leaves'] = 40  # TODO: change
            # max_leaves = np.power(2, model_params["max_depth"])
            # model_params['num_leaves'] = int(self.sample_hp('num_leaves', "log-uniform",
            #                                        [max_leaves/2, max_leaves]))  # TODO: change?

        # TODO: check out pos_bagging and neg_bagging
        if model_params["boosting_type"] != "goss":
            model_params['subsample_freq'] = 1
            model_params['subsample'] = self.sample_hp("subsample", "uniform", [0.4, 1.0])

        # if model_params["objective"] in ["binary", "multiclassova", "cross_entropy"]:
        model_params['boost_from_average'] = True

        if model_params['boosting_type'] == 'dart':
            model_params['xgboost_dart_mode'] = self.sample_hp('xgboost_dart_mode', "categorical", [False, True])
            model_params['drop_rate'] = self.sample_hp('drop_rate', "log_uniform", [1e-8, 1.0])
            model_params['skip_drop'] = self.sample_hp('skip_drop', "log_uniform", [1e-8, 1.0])
        if model_params['boosting_type'] == 'goss':
            model_params['top_rate'] = self.sample_hp('top_rate', "uniform", [0.0, 1.0])
            model_params['other_rate'] = self.sample_hp('other_rate', "uniform", [0.0, 1.0 - model_params['top_rate']])

        # TODO: add categorical hps. starting at "min_data_per_group"
        # https://lightgbm.readthedocs.io/en/latest/Parameters.html

        self.model_params = model_params

    def predict(self):
        preds = self.current_model.predict(self.dtest)
        return preds

    def create_model(self):
        # TODO: if learning rates are identical throughout - create a regular Classifier
        self.model_params['n_estimators'] = self.best_n_iterations
        self.model_params["learning_rate"] = self.learning_rates[0]  # TODO change

        final_model = LGBMRegressor(**self.model_params)
        return final_model

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