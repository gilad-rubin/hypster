import xgboost as xgb
from copy import deepcopy
from .base import HypsterEstimator

#TODO separate into tree and linear
class XGBModelHypster(HypsterEstimator):
    def __init__(self, lr_decay=0.1, n_iter_per_round=1, random_state=1, n_jobs=1, param_dict={}):
        self.lr_decay = lr_decay
        self.best_n_iterations = 0
        self.learning_rates = []

        super(XGBModelHypster, self).__init__(n_iter_per_round=n_iter_per_round, n_jobs=n_jobs,
                                              random_state=random_state, param_dict=param_dict)

    def set_train(self, X, y, sample_weight=None, missing=None):
        # TODO make sample_weight works with class_weight
        self.dtrain = xgb.DMatrix(X, y, weight=sample_weight, missing=missing, nthread=self.n_jobs)

    def set_test(self, X, y, sample_weight=None, missing=None):
        # TODO make sample_weight works with class_weight
        # TODO make sure sample_weight passes to XGBClassifierLR
        self.dtest = xgb.DMatrix(X, y, weight=sample_weight, missing=missing, nthread=self.n_jobs)

    def fit(self, sample_weight=None, warm_start=True):
        learning_rates = [self.model_params['eta']] * self.n_iter_per_round

        if warm_start:
            model = self.get_current_model()
        if model is None:
            self.current_model = xgb.train(self.model_params
                                           ,self.dtrain
                                           ,num_boost_round=self.n_iter_per_round
                                           ,callbacks=[xgb.callback.reset_learning_rate(learning_rates)]
                                           )
        else:
            self.current_model = xgb.train(self.model_params
                                           ,self.dtrain
                                           ,xgb_model=model
                                           ,num_boost_round=self.n_iter_per_round
                                           ,callbacks=[xgb.callback.reset_learning_rate(learning_rates)]
                                           )

    def lower_complexity(self):
        self.model_params['eta'] = self.model_params['eta'] * self.lr_decay

    def save_best(self):
        self.learning_rates += [self.model_params['eta']] * self.n_iter_per_round
        self.best_n_iterations = len(self.learning_rates)
        self.set_best_model(deepcopy(self.get_current_model()))