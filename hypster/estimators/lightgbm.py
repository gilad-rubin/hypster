import lightgbm as lgb
import numpy as np
from sklearn.base import clone
from copy import copy, deepcopy
from .base import HypsterEstimator

class LGBModelHypster(HypsterEstimator):
    def __init__(self, lr_decay=0.1, n_iter_per_round=1,
                 random_state=1, n_jobs=1, param_dict={}):
        self.lr_decay = lr_decay
        self.best_n_iterations = 0
        self.learning_rates = []
        self.best_ptrain = None
        self.best_ptest = None

        super(LGBModelHypster, self).__init__(n_iter_per_round=n_iter_per_round, n_jobs=n_jobs,
                                              random_state=random_state, param_dict=param_dict)

    def set_train(self, X, y, sample_weight=None, missing=None):
        #init_score = np.zeros(len(y))
        #self.current_ptrain = init_score
        self.dtrain = lgb.Dataset(data=X, label=y, weight=sample_weight,
                                  #init_score=init_score,
                                  #params={"verbose": -1},
                                  #silent=True,
                                  free_raw_data=False)


    def set_test(self, X, y, sample_weight=None, missing=None):
        #init_score = np.zeros(X.shape[0])
        #self.current_ptest = init_score
        # self.dtest = lgb.Dataset(data=X, label=y, weight=sample_weight,
        #                          init_score=init_score,
        #                          silent=True, free_raw_data=False).construct()
        self.dtest = X

    def fit(self, sample_weight=None, warm_start=True):
        learning_rates = [self.model_params['learning_rate']] * self.n_iter_per_round

        if warm_start:
           model = self.get_current_model()

        if model is None:
            self.current_model = lgb.train(self.model_params,
                                           self.dtrain,
                                           verbose_eval=False,
                                           num_boost_round=self.n_iter_per_round,
                                           keep_training_booster=True,
                                           callbacks=[lgb.reset_parameter(learning_rate=learning_rates)]
                                           )
        for i in range(self.n_iter_per_round):
            self.current_model.update()

        # self.current_model = lgb.train(self.model_params,
        #                                self.dtrain,
        #                                num_boost_round=self.n_iter_per_round,
        #                                callbacks=[lgb.reset_parameter(learning_rate=learning_rates)]
        #                                )
        # self.current_ptrain += self.current_model.predict(self.dtrain.get_data(), raw_score=True,
        #                                                   num_iteration=self.n_iter_per_round)
        #
        # self.current_ptest += self.current_model.predict(self.dtest.get_data(), raw_score=True,
        #                                                  num_iteration=self.n_iter_per_round)
        #
        #
        # self.dtrain.set_init_score(self.current_ptrain)
        # self.dtest.set_init_score(self.current_ptest)

    def lower_complexity(self):
        self.model_params['learning_rate'] = self.model_params['learning_rate'] * self.lr_decay
        self.current_model.set_attr("learning_rate", self.model_params['learning_rate'])
        #self.current_ptrain = copy(self.best_ptrain)
        #self.current_ptest = copy(self.best_ptest)

    def save_best(self):
        self.learning_rates += [self.model_params['learning_rate']] * self.n_iter_per_round
        self.best_n_iterations = len(self.learning_rates)
        model_str = self.current_model.model_to_string(num_iteration=-1)
        self.set_best_model(lgb.Booster(model_str=model_str, silent=True))
        #self.best_ptrain = copy(self.current_ptrain)
        #self.best_ptest = copy(self.current_ptest)