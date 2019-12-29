import lightgbm as lgb
from .base import HypsterEstimator

class LGBModelHypster(HypsterEstimator):
    def __init__(self, lr_decay=0.1, n_iter_per_round=1,
                 random_state=1, n_jobs=1, param_dict={}):
        self.lr_decay = lr_decay
        self.best_n_iterations = 0
        self.learning_rates = []

        super(LGBModelHypster, self).__init__(n_iter_per_round=n_iter_per_round, n_jobs=n_jobs,
                                              random_state=random_state, param_dict=param_dict)

    def set_train(self, X, y, sample_weight=None, missing=None):
        self.dtrain = lgb.Dataset(data=X, label=y, weight=sample_weight,
                                  params={"verbose": -1},
                                  free_raw_data=False)


    def set_test(self, X, y, sample_weight=None, missing=None):
        self.dtest = X

    def fit(self, sample_weight=None, warm_start=True):
        learning_rates = [self.model_params['learning_rate']] * self.n_iter_per_round

        if warm_start:
           model = self.get_current_model()

        if model is None:
            self.current_model = lgb.train(self.model_params,
                                           self.dtrain,
                                           num_boost_round=self.n_iter_per_round,
                                           keep_training_booster=True,
                                           callbacks=[lgb.reset_parameter(learning_rate=learning_rates)]
                                           )
        else:
            self.current_model = lgb.train(self.model_params,
                                           self.dtrain,
                                           init_model=model,
                                           num_boost_round=self.n_iter_per_round,
                                           callbacks=[lgb.reset_parameter(learning_rate=learning_rates)]
                                           )

    def lower_complexity(self):
        self.model_params['learning_rate'] = self.model_params['learning_rate'] * self.lr_decay
        self.current_model.set_attr("learning_rate", self.model_params['learning_rate'])

    def save_best(self):
        self.learning_rates += [self.model_params['learning_rate']] * self.n_iter_per_round
        self.best_n_iterations = len(self.learning_rates)
        model_str = self.current_model.model_to_string(num_iteration=-1)
        self.set_best_model(lgb.Booster(model_str=model_str, silent=True))