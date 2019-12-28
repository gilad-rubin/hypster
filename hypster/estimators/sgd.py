import numpy as np
from sklearn.base import clone
from copy import deepcopy
from .base import HypsterEstimator

class SGDModelHypster(HypsterEstimator):
    def __init__(self, lr_decay=0.1, n_iter_per_round=1, random_state=1, n_jobs=1, param_dict={}):
        self.lr_decay = lr_decay
        self.best_n_iterations = 0
        self.learning_rates = []

        super(SGDModelHypster, self).__init__(n_iter_per_round=n_iter_per_round, n_jobs=n_jobs,
                                              random_state=random_state, param_dict=param_dict)

    def set_train(self, X, y, sample_weight=None, missing=None):
        self.X_train = X
        self.y_train = y
        self.classes = np.unique(y)

    def set_test(self, X, y, sample_weight=None, missing=None):
        self.X_test = X
        self.y_test = y

    def lower_complexity(self):
        self.model_params['eta0'] = self.model_params['eta0'] * self.lr_decay

    def save_best(self):
        self.learning_rates += [self.model_params['eta0']] * self.n_iter_per_round
        self.best_n_iterations = len(self.learning_rates)
        self.set_best_model(deepcopy(self.get_current_model()))