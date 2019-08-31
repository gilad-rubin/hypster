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