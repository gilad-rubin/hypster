from sklearn.linear_model import SGDClassifier
import numpy as np
from sklearn.base import clone
from copy import deepcopy
from ..sgd import SGDModelHypster
from sklearn.base import ClassifierMixin

class SGDClassifierHypster(SGDModelHypster):
    @staticmethod
    def get_name():
        return 'SGD Classifier'

    def set_default_tags(self):
        self.tags = {'alias' : ['sgd'],
                    'supports regression': False,
                    'supports ranking': False,
                    'supports classification': True,
                    'supports multiclass': True,
                    'supports multilabel': True,
                    'handles categorical' : False,
                    'handles categorical nan': False,
                    'handles sparse': True,
                    'handles numeric nan': False,
                    'nan value when sparse': 0,
                    'sensitive to feature scaling': True,
                    'has predict_proba' : True,
                    'has model embeddings': True,
                    'adjustable model complexity' : True,
                    'tree based': False
                    }

    def choose_and_set_params(self, trial, class_counts, missing):
        n_classes = len(class_counts)
        n_samples = sum(class_counts)
        lst = n_samples / (n_classes * class_counts)
        scale_dict = {i:lst[i] for i in range(len(lst))}

        losses = ['log', 'modified_huber'] #, 'squared_hinge', 'perceptron', 'hinge'
        #learning_rates = ['constant'] #, 'optimal', 'invscaling'

        model_params = {'random_state': self.random_state
                        ,'n_jobs' : self.n_jobs
                        ,'verbose' : 0
                        ,'loss': trial.suggest_categorical('loss', losses)
                        ,'penalty': trial.suggest_categorical('penalty', ['none', 'l1', 'l2', 'elasticnet'])
                        ,'alpha': trial.suggest_loguniform('alpha', 1e-8, 1.0)
                        ,'learning_rate': 'constant'
                        ,'eta0': trial.suggest_loguniform('eta0', 1e-3, 1.0)
                        ,'class_weight': trial.suggest_categorical("class_weight", [None, scale_dict])
                        ,'shuffle' : trial.suggest_categorical("shuffle", [True, False])
                        }

        if model_params['penalty'] == 'elasticnet':
            model_params['l1_ratio'] = trial.suggest_uniform('l1_ratio', 0, 1.0)

        # if model_params['learning_rate'] == 'invscaling':
        #     model_params['power_t'] = trial.suggest_uniform('eta0', 0.2, 1.0)

        self.model_params = model_params

    def fit(self, sample_weight=None, warm_start=False):
        if (warm_start==False) or (self.current_model is None):
            self.current_model = SGDClassifier(**self.model_params
                                              ,max_iter=self.n_iter_per_round
                                              ,warm_start=True
                                              ,early_stopping=False
                                              ,tol=1e-5
                                             )
        else:
            self.current_model.set_params(eta0=self.model_params['eta0'])

        self.current_model.partial_fit(self.X_train, self.y_train, self.classes, sample_weight)

    def predict_proba(self):
        if self.model_params["loss"] in ['log', 'modified_huber']:
            preds = self.current_model.predict_proba(self.X_test)
        else:
            preds = self.current_model.decision_function(self.X_test)

        return preds

    def create_model(self):
        #TODO: if learning rates are identical throughout - create a regular Classifier
        final_model = SGDClassifierLR(self.model_params, self.learning_rates)
        return final_model


class SGDClassifierLR(SGDClassifier):
    def __init__(self, model_params=None, learning_rates=None):
        self.model_params = model_params
        self.learning_rates = learning_rates

    def fit(self, X, y, sample_weight=None):
        classes = np.unique(y)
        model = SGDClassifier(**self.model_params
                             ,max_iter=1
                             ,warm_start=True
                             ,early_stopping=False
                             ,tol=1e-5
                             )

        from itertools import groupby
        for lr,group in groupby(self.learning_rates):
            model.set_params(eta0=lr)
            model.set_params(max_iter=len(list(group)))
            model.partial_fit(X, y, classes)

        self.model = model

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        try:
            preds = self.model.predict_proba(X)
        except:
            preds = self.model.decision_function(X)

        return preds

    # def get_params(self):
    #     return self.learning_rates