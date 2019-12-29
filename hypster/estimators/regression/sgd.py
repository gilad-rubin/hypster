from sklearn.linear_model import SGDRegressor
from ..sgd import SGDModelHypster

class SGDRegressorHypster(SGDModelHypster):
    def get_tags(self):
        self.tags = {'name' : "SGD Regressor",
                     'model type': "linear",
                     'supports regression': True,
                     'supports ranking': False,
                     'supports classification': False,
                     'supports multiclass': False,
                     'supports multilabel': False,
                     'handles categorical' : False,
                     'handles categorical nan': False,
                     'handles sparse': True,
                     'handles numeric nan': False,
                     'nan value when sparse': 0,
                     'sensitive to feature scaling': True,
                     'has predict_proba' : False,
                     'has model embeddings': False,
                     'adjustable model complexity' : True
                    }
        return self.tags

    def choose_and_set_params(self, trial, class_counts, missing):
        self.trial = trial #TODO move to core
        losses = ['squared_loss', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive']
        #learning_rates = ['constant'] #, 'optimal', 'invscaling'

        model_params = {'random_state': self.random_state
                        #,'n_jobs' : self.n_jobs #TODO: add n_jobs for later versions of SGDRegressor
                        ,'verbose' : 0
                        ,'loss': self.sample_hp('loss', "categorical", losses)
                        ,'penalty': self.sample_hp('penalty', "categorical", ['none', 'l1', 'l2', 'elasticnet'])
                        ,'alpha': self.sample_hp('alpha', "log-uniform", [1e-8, 1.0])
                        ,'learning_rate': 'constant'
                        ,'eta0': self.sample_hp('eta0', "log-uniform", [1e-4, 0.1])
                        ,'shuffle' : self.sample_hp("shuffle", "categorical", [True, False])
                        }

        if model_params['penalty'] == 'elasticnet':
            model_params['l1_ratio'] = self.sample_hp('l1_ratio', "uniform", [0, 1.0])

        if model_params["loss"] in ["huber", "epsilon_insensitive", "squared_epsilon_insensitive"]:
            model_params["epsilon"] = self.sample_hp("epsilon", "log-uniform", [1e-3,1.0]) #TODO check range

        # if model_params['learning_rate'] == 'invscaling':
        #     model_params['power_t'] = self.sample_hp('eta0', "uniform", [0.2, 1.0])

        self.model_params = model_params

    def fit(self, sample_weight=None, warm_start=True):
        if (warm_start==False) or (self.current_model is None):
            self.current_model = SGDRegressor(**self.model_params
                                              ,max_iter=self.n_iter_per_round
                                              ,warm_start=True
                                              ,early_stopping=False
                                              ,tol=1e-10
                                             )
        else:
            self.current_model.set_params(eta0=self.model_params['eta0'])

        self.current_model.partial_fit(self.X_train, self.y_train, sample_weight)

    def predict(self):
        preds = self.current_model.predict(self.X_test)
        return preds

    def create_model(self):
        #TODO: if learning rates are identical throughout - create a regular Classifier
        final_model = SGDRegressor(**self.model_params)
        #final_model = SGDRegressorLR(self.model_params, self.learning_rates)
        return final_model


# class SGDClassifierLR(SGDClassifier):
#     def __init__(self, model_params=None, learning_rates=None):
#         self.model_params = model_params
#         self.learning_rates = learning_rates
#
#     def fit(self, X, y, sample_weight=None):
#         classes = np.unique(y)
#         model = SGDClassifier(**self.model_params
#                              ,max_iter=1
#                              ,warm_start=True
#                              ,early_stopping=False
#                              ,tol=1e-5
#                              )
#
#         from itertools import groupby
#         for lr,group in groupby(self.learning_rates):
#             model.set_params(eta0=lr)
#             model.set_params(max_iter=len(list(group)))
#             model.partial_fit(X, y, classes)
#
#         self.model = model
#
#     def predict(self, X):
#         return self.model.predict(X)
#
#     def predict_proba(self, X):
#         try:
#             preds = self.model.predict_proba(X)
#         except:
#             preds = self.model.decision_function(X)
#
#         return preds