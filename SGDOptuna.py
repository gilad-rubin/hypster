class SGDClassifierOptuna(object):
    def __init__(self, lr_decay=0.5, seed=42):
        self.random_state = seed
        self.lr_decay = lr_decay
        self.n_estimators = 0
        self.learning_rates = []

    def set_train(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.classes = np.unique(self.y_train)

    def set_test(self, X_test, y_test):
        self.X_test = X_test
        self.y_test = y_test

    def choose_and_set_params(self, trial, weights):
        scale_dict = {0: weights[0], 1: weights[1]}
        losses = ['log', 'modified_huber', 'squared_hinge', 'perceptron', 'hinge']
        learning_rates = ['constant', 'optimal', 'invscaling']  # 'adaptive'

        model_params = {'random_state': self.random_state
            , 'loss': trial.suggest_categorical('loss', losses)
            , 'penalty': trial.suggest_categorical('penalty', ['none', 'l1', 'l2', 'elasticnet'])
            , 'alpha': trial.suggest_loguniform('alpha', 1e-8, 1.0)
            , 'learning_rate': 'constant'
                        # ,'learning_rate'    : trial.suggest_categorical('sgd_learning_rate', learning_rates)
            , 'eta0': trial.suggest_loguniform('eta0', 1e-2, 1.0)
            , 'class_weight': scale_dict
                        }

        if model_params['penalty'] == 'elasticnet':
            model_params['l1_ratio'] = trial.suggest_uniform('l1_ratio', 0, 1.0)

        if model_params['learning_rate'] == 'invscaling':
            model_params['power_t'] = trial.suggest_uniform('eta0', 0.2, 1.0)

        self.model_params = model_params

    def train_one_iteration(self):
        if self.n_estimators == 0:
            self.model = sklearn.linear_model.SGDClassifier(**self.model_params
                                                            , max_iter=1
                                                            , warm_start=True
                                                            , verbose=0
                                                            , early_stopping=False
                                                            , n_jobs=-1
                                                            , tol=1e-5
                                                            )
        else:
            self.model = deepcopy(self.best_model)
            self.model.set_params(eta0=self.model_params['eta0'])

        self.model.partial_fit(self.X_train, self.y_train, self.classes)

    def score_test(self, scorer):
        if self.model_params["loss"] in ['log', 'modified_huber']:
            preds = self.model.predict_proba(self.X_test)[:, 1]
        else:
            preds = self.model.decision_function(self.X_test)

        return scorer(self.y_test, preds)

    def lower_complexity(self):
        self.model_params['eta0'] *= self.lr_decay

    def save_best(self):
        self.best_model = deepcopy(self.model)
        self.learning_rates.append(self.model_params['eta0'])
        self.n_estimators += 1

    def create_model(self):
        final_model = SGDClassifierLR(self.model_params, self.n_estimators, self.learning_rates)
        return final_model

class SGDClassifierLR(ClassifierMixin):
    def __init__(self, model_params=None, n_estimators=None, learning_rates=None):
        self.model_params = model_params
        self.n_estimators = n_estimators
        self.learning_rates = learning_rates

    def fit(self, X, y, sample_weight=None):
        classes = np.unique(y)
        model = sklearn.linear_model.SGDClassifier(**self.model_params
                                                   , max_iter=1
                                                   , warm_start=True
                                                   , verbose=0
                                                   , early_stopping=False
                                                   , n_jobs=-1
                                                   , tol=1e-5
                                                   )
        for i in range(self.n_estimators):
            model.set_params(eta0=self.learning_rates[i])
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

    def get_params(self):
        return self.learning_rates