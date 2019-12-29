#TODO: finish
class RFClassifierOptuna(object):
    def __init__(self, n_trees_per_iter=20, seed=42):
        self.random_state = seed
        self.n_trees_per_iter = n_trees_per_iter
        self.n_estimators = 0
        self.best_n_estimators = 0

    def set_train(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def set_test(self, X_test, y_test):
        self.X_test = X_test
        self.y_test = y_test

    def choose_and_set_params(self, trial, weights):
        scale_dict = {0: weights[0], 1: weights[1]}
        model_params = {'class_weight': scale_dict
            , 'random_state': self.random_state
            , 'criterion': trial.suggest_categorical('tree_criterion', ['gini', 'entropy'])
            , 'bootstrap': trial.suggest_categorical('bootstrap', [False, True])
            , 'max_depth': trial.suggest_int('max_depth', 2, 100)
            , 'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20)
            , 'max_features': 'auto'
            ,  # 'max_features'      : trial.suggest_uniform('max_features', 0.1, 1.0)
                        }

        self.model_params = model_params

    def train_one_iteration(self):
        if self.n_estimators == 0:
            self.model = RandomForestClassifier(**self.model_params
                                                , n_estimators=self.n_trees_per_iter
                                                , n_jobs=1
                                                , warm_start=True)

        self.model.fit(self.X_train, self.y_train)
        self.model.n_estimators += self.n_trees_per_iter
        self.n_estimators += self.n_trees_per_iter

    def score_test(self, scorer):
        preds = self.model.predict_proba(self.X_test)[:, 1]
        return scorer(self.y_test, preds)

    def save_best(self):
        self.best_n_estimators += self.n_trees_per_iter

    def create_model(self):
        self.model_params['n_estimators'] = self.best_n_estimators
        self.model_params['n_jobs'] = -1
        self.model_params['class_weight'] = "balanced"
        final_model = RandomForestClassifier(**self.model_params)
        return final_model