class LGBClassifierOptuna(object):
    def __init__(self, lr_decay=0.5, seed=42):
        self.random_state = seed
        self.lr_decay = lr_decay
        self.n_estimators = 0
        self.learning_rates = []

    def set_train(self, X_train, y_train):
        self.ratio = sum(y_train == 1) / len(y_train)
        self.dtrain = lgb.Dataset(X_train,
                                  label=y_train,
                                  free_raw_data=False,
                                  )

    def set_test(self, X_test, y_test):
        self.dtest = lgb.Dataset(X_test,
                                 label=y_test,
                                 free_raw_data=False,
                                 )
        self.X_test = X_test

    def choose_and_set_params(self, trial, weights):
        pos_weight = weights[1]
        model_params = {'seed': self.random_state
            , 'metric': 'auc'
            , 'n_jobs': -1
            , 'verbose': 0
            , 'boost_from_average': False
                        # ,'max_bin'            : 255
            , 'objective': trial.suggest_categorical('objective', ['xentropy'])  # 'binary'
            , 'learning_rate': trial.suggest_loguniform('init_learning_rate', 1e-2, 1.0)
            , 'tree_learner': 'feature'
            , 'boosting_type': trial.suggest_categorical('boosting', ['gbdt', 'goss', 'dart'])
            , 'scale_pos_weight': pos_weight  # trial.suggest_uniform('scale_pos_weight', 1.0, 10.0)
            , 'num_leaves': trial.suggest_int('num_leaves', 10, 100)
            , 'min_child_weight': trial.suggest_loguniform('lgb_min_child_weight', 1e-3, 30.0)
            , 'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 30)
            , 'subsample': trial.suggest_uniform('subsample', 0.5, 1.0)
            , 'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.1, 1.0)
                        }

        if model_params['boosting_type'] == 'dart':
            model_params['drop_rate'] = trial.suggest_loguniform('drop_rate', 1e-8, 1.0)
            model_params['skip_drop'] = trial.suggest_loguniform('skip_drop', 1e-8, 1.0)

        if model_params['boosting_type'] == 'goss':
            model_params['top_rate'] = trial.suggest_uniform('top_rate', 0.0, 1.0)
            model_params['other_rate'] = trial.suggest_uniform('other_rate', 0.0, 1.0 - model_params['top_rate'])

        if model_params['boosting_type'] == 'rf':
            model_params['bagging_freq'] = 2

        self.model_params = model_params

    def train_one_iteration(self):
        if self.n_estimators == 0:
            init_model = None
        else:
            init_model = self.best_model
            self.dtrain = self.dtrain.set_init_score(None)
            self.dtest = self.dtest.set_init_score(None)

        self.model = lgb.train(self.model_params
                               , self.dtrain
                               , init_model=init_model
                               , num_boost_round=1
                               , valid_sets=[self.dtest]
                               , verbose_eval=False
                               )

    def score_test(self, scorer):
        preds = self.model.predict(self.X_test)
        return scorer(self.dtest.get_label(), preds)

    def lower_complexity(self):
        self.model_params['learning_rate'] *= self.lr_decay

    def save_best(self):
        self.best_model = self.model
        self.learning_rates.append(self.model_params['learning_rate'])
        self.n_estimators += 1

    def create_model(self):
        final_model = LGBClassifierLR(self.model_params, self.n_estimators, self.learning_rates)
        return final_model

class LGBClassifierLR(ClassifierMixin):
    def __init__(self, model_params=None, n_estimators=None, learning_rates=None):
        self.model_params = model_params
        self.n_estimators = n_estimators
        self.learning_rates = learning_rates

    def fit(self, X, y, sample_weight=None):
        dtrain = lgb.Dataset(X, label=y)
        model = lgb.train(self.model_params
                          , dtrain
                          , num_boost_round=self.n_estimators
                          , learning_rates=self.learning_rates
                          )
        self.model = model

    def predict(self, X):
        return self.model.predict(X)
        # TODO Fix

    def predict_proba(self, X):
        return self.model.predict(X)

    def get_params(self):
        return self.learning_rates
