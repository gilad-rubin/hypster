import pandas as pd
import numpy as np
import scipy
import sklearn
import optuna

from sklearn.base import clone
from copy import deepcopy
from sklearn.utils import safe_indexing
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

class Objective(object):
    def __init__(self, X, y, cat_columns, estimator, pipeline=None, pipe_params=None,
                 cv='warn', scoring=None,  # sklearn, support multiple?
                 refit=False, tol=1e-5, agg_func=np.mean, max_iter=5000, max_fails=3):

        self.X = X
        self.y = y
        self.cat_columns = cat_columns
        self.pipeline = pipeline
        self.pipe_params = pipe_params
        self.estimator = estimator
        self.cv = cv
        self.scoring = scoring
        self.refit = refit
        #self.agg_func = agg_func #TODO fix func regarded as "tuple"
        self.agg_func = np.mean
        self.tol = tol
        self.max_iter = max_iter
        self.max_fails = max_fails

    def __call__(self, trial):
        if self.pipeline is not None:
            pipeline = clone(self.pipeline)
            pipe_params = self._get_params(trial)
            pipeline.set_params(**pipe_params)

        weights = sklearn.utils.class_weight.compute_class_weight("balanced", np.unique(self.y), self.y)
        n_classes = len(np.unique(self.y))
        estimator_list = []

        # create k folds and estimators
        #TODO use cv like in GridSearchCV
        for train, test in self.cv.split(self.X, self.y):  # groups=self.groups
            X_train, y_train = safe_indexing(self.X, train), safe_indexing(self.y, train)
            X_test, y_test = safe_indexing(self.X, test), safe_indexing(self.y, test)

            #convert categorical columns with OneHotEncoding
            if (self.cat_columns is not None) and (estimator.get_properties['handles_categorical'] == False):
                ohe = ColumnTransformer([('ohe', OneHotEncoder(handle_unknown='ignore'), self.cat_columns)], remainder="passthrough")
                if self.pipeline is not None:
                    pipeline.steps.append(['ohe', ohe])
                else:
                    pipeline = Pipeline([('ohe', ohe)])

            if self.pipeline is not None:
                X_train = pipeline.fit_transform(X_train, y_train)
                X_test = pipeline.transform(X_test)

            self.estimator.choose_and_set_params(trial, weights, n_classes)
            estimator = deepcopy(self.estimator) #TODO: check if "copy" or "clone" is better
            # TODO: if estimator doesn't support categorical featues - automatically
            #  convert categorical to onehot (like lgbm) + add to pipeline
            estimator.set_train_test(X_train, y_train, X_test, y_test, self.cat_columns)

            estimator_list.append(estimator)

        best_score = 0.0
        for step in range(self.max_iter):
            # print("Iteration #", step)
            scores = []
            for estimator in estimator_list:
                estimator.train_one_iteration()
                fold_score = estimator.score_test(self.scoring)
                scores.append(fold_score)

            intermediate_value = self.agg_func(scores)
            trial.report(intermediate_value, step)

            #print("intermediate result = ", intermediate_value)

            if trial.should_prune(step):
                raise optuna.structs.TrialPruned()

            if intermediate_value >= best_score + self.tol:
                best_score = intermediate_value
                fail_count = 0
                for estimator in estimator_list:
                    estimator.save_best()
            else:
                fail_count += 1
                if (fail_count >= self.max_fails) or (estimator_list[0].lower_complexity()==False):
                    break

                # TODO: make this step only after k times
                for estimator in estimator_list:
                    estimator.lower_complexity()

        model = estimator.create_model()
        if self.pipeline is not None:
            pipeline.steps.append(['classifier', model])
        else:
            pipeline = Pipeline([("classifier", model)])

        print('Score: ' + str(round(best_score, 5)))

        trial.set_user_attr('pipeline', pipeline)

        return best_score

    def _get_params(self, trial):
        param_dict = {name: trial._suggest(name, distribution) for name,
                                                                   distribution in self.pipe_params.items()}
        return param_dict


class HyPSTERClassifier():
    def __init__(self, estimators,  # names and/or instances,
                 pipeline=None, pipe_params=None,
                 scoring=None,  # sklearn, support multiple? #TODO: add support for strings like sklearn
                 greater_is_better=None,  # TODO: check how to extract from scorer/metric
                 cv=3,
                 agg_func=np.mean,
                 refit=True,
                 tol=1e-5, max_iter=1000, time_limit=None, max_fails=3,
                 study_name="",  # TODO: think how to allow the user to define if they want to keep training or not?
                 save_cv_probs=False, #TODO: add support for stacking
                 pruner=optuna.pruners.SuccessiveHalvingPruner(min_resource=3, reduction_factor=3),
                 sampler=optuna.samplers.TPESampler(),
                 n_jobs=1,
                 verbose=1, #TODO Add support for verbosity
                 random_state=None):

        self.estimators = estimators
        self.pipeline = pipeline
        self.pipe_params = pipe_params
        self.scoring = scoring
        self.greater_is_better = greater_is_better
        self.cv = cv
        self.agg_func = agg_func,
        self.refit = refit
        self.tol = tol
        self.max_iter = max_iter
        self.time_limit = time_limit
        self.max_fails = max_fails
        self.study_name = study_name #TODO:needed? maybe replace with "resume last study"?
        self.save_cv_probs = save_cv_probs
        self.pruner = pruner
        self.sampler = sampler
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.random_state = random_state

        # TODO: set_seed function for samplers & cv_function(?) with random_state

    def fit(self, X=None, y=None, cat_columns=None, n_trials_per_estimator=10): #dataset_name
        studies = []
        for i in range(len(self.estimators)):
            estimator = self.estimators[i]

            if estimator.get_seed() == 1: #TODO make it nice and consider removing setters and getters
                estimator.set_seed(self.random_state)

            print(estimator.get_properties()["name"]) #TODO: convert to static method?
            #TODO cat_columns = list of indices or names
            objective = Objective(X, y, cat_columns, estimator, self.pipeline, self.pipe_params,
                                  cv=self.cv, scoring=self.scoring, agg_func=self.agg_func,
                                  tol=self.tol, max_iter=self.max_iter, max_fails=self.max_fails)

            if self.greater_is_better:
                direction = "maximize"
            else:
                direction = "minimize"

            if self.verbose > 0:
                optuna.logging.set_verbosity(optuna.logging.WARN)
                # TODO: change

            study = optuna.create_study(pruner=self.pruner, sampler=self.sampler, direction=direction)
            studies.append(study)

            if type(n_trials_per_estimator) == list:
                n_trials = n_trials_per_estimator[i]
            else:
                n_trials = n_trials_per_estimator

            studies[i].optimize(objective, n_trials=n_trials, n_jobs=self.n_jobs)

        self.studies = studies

        if self.greater_is_better:
            func = np.argmax
        else:
            func = np.argmin
        index = func([study.best_value for study in studies])
        self.best_estimator_ = studies[index].best_trial.user_attrs['pipeline'] #TODO: what if refit=true?
        self.best_score_ = studies[index].best_value
        self.best_params_ = studies[index].best_params

        # append results
        # find best result + refit
        # output combined results
        # return studies
        # return cv_results_, best_index_

    def refit(self): #TODO
        return

    def predict(self): #TODO
        #only if refit=true
        return

    def predict_proba(self): #TODO
        # only if refit=true
        return

    def visualize_results(self):
        return
        # TODO: plot... with matplotlib/plotly/hvplot
        # TODO: show if it will help to increase max_iter