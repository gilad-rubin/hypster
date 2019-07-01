import pandas as pd
import numpy as np
import scipy
import sklearn
import optuna

from sklearn.base import clone
from copy import deepcopy
from sklearn.utils import safe_indexing
from sklearn.pipeline import Pipeline

from joblib import parallel_backend, delayed, Parallel

class Objective(object):
    def __init__(self, X, y, estimator, pipeline=None, pipe_params=None,
                 cv='warn', scoring=None,  # sklearn, support multiple?
                 refit=False, agg_func=np.mean, tol=1e-5, max_iter=5000, max_fails=3):

        self.X = X
        self.y = y
        self.pipeline = pipeline
        self.pipe_params = pipe_params
        self.estimator = estimator
        self.cv = cv
        self.scoring = scoring
        self.refit = refit
        self.agg_func = agg_func
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
        for train, test in self.cv.split(self.X, self.y):  # groups=self.groups
            X_train, y_train = safe_indexing(self.X, train), safe_indexing(self.y, train)
            X_test, y_test = safe_indexing(self.X, test), safe_indexing(self.y, test)
            if self.pipeline is not None:
                X_train = pipeline.fit_transform(X_train, y_train)
                X_test = pipeline.transform(X_test)

            self.estimator.choose_and_set_params(trial, weights, n_classes)
            estimator = deepcopy(self.estimator) #TODO: check if "copy" or "clone" is better
            estimator.set_train(X_train, y_train)
            estimator.set_test(X_test, y_test)

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


class Study():
    def __init__(self, X, y, estimators,  # names and/or instances,
                 pipeline=None, pipe_params=None,
                 cv='warn', scoring=None,  # sklearn, support multiple?
                 greater_is_better=True,  # TODO: check how to extract from scorer/metric
                 refit=False, agg_func=np.mean, tol=1e-5, max_iter=1000, max_fails=5,
                 study_name="",  # TODO: think how to allow the user to define if they want to keep training or not?
                 save_only_best_estimators=False,  # TODO: think about it
                 verbosity=1, pruner=optuna.pruners.SuccessiveHalvingPruner(min_resource=2, reduction_factor=2),
                 sampler=optuna.samplers.TPESampler(), random_state=None):

        self.X = X
        self.y = y
        self.pipeline = pipeline
        self.pipe_params = pipe_params
        self.estimators = estimators
        self.cv = cv
        self.scoring = scoring
        self.greater_is_better = greater_is_better
        self.refit = refit
        self.agg_func = agg_func
        self.tol = tol
        self.max_iter = max_iter
        self.max_fails = max_fails
        # self.study_name = study_name TODO:needed? maybe replace with "resume last study"?
        self.save_only_best_estimators = save_only_best_estimators
        self.verbosity = verbosity
        self.pruner = pruner
        self.sampler = sampler
        self.random_state = random_state

        # TODO: set_seed function for samplers & cv_function(?) with random_state

    def run(self, n_trials_per_estimator=10, n_jobs=1):
        studies = []
        for i in range(len(self.estimators)):
            estimator = self.estimators[i]

            #add "setters" to estimators?
            estimator.n_jobs = n_jobs
            estimator.seed = self.random_state

            print(estimator.get_properties()["name"]) #TODO: convert to static method?

            objective = Objective(self.X, self.y, estimator, self.pipeline, self.pipe_params,
                                  cv=self.cv, scoring=self.scoring, agg_func=self.agg_func,
                                  tol=self.tol, max_iter=self.max_iter, max_fails=self.max_fails)

            if self.greater_is_better:
                direction = "maximize"
            else:
                direction = "minimize"

            if self.verbosity > 0:
                optuna.logging.set_verbosity(optuna.logging.WARN)
                # TODO: change

            study = optuna.create_study(pruner=self.pruner, sampler=self.sampler, direction=direction)
            studies.append(study)

            if type(n_trials_per_estimator) == list:
                n_trials = n_trials_per_estimator[i]
            else:
                n_trials = n_trials_per_estimator

            studies[i].optimize(objective, n_trials=n_trials, n_jobs=n_jobs)

        self.studies = studies
        # append results
        # find best result + refit
        # output combined results
        # return studies

    def visualize_results(self):
        return
        # TODO: plot... with matplotlib/plotly/hvplot
        # TODO: show if it will help to increase max_iter
