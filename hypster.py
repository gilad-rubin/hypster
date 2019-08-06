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
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError

from sklearn.base import BaseEstimator, TransformerMixin

class IdentityTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, input_array, y=None):
        return self

    def transform(self, X, y=None):
        return X * 1

def get_best_study_index(self, studies, greater_is_better=True):
    if greater_is_better:
        func = np.argmax
    else:
        func = np.argmin
    print()
    print(func)
    lst = [study.best_value for study in studies]
    print(lst)
    res = func(lst)
    print(res)
    return res

class Objective(object):
    def __init__(self, X, y, cat_columns, objective_type=None, class_counts=None,
                 estimator=None, pipeline=None, pipe_params=None,
                 cv='warn', scoring=None, scorer_type=None,
                 refit=False, tol=1e-5, agg_func=np.mean, max_iter=5000, max_fails=3):

        self.X = X
        self.y = y
        self.cat_columns = cat_columns
        self.objective_type = objective_type
        self.class_counts = class_counts
        self.pipeline = pipeline
        self.pipe_params = pipe_params
        self.estimator = estimator
        self.cv = cv
        self.scoring = scoring
        self.scorer_type = scorer_type
        self.refit_ = refit
        #self.agg_func = agg_func #TODO fix func regarded as "tuple"
        self.agg_func = np.mean
        self.tol = tol
        self.max_iter = max_iter
        self.max_fails = max_fails

    def __call__(self, trial):
        pipeline = None
        if self.pipeline is not None:
            pipeline = clone(self.pipeline)
            if self.pipe_params is not None:
                pipe_params = _get_params(trial, self.pipe_params)
                pipeline.set_params(**pipe_params)

        # convert categorical columns with OneHotEncoding
        #TODO return sparse or dense? consider that in sparse sometimes na is 0
        #TODO what if model handles imputation?
        if (self.cat_columns is not None) and (self.estimator.get_properties()['handles_categorical'] == False):
            impute_ohe_pipe = Pipeline([('impute', SimpleImputer(strategy="constant", fill_value="unknown")),
                                         ('ohe', OneHotEncoder(categories="auto", handle_unknown='ignore'))])
            impute_ohe_ct = ColumnTransformer([('impute_ohe', impute_ohe_pipe, self.cat_columns)], remainder="passthrough")
            if pipeline is not None:
                #TODO where to append? at the beginning or end?
                pipeline.steps.append(['impute_ohe', impute_ohe_ct])
            else:
                pipeline = Pipeline([('impute_ohe', impute_ohe_ct)])

        estimator_list = []

        # create k folds and estimators
        #TODO use cv like in GridSearchCV
        for train, test in self.cv.split(self.X, self.y):  # groups=self.groups
            X_train, y_train = safe_indexing(self.X, train), safe_indexing(self.y, train)
            X_test, y_test = safe_indexing(self.X, test), safe_indexing(self.y, test)

            if pipeline is not None:
                X_train = pipeline.fit_transform(X_train, y_train)
                X_test = pipeline.transform(X_test)

            estimator = deepcopy(self.estimator) #TODO: check if "copy" or "clone" is better

            if estimator.param_dict is not None:
                user_params = _get_params(trial, estimator.param_dict)

            if self.objective_type=="classification":
                estimator.choose_and_set_params(trial, self.class_counts)
            else:
                estimator.choose_and_set_params(trial)

            # avoid hp's that don't work with conditions
            # TODO fix it in cases where the user wants to add a hp
            # TODO get this out of the loop
            if estimator.param_dict is not None:
                for (key, value) in user_params.items():  # override default params
                    if key in estimator.model_params.keys():
                        estimator.model_params[key] = value

            estimator.set_train_test(X_train, y_train, X_test, y_test, self.cat_columns)

            estimator_list.append(estimator)

        best_score = 0.0
        for step in range(self.max_iter):
            # print("Iteration #", step)
            scores = []
            for estimator in estimator_list:
                estimator.train_one_iteration()
                fold_score = estimator.score_test(self.scoring, self.scorer_type)
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
        if pipeline is not None:
            pipeline.steps.append(["model", model])
        else:
            pipeline = Pipeline([("model", model)])

        print('Score: ' + str(round(best_score, 5)))

        trial.set_user_attr('pipeline', pipeline)

        return best_score

def _get_params(trial, params):
    param_dict = {}
    for (key, value) in params.items():
        if "optuna.distributions" in str(type(value)):
            param_dict[key] = trial._suggest(key, value)
        else:
            param_dict[key] = params[key]
    return param_dict

class HyPSTEREstimator():
    def __init__(self, estimators,  # names and/or instances,
                 pipeline=None, pipe_params=None,
                 scoring=None,  # sklearn, support multiple? #TODO: add support for strings like sklearn
                 cv=3,
                 agg_func=np.mean,
                 refit=True,
                 tol=1e-5, max_iter=1000, time_limit=None, max_fails=3,
                 study_name="",  # TODO: think how to allow the user to define if they want to keep training or not?
                 save_cv_preds=False, #TODO: add support for stacking
                 pruner=optuna.pruners.SuccessiveHalvingPruner(min_resource=3, reduction_factor=3),
                 sampler=optuna.samplers.TPESampler(),
                 n_jobs=1,
                 verbose=1, #TODO Add support for verbosity
                 random_state=None):

        self.estimators = estimators
        self.pipeline = pipeline
        self.pipe_params = pipe_params
        self.scoring = scoring
        self.cv = cv
        self.agg_func = agg_func,
        self.refit_ = refit
        self.tol = tol
        self.max_iter = max_iter
        self.time_limit = time_limit
        self.max_fails = max_fails
        self.study_name = study_name #TODO:needed? maybe replace with "resume last study"?
        self.save_cv_probs = save_cv_preds
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
                                  cv=self.cv, scorer=scorer, agg_func=self.agg_func,
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

        #find best study
        if self.greater_is_better:
            func = np.argmax
        else:
            func = np.argmin
        index = func([study.best_value for study in studies])
        #index = get_best_study_index(studies, self.greater_is_better) #TODO convert to function

        self.best_estimator_ = studies[index].best_trial.user_attrs['pipeline'] #TODO: what if refit=true?
        self.best_score_ = studies[index].best_value
        self.best_params_ = studies[index].best_params

        if len(self.best_estimator_.steps) > 1: #pipeline has more than just a classifier
            self.best_transformer_ = Pipeline(self.best_estimator_.steps[:-1]) #return all steps but last (classifier)
        else:
            self.best_transformer_ = IdentityTransformer() #TODO check if it's neccessary

        self.studies = studies

        if self.refit_:
            self.best_estimator_.fit(X, y)

        # append results
        # output combined results
        # return studies
        # return cv_results_, best_index_

    def refit(self, X, y):
        self.refit_=True
        self.best_estimator_.fit(X, y)

    def _check_is_fitted(self, method_name):
        if not self.refit_:
            raise NotFittedError('This instance was initialized '
                                 'with refit=False. %s is '
                                 'available only after refitting on the best '
                                 'parameters. You can refit an estimator '
                                 'by using applying the ".refit" method manually'
                                 % (method_name))
        else:
            check_is_fitted(self, 'best_estimator_')

    def predict(self, X): #TODO
        self._check_is_fitted('predict')
        return self.best_estimator_.predict(X)

    def predict_proba(self, X):
        self._check_is_fitted('predict_proba')
        return self.best_estimator_.predict_proba(X)

    def visualize_results(self):
        return
        # TODO: plot... with matplotlib/plotly/hvplot
        # TODO: show if it will help to increase max_iter

class HyPSTERClassifier(HyPSTEREstimator):
    def fit(self, X=None, y=None, cat_columns=None, n_trials_per_estimator=10): #dataset_name
        #TODO check that y is classification and not regression
        le = LabelEncoder()
        y = le.fit_transform(y)
        class_counts = np.bincount(y)

        studies = []
        for i in range(len(self.estimators)):
            estimator = self.estimators[i]

            if estimator.get_seed() == 1: #TODO make it nice and consider removing setters and getters
                estimator.set_seed(self.random_state)

            print(estimator.get_properties()["name"]) #TODO: convert to static method?
            #TODO cat_columns = list of indices or names

            scorer = sklearn.metrics.get_scorer(self.scoring)
            #TODO check if we can make it a bit nicer
            if "_Threshold" in str(type(scorer)):
                scorer_type = "threshold"
            elif "_Predict" in str(type(scorer)):
                scorer_type = "predict"
            else:
                scorer_type = "proba"

            # TODO check if we can make it a bit nicer
            if "greater_is_better=False" in str(scorer):
                greater_is_better=False
            else:
                greater_is_better = True

            objective = Objective(X, y, cat_columns, objective_type="classification",
                                  class_counts=class_counts, estimator=estimator,
                                  pipeline=self.pipeline, pipe_params=self.pipe_params,
                                  cv=self.cv, scoring=scorer._score_func, scorer_type = scorer_type,
                                  agg_func=self.agg_func, tol=self.tol, max_iter=self.max_iter, max_fails=self.max_fails)

            if self.verbose > 0:
                optuna.logging.set_verbosity(optuna.logging.WARN)
                # TODO: change

            if greater_is_better==True:
                direction = "maximize"
            else:
                direction = "minimize"

            study = optuna.create_study(pruner=self.pruner, sampler=self.sampler, direction=direction)
            studies.append(study)

            if type(n_trials_per_estimator) == list:
                n_trials = n_trials_per_estimator[i]
            else:
                n_trials = n_trials_per_estimator

            studies[i].optimize(objective, n_trials=n_trials, n_jobs=self.n_jobs)

        #find best study
        if greater_is_better:
            func = np.argmax
        else:
            func = np.argmin
        index = func([study.best_value for study in studies])
        #index = get_best_study_index(studies, greater_is_better) #TODO convert to function

        self.best_estimator_ = studies[index].best_trial.user_attrs['pipeline'] #TODO: what if refit=true?
        self.best_score_ = studies[index].best_value
        self.best_params_ = studies[index].best_params

        if len(self.best_estimator_.steps) > 1: #pipeline has more than just a classifier
            self.best_transformer_ = Pipeline(self.best_estimator_.steps[:-1]) #return all steps but last (classifier)
        else:
            self.best_transformer_ = IdentityTransformer() #TODO check if it's neccessary

        self.studies = studies

        if self.refit_:
            self.best_estimator_.fit(X, y)

        # append results
        # output combined results
        # return studies
        # return cv_results_, best_index_

    def predict_proba(self, X):
        self._check_is_fitted('predict_proba')
        return self.best_estimator_.predict_proba(X)