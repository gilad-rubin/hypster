#TODO: stick with eith "" or ''
#TODO: clean imports
import pandas as pd
import numpy as np
import scipy
import sklearn
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import SuccessiveHalvingPruner

from sklearn.base import clone
from copy import deepcopy
from sklearn.utils import safe_indexing
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import check_cv
from sklearn.utils.validation import indexable
from sklearn.base import BaseEstimator, TransformerMixin

from scipy.sparse import issparse

def _init_pipeline(pipeline, pipe_params, trial):
    if pipeline is not None:
        final_pipeline = clone(pipeline)
        if pipe_params is not None:
            pipe_params = _get_params(trial, pipe_params)
            final_pipeline.set_params(**pipe_params)
        return final_pipeline
    return None

def _get_params(trial, params):
    param_dict = {}
    for (key, value) in params.items():
        if "optuna.distributions" in str(type(value)):
            param_dict[key] = trial._suggest(key, value)
        else:
            param_dict[key] = params[key]
    return param_dict

class Objective(object):
    def __init__(self, X, y, estimator, sample_weight=None, groups=None, cat_columns=None, objective_type="classification",
                 y_stats=None, pipeline=None, pipe_params=None, greater_is_better=True,
                 cv='warn', save_cv_preds=False, scoring=None, scorer_type=None,
                 refit=False, tol=1e-5, agg_func=np.mean, max_iter=30, max_fails=3):

        self.X = X
        self.y = y
        self.estimator = estimator
        self.sample_weight = sample_weight
        self.groups = groups
        self.cat_columns = cat_columns
        self.objective_type = objective_type
        self.y_stats = y_stats
        self.pipeline = pipeline
        self.pipe_params = pipe_params
        self.greater_is_better = greater_is_better
        self.cv = cv
        self.save_cv_preds = save_cv_preds
        self.scoring = scoring
        self.scorer_type = scorer_type
        self.refit_ = refit
        self.agg_func = agg_func
        self.tol = tol
        self.max_iter = max_iter
        self.max_fails = max_fails

    def __call__(self, trial):
        pipeline = _init_pipeline(self.pipeline, self.pipe_params, trial)

        ## choose estimator from list
        estimator_list = []
        estimator = deepcopy(self.estimator)
        estimator.choose_and_set_params(trial, self.y_stats)

        ## set params on main estimator
        if estimator.param_dict is not None:
            user_params = _get_params(trial, estimator.param_dict)

        # overwrite params with user dictionary
        # to avoid hp's that don't comply to the structure of other sampled HPs
        if estimator.param_dict is not None:
            for (key, value) in user_params.items():
                if key in estimator.model_params.keys():
                    estimator.model_params[key] = value
                #TODO: 'else:' warn user that the key is not compatible with the structure of the other sampled HPs

        estimator.update_tags()

        ## impute & convert categorical columns
        # TODO cat_columns = list of indices or names?
        # TODO add imputation for numeric columns
        if (self.cat_columns is not None) and (estimator.get_tags()['handles categorical'] == False):
            impute_ohe_pipe = Pipeline([('impute', SimpleImputer(strategy="constant", fill_value="unknown")),
                                         ('ohe', OneHotEncoder(categories="auto", handle_unknown='ignore'))])
            impute_ohe_ct = ColumnTransformer([('impute_ohe', impute_ohe_pipe, self.cat_columns)],
                                              remainder="passthrough")
            if pipeline is not None:
                #TODO where to append? at the beginning or end?
                pipeline.steps.append(['impute_ohe', impute_ohe_ct])
            else:
                pipeline = Pipeline([('impute_ohe', impute_ohe_ct)])

        can_lower_complexity = estimator.get_tags()["adjustable model complexity"]

        ## create k folds and estimators
        folds = []
        for train_idx, test_idx in self.cv.split(self.X, self.y, groups=self.groups):
            X_train, y_train = safe_indexing(self.X, train_idx), safe_indexing(self.y, train_idx)
            X_test, y_test = safe_indexing(self.X, test_idx), safe_indexing(self.y, test_idx)

            ## apply pipeline to train+test
            if pipeline is not None:
                X_train = pipeline.fit_transform(X_train, y_train)
                X_test = pipeline.transform(X_test)

            fold_estimator = deepcopy(estimator)

            if self.sample_weight is not None:
                train_sample_weight = safe_indexing(self.sample_weight, train_idx)
                test_sample_weight = safe_indexing(self.sample_weight, test_idx)
            else:
                train_sample_weight = None
                test_sample_weight = None

            fold_estimator.set_train(X_train, y_train, train_sample_weight)
            fold_estimator.set_test(X_test, y_test, test_sample_weight)

            folds.append({"y_test" : y_test,
                          "train_idx" : train_idx, "test_idx" : test_idx,
                          "estimator": fold_estimator})

        if self.greater_is_better:
            best_score = -np.inf
        else:
            best_score = np.inf

        for step in range(self.max_iter):
            # print("Iteration #", step)
            scores = []
            raw_preds_list = []
            for fold in folds:
                ## train for n_iter while resuming from current model
                fold['estimator'].fit(warm_start=True)

                ## get raw predictions
                if self.objective_type == "regression":
                    raw_preds = fold['estimator'].predict()
                else:
                    raw_preds = fold['estimator'].predict_proba()
                    #TODO: what about decision_function? and those who don't have predict_proba?

                if self.save_cv_preds:
                    raw_preds_list.append(raw_preds)

                ## get classes for metrics that deal with classes
                if self.scorer_type == "predict" and self.objective_type == "classification":
                    #TODO handle multiclass
                    threshold = 0.5 #TODO: find optimal threshold w.r.t scoring function
                    raw_preds = (raw_preds >= threshold).astype(int)

                if self.scorer_type == "threshold":
                    raw_preds = raw_preds[:,1] #TODO handle multiclass and other scorers

                ##get score & append
                fold_score = self.scoring(fold["y_test"], raw_preds)
                scores.append(fold_score)

            intermediate_value = self.agg_func(scores)
            # Using "max" in order to avoid pruning just because of overfitting at one certain step:
            report_value = max(intermediate_value, best_score)
            trial.report(report_value, step)

            #print("intermediate result = ", intermediate_value)

            if trial.should_prune(step):
                raise optuna.structs.TrialPruned()

            ## if result improved
            if intermediate_value - best_score >= self.tol: #TODO: should I make it self.tol * estimator.n_iter_per_round?
                best_score = intermediate_value
                fail_count = 0
                for (i, fold) in enumerate(folds):
                    fold['estimator'].save_best()
                    if self.save_cv_preds:
                        #TODO handle cases where:
                        # self.cv does not cover the whole dataset (e.g train/test)
                        # self.cv is repeated cross validation. then we should perhaps choose one cover of the whole dataset
                        fold["raw_predictions"] = raw_preds_list[i]

            else:
                fail_count += 1
                if (can_lower_complexity == False) or (fail_count >= self.max_fails):
                    break

                for fold in folds:
                    fold['estimator'].lower_complexity()
                    best_model = deepcopy(fold['estimator'].get_best_model)
                    fold['estimator'].set_current_model(best_model)

        model = folds[0]['estimator'].get_best_model()

        if pipeline is not None:
            pipeline.steps.append(["model", model])
        else:
            pipeline = Pipeline([("model", model)])

        print('Score: ' + str(round(best_score, 5))) #TODO: change to logging

        trial.set_user_attr('pipeline', pipeline)

        if self.save_cv_preds:
            n_rows = self.X.shape[0]
            n_columns = folds[0]["raw_predictions"].shape[1]
            raw_preds = np.zeros((n_rows, n_columns))
            for fold in folds:
                raw_preds[fold["test_idx"],:] = fold['raw_predictions']
            trial.set_user_attr("cv_preds", raw_preds)

        return best_score

class HyPSTEREstimator():
    def __init__(self, estimators,
                 pipeline=None,
                 pipe_params=None,
                 scoring=None,
                 cv=3,
                 agg_func=np.mean,
                 refit=True,
                 tol=1e-5,
                 max_iter=50,
                 time_limit=None, max_fails=3,
                 study_name="",
                 save_cv_preds=False,
                 pruner=SuccessiveHalvingPruner(min_resource=3, reduction_factor=3),
                 sampler=TPESampler(**TPESampler.hyperopt_parameters()),
                 n_jobs=1,
                 verbose=1,
                 random_state=None):

        self.estimators = estimators
        self.pipeline = pipeline
        self.pipe_params = pipe_params
        self.scoring = scoring
        self.cv = cv
        self.agg_func = agg_func
        self.refit_ = refit
        self.tol = tol
        self.max_iter = max_iter
        self.time_limit = time_limit
        self.max_fails = max_fails
        self.study_name = study_name
        self.save_cv_preds = save_cv_preds
        self.pruner = pruner
        self.sampler = sampler
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.random_state = random_state
        self.best_estimator_ = None

    def fit(self, X=None, y=None, sample_weight=None, groups=None, cat_columns=None, n_trials=10):
        raise NotImplementedError

    def refit(self, X, y):
        #TODO check if best_estimator exists
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

    def predict(self, X):
        self._check_is_fitted('predict')
        return self.best_estimator_.predict(X)
    
    #TODO: check if we should implement "score" and "predict_log_proba"

    def visualize_results(self):
        return
        # TODO: plot... with matplotlib/plotly/hvplot
        # TODO: show if it will help to increase max_iter

    def summary(self):
        #TODO estimators tested, estimators left out
        #TODO more statistics about estimators
        return

class HyPSTERClassifier(HyPSTEREstimator):
    def fit(self, X, y, sample_weight=None, groups=None, cat_columns=None,
            n_trials_per_estimator=10):

        if isinstance(n_trials_per_estimator, list) and (len(n_trials_per_estimator) < len(self.estimators)):
            print("n_trials_per_estimator size is smaller than the number of estimators!") #TODO error
            return

        ##initialize seeds
        if self.sampler.seed is None:
            self.sampler.seed = self.random_state

        X, y, groups = indexable(X, y, groups)
        cv = check_cv(self.cv, y, classifier=True)
        if cv.random_state is None:
            cv.random_state = self.random_state

        scorer = sklearn.metrics.get_scorer(self.scoring)
        # https://github.com/scikit-learn/scikit-learn/blob/1495f6924/sklearn/metrics/scorer.py
        if "_Threshold" in str(type(scorer)):
            scorer_type = "threshold"
        elif "_Predict" in str(type(scorer)):
            scorer_type = "predict"
        else:
            scorer_type = "proba"

        # TODO check if we can make it a bit nicer
        if "greater_is_better=False" in str(scorer):
            greater_is_better = False
            direction = "minimize"
        else:
            greater_is_better = True
            direction = "maximize"

        ## convert labels to np.array
        le = LabelEncoder()
        y = le.fit_transform(y)
        class_counts = np.bincount(y)

        valid_estimators = []
        ## filter out estimators & transformer
        for i in range(len(self.estimators)):
            remove_estimator=False
            estimator = self.estimators[i]
            estimator.set_default_tags()
            tags = estimator.get_tags()
            if issparse(X) and (tags["handles sparse"] == False):
                remove_estimator=True
                #TODO add logging
            elif (tags["supports multiclass"] == False) and (class_counts.shape[1] > 2):
                remove_estimator = True
                # TODO add logging
            if remove_estimator:
                if isinstance(n_trials_per_estimator, list):
                    del n_trials_per_estimator[i]
            else:
                valid_estimators.append(estimator)

        if len(valid_estimators)==0:
            print("No valid estimators available for this type of input") #TODO convert to error
            return

        study = None
        for i in range(len(valid_estimators)):
            estimator = valid_estimators[i]

            print(estimator.get_name())

            if estimator.get_seed() == 1: #TODO make it nice and consider removing setters and getters
                estimator.set_seed(self.random_state)

            estimator.set_n_jobs(self.n_jobs)

            objective = Objective(X, y, estimator, sample_weight, groups, cat_columns,
                                  objective_type="classification", y_stats=class_counts,
                                  pipeline=self.pipeline, pipe_params=self.pipe_params,
                                  greater_is_better=greater_is_better,
                                  cv=self.cv, save_cv_preds=self.save_cv_preds,
                                  scoring=scorer._score_func,
                                  scorer_type = scorer_type,
                                  agg_func=self.agg_func, tol=self.tol,
                                  max_iter=self.max_iter, max_fails=self.max_fails)

            if self.verbose > 0:
                optuna.logging.set_verbosity(optuna.logging.WARN)

            if study is None:
                study = optuna.create_study(pruner=self.pruner,
                                            sampler=self.sampler,
                                            study_name="study",
                                            load_if_exists=True,
                                            direction=direction)
            else:
                #currently there's a bug in optuna that doesn't allow changes in distribution options
                #this is a temporary fix
                study.storage.param_distribution = {}

            if isinstance(n_trials_per_estimator, list):
                n_trials = n_trials_per_estimator[i]
            else:
                n_trials = n_trials_per_estimator

            study.optimize(objective, n_trials=n_trials, n_jobs=self.n_jobs)

        self.study = study
        self.best_estimator_ = study.best_trial.user_attrs['pipeline']
        self.best_score_ = study.best_value
        self.best_params_ = study.best_params
        self.best_index_ = study.best_trial.trial_id

        if self.refit_:
            self.best_estimator_.fit(X, y)

        if len(self.best_estimator_.steps) > 1: #pipeline has more than just a classifier
            self.best_transformer_ = Pipeline(self.best_estimator_.steps[:-1]) #return all steps but last (classifier)
        else:
            self.best_transformer_ = IdentityTransformer() #TODO check if it's neccessary

        self.best_model_ = self.best_estimator_.named_steps["model"]
    
    def predict_proba(self, X):
        self._check_is_fitted('predict_proba')
        return self.best_estimator_.predict_proba(X)

class HyPSTERRegressor(HyPSTEREstimator):
    def fit(self, X=None, y=None, cat_columns=None, n_trials_per_estimator=10): #dataset_name
        #TODO check that y is regression and not classification
        #TODO: consider log-transform y?
        
        #TODO: get compatible transformers and estimators
            #dataset_properties = get_dataset_properties(X, y, cat_columns, type="regression")
            #transformers = get_compatible_estimators(dataset_properties)
            #estimators = get_compatible_estimators(dataset_properties)
        y = np.array(y) #TODO check if this makes sense
        studies = []
        for i in range(len(self.estimators)):
            estimator = self.estimators[i]

            if estimator.get_seed() == 1: #TODO make it nice and consider removing setters and getters
                estimator.set_seed(self.random_state)

            print(estimator.get_tags()["name"]) #TODO: convert to static method?
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
            #TODO check if greater_is_better = False also works (with spaces " = ")
            if "greater_is_better=False" in str(scorer):
                greater_is_better=False
            else:
                greater_is_better = True

            objective = Objective(X, y, cat_columns, objective_type="regression",
                                  estimator=estimator, pipeline=self.pipeline, pipe_params=self.pipe_params,
                                  greater_is_better=greater_is_better, cv=self.cv, scoring=scorer._score_func,
                                  scorer_type = scorer_type, agg_func=self.agg_func, tol=self.tol,
                                  max_iter=self.max_iter, max_fails=self.max_fails)

            if self.verbose > 0:
                optuna.logging.set_verbosity(optuna.logging.WARN)
                # TODO: change

            if greater_is_better:
                direction = "maximize"
            else:
                direction = "minimize"

            study = optuna.create_study(pruner=self.pruner,
                                        sampler=self.sampler, direction=direction)
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
        index = get_best_study_index(studies, greater_is_better) #TODO convert to function

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

class IdentityTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, input_array, y=None):
        return self

    def transform(self, X, y=None):
        return X