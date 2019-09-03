#TODO: stick with eith "" or ''
#TODO: clean imports
import pandas as pd
import numpy as np
import scipy
import scipy.sparse as sp
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
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from category_encoders import OneHotEncoder, BinaryEncoder, CatBoostEncoder, TargetEncoder, WOEEncoder
from sklearn.pipeline import FeatureUnion
from scipy.sparse import issparse

from hypster.utils import *
from hypster.preprocessors import *

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
    def __init__(self, X, y, estimators, sample_weight=None,
                 missing=None, groups=None, cat_cols=None,
                 numeric_cols=None,#TODO add missing
                 objective_type="classification", y_stats=None, pipeline=None,
                 pipe_params=None, greater_is_better=True,
                 cv='warn', save_cv_preds=False, scoring=None, scorer_type=None,
                 refit=False, tol=1e-5, agg_func=np.mean, max_iter=30, max_fails=3, random_state=1):

        self.X = X
        self.y = y
        self.estimators = estimators
        self.sample_weight = sample_weight
        self.groups = groups
        self.missing = missing
        self.cat_cols = cat_cols
        self.numeric_cols = numeric_cols
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
        self.random_state=random_state

    def __call__(self, trial):
        estimator = deepcopy(trial.suggest_categorical("estimator", self.estimators))

        #print(estimator.get_name())

        pipeline = _init_pipeline(self.pipeline, self.pipe_params, trial)

        ## choose estimator from list
        estimator_list = []
        #estimator = deepcopy(self.estimator)
        estimator.choose_and_set_params(trial, self.y_stats, self.missing)

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

        # TODO replace with self.""?
        X = self.X
        y = self.y
        cat_cols = self.cat_cols
        numeric_cols = self.numeric_cols
        random_state = self.random_state
        tags = estimator.get_tags()

        cat_transforms = ["encode"] #"impute" #TODO: fix numpy array output from imputation
        transformers = []
        cat_steps = None

        if cat_cols is not None:
            if "impute" in cat_transforms:
                cat_imputer = CatImputer(X, cat_cols, tags, trial, random_state)
                if cat_imputer is not None:
                    transformers.append(("cat_imputer", cat_imputer))
            if "encode" in cat_transforms:
                n_classes = 1 if self.objective_type=="regression" else len(self.y_stats)
                cat_encoder = CatEncoder(X, cat_cols, tags, self.objective_type, trial, n_classes, random_state)
                if cat_encoder is not None:
                    transformers.append(("cat_encoder", cat_encoder))
            if len(transformers) == 1:
                cat_steps_name = transformers[0][0]
                cat_steps = transformers[0][1]
            elif len(transformers) >= 2:
                cat_steps_name = "cat_transforms"
                cat_steps = Pipeline(transformers)
            if (numeric_cols is not None) and (cat_steps is not None):
                cat_steps = ColumnTransformer([(cat_steps_name, cat_steps, cat_cols)],
                                              remainder="drop", sparse_threshold=0)
        numeric_transforms = ["impute", "scale"]
        transformers = []
        numeric_steps = None
        if numeric_cols is not None:
            if "impute" in numeric_transforms:
                imputer = NumericImputer(X, numeric_cols, trial, tags)
                if imputer is not None:
                    transformers.append(("numeric_imputer", imputer))
            if "scale" in numeric_transforms:
                scaler = Scaler(X, numeric_cols, trial, tags)
                if scaler is not None:
                    transformers.append(("scaler", scaler))
            if len(transformers) == 1:
                numeric_steps_name = transformers[0][0]
                numeric_steps = transformers[0][1]
            elif len(transformers) >= 2:
                numeric_steps_name = "numeric_transforms"
                numeric_steps = Pipeline(transformers)
            if (cat_cols is not None) and (numeric_steps is not None):
                numeric_steps = ColumnTransformer([(numeric_steps_name, numeric_steps, numeric_cols)],
                                                  remainder="drop", sparse_threshold=0)

        if (cat_steps is not None) and (numeric_steps is not None):
            union = FeatureUnion([("cat", cat_steps), ("numeric", numeric_steps)])
            pipeline = add_to_pipe(pipeline, "cat_numeric_transforms", union)
        elif cat_steps is not None:
            pipeline = add_to_pipe(pipeline, cat_steps_name, cat_steps)
        elif numeric_steps is not None:
            pipeline = add_to_pipe(pipeline, numeric_steps_name, numeric_steps)

        can_lower_complexity = tags["adjustable model complexity"]
        ## create k folds and estimators
        folds = []
        for train_idx, test_idx in self.cv.split(X, y, groups=self.groups):
            X_train, y_train = safe_indexing(X, train_idx), safe_indexing(y, train_idx)
            X_test, y_test = safe_indexing(X, test_idx), safe_indexing(y, test_idx)

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

            fold_estimator.set_train(X_train, y_train, sample_weight=train_sample_weight, missing=self.missing)
            fold_estimator.set_test(X_test, y_test, sample_weight=test_sample_weight, missing=self.missing)

            folds.append({"y_test" : y_test,
                          "train_idx" : train_idx, "test_idx" : test_idx,
                          "estimator": fold_estimator})

        best_score = np.nan
        for step in range(self.max_iter):
            #print("Iteration #", step)
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
            # Using "func" in order to avoid pruning just because of overfitting at one certain step:
            if self.greater_is_better:
                func = np.nanmax
            else:
                func = np.nanmin

            report_value = func([intermediate_value, best_score])
            trial.report(report_value, step)

            #print("intermediate result = ", intermediate_value)

            if trial.should_prune():
                raise optuna.structs.TrialPruned()

            if self.greater_is_better:
                # TODO: should I make it self.tol * estimator.n_iter_per_round?
                condition = (np.isnan(best_score)) or (intermediate_value - best_score >= self.tol)
            else:
                condition = (np.isnan(best_score)) or (best_score - intermediate_value >= self.tol)

            if condition:
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
                    best_model = deepcopy(fold['estimator'].get_best_model())
                    fold['estimator'].set_current_model(best_model)

        model = folds[0]['estimator'].create_model()

        if pipeline is not None:
            pipeline.steps.append(["model", model])
        else:
            pipeline = Pipeline([("model", model)])

        #print(pipeline)
        print('Score: ' + str(round(best_score, 5)))

        trial.set_user_attr('pipeline', pipeline)

        if self.save_cv_preds:
            n_rows = X.shape[0]

            if self.objective_type=="regression":
                n_columns = 1
            else:
                n_columns = folds[0]["raw_predictions"].shape[1]

            raw_preds = np.zeros((n_rows, n_columns))
            for fold in folds:
                if n_columns == 1:
                    fold_raw_preds = fold['raw_predictions'].reshape(-1, 1)
                else:
                    fold_raw_preds = fold['raw_predictions']
                raw_preds[fold["test_idx"],:] = fold_raw_preds
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
                 max_fails=3,
                 time_limit=None,
                 study_name=None,
                 save_cv_preds=False,
                 pruner=SuccessiveHalvingPruner(min_resource=3, reduction_factor=3),
                 sampler=TPESampler(**TPESampler.hyperopt_parameters()),
                 storage=None,
                 n_jobs=1,
                 verbose=1,
                 random_state=None):

        self.estimators = estimators if isinstance(estimators, list) else [estimators]
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
        self.storage = storage
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.random_state = random_state
        self.best_pipeline_ = None

    def fit(self, X=None, y=None, sample_weight=None, groups=None, missing=None, cat_cols=None, n_trials=10):
        raise NotImplementedError

    def run_study_save_results(self, X, y, valid_estimators, cv, scorer, scorer_type,
                              greater_is_better, y_stats, objective_type, sample_weight,
                              groups, missing, cat_cols, timeout_per_estimator, n_trials):

        direction = "maximize" if greater_is_better else "minimize"
        #study = None
        for i in range(len(valid_estimators)):
            estimator = valid_estimators[i]

            # print(estimator.get_name())
            #
            if estimator.get_seed() == 1:
                estimator.set_seed(self.random_state)

            estimator.set_n_jobs(self.n_jobs)

        numeric_cols = get_numeric_cols(X, cat_cols)
        objective = Objective(X, y, valid_estimators, sample_weight, groups, missing, cat_cols,
                              numeric_cols=numeric_cols, objective_type=objective_type, y_stats=y_stats,
                              pipeline=self.pipeline, pipe_params=self.pipe_params,
                              greater_is_better=greater_is_better,
                              cv=cv, save_cv_preds=self.save_cv_preds,
                              scoring=scorer._score_func,
                              scorer_type = scorer_type,
                              agg_func=self.agg_func, tol=self.tol,
                              max_iter=self.max_iter, max_fails=self.max_fails,
                              random_state=self.random_state)

        if self.verbose > 0:
            optuna.logging.set_verbosity(optuna.logging.WARN)

        #if study is None:
        study_name = self.study_name if self.study_name else "study"
        study = optuna.create_study(storage=self.storage,
                                    pruner=self.pruner,
                                    sampler=self.sampler,
                                    study_name=study_name,
                                    load_if_exists=True,
                                    direction=direction)

        study.optimize(objective, n_trials=n_trials, n_jobs=self.n_jobs, timeout=timeout_per_estimator)

        self.study = study
        self.best_pipeline_ = study.best_trial.user_attrs['pipeline']
        self.best_score_ = study.best_value
        self.best_params_ = study.best_params
        self.best_index_ = study.best_trial.trial_id

        if self.refit_:
            self.best_pipeline_.fit(X, y)

        if len(self.best_pipeline_.steps) > 1: #pipeline has more than just a classifier
            self.best_transformer_ = Pipeline(self.best_pipeline_.steps[:-1]) #return all steps but last (classifier)
        else:
            self.best_transformer_ = IdentityTransformer()

        self.best_model_ = self.best_pipeline_.named_steps["model"]

    def _check_is_fitted(self, method_name):
        if not self.refit_:
            raise NotFittedError('This instance was initialized '
                                 'with refit=False. %s is '
                                 'available only after refitting on the best '
                                 'parameters. You can refit an estimator '
                                 'by using applying the ".refit" method manually'
                                 % (method_name))
        else:
            check_is_fitted(self, 'best_pipeline_')

    def predict(self, X):
        self._check_is_fitted('predict')
        return self.best_pipeline_.predict(X)
    
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
    def fit(self, X, y, sample_weight=None, groups=None, missing=None, cat_cols=None,
            n_trials=10, timeout_per_estimator=None):

        X, y, groups = indexable(X, y, groups)

        ## convert labels to np.array
        le = LabelEncoder()
        y = le.fit_transform(y)
        class_counts = np.bincount(y)

        cv = check_cv(self.cv, y, classifier=True)
        if cv.random_state is None:
            cv.random_state = self.random_state

        if self.sampler.seed is None:
            self.sampler.seed = self.random_state

        scorer, scorer_type, greater_is_better = get_scorer_type(self.scoring)

        valid_estimators = []
        ## filter out estimators
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
                if isinstance(n_trials, list):
                    del n_trials[i]
            else:
                valid_estimators.append(estimator)

        if len(valid_estimators)==0:
            print("No valid estimators available for this type of input") #TODO convert to error
            return

        self.run_study_save_results(X, y, valid_estimators, cv, scorer, scorer_type,
                                    greater_is_better, y_stats=class_counts, objective_type="classification",
                                    sample_weight=sample_weight, groups=groups, missing=missing,
                                    cat_cols=cat_cols, timeout_per_estimator=timeout_per_estimator,
                                    n_trials=n_trials)

    def refit(self, X, y):
        #TODO check if best_pipeline exists
        le = LabelEncoder()
        y = le.fit_transform(y)
        self.refit_ = True
        self.best_pipeline_.fit(X, y)

    def predict(self, X):
        self._check_is_fitted('predict')
        return self.best_pipeline_.predict(X)

    def predict_proba(self, X):
        self._check_is_fitted('predict_proba')
        return self.best_pipeline_.predict_proba(X)


class HyPSTERRegressor(HyPSTEREstimator):
    def fit(self, X, y, sample_weight=None, groups=None, missing=None, cat_cols=None,
            n_trials=10, timeout_per_estimator=None):

        #TODO check that y is regression and not classification
        #TODO: consider log-transform y?

        X, y, groups = indexable(X, y, groups)

        ## convert labels to np.array
        y = np.array(y)
        y_mean = np.mean(y)

        cv = check_cv(self.cv, y, classifier=False)
        if cv.random_state is None:
            cv.random_state = self.random_state

        ##initialize seeds
        if self.sampler.seed is None:
            self.sampler.seed = self.random_state

        scorer, scorer_type, greater_is_better = get_scorer_type(self.scoring)

        valid_estimators = []
        ## filter out estimators & transformer
        for i in range(len(self.estimators)):
            remove_estimator = False
            estimator = self.estimators[i]
            estimator.set_default_tags()
            tags = estimator.get_tags()

            if issparse(X) and (tags["handles sparse"] == False):
                remove_estimator = True
                # TODO add logging
            if tags["supports regression"] == False:
                remove_estimator = True
                # TODO add logging

            if not remove_estimator:
                valid_estimators.append(estimator)

        if len(valid_estimators) == 0:
            print("No valid estimators available for this type of input")  # TODO convert to error
            return

        self.run_study_save_results(X, y, valid_estimators, cv, scorer, scorer_type,
                                    greater_is_better, y_stats=y_mean, objective_type="regression",
                                    sample_weight=sample_weight, groups=groups, missing=missing,
                                    cat_cols=cat_cols, timeout_per_estimator=timeout_per_estimator,
                                    n_trials=n_trials)

    def refit(self, X, y):
        #TODO check if best_pipeline exists
        y = np.array(y)
        self.refit_ = True
        self.best_pipeline_.fit(X, y)

    def predict(self, X):
        self._check_is_fitted('predict')
        return self.best_pipeline_.predict(X)

class IdentityTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, input_array, y=None):
        return self

    def transform(self, X, y=None):
        return X