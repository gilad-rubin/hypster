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

from sklearn.base import BaseEstimator, TransformerMixin

class Objective(object):
    def __init__(self, X, y, cat_columns, estimator, objective_type="classification",
                 y_stats=None, pipeline=None, pipe_params=None, greater_is_better=True,
                 cv='warn', save_cv_preds=False, scoring=None, scorer_type=None,
                 refit=False, tol=1e-5, agg_func=np.mean, max_iter=50, max_fails=3):

        self.X = X
        self.y = y
        self.cat_columns = cat_columns
        self.estimator = estimator
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

    # def init_pipeline(self, trial):
    #     pipeline = None
    #     if self.pipeline is not None:
    #         pipeline = clone(self.pipeline)
    #         if self.pipe_params is not None:
    #             pipe_params = _get_params(trial, self.pipe_params)
    #             pipeline.set_params(**pipe_params)

    def __call__(self, trial):

        # init_pipeline(self.pipeline)
        ## init pipeline

        pipeline = None
        if self.pipeline is not None:
            pipeline = clone(self.pipeline)
            if self.pipe_params is not None:
                pipe_params = _get_params(trial, self.pipe_params)
                pipeline.set_params(**pipe_params)

        ## impute & convert categorical columns

        #TODO return sparse or dense? consider that in sparse sometimes na is 0
        #TODO what if model handles imputation?
        #TODO make this into a seperate class
        if (self.cat_columns is not None) and (self.estimator.get_properties()['handles_categorical'] == False):
            impute_ohe_pipe = Pipeline([('impute', SimpleImputer(strategy="constant", fill_value="unknown")),
                                         ('ohe', OneHotEncoder(categories="auto", handle_unknown='ignore'))])
            impute_ohe_ct = ColumnTransformer([('impute_ohe', impute_ohe_pipe, self.cat_columns)],
                                              remainder="passthrough")
            if pipeline is not None:
                #TODO where to append? at the beginning or end?
                pipeline.steps.append(['impute_ohe', impute_ohe_ct])
            else:
                pipeline = Pipeline([('impute_ohe', impute_ohe_ct)])

        can_lower_complexity = self.estimator.get_properties()["can_lower_complexity"]
        ##TODO think of a better name than "can_lower..."
        
        ## choose estimator from list
        estimator_list = []
        
        ## set params on main estimator
        ## TODO Create LRFinder for sklearn compatible objects (Separate greedy/global estimators)
        if self.estimator.param_dict is not None:
            user_params = _get_params(trial, self.estimator.param_dict)

        estimator = deepcopy(self.estimator)
        estimator.choose_and_set_params(trial, self.y_stats)

        ## overwrite params with user dictionary
        # to avoid hp's that don't work with conditions
        # TODO fix it in cases where the user wants to add a hp
        # TODO get this out of the loop
        if estimator.param_dict is not None:
            for (key, value) in user_params.items():  # override default params
                if key in estimator.model_params.keys():
                    estimator.model_params[key] = value

        ## create k folds and estimators
        #TODO save indices for splits
        #TODO use cv like in GridSearchCV
        folds = []
        for train_idx, test_idx in self.cv.split(self.X, self.y):  # groups=self.groups
            X_train, y_train = safe_indexing(self.X, train_idx), safe_indexing(self.y, train_idx)
            X_test, y_test = safe_indexing(self.X, test_idx), safe_indexing(self.y, test_idx)

            ## apply pipeline to train+test
            if pipeline is not None:
                X_train = pipeline.fit_transform(X_train, y_train)
                X_test = pipeline.transform(X_test)

            fold_estimator = deepcopy(estimator)
            #fold_estimator.set_train_test(X_train, y_train, X_test, y_test, self.cat_columns)
            #estimator.init_model() #TODO inspect

            folds.append({"X_train" : X_train, "y_train" : y_train,
                          "X_test" : X_test, "y_test" : y_test,
                          "train_idx" : train_idx, "test_idx" : test_idx,
                          "estimator": fold_estimator})

        if self.greater_is_better:
            best_score = 0.0
        else:
            best_score = np.inf

        n_iter_per_round = estimator.get_n_iter_per_round()
        total_iterations = 0 #counts the number of total iterations

        for step in range(self.max_iter):
            # print("Iteration #", step)
            scores = []
            raw_preds_list = []
            for fold in folds:
                ## train for n_iter while resuming from current model
                #model = fold['estimator'].get_current_model()
                fold['estimator'].fit(fold['X_train'], fold['y_train'])
                total_iterations += n_iter_per_round

                ## get raw predictions
                if self.objective_type == "regression":
                    raw_preds = fold['estimator'].predict(fold['X_test'])
                else:
                    # TODO check that the result is equal in shape to n_rows, n_classes
                    raw_preds = fold['estimator'].predict_proba(fold['X_test'])

                raw_preds_list.append(raw_preds)

                ## get classes for metrics that deal with classes
                if self.scorer_type == "predict" and self.objective_type == "classification":
                    # TODO handle multiclass
                    threshold = 0.5 #TODO: find optimal threshold
                    raw_preds = (raw_preds >= threshold).astype(int)

                if self.scorer_type == "threshold":
                    raw_preds = raw_preds[:,1] #TODO handle multiclass and other scorers

                ##get score & append
                fold_score = self.scoring(fold['y_test'], raw_preds)
                scores.append(fold_score)

            intermediate_value = self.agg_func(scores)
            trial.report(intermediate_value, step)

            #print("intermediate result = ", intermediate_value)

            if trial.should_prune(step):
                raise optuna.structs.TrialPruned()

            if intermediate_value >= best_score + self.tol:
                best_score = intermediate_value
                fail_count = 0
                for (i, fold) in enumerate(folds):
                    fold['estimator'].save_best()
                    if self.save_cv_preds:
                        fold["raw_predictions"] = raw_preds[i]
            else:
                fail_count += 1
                if (can_lower_complexity == False) or (fail_count >= self.max_fails):
                    break

                # TODO: make this step only after k times
                for fold in folds:
                    fold['estimator'].lower_complexity()

        model = folds[0]['estimator'].get_best_model() #TODO do it nicer
        if pipeline is not None:
            pipeline.steps.append(["model", model])
        else:
            pipeline = Pipeline([("model", model)]) #TODO should we just return the model?

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
                 tol=1e-5,
                 max_iter=200, #TODO: add iterations if the learning curve prediction is high?
                 time_limit=None, max_fails=3,
                 study_name="",  # TODO: think how to allow the user to define if they want to keep training or not?
                 save_cv_preds=False, #TODO: add support for stacking
                 pruner=SuccessiveHalvingPruner(min_resource=3, reduction_factor=3),
                 sampler=TPESampler(**TPESampler.hyperopt_parameters()),
                 n_jobs=1,
                 verbose=1, #TODO Add support for verbosity
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
        self.study_name = study_name #TODO:needed? maybe replace with "resume last study"?
        self.save_cv_preds = save_cv_preds
        self.pruner = pruner
        self.sampler = sampler
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.random_state = random_state
        self.best_estimator_ = None

        # TODO: set_seed function for samplers & cv_function(?) with random_state

    def fit(self, X=None, y=None, cat_columns=None, n_trials_per_estimator=10): #dataset_name
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

    def predict(self, X): #TODO
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
    def fit(self, X=None, y=None, cat_columns=None, n_trials=10): #dataset_name
        #TODO check that y is classification and not regression
        le = LabelEncoder()
        y = le.fit_transform(y)
        class_counts = np.bincount(y)

        scorer = sklearn.metrics.get_scorer(self.scoring)
        # TODO check if we can make it a bit nicer
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
        else:
            greater_is_better = True

        ## Get only valid estimators and transformers, using: scorer_type(proba/threshold),
        ## dense/sparse, multiclass, multilabel, positive_only X

        studies = []
        for i in range(len(self.estimators)):
            estimator = self.estimators[i]

            if estimator.get_seed() == 1: #TODO make it nice and consider removing setters and getters
                estimator.set_seed(self.random_state)

            estimator.n_jobs =  self.n_jobs #TODO make it nice and consider removing setters and getters

            print(estimator.get_properties()["name"]) #TODO: convert to static method?

            #TODO cat_columns = list of indices or names?
            objective = Objective(X, y, cat_columns, objective_type="classification",
                                  y_stats=class_counts, estimator=estimator,
                                  pipeline=self.pipeline, pipe_params=self.pipe_params,
                                  greater_is_better=greater_is_better,
                                  cv=self.cv, save_cv_preds=self.save_cv_preds,
                                  scoring=scorer._score_func,
                                  scorer_type = scorer_type,
                                  agg_func=self.agg_func, tol=self.tol,
                                  max_iter=self.max_iter, max_fails=self.max_fails)

            if self.verbose > 0:
                optuna.logging.set_verbosity(optuna.logging.WARN)
                # TODO: change

            if greater_is_better==True:
                direction = "maximize"
            else:
                direction = "minimize"

            study = optuna.create_study(pruner=self.pruner,
                                        sampler=self.sampler,
                                        study_name="study",
                                        direction=direction)

            study.optimize(objective, n_trials=n_trials, n_jobs=self.n_jobs)
            studies.append(study)

            # if type(n_trials_per_estimator) == list:
            #     n_trials = n_trials_per_estimator[i] #TODO what if the list is shorter than num of estimators?
            # else:
            #     n_trials = n_trials_per_estimator

            #studies[i].optimize(objective, n_trials=n_trials, n_jobs=self.n_jobs)

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
            #TODO: check if estimator has predict_proba. maybe remove from abstract class
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

            if greater_is_better==True:
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

class IdentityTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, input_array, y=None):
        return self

    def transform(self, X, y=None):
        return X

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