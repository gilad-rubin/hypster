import optuna
from optuna.samplers import TPESampler
from hypster.pruners import LinearExtrapolationPruner
from copy import copy, deepcopy
from sklearn.utils import _safe_indexing  # TODO switch to _safe_indexing
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import check_cv
from sklearn.utils.validation import indexable
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion
from scipy.sparse import issparse

from hypster.utils import *
from hypster.preprocessors import *

# TODO move pipeline functions into utils
def _init_pipeline(pipeline, pipe_params, trial):
    if pipeline is not None:
        final_pipeline = clone(pipeline)
        if pipe_params is not None:
            pipe_params = _get_params(trial, pipe_params)
            final_pipeline.set_params(**pipe_params)
        return final_pipeline
    return None

# TODO: move into utils
def _get_params(trial, params):
    param_dict = {}
    for (key, value) in params.items():
        if "optuna.distributions" in str(type(value)):
            param_dict[key] = trial._suggest(key, value)
        else:
            param_dict[key] = params[key]
    return param_dict


def a_better_equal_b(a, b, greater_is_better):
    if greater_is_better:
        return a >= b
    return a <= b


# TODO move objective into another file (?)
class Objective(object):
    def __init__(self, X, y, estimators, sample_weight=None,
                 missing=None, groups=None, cat_cols=None,
                 numeric_cols=None,  # TODO add missing
                 objective_type="classification", y_stats=None, pipeline=None,
                 pipe_params=None, greater_is_better=True,
                 cv='warn', save_cv_preds=False, scoring=None, scorer_type=None,
                 refit=False, tol=1e-7, agg_func=np.mean, max_iter=30, max_fails=0,
                 random_state=1):

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
        self.random_state = random_state

    def __call__(self, trial):
        #######################
        ### Initializations ###
        #######################

        pipeline = _init_pipeline(self.pipeline, self.pipe_params, trial)
        # TODO replace with self.""?
        X = self.X
        y = self.y
        cat_cols = self.cat_cols
        numeric_cols = self.numeric_cols
        random_state = self.random_state

        ##############################
        ### Choose Estimator & HPs ###
        ##############################
        estimator = deepcopy(trial.suggest_categorical("estimator", self.estimators))
        estimator.choose_and_set_params(trial, self.y_stats, self.missing)
        estimator.remove_trial() #TODO: change it to be more elegant

        ## set params on main estimator
        if estimator.param_dict is not None:
            user_params = _get_params(trial, estimator.param_dict)

        # print(estimator.model_params)

        # overwrite params with user dictionary
        # to avoid hp's that don't comply to the structure of other sampled HPs
        if estimator.param_dict is not None:
            for key, value in user_params.items():
                if key in estimator.model_params.keys():
                    estimator.model_params[key] = value
                # TODO: 'else:' warn user that the key is not compatible with the structure of the other sampled HPs

        tags = estimator.get_tags()
        estimator_name = tags["name"]

        ###########################
        ### Choose Transformers ###
        ###########################

        cat_transforms = ["encode"]  # "impute" #TODO: fix numpy array output from imputation
        transformers = []
        cat_steps = None

        if cat_cols is not None:
            if "impute" in cat_transforms:
                cat_imputer = CatImputer(X, cat_cols, tags, trial, random_state)
                if cat_imputer is not None:
                    transformers.append(("cat_imputer", cat_imputer))
            if "encode" in cat_transforms:
                n_classes = 1 if self.objective_type == "regression" else len(self.y_stats)
                cat_encoder = CatEncoder(X, cat_cols, tags, estimator_name,
                                         self.objective_type, trial, n_classes, random_state)
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
        numeric_transforms = ["impute"]  # , "scale"]
        transformers = []
        numeric_steps = None
        if numeric_cols is not None:
            if "impute" in numeric_transforms:
                imputer = NumericImputer(X, numeric_cols, trial, tags)
                if imputer is not None:
                    transformers.append(("numeric_imputer", imputer))
            if "scale" in numeric_transforms:
                scaler = Scaler(X, numeric_cols, trial, estimator_name, tags)
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

        ###################################
        ### Create K Folds & Estimators ###
        ###################################

        folds = []
        for train_idx, test_idx in self.cv.split(X, y, groups=self.groups):
            X_train, y_train = _safe_indexing(X, train_idx), _safe_indexing(y, train_idx)
            X_test, y_test = _safe_indexing(X, test_idx), _safe_indexing(y, test_idx)

            ## apply pipeline to train+test
            if pipeline is not None:
                X_train = pipeline.fit_transform(X_train, y_train)
                X_test = pipeline.transform(X_test)

            fold_estimator = deepcopy(estimator)

            if self.sample_weight is not None:
                train_sample_weight = _safe_indexing(self.sample_weight, train_idx)
                test_sample_weight = _safe_indexing(self.sample_weight, test_idx)
            else:
                train_sample_weight = None
                test_sample_weight = None

            fold_estimator.set_train(X_train, y_train, sample_weight=train_sample_weight, missing=self.missing)
            fold_estimator.set_test(X_test, y_test, sample_weight=test_sample_weight, missing=self.missing)

            folds.append({"y_test": y_test,
                          "train_idx": train_idx, "test_idx": test_idx,
                          "estimator": fold_estimator})

        ###########################
        ### Train By Iterations ###
        ###########################

        best_score = np.nan
        prune = False
        for step in range(self.max_iter):
            # print("Iteration #", step)
            scores = []
            raw_preds_list = []
            for fold in folds:
                ## train for n_iter while resuming from current model
                fold['estimator'].fit()

                ## get raw predictions
                if self.objective_type == "regression":
                    raw_preds = fold['estimator'].predict()
                else:
                    raw_preds = fold['estimator'].predict_proba()
                    # TODO: what about decision_function? and those who don't have predict_proba?

                # TODO: check that there is a random seed so that stacking will work on the same cv_folds.
                # if not - export the random seed or splits
                if self.save_cv_preds:
                    raw_preds_list.append(raw_preds)

                ## get classes for metrics that deal with classes
                if self.scorer_type == "predict" and self.objective_type == "classification":
                    # TODO handle multiclass
                    threshold = 0.5  # TODO: find optimal threshold w.r.t scoring function
                    raw_preds = (raw_preds >= threshold).astype(int)

                if self.scorer_type == "threshold":
                    raw_preds = raw_preds[:, 1]  # TODO handle multiclass and other scorers

                ##get score & append
                if not np.any(np.isnan(raw_preds)):
                    fold_score = self.scoring(fold["y_test"], raw_preds)
                    scores.append(fold_score)
                else:
                    break

            intermediate_value = self.agg_func(scores)
            # Using "func" in order to avoid pruning just because of overfitting at one certain step:
            if self.greater_is_better:
                func = np.nanmax
            else:
                func = np.nanmin

            report_value = func([intermediate_value, best_score])
            trial.report(report_value, step)

            #print(report_value)

            if a_better_equal_b(intermediate_value, best_score, self.greater_is_better) or step == 0:
                if trial.should_prune():
                    prune = True
                    break

            #########################
            ### Reduce Complexity ###
            #########################
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
                        # TODO handle cases where:
                        # self.cv does not cover the whole dataset (e.g train/test)
                        # self.cv is repeated cross validation. then we should perhaps choose one cover of the whole dataset
                        fold["raw_predictions"] = raw_preds_list[i]

            else:
                break
                # fail_count += 1
                # if (can_lower_complexity == False) or (fail_count >= self.max_fails):
                #     break
                #
                # for fold in folds:
                #     fold['estimator'].lower_complexity()
                #     best_model = deepcopy(fold['estimator'].get_best_model())
                #     fold['estimator'].set_current_model(best_model)

        #####################
        ### Wrap Up Trial ###
        #####################
        # print(pipeline)
        if prune == False:
            model = folds[0]['estimator'].create_model()

            if pipeline is not None:
                pipeline.steps.append(["model", model])
            else:
                pipeline = Pipeline([("model", model)])
            print(estimator_name + ' Score: ' + str(round(best_score, 5)))

            trial.set_user_attr('pipeline', pipeline)

            if self.save_cv_preds:
                n_rows = X.shape[0]

                if self.objective_type == "regression":
                    n_columns = 1
                else:
                    n_columns = folds[0]["raw_predictions"].shape[1]

                raw_preds = np.zeros((n_rows, n_columns))
                for fold in folds:
                    if n_columns == 1:
                        fold_raw_preds = fold['raw_predictions'].reshape(-1, 1)
                    else:
                        fold_raw_preds = fold['raw_predictions']
                    raw_preds[fold["test_idx"], :] = fold_raw_preds
                trial.set_user_attr("cv_preds", raw_preds)

        if np.isnan(best_score):  # TODO fix this
            return report_value

        return best_score

class HyPSTEREstimator():
    def __init__(self, frameworks,
                 model_types,
                 pipeline=None,
                 pipe_params=None,
                 scoring=None,
                 cv=3,
                 agg_func=np.mean,
                 tol=1e-7,
                 max_iter=50,
                 max_fails=0,
                 time_limit=None,
                 study_name=None,
                 save_cv_preds=False,
                 pruner=LinearExtrapolationPruner(n_steps_back=2, n_steps_forward=10, percentage_from_best=90),
                 sampler=TPESampler(**TPESampler.hyperopt_parameters()),
                 storage=None,
                 refit=True,
                 n_jobs=1,
                 verbose=1,
                 random_state=None):

        self.frameworks = frameworks if isinstance(frameworks, list) else [frameworks]
        self.model_types = model_types if isinstance(model_types, list) else [model_types]
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
        # TODO: add CMA Sampler
        self.storage = storage
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.random_state = random_state
        self.best_pipeline_ = None
        # TODO: check validity of columns with X (e.g. pandas with int as columns)

    def fit(self, X=None, y=None, sample_weight=None, groups=None, missing=None, cat_cols=None, n_trials=10):
        raise NotImplementedError

    def run_study(self, X, y, valid_estimators, cv, scorer, scorer_type,
                  greater_is_better, y_stats, objective_type, sample_weight,
                  groups, missing, cat_cols, timeout_per_estimator, n_trials):

        direction = "maximize" if greater_is_better else "minimize"

        for i, estimator in enumerate(valid_estimators):
            if estimator.get_seed() == 1:
                estimator.set_seed(self.random_state)
            if estimator.n_jobs != self.n_jobs:
                estimator.set_n_jobs(self.n_jobs)

        numeric_cols = get_numeric_cols(X, cat_cols)
        objective = Objective(X, y, valid_estimators, sample_weight, groups, missing, cat_cols,
                              numeric_cols=numeric_cols, objective_type=objective_type, y_stats=y_stats,
                              pipeline=self.pipeline, pipe_params=self.pipe_params,
                              greater_is_better=greater_is_better,
                              cv=cv, save_cv_preds=self.save_cv_preds,
                              scoring=scorer._score_func,
                              scorer_type=scorer_type,
                              agg_func=self.agg_func, tol=self.tol,
                              max_iter=self.max_iter, max_fails=self.max_fails,
                              random_state=self.random_state)

        if self.verbose > 0:
            optuna.logging.set_verbosity(optuna.logging.WARN)

        # if study is None:
        study_name = self.study_name if self.study_name else "study"
        study = optuna.create_study(storage=self.storage,
                                    pruner=self.pruner,
                                    sampler=self.sampler,
                                    study_name=study_name,
                                    load_if_exists=False,
                                    direction=direction)

        study.optimize(objective, n_trials=n_trials, n_jobs=self.n_jobs,
                       timeout=timeout_per_estimator)

        self.study = study

    def save_results(self):
        self.best_pipeline_ = self.study.best_trial.user_attrs['pipeline']
        self.best_score_ = self.study.best_value
        self.best_params_ = self.study.best_params
        self.best_index_ = self.study.best_trial.number

        if len(self.best_pipeline_.steps) > 1:  # pipeline has more than just a classifier
            self.best_transformer_ = Pipeline(self.best_pipeline_.steps[:-1])  # return all steps but last (classifier)
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
            return True
            # TODO: replace deprecated _check_is_fitted(self, 'best_pipeline_')

        return valid_estimators

    def predict(self, X):
        self._check_is_fitted('predict')
        return self.best_pipeline_.predict(X)

    # TODO: check if we should implement "score" and "predict_log_proba"

    def visualize_results(self):
        return
        # TODO: plot... with matplotlib/plotly/hvplot
        # TODO: show if it will help to increase max_iter

    def summary(self):
        # TODO estimators tested, estimators left out
        # TODO more statistics about estimators
        return


class HyPSTERClassifier(HyPSTEREstimator):
    def fit(self, X, y, sample_weight=None, groups=None,
            missing=None, cat_cols=None, n_trials=10, timeout_per_estimator=None):

        X, y, groups = indexable(X, y, groups)

        ## convertc labels to np.array
        le = LabelEncoder()
        y = le.fit_transform(y)
        class_counts = np.bincount(y)

        cv = check_cv(self.cv, y, classifier=True)
        if cv.random_state is None:
            cv.random_state = self.random_state

        if self.sampler.seed is None: #TODO: check for CMA
            self.sampler.seed = self.random_state

        scorer, scorer_type, greater_is_better = get_scorer_type(self.scoring)

        valid_estimators = get_estimators(self.frameworks, self.model_types,
                                          objective_type="classification")
        valid_estimators = filter_estimators(X, valid_estimators, class_counts, "classification")

        self.run_study(X, y, valid_estimators, cv, scorer, scorer_type,
                       greater_is_better, y_stats=class_counts, objective_type="classification",
                       sample_weight=sample_weight, groups=groups, missing=missing,
                       cat_cols=cat_cols, timeout_per_estimator=timeout_per_estimator,
                       n_trials=n_trials)

        self.save_results()

        if self.refit_:
            self.best_pipeline_.fit(X, y)

    def refit(self, X, y):
        # TODO check if best_pipeline exists
        le = LabelEncoder()
        y = le.fit_transform(y)
        self.best_pipeline_.fit(X, y)
        self.refit_ = True

    def predict(self, X):
        self._check_is_fitted('predict')
        return self.best_pipeline_.predict(X)

    def predict_proba(self, X):
        self._check_is_fitted('predict_proba')
        return self.best_pipeline_.predict_proba(X)


class HyPSTERRegressor(HyPSTEREstimator):
    def fit(self, X, y, sample_weight=None, groups=None, missing=None, cat_cols=None,
            n_trials=10, timeout_per_estimator=None):

        # TODO check that y is regression and not classification
        # TODO: consider log-transform y?

        X, y, groups = indexable(X, y, groups)

        y = np.array(y)
        y_mean = np.mean(y)

        cv = check_cv(self.cv, y, classifier=False)
        if cv.random_state is None:
            cv.random_state = self.random_state

        if self.sampler.seed is None:
            self.sampler.seed = self.random_state

        scorer, scorer_type, greater_is_better = get_scorer_type(self.scoring)

        valid_estimators = get_estimators(self.frameworks, self.model_types,
                                          objective_type="regression")

        valid_estimators = filter_estimators(X, valid_estimators, y_mean, "regression")

        self.run_study(X, y, valid_estimators, cv, scorer, scorer_type,
                       greater_is_better, y_stats=y_mean, objective_type="regression",
                       sample_weight=sample_weight, groups=groups, missing=missing,
                       cat_cols=cat_cols, timeout_per_estimator=timeout_per_estimator,
                       n_trials=n_trials)

        self.save_results()

        if self.refit_:
            self.best_pipeline_.fit(X, y)

    def refit(self, X, y):
        # TODO check if best_pipeline exists
        y = np.array(y)
        self.refit_ = True
        self.best_pipeline_.fit(X, y)

    def predict(self, X):
        self._check_is_fitted('predict')
        return self.best_pipeline_.predict(X)


# TODO move to another file
class IdentityTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, input_array, y=None):
        return self

    def transform(self, X, y=None):
        return X


def get_estimators(frameworks, model_types, objective_type):
    # frameworks: ["xgboost", "lightgbm", "sklearn", "catboost", "tf", "keras", "fastai", "all"]
    # model types: ["tree_based", "linear", "deep_learning"]
    xgb_min_ver = "0.9"
    lgb_min_ver = "2.3.1"
    sklearn_min_ver = "0.22"
    # TODO: add exclude frameworks
    estimators = []
    for framework in frameworks:
        if str.startswith(framework, "xgb"):
            try:
                import xgboost as xgb
                xgb_ver = xgb.__version__
                if ge_version(xgb_ver, xgb_min_ver):
                    if any(item.startswith('linear') for item in model_types):
                        if objective_type == "classification":
                            from hypster.estimators.classification.xgboost import XGBLinearClassifierHypster
                            estimators.append(XGBLinearClassifierHypster())
                        else:
                            from hypster.estimators.regression.xgboost import XGBLinearRegressorHypster
                            estimators.append(XGBLinearRegressorHypster())
                    if any(item.startswith('tree') for item in model_types):
                        if objective_type == "classification":
                            from hypster.estimators.classification.xgboost import XGBTreeClassifierHypster
                            estimators.append(XGBTreeClassifierHypster())
                        else:
                            from hypster.estimators.regression.xgboost import XGBTreeRegressorHypster
                            estimators.append(XGBTreeRegressorHypster())
            except:
                # TODO: log that xgboost is not installed in the right version
                continue

        if str.startswith(framework, ("lgb", "lightgbm")):
            try:
                import lightgbm as lgb
                lgb_ver = lgb.__version__
                if ge_version(lgb_ver, lgb_min_ver):
                    if any(item.startswith('tree') for item in model_types):
                        if objective_type == "classification":
                            from hypster.estimators.classification.lightgbm import LGBClassifierHypster
                            estimators.append(LGBClassifierHypster())
                        else:
                            from hypster.estimators.regression.lightgbm import LGBRegressorHypster
                            estimators.append(LGBRegressorHypster())
            except:
                # TODO: log that lightgbm is not installed in the right version
                continue

        if str.startswith(framework, ("sklearn", "scikit", "sci-kit")):
            try:
                import sklearn
                sklearn_ver = sklearn.__version__
                if ge_version(sklearn_ver, sklearn_min_ver):
                    if any(item.startswith('linear') for item in model_types):
                        if objective_type == "classification":
                            from hypster.estimators.classification.sgd import SGDClassifierHypster
                            estimators.append(SGDClassifierHypster())
                        else:
                            from hypster.estimators.regression.sgd import SGDRegressorHypster
                            estimators.append(SGDRegressorHypster())
                    # if any(item.startswith('tree') for item in model_types):
                    #     if objective_type == "classification":
                    #         from hypster.estimators.classification.ensembles import RFClassifierHyPSTER, \
                    #             ETClassifierHyPSTER, GBTClassifierHyPSTER
                    #         estimators.extend([RFClassifierHyPSTER(), ETClassifierHyPSTER(), GBTClassifierHyPSTER()])
                    #     else:
                    #         from hypster.estimators.regression.ensembles import RFRegressorHyPSTER, \
                    #             ETRegressorHyPSTER, GBTRegressorHyPSTER
                    #         estimators.extend([RFRegressorHyPSTER(), ETRegressorHyPSTER(), GBTRegressorHyPSTER()])
            except:
                # TODO: log that sklearn is not installed in the right version
                continue
    return estimators

def filter_estimators(X, estimators, y_stats, objective_type):
    if objective_type == "classification":
        n_classes = len(y_stats)

    valid_estimators = []
    for i, estimator in enumerate(estimators):
        tags = estimator.get_tags()
        if issparse(X) and (tags["handles sparse"] == False):
            continue
            # TODO add logging
        elif (objective_type == "classification") and (tags["supports multiclass"] == False) and (n_classes > 2):
            continue
            #TODO: add logging
        else:
            valid_estimators.append(estimator)

    if len(valid_estimators) == 0:
        print("No valid estimators available for this type of input")  # TODO convert to error
        return

    return valid_estimators
