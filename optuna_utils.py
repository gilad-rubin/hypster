import pandas as pd
import numpy as np
import scipy
import sklearn

class Objective(object):
    def __init__(self
                 , X
                 , y
                 , pipeline
                 , pipe_params
                 , estimator
                 , tol=1e-7
                 , cv=5
                 , groups=None
                 , agg_func=np.mean
                 , max_iter=1000
                 , max_fails=10
                 , scoring=sklearn.metrics.balanced_accuracy_score
                 # ,num_iter_check=5
                 ):
        self.X = X
        self.y = y
        self.pipeline = pipeline
        self.pipe_params = pipe_params
        self.estimator = estimator
        self.tol = tol
        self.cv = cv
        self.agg_func = agg_func
        self.groups = groups
        self.max_iter = max_iter
        self.max_fails = max_fails
        self.scoring = scoring

    def __call__(self, trial):
        if self.pipeline is not None:
            pipeline = clone(self.pipeline)
            pipe_params = self._get_params(trial)
            pipeline.set_params(**pipe_params)

        weights = sklearn.utils.class_weight.compute_class_weight("balanced", np.unique(self.y), self.y)
        estimator_list = []

        # create k folds and estimators
        for train, test in self.cv.split(self.X, self.y, groups=self.groups):
            X_train, y_train = safe_indexing(self.X, train), safe_indexing(self.y, train)
            X_test, y_test = safe_indexing(self.X, test), safe_indexing(self.y, test)

            if pipeline is not None:
                X_train = pipeline.fit_transform(X_train, y_train)
                X_test = pipeline.transform(X_test)

            self.estimator.choose_and_set_params(trial, weights)
            estimator = deepcopy(self.estimator)

            estimator.set_train(X_train, y_train)
            estimator.set_test(X_test, y_test)

            estimator_list.append(estimator)

        best_score = 0.0
        for step in range(self.max_iter):
            scores = []
            for estimator in estimator_list:
                estimator.train_one_iteration()
                fold_score = estimator.score_test(self.scoring)
                scores.append(fold_score)

            intermediate_value = self.agg_func(scores)
            trial.report(intermediate_value, step)

            # print("intermediate result = ", intermediate_value)

            if trial.should_prune(step):
                raise optuna.structs.TrialPruned()

            if intermediate_value >= best_score + self.tol:
                best_score = intermediate_value
                fail_count = 0
                for estimator in estimator_list:
                    estimator.save_best()
            else:
                fail_count += 1
                if (fail_count >= self.max_fails) or (not hasattr(estimator_list[0], 'lower_complexity')):
                    break

                # TODO: make this step only after k times
                for estimator in estimator_list:
                    estimator.lower_complexity()

        model = estimator.create_model()
        if pipeline is not None:
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