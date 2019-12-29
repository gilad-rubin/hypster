import math

import numpy as np

from optuna.pruners import BasePruner
from optuna import structs

def a_better_equal_b(a, b, direction):
    if direction == structs.StudyDirection.MAXIMIZE:
        return a >= b
    return a <= b

def _get_extrapolated_step_increment(start_step, end_step, intermediate_values):
    start_value = intermediate_values[start_step]
    end_value = intermediate_values[end_step]
    return (end_value - start_value) / (end_step - start_step)

def _get_best_intermediate_step_value(all_trials, direction, step):
    #completed_trials = [t for t in all_trials if t.state == structs.TrialState.COMPLETE]
    completed_trials = all_trials

    if len(completed_trials) == 0:
        raise ValueError("No trials have been completed.")

    if direction == structs.StudyDirection.MAXIMIZE:
        func = np.nanmax
    else:
        func = np.nanmin

    return float(
        func(
            np.array([
                t.intermediate_values[step]
                for t in completed_trials if step in t.intermediate_values
            ], np.float)
            ))


class LinearExtrapolationPruner(BasePruner):
    def __init__(self, n_steps_back=2, n_steps_forward=5, percentage_from_best=90,
                 n_startup_trials=0, n_warmup_steps=0, interval_steps=1):

        if n_startup_trials < 0:
            raise ValueError(
                'Number of startup trials cannot be negative but got {}.'.format(n_startup_trials))
        if n_warmup_steps < 0:
            raise ValueError(
                'Number of warmup steps cannot be negative but got {}.'.format(n_warmup_steps))
        if interval_steps < 1:
            raise ValueError(
                'Pruning interval steps must be at least 1 but got {}.'.format(interval_steps))

        self._n_steps_back = n_steps_back #TODO Change name
        self._n_steps_forward = n_steps_forward # TODO Change name
        if percentage_from_best > 1.0:
            self._percentage_from_best = percentage_from_best * 0.01
        else:
            self._percentage_from_best = percentage_from_best
        #TODO add checkups
        self._n_startup_trials = n_startup_trials
        self._n_warmup_steps = n_warmup_steps
        self._interval_steps = interval_steps

    def prune(self, study, trial):
        intermediate_values = trial.intermediate_values
        all_trials = study.trials
        #all_trials = study.get_trials(deepcopy=False)

        #n_trials = len([t for t in all_trials if t.state == structs.TrialState.COMPLETE])
        n_trials = len(all_trials)

        if n_trials == 0:
            return False

        if n_trials < self._n_startup_trials:
            return False

        step = trial.last_step
        if step is None:
            return False

        n_warmup_steps = self._n_warmup_steps
        if step < n_warmup_steps:
            return False

        direction = study.direction
        step_value = trial.value
        if math.isnan(step_value):
            return True

        best_intermediate_value = _get_best_intermediate_step_value(all_trials, direction, step)
        if math.isnan(best_intermediate_value):
            return False

        if a_better_equal_b(step_value, best_intermediate_value, direction):
            return False

        n_steps_back = self._n_steps_back
        n_steps_forward = self._n_steps_forward
        percentage_from_best = self._percentage_from_best
        if step == 0:
            if direction == structs.StudyDirection.MINIMIZE:
                percentage_from_best = 1.0 + percentage_from_best
            perc_best_value = best_intermediate_value * percentage_from_best
            if a_better_equal_b(step_value, perc_best_value, direction):
                return False
            else:
                return True

        if step <= n_steps_back:
            start_step = 0
        else:
            start_step = step - n_steps_back

        step_inc = _get_extrapolated_step_increment(start_step, step, intermediate_values)
        extrapolated_value = step_value + (n_steps_forward * step_inc)
        if a_better_equal_b(extrapolated_value, best_intermediate_value, direction):
            return False
        return True

        # TODO: handle plateau or worse?
        # if improve_value == 0:
        #     steps_back = min(len(intermediate_values) - 1, fail_count + 2)
        #     improve_value = (report_value - intermediate_values[step - steps_back]) / steps_back