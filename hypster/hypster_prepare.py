# AUTOGENERATED! DO NOT EDIT! File to edit: 03_hypster_prepare.ipynb (unless otherwise specified).

__all__ = ['set_name', 'set_name_from_arg', 'set_names_from_args', 'HypsterPrepare', 'prepare', 'run_func_test']

# Cell
from .oo_hp import *

# Cell
from inspect import signature
import functools
from collections import OrderedDict

# Cell
def set_name(arg, name):
    if arg.manual_name == False:
        arg.name = name

# Cell
def set_name_from_arg(arg, name):
    if isinstance(arg, HpAtomic):
        set_name(arg, name)

    if isinstance(arg, HpToggle):
        set_name_from_arg(arg.hp, name)
        if arg.manual_name == False:
            arg.name = f"toggle_{get_name(arg.hp)}"

    if isinstance(arg, (list, tuple)):
        [set_name_from_arg(item, name) for item in arg]

    if isinstance(arg, HpVarLenIterable):
        set_name(arg, f"n_{name}")
        set_name_from_arg(arg.hp, name)

    if isinstance(arg, HpIterable):
        #TODO: continue
        list(set([item for item in arg]))

    if isinstance(arg, HypsterPrepare):
        print(arg.call)
        if hasattr(arg.call, "__class__") and hasattr(arg.call.__class__, "__name__"):
            arg.name = f"{arg.call.__class__.__name__}_{arg.name}"

# Cell
def set_names_from_args(arg_names, args, kwargs):
    # if argument is an iterable -> (start_mom, ..., ...) use its object name + counting index
    # fit_method = learner.fit_one_cycle(2, lr) --> during sampling, get the function signature and change the name
    for i, arg in enumerate(args):
        set_name_from_arg(arg, arg_names[i])

    for key, value in kwargs.items():
        set_name_from_arg(value, key)

# Cell
class HypsterPrepare(HypsterBase):
    def __init__(self, call, base_call, *args, **kwargs):
        #allow future option to add a prefix of a model name
        self.call            = call
        self.base_call       = base_call
        self.args            = args
        self.kwargs          = kwargs
        self.trials_sampled  = set()
        self.studies_sampled = set()
        self.base_object     = None
        self.result          = None

        if callable(call):
            self.arg_names = list(signature(call).parameters.keys())
            set_names_from_args(self.arg_names, self.args, self.kwargs)

    def sample(self, trial):
        if trial.study.study_name not in self.studies_sampled:
            self.trials_sampled = set()
        elif trial.number in self.trials_sampled:
            return self.result

        if self.base_call is not None:
            self.base_object = self.base_call.sample(trial)

        self.sampled_args   = populate_iterable(self.args, trial)
        self.sampled_kwargs = populate_dict(self.kwargs, trial)

        self.trials_sampled.add(trial.number)
        self.studies_sampled.add(trial.study.study_name)

        if self.base_object:
            if len(self.sampled_args) == 0 and len(self.sampled_kwargs) == 0:
                self.result = getattr(self.base_object, self.call)
            else:
                self.result = getattr(self.base_object, self.call)(*self.sampled_args, **self.sampled_kwargs)
        else:
            self.result = self.call(*self.sampled_args, **self.sampled_kwargs)
        return self.result

    def __call__(self, *args, **kwargs):
        #print(f"args {args}, kwargs {kwargs}")
        self.args = args
        self.kwargs = kwargs
        return self

    def __getattr__(self, name, *args, **kwargs):
        #print(f"name {name}, args {args}, kwargs {kwargs}")
        return HypsterPrepare(name, self, *args, **kwargs)

# Cell
def prepare(call):
    @functools.wraps(call)
    def wrapper_decorator(*args, **kwargs):
        all_args = list(args) + list(kwargs.values())
        if any([contains_hypster(arg, HYPSTER_TYPES) for arg in all_args]):
            return HypsterPrepare(call, None, *args, **kwargs)
        else:
            return call(*args, **kwargs)
    return wrapper_decorator

# Cell
import optuna

# Cell
def run_func_test(x, n_trials=5):
    def objective(trial):
        print(x.sample(trial))
        return 1.0

    optuna.logging.set_verbosity(0)
    pruner = optuna.pruners.NopPruner()
    study = optuna.create_study(direction="maximize", pruner=pruner)
    study.optimize(objective, n_trials=n_trials, timeout=600)