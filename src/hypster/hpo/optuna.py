"""Optuna adapter: runs a config with trial-suggested values via HP's value-provider seam.

Any trial-like object with Optuna's ``suggest_int``/``suggest_float``/
``suggest_categorical`` methods works; optuna itself is not imported.
"""

from __future__ import annotations

from typing import Any, Dict, List

from .._sentinels import NOT_PROVIDED
from ..core import ConfigFunc, _run_config
from .types import HpoCategorical, HpoFloat, HpoInt

#: Parameter kinds the Optuna adapter can tune. Other kinds (bool, text,
#: multi_*, rules, schema) fall back to their defaults during suggestion.
TUNABLE_KINDS = frozenset({"int", "float", "select"})


def _exclusive_high_for_int(low: int, high: int, step: int | None) -> int:
    actual_step = step or 1
    if actual_step <= 0:
        raise ValueError("HpoInt(step=...) must be positive.")
    if high <= low:
        raise ValueError("HpoInt(include_max=False) leaves no valid integer values below max.")
    return low + ((high - low - 1) // actual_step) * actual_step


def _validate_int_spec(full: str, hpo_spec: HpoInt | None) -> None:
    if hpo_spec is None:
        return
    if hpo_spec.base != 10.0:
        raise ValueError(f"Parameter '{full}': HpoInt(base=...) is not supported by the Optuna suggest_int adapter.")


def _float_log_flag(full: str, hpo_spec: HpoFloat | None) -> bool:
    if hpo_spec is None:
        return False
    if hpo_spec.base != 10.0:
        raise ValueError(
            f"Parameter '{full}': HpoFloat(base=...) is not supported by the Optuna suggest_float adapter."
        )
    if hpo_spec.center is not None or hpo_spec.spread is not None:
        raise ValueError(
            f"Parameter '{full}': HpoFloat(center=..., spread=...) is only meaningful for normal/lognormal "
            "distributions, which are not supported by the Optuna suggest_float adapter."
        )
    if hpo_spec.distribution in {"normal", "lognormal"}:
        raise ValueError(
            f"Parameter '{full}': HpoFloat(distribution={hpo_spec.distribution!r}) is not supported by the "
            "Optuna suggest_float adapter."
        )
    if hpo_spec.distribution == "uniform":
        return False
    if hpo_spec.distribution == "loguniform":
        return True
    return hpo_spec.scale == "log"


def _validate_categorical_spec(full: str, hpo_spec: HpoCategorical | None) -> None:
    if hpo_spec is None:
        return
    if hpo_spec.ordered:
        raise ValueError(
            f"Parameter '{full}': HpoCategorical(ordered=True) is not supported by the Optuna "
            "suggest_categorical adapter."
        )
    if hpo_spec.weights is not None:
        raise ValueError(
            f"Parameter '{full}': HpoCategorical(weights=...) is not supported by the Optuna "
            "suggest_categorical adapter."
        )


def _reject_nullable_numeric(full: str, allow_none: bool, default: Any) -> None:
    if allow_none:
        raise ValueError(
            f"Parameter '{full}': allow_none=True is not supported for HPO numeric suggestions yet.\n\n"
            "How to fix: remove allow_none=True from HPO numeric parameters, or make this choice categorical."
        )
    if default is None:
        raise ValueError(
            f"Parameter '{full}': default=None requires allow_none=True, "
            "but nullable numeric HPO suggestions are not supported yet."
        )


class TrialValueProvider:
    """Asks an Optuna-style trial for values of tunable parameter kinds.

    Declines every other kind, so those parameters keep their defaults and
    still pass through HP's regular validation.
    """

    def __init__(self, trial: Any):
        self.trial = trial

    def provide_value(
        self,
        *,
        path: str,
        kind: str,
        default: Any,
        allow_none: bool = False,
        strict: bool = False,
        options: List[Any] | None = None,
        min: Any = None,
        max: Any = None,
        hpo_spec: Any = None,
    ) -> Any:
        if kind == "int":
            _validate_int_spec(path, hpo_spec)
            _reject_nullable_numeric(path, allow_none, default)
            low = default if min is None else min
            high = default if max is None else max
            step = hpo_spec.step if hpo_spec else None
            log = (hpo_spec.scale == "log") if hpo_spec else False
            if hpo_spec and not hpo_spec.include_max and high is not None:
                high = _exclusive_high_for_int(low, high, step)
            return self.trial.suggest_int(path, low, high, step=step, log=log)

        if kind == "float":
            log = _float_log_flag(path, hpo_spec)
            _reject_nullable_numeric(path, allow_none, default)
            low = default if min is None else min
            high = default if max is None else max
            step = hpo_spec.step if hpo_spec else None
            return self.trial.suggest_float(path, low, high, step=step, log=log)

        if kind == "select":
            _validate_categorical_spec(path, hpo_spec)
            return self.trial.suggest_categorical(path, options)

        return NOT_PROVIDED


class _TunedParamsCollector:
    """Collect the values of tunable params — suggested or overridden — for replay."""

    def __init__(self) -> None:
        self.params: Dict[str, Any] = {}

    def record_parameter(self, *, path: str, kind: str, selected_value: Any, **event: Any) -> None:
        if kind in TUNABLE_KINDS:
            self.params[path] = selected_value

    def record_nest(self, **event: Any) -> None:
        pass


def suggest_values(trial: Any, config: ConfigFunc[Any], /, **kwargs: Any) -> Dict[str, Any]:
    """Run the config with a trial-backed HP to produce a values dict.

    This respects conditionals: only parameters touched in the executed path
    are suggested and returned. Parameter kinds without an Optuna mapping
    (bool, text, multi_*, rules, schema) fall back to their defaults and are
    not part of the returned dict.
    """
    provider = TrialValueProvider(trial)
    collector = _TunedParamsCollector()
    _run_config(config, kwargs=kwargs, parameter_tracker=collector, value_provider=provider)
    return collector.params
