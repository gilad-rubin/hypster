from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

try:  # optional dependency
    import optuna as _optuna
except Exception:  # pragma: no cover
    _optuna = None

from .types import HpoCategorical, HpoFloat, HpoInt


def _flatten(d: Mapping[str, Any], prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        else:
            out[key] = v
    return out


class _HPProxy:
    def __init__(
        self,
        trial: Any,
        collector: Dict[str, Any],
        ns: list[str] | None = None,
        overrides: Dict[str, Any] | None = None,
    ):
        self.trial = trial
        self.collector = collector
        self.ns = ns or []
        self.overrides = overrides or {}

    def _full(self, name: str) -> str:
        return ".".join(self.ns + [name]) if self.ns else name

    # integers
    def int(
        self,
        default: int,
        *,
        name: str,
        min: int | None = None,
        max: int | None = None,
        strict: bool = False,
        hpo_spec: HpoInt | None = None,
    ) -> int:
        full = self._full(name)
        if full in self.overrides:
            val = int(self.overrides[full])
            self.collector[full] = val
            return val
        low = default if min is None else min
        high = default if max is None else max
        step = hpo_spec.step if hpo_spec else None
        log = (hpo_spec.scale == "log") if hpo_spec else False
        if hpo_spec and not hpo_spec.include_max and high is not None:
            high = high - (step or 1)
        val = self.trial.suggest_int(full, low, high, step=step, log=log)
        self.collector[full] = val
        return val

    # floats
    def float(
        self,
        default: float,
        *,
        name: str,
        min: float | None = None,
        max: float | None = None,
        strict: bool = False,
        hpo_spec: HpoFloat | None = None,
    ) -> float:
        full = self._full(name)
        if full in self.overrides:
            val = float(self.overrides[full])
            self.collector[full] = val
            return val
        low = default if min is None else min
        high = default if max is None else max
        step = hpo_spec.step if hpo_spec else None
        log = (hpo_spec.scale == "log") if hpo_spec else False
        val = self.trial.suggest_float(full, low, high, step=step, log=log)
        self.collector[full] = val
        return val

    # categorical
    def select(
        self,
        options: Sequence[Any] | Mapping[Any, Any],
        *,
        name: str,
        default: Any | None = None,
        options_only: bool = False,
        hpo_spec: HpoCategorical | None = None,
    ) -> Any:
        full = self._full(name)
        if full in self.overrides:
            key_or_val = self.overrides[full]
            if isinstance(options, dict):
                self.collector[full] = key_or_val
                return options[key_or_val]
            self.collector[full] = key_or_val
            return key_or_val

        if isinstance(options, dict):
            keys = list(options.keys())
            key = self.trial.suggest_categorical(full, keys)
            self.collector[full] = key
            return options[key]
        else:
            choice = self.trial.suggest_categorical(full, list(options))
            self.collector[full] = choice
            return choice

    # nesting
    def nest(
        self,
        child,
        *,
        name: str,
        values: Dict[str, Any] | None = None,
        args: tuple = (),
        kwargs: Dict[str, Any] | None = None,
    ) -> Any:
        overrides = dict(self.overrides)
        if values:
            flat = _flatten(values)
            for k, v in flat.items():
                overrides[f"{self._full(name)}.{k}"] = v
        nested = _HPProxy(self.trial, self.collector, ns=self.ns + [name], overrides=overrides)
        kwargs = kwargs or {}
        return child(nested, *args, **kwargs)


def suggest_values(trial: Any, *, config, args: tuple = (), kwargs: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Run the config with a trial-backed HP to produce a values dict.

    This respects conditionals: only parameters touched in the executed path
    are suggested and returned.
    """
    # Optuna is optional; any trial-like object with suggest_* methods works.
    # We intentionally do not import or require optuna here for testability.
    collector: Dict[str, Any] = {}
    hp = _HPProxy(trial, collector)
    kwargs = kwargs or {}
    config(hp, *args, **kwargs)
    return collector
