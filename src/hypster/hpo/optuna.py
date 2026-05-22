from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Dict

try:  # optional dependency
    import optuna as _optuna
except Exception:  # pragma: no cover
    _optuna = None

from .._sentinels import NO_DEFAULT as _NO_DEFAULT
from ..utils import normalize_values, validate_identifier_name, validate_select_choice
from .types import HpoCategorical, HpoFloat, HpoInt


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
        validate_identifier_name(name, kind="parameter name")
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
        allow_none: bool = False,
        hpo_spec: HpoInt | None = None,
    ) -> int:
        full = self._full(name)
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
        if full in self.overrides:
            if self.overrides[full] is None:
                raise ValueError(
                    f"Parameter '{full}': None overrides are not supported for HPO numeric suggestions yet."
                )
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
        allow_none: bool = False,
        hpo_spec: HpoFloat | None = None,
    ) -> float:
        full = self._full(name)
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
        if full in self.overrides:
            if self.overrides[full] is None:
                raise ValueError(
                    f"Parameter '{full}': None overrides are not supported for HPO numeric suggestions yet."
                )
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
        default: Any = _NO_DEFAULT,
        options_only: bool = False,
        allow_none: bool = False,
        hpo_spec: HpoCategorical | None = None,
    ) -> Any:
        full = self._full(name)
        if isinstance(options, Mapping):
            keys = list(options.keys())
        else:
            keys = list(options)
        for index, option in enumerate(keys):
            validate_select_choice(option, param_path=f"{full} option #{index}", allow_none=allow_none)
        if default is not _NO_DEFAULT:
            validate_select_choice(default, param_path=full, allow_none=allow_none)

        if full in self.overrides:
            key_or_val = self.overrides[full]
            validate_select_choice(key_or_val, param_path=full, allow_none=allow_none)
            if options_only and key_or_val not in keys:
                options_str = ", ".join(repr(o) for o in keys[:5])
                if len(keys) > 5:
                    options_str += f", ... ({len(keys) - 5} more)"
                raise ValueError(
                    f"Parameter '{full}': '{key_or_val}' not in allowed options. Available: [{options_str}]"
                )
            if isinstance(options, Mapping) and key_or_val in options:
                self.collector[full] = key_or_val
                return options[key_or_val]
            self.collector[full] = key_or_val
            return key_or_val

        if isinstance(options, Mapping):
            key = self.trial.suggest_categorical(full, keys)
            self.collector[full] = key
            return options[key]
        else:
            choice = self.trial.suggest_categorical(full, keys)
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
        validate_identifier_name(name, kind="nest name")
        overrides = dict(self.overrides)
        if values:
            flat = normalize_values(values)
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
