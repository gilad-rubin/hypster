from __future__ import annotations

from typing import Any, Dict, Sequence

from hypster import HP, instantiate
from hypster.hpo.optuna import suggest_values

# NOTE: We import types and integration from the new modules under hypster.hpo
from hypster.hpo.types import HpoCategorical, HpoFloat, HpoInt


class FakeTrial:
    """Minimal trial-like object with Optuna's suggest_* API surface.

    Used to avoid depending on optuna for tests; any backend with the same
    methods will work.
    """

    def __init__(self, choices: Dict[str, Any] | None = None):
        self.choices = choices or {}
        self.calls: list[dict] = []

    def suggest_categorical(self, name: str, options: Sequence[Any]) -> Any:
        self.calls.append({"fn": "categorical", "name": name, "options": list(options)})
        return self.choices.get(name, options[0])

    def suggest_int(self, name: str, low: int, high: int, step: int | None = None, log: bool = False) -> int:
        self.calls.append({"fn": "int", "name": name, "low": low, "high": high, "step": step, "log": log})
        return int(self.choices.get(name, low))

    def suggest_float(
        self,
        name: str,
        low: float,
        high: float,
        step: float | None = None,
        log: bool = False,
    ) -> float:
        self.calls.append({"fn": "float", "name": name, "low": low, "high": high, "step": step, "log": log})
        return float(self.choices.get(name, low))


class _RF:
    def __init__(self, n_estimators: int, max_depth: float):
        self.n_estimators = n_estimators
        self.max_depth = max_depth


class _LR:
    def __init__(self, C: float, solver: str):
        self.C = C
        self.solver = solver


def rf_cfg(hp: HP):
    n_estimators = hp.int(100, name="n_estimators", min=50, max=300, hpo_spec=HpoInt(step=50))
    max_depth = hp.float(10.0, name="max_depth", min=2.0, max=30.0, hpo_spec=HpoFloat(step=0.5))
    return _RF(n_estimators=n_estimators, max_depth=max_depth)


def lr_cfg(hp: HP):
    C = hp.float(1.0, name="C", min=1e-5, max=10.0, hpo_spec=HpoFloat(scale="log"))
    solver = hp.select(["lbfgs", "saga"], name="solver", hpo_spec=HpoCategorical(ordered=False))
    return _LR(C=C, solver=solver)


def model_cfg(hp: HP):
    family = hp.select(["rf", "lr"], name="family", hpo_spec=HpoCategorical(ordered=False))
    if family == "rf":
        return hp.nest(rf_cfg, name="rf")
    else:
        return hp.nest(lr_cfg, name="lr")


def test_optuna_suggest_values_rf_branch_and_instantiate():
    trial = FakeTrial({"family": "rf", "rf.n_estimators": 200, "rf.max_depth": 12.5})
    values = suggest_values(trial, model_cfg)
    assert values["family"] == "rf"
    assert values["rf.n_estimators"] == 200
    assert values["rf.max_depth"] == 12.5

    model = instantiate(model_cfg, values=values)
    assert isinstance(model, _RF)

    # Check that step was forwarded on int and log was False
    int_calls = [c for c in trial.calls if c["fn"] == "int" and c["name"] == "rf.n_estimators"]
    assert int_calls and int_calls[0]["step"] == 50 and int_calls[0]["log"] is False


def test_optuna_suggest_values_lr_branch_and_log_float():
    trial = FakeTrial({"family": "lr", "lr.C": 0.01, "lr.solver": "saga"})
    values = suggest_values(trial, model_cfg)
    assert values["family"] == "lr"
    assert values["lr.C"] == 0.01
    assert values["lr.solver"] == "saga"

    model = instantiate(model_cfg, values=values)
    assert isinstance(model, _LR)

    # Float with log scale true
    float_calls = [c for c in trial.calls if c["fn"] == "float" and c["name"] == "lr.C"]
    assert float_calls and float_calls[0]["log"] is True


def test_nested_overrides_passed_to_child():
    def parent(hp: HP):
        # override child param via nest(values=...)
        return hp.nest(
            lambda hp: hp.int(0, name="depth", min=0, max=10, hpo_spec=HpoInt(step=2)),
            name="tree",
            values={"depth": 6},
        )

    trial = FakeTrial()
    values = suggest_values(trial, parent)
    assert values["tree.depth"] == 6  # took override, no suggestion
    # no int call for tree.depth recorded
    assert not any(c for c in trial.calls if c["fn"] == "int" and c["name"] == "tree.depth")

    out = instantiate(parent, values=values)
    assert out == 6


def test_optuna_suggest_values_forwards_execution_kwargs() -> None:
    def config(hp: HP, branch: str):
        if branch == "rf":
            return hp.nest(rf_cfg, name="rf")
        return hp.nest(lr_cfg, name="lr")

    trial = FakeTrial({"rf.n_estimators": 150, "rf.max_depth": 8.0})

    values = suggest_values(trial, config, branch="rf")

    assert values == {"rf.n_estimators": 150, "rf.max_depth": 8.0}


def test_optuna_proxy_accepts_metadata_on_supported_parameter_calls() -> None:
    def config(hp: HP):
        n = hp.int(
            1,
            name="n",
            min=0,
            max=10,
            hpo_spec=HpoInt(),
            description="Trial-backed integer.",
            metadata={"ui": "slider"},
        )
        x = hp.float(
            0.1,
            name="x",
            min=0.0,
            max=1.0,
            hpo_spec=HpoFloat(),
            description="Trial-backed float.",
            metadata={"ui": "number"},
        )
        choice = hp.select(
            ["a", "b"],
            name="choice",
            hpo_spec=HpoCategorical(),
            description="Trial-backed choice.",
            metadata={"ui": "radio"},
        )
        return {"n": n, "x": x, "choice": choice}

    values = suggest_values(FakeTrial({"n": 3, "x": 0.4, "choice": "b"}), config)

    assert values == {"n": 3, "x": 0.4, "choice": "b"}


def test_optuna_nested_proxy_forwards_execution_kwargs() -> None:
    def child(hp: HP, min_depth: int) -> int:
        return hp.int(min_depth, name="depth", min=min_depth, max=10, hpo_spec=HpoInt())

    def parent(hp: HP) -> int:
        return hp.nest(child, name="tree", min_depth=4)

    values = suggest_values(FakeTrial(), parent)

    assert values == {"tree.depth": 4}


def test_hpo_falls_back_to_defaults_for_untunable_kinds() -> None:
    """bool/text/multi_* params must not crash under HPO; they keep their defaults."""

    def config(hp: HP):
        return {
            "cache": hp.bool(True, name="use_cache"),
            "model": hp.text("gpt-4o", name="model"),
            "layers": hp.multi_int([64, 32], name="layers", min=1, max=128),
            "tags": hp.multi_select(["a", "b"], default=["a"], name="tags"),
            "n": hp.int(3, name="n", min=1, max=8),
        }

    trial = FakeTrial({"n": 5})
    values = suggest_values(trial, config)

    assert values == {"n": 5}  # only tunable kinds are suggested and returned

    out = instantiate(config, values=values)
    assert out == {"cache": True, "model": "gpt-4o", "layers": [64, 32], "tags": ["a"], "n": 5}


def test_suggested_values_go_through_regular_validation() -> None:
    import pytest

    class RogueTrial(FakeTrial):
        def suggest_int(self, name: str, low: int, high: int, step: int | None = None, log: bool = False) -> int:
            return 999

    def config(hp: HP):
        return hp.int(1, name="n", min=0, max=10)

    with pytest.raises(ValueError, match="exceeds maximum bound 10"):
        suggest_values(RogueTrial(), config)
