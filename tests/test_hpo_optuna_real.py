"""Tests against REAL Optuna trials.

The rest of the HPO suite uses FakeTrial, which tolerates argument
combinations real Optuna rejects (e.g. step=None). These tests exercise the
adapter through optuna.Study.ask() so such gaps can't hide.
"""

from typing import Any

import pytest

optuna = pytest.importorskip("optuna")

from hypster import HP, instantiate  # noqa: E402
from hypster.hpo.optuna import suggest_values  # noqa: E402
from hypster.hpo.types import HpoFloat, HpoInt  # noqa: E402

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _trial() -> Any:
    return optuna.create_study().ask()


def test_plain_int_without_hpo_spec_suggests_with_real_optuna() -> None:
    def config(hp: HP) -> int:
        return hp.int(5, name="n", min=1, max=10)

    values = suggest_values(_trial(), config)

    assert 1 <= values["n"] <= 10
    assert instantiate(config, values=values) == values["n"]


def test_log_scale_int_without_step_suggests_with_real_optuna() -> None:
    def config(hp: HP) -> int:
        return hp.int(12, name="depth", min=2, max=64, hpo_spec=HpoInt(scale="log"))

    values = suggest_values(_trial(), config)

    assert 2 <= values["depth"] <= 64


def test_log_scale_int_with_step_raises_friendly_error() -> None:
    def config(hp: HP) -> int:
        return hp.int(12, name="depth", min=2, max=64, hpo_spec=HpoInt(scale="log", step=4))

    with pytest.raises(ValueError, match="cannot be combined with scale='log'"):
        suggest_values(_trial(), config)


def test_log_scale_float_suggests_with_real_optuna() -> None:
    def config(hp: HP) -> float:
        return hp.float(0.1, name="lr", min=1e-4, max=1.0, hpo_spec=HpoFloat(scale="log"))

    values = suggest_values(_trial(), config)

    assert 1e-4 <= values["lr"] <= 1.0


def test_log_scale_float_with_step_raises_friendly_error() -> None:
    def config(hp: HP) -> float:
        return hp.float(0.1, name="lr", min=1e-4, max=1.0, hpo_spec=HpoFloat(scale="log", step=0.1))

    with pytest.raises(ValueError, match="cannot be combined with a log scale"):
        suggest_values(_trial(), config)


def test_full_study_optimizes_branching_config_end_to_end() -> None:
    def config(hp: HP) -> dict:
        family = hp.select(["linear", "forest"], name="family")
        if family == "linear":
            return {"family": family, "c": hp.float(1.0, name="c", min=0.01, max=10.0, hpo_spec=HpoFloat(scale="log"))}
        return {"family": family, "trees": hp.int(100, name="trees", min=10, max=500, hpo_spec=HpoInt(step=10))}

    def objective(trial: Any) -> float:
        values = suggest_values(trial, config)
        model = instantiate(config, values=values)
        return model.get("c", 0.0) + model.get("trees", 0)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=8)

    assert len(study.trials) == 8
    replayed = instantiate(config, values=suggest_values(study.ask(), config))
    assert replayed["family"] in {"linear", "forest"}
