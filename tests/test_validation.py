"""Test parameter validation and error messages."""

from typing import Any, Dict

import pytest

from hypster import HP, instantiate


def test_unknown_parameter_error() -> None:
    """Test error for unknown parameters with suggestions."""

    def config(hp: HP) -> Dict[str, float]:
        lr = hp.float(0.1, name="learning_rate")
        return {"lr": lr}

    # Typo in parameter name
    with pytest.raises(ValueError, match="Did you mean 'learning_rate'"):
        instantiate(config, values={"lerning_rate": 0.05})


def test_unknown_or_unreachable_parameter_error() -> None:
    """Test error for unknown or conditionally unreachable parameters."""

    def config(hp: HP) -> Dict[str, Any]:
        model_type = hp.select(["logistic", "rf"], name="model_type", default="logistic")

        if model_type == "rf":
            n_trees = hp.int(100, name="n_trees")
            return {"model": "rf", "n_trees": n_trees}
        else:
            penalty = hp.text("l2", name="penalty")
            return {"model": "logistic", "penalty": penalty}

    # Try to set rf parameter when logistic is selected
    # This should fail as unknown or unreachable since n_trees wasn't encountered
    with pytest.raises(ValueError, match="Unknown or unreachable.*n_trees"):
        instantiate(config, values={"n_trees": 200})


def test_type_mismatch_with_context() -> None:
    """Test type mismatch errors include full path context."""

    def optimizer(hp: HP) -> Dict[str, float]:
        lr = hp.float(0.1, name="lr")
        return {"lr": lr}

    def config(hp: HP) -> Dict[str, Dict[str, float]]:
        opt = hp.nest(optimizer, name="optimizer")
        return {"optimizer": opt}

    # Integer provided for float in nested context
    with pytest.raises(ValueError, match="Parameter 'optimizer.lr': expected float but got int"):
        instantiate(config, values={"optimizer.lr": 1})


def test_bounds_validation_error() -> None:
    """Test bounds validation with helpful messages."""

    def config(hp: HP) -> int:
        return hp.int(50, name="value", min=0, max=100)

    with pytest.raises(ValueError, match="exceeds maximum bound 100"):
        instantiate(config, values={"value": 150})

    with pytest.raises(ValueError, match="below minimum bound 0"):
        instantiate(config, values={"value": -10})
