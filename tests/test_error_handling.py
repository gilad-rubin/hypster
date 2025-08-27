"""Test error handling strategies for unknown parameters."""

import warnings
from typing import Any, Dict

import pytest

from hypster import HP, instantiate


def test_on_unknown_warn_default() -> None:
    """Test that on_unknown='warn' is the default behavior."""

    def config(hp: HP) -> Dict[str, float]:
        lr = hp.float(0.1, name="learning_rate")
        return {"lr": lr}

    # Default behavior should warn
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = instantiate(config, values={"unknown_param": 0.05})

        # Should complete successfully with defaults
        assert result == {"lr": 0.1}

        # Should have issued a warning
        assert len(w) == 1
        assert "'unknown_param': Unknown parameter" in str(w[0].message)

    # Explicit warn should behave the same
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = instantiate(config, values={"unknown_param": 0.05}, on_unknown="warn")

        assert result == {"lr": 0.1}
        assert len(w) == 1
        assert "'unknown_param': Unknown parameter" in str(w[0].message)


def test_on_unknown_raise() -> None:
    """Test that on_unknown='raise' fails immediately."""

    def config(hp: HP) -> Dict[str, float]:
        lr = hp.float(0.1, name="learning_rate")
        return {"lr": lr}

    # Should fail with raise mode
    with pytest.raises(ValueError, match="Unknown or unreachable"):
        instantiate(config, values={"unknown_param": 0.05}, on_unknown="raise")


def test_on_unknown_ignore() -> None:
    """Test that on_unknown='ignore' silently ignores unknown parameters."""

    def config(hp: HP) -> Dict[str, float]:
        lr = hp.float(0.1, name="learning_rate")
        return {"lr": lr}

    # Should complete successfully without warnings
    result = instantiate(config, values={"unknown_param": 0.05}, on_unknown="ignore")
    assert result == {"lr": 0.1}


def test_on_unknown_with_suggestions() -> None:
    """Test that error messages include helpful suggestions."""

    def config(hp: HP) -> Dict[str, float]:
        learning_rate = hp.float(0.1, name="learning_rate")
        temperature = hp.float(0.7, name="temperature")
        return {"lr": learning_rate, "temp": temperature}

    # Typo should suggest similar parameter (in warn mode by default)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = instantiate(config, values={"lerning_rate": 0.05})

        assert len(w) == 1
        warning_msg = str(w[0].message)
        assert "lerning_rate" in warning_msg
        assert "learning_rate" in warning_msg  # suggestion

    # In raise mode, should get exception with suggestions
    with pytest.raises(ValueError) as exc_info:
        instantiate(
            config,
            values={
                "lerning_rate": 0.05,  # typo
                "temp_value": 0.8,  # typo
                "totally_unknown": 42,  # no suggestion
            },
            on_unknown="raise",
        )

    error_msg = str(exc_info.value)
    assert "lerning_rate" in error_msg
    assert "learning_rate" in error_msg  # suggestion
    assert "temp_value" in error_msg
    assert "temperature" in error_msg  # suggestion
    assert "totally_unknown" in error_msg


def test_on_unknown_conditional_reachability() -> None:
    """Test error handling for conditionally unreachable parameters."""

    def config(hp: HP) -> Dict[str, Any]:
        model_type = hp.select(["logistic", "rf"], name="model_type", default="logistic")

        if model_type == "rf":
            n_trees = hp.int(100, name="n_trees")
            return {"model": "rf", "n_trees": n_trees}
        else:
            penalty = hp.text("l2", name="penalty")
            return {"model": "logistic", "penalty": penalty}

    # Default warn mode should warn about unreachable parameter
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = instantiate(
            config,
            values={"n_trees": 200},  # unreachable when model_type='logistic'
        )

        # Should complete with logistic model
        assert result == {"model": "logistic", "penalty": "l2"}

        # Should warn about unreachable parameter
        assert len(w) == 1
        assert "n_trees" in str(w[0].message)
        assert "unreachable" in str(w[0].message).lower()
