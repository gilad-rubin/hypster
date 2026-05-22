"""Test error handling strategies for unknown parameters."""

from typing import Any, Dict

import pytest

from hypster import HP, instantiate


def test_on_unknown_raise_default() -> None:
    """Test that on_unknown='raise' is the default behavior."""

    def config(hp: HP) -> Dict[str, float]:
        lr = hp.float(0.1, name="learning_rate")
        return {"lr": lr}

    with pytest.raises(ValueError) as exc_info:
        instantiate(config, values={"unknown_param": 0.05})

    assert "'unknown_param': Unknown parameter" in str(exc_info.value)
    assert "explore(config, values=...)" in str(exc_info.value)

    # Explicit warn should behave the same
    with pytest.warns(UserWarning, match="'unknown_param': Unknown parameter"):
        result = instantiate(config, values={"unknown_param": 0.05}, on_unknown="warn")

    assert result == {"lr": 0.1}


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


def test_on_unknown_invalid_policy_has_guidance() -> None:
    calls = []

    def config(hp: HP) -> Dict[str, float]:
        calls.append("executed")
        return {"lr": hp.float(0.1, name="learning_rate")}

    with pytest.raises(ValueError, match="on_unknown must be one of"):
        instantiate(config, values={"learning_rate": 0.2}, on_unknown="silent")  # type: ignore[arg-type]

    assert calls == []


def test_values_must_be_a_mapping() -> None:
    def config(hp: HP) -> Dict[str, float]:
        return {"lr": hp.float(0.1, name="learning_rate")}

    with pytest.raises(ValueError, match="expected a dictionary"):
        instantiate(config, values=["learning_rate"])  # type: ignore[arg-type]


def test_on_unknown_with_suggestions() -> None:
    """Test that error messages include helpful suggestions."""

    def config(hp: HP) -> Dict[str, float]:
        learning_rate = hp.float(0.1, name="learning_rate")
        temperature = hp.float(0.7, name="temperature")
        return {"lr": learning_rate, "temp": temperature}

    # Typo should suggest similar parameter in warn mode
    with pytest.warns(UserWarning) as w:
        result = instantiate(config, values={"lerning_rate": 0.05}, on_unknown="warn")

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

    with pytest.raises(ValueError) as exc_info:
        instantiate(
            config,
            values={"n_trees": 200},  # unreachable when model_type='logistic'
        )

    assert "n_trees" in str(exc_info.value)
    assert "explore(config, values=...)" in str(exc_info.value)


def test_nested_values_unknown_keys_participate_in_on_unknown() -> None:
    def child(hp: HP) -> Dict[str, float]:
        return {"lr": hp.float(0.1, name="learning_rate")}

    def config(hp: HP) -> Dict[str, Dict[str, float]]:
        return {"model": hp.nest(child, name="model")}

    with pytest.raises(ValueError) as exc_info:
        instantiate(config, values={"model": {"lerning_rate": 0.2}})

    error = str(exc_info.value)
    assert "model.lerning_rate" in error
    assert "model.learning_rate" in error
    assert "Nested dict values are interpreted as parameter paths" in error
