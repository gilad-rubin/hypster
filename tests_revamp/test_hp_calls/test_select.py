"""
Tests for hp.select with explicit naming requirements.
"""

import pytest

from hypster import HP, config


def test_select_basic():
    """Test basic select functionality with explicit names"""

    @config
    def test_config(hp: HP):
        choice = hp.select(["option1", "option2", "option3"], name="choice")
        return {"choice": choice}

    result = test_config()
    assert result["choice"] == "option1"  # First option as default


def test_select_with_default():
    """Test select with explicit default"""

    @config
    def test_config(hp: HP):
        choice = hp.select(["option1", "option2", "option3"], name="choice", default="option2")
        return {"choice": choice}

    result = test_config()
    assert result["choice"] == "option2"


def test_select_with_values_override():
    """Test select with values override using explicit name"""

    @config
    def test_config(hp: HP):
        choice = hp.select(["option1", "option2", "option3"], name="choice")
        return {"choice": choice}

    result = test_config(values={"choice": "option3"})
    assert result["choice"] == "option3"


def test_select_dict_options():
    """Test select with dictionary options"""

    @config
    def test_config(hp: HP):
        model = hp.select(
            {"rf": "RandomForest", "svm": "SupportVectorMachine", "lr": "LogisticRegression"}, name="model"
        )
        return {"model": model}

    result = test_config()
    assert result["model"] == "RandomForest"  # First value

    result = test_config(values={"model": "svm"})
    assert result["model"] == "SupportVectorMachine"


def test_select_options_only():
    """Test select with options_only constraint"""

    @config
    def test_config(hp: HP):
        choice = hp.select(["option1", "option2"], name="choice", options_only=True)
        return {"choice": choice}

    # Valid option should work
    result = test_config(values={"choice": "option2"})
    assert result["choice"] == "option2"

    # Invalid option should raise error
    with pytest.raises(Exception):
        test_config(values={"choice": "invalid_option"})


def test_select_dotted_names():
    """Test select with dotted names for hierarchy"""

    @config
    def test_config(hp: HP):
        model_type = hp.select(["rf", "svm"], name="model.type")
        optimizer_type = hp.select(["adam", "sgd"], name="optimizer.type")

        return {"model": {"type": model_type}, "optimizer": {"type": optimizer_type}}

    result = test_config(values={"model.type": "svm", "optimizer.type": "sgd"})

    assert result["model"]["type"] == "svm"
    assert result["optimizer"]["type"] == "sgd"


def test_select_without_name_no_override():
    """Test that select without name works when no override is attempted"""

    @config
    def test_config(hp: HP):
        choice = hp.select(["option1", "option2"])  # No name
        return {"choice": choice}

    # Should work fine without override
    result = test_config()
    assert result["choice"] == "option1"


def test_select_without_name_with_override_error():
    """Test that select without name raises error when override is attempted"""

    @config
    def test_config(hp: HP):
        choice = hp.select(["option1", "option2"])  # No name
        return {"choice": choice}

    # Should raise error when trying to override unnamed parameter
    with pytest.raises(Exception):
        test_config(values={"choice": "option2"})


def test_select_empty_options():
    """Test select with empty options list"""

    @config
    def test_config(hp: HP):
        choice = hp.select([], name="choice")
        return {"choice": choice}

    with pytest.raises(Exception):
        test_config()


def test_select_single_option():
    """Test select with single option"""

    @config
    def test_config(hp: HP):
        choice = hp.select(["only_option"], name="choice")
        return {"choice": choice}

    result = test_config()
    assert result["choice"] == "only_option"

    # Should work with override to same value
    result = test_config(values={"choice": "only_option"})
    assert result["choice"] == "only_option"


def test_select_complex_values():
    """Test select with complex value types"""

    @config
    def test_config(hp: HP):
        config_choice = hp.select(
            {"simple": {"layers": 2, "lr": 0.01}, "complex": {"layers": 5, "lr": 0.001}}, name="config"
        )
        return {"config": config_choice}

    result = test_config()
    assert result["config"]["layers"] == 2
    assert result["config"]["lr"] == 0.01

    result = test_config(values={"config": "complex"})
    assert result["config"]["layers"] == 5
    assert result["config"]["lr"] == 0.001
