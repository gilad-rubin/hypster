"""
Tests for hp.number with explicit naming requirements.
"""

import pytest

from hypster import HP, config


def test_number_basic():
    """Test basic number input with explicit names"""

    @config
    def test_config(hp: HP):
        learning_rate = hp.number(0.01, name="learning_rate")
        return {"learning_rate": learning_rate}

    result = test_config()
    assert result["learning_rate"] == 0.01


def test_number_with_bounds():
    """Test number input with min/max bounds"""

    @config
    def test_config(hp: HP):
        learning_rate = hp.number(0.01, name="learning_rate", min=0.001, max=0.1)
        return {"learning_rate": learning_rate}

    result = test_config()
    assert result["learning_rate"] == 0.01

    # Test override within bounds
    result = test_config(values={"learning_rate": 0.05})
    assert result["learning_rate"] == 0.05

    # Test override at boundaries
    result = test_config(values={"learning_rate": 0.001})
    assert result["learning_rate"] == 0.001

    result = test_config(values={"learning_rate": 0.1})
    assert result["learning_rate"] == 0.1


def test_number_bounds_validation():
    """Test that number bounds are validated"""

    @config
    def test_config(hp: HP):
        learning_rate = hp.number(0.01, name="learning_rate", min=0.001, max=0.1)
        return {"learning_rate": learning_rate}

    # Test below minimum
    with pytest.raises(Exception):
        test_config(values={"learning_rate": 0.0005})

    # Test above maximum
    with pytest.raises(Exception):
        test_config(values={"learning_rate": 0.2})


def test_number_without_name():
    """Test number without name works when no override attempted"""

    @config
    def test_config(hp: HP):
        learning_rate = hp.number(0.01)  # No name
        return {"learning_rate": learning_rate}

    # Should work without override
    result = test_config()
    assert result["learning_rate"] == 0.01


def test_number_dotted_names():
    """Test number with dotted names for hierarchy"""

    @config
    def test_config(hp: HP):
        model_lr = hp.number(0.01, name="model.learning_rate")
        optimizer_lr = hp.number(0.001, name="optimizer.learning_rate")

        return {"model": {"learning_rate": model_lr}, "optimizer": {"learning_rate": optimizer_lr}}

    result = test_config(values={"model.learning_rate": 0.05, "optimizer.learning_rate": 0.002})

    assert result["model"]["learning_rate"] == 0.05
    assert result["optimizer"]["learning_rate"] == 0.002


def test_number_float_precision():
    """Test number handles float precision correctly"""

    @config
    def test_config(hp: HP):
        precise_value = hp.number(0.123456789, name="precise_value")
        return {"precise_value": precise_value}

    result = test_config()
    assert result["precise_value"] == 0.123456789

    result = test_config(values={"precise_value": 0.987654321})
    assert result["precise_value"] == 0.987654321


def test_number_int_values():
    """Test number with integer values"""

    @config
    def test_config(hp: HP):
        rate = hp.number(5, name="rate")  # Integer default
        return {"rate": rate}

    result = test_config()
    assert result["rate"] == 5
    assert isinstance(result["rate"], (int, float))

    # Can override with float
    result = test_config(values={"rate": 5.5})
    assert result["rate"] == 5.5
