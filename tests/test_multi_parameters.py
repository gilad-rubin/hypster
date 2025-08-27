"""Test multi-value parameter types."""

from typing import List

import pytest

from hypster import HP, instantiate


def test_multi_int() -> None:
    """Test hp.multi_int for lists of integers."""

    def config(hp: HP) -> List[int]:
        return hp.multi_int([1, 2, 3], name="values", min=0, max=10)

    result = instantiate(config)
    assert result == [1, 2, 3]

    result = instantiate(config, values={"values": [5, 6, 7]})
    assert result == [5, 6, 7]

    # Bounds validation on each element
    with pytest.raises(ValueError, match="exceeds maximum"):
        instantiate(config, values={"values": [5, 15, 7]})


def test_multi_float() -> None:
    """Test hp.multi_float for lists of floats."""

    def config(hp: HP) -> List[float]:
        return hp.multi_float([1.0, 2.5, 3.7], name="values", min=0.0, max=10.0)

    result = instantiate(config)
    assert result == [1.0, 2.5, 3.7]

    result = instantiate(config, values={"values": [5.5, 6.1, 7.9]})
    assert result == [5.5, 6.1, 7.9]

    # Test strict parameter behavior
    def strict_config(hp: HP) -> List[float]:
        return hp.multi_float([1.0, 2.0], name="values", strict=True)

    # Should reject integers when strict=True
    with pytest.raises(ValueError, match="expected float but got int"):
        instantiate(strict_config, values={"values": [1, 2]})

    # Should accept integers when strict=False (default)
    def flexible_config(hp: HP) -> List[float]:
        return hp.multi_float([1.0, 2.0], name="values")

    result = instantiate(flexible_config, values={"values": [1, 2]})
    assert result == [1.0, 2.0]  # Converted to floats


def test_multi_select() -> None:
    """Test hp.multi_select for multiple selections."""

    def config(hp: HP) -> List[str]:
        return hp.multi_select(["a", "b", "c"], name="choices", default=["a"])

    result = instantiate(config)
    assert result == ["a"]

    result = instantiate(config, values={"choices": ["a", "b"]})
    assert result == ["a", "b"]


def test_multi_select_with_dict() -> None:
    """Test hp.multi_select with dictionary options."""

    def config(hp: HP) -> List[str]:
        return hp.multi_select({"opt1": "value1", "opt2": "value2", "opt3": "value3"}, name="choices", default=["opt1"])

    # Default (should return mapped values)
    result = instantiate(config)
    assert result == ["value1"]

    # Override with keys
    result = instantiate(config, values={"choices": ["opt1", "opt2"]})
    assert result == ["value1", "value2"]

    result = instantiate(config, values={"choices": ["opt2", "opt3"]})
    assert result == ["value2", "value3"]

    # Test options_only=True with dict
    def strict_multi_dict_config(hp: HP) -> List[str]:
        return hp.multi_select(
            {"a": "alpha", "b": "beta", "c": "gamma"}, name="items", default=["a"], options_only=True
        )

    result = instantiate(strict_multi_dict_config)
    assert result == ["alpha"]

    result = instantiate(strict_multi_dict_config, values={"items": ["a", "b"]})
    assert result == ["alpha", "beta"]

    # Should reject invalid keys when options_only=True
    with pytest.raises(ValueError, match="not in allowed options"):
        instantiate(strict_multi_dict_config, values={"items": ["a", "invalid"]})

    # Should reject values (not keys) when options_only=True
    with pytest.raises(ValueError, match="not in allowed options"):
        instantiate(strict_multi_dict_config, values={"items": ["alpha"]})  # This is a value, not a key

    # Test options_only=False with dict
    def flexible_multi_dict_config(hp: HP) -> List[str]:
        return hp.multi_select({"x": "ex", "y": "why"}, name="items", default=["x"], options_only=False)

    result = instantiate(flexible_multi_dict_config)
    assert result == ["ex"]

    result = instantiate(flexible_multi_dict_config, values={"items": ["x", "y"]})
    assert result == ["ex", "why"]

    result = instantiate(flexible_multi_dict_config, values={"items": ["custom", "values"]})
    assert result == ["custom", "values"]


def test_multi_select_dict_mixed_usage() -> None:
    """Test hp.multi_select with dict allowing both keys and custom values."""

    def config(hp: HP) -> List[str]:
        return hp.multi_select(
            {"preset1": "config1", "preset2": "config2"}, name="configs", default=["preset1"], options_only=False
        )

    # Mix of preset keys and custom values
    result = instantiate(config, values={"configs": ["preset1", "custom", "preset2"]})
    assert result == ["config1", "custom", "config2"]
