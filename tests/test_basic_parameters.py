"""Test basic parameter types and their validation."""

from typing import Dict

import pytest

from hypster import HP, instantiate


def test_int_parameter() -> None:
    """Test hp.int with explicit naming."""

    def config(hp: HP) -> Dict[str, int]:
        batch_size = hp.int(32, name="batch_size", min=1, max=128)
        return {"batch_size": batch_size}

    # Default value
    result = instantiate(config)
    assert result == {"batch_size": 32}

    # Override
    result = instantiate(config, values={"batch_size": 64})
    assert result == {"batch_size": 64}

    # Bounds validation
    with pytest.raises(ValueError, match="exceeds maximum"):
        instantiate(config, values={"batch_size": 256})


def test_float_parameter() -> None:
    """Test hp.float with type enforcement."""

    def config(hp: HP) -> float:
        return hp.float(0.5, name="lr", min=0.0, max=1.0)

    # Default
    result = instantiate(config)
    assert result == 0.5

    # Override with float
    result = instantiate(config, values={"lr": 0.7})
    assert result == 0.7

    # Accept integer when strict=False (default)
    result = instantiate(config, values={"lr": 1})
    assert result == 1.0  # Converted to float

    # Test strict=True behavior
    def strict_config(hp: HP) -> float:
        return hp.float(0.5, name="lr", min=0.0, max=1.0, strict=True)

    # Should reject integer when strict=True
    with pytest.raises(ValueError, match="expected float but got int"):
        instantiate(strict_config, values={"lr": 1})


def test_int_parameter_with_strict() -> None:
    """Test hp.int with strict parameter."""

    def config(hp: HP) -> int:
        return hp.int(32, name="batch_size", min=1, max=128)

    # Accept float that can be converted when strict=False (default)
    result = instantiate(config, values={"batch_size": 64.0})
    assert result == 64  # Converted to int

    # Reject float that loses precision
    with pytest.raises(ValueError, match="would lose precision"):
        instantiate(config, values={"batch_size": 64.5})

    # Test strict=True behavior
    def strict_config(hp: HP) -> int:
        return hp.int(32, name="batch_size", min=1, max=128, strict=True)

    # Should reject float when strict=True
    with pytest.raises(ValueError, match="expected int but got float"):
        instantiate(strict_config, values={"batch_size": 64.0})

    # Should also reject float with precision loss when strict=True
    with pytest.raises(ValueError, match="expected int but got float"):
        instantiate(strict_config, values={"batch_size": 64.5})


def test_text_parameter() -> None:
    """Test hp.text for string values."""

    def config(hp: HP) -> str:
        return hp.text("hello", name="greeting")

    result = instantiate(config)
    assert result == "hello"

    result = instantiate(config, values={"greeting": "hi"})
    assert result == "hi"

    # Type validation
    with pytest.raises(ValueError, match="expected string"):
        instantiate(config, values={"greeting": 123})


def test_bool_parameter() -> None:
    """Test hp.bool for boolean values."""

    def config(hp: HP) -> bool:
        return hp.bool(True, name="flag")

    result = instantiate(config)
    assert result is True

    result = instantiate(config, values={"flag": False})
    assert result is False

    # Type validation
    with pytest.raises(ValueError, match="expected boolean"):
        instantiate(config, values={"flag": "true"})
