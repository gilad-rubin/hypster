"""Test the requirement for explicit naming in v2."""

import pytest

from hypster import HP, instantiate


def test_missing_name_raises_error() -> None:
    """Test that missing name parameter raises error."""

    def config(hp: HP) -> int:
        return hp.int(10)  # Missing name

    with pytest.raises(TypeError, match="missing 1 required keyword-only argument: 'name'"):
        instantiate(config)


def test_name_parameter_is_keyword_only() -> None:
    """Test that name must be passed as keyword argument."""

    def config(hp: HP) -> int:
        return hp.int(10, "my_param")  # Positional name not allowed

    with pytest.raises(TypeError):
        instantiate(config)
