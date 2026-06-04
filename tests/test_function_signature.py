"""Test function signature validation."""

import pytest

from hypster import HP, instantiate


def test_valid_signature() -> None:
    """Test that valid signatures are accepted."""

    def config(hp: HP) -> int:
        return hp.int(10, name="value")

    result = instantiate(config)
    assert result == 10


def test_invalid_first_param_name() -> None:
    """Test error when first param is not named 'hp'."""

    def bad_config(hypster: HP) -> int:  # Wrong parameter name
        return hypster.int(10, name="value")

    with pytest.raises(ValueError, match="first param.*named 'hp'"):
        instantiate(bad_config)


def test_missing_hp_parameter() -> None:
    """Test error when hp parameter is missing."""

    def bad_config() -> int:
        return 10

    with pytest.raises(ValueError, match="must have.*hp: HP"):
        instantiate(bad_config)


def test_extra_args_allowed() -> None:
    """Test that extra arguments are allowed and forwarded."""

    def config(hp: HP, multiplier: int = 1) -> int:
        base = hp.int(10, name="base")
        return base * multiplier

    result = instantiate(config, multiplier=3)
    assert result == 30


def test_removed_execution_argument_containers_are_rejected() -> None:
    calls = []

    def config(hp: HP, **execution_kwargs: object) -> int:
        calls.append(execution_kwargs)
        return hp.int(10, name="base")

    with pytest.raises(TypeError, match="no longer accepts args= or kwargs="):
        instantiate(config, args=(3,))

    with pytest.raises(TypeError, match="no longer accepts args= or kwargs="):
        instantiate(config, kwargs={"multiplier": 3})

    assert calls == []


def test_keyword_only_hp_is_rejected_during_signature_validation() -> None:
    calls = []

    def bad_config(*, hp: HP) -> int:
        calls.append("executed")
        return hp.int(10, name="value")

    with pytest.raises(ValueError, match="positional"):
        instantiate(bad_config)

    assert calls == []


def test_callable_object_signature_errors_use_class_name() -> None:
    class BadConfig:
        def __call__(self) -> int:
            return 10

    with pytest.raises(ValueError, match="BadConfig"):
        instantiate(BadConfig())
