"""Test select parameter functionality."""

from typing import Any, Dict

import pytest

from hypster import HP, instantiate


def test_select_parameter() -> None:
    """Test hp.select with options."""

    def config(hp: HP) -> str:
        return hp.select(["a", "b", "c"], name="choice", default="a")

    result = instantiate(config)
    assert result == "a"

    result = instantiate(config, values={"choice": "b"})
    assert result == "b"

    # options_only enforcement - default is False
    def strict_config(hp: HP) -> str:
        return hp.select(["a", "b"], name="choice", default="a", options_only=True)

    with pytest.raises(ValueError, match="not in allowed options"):
        instantiate(strict_config, values={"choice": "c"})

    # options_only=False allows any value
    def flexible_config(hp: HP) -> str:
        return hp.select(["a", "b"], name="choice", default="a", options_only=False)

    result = instantiate(flexible_config, values={"choice": "custom"})
    assert result == "custom"


def test_select_parameter_with_dict() -> None:
    """Test hp.select with dictionary options."""

    def config(hp: HP) -> str:
        return hp.select({"fast": "gpt-4o-mini", "smart": "gpt-4"}, name="model", default="fast")

    # Default value (should return the mapped value)
    result = instantiate(config)
    assert result == "gpt-4o-mini"

    # Override with key
    result = instantiate(config, values={"model": "smart"})
    assert result == "gpt-4"

    # Test options_only=True with dict
    def strict_dict_config(hp: HP) -> str:
        return hp.select({"small": "gpt-3.5", "large": "gpt-4"}, name="model", default="small", options_only=True)

    result = instantiate(strict_dict_config)
    assert result == "gpt-3.5"

    result = instantiate(strict_dict_config, values={"model": "large"})
    assert result == "gpt-4"

    # Should reject keys not in dictionary when options_only=True
    with pytest.raises(ValueError, match="not in allowed options"):
        instantiate(strict_dict_config, values={"model": "invalid"})

    # Should also reject values when options_only=True
    with pytest.raises(ValueError, match="not in allowed options"):
        instantiate(strict_dict_config, values={"model": "gpt-4"})  # This is a value, not a key

    # Test options_only=False with dict (should allow any value)
    def flexible_dict_config(hp: HP) -> str:
        return hp.select(
            {"preset1": "value1", "preset2": "value2"}, name="choice", default="preset1", options_only=False
        )

    result = instantiate(flexible_dict_config)
    assert result == "value1"

    result = instantiate(flexible_dict_config, values={"choice": "preset2"})
    assert result == "value2"

    result = instantiate(flexible_dict_config, values={"choice": "custom"})
    assert result == "custom"


def test_select_dict_complex_types() -> None:
    """Test hp.select with dictionary containing complex values."""

    def config(hp: HP) -> Dict[str, Any]:
        model_config = hp.select(
            {"small": {"name": "gpt-3.5-turbo", "max_tokens": 4096}, "large": {"name": "gpt-4", "max_tokens": 8192}},
            name="model",
            default="small",
        )
        return model_config

    # Default
    result = instantiate(config)
    assert result == {"name": "gpt-3.5-turbo", "max_tokens": 4096}

    # Override
    result = instantiate(config, values={"model": "large"})
    assert result == {"name": "gpt-4", "max_tokens": 8192}


def test_select_dict_none_and_tuples() -> None:
    """Test hp.select with dictionary containing None values and tuples."""

    def config(hp: HP) -> Dict[str, Any]:
        # Test None values
        tokenizer = hp.select({"none": None, "basic": "basic_tokenizer"}, name="tokenizer", default="none")

        # Test tuple values (e.g., n-gram ranges)
        ngram_range = hp.select(
            {"unigram": (1, 1), "bigram": (1, 2), "trigram": (1, 3)}, name="ngram_range", default="bigram"
        )

        return {"tokenizer": tokenizer, "ngram_range": ngram_range}

    # Test defaults
    result = instantiate(config)
    assert result["tokenizer"] is None
    assert result["ngram_range"] == (1, 2)

    # Test overrides
    result = instantiate(config, values={"tokenizer": "basic", "ngram_range": "trigram"})
    assert result["tokenizer"] == "basic_tokenizer"
    assert result["ngram_range"] == (1, 3)


def test_select_none_choice_requires_allow_none() -> None:
    def nullable_config(hp: HP) -> object:
        return hp.select([None, "hello"], name="model", default=None, allow_none=True)

    assert instantiate(nullable_config) is None
    assert instantiate(nullable_config, values={"model": "hello"}) == "hello"
    assert instantiate(nullable_config, values={"model": None}) is None

    def missing_allow_none(hp: HP) -> object:
        return hp.select([None, "hello"], name="model", default=None)

    with pytest.raises(ValueError, match="allow_none=True"):
        instantiate(missing_allow_none)


def test_select_with_no_options_can_default_to_none_when_nullable() -> None:
    def config(hp: HP) -> object:
        return hp.select([], name="choice", allow_none=True)

    assert instantiate(config) is None


def test_select_rejects_complex_list_options_with_dict_guidance() -> None:
    def config(hp: HP) -> object:
        return hp.select([{"layers": 2}, {"layers": 4}], name="model")

    with pytest.raises(ValueError) as exc_info:
        instantiate(config)

    error = str(exc_info.value)
    assert "Select choices must be logging-safe" in error
    assert "Use dict-backed select" in error


def test_select_choice_validation_preserves_bool_int_identity() -> None:
    from hypster.utils import validate_select_choice

    assert type(validate_select_choice(True, param_path="choice")) is bool
    assert type(validate_select_choice(1, param_path="choice")) is int
    assert type(validate_select_choice(1.0, param_path="choice")) is float


def test_select_dict_keys_are_logged_even_when_values_are_complex() -> None:
    from hypster import instantiate_with_params

    def config(hp: HP) -> Dict[str, Any]:
        return hp.select(
            {
                "small": {"layers": 2, "units": [64, 32]},
                "large": {"layers": 4, "units": [256, 128]},
            },
            name="model",
            default="small",
        )

    output = instantiate_with_params(config, values={"model": "large"})

    assert output.value == {"layers": 4, "units": [256, 128]}
    assert output.params == {"model": "large"}
