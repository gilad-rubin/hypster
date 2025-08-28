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
