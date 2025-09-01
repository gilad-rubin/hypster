"""Test how None is currently handled in select."""

from hypster import HP, instantiate


def test_none_in_dict():
    """Test dictionary with None values."""

    def config(hp: HP):
        tokenizers = {"none": None, "basic": "basic_tokenizer"}
        tokenizer = hp.select(tokenizers, default="none", name="tokenizer")
        print(f"Selected tokenizer: {tokenizer}")
        print(f"Type: {type(tokenizer)}")
        return {"tokenizer": tokenizer}

    # Test default (None)
    result = instantiate(config)
    print(f"Result with None: {result}")
    assert result["tokenizer"] is None

    # Test override with string
    result = instantiate(config, values={"tokenizer": "basic"})
    print(f"Result with basic: {result}")
    assert result["tokenizer"] == "basic_tokenizer"


def test_none_in_list():
    """Test list with None values."""

    def config(hp: HP):
        value = hp.select([None, "a", "b"], default=None, name="value")
        print(f"Selected value: {value}")
        print(f"Type: {type(value)}")
        return {"value": value}

    result = instantiate(config)
    print(f"Result: {result}")
    assert result["value"] is None


if __name__ == "__main__":
    print("Testing None in dictionary options:")
    test_none_in_dict()
    print("\nTesting None in list options:")
    test_none_in_list()
