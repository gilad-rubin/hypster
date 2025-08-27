import pytest

from hypster import HP, config


# Text Input Tests
def test_text_with_default():
    @config
    def config_func(hp: HP):
        value = hp.text(default="hello", name="param")

    result = config_func()
    assert result["value"] == "hello"


def test_text_without_default():
    with pytest.raises(Exception):

        @config
        def config_func(hp: HP):
            value = hp.text(name="param")

        config_func()


def test_text_invalid_default():
    with pytest.raises(Exception):

        @config
        def config_func(hp: HP):
            value = hp.text(default=123, name="param")  # Not a string

        config_func()


def test_text_with_values():
    @config
    def config_func(hp: HP):
        value = hp.text(default="hello", name="param")

    result = config_func(values={"param": "world"})
    assert result["value"] == "world"


def test_text_invalid_values():
    @config
    def config_func(hp: HP):
        value = hp.text(default="hello", name="param")

    with pytest.raises(Exception):
        config_func(values={"param": 123})  # Not a string


# Multi Text Tests
def test_multi_text_with_default():
    @config
    def config_func(hp: HP):
        values = hp.multi_text(default=["a", "b"], name="param")

    result = config_func()
    assert result["values"] == ["a", "b"]


def test_multi_text_without_default():
    @config
    def config_func(hp: HP):
        values = hp.multi_text(name="param")

    results = config_func()
    assert results["values"] == []


def test_multi_text_invalid_default():
    with pytest.raises(Exception):

        @config
        def config_func(hp: HP):
            values = hp.multi_text(default=[1, 2], name="param")  # Not strings

        config_func()


def test_multi_text_with_values():
    @config
    def config_func(hp: HP):
        values = hp.multi_text(default=["a", "b"], name="param")

    result = config_func(values={"param": ["c", "d"]})
    assert result["values"] == ["c", "d"]


def test_multi_text_invalid_values():
    @config
    def config_func(hp: HP):
        values = hp.multi_text(default=["a", "b"], name="param")

    with pytest.raises(Exception):
        config_func(values={"param": [1, 2]})  # Not strings


def test_multi_text_invalid_values_type():
    @config
    def config_func(hp: HP):
        values = hp.multi_text(default=["a", "b"], name="param")

    with pytest.raises(Exception):
        config_func(values={"param": "c"})  # Not a list


if __name__ == "__main__":
    pytest.main([__file__])
