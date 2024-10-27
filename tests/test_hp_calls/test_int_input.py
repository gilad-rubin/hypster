import pytest

from hypster import HP, config


# Int Input Tests
def test_int_input_with_default():
    @config
    def config_func(hp: HP):
        value = hp.int_input(default=42, name="param")

    result = config_func()
    assert result["value"] == 42
    assert isinstance(result["value"], int)


def test_int_input_without_default():
    with pytest.raises(TypeError):

        @config
        def config_func(hp: HP):
            value = hp.int_input(name="param")

        config_func()


def test_int_input_invalid_default_float():
    with pytest.raises(TypeError):

        @config
        def config_func(hp: HP):
            value = hp.int_input(default=42.5, name="param")

        config_func()


def test_int_input_invalid_default_string():
    with pytest.raises(TypeError):

        @config
        def config_func(hp: HP):
            value = hp.int_input(default="42", name="param")

        config_func()


def test_int_input_with_override():
    @config
    def config_func(hp: HP):
        value = hp.int_input(default=42, name="param")

    result = config_func(overrides={"param": 100})
    assert result["value"] == 100
    assert isinstance(result["value"], int)


def test_int_input_invalid_override():
    @config
    def config_func(hp: HP):
        value = hp.int_input(default=42, name="param")

    with pytest.raises(TypeError):
        config_func(overrides={"param": 42.5})  # Float not allowed
