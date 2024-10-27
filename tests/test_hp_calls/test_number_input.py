import pytest

from hypster import HP, config


# Number Input Tests
def test_number_input_with_default():
    @config
    def config_func(hp: HP):
        value = hp.number_input(default=0.5, name="param")

    result = config_func()
    assert result["value"] == 0.5


def test_number_input_without_default():
    with pytest.raises(ValueError):

        @config
        def config_func(hp: HP):
            value = hp.number_input(name="param")

        config_func()


def test_number_input_invalid_default():
    with pytest.raises(TypeError):

        @config
        def config_func(hp: HP):
            value = hp.number_input(default="not a number", name="param")

        config_func()


def test_number_input_with_override():
    @config
    def config_func(hp: HP):
        value = hp.number_input(default=0.5, name="param")

    result = config_func(overrides={"param": 1.5})
    assert result["value"] == 1.5


def test_number_input_invalid_override():
    @config
    def config_func(hp: HP):
        value = hp.number_input(default=0.5, name="param")

    with pytest.raises(TypeError):
        config_func(overrides={"param": "not a number"})


# Multi Number Tests
def test_multi_number_with_default():
    @config
    def config_func(hp: HP):
        values = hp.multi_number(default=[1.0, 2.0], name="param")

    result = config_func()
    assert result["values"] == [1.0, 2.0]


def test_multi_number_without_default():
    with pytest.raises(ValueError):

        @config
        def config_func(hp: HP):
            values = hp.multi_number(name="param")

        config_func()


def test_multi_number_invalid_default():
    with pytest.raises(TypeError):

        @config
        def config_func(hp: HP):
            values = hp.multi_number(default=["a", "b"], name="param")  # Not numbers

        config_func()


def test_multi_number_with_override():
    @config
    def config_func(hp: HP):
        values = hp.multi_number(default=[1.0, 2.0], name="param")

    result = config_func(overrides={"param": [3.0, 4.0]})
    assert result["values"] == [3.0, 4.0]


def test_multi_number_invalid_override():
    @config
    def config_func(hp: HP):
        values = hp.multi_number(default=[1.0, 2.0], name="param")

    with pytest.raises(TypeError):
        config_func(overrides={"param": ["a", "b"]})  # Not numbers
