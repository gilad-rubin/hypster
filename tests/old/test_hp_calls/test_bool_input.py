import pytest

from hypster import HP, config


# Bool Input Tests
def test_bool_with_default():
    @config
    def config_func(hp: HP):
        value = hp.bool(default=True, name="param")

    result = config_func()
    assert result["value"] is True
    assert isinstance(result["value"], bool)


def test_bool_without_default():
    with pytest.raises(Exception):

        @config
        def config_func(hp: HP):
            value = hp.bool(name="param")

        config_func()


def test_bool_invalid_default():
    with pytest.raises(Exception):

        @config
        def config_func(hp: HP):
            value = hp.bool(default="abc", name="param")  # Not a boolean

        config_func()


def test_bool_with_values():
    @config
    def config_func(hp: HP):
        value = hp.bool(default=True, name="param")

    result = config_func(values={"param": False})
    assert result["value"] is False
    assert isinstance(result["value"], bool)


def test_bool_invalid_values():
    @config
    def config_func(hp: HP):
        value = hp.bool(default=True, name="param")

    with pytest.raises(Exception):
        config_func(values={"param": "true"})  # Not a boolean


# Multi Bool Tests
def test_multi_bool_with_default():
    @config
    def config_func(hp: HP):
        values = hp.multi_bool(default=[True, False], name="param")

    result = config_func()
    assert result["values"] == [True, False]
    assert all(isinstance(x, bool) for x in result["values"])


def test_multi_bool_without_default():
    @config
    def config_func(hp: HP):
        values = hp.multi_bool(name="param")

    results = config_func()
    assert results["values"] == []


def test_multi_bool_invalid_default():
    with pytest.raises(Exception):

        @config
        def config_func(hp: HP):
            values = hp.multi_bool(default=[True, "123"], name="param")  # Not all booleans

        config_func()


def test_multi_bool_with_values():
    @config
    def config_func(hp: HP):
        values = hp.multi_bool(default=[True, True], name="param")

    result = config_func(values={"param": [False, False]})
    assert result["values"] == [False, False]


def test_multi_bool_invalid_values():
    @config
    def config_func(hp: HP):
        values = hp.multi_bool(default=[True, False], name="param")

    with pytest.raises(Exception):
        config_func(values={"param": [True, "3alse"]})  # Not all booleans


def test_multi_bool_invalid_values_type():
    @config
    def config_func(hp: HP):
        values = hp.multi_bool(default=[True, False], name="param")

    with pytest.raises(Exception):
        config_func(values={"param": True})  # Not a list


if __name__ == "__main__":
    pytest.main([__file__])
