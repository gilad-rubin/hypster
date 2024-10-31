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
    with pytest.raises(Exception):

        @config
        def config_func(hp: HP):
            value = hp.int_input(name="param")

        config_func()


def test_int_input_invalid_default_float():
    with pytest.raises(Exception):

        @config
        def config_func(hp: HP):
            value = hp.int_input(default=42.5, name="param")

        config_func()


def test_int_input_invalid_default_string():
    with pytest.raises(Exception):

        @config
        def config_func(hp: HP):
            value = hp.int_input(default="42", name="param")

        config_func()


def test_int_input_with_values():
    @config
    def config_func(hp: HP):
        value = hp.int_input(default=42, name="param")

    result = config_func(values={"param": 100})
    assert result["value"] == 100
    assert isinstance(result["value"], int)


def test_int_input_invalid_values():
    @config
    def config_func(hp: HP):
        value = hp.int_input(default=42, name="param")

    with pytest.raises(Exception):
        config_func(values={"param": 42.5})  # Float not allowed


def test_multi_int_with_default():
    @config
    def config_func(hp: HP):
        values = hp.multi_int(default=[1, 2, 3], name="param")

    result = config_func()
    assert result["values"] == [1, 2, 3]
    assert all(isinstance(x, int) for x in result["values"])


def test_multi_int_without_default():
    @config
    def config_func(hp: HP):
        values = hp.multi_int(name="param")

    results = config_func()
    assert results["values"] == []


def test_multi_int_invalid_default():
    with pytest.raises(Exception):

        @config
        def config_func(hp: HP):
            values = hp.multi_int(default=[1, 2.5], name="param")  # Not all integers

        config_func()


def test_multi_int_with_values():
    @config
    def config_func(hp: HP):
        values = hp.multi_int(default=[1, 2], name="param")

    result = config_func(values={"param": [3, 4]})
    assert result["values"] == [3, 4]


def test_multi_int_invalid_values():
    @config
    def config_func(hp: HP):
        values = hp.multi_int(default=[1, 2], name="param")

    with pytest.raises(Exception):
        config_func(values={"param": [3, 4.5]})  # Not all integers


def test_multi_int_invalid_values_type():
    @config
    def config_func(hp: HP):
        values = hp.multi_int(default=[1, 2], name="param")

    with pytest.raises(Exception):
        config_func(values={"param": 3})  # Not a list


# Min/Max validation tests for int_input
def test_int_input_with_min():
    @config
    def config_func(hp: HP):
        value = hp.int_input(default=5, min=0, name="param")

    result = config_func()
    assert result["value"] == 5


def test_int_input_with_max():
    @config
    def config_func(hp: HP):
        value = hp.int_input(default=5, max=10, name="param")

    result = config_func()
    assert result["value"] == 5


def test_int_input_with_min_max():
    @config
    def config_func(hp: HP):
        value = hp.int_input(default=5, min=0, max=10, name="param")

    result = config_func()
    assert result["value"] == 5


def test_int_input_default_below_min():
    with pytest.raises(Exception):

        @config
        def config_func(hp: HP):
            value = hp.int_input(default=-1, min=0, name="param")

        config_func()


def test_int_input_default_above_max():
    with pytest.raises(Exception):

        @config
        def config_func(hp: HP):
            value = hp.int_input(default=11, max=10, name="param")

        config_func()


def test_int_input_values_below_min():
    @config
    def config_func(hp: HP):
        value = hp.int_input(default=5, min=0, name="param")

    with pytest.raises(Exception):
        config_func(values={"param": -1})


def test_int_input_values_above_max():
    @config
    def config_func(hp: HP):
        value = hp.int_input(default=5, max=10, name="param")

    with pytest.raises(Exception):
        config_func(values={"param": 11})


# Min/Max validation tests for multi_int
def test_multi_int_with_min():
    @config
    def config_func(hp: HP):
        values = hp.multi_int(default=[5, 6], min=0, name="param")

    result = config_func()
    assert result["values"] == [5, 6]


def test_multi_int_with_max():
    @config
    def config_func(hp: HP):
        values = hp.multi_int(default=[5, 6], max=10, name="param")

    result = config_func()
    assert result["values"] == [5, 6]


def test_multi_int_with_min_max():
    @config
    def config_func(hp: HP):
        values = hp.multi_int(default=[5, 6], min=0, max=10, name="param")

    result = config_func()
    assert result["values"] == [5, 6]


def test_multi_int_default_below_min():
    with pytest.raises(Exception):

        @config
        def config_func(hp: HP):
            values = hp.multi_int(default=[5, -1], min=0, name="param")

        config_func()


def test_multi_int_default_above_max():
    with pytest.raises(Exception):

        @config
        def config_func(hp: HP):
            values = hp.multi_int(default=[5, 11], max=10, name="param")

        config_func()


def test_multi_int_values_below_min():
    @config
    def config_func(hp: HP):
        values = hp.multi_int(default=[5, 6], min=0, name="param")

    with pytest.raises(Exception):
        config_func(values={"param": [5, -1]})


def test_multi_int_values_above_max():
    @config
    def config_func(hp: HP):
        values = hp.multi_int(default=[5, 6], max=10, name="param")

    with pytest.raises(Exception):
        config_func(values={"param": [5, 11]})
