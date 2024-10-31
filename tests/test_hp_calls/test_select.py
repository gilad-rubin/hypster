import pytest

from hypster import HP, config


# Options validation
def test_select_valid_options_list():
    @config
    def config_func(hp: HP):
        value = hp.select(["a", "b", "c"], default="a", name="param")

    result = config_func()
    assert result["value"] == "a"


def test_select_valid_options_dict():
    @config
    def config_func(hp: HP):
        value = hp.select({"k1": "v1", "k2": "v2"}, default="k1", name="param")

    result = config_func()
    assert result["value"] == "v1"


def test_select_invalid_options_empty():
    with pytest.raises(Exception):

        @config
        def config_func(hp: HP):
            value = hp.select([], default="a", name="param")

        config_func()


def test_select_invalid_options_wrong_type():
    with pytest.raises(Exception):

        @config
        def config_func(hp: HP):
            value = hp.select(123, default="a", name="param")  # Not a list/dict

        config_func()


# Default validation
def test_select_valid_default_from_options_list():
    @config
    def config_func(hp: HP):
        value = hp.select(["a", "b", "c"], default="b", name="param")

    result = config_func()
    assert result["value"] == "b"


def test_select_valid_default_from_options_dict():
    @config
    def config_func(hp: HP):
        value = hp.select({"k1": "v1", "k2": "v2"}, default="k2", name="param")

    result = config_func()
    assert result["value"] == "v2"


def test_select_invalid_default_not_in_options_list():
    with pytest.raises(Exception):

        @config
        def config_func(hp: HP):
            value = hp.select(["a", "b", "c"], default="d", name="param")

        config_func()


def test_select_invalid_default_not_in_options_dict():
    with pytest.raises(Exception):

        @config
        def config_func(hp: HP):
            value = hp.select({"k1": "v1", "k2": "v2"}, default="k3", name="param")

        config_func()


def test_select_with_list():
    @config
    def config_func(hp: HP):
        value = hp.select(["a", "b", "c"], default="a", name="param")

    result = config_func(values={"param": "b"})
    assert result["value"] == "b"


def test_select_with_dict():
    @config
    def config_func(hp: HP):
        value = hp.select({"k1": "v1", "k2": "v2"}, default="k1", name="param")

    result = config_func(values={"param": "k2"})
    assert result["value"] == "v2"


def test_select_without_values():
    @config
    def config_func(hp: HP):
        value = hp.select(["a", "b", "c"], default="a", name="param")

    result = config_func()
    assert result["value"] == "a"


def test_invalid_dict_key_types():
    with pytest.raises(Exception):

        @config
        def invalid_dict_keys(hp: HP):
            var = hp.select({1: "a", 2.0: "b", True: "c", "str": "d", complex(1, 2): "e"}, default="d", name="param")

        invalid_dict_keys()


def test_invalid_list_value_types():
    with pytest.raises(Exception):

        @config
        def invalid_list_values(hp: HP):
            var = hp.select([1, 2.0, True, "str", complex(1, 2)], default="str", name="param")

        invalid_list_values()


def test_select_missing_default():
    with pytest.raises(Exception):

        @config
        def missing_default(hp: HP):
            hp.select(["a", "b"], name="param")

        missing_default()


def test_select_with_options_only():
    @config
    def config_func(hp: HP):
        value = hp.select(["a", "b", "c"], default="a", name="param", options_only=True)

    # Should work with values
    result = config_func(values={"param": "b"})
    assert result["value"] == "b"

    # Should fail with values
    with pytest.raises(Exception):
        config_func(values={"param": "d"})


def test_select_rejects_list_selection():
    @config
    def config_func(hp: HP):
        value = hp.select(["a", "b", "c"], default="a", name="param")

    with pytest.raises(Exception):
        config_func(values={"param": ["b"]})
