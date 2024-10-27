import pytest

from hypster import HP, config


# Options validation
def test_multi_select_valid_options_list():
    @config
    def config_func(hp: HP):
        values = hp.multi_select(["a", "b", "c"], default=["a", "b"], name="param")

    result = config_func()
    assert result["values"] == ["a", "b"]


def test_multi_select_valid_options_dict():
    @config
    def config_func(hp: HP):
        values = hp.multi_select({"k1": "v1", "k2": "v2", "k3": "v3"}, default=["k1", "k2"], name="param")

    result = config_func()
    assert result["values"] == ["v1", "v2"]


def test_multi_select_invalid_options_empty():
    with pytest.raises(ValueError):

        @config
        def config_func(hp: HP):
            values = hp.multi_select([], default=["a"], name="param")

        config_func()


def test_multi_select_invalid_options_wrong_type():
    with pytest.raises(ValueError):

        @config
        def config_func(hp: HP):
            values = hp.multi_select(123, default=["a"], name="param")  # Not a list/dict

        config_func()


# Default validation
def test_multi_select_valid_default_from_options_list():
    @config
    def config_func(hp: HP):
        values = hp.multi_select(["a", "b", "c"], default=["b", "c"], name="param")

    result = config_func()
    assert result["values"] == ["b", "c"]


def test_multi_select_valid_default_from_options_dict():
    @config
    def config_func(hp: HP):
        values = hp.multi_select({"k1": "v1", "k2": "v2", "k3": "v3"}, default=["k2", "k3"], name="param")

    result = config_func()
    assert result["values"] == ["v2", "v3"]


def test_multi_select_invalid_default_not_in_options_list():
    with pytest.raises(ValueError):

        @config
        def config_func(hp: HP):
            values = hp.multi_select(["a", "b", "c"], default=["d", "e"], name="param")

        config_func()


def test_multi_select_invalid_default_not_in_options_dict():
    with pytest.raises(ValueError):

        @config
        def config_func(hp: HP):
            values = hp.multi_select({"k1": "v1", "k2": "v2"}, default=["k3", "k4"], name="param")

        config_func()


def test_multi_select_empty_default_list():
    @config
    def config_func(hp: HP):
        values = hp.multi_select(["a", "b", "c"], default=[], name="param")

    result = config_func()
    assert result["values"] == []


def test_multi_select_empty_default_dict():
    @config
    def config_func(hp: HP):
        values = hp.multi_select({"k1": "v1", "k2": "v2"}, default=[], name="param")

    result = config_func()
    assert result["values"] == []


# Selection and override behavior with list options
def test_multi_select_with_selection_list():
    @config
    def config_func(hp: HP):
        values = hp.multi_select(["a", "b", "c"], default=["a"], name="param")

    result = config_func(selections={"param": ["b", "c"]})
    assert result["values"] == ["b", "c"]


def test_multi_select_with_override_list():
    @config
    def config_func(hp: HP):
        values = hp.multi_select(["a", "b", "c"], default=["a"], name="param")

    result = config_func(overrides={"param": ["c"]})
    assert result["values"] == ["c"]


# Selection and override behavior with dict options
def test_multi_select_with_selection_dict():
    @config
    def config_func(hp: HP):
        values = hp.multi_select({"k1": "v1", "k2": "v2", "k3": "v3"}, default=["k1"], name="param")

    result = config_func(selections={"param": ["k2", "k3"]})
    assert result["values"] == ["v2", "v3"]


def test_multi_select_with_override_dict():
    @config
    def config_func(hp: HP):
        values = hp.multi_select({"k1": "v1", "k2": "v2", "k3": "v3"}, default=["k1"], name="param")

    result = config_func(overrides={"param": ["k2", "k3"]})
    assert result["values"] == ["v2", "v3"]


# Override precedence tests
def test_multi_select_override_precedence_list():
    @config
    def config_func(hp: HP):
        values = hp.multi_select(["a", "b", "c"], default=["a"], name="param")

    # Test with valid options
    result = config_func(selections={"param": ["b"]}, overrides={"param": ["c"]})
    assert result["values"] == ["c"]

    # Test with override value not in options (should still work)
    result = config_func(selections={"param": ["b"]}, overrides={"param": ["d", "e"]})
    assert result["values"] == ["d", "e"]


def test_multi_select_override_precedence_dict():
    @config
    def config_func(hp: HP):
        values = hp.multi_select({"k1": "v1", "k2": "v2", "k3": "v3"}, default=["k1"], name="param")

    # Test with valid keys
    result = config_func(selections={"param": ["k1"]}, overrides={"param": ["k2", "k3"]})
    assert result["values"] == ["v2", "v3"]

    # Test with override keys not in options (should still work)
    result = config_func(selections={"param": ["k1"]}, overrides={"param": ["k4", "k5"]})
    assert result["values"] == ["k4", "k5"]


def test_multi_select_without_selection_or_override():
    @config
    def config_func(hp: HP):
        values = hp.multi_select(["a", "b", "c"], default=["a", "b"], name="param")

    result = config_func()
    assert result["values"] == ["a", "b"]


def test_multi_select_invalid_selection_type():
    @config
    def config_func(hp: HP):
        values = hp.multi_select({"k1": "v1", "k2": "v2"}, default=["k1"], name="param")

    # Test with string instead of list
    with pytest.raises(TypeError):
        config_func(selections={"param": "k1"})  # Should be ["k1"]


def test_multi_select_invalid_dict_key_types():
    with pytest.raises(ValueError):

        @config
        def invalid_dict_keys(hp: HP):
            var = hp.multi_select(
                {1: "a", 2.0: "b", True: "c", "str": "d", complex(1, 2): "e"}, default=["d"], name="param"
            )

        invalid_dict_keys()


def test_multi_select_invalid_list_value_types():
    with pytest.raises(ValueError):

        @config
        def invalid_list_values(hp: HP):
            var = hp.multi_select([1, 2.0, True, "str", complex(1, 2)], default=["str"], name="param")

        invalid_list_values()


def test_multi_select_missing_default():
    with pytest.raises(ValueError):

        @config
        def missing_default(hp: HP):
            hp.multi_select(["a", "b"], name="param")

        missing_default()
