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
    with pytest.raises(ValueError):

        @config
        def config_func(hp: HP):
            value = hp.select([], default="a", name="param")

        config_func()


def test_select_invalid_options_wrong_type():
    with pytest.raises(ValueError):

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
    with pytest.raises(ValueError):

        @config
        def config_func(hp: HP):
            value = hp.select(["a", "b", "c"], default="d", name="param")

        config_func()


def test_select_invalid_default_not_in_options_dict():
    with pytest.raises(ValueError):

        @config
        def config_func(hp: HP):
            value = hp.select({"k1": "v1", "k2": "v2"}, default="k3", name="param")

        config_func()


# Selection and override behavior with list options
def test_select_with_selection_list():
    @config
    def config_func(hp: HP):
        value = hp.select(["a", "b", "c"], default="a", name="param")

    result = config_func(selections={"param": "b"})
    assert result["value"] == "b"


def test_select_with_override_list():
    @config
    def config_func(hp: HP):
        value = hp.select(["a", "b", "c"], default="a", name="param")

    result = config_func(overrides={"param": "c"})
    assert result["value"] == "c"


# Selection and override behavior with dict options
def test_select_with_selection_dict():
    @config
    def config_func(hp: HP):
        value = hp.select({"k1": "v1", "k2": "v2"}, default="k1", name="param")

    result = config_func(selections={"param": "k2"})
    assert result["value"] == "v2"


def test_select_with_override_dict():
    @config
    def config_func(hp: HP):
        value = hp.select({"k1": "v1", "k2": "v2"}, default="k1", name="param")

    result = config_func(overrides={"param": "k2"})
    assert result["value"] == "v2"


# Override precedence tests
def test_select_override_precedence_list():
    @config
    def config_func(hp: HP):
        value = hp.select(["a", "b", "c"], default="a", name="param")

    # Test with valid options
    result = config_func(selections={"param": "b"}, overrides={"param": "c"})
    assert result["value"] == "c"

    # Test with override value not in options (should still work)
    result = config_func(selections={"param": "b"}, overrides={"param": "d"})
    assert result["value"] == "d"


def test_select_override_precedence_dict():
    @config
    def config_func(hp: HP):
        value = hp.select({"k1": "v1", "k2": "v2"}, default="k1", name="param")

    # Test with valid keys
    result = config_func(selections={"param": "k1"}, overrides={"param": "k2"})
    assert result["value"] == "v2"

    # Test with override key not in options (should still work)
    result = config_func(selections={"param": "k1"}, overrides={"param": "k3"})
    assert result["value"] == "k3"


def test_select_without_selection_or_override():
    @config
    def config_func(hp: HP):
        value = hp.select(["a", "b", "c"], default="a", name="param")

    result = config_func()
    assert result["value"] == "a"


def test_select_invalid_dict_key_types():
    with pytest.raises(ValueError):

        @config
        def invalid_dict_keys(hp: HP):
            var = hp.select({1: "a", 2.0: "b", True: "c", "str": "d", complex(1, 2): "e"}, default="d", name="param")

        invalid_dict_keys()


def test_select_invalid_list_value_types():
    with pytest.raises(ValueError):

        @config
        def invalid_list_values(hp: HP):
            var = hp.select([1, 2.0, True, "str", complex(1, 2)], default="str", name="param")

        invalid_list_values()


def test_select_missing_default():
    with pytest.raises(ValueError):

        @config
        def missing_default(hp: HP):
            hp.select(["a", "b"], name="param")

        missing_default()


def test_select_with_disable_overrides():
    @config
    def config_func(hp: HP):
        value = hp.select(["a", "b", "c"], default="a", name="param", disable_overrides=True)

    # Should work with selections
    result = config_func(selections={"param": "b"})
    assert result["value"] == "b"

    # Should fail with overrides
    with pytest.raises(ValueError, match="Overrides are disabled for 'param'"):
        config_func(overrides={"param": "c"})


def test_select_rejects_list_selection():
    @config
    def config_func(hp: HP):
        value = hp.select(["a", "b", "c"], default="a", name="param")

    with pytest.raises(TypeError, match="Selection for 'param' must not be a list"):
        config_func(selections={"param": ["b"]})


def test_select_rejects_list_override_with_valid_options():
    @config
    def config_func(hp: HP):
        value = hp.select(["a", "b", "c"], default="a", name="param")

    with pytest.raises(ValueError, match="Override values \\['b', 'c'\\] for 'param' are not all valid options"):
        config_func(overrides={"param": ["b", "c"]})
