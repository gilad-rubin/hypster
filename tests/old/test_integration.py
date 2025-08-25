import pytest

import hypster
from hypster import HP, config


def test_defaults_selections_overrides():
    @config
    def config_func(hp: HP):
        model = hp.select(["cnn", "rnn", "transformer"], default="cnn", name="model")
        lr = hp.number(0.001, name="lr")
        epochs = hp.number(10, name="epochs")

    # Test defaults
    result = config_func()
    assert result["model"] == "cnn"
    assert result["lr"] == 0.001
    assert result["epochs"] == 10

    # Test selections
    result = config_func(values={"model": "rnn"})
    assert result["model"] == "rnn"
    assert result["lr"] == 0.001

    # Test overrides
    result = config_func(values={"model": "transformer", "lr": 0.01})
    assert result["model"] == "transformer"
    assert result["lr"] == 0.01

    # Test precedence: overrides > selections > defaults
    result = config_func(values={"model": "rnn"})
    assert result["model"] == "rnn"

    # Test that overrides returns the value of the selection if present in the dictionary
    result = config_func(values={"model": "rnn"})
    assert result["model"] == "rnn"


def test_error_cases():
    # Test empty options
    with pytest.raises(Exception):

        @config
        def empty_options(hp: HP):
            a = hp.select([])

        empty_options()

    # Test non-existent default
    with pytest.raises(Exception):

        @config
        def bad_default(hp: HP):
            a = hp.select(["a"], default="b")

        bad_default()

    # Test non-existent default
    with pytest.raises(Exception):

        @config
        def bad_default_dct(hp: HP):
            a = hp.select({"a": 5}, default=5)

        bad_default_dct()

    # Test invalid key types in dictionary options
    with pytest.raises(Exception):

        @config
        def invalid_dict_keys(hp: HP):
            var = hp.select(
                {1: "a", 2.0: "b", True: "c", "str": "d", complex(1, 2): "e"},
                default="d",
            )

        invalid_dict_keys()

    # Test invalid value types in list options
    with pytest.raises(Exception):

        @config
        def invalid_list_values(hp: HP):
            var = hp.select([1, 2.0, True, "str", complex(1, 2)], default="str")

        invalid_list_values()

    # Test missing default and no selection/override
    with pytest.raises(Exception):

        @config
        def missing_default(hp: HP):
            hp.select(["a", "b"])

        missing_default()


def test_final_vars():
    @config
    def config_func(hp: HP):
        model = hp.select(["cnn", "rnn"], default="cnn", name="model")
        lr = hp.number(0.001, name="lr")
        epochs = hp.number(10, name="epochs")

    # Test partial final_vars
    result = config_func(final_vars=["model", "lr"])
    assert "model" in result
    assert "lr" in result
    assert "epochs" not in result

    # Test empty final_vars
    result = config_func(final_vars=[])
    assert "model" in result
    assert "lr" in result
    assert "epochs" in result

    # Test non-existent final_vars
    with pytest.raises(Exception) as exc_info:
        config_func(final_vars=["non_existent"])
    assert "do not exist in the configuration" in str(exc_info.value)

    # Test mixture of existing and non-existing final_vars
    with pytest.raises(Exception) as exc_info:
        config_func(final_vars=["model", "non_existent"])
    assert "do not exist in the configuration: non_existent" in str(exc_info.value)


def test_pythonic_api():
    @config
    def config_func(hp: HP):
        # Test conditional statements
        dataset_size = hp.select(["small", "medium", "large"], default="medium", name="dataset_size")
        if dataset_size == "small":
            model = hp.select(["simple_cnn", "small_rnn"], default="simple_cnn", name="model")
        elif dataset_size == "medium":
            model = hp.select(["resnet", "lstm"], default="lstm", name="model")
        else:
            model = hp.select(["transformer", "large_cnn"], default="transformer", name="model")

        # Test loops
        num_layers = hp.select([3, 5, 7], default=5, name="num_layers")
        layer_sizes = []
        for i in range(num_layers):
            layer_sizes.append(hp.select([32, 64, 128], default=64, name=f"layer_{i}_size"))

    # Test defaults
    result = config_func(values={"model": "resnet"})
    assert result["dataset_size"] == "medium"
    assert result["model"] in ["resnet", "lstm"]
    # Check that we have the expected number of layer parameters
    layer_params = [k for k in result.keys() if k.startswith("layer_") and k.endswith("_size")]
    assert len(layer_params) == 5
    assert all(result[layer_param] == 64 for layer_param in layer_params)

    # Test with different selections
    result = config_func(values={"dataset_size": "large", "num_layers": 3})
    assert result["dataset_size"] == "large"
    assert result["model"] in ["transformer", "large_cnn"]
    # Check that we have the expected number of layer parameters
    layer_params = [k for k in result.keys() if k.startswith("layer_") and k.endswith("_size")]
    assert len(layer_params) == 3

    # Test overrides for loop-generated values
    result = config_func(values={"layer_0_size": 32, "layer_1_size": 128})
    assert result["layer_0_size"] == 32
    assert result["layer_1_size"] == 128


@config
def complex_config(hp: HP):
    import os

    def func(x):
        return x

    class TestClass:
        def __init__(self, hello):
            self.hello = hello

    b = func(6)
    c = TestClass("hey")
    cwd = os.getcwd()
    nested_param = hp.select(["a", "b"], name="nested_param", default="a")


complex_config.save("tests/helper_configs/complex_config.py")


def test_save_load_complex_module():
    complex_config = hypster.load("tests/helper_configs/complex_config.py")

    results = complex_config()

    # Only HP parameters are returned in the new behavior
    assert results["nested_param"] == "a"

    # Test with different values
    results = complex_config(values={"nested_param": "b"})
    assert results["nested_param"] == "b"


def test_multi_select_defaults_values():
    @config
    def config_func(hp: HP):
        options = hp.multi_select(["option1", "option2", "option3"], name="options", default=["option1", "option2"])

    # Test defaults
    result = config_func()
    assert result["options"] == ["option1", "option2"]

    # Test selections
    result = config_func(values={"options": ["option2", "option3"]})
    assert result["options"] == ["option2", "option3"]

    # Test overrides
    result = config_func(values={"options": ["option1", "option3"]})
    assert result["options"] == ["option1", "option3"]

    # Test precedence: overrides > selections > defaults
    result = config_func(values={"options": ["option2"]})
    assert result["options"] == ["option2"]


def test_nesting():
    @config
    def nested_config(hp: HP):
        nested_param = hp.select(["a", "b"], name="nested_param", default="a")

    hypster.save(nested_config, "tests/helper_configs/nested_config.py")

    @config
    def main_config(hp: HP):
        nested = hp.nest(
            "tests/helper_configs/nested_config.py",
            name="nested",
        )

        main_param = hp.select(["x", "y"], name="main_param", default="x")

    # Test defaults
    result = main_config()
    assert result["main_param"] == "x"
    assert result["nested"]["nested_param"] == "a"

    # Test selections for nested config
    result = main_config(values={"nested.nested_param": "b"})
    assert result["main_param"] == "x"
    assert result["nested"]["nested_param"] == "b"

    # Test selections for nested config
    result = main_config(values={"nested.nested_param": "b"})
    assert result["main_param"] == "x"
    assert result["nested"]["nested_param"] == "b"

    # Test overrides for both main and nested configs
    result = main_config(values={"main_param": "y", "nested.nested_param": "b"})
    assert result["main_param"] == "y"
    assert result["nested"]["nested_param"] == "b"

    # Test selections for nested config
    result = main_config(values={"nested": {"nested_param": "b"}})
    assert result["main_param"] == "x"
    assert result["nested"]["nested_param"] == "b"

    # Test overrides for both main and nested configs
    result = main_config(values={"main_param": "y", "nested": {"nested_param": "b"}})
    assert result["main_param"] == "y"
    assert result["nested"]["nested_param"] == "b"


def test_multi_select_defaults_values():
    @config
    def config_func(hp: HP):
        options = hp.multi_select(["option1", "option2", "option3"], name="options", default=["option1", "option2"])

    # Test defaults
    result = config_func()
    assert result["options"] == ["option1", "option2"]

    # Test selections
    result = config_func(values={"options": ["option2", "option3"]})
    assert result["options"] == ["option2", "option3"]

    # Test overrides
    result = config_func(values={"options": ["option1", "option3"]})
    assert result["options"] == ["option1", "option3"]

    # Test precedence: overrides > selections > defaults
    result = config_func(values={"options": ["option2"]})
    assert result["options"] == ["option2"]


def test_multi_select_dict_defaults_overrides_selections():
    @config
    def config_func(hp: HP):
        class TestClass:
            def __init__(self, name):
                self.name = name

        options = hp.multi_select(
            {"option1": 1, "option2": 2},
            name="options",
            default=["option1", "option2"],
        )

    # Test defaults
    result = config_func()
    assert result["options"] == [1, 2]

    # Test selections
    result = config_func(values={"options": ["option2"]})
    assert result["options"] == [2]

    # Test overrides
    result = config_func(values={"options": ["option1"]})
    assert result["options"] == [1]

    # Test precedence: overrides > selections > defaults
    result = config_func(values={"options": ["option2"]})
    assert result["options"] == [2]


def test_multi_text_defaults_overrides_selections():
    @config
    def config_func(hp: HP):
        texts = hp.multi_text(default=["default1", "default2"], name="texts")

    # Test defaults
    result = config_func()
    assert result["texts"] == ["default1", "default2"]

    # Test overrides
    result = config_func(values={"texts": ["override1", "override2"]})
    assert result["texts"] == ["override1", "override2"]

    # Test partial overrides
    result = config_func(values={"texts": ["override1"]})
    assert result["texts"] == ["override1"]

    # Test empty overrides
    result = config_func(values={"texts": []})
    assert result["texts"] == []


def test_multi_number_defaults_overrides_selections():
    @config
    def config_func(hp: HP):
        numbers = hp.multi_number(default=[1, 2, 3], name="numbers")

    # Test defaults
    result = config_func()
    assert result["numbers"] == [1, 2, 3]

    # Test overrides
    result = config_func(values={"numbers": [4, 5, 6]})
    assert result["numbers"] == [4, 5, 6]

    # Test partial overrides
    result = config_func(values={"numbers": [7]})
    assert result["numbers"] == [7]

    # Test empty overrides
    result = config_func(values={"numbers": []})
    assert result["numbers"] == []


def test_multi_select_error_cases():
    @config
    def config_func(hp: HP):
        hp.multi_select([], name="empty_options", default=[])

    # Test empty options
    with pytest.raises(Exception):
        config_func()

    # Test invalid default
    with pytest.raises(Exception):

        @config
        def bad_default(hp: HP):
            hp.multi_select(["a", "b"], name="bad_default", default=["c"])

        bad_default()


def test_multi_text_error_cases():
    @config
    def config_func(hp: HP):
        hp.multi_text(name="multi_text", default=["default"])

    # multi_text does not have complex validation, but we can test type enforcement if implemented.


def test_multi_number_error_cases():
    @config
    def config_func(hp: HP):
        hp.multi_number(name="multi_number", default=[1, 2])

    # multi_number does not have complex validation, but we can test type enforcement if implemented.
