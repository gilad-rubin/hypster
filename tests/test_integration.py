import pytest

import hypster
from hypster import HP, config


def test_defaults_selections_overrides():
    @config
    def config_func(hp: HP):
        model = hp.select(["cnn", "rnn", "transformer"], default="cnn")
        lr = hp.number_input(0.001)
        epochs = hp.number_input(10)

    # Test defaults
    result = config_func()
    assert result["model"] == "cnn"
    assert result["lr"] == 0.001
    assert result["epochs"] == 10

    # Test selections
    result = config_func(selections={"model": "rnn"})
    assert result["model"] == "rnn"
    assert result["lr"] == 0.001

    # Test overrides
    result = config_func(overrides={"model": "transformer", "lr": 0.01})
    assert result["model"] == "transformer"
    assert result["lr"] == 0.01

    # Test precedence: overrides > selections > defaults
    result = config_func(selections={"model": "rnn"}, overrides={"model": "transformer"})
    assert result["model"] == "transformer"

    # Test that overrides returns the value of the selection if present in the dictionary
    result = config_func(overrides={"model": "rnn"})
    assert result["model"] == "rnn"


def test_error_cases():
    # Test empty options
    with pytest.raises(ValueError):

        @config
        def empty_options(hp: HP):
            a = hp.select([])

        empty_options()

    # Test non-existent default
    with pytest.raises(ValueError):

        @config
        def bad_default(hp: HP):
            a = hp.select(["a"], default="b")

        bad_default()

    # Test non-existent default
    with pytest.raises(ValueError):

        @config
        def bad_default_dct(hp: HP):
            a = hp.select({"a": 5}, default=5)

        bad_default_dct()

    # Test invalid key types in dictionary options
    with pytest.raises(ValueError):

        @config
        def invalid_dict_keys(hp: HP):
            var = hp.select({1: "a", 2.0: "b", True: "c", "str": "d", complex(1, 2): "e"}, default="d")

        invalid_dict_keys()

    # Test invalid value types in list options
    with pytest.raises(ValueError):

        @config
        def invalid_list_values(hp: HP):
            var = hp.select([1, 2.0, True, "str", complex(1, 2)], default="str")

        invalid_list_values()

    # Test missing default and no selection/override
    with pytest.raises(ValueError):

        @config
        def missing_default(hp: HP):
            hp.select(["a", "b"])

        missing_default()


def test_final_vars():
    @config
    def config_func(hp: HP):
        model = hp.select(["cnn", "rnn"], default="cnn")
        lr = hp.number_input(0.001)
        epochs = hp.number_input(10)

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
    with pytest.raises(ValueError) as exc_info:
        config_func(final_vars=["non_existent"])
    assert "do not exist in the configuration" in str(exc_info.value)

    # Test mixture of existing and non-existing final_vars
    with pytest.raises(ValueError) as exc_info:
        config_func(final_vars=["model", "non_existent", "another_non_existent"])
    assert "do not exist in the configuration: non_existent, another_non_existent" in str(exc_info.value)


def test_pythonic_api():
    @config
    def config_func(hp: HP):
        # Test conditional statements
        dataset_size = hp.select(["small", "medium", "large"], default="medium")
        if dataset_size == "small":
            model = hp.select(["simple_cnn", "small_rnn"], default="simple_cnn")
        elif dataset_size == "medium":
            model = hp.select(["resnet", "lstm"], default="lstm")
        else:
            model = hp.select(["transformer", "large_cnn"], default="transformer")

        # Test loops
        num_layers = hp.select([3, 5, 7], default=5)
        layer_sizes = []
        for i in range(num_layers):
            layer_sizes.append(hp.select([32, 64, 128], default=64, name=f"layer_{i}_size"))

    # Test defaults
    result = config_func(selections={"model": "resnet"})
    assert result["dataset_size"] == "medium"
    assert result["model"] in ["resnet", "lstm"]
    assert len(result["layer_sizes"]) == 5
    assert all(size == 64 for size in result["layer_sizes"])

    # Test with different selections
    result = config_func(selections={"dataset_size": "large", "num_layers": 3})
    assert result["dataset_size"] == "large"
    assert result["model"] in ["transformer", "large_cnn"]
    assert len(result["layer_sizes"]) == 3

    # Test overrides for loop-generated values
    result = config_func(overrides={"layer_0_size": 32, "layer_1_size": 128})
    assert result["layer_sizes"][0] == 32
    assert result["layer_sizes"][1] == 128


def test_propagation():
    @config
    def nested_config(hp: HP):
        nested_param = hp.select(["a", "b"], default="a")

    hypster.save(nested_config, "tests/nested_config.py")

    @config
    def main_config(hp: HP):
        import hypster

        nested_config = hypster.load("tests/nested_config.py")
        nested = hp.propagate(nested_config)

        main_param = hp.select(["x", "y"], default="x")

    # Test defaults
    result = main_config()
    assert result["main_param"] == "x"
    assert result["nested"]["nested_param"] == "a"

    # Test selections for nested config
    result = main_config(selections={"nested.nested_param": "b"})
    assert result["main_param"] == "x"
    assert result["nested"]["nested_param"] == "b"

    # Test overrides for both main and nested configs
    result = main_config(overrides={"main_param": "y", "nested.nested_param": "b"})
    assert result["main_param"] == "y"
    assert result["nested"]["nested_param"] == "b"
