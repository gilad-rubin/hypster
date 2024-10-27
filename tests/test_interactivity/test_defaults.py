import pytest

from hypster import HP, config, save


# Test 12: Simple defaults
def test_simple_defaults():
    @config
    def simple_config(hp: HP):
        model = hp.select(["cnn", "rnn"], default="cnn")
        lr = hp.select([0.001, 0.01], default=0.001)

    defaults = simple_config.get_defaults()
    assert defaults == {"model": ["cnn"], "lr": [0.001]}


# Test 13: Multiple defaults in conditional branches
def test_multiple_defaults():
    @config
    def multi_default_config(hp: HP):
        model = hp.select(["cnn", "rnn"], default="cnn")
        if model == "cnn":
            layers = hp.select([2, 3], default=2)
        else:
            layers = hp.select([4, 5], default=4)

    defaults = multi_default_config.get_defaults()
    assert defaults == {"model": ["cnn"], "layers": [2, 4]}


# Test 14: Nested defaults with propagation
def test_nested_defaults(tmp_path):
    @config
    def inner_config(hp: HP):
        activation = hp.select(["relu", "tanh"], default="relu")
        dropout = hp.select([0.1, 0.2], default=0.1)

    save(inner_config, "tests/helper_configs/inner_config.py")

    @config
    def outer_config(hp: HP):
        from hypster import load

        model = hp.select(["cnn", "rnn"], default="cnn")
        inner = hp.propagate(load("tests/helper_configs/inner_config.py"))

    defaults = outer_config.get_defaults()
    assert defaults == {"model": ["cnn"], "inner": {"activation": ["relu"], "dropout": [0.1]}}


# Test 15: Defaults with multi_select
def test_multi_select_defaults():
    @config
    def multi_select_config(hp: HP):
        frameworks = hp.multi_select(
            {"pytorch": "PyTorch", "tensorflow": "TensorFlow", "jax": "JAX"}, default=["pytorch", "tensorflow"]
        )
        optimizer = hp.select(["adam", "sgd"], default="adam")

    defaults = multi_select_config.get_defaults()
    assert defaults == {"frameworks": [["pytorch", "tensorflow"]], "optimizer": ["adam"]}


# Test 16: Defaults with text and number inputs
def test_input_defaults():
    @config
    def input_config(hp: HP):
        name = hp.text_input(default="model_v1")
        epochs = hp.number_input(default=100)

    defaults = input_config.get_defaults()
    assert defaults == {"name": ["model_v1"], "epochs": [100]}


# Test 17: Defaults with no default values
def test_no_defaults():
    @config
    def no_default_config(hp: HP):
        model = hp.select(["cnn", "rnn"])
        lr = hp.select([0.001, 0.01])

    defaults = no_default_config.get_defaults()
    assert defaults == {}


# Test 18: Complex nested defaults
def test_complex_nested_defaults(tmp_path):
    @config
    def innermost_config(hp: HP):
        dropout = hp.select([0.1, 0.2], default=0.1)

    save(innermost_config, "tests/helper_configs/innermost_config.py")

    @config
    def inner_config(hp: HP):
        from hypster import load

        activation = hp.select(["relu", "tanh"], default="relu")
        innermost = hp.propagate(load("tests/helper_configs/innermost_config.py"), name="innermost")

    save(inner_config, "tests/helper_configs/inner_config.py")

    @config
    def outer_config(hp: HP):
        from hypster import load

        model = hp.select(["cnn", "rnn"], default="cnn")
        inner = hp.propagate(load("tests/helper_configs/inner_config.py"), name="inner")

    defaults = outer_config.get_defaults()
    assert defaults == {"model": ["cnn"], "inner": {"activation": ["relu"], "innermost": {"dropout": [0.1]}}}


if __name__ == "__main__":
    pytest.main([__file__])
