import pytest

from hypster import HP, config


def test_variable_naming():
    @config
    def config_func(hp: HP):
        var1 = hp.select(["a", "b"], name="var1", default="a")
        var2 = hp.number(10, name="var2")

    # Test with defaults
    result = config_func()
    assert result["var1"] == "a"
    assert result["var2"] == 10

    # Test with selections
    result = config_func(values={"var1": "b"})
    assert result["var1"] == "b"
    assert result["var2"] == 10

    # Test with overrides
    result = config_func(values={"var2": 20})
    assert result["var1"] == "a"
    assert result["var2"] == 20


def test_dict_naming():
    @config
    def config_func(hp: HP):
        config = {
            "model_type": hp.select(["cnn", "rnn"], name="config.model_type", default="cnn"),
            "learning_rate": hp.number(0.001, name="config.learning_rate"),
        }

    # Test with defaults
    result = config_func()
    assert result["config"]["model_type"] == "cnn"
    assert result["config"]["learning_rate"] == 0.001

    # Test with selections
    result = config_func(values={"config.model_type": "rnn"})
    assert result["config"]["model_type"] == "rnn"
    assert result["config"]["learning_rate"] == 0.001

    # Test with overrides
    result = config_func(values={"config.learning_rate": 0.01})
    assert result["config"]["model_type"] == "cnn"
    assert result["config"]["learning_rate"] == 0.01


def test_nested_naming():
    @config
    def config_func(hp: HP):
        outer = {
            "inner": {
                "deep": hp.select(["x", "y"], name="outer.inner.deep", default="x"),
                "value": hp.number(5, name="outer.inner.value"),
            }
        }

    # Test with defaults
    result = config_func()
    assert result["outer"]["inner"]["deep"] == "x"
    assert result["outer"]["inner"]["value"] == 5

    # Test with selections
    result = config_func(values={"outer.inner.deep": "y"})
    assert result["outer"]["inner"]["deep"] == "y"
    assert result["outer"]["inner"]["value"] == 5

    # Test with overrides
    result = config_func(values={"outer.inner.value": 10})
    assert result["outer"]["inner"]["deep"] == "x"
    assert result["outer"]["inner"]["value"] == 10


def test_class_naming():
    @config
    def config_func(hp: HP):
        class ModelConfig:
            def __init__(self, model_type, learning_rate):
                self.model_type = model_type
                self.learning_rate = learning_rate

        model = ModelConfig(
            model_type=hp.select(["cnn", "rnn"], name="model.model_type", default="cnn"),
            learning_rate=hp.number(0.001, name="model.learning_rate"),
        )

    # Test with defaults - check the nested structure created from dotted names
    result = config_func()
    assert result["model"]["model_type"] == "cnn"
    assert result["model"]["learning_rate"] == 0.001

    # Test with selections
    result = config_func(values={"model.model_type": "rnn"})
    assert result["model"]["model_type"] == "rnn"
    assert result["model"]["learning_rate"] == 0.001

    # Test with overrides
    result = config_func(values={"model.learning_rate": 0.01})
    assert result["model"]["model_type"] == "cnn"
    assert result["model"]["learning_rate"] == 0.01


def test_function_naming():
    @config
    def config_func(hp: HP):
        def inner_func(param1, param2):
            return param1, param2

        result = inner_func(
            param1=hp.select(["a", "b"], name="result.param1", default="a"),
            param2=hp.number(10, name="result.param2"),
        )

    # Test with defaults - check the nested structure created from dotted names
    result = config_func()
    assert result["result"]["param1"] == "a"
    assert result["result"]["param2"] == 10

    # Test with selections
    result = config_func(values={"result.param1": "b"})
    assert result["result"]["param1"] == "b"
    assert result["result"]["param2"] == 10

    # Test with overrides
    result = config_func(values={"result.param2": 20})
    assert result["result"]["param1"] == "a"
    assert result["result"]["param2"] == 20


def test_disable_automatic_naming_with_explicit_names():
    @config
    def class_kwargs_naming(hp: HP):
        class ModelConfig:
            def __init__(self, model_type, learning_rate):
                self.model_type = model_type
                self.learning_rate = learning_rate

        model = ModelConfig(
            model_type=hp.select(["cnn", "rnn"], name="model_type", default="cnn"),
            learning_rate=hp.number(0.001, name="learning_rate"),
        )

    # Test with explicit names - parameters appear directly in result
    result = class_kwargs_naming()
    assert result["model_type"] == "cnn"
    assert result["learning_rate"] == 0.001

    result = class_kwargs_naming(values={"model_type": "rnn"})
    assert result["model_type"] == "rnn"

    result = class_kwargs_naming(values={"learning_rate": 0.01})
    assert result["learning_rate"] == 0.01


def test_disable_automatic_naming_missing_name_error():
    @config
    def no_injection_config(hp: HP):
        a = hp.select(["a", "b"])  # This should raise an error

    with pytest.raises(Exception):
        no_injection_config()


def test_non_constant_name():
    @config
    def my_config(hp: HP):
        var = "a"
        a = hp.select(["a", "b"], name=f"hey_{var}")

    results = my_config(values={"hey_a": "a"})
    assert results["hey_a"] == "a"
