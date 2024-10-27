import pytest

from hypster import HP, SelectionHandler, config, save


# Test initialization
def test_initialization():
    @config
    def simple_config(hp: HP):
        model = hp.select(["cnn", "rnn"], default="cnn")
        lr = hp.select([0.001, 0.01], default=0.001)
        optimizer = hp.select(["adam", "sgd"], default="adam")

    handler = SelectionHandler(simple_config)
    handler.initialize()

    assert handler.selected_params == {"model": "cnn", "lr": 0.001, "optimizer": "adam"}
    assert handler.current_options == {"model": {"cnn", "rnn"}, "lr": {0.001, 0.01}, "optimizer": {"adam", "sgd"}}


# Test updating a parameter
def test_update_param():
    @config
    def simple_config(hp: HP):
        model = hp.select(["cnn", "rnn"], default="cnn")
        lr = hp.select([0.001, 0.01], default=0.001)
        optimizer = hp.select(["adam", "sgd"], default="adam")

    handler = SelectionHandler(simple_config)
    handler.initialize()

    handler.update_param("model", "rnn")

    assert handler.selected_params == {"model": "rnn", "lr": 0.001, "optimizer": "adam"}
    assert handler.current_options == {"model": {"cnn", "rnn"}, "lr": {0.001, 0.01}, "optimizer": {"adam", "sgd"}}


# Test multi_select
def test_multi_select():
    @config
    def multi_select_config(hp: HP):
        frameworks = hp.multi_select(
            {"pytorch": "PyTorch", "tensorflow": "TensorFlow", "jax": "JAX", "mxnet": "MXNet"},
            default=["pytorch", "tensorflow"],
        )
        optimizer = hp.select(["adam", "sgd"], default="adam")

    handler = SelectionHandler(multi_select_config)
    handler.initialize()

    assert handler.selected_params == {"frameworks": ["pytorch", "tensorflow"], "optimizer": "adam"}
    assert handler.current_options == {
        "frameworks": {"pytorch", "tensorflow", "jax", "mxnet"},
        "optimizer": {"adam", "sgd"},
    }

    handler.update_param("frameworks", ["jax", "mxnet"])

    assert handler.selected_params == {"frameworks": ["jax", "mxnet"], "optimizer": "adam"}
    assert handler.current_options == {
        "frameworks": {"pytorch", "tensorflow", "jax", "mxnet"},
        "optimizer": {"adam", "sgd"},
    }


def test_propagate(tmp_path):
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

    handler = SelectionHandler(outer_config)
    handler.initialize()

    assert handler.selected_params == {"model": "cnn", "inner": {"activation": "relu", "dropout": 0.1}}
    assert handler.current_options == {
        "model": {"cnn", "rnn"},
        "inner": {"activation": {"relu", "tanh"}, "dropout": {0.1, 0.2}},
    }

    handler.update_param("inner", {"activation": "tanh"})

    assert handler.selected_params == {"model": "cnn", "inner": {"activation": "tanh", "dropout": 0.1}}
    assert handler.current_options == {
        "model": {"cnn", "rnn"},
        "inner": {"activation": {"relu", "tanh"}, "dropout": {0.1, 0.2}},
    }


# Test conditional parameters
def test_conditional_parameters():
    @config
    def conditional_config(hp: HP):
        model = hp.select(["cnn", "rnn", "transformer"], default="cnn")

        if model == "cnn":
            layers = hp.select([3, 5, 7], default=5)
            kernel_size = hp.select([3, 5], default=3)
        elif model == "rnn":
            cell_type = hp.select(["lstm", "gru"], default="lstm")
            num_layers = hp.select([1, 2, 12], default=2)
        else:  # transformer
            num_heads = hp.select([4, 8, 16], default=8)
            num_layers = hp.select([3, 6, 12], default=6)

        lr = hp.select([0.001, 0.01, 0.1], default=0.01)

    handler = SelectionHandler(conditional_config)
    handler.initialize()

    # Test initial state (default: CNN)
    assert handler.selected_params == {"model": "cnn", "layers": 5, "kernel_size": 3, "lr": 0.01}
    assert handler.current_options == {
        "model": {"cnn", "rnn", "transformer"},
        "layers": {3, 5, 7},
        "kernel_size": {3, 5},
        "lr": {0.001, 0.01, 0.1},
    }

    # Test switching to RNN
    handler.update_param("model", "rnn")
    assert handler.selected_params == {"model": "rnn", "cell_type": "lstm", "num_layers": 2, "lr": 0.01}
    assert handler.current_options == {
        "model": {"cnn", "rnn", "transformer"},
        "cell_type": {"lstm", "gru"},
        "num_layers": {1, 2, 12},
        "lr": {0.001, 0.01, 0.1},
    }
    assert "layers" not in handler.current_options
    assert "kernel_size" not in handler.current_options

    # Test switching to Transformer
    handler.update_param("model", "transformer")
    assert handler.selected_params == {"model": "transformer", "num_heads": 8, "num_layers": 6, "lr": 0.01}
    assert handler.current_options == {
        "model": {"cnn", "rnn", "transformer"},
        "num_heads": {4, 8, 16},
        "num_layers": {3, 6, 12},
        "lr": {0.001, 0.01, 0.1},
    }
    assert "cell_type" not in handler.current_options
    # Test updating a conditional parameter
    handler.update_param("num_heads", 16)
    handler.update_param("num_layers", 3)
    assert handler.selected_params == {"model": "transformer", "num_heads": 16, "num_layers": 3, "lr": 0.01}

    # Test switching back to CNN
    handler.update_param("num_layers", 12)  # this is mutual with cnn - so when we switch we're expected to keep this
    handler.update_param("model", "cnn")
    assert handler.selected_params == {"model": "cnn", "layers": 5, "kernel_size": 3, "lr": 0.01}
    assert handler.current_options == {
        "model": {"cnn", "rnn", "transformer"},
        "layers": {3, 5, 7},
        "kernel_size": {3, 5},
        "lr": {0.001, 0.01, 0.1},
    }
    assert "num_heads" not in handler.current_options
    assert "num_layers" not in handler.current_options

    handler.update_param("lr", 0.1)
    assert handler.selected_params == {"model": "cnn", "layers": 5, "kernel_size": 3, "lr": 0.1}

    handler.update_param("model", "rnn")
    assert handler.selected_params == {"model": "rnn", "cell_type": "lstm", "num_layers": 12, "lr": 0.1}


if __name__ == "__main__":
    pytest.main([__file__])
