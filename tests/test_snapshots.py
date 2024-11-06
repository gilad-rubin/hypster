import pytest

from hypster import HP, config, save


def test_basic_snapshot():
    @config
    def basic_config(hp: HP):
        model = hp.select(["cnn", "rnn"], default="cnn")
        lr = hp.number(0.001)
        epochs = hp.number(10)

    # Run with default values
    result1 = basic_config()
    snapshot1 = basic_config.get_latest_snapshot()

    assert snapshot1 == {"model": "cnn", "lr": 0.001, "epochs": 10}

    # Run with the snapshot as values
    result2 = basic_config(values=snapshot1)

    assert result1 == result2


def test_nested_snapshot():
    @config
    def nested_config(hp: HP):
        optimizer = hp.select(["adam", "sgd"], default="adam")
        lr = hp.number(0.001)

    save(nested_config, "tests/helper_configs/nested_config.py")

    @config
    def main_config(hp: HP):
        from hypster import load

        model = hp.select(["cnn", "rnn"], default="cnn")
        nested_config = load("tests/helper_configs/nested_config.py")
        nested_inputs = hp.propagate(nested_config)

    # Run with some values
    result1 = main_config(final_vars=["model", "nested_inputs"], values={"nested_inputs.optimizer": "sgd"})
    snapshot1 = main_config.get_latest_snapshot()

    assert snapshot1 == {"model": "cnn", "nested_inputs.optimizer": "sgd", "nested_inputs.lr": 0.001}

    # Run with the snapshot as values
    result2 = main_config(final_vars=["model", "nested_inputs"], values=snapshot1)

    assert result1 == result2


def test_multi_select_snapshot():
    @config
    def multi_select_config(hp: HP):
        frameworks = hp.multi_select(["pytorch", "tensorflow", "jax"], default=["pytorch"])
        batch_size = hp.number(32)

    # Run with some values
    result1 = multi_select_config(values={"frameworks": ["pytorch", "tensorflow"]})
    snapshot1 = multi_select_config.get_latest_snapshot()

    assert snapshot1 == {"frameworks": ["pytorch", "tensorflow"], "batch_size": 32}

    # Run with the snapshot as values
    result2 = multi_select_config(values=snapshot1)

    assert result1 == result2


def test_conditional_snapshot():
    @config
    def conditional_config(hp: HP):
        model = hp.select(["cnn", "rnn"], default="cnn")
        if model == "cnn":
            filters = hp.number(64)
        else:
            units = hp.number(128)

    # Run with CNN selection
    result_cnn = conditional_config(values={"model": "cnn"})
    snapshot_cnn = conditional_config.get_latest_snapshot()

    assert snapshot_cnn == {"model": "cnn", "filters": 64}

    # Run with RNN selection
    result_rnn = conditional_config(values={"model": "rnn"})
    snapshot_rnn = conditional_config.get_latest_snapshot()

    assert snapshot_rnn == {"model": "rnn", "units": 128}

    # Run with snapshots as values
    assert conditional_config(values=snapshot_cnn) == result_cnn
    assert conditional_config(values=snapshot_rnn) == result_rnn


def test_snapshot_with_text():
    @config
    def text_config(hp: HP):
        model = hp.select(["cnn", "rnn"], default="cnn")
        name = hp.text("default_model")

    # Run with some values and values
    result1 = text_config(values={"model": "rnn", "name": "custom_rnn"})
    snapshot1 = text_config.get_latest_snapshot()

    assert snapshot1 == {"model": "rnn", "name": "custom_rnn"}

    # Run with the snapshot as values
    result2 = text_config(values=snapshot1)

    assert result1 == result2


def test_snapshot_history():
    @config
    def history_config(hp: HP):
        model = hp.select(["cnn", "rnn", "transformer"], default="cnn")
        lr = hp.number(0.001)

    # Run multiple times with different values
    history_config(values={"model": "cnn"})
    history_config(values={"model": "rnn"})
    history_config(values={"model": "transformer"})

    # # Check if we can access all snapshots
    # snapshots = history_config.snapshot_history

    # assert len(snapshots) == 3
    # assert snapshots[0] == {"model": "cnn", "lr": 0.001}
    # assert snapshots[1] == {"model": "rnn", "lr": 0.001}
    # assert snapshots[2] == {"model": "transformer", "lr": 0.001}


if __name__ == "__main__":
    pytest.main([__file__])
