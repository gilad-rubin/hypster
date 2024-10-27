import pytest

from hypster import HP, config, save


def test_basic_snapshot():
    @config
    def basic_config(hp: HP):
        model = hp.select(["cnn", "rnn"], default="cnn")
        lr = hp.number_input(0.001)
        epochs = hp.number_input(10)

    # Run with default values
    result1 = basic_config()
    snapshot1 = basic_config.get_last_snapshot()

    assert snapshot1 == {"model": "cnn", "lr": 0.001, "epochs": 10}

    # Run with the snapshot as overrides
    result2 = basic_config(overrides=snapshot1)

    assert result1 == result2


def test_nested_snapshot():
    @config
    def nested_config(hp: HP):
        optimizer = hp.select(["adam", "sgd"], default="adam")
        lr = hp.number_input(0.001)

    save(nested_config, "tests/helper_configs/nested_config.py")

    @config
    def main_config(hp: HP):
        from hypster import load

        model = hp.select(["cnn", "rnn"], default="cnn")
        nested_config = load("tests/helper_configs/nested_config.py")
        nested_inputs = hp.propagate(nested_config)

    # Run with some selections
    result1 = main_config(final_vars=["model", "nested_inputs"], selections={"nested_inputs.optimizer": "sgd"})
    snapshot1 = main_config.get_last_snapshot()

    assert snapshot1 == {"model": "cnn", "nested_inputs": {"optimizer": "sgd", "lr": 0.001}}

    # Run with the snapshot as overrides
    result2 = main_config(final_vars=["model", "nested_inputs"], overrides=snapshot1)

    assert result1 == result2


def test_multi_select_snapshot():
    @config
    def multi_select_config(hp: HP):
        frameworks = hp.multi_select(["pytorch", "tensorflow", "jax"], default=["pytorch"])
        batch_size = hp.number_input(32)

    # Run with some selections
    result1 = multi_select_config(selections={"frameworks": ["pytorch", "tensorflow"]})
    snapshot1 = multi_select_config.get_last_snapshot()

    assert snapshot1 == {"frameworks": ["pytorch", "tensorflow"], "batch_size": 32}

    # Run with the snapshot as overrides
    result2 = multi_select_config(overrides=snapshot1)

    assert result1 == result2


def test_conditional_snapshot():
    @config
    def conditional_config(hp: HP):
        model = hp.select(["cnn", "rnn"], default="cnn")
        if model == "cnn":
            filters = hp.number_input(64)
        else:
            units = hp.number_input(128)

    # Run with CNN selection
    result_cnn = conditional_config(selections={"model": "cnn"})
    snapshot_cnn = conditional_config.get_last_snapshot()

    assert snapshot_cnn == {"model": "cnn", "filters": 64}

    # Run with RNN selection
    result_rnn = conditional_config(selections={"model": "rnn"})
    snapshot_rnn = conditional_config.get_last_snapshot()

    assert snapshot_rnn == {"model": "rnn", "units": 128}

    # Run with snapshots as overrides
    assert conditional_config(overrides=snapshot_cnn) == result_cnn
    assert conditional_config(overrides=snapshot_rnn) == result_rnn


def test_snapshot_with_text_input():
    @config
    def text_input_config(hp: HP):
        model = hp.select(["cnn", "rnn"], default="cnn")
        name = hp.text_input("default_model")

    # Run with some selections and overrides
    result1 = text_input_config(selections={"model": "rnn"}, overrides={"name": "custom_rnn"})
    snapshot1 = text_input_config.get_last_snapshot()

    assert snapshot1 == {"model": "rnn", "name": "custom_rnn"}

    # Run with the snapshot as overrides
    result2 = text_input_config(overrides=snapshot1)

    assert result1 == result2


def test_snapshot_history():
    @config
    def history_config(hp: HP):
        model = hp.select(["cnn", "rnn", "transformer"], default="cnn")
        lr = hp.number_input(0.001)

    # Run multiple times with different selections
    history_config(selections={"model": "cnn"})
    history_config(selections={"model": "rnn"})
    history_config(selections={"model": "transformer"})

    # Check if we can access all snapshots
    snapshots = history_config.snapshot_history

    assert len(snapshots) == 3
    assert snapshots[0] == {"model": "cnn", "lr": 0.001}
    assert snapshots[1] == {"model": "rnn", "lr": 0.001}
    assert snapshots[2] == {"model": "transformer", "lr": 0.001}


if __name__ == "__main__":
    pytest.main([__file__])
