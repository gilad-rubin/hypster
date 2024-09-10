import pytest

from hypster import HP, config, query_combinations, save


# Test 1: Simple configuration
def test_simple_config():
    @config
    def simple_config(hp: HP):
        model = hp.select(["cnn", "rnn"], default="cnn")
        lr = hp.select([0.001, 0.01], default=0.001)

    combinations = simple_config.get_combinations()
    assert len(combinations) == 4
    assert {"model": "cnn", "lr": 0.001} in combinations
    assert {"model": "cnn", "lr": 0.01} in combinations
    assert {"model": "rnn", "lr": 0.001} in combinations
    assert {"model": "rnn", "lr": 0.01} in combinations


# Test 2: Configuration with conditional branches
def test_conditional_config():
    @config
    def conditional_config(hp: HP):
        model = hp.select(["cnn", "rnn"], default="cnn")
        if model == "cnn":
            layers = hp.select([3, 5], default=3)
        else:
            cell_type = hp.select(["lstm", "gru"], default="lstm")

    combinations = conditional_config.get_combinations()
    assert len(combinations) == 4
    assert {"model": "cnn", "layers": 3} in combinations
    assert {"model": "cnn", "layers": 5} in combinations
    assert {"model": "rnn", "cell_type": "lstm"} in combinations
    assert {"model": "rnn", "cell_type": "gru"} in combinations


# Test 3: Nested configurations
def test_nested_config(tmp_path):
    @config
    def inner_config(hp: HP):
        activation = hp.select(["relu", "tanh"], default="relu")

    save(inner_config, "inner_config.py")

    @config
    def outer_config(hp: HP):
        from hypster import load

        model = hp.select(["cnn", "rnn"], default="cnn")
        inner = hp.propagate(load("inner_config.py"))

    combinations = outer_config.get_combinations()
    print(combinations)
    assert len(combinations) == 4
    assert {"model": "cnn", "inner.activation": "relu"} in combinations
    assert {"model": "cnn", "inner.activation": "tanh"} in combinations
    assert {"model": "rnn", "inner.activation": "relu"} in combinations
    assert {"model": "rnn", "inner.activation": "tanh"} in combinations


# Test 4: Multiple nested levels
def test_multiple_nested_levels(tmp_path):
    @config
    def innermost_config(hp: HP):
        dropout = hp.select([0.1, 0.2], default=0.1)

    save(innermost_config, "innermost_config.py")

    @config
    def inner_config(hp: HP):
        from hypster import load

        activation = hp.select(["relu", "tanh"], default="relu")
        innermost = hp.propagate(load("innermost_config.py"), name="innermost")

    save(inner_config, "inner_config.py")

    @config
    def outer_config(hp: HP):
        from hypster import load

        model = hp.select(["cnn", "rnn"], default="cnn")
        inner = hp.propagate(load("inner_config.py"), name="inner")

    combinations = outer_config.get_combinations()
    print(combinations)
    assert len(combinations) == 8
    assert {"model": "cnn", "inner.activation": "relu", "inner.innermost.dropout": 0.1} in combinations
    assert {"model": "rnn", "inner.activation": "tanh", "inner.innermost.dropout": 0.2} in combinations


# Test 5: Query combinations with exact matches
def test_query_exact_match():
    @config
    def config_func(hp: HP):
        model = hp.select(["cnn", "rnn"], default="cnn")
        lr = hp.select([0.001, 0.01], default=0.001)

    combinations = config_func.get_combinations()
    filtered = query_combinations(combinations, {"model": "cnn", "lr": 0.001})
    assert len(filtered) == 1
    assert filtered[0] == {"model": "cnn", "lr": 0.001}


# Test 6: Query combinations with partial matches
def test_query_partial_match():
    @config
    def config_func(hp: HP):
        model = hp.select(["cnn", "rnn"], default="cnn")
        lr = hp.select([0.001, 0.01], default=0.001)

    combinations = config_func.get_combinations()
    filtered = query_combinations(combinations, {"model": "cnn"})
    assert len(filtered) == 2
    assert {"model": "cnn", "lr": 0.001} in filtered
    assert {"model": "cnn", "lr": 0.01} in filtered


# Test 7: Query combinations with no matches
def test_query_no_match():
    @config
    def config_func(hp: HP):
        model = hp.select(["cnn", "rnn"], default="cnn")
        lr = hp.select([0.001, 0.01], default=0.001)

    combinations = config_func.get_combinations()
    filtered = query_combinations(combinations, {"model": "transformer"})
    assert len(filtered) == 0


# Test 8: Combinations with non-select parameters
def test_non_select_params():
    @config
    def config_func(hp: HP):
        model = hp.select(["cnn", "rnn"], default="cnn")
        lr = hp.number_input(default=0.001)
        name = hp.text_input(default="model")

    combinations = config_func.get_combinations()
    assert len(combinations) == 2
    assert all("lr" not in combo for combo in combinations)
    assert all("name" not in combo for combo in combinations)


# Test 9: Error handling for invalid configurations
def test_invalid_config():
    with pytest.raises(ValueError):

        @config
        def invalid_config(hp: HP):
            model = hp.select([], default="cnn")

        invalid_config.get_combinations()


# Test 10: Performance with a large number of combinations
def test_large_combinations():
    @config
    def large_config(hp: HP):
        for i in range(10):
            hp.select([f"option1_{i}", f"option2_{i}"], default=f"option1_{i}", name=f"option_{i}")

    combinations = large_config.get_combinations()
    assert len(combinations) == 2**10  # 1024 combinations


if __name__ == "__main__":
    pytest.main([__file__])
