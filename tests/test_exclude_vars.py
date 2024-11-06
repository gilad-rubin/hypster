from hypster import HP, config


def test_basic_exclude_vars():
    @config
    def config_func(hp: HP):
        model = hp.select(["cnn", "rnn"], default="cnn")
        lr = hp.number(0.001)
        epochs = hp.number(10)

    # Test excluding single variable
    result = config_func(exclude_vars=["lr"])
    assert "model" in result
    assert "epochs" in result
    assert "lr" not in result

    # Test excluding multiple variables
    result = config_func(exclude_vars=["lr", "epochs"])
    assert "model" in result
    assert "lr" not in result
    assert "epochs" not in result

    # Test excluding non-existent variables (should not raise error)
    result = config_func(exclude_vars=["non_existent"])
    assert "model" in result
    assert "lr" in result
    assert "epochs" in result


def test_exclude_vars_with_final_vars():
    @config
    def config_func(hp: HP):
        model = hp.select(["cnn", "rnn"], default="cnn")
        lr = hp.number(0.001)
        epochs = hp.number(10)

    # Test final_vars and exclude_vars together
    result = config_func(final_vars=["model", "lr", "epochs"], exclude_vars=["lr"])
    assert "model" in result
    assert "epochs" in result
    assert "lr" not in result


def test_exclude_vars_with_propagation():
    @config
    def nested_config(hp: HP):
        nested_param = hp.select(["a", "b"], default="a")
        nested_number = hp.number(1.0)

    nested_config.save("tests/helper_configs/nested_config.py")

    @config
    def main_config(hp: HP):
        nested = hp.propagate("tests/helper_configs/nested_config.py", name="nested")
        main_param = hp.select(["x", "y"], default="x")

    # Test excluding nested variables with dot notation
    result = main_config(exclude_vars=["nested.nested_number"])
    assert "nested_param" in result["nested"]
    assert "nested_number" not in result["nested"]
    assert "main_param" in result

    # Test excluding top-level and nested variables
    result = main_config(exclude_vars=["main_param", "nested.nested_number"])
    assert "nested_param" in result["nested"]
    assert "nested_number" not in result["nested"]
    assert "main_param" not in result

    # Test excluding entire nested config
    result = main_config(exclude_vars=["nested"])
    assert "main_param" in result
    assert "nested" not in result


def test_exclude_vars_with_nested_propagation():
    @config
    def deep_nested_config(hp: HP):
        deep_param = hp.select(["deep1", "deep2"], default="deep1")
        deep_number = hp.number(2.0)

    deep_nested_config.save("tests/helper_configs/deep_nested_config.py")

    @config
    def middle_config(hp: HP):
        deep = hp.propagate("tests/helper_configs/deep_nested_config.py", name="deep")
        middle_param = hp.select(["mid1", "mid2"], default="mid1")

    middle_config.save("tests/helper_configs/middle_config.py")

    @config
    def main_config(hp: HP):
        middle = hp.propagate("tests/helper_configs/middle_config.py", name="middle")
        main_param = hp.select(["x", "y"], default="x")

    # Test excluding deeply nested variables
    result = main_config(exclude_vars=["middle.deep.deep_number"])
    assert "deep_param" in result["middle"]["deep"]
    assert "deep_number" not in result["middle"]["deep"]
    assert "middle_param" in result["middle"]
    assert "main_param" in result

    # Test excluding middle level config
    result = main_config(exclude_vars=["middle.middle_param"])
    assert "deep" in result["middle"]
    assert "middle_param" not in result["middle"]
    assert "main_param" in result


def test_exclude_vars_in_propagate_call():
    @config
    def nested_config(hp: HP):
        nested_param = hp.select(["a", "b"], default="a")
        nested_number = hp.number(1.0)
        extra_param = hp.select(["x", "y"], default="x")

    nested_config.save("tests/helper_configs/nested_config.py")

    @config
    def main_config(hp: HP):
        nested = hp.propagate(
            "tests/helper_configs/nested_config.py",
            name="nested",
            exclude_vars=["extra_param"],  # Exclude within propagate call
        )
        main_param = hp.select(["x", "y"], default="x")

    # Test excluding variables in propagate call
    result = main_config()
    assert "nested_param" in result["nested"]
    assert "nested_number" in result["nested"]
    assert "extra_param" not in result["nested"]
    assert "main_param" in result
