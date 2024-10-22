import pytest

from hypster import HP, config


def test_case_0():
    @config
    def config_func(hp: HP):
        var = hp.select(["option1", "option2"], name="var", default="option1")

    result = config_func()
    assert result["var"] == "option1"
    result = config_func(selections={"var": "option2"})
    assert result["var"] == "option2"
    result = config_func(overrides={"var": "option2"})
    assert result["var"] == "option2"


def test_case_1():
    @config
    def config_func(hp: HP):
        var = hp.select(["option1", "option2"], name="var")

    with pytest.raises(ValueError):
        config_func()
    result = config_func(selections={"var": "option2"})
    assert result["var"] == "option2"
    result = config_func(overrides={"var": "option2"})
    assert result["var"] == "option2"


def test_case_2():
    @config
    def config_func(hp: HP):
        if hp.select(["condition1", "condition2"], name="condition", default="condition1") == "condition1":
            var = hp.select(["option1", "option2"], name="var", default="option1")
        else:
            var = hp.select(["option3", "option2"], name="var", default="option3")

    result = config_func()
    assert result["var"] == "option1"
    result = config_func(selections={"var": "option2"})
    assert result["var"] == "option2"
    result = config_func(overrides={"var": "option2"})
    assert result["var"] == "option2"


def test_case_3():
    @config
    def config_func(hp: HP):
        if hp.select(["condition1", "condition2"], name="condition") == "condition1":
            var = hp.select(["option1", "option2"], name="var")
        else:
            var = hp.select(["option3", "option2"], name="var")

    with pytest.raises(ValueError):
        config_func()
    result = config_func(selections={"condition": "condition1", "var": "option2"})
    assert result["var"] == "option2"
    result = config_func(overrides={"condition": "condition1", "var": "option2"})
    assert result["var"] == "option2"


def test_case_4():
    @config
    def nested_func(hp):
        var = hp.select(["option1", "option2"], name="var", default="option1")

    nested_func.save("tests/nested_func.py")

    @config
    def config_func(hp: HP):
        result = hp.propagate("tests/nested_func.py")

    result = config_func()
    assert result["result"]["var"] == "option1"
    result = config_func(selections={"result.var": "option2"})
    assert result["result"]["var"] == "option2"
    result = config_func(overrides={"result.var": "option2"})
    assert result["result"]["var"] == "option2"


def test_case_5():
    @config
    def nested_func(hp):
        var = hp.select(["option1", "option2"], name="var")

    nested_func.save("tests/nested_func.py")

    @config
    def config_func(hp: HP):
        result = hp.propagate("tests/nested_func.py")

    with pytest.raises(ValueError):
        config_func()
    result = config_func(selections={"result.var": "option2"})
    assert result["result"]["var"] == "option2"
    result = config_func(overrides={"result.var": "option2"})
    assert result["result"]["var"] == "option2"


def test_case_6():
    @config
    def config_func(hp: HP):
        var = hp.multi_select(["option1", "option2", "option3"], name="var", default=["option1", "option2"])

    result = config_func()
    assert result["var"] == ["option1", "option2"]
    result = config_func(selections={"var": "option2"})
    assert result["var"] == "option2"
    result = config_func(overrides={"var": "option2"})
    assert result["var"] == "option2"


def test_case_7():
    @config
    def config_func(hp: HP):
        var = hp.multi_select(["option1", "option2", "option3"], name="var")

    with pytest.raises(ValueError):
        config_func()
    result = config_func(selections={"var": "option2"})
    assert result["var"] == "option2"
    result = config_func(overrides={"var": "option2"})
    assert result["var"] == "option2"


def test_case_8():
    @config
    def config_func(hp: HP):
        if hp.select(["condition1", "condition2"], name="condition", default="condition1") == "condition1":
            var = hp.multi_select(["option1", "option2", "option3"], name="var", default=["option1", "option2"])
        else:
            var = hp.multi_select(["option3", "option2", "option3"], name="var", default=["option3", "option2"])

    result = config_func()
    assert result["var"] == ["option1", "option2"]
    result = config_func(selections={"var": "option2"})
    assert result["var"] == "option2"
    result = config_func(overrides={"var": "option2"})
    assert result["var"] == "option2"


def test_case_9():
    @config
    def config_func(hp: HP):
        if hp.select(["condition1", "condition2"], name="condition") == "condition1":
            var = hp.multi_select(["option1", "option2", "option3"], name="var")
        else:
            var = hp.multi_select(["option3", "option2", "option3"], name="var")

    with pytest.raises(ValueError):
        config_func()
    result = config_func(selections={"condition": "condition1", "var": "option2"})
    assert result["var"] == "option2"
    result = config_func(overrides={"condition": "condition1", "var": "option2"})
    assert result["var"] == "option2"


def test_case_10():
    @config
    def nested_func(hp):
        var = hp.multi_select(["option1", "option2", "option3"], name="var", default=["option1", "option2"])

    nested_func.save("tests/nested_func.py")

    @config
    def config_func(hp: HP):
        result = hp.propagate("tests/nested_func.py")

    result = config_func()
    assert result["result"]["var"] == ["option1", "option2"]
    result = config_func(selections={"result.var": "option2"})
    assert result["result"]["var"] == "option2"
    result = config_func(overrides={"result.var": "option2"})
    assert result["result"]["var"] == "option2"


def test_case_11():
    @config
    def nested_func(hp):
        var = hp.multi_select(["option1", "option2", "option3"], name="var")

    nested_func.save("tests/nested_func.py")

    @config
    def config_func(hp: HP):
        result = hp.propagate("tests/nested_func.py")

    with pytest.raises(ValueError):
        config_func()
    result = config_func(selections={"result.var": "option2"})
    assert result["result"]["var"] == "option2"
    result = config_func(overrides={"result.var": "option2"})
    assert result["result"]["var"] == "option2"


def test_case_12():
    @config
    def config_func(hp: HP):
        var = hp.number_input(name="var", default=10)

    result = config_func()
    assert result["var"] == 10
    result = config_func(overrides={"var": 5})
    assert result["var"] == 5


def test_case_13():
    @config
    def config_func(hp: HP):
        if hp.select(["condition1", "condition2"], name="condition", default="condition1") == "condition1":
            var = hp.number_input(name="var", default=10)
        else:
            var = hp.number_input(name="var", default=10)

    result = config_func()
    assert result["var"] == 10
    result = config_func(overrides={"var": 5})
    assert result["var"] == 5


def test_case_14():
    @config
    def nested_func(hp):
        var = hp.number_input(name="var", default=10)

    nested_func.save("tests/nested_func.py")

    @config
    def config_func(hp: HP):
        result = hp.propagate("tests/nested_func.py")

    result = config_func()
    assert result["result"]["var"] == 10
    result = config_func(overrides={"result.var": 5})
    assert result["result"]["var"] == 5
