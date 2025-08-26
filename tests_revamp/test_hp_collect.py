"""
Tests for the hp.collect locals collection helper.
"""

from hypster import HP, config


def test_hp_collect_full_collection():
    """Test hp.collect removes noise and keeps data objects"""

    @config
    def test_config(hp: HP):
        # Valid variables that should be included
        learning_rate = hp.number(0.01, name="learning_rate")
        model_type = hp.select(["rf", "svm"], name="model_type")
        plain_var = "some_value"
        number_var = 42

        # Variables that should be excluded
        def some_function():
            return "test"

        import os

        class SomeClass:
            pass

        _private_var = "private"
        __dunder_var__ = "dunder"

        # Test hp.collect removes noise
        collected = hp.collect(locals())

        return {
            "collected": collected,
            "all_vars": {
                "learning_rate": learning_rate,
                "model_type": model_type,
                "plain_var": plain_var,
                "number_var": number_var,
                "some_function": some_function,
                "os": os,
                "SomeClass": SomeClass,
                "_private_var": _private_var,
                "__dunder_var__": __dunder_var__,
            },
        }

    result = test_config()
    collected = result["collected"]

    # Should include data variables
    assert "learning_rate" in collected
    assert "model_type" in collected
    assert "plain_var" in collected
    assert "number_var" in collected

    # Should exclude noise
    assert "hp" not in collected
    assert "some_function" not in collected
    assert "os" not in collected
    assert "SomeClass" not in collected
    assert "_private_var" not in collected
    assert "__dunder_var__" not in collected

    # Check values are correct
    assert collected["learning_rate"] == 0.01
    assert collected["model_type"] == "rf"
    assert collected["plain_var"] == "some_value"
    assert collected["number_var"] == 42


def test_hp_collect_include_only():
    """Test hp.collect with include parameter"""

    @config
    def test_config(hp: HP):
        learning_rate = hp.number(0.01, name="learning_rate")
        model_type = hp.select(["rf", "svm"], name="model_type")
        epochs = hp.int(10, name="epochs")
        extra_var = "should_not_be_included"

        # Only include specific variables
        collected = hp.collect(locals(), include=["learning_rate", "epochs"])

        return collected

    result = test_config()

    # Should only include specified variables
    assert "learning_rate" in result
    assert "epochs" in result

    # Should exclude others, even if they would normally be included
    assert "model_type" not in result
    assert "extra_var" not in result

    # Check values
    assert result["learning_rate"] == 0.01
    assert result["epochs"] == 10


def test_hp_collect_exclude_only():
    """Test hp.collect with exclude parameter"""

    @config
    def test_config(hp: HP):
        learning_rate = hp.number(0.01, name="learning_rate")
        model_type = hp.select(["rf", "svm"], name="model_type")
        epochs = hp.int(10, name="epochs")
        temp_var = "temporary"

        # Exclude specific variables
        collected = hp.collect(locals(), exclude=["temp_var", "epochs"])

        return collected

    result = test_config()

    # Should include non-excluded data variables
    assert "learning_rate" in result
    assert "model_type" in result

    # Should exclude specified variables
    assert "temp_var" not in result
    assert "epochs" not in result

    # Check values
    assert result["learning_rate"] == 0.01
    assert result["model_type"] == "rf"


def test_hp_collect_include_and_exclude():
    """Test hp.collect with both include and exclude parameters"""

    @config
    def test_config(hp: HP):
        learning_rate = hp.number(0.01, name="learning_rate")
        model_type = hp.select(["rf", "svm"], name="model_type")
        epochs = hp.int(10, name="epochs")
        batch_size = hp.int(32, name="batch_size")

        # Include specific vars but exclude some of them
        collected = hp.collect(
            locals(), include=["learning_rate", "model_type", "epochs", "batch_size"], exclude=["epochs"]
        )

        return collected

    result = test_config()

    # Should include vars in include list but not in exclude list
    assert "learning_rate" in result
    assert "model_type" in result
    assert "batch_size" in result

    # Should exclude even if in include list
    assert "epochs" not in result


def test_hp_collect_with_hp_calls_results():
    """Test that hp.collect works with results of HP calls"""

    @config
    def test_config(hp: HP):
        # HP call results should be included
        learning_rate = hp.number(0.01, name="learning_rate")
        selected_option = hp.select(["option1", "option2"], name="selected")

        # Mix with plain Python variables
        computed_value = learning_rate * 2
        string_value = f"lr_{learning_rate}"

        collected = hp.collect(locals())
        return collected

    result = test_config()

    # Should include HP call results
    assert "learning_rate" in result
    assert "selected_option" in result

    # Should include computed values
    assert "computed_value" in result
    assert "string_value" in result

    # Check values are correct
    assert result["learning_rate"] == 0.01
    assert result["selected_option"] == "option1"
    assert result["computed_value"] == 0.02
    assert result["string_value"] == "lr_0.01"


def test_hp_collect_empty_locals():
    """Test hp.collect with minimal locals"""

    @config
    def test_config(hp: HP):
        # Just collect with minimal content
        collected = hp.collect(locals())
        return collected

    result = test_config()

    # Should be empty or contain only safe variables (no hp, no noise)
    assert "hp" not in result

    # Should be a dict
    assert isinstance(result, dict)


def test_hp_collect_include_nonexistent():
    """Test hp.collect include with non-existent variables"""

    @config
    def test_config(hp: HP):
        learning_rate = hp.number(0.01, name="learning_rate")

        # Include a variable that doesn't exist
        collected = hp.collect(locals(), include=["learning_rate", "nonexistent_var"])
        return collected

    result = test_config()

    # Should include existing variables, ignore non-existent ones
    assert "learning_rate" in result
    assert "nonexistent_var" not in result
    assert result["learning_rate"] == 0.01
