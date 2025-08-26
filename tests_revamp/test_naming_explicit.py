"""
Tests for explicit naming behavior and error pathways.
Replaces auto-naming tests with explicit naming requirements.
"""

import pytest

from hypster import HP, config


def test_explicit_names_assignment():
    """Test explicit names work at assignment"""

    @config
    def test_config(hp: HP):
        learning_rate = hp.number(0.01, name="cfg.learning_rate")
        model_type = hp.select(["rf", "svm"], name="model.type")

        return hp.collect(locals())

    result = test_config()
    assert result["learning_rate"] == 0.01
    assert result["model_type"] == "rf"


def test_explicit_names_with_values_override():
    """Test explicit names work with values override"""

    @config
    def test_config(hp: HP):
        learning_rate = hp.number(0.01, name="cfg.learning_rate")
        model_type = hp.select(["rf", "svm"], name="model.type")

        return hp.collect(locals())

    # Override using the explicit names
    result = test_config(values={"cfg.learning_rate": 0.05, "model.type": "svm"})

    assert result["learning_rate"] == 0.05
    assert result["model_type"] == "svm"


def test_dotted_names_resolution():
    """Test dotted names work for hierarchical parameters"""

    @config
    def test_config(hp: HP):
        # Use dotted names for hierarchy
        optimizer_lr = hp.number(0.01, name="optimizer.learning_rate")
        optimizer_type = hp.select(["adam", "sgd"], name="optimizer.type")
        model_layers = hp.int(3, name="model.layers")

        return {"optimizer": {"learning_rate": optimizer_lr, "type": optimizer_type}, "model": {"layers": model_layers}}

    result = test_config(values={"optimizer.learning_rate": 0.001, "optimizer.type": "sgd", "model.layers": 5})

    assert result["optimizer"]["learning_rate"] == 0.001
    assert result["optimizer"]["type"] == "sgd"
    assert result["model"]["layers"] == 5


def test_missing_name_raises_error_when_overriding():
    """Test that HP calls without name raise error when trying to override"""

    @config
    def test_config(hp: HP):
        # HP call without name - should work fine by itself
        learning_rate = hp.number(0.01)  # No name parameter

        return {"learning_rate": learning_rate}

    # Should work fine when no override is attempted
    result = test_config()
    assert result["learning_rate"] == 0.01

    # Should raise error when trying to override unnamed parameter
    with pytest.raises(Exception) as exc_info:
        test_config(values={"learning_rate": 0.05})

    # Error should mention naming or missing name
    error_msg = str(exc_info.value).lower()
    assert "name" in error_msg or "unnamed" in error_msg


def test_missing_name_no_error_without_override():
    """Test that HP calls without name work fine when not overriding"""

    @config
    def test_config(hp: HP):
        # Multiple HP calls without names - should work
        learning_rate = hp.number(0.01)
        model_type = hp.select(["rf", "svm"])
        epochs = hp.int(10)

        return {"learning_rate": learning_rate, "model_type": model_type, "epochs": epochs}

    # Should work fine with no values override
    result = test_config()
    assert result["learning_rate"] == 0.01
    assert result["model_type"] == "rf"
    assert result["epochs"] == 10


def test_partial_naming():
    """Test configs where some params have names and others don't"""

    @config
    def test_config(hp: HP):
        # Some with names, some without
        learning_rate = hp.number(0.01, name="learning_rate")
        model_type = hp.select(["rf", "svm"])  # No name
        epochs = hp.int(10, name="epochs")

        return {"learning_rate": learning_rate, "model_type": model_type, "epochs": epochs}

    # Can override only the named parameters
    result = test_config(values={"learning_rate": 0.05, "epochs": 20})

    assert result["learning_rate"] == 0.05
    assert result["model_type"] == "rf"  # Default, can't be overridden
    assert result["epochs"] == 20


def test_override_nonexistent_name_error():
    """Test that trying to override a non-existent name raises error"""

    @config
    def test_config(hp: HP):
        learning_rate = hp.number(0.01, name="learning_rate")
        return {"learning_rate": learning_rate}

    # Try to override a parameter that doesn't exist
    with pytest.raises(Exception) as exc_info:
        test_config(values={"nonexistent_param": 0.05})

    # Should mention that the parameter doesn't exist
    error_msg = str(exc_info.value).lower()
    assert "nonexistent_param" in error_msg or "not found" in error_msg


def test_complex_dotted_names():
    """Test complex hierarchical naming patterns"""

    @config
    def test_config(hp: HP):
        # Complex nested naming
        model_cnn_filters = hp.int(32, name="model.cnn.filters")
        model_cnn_kernel = hp.int(3, name="model.cnn.kernel_size")
        model_dense_units = hp.int(128, name="model.dense.units")
        optim_lr = hp.number(0.01, name="optimizer.learning_rate")
        optim_decay = hp.number(0.0001, name="optimizer.weight_decay")

        return {
            "model": {
                "cnn": {"filters": model_cnn_filters, "kernel_size": model_cnn_kernel},
                "dense": {"units": model_dense_units},
            },
            "optimizer": {"learning_rate": optim_lr, "weight_decay": optim_decay},
        }

    result = test_config(values={"model.cnn.filters": 64, "model.dense.units": 256, "optimizer.learning_rate": 0.001})

    assert result["model"]["cnn"]["filters"] == 64
    assert result["model"]["cnn"]["kernel_size"] == 3  # Default
    assert result["model"]["dense"]["units"] == 256
    assert result["optimizer"]["learning_rate"] == 0.001
    assert result["optimizer"]["weight_decay"] == 0.0001  # Default


def test_name_validation():
    """Test that names are properly validated"""

    @config
    def test_config(hp: HP):
        # Valid names
        param1 = hp.number(0.01, name="valid_name")
        param2 = hp.number(0.02, name="valid.dotted.name")
        param3 = hp.number(0.03, name="valid-dashed-name")

        return {"param1": param1, "param2": param2, "param3": param3}

    result = test_config(values={"valid_name": 0.1, "valid.dotted.name": 0.2, "valid-dashed-name": 0.3})

    assert result["param1"] == 0.1
    assert result["param2"] == 0.2
    assert result["param3"] == 0.3
