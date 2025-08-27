#!/usr/bin/env python3
"""Comprehensive test demonstrating full Hypster v2 capabilities."""

import os
import sys

# Add src to path to import hypster
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from hypster import HP, instantiate


def optimizer_config(hp: HP) -> dict:
    """Nested configuration for optimizer settings."""
    optimizer_type = hp.select(["adam", "sgd", "rmsprop"], name="type", default="adam")
    learning_rate = hp.float(0.001, name="lr", min=1e-6, max=1.0)

    if optimizer_type == "adam":
        beta1 = hp.float(0.9, name="beta1", min=0.0, max=1.0)
        beta2 = hp.float(0.999, name="beta2", min=0.0, max=1.0)
        return {"type": optimizer_type, "lr": learning_rate, "beta1": beta1, "beta2": beta2}
    elif optimizer_type == "sgd":
        momentum = hp.float(0.9, name="momentum", min=0.0, max=1.0)
        return {"type": optimizer_type, "lr": learning_rate, "momentum": momentum}
    else:  # rmsprop
        alpha = hp.float(0.99, name="alpha", min=0.0, max=1.0)
        return {"type": optimizer_type, "lr": learning_rate, "alpha": alpha}


def model_config(hp: HP) -> dict:
    """Nested configuration for model architecture."""
    model_type = hp.select(["cnn", "transformer"], name="type", default="cnn")

    if model_type == "cnn":
        layers = hp.multi_int([32, 64, 128], name="layers", min=1, max=512)
        dropout = hp.float(0.1, name="dropout", min=0.0, max=0.5)
        return {"type": model_type, "layers": layers, "dropout": dropout}
    else:  # transformer
        num_heads = hp.int(8, name="num_heads", min=1, max=32)
        hidden_size = hp.int(512, name="hidden_size", min=64, max=2048)
        num_layers = hp.int(6, name="num_layers", min=1, max=24)
        return {"type": model_type, "num_heads": num_heads, "hidden_size": hidden_size, "num_layers": num_layers}


def training_config(hp: HP) -> dict:
    """Main configuration combining model and optimizer."""
    # Basic training parameters
    batch_size = hp.int(32, name="batch_size", min=1, max=512)
    epochs = hp.int(10, name="epochs", min=1, max=1000)

    # Text parameters
    experiment_name = hp.text("experiment_1", name="experiment_name")
    dataset_path = hp.text("/data/train", name="dataset_path")

    # Boolean flags
    use_cuda = hp.bool(True, name="use_cuda")
    save_checkpoints = hp.bool(True, name="save_checkpoints")

    # Multi-select for data augmentation
    augmentations = hp.multi_select(
        ["rotation", "flip", "crop", "noise"], name="augmentations", default=["rotation", "flip"], options_only=True
    )

    # Multi-value parameters
    validation_splits = hp.multi_float([0.8, 0.1, 0.1], name="splits", min=0.0, max=1.0)

    # Nested configurations
    model = hp.nest(model_config, name="model")
    optimizer = hp.nest(optimizer_config, name="optimizer")

    # Use hp.collect for convenient result gathering
    return hp.collect(locals())


def test_default_configuration():
    """Test with all default values."""
    print("=== Testing default configuration ===")

    result = instantiate(training_config)

    # Verify structure
    assert "model" in result
    assert "optimizer" in result
    assert result["batch_size"] == 32
    assert result["epochs"] == 10
    assert result["use_cuda"] is True
    assert result["augmentations"] == ["rotation", "flip"]

    print(f"✓ Default configuration: {result}")


def test_override_basic_parameters():
    """Test overriding basic parameters."""
    print("\n=== Testing basic parameter overrides ===")

    values = {"batch_size": 64, "epochs": 20, "experiment_name": "test_experiment", "use_cuda": False}

    result = instantiate(training_config, values=values)

    assert result["batch_size"] == 64
    assert result["epochs"] == 20
    assert result["experiment_name"] == "test_experiment"
    assert result["use_cuda"] is False

    print(f"✓ Basic overrides work: batch_size={result['batch_size']}, epochs={result['epochs']}")


def test_override_nested_parameters():
    """Test overriding nested configuration parameters."""
    print("\n=== Testing nested parameter overrides ===")

    values = {
        "model.type": "transformer",
        "model.num_heads": 16,
        "model.hidden_size": 1024,
        "optimizer.type": "sgd",
        "optimizer.lr": 0.01,
        "optimizer.momentum": 0.8,
    }

    result = instantiate(training_config, values=values)

    assert result["model"]["type"] == "transformer"
    assert result["model"]["num_heads"] == 16
    assert result["model"]["hidden_size"] == 1024
    assert result["optimizer"]["type"] == "sgd"
    assert result["optimizer"]["lr"] == 0.01
    assert result["optimizer"]["momentum"] == 0.8

    print(f"✓ Nested overrides work: model={result['model']['type']}, optimizer={result['optimizer']['type']}")


def test_multi_value_parameters():
    """Test multi-value parameter overrides."""
    print("\n=== Testing multi-value parameter overrides ===")

    values = {
        "model.type": "cnn",
        "model.layers": [64, 128, 256, 512],
        "augmentations": ["rotation", "crop", "noise"],
        "splits": [0.7, 0.2, 0.1],
    }

    result = instantiate(training_config, values=values)

    assert result["model"]["layers"] == [64, 128, 256, 512]
    assert result["augmentations"] == ["rotation", "crop", "noise"]
    assert result["validation_splits"] == [0.7, 0.2, 0.1]  # Variable name is validation_splits

    print(f"✓ Multi-value overrides work: layers={result['model']['layers']}")


def test_conditional_configuration():
    """Test conditional configuration paths."""
    print("\n=== Testing conditional configurations ===")

    # Test CNN path
    cnn_values = {"model.type": "cnn"}
    cnn_result = instantiate(training_config, values=cnn_values)
    assert "layers" in cnn_result["model"]
    assert "dropout" in cnn_result["model"]
    assert "num_heads" not in cnn_result["model"]

    # Test Transformer path
    transformer_values = {"model.type": "transformer"}
    transformer_result = instantiate(training_config, values=transformer_values)
    assert "num_heads" in transformer_result["model"]
    assert "hidden_size" in transformer_result["model"]
    assert "layers" not in transformer_result["model"]

    print("✓ Conditional configurations work correctly")


def test_error_handling():
    """Test that validation errors are properly caught."""
    print("\n=== Testing error handling ===")

    # Test bounds validation
    try:
        instantiate(training_config, values={"batch_size": 1000})  # Exceeds max
        assert False, "Should have raised error"
    except ValueError as e:
        assert "exceeds maximum" in str(e)
        print("✓ Bounds validation works")

    # Test type validation
    try:
        instantiate(training_config, values={"use_cuda": "true"})  # String instead of bool
        assert False, "Should have raised error"
    except ValueError as e:
        assert "expected boolean" in str(e)
        print("✓ Type validation works")

    # Test unknown parameters
    import warnings

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        instantiate(training_config, values={"unknown_param": 1})
        assert len(w) > 0
        print("✓ Unknown parameter warning works")


def main():
    """Run comprehensive test suite."""
    print("🚀 Running comprehensive Hypster v2 test suite...")

    test_default_configuration()
    test_override_basic_parameters()
    test_override_nested_parameters()
    test_multi_value_parameters()
    test_conditional_configuration()
    test_error_handling()

    print("\n" + "=" * 60)
    print("🎉 ALL COMPREHENSIVE TESTS PASSED!")
    print("Hypster v2 implementation is complete and fully functional!")
    print("=" * 60)

    # Show final example
    print("\n📋 Example configuration:")
    example_values = {
        "batch_size": 128,
        "epochs": 50,
        "experiment_name": "production_model",
        "model.type": "transformer",
        "model.num_heads": 12,
        "model.hidden_size": 768,
        "optimizer.type": "adam",
        "optimizer.lr": 2e-4,
        "augmentations": ["rotation", "flip", "crop"],
    }

    result = instantiate(training_config, values=example_values)
    print("Values:", example_values)
    print("Result:", result)

    return 0


if __name__ == "__main__":
    sys.exit(main())
