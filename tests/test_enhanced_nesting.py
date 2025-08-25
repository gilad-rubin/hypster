"""Tests for enhanced nesting functionality with registry support."""

import os
import tempfile

import pytest

from hypster import HP, config, registry


def test_nest_from_registry():
    """Test nesting configurations from the registry."""
    registry.clear()

    # Register a nested configuration
    @config(register="models.test_model")
    def model_config(hp: HP):
        hidden_size = hp.int(default=768, min=100, max=1000, name="hidden_size")
        layers = hp.int(default=12, min=6, max=24, name="layers")

    # Use it in another configuration
    @config
    def main_config(hp: HP):
        model = hp.nest("models.test_model", name="model")
        batch_size = hp.int(default=32, min=16, max=128, name="batch_size")

    # Test instantiation
    result = main_config()
    assert "model" in result
    assert "batch_size" in result
    assert "hidden_size" in result["model"]
    assert "layers" in result["model"]


def test_dynamic_config_selection():
    """Test dynamic configuration selection based on HP values."""
    registry.clear()

    # Register multiple models
    @config(register="models.small")
    def small_model(hp: HP):
        size = hp.int(default=256, min=100, max=500, name="size")

    @config(register="models.large")
    def large_model(hp: HP):
        size = hp.int(default=2048, min=1000, max=5000, name="size")  # Configuration that dynamically selects model

    @config
    def experiment_config(hp: HP):
        model_type = hp.select(["small", "large"], default="small", name="model_type")
        model = hp.nest(f"models.{model_type}", name="model")

    # Test with small model
    result_small = experiment_config(values={"model_type": "small"})
    assert result_small["model_type"] == "small"
    assert "model" in result_small

    # Test with large model
    result_large = experiment_config(values={"model_type": "large"})
    assert result_large["model_type"] == "large"
    assert "model" in result_large


def test_nest_file_path():
    """Test nesting from file paths."""
    # Create a temporary config file
    config_content = """from hypster import HP

def temp_config(hp: HP):
    value = hp.int(default=50, min=1, max=100, name="value")
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(config_content)
        temp_path = f.name

    try:
        # Test loading from file path
        @config
        def main_config(hp: HP):
            nested = hp.nest(temp_path, name="nested")
            other = hp.text("default", name="other")

        result = main_config()
        assert "nested" in result
        assert "other" in result
        assert "value" in result["nested"]

    finally:
        os.unlink(temp_path)


def test_nest_specific_function_from_file():
    """Test nesting specific function from file with colon notation."""
    # Create a file with multiple configs
    config_content = """from hypster import HP

def config_a(hp: HP):
    value_a = hp.int(default=5, min=1, max=10, name="value_a")

def config_b(hp: HP):
    value_b = hp.int(default=25, min=20, max=30, name="value_b")

def not_a_config():
    return "not a config"
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(config_content)
        temp_path = f.name

    try:
        # Test loading specific function
        @config
        def main_config(hp: HP):
            nested_a = hp.nest(f"{temp_path}:config_a", name="nested_a")
            nested_b = hp.nest(f"{temp_path}:config_b", name="nested_b")

        result = main_config()
        assert "nested_a" in result
        assert "nested_b" in result
        assert "value_a" in result["nested_a"]
        assert "value_b" in result["nested_b"]

    finally:
        os.unlink(temp_path)


def test_registry_resolution_order():
    """Test the resolution order: registry first, then file loading."""
    registry.clear()

    # Register a config in registry
    @config(register="test.priority")
    def registry_config(hp: HP):
        source = hp.text("registry", name="source")

    # Create a file with same name (should not be used)
    config_content = """from hypster import HP

def file_config(hp: HP):
    source = hp.text("file", name="source")
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(config_content)
        temp_path = f.name

    try:

        @config
        def main_config(hp: HP):
            # This should use registry, not file
            nested = hp.nest("test.priority", name="nested")

        result = main_config()
        assert result["nested"]["source"] == "registry"

    finally:
        os.unlink(temp_path)


def test_helpful_error_messages():
    """Test that helpful error messages are provided for resolution failures."""
    registry.clear()

    # Register some configs for suggestions
    @config(register="models.bert")
    def bert_config(hp: HP):
        size = hp.int(default=768, min=100, max=1000, name="size")

    @config(register="models.gpt")
    def gpt_config(hp: HP):
        layers = hp.int(default=12, min=6, max=48, name="layers")

    @config
    def main_config(hp: HP):
        # This should fail with helpful message
        model = hp.nest("models.bert_base", name="model")  # Typo: should be "models.bert"

    with pytest.raises(ValueError) as exc_info:
        main_config()

    error_msg = str(exc_info.value)
    assert "Could not resolve configuration target" in error_msg
    assert "models.bert_base" in error_msg
    # Should suggest close matches
    assert "models.bert" in error_msg or "Available keys" in error_msg


def test_module_import_loading():
    """Test loading from module imports (if available)."""
    # This test would need an actual importable module
    # For now, test the error handling for missing modules

    @config
    def main_config(hp: HP):
        model = hp.nest("nonexistent.module:config", name="model")

    with pytest.raises(ValueError) as exc_info:
        main_config()

    error_msg = str(exc_info.value)
    assert "Could not resolve configuration target" in error_msg
    assert "nonexistent.module:config" in error_msg


def test_invalid_file_path():
    """Test error handling for invalid file paths."""

    @config
    def main_config(hp: HP):
        model = hp.nest("/nonexistent/path.py", name="model")

    with pytest.raises(ValueError) as exc_info:
        main_config()

    error_msg = str(exc_info.value)
    assert "Could not resolve configuration target" in error_msg


def test_missing_object_in_file():
    """Test error when specific object doesn't exist in file."""
    config_content = """from hypster import HP

def existing_config(hp: HP):
    value = hp.int(1, 10, name="value")
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(config_content)
        temp_path = f.name

    try:

        @config
        def main_config(hp: HP):
            model = hp.nest(f"{temp_path}:nonexistent_config", name="model")

        with pytest.raises(ValueError) as exc_info:
            main_config()

        error_msg = str(exc_info.value)
        assert "Object 'nonexistent_config' not found" in error_msg

    finally:
        os.unlink(temp_path)


def test_complex_nesting_scenario():
    """Test a complex scenario with multiple levels of nesting."""
    registry.clear()

    # Base components
    @config(register="components.encoder")
    def encoder_config(hp: HP):
        hidden_size = hp.int(default=512, min=256, max=1024, name="hidden_size")
        num_layers = hp.int(default=8, min=6, max=12, name="num_layers")

    @config(register="components.decoder")
    def decoder_config(hp: HP):
        hidden_size = hp.int(default=512, min=256, max=1024, name="hidden_size")
        num_layers = hp.int(default=6, min=6, max=12, name="num_layers")

    # Model that uses components
    @config(register="models.transformer")
    def transformer_config(hp: HP):
        encoder = hp.nest("components.encoder", name="encoder")
        decoder = hp.nest("components.decoder", name="decoder")
        dropout = hp.number(default=0.1, min=0.0, max=0.5, name="dropout")

    # Training config that uses model
    @config
    def training_config(hp: HP):
        model = hp.nest("models.transformer", name="model")
        learning_rate = hp.number(default=1e-3, min=1e-5, max=1e-2, name="learning_rate")
        batch_size = hp.int(default=32, min=16, max=128, name="batch_size")

    result = training_config()

    # Verify nested structure
    assert "model" in result
    assert "learning_rate" in result
    assert "batch_size" in result

    model = result["model"]
    assert "encoder" in model
    assert "decoder" in model
    assert "dropout" in model

    encoder = model["encoder"]
    assert "hidden_size" in encoder
    assert "num_layers" in encoder


if __name__ == "__main__":
    pytest.main([__file__])
