"""
Tests for simplified save/load round-trip via importable modules.
"""

import os
import tempfile

import pytest

from hypster import HP, config, load, save


def test_save_load_roundtrip_basic():
    """Test saving a config and re-importing it"""

    @config
    def test_config(hp: HP):
        learning_rate = hp.number(0.01, name="learning_rate")
        model_type = hp.select(["rf", "svm"], name="model_type")

        return {"learning_rate": learning_rate, "model_type": model_type}

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        temp_path = f.name

    try:
        # Save the config
        save(test_config, temp_path)

        # Load it back
        loaded_config = load(temp_path)

        # Should be a Hypster instance
        assert hasattr(loaded_config, "__call__")

        # Should work the same as original
        original_result = test_config()
        loaded_result = loaded_config()

        assert original_result == loaded_result
        assert loaded_result["learning_rate"] == 0.01
        assert loaded_result["model_type"] == "rf"

        # Should work with overrides
        original_override = test_config(values={"learning_rate": 0.05})
        loaded_override = loaded_config(values={"learning_rate": 0.05})

        assert original_override == loaded_override
        assert loaded_override["learning_rate"] == 0.05

    finally:
        # Clean up
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_saved_file_is_importable():
    """Test that saved files are valid Python modules that can be imported"""

    @config
    def my_config(hp: HP):
        param = hp.number(0.123, name="param")
        return {"value": param}

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        temp_path = f.name

    try:
        # Save the config
        save(my_config, temp_path)

        # Read the saved file and check it's valid Python
        with open(temp_path, "r") as f:
            saved_content = f.read()

        # Should contain necessary imports
        assert "from hypster import" in saved_content or "import hypster" in saved_content

        # Should contain the original function
        assert "def my_config" in saved_content

        # Should be valid Python (compile doesn't raise)
        compile(saved_content, temp_path, "exec")

        # Should be importable via importlib
        import importlib.util

        spec = importlib.util.spec_from_file_location("test_module", temp_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Should have a bound Hypster object
        # The exact attribute name might vary based on implementation
        hypster_attrs = [
            attr for attr in dir(module) if hasattr(getattr(module, attr), "__call__") and not attr.startswith("_")
        ]

        assert len(hypster_attrs) > 0, "No callable Hypster found in saved module"

    finally:
        # Clean up
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_save_load_with_complex_config():
    """Test save/load with more complex configuration"""

    @config
    def complex_config(hp: HP):
        # Multiple parameter types
        learning_rate = hp.number(0.01, name="learning_rate", min=0.001, max=0.1)
        epochs = hp.int(10, name="epochs", min=1, max=100)
        model_type = hp.select(["rf", "svm", "neural_net"], name="model_type")
        use_validation = hp.bool(True, name="use_validation")
        optimizer = hp.text("adam", name="optimizer")

        # Some computation
        effective_lr = learning_rate * (0.5 if model_type == "neural_net" else 1.0)

        return {
            "learning_rate": learning_rate,
            "effective_lr": effective_lr,
            "epochs": epochs,
            "model_type": model_type,
            "use_validation": use_validation,
            "optimizer": optimizer,
        }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        temp_path = f.name

    try:
        # Save and load
        save(complex_config, temp_path)
        loaded_config = load(temp_path)

        # Test default values
        result = loaded_config()
        assert result["learning_rate"] == 0.01
        assert result["epochs"] == 10
        assert result["model_type"] == "rf"
        assert result["use_validation"] is True
        assert result["optimizer"] == "adam"
        assert result["effective_lr"] == 0.01  # No adjustment for rf

        # Test with overrides
        result = loaded_config(
            values={"learning_rate": 0.05, "model_type": "neural_net", "epochs": 20, "use_validation": False}
        )

        assert result["learning_rate"] == 0.05
        assert result["model_type"] == "neural_net"
        assert result["epochs"] == 20
        assert result["use_validation"] is False
        assert result["effective_lr"] == 0.025  # Adjusted for neural_net

    finally:
        # Clean up
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_save_load_preserves_function_logic():
    """Test that saved configs preserve complex function logic"""

    @config
    def logic_config(hp: HP):
        model_type = hp.select(["simple", "complex"], name="model_type")
        base_lr = hp.number(0.01, name="base_lr")

        # Complex logic that should be preserved
        if model_type == "simple":
            layers = hp.int(2, name="simple.layers")
            learning_rate = base_lr
        else:
            layers = hp.int(5, name="complex.layers")
            learning_rate = base_lr * 0.1

        # Some computation
        total_params = layers * 100

        return {
            "model_type": model_type,
            "layers": layers,
            "learning_rate": learning_rate,
            "total_params": total_params,
        }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        temp_path = f.name

    try:
        # Save and load
        save(logic_config, temp_path)
        loaded_config = load(temp_path)

        # Test simple model path
        result = loaded_config(values={"model_type": "simple"})
        assert result["model_type"] == "simple"
        assert result["layers"] == 2
        assert result["learning_rate"] == 0.01
        assert result["total_params"] == 200

        # Test complex model path
        result = loaded_config(values={"model_type": "complex"})
        assert result["model_type"] == "complex"
        assert result["layers"] == 5
        assert result["learning_rate"] == 0.001  # base_lr * 0.1
        assert result["total_params"] == 500

    finally:
        # Clean up
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_save_load_with_external_imports():
    """Test that configs with external imports work after save/load"""

    # Note: This test may need to be adjusted based on how external imports are handled
    @config
    def import_config(hp: HP):
        import math

        angle = hp.number(0.5, name="angle")
        use_degrees = hp.bool(False, name="use_degrees")

        # Use external import
        if use_degrees:
            angle_rad = math.radians(angle)
        else:
            angle_rad = angle

        result = math.sin(angle_rad)

        return {"angle": angle, "use_degrees": use_degrees, "sin_value": result}

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        temp_path = f.name

    try:
        # Save and load
        save(import_config, temp_path)
        loaded_config = load(temp_path)

        # Test that imports work
        result = loaded_config(values={"angle": 1.0, "use_degrees": False})
        import math

        expected_sin = math.sin(1.0)

        assert abs(result["sin_value"] - expected_sin) < 1e-10
        assert result["angle"] == 1.0
        assert result["use_degrees"] is False

    finally:
        # Clean up
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_load_nonexistent_file():
    """Test that loading non-existent file raises clear error"""
    with pytest.raises(Exception) as exc_info:
        load("/nonexistent/path/config.py")

    # Should mention file not found or similar
    error_msg = str(exc_info.value).lower()
    assert "file" in error_msg or "not found" in error_msg or "exist" in error_msg


def test_save_to_directory():
    """Test saving to a directory creates the directory if needed"""
    with tempfile.TemporaryDirectory() as temp_dir:
        nested_path = os.path.join(temp_dir, "configs", "my_config.py")

        @config
        def test_config(hp: HP):
            param = hp.number(0.5, name="param")
            return {"param": param}

        # Save to nested path (directory doesn't exist yet)
        save(test_config, nested_path)

        # Should create the directory and file
        assert os.path.exists(nested_path)

        # Should be loadable
        loaded_config = load(nested_path)
        result = loaded_config()
        assert result["param"] == 0.5
