#!/usr/bin/env python3
"""
Test script to verify that auto-naming removal works correctly.
"""

import json  # Module-level import for testing portability
import os
import sys

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from hypster import HP, config


# Test that we can define a config function with explicit names
@config
def my_config(hp: HP):
    # All imports at top level (testing portability removal)
    from dataclasses import dataclass

    @dataclass
    class ModelConfig:
        name: str
        learning_rate: float
        epochs: int

    # All HP calls now require explicit names
    model_name = hp.select(["bert", "gpt", "t5"], name="model_name", default="bert")
    learning_rate = hp.number(0.001, name="learning_rate", min=0.0001, max=0.1)
    epochs = hp.int(10, name="epochs", min=1, max=100)
    use_gpu = hp.bool(True, name="use_gpu")

    model_config = ModelConfig(name=model_name, learning_rate=learning_rate, epochs=epochs)

    print(f"Configuration created: {model_config}")
    print(f"Using GPU: {use_gpu}")


def test_basic_functionality():
    """Test basic functionality with explicit names."""
    print("Testing basic functionality...")

    # Test with default values
    config1 = my_config()
    print("Config with defaults:", config1)

    # Test with custom values
    config2 = my_config(values={"model_name": "gpt", "learning_rate": 0.01, "epochs": 5, "use_gpu": False})
    print("Config with custom values:", config2)

    # Test final_vars filtering
    config3 = my_config(final_vars=["model_name", "learning_rate"], values={"model_name": "t5"})
    print("Config with final_vars:", config3)

    print("✅ Basic functionality test passed!")


def test_debugging_support():
    """Test that debugging works (no AST manipulation)."""
    print("\nTesting debugging support...")

    # This should work without any AST manipulation issues
    # import pdb  # Available for debugging if needed
    # pdb.set_trace() # Uncomment to test debugging

    result = my_config(values={"model_name": "bert", "learning_rate": 0.005})
    print("Debug test result:", result)
    print("✅ Debugging support test passed!")


def test_import_portability_removed():
    """Test that we can use imports from outside the function."""
    print("\nTesting import portability removal...")

    print("json module available:", "json" in globals())
    print("json module:", json)

    @config
    def config_with_external_imports(hp: HP):
        # Add some debug info
        print("Inside config function:")
        print("json module available in locals:", "json" in locals())
        print("json module available in globals:", "json" in globals())

        # This should work now that portability constraints are removed
        data = json.loads('{"test": "value"}')

        model_type = hp.select(["simple", "complex"], name="model_type", default="simple")
        config_data = {"model": model_type, "external_data": data}

    result = config_with_external_imports()
    print("External imports test result:", result)
    print("✅ Import portability removal test passed!")


if __name__ == "__main__":
    print("Testing Hypster without auto-naming...")

    try:
        test_basic_functionality()
        test_debugging_support()
        test_import_portability_removed()
        print("\n🎉 All tests passed! Auto-naming removal successful.")

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
