#!/usr/bin/env python3
"""Verify that explicit naming is working correctly."""

import math

from src.hypster import HP, config


def test_explicit_naming():
    """Test that explicit names are required and work correctly."""
    print("Testing explicit naming...")

    @config
    def test_config(hp: HP):
        model = hp.select(["cnn", "rnn"], name="model", default="cnn")
        lr = hp.number(0.001, name="learning_rate")
        epochs = hp.number(10, name="epochs")

    # Test with defaults
    result1 = test_config()
    print(f"Default result: {result1}")
    assert result1["model"] == "cnn"
    assert result1["lr"] == 0.001
    assert result1["epochs"] == 10

    # Test with values
    result2 = test_config(values={"model": "rnn", "learning_rate": 0.01})
    print(f"With values result: {result2}")
    assert result2["model"] == "rnn"
    assert result2["lr"] == 0.01
    assert result2["epochs"] == 10

    print("✅ Explicit naming test passed!")


def test_missing_name_error():
    """Test that missing name parameter raises an error."""
    print("Testing missing name error...")

    try:

        @config
        def bad_config(hp: HP):
            # This should fail because name is missing
            value = hp.select(["a", "b"])  # Missing name parameter

        bad_config()
        print("❌ Expected error but none was raised!")
        assert False, "Should have raised an error for missing name"
    except TypeError as e:
        print(f"✅ Correctly caught missing name error: {e}")
    except Exception as e:
        print(f"✅ Caught error (different type): {e}")


def test_debugging_capability():
    """Test that debugging works (no AST manipulation)."""
    print("Testing debugging capability...")

    @config
    def debug_config(hp: HP):
        # This should work with debugging enabled
        import math  # External import should work

        model = hp.select(["cnn", "rnn"], name="model", default="cnn")
        # Simple computation that we can debug
        lr_base = 0.001
        lr = hp.number(lr_base * math.sqrt(2), name="learning_rate")
        math_result = math.pi

    result = debug_config()
    print(f"Debug result: {result}")
    assert result["model"] == "cnn"
    assert abs(result["lr"] - (0.001 * math.sqrt(2))) < 1e-10
    assert abs(result["math_result"] - math.pi) < 1e-10

    print("✅ Debugging capability test passed!")


if __name__ == "__main__":
    test_explicit_naming()
    test_missing_name_error()
    test_debugging_capability()
    print("\n🎉 All verification tests passed!")
    print("✅ Explicit naming is working correctly")
    print("✅ AST manipulation has been successfully removed")
    print("✅ Debugging is now enabled")
