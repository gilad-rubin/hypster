#!/usr/bin/env python3
"""Simple test script to verify our Hypster implementation."""

import os
import sys

# Add src to path to import hypster
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from hypster import HP, instantiate


def test_basic_int():
    """Test basic int parameter."""

    def config(hp: HP) -> dict:
        batch_size = hp.int(32, name="batch_size", min=1, max=128)
        return {"batch_size": batch_size}

    # Test default value
    result = instantiate(config)
    assert result == {"batch_size": 32}, f"Expected {{'batch_size': 32}}, got {result}"

    # Test override
    result = instantiate(config, values={"batch_size": 64})
    assert result == {"batch_size": 64}, f"Expected {{'batch_size': 64}}, got {result}"

    print("✓ Basic int test passed")


def test_basic_float():
    """Test basic float parameter."""

    def config(hp: HP) -> float:
        return hp.float(0.5, name="lr", min=0.0, max=1.0)

    # Test default
    result = instantiate(config)
    assert result == 0.5, f"Expected 0.5, got {result}"

    # Test override
    result = instantiate(config, values={"lr": 0.7})
    assert result == 0.7, f"Expected 0.7, got {result}"

    # Test int conversion (non-strict)
    result = instantiate(config, values={"lr": 1})
    assert result == 1.0, f"Expected 1.0, got {result}"

    print("✓ Basic float test passed")


def test_strict_mode():
    """Test strict type checking."""

    def strict_config(hp: HP) -> float:
        return hp.float(0.5, name="lr", strict=True)

    try:
        instantiate(strict_config, values={"lr": 1})
        assert False, "Should have raised error for int in strict mode"
    except ValueError as e:
        assert "expected float but got int" in str(e), f"Wrong error message: {e}"

    print("✓ Strict mode test passed")


def test_text_parameter():
    """Test text parameter."""

    def config(hp: HP) -> str:
        return hp.text("hello", name="greeting")

    result = instantiate(config)
    assert result == "hello", f"Expected 'hello', got {result}"

    result = instantiate(config, values={"greeting": "hi"})
    assert result == "hi", f"Expected 'hi', got {result}"

    print("✓ Text parameter test passed")


def test_bool_parameter():
    """Test bool parameter."""

    def config(hp: HP) -> bool:
        return hp.bool(True, name="flag")

    result = instantiate(config)
    assert result is True, f"Expected True, got {result}"

    result = instantiate(config, values={"flag": False})
    assert result is False, f"Expected False, got {result}"

    print("✓ Bool parameter test passed")


def test_select_parameter():
    """Test select parameter."""

    def config(hp: HP) -> str:
        return hp.select(["a", "b", "c"], name="choice", default="a")

    result = instantiate(config)
    assert result == "a", f"Expected 'a', got {result}"

    result = instantiate(config, values={"choice": "b"})
    assert result == "b", f"Expected 'b', got {result}"

    print("✓ Select parameter test passed")


def test_missing_name_error():
    """Test that missing name raises error."""

    def config(hp: HP) -> int:
        return hp.int(10)  # Missing name

    try:
        instantiate(config)
        assert False, "Should have raised error for missing name"
    except TypeError as e:
        assert "missing 1 required keyword-only argument: 'name'" in str(e), f"Wrong error message: {e}"

    print("✓ Missing name error test passed")


def test_bounds_validation():
    """Test bounds validation."""

    def config(hp: HP) -> int:
        return hp.int(32, name="batch_size", min=1, max=128)

    try:
        instantiate(config, values={"batch_size": 256})
        assert False, "Should have raised error for value out of bounds"
    except ValueError as e:
        assert "exceeds maximum" in str(e), f"Wrong error message: {e}"

    print("✓ Bounds validation test passed")


def main():
    """Run all tests."""
    print("Testing Hypster v2 implementation...")

    try:
        test_basic_int()
        test_basic_float()
        test_strict_mode()
        test_text_parameter()
        test_bool_parameter()
        test_select_parameter()
        test_missing_name_error()
        test_bounds_validation()

        print("\n🎉 All tests passed! Implementation is working correctly.")
        return 0

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
