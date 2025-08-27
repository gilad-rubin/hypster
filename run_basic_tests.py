#!/usr/bin/env python3
"""Manual test runner for specific test functions."""

import os
import sys

# Add src to path to import hypster
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from hypster import HP, instantiate


def run_test_function(test_func, test_name):
    """Run a single test function and report results."""
    try:
        test_func()
        print(f"✓ {test_name} passed")
        return True
    except Exception as e:
        print(f"✗ {test_name} failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_int_parameter():
    """Test hp.int with explicit naming."""

    def config(hp: HP) -> dict:
        batch_size = hp.int(32, name="batch_size", min=1, max=128)
        return {"batch_size": batch_size}

    # Default value
    result = instantiate(config)
    assert result == {"batch_size": 32}

    # Override
    result = instantiate(config, values={"batch_size": 64})
    assert result == {"batch_size": 64}

    # Bounds validation
    try:
        instantiate(config, values={"batch_size": 256})
        assert False, "Should have raised error"
    except ValueError as e:
        assert "exceeds maximum" in str(e)


def test_float_parameter():
    """Test hp.float with type enforcement."""

    def config(hp: HP) -> float:
        return hp.float(0.5, name="lr", min=0.0, max=1.0)

    # Default
    result = instantiate(config)
    assert result == 0.5

    # Override with float
    result = instantiate(config, values={"lr": 0.7})
    assert result == 0.7

    # Accept integer when strict=False (default)
    result = instantiate(config, values={"lr": 1})
    assert result == 1.0  # Converted to float

    # Test strict=True behavior
    def strict_config(hp: HP) -> float:
        return hp.float(0.5, name="lr", min=0.0, max=1.0, strict=True)

    # Should reject integer when strict=True
    try:
        instantiate(strict_config, values={"lr": 1})
        assert False, "Should have raised error"
    except ValueError as e:
        assert "expected float but got int" in str(e)


def test_int_parameter_with_strict():
    """Test hp.int with strict parameter."""

    def config(hp: HP) -> int:
        return hp.int(32, name="batch_size", min=1, max=128)

    # Accept float that can be converted when strict=False (default)
    result = instantiate(config, values={"batch_size": 64.0})
    assert result == 64  # Converted to int

    # Reject float that loses precision
    try:
        instantiate(config, values={"batch_size": 64.5})
        assert False, "Should have raised error"
    except ValueError as e:
        assert "would lose precision" in str(e)

    # Test strict=True behavior
    def strict_config(hp: HP) -> int:
        return hp.int(32, name="batch_size", min=1, max=128, strict=True)

    # Should reject float when strict=True
    try:
        instantiate(strict_config, values={"batch_size": 64.0})
        assert False, "Should have raised error"
    except ValueError as e:
        assert "expected int but got float" in str(e)


def test_text_parameter():
    """Test hp.text for string values."""

    def config(hp: HP) -> str:
        return hp.text("hello", name="greeting")

    result = instantiate(config)
    assert result == "hello"

    result = instantiate(config, values={"greeting": "hi"})
    assert result == "hi"

    # Type validation
    try:
        instantiate(config, values={"greeting": 123})
        assert False, "Should have raised error"
    except ValueError as e:
        assert "expected string" in str(e)


def test_bool_parameter():
    """Test hp.bool for boolean values."""

    def config(hp: HP) -> bool:
        return hp.bool(True, name="flag")

    result = instantiate(config)
    assert result is True

    result = instantiate(config, values={"flag": False})
    assert result is False

    # Type validation
    try:
        instantiate(config, values={"flag": "true"})
        assert False, "Should have raised error"
    except ValueError as e:
        assert "expected boolean" in str(e)


def main():
    """Run all test functions."""
    print("Running basic parameters tests...")

    tests = [
        (test_int_parameter, "test_int_parameter"),
        (test_float_parameter, "test_float_parameter"),
        (test_int_parameter_with_strict, "test_int_parameter_with_strict"),
        (test_text_parameter, "test_text_parameter"),
        (test_bool_parameter, "test_bool_parameter"),
    ]

    passed = 0
    total = len(tests)

    for test_func, test_name in tests:
        if run_test_function(test_func, test_name):
            passed += 1

    print(f"\nResults: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All basic parameter tests passed!")
        return 0
    else:
        print("❌ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
