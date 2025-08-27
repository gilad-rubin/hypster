#!/usr/bin/env python3
"""Test multi-value parameter types."""

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


def test_multi_int():
    """Test hp.multi_int for lists of integers."""

    def config(hp: HP) -> list:
        return hp.multi_int([1, 2, 3], name="values", min=0, max=10)

    result = instantiate(config)
    assert result == [1, 2, 3]

    result = instantiate(config, values={"values": [5, 6, 7]})
    assert result == [5, 6, 7]

    # Bounds validation on each element
    try:
        instantiate(config, values={"values": [5, 15, 7]})
        assert False, "Should have raised error"
    except ValueError as e:
        assert "exceeds maximum" in str(e)


def test_multi_float():
    """Test hp.multi_float for lists of floats."""

    def config(hp: HP) -> list:
        return hp.multi_float([1.0, 2.5, 3.7], name="values", min=0.0, max=10.0)

    result = instantiate(config)
    assert result == [1.0, 2.5, 3.7]

    result = instantiate(config, values={"values": [5.5, 6.1, 7.9]})
    assert result == [5.5, 6.1, 7.9]

    # Test strict parameter behavior
    def strict_config(hp: HP) -> list:
        return hp.multi_float([1.0, 2.0], name="values", strict=True)

    # Should reject integers when strict=True
    try:
        instantiate(strict_config, values={"values": [1, 2]})
        assert False, "Should have raised error"
    except ValueError as e:
        assert "expected float but got int" in str(e)

    # Should accept integers when strict=False (default)
    def flexible_config(hp: HP) -> list:
        return hp.multi_float([1.0, 2.0], name="values")

    result = instantiate(flexible_config, values={"values": [1, 2]})
    assert result == [1.0, 2.0]  # Converted to floats


def test_multi_text():
    """Test hp.multi_text for lists of strings."""

    def config(hp: HP) -> list:
        return hp.multi_text(["hello", "world"], name="greetings")

    result = instantiate(config)
    assert result == ["hello", "world"]

    result = instantiate(config, values={"greetings": ["hi", "there"]})
    assert result == ["hi", "there"]

    # Type validation
    try:
        instantiate(config, values={"greetings": ["hello", 123]})
        assert False, "Should have raised error"
    except ValueError as e:
        assert "expected string" in str(e)


def test_multi_bool():
    """Test hp.multi_bool for lists of booleans."""

    def config(hp: HP) -> list:
        return hp.multi_bool([True, False], name="flags")

    result = instantiate(config)
    assert result == [True, False]

    result = instantiate(config, values={"flags": [False, True, False]})
    assert result == [False, True, False]

    # Type validation
    try:
        instantiate(config, values={"flags": [True, "false"]})
        assert False, "Should have raised error"
    except ValueError as e:
        assert "expected boolean" in str(e)


def test_multi_select():
    """Test hp.multi_select for lists of selections."""

    def config(hp: HP) -> list:
        return hp.multi_select(["a", "b", "c"], name="choices", default=["a"])

    result = instantiate(config)
    assert result == ["a"]

    result = instantiate(config, values={"choices": ["b", "c"]})
    assert result == ["b", "c"]

    # options_only enforcement
    def strict_config(hp: HP) -> list:
        return hp.multi_select(["a", "b"], name="choices", default=["a"], options_only=True)

    try:
        instantiate(strict_config, values={"choices": ["a", "invalid"]})
        assert False, "Should have raised error"
    except ValueError as e:
        assert "not in allowed options" in str(e)


def main():
    """Run all test functions."""
    print("Running multi-parameter tests...")

    tests = [
        (test_multi_int, "test_multi_int"),
        (test_multi_float, "test_multi_float"),
        (test_multi_text, "test_multi_text"),
        (test_multi_bool, "test_multi_bool"),
        (test_multi_select, "test_multi_select"),
    ]

    passed = 0
    total = len(tests)

    for test_func, test_name in tests:
        if run_test_function(test_func, test_name):
            passed += 1

    print(f"\nResults: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All multi-parameter tests passed!")
        return 0
    else:
        print("❌ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
