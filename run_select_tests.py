#!/usr/bin/env python3
"""Test select parameter functionality."""

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


def test_select_parameter():
    """Test hp.select with options."""

    def config(hp: HP) -> str:
        return hp.select(["a", "b", "c"], name="choice", default="a")

    result = instantiate(config)
    assert result == "a"

    result = instantiate(config, values={"choice": "b"})
    assert result == "b"

    # options_only enforcement - default is False
    def strict_config(hp: HP) -> str:
        return hp.select(["a", "b"], name="choice", default="a", options_only=True)

    try:
        instantiate(strict_config, values={"choice": "c"})
        assert False, "Should have raised error"
    except ValueError as e:
        assert "not in allowed options" in str(e)

    # options_only=False allows any value
    def flexible_config(hp: HP) -> str:
        return hp.select(["a", "b"], name="choice", default="a", options_only=False)

    result = instantiate(flexible_config, values={"choice": "custom"})
    assert result == "custom"


def test_select_parameter_with_dict():
    """Test hp.select with dictionary options."""

    def config(hp: HP) -> str:
        return hp.select({"fast": "gpt-4o-mini", "smart": "gpt-4"}, name="model", default="fast")

    # Default value (should return the mapped value)
    result = instantiate(config)
    assert result == "gpt-4o-mini"

    # Override with key
    result = instantiate(config, values={"model": "smart"})
    assert result == "gpt-4"

    # Test options_only=True with dict
    def strict_dict_config(hp: HP) -> str:
        return hp.select({"fast": "gpt-4o-mini", "smart": "gpt-4"}, name="model", default="fast", options_only=True)

    try:
        instantiate(strict_dict_config, values={"model": "invalid"})
        assert False, "Should have raised error"
    except ValueError as e:
        assert "not in allowed options" in str(e)


def test_select_with_no_default():
    """Test hp.select with no explicit default."""

    def config(hp: HP) -> str:
        return hp.select(["option1", "option2"], name="choice")

    # Should use first option as default
    result = instantiate(config)
    assert result == "option1"


def main():
    """Run all test functions."""
    print("Running select parameter tests...")

    tests = [
        (test_select_parameter, "test_select_parameter"),
        (test_select_parameter_with_dict, "test_select_parameter_with_dict"),
        (test_select_with_no_default, "test_select_with_no_default"),
    ]

    passed = 0
    total = len(tests)

    for test_func, test_name in tests:
        if run_test_function(test_func, test_name):
            passed += 1

    print(f"\nResults: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All select parameter tests passed!")
        return 0
    else:
        print("❌ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
