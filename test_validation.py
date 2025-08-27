#!/usr/bin/env python3
"""Test error handling and validation."""

import os
import sys

# Add src to path to import hypster
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from hypster import HP, instantiate


def test_bounds_validation():
    """Test bounds validation errors."""

    def config(hp: HP) -> int:
        return hp.int(50, name="value", min=10, max=100)

    # Test exceeding maximum
    try:
        instantiate(config, values={"value": 150})
        assert False, "Should have raised error"
    except ValueError as e:
        assert "exceeds maximum bound" in str(e)
        print("✓ Maximum bound validation works")

    # Test below minimum
    try:
        instantiate(config, values={"value": 5})
        assert False, "Should have raised error"
    except ValueError as e:
        assert "below minimum bound" in str(e)
        print("✓ Minimum bound validation works")


def test_type_validation():
    """Test type validation errors."""

    # Test invalid text type
    def text_config(hp: HP) -> str:
        return hp.text("hello", name="message")

    try:
        instantiate(text_config, values={"message": 123})
        assert False, "Should have raised error"
    except ValueError as e:
        assert "expected string" in str(e)
        print("✓ Text type validation works")

    # Test invalid bool type
    def bool_config(hp: HP) -> bool:
        return hp.bool(True, name="flag")

    try:
        instantiate(bool_config, values={"flag": "true"})
        assert False, "Should have raised error"
    except ValueError as e:
        assert "expected boolean" in str(e)
        print("✓ Bool type validation works")


def test_select_validation():
    """Test select validation with options_only."""

    def config(hp: HP) -> str:
        return hp.select(["a", "b", "c"], name="choice", options_only=True)

    try:
        instantiate(config, values={"choice": "invalid"})
        assert False, "Should have raised error"
    except ValueError as e:
        assert "not in allowed options" in str(e)
        print("✓ Select options_only validation works")


def test_function_signature_validation():
    """Test function signature validation."""

    # Function without hp parameter
    def bad_config():
        return "bad"

    try:
        instantiate(bad_config)
        assert False, "Should have raised error"
    except ValueError as e:
        assert "must have 'hp: HP' as first parameter" in str(e)
        print("✓ Function signature validation works")

    # Function with wrong first parameter name
    def bad_config2(config):
        return "bad"

    try:
        instantiate(bad_config2)
        assert False, "Should have raised error"
    except ValueError as e:
        assert "must have 'hp' as first parameter name" in str(e)
        print("✓ First parameter name validation works")


def test_duplicate_parameter_names():
    """Test that duplicate parameter names in same execution path raise errors."""

    def config(hp: HP) -> dict:
        x = hp.int(10, name="param")
        y = hp.int(20, name="param")  # Duplicate name
        return {"x": x, "y": y}

    try:
        instantiate(config)
        assert False, "Should have raised error"
    except ValueError as e:
        assert "already been defined" in str(e)
        print("✓ Duplicate parameter name validation works")


def test_unknown_parameter_warnings():
    """Test unknown parameter handling."""

    def config(hp: HP) -> int:
        return hp.int(10, name="value")

    import warnings

    # Test warning mode (default)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = instantiate(config, values={"unknown_param": 1})
        assert len(w) > 0
        assert "unknown_param" in str(w[0].message).lower()
        print("✓ Unknown parameter warning works")

    # Test raise mode
    try:
        instantiate(config, values={"unknown_param": 1}, on_unknown="raise")
        assert False, "Should have raised error"
    except ValueError as e:
        assert "Unknown or unreachable parameters" in str(e)
        print("✓ Unknown parameter raise mode works")

    # Test ignore mode
    result = instantiate(config, values={"unknown_param": 1}, on_unknown="ignore")
    assert result == 10  # Should use default
    print("✓ Unknown parameter ignore mode works")


def main():
    print("Testing error handling and validation...")

    test_bounds_validation()
    test_type_validation()
    test_select_validation()
    test_function_signature_validation()
    test_duplicate_parameter_names()
    test_unknown_parameter_warnings()

    print("\n🎉 All validation tests passed!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
