#!/usr/bin/env python3
"""Test explicit naming requirement."""

import os
import sys

# Add src to path to import hypster
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from hypster import HP, instantiate


def test_missing_name_raises_error():
    """Test that missing name parameter raises error."""

    def config(hp: HP) -> int:
        return hp.int(10)  # Missing name

    try:
        instantiate(config)
        assert False, "Should have raised error"
    except TypeError as e:
        assert "missing 1 required keyword-only argument: 'name'" in str(e)
        print("✓ Missing name correctly raises TypeError")
    except ValueError as e:
        if "requires 'name'" in str(e):
            print("✓ Missing name correctly raises ValueError")
        else:
            raise


def test_name_parameter_is_keyword_only():
    """Test that name must be passed as keyword argument."""

    def config(hp: HP) -> int:
        return hp.int(10, "my_param")  # Positional name not allowed

    try:
        instantiate(config)
        assert False, "Should have raised TypeError"
    except TypeError:
        print("✓ Positional name correctly raises TypeError")


def test_successful_explicit_naming():
    """Test that explicit naming works correctly."""

    def config(hp: HP) -> dict:
        x = hp.int(10, name="x")
        y = hp.float(0.5, name="y")
        return {"x": x, "y": y}

    result = instantiate(config)
    assert result == {"x": 10, "y": 0.5}
    print("✓ Explicit naming works correctly")

    result = instantiate(config, values={"x": 20, "y": 0.8})
    assert result == {"x": 20, "y": 0.8}
    print("✓ Explicit naming with overrides works correctly")


def main():
    print("Testing explicit naming requirements...")

    test_missing_name_raises_error()
    test_name_parameter_is_keyword_only()
    test_successful_explicit_naming()

    print("\n🎉 All explicit naming tests passed!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
