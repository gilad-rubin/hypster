#!/usr/bin/env python3
"""Test hp.collect helper functionality."""

import os
import sys

# Add src to path to import hypster
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from hypster import HP, instantiate


def test_collect_basic():
    """Test basic hp.collect functionality."""

    def config(hp: HP) -> dict:
        batch_size = hp.int(32, name="batch_size")
        learning_rate = hp.float(0.01, name="lr")
        model_name = hp.text("resnet", name="model")

        # Use hp.collect to gather local variables
        return hp.collect(locals())

    result = instantiate(config)
    expected = {"batch_size": 32, "learning_rate": 0.01, "model_name": "resnet"}
    assert result == expected
    print("✓ Basic hp.collect works")


def test_collect_with_include():
    """Test hp.collect with include filter."""

    def config(hp: HP) -> dict:
        batch_size = hp.int(32, name="batch_size")
        learning_rate = hp.float(0.01, name="lr")
        model_name = hp.text("resnet", name="model")
        temp_var = "should not be included"

        # Only include specific variables
        return hp.collect(locals(), include=["batch_size", "learning_rate"])

    result = instantiate(config)
    expected = {
        "batch_size": 32,
        "learning_rate": 0.01,
    }
    assert result == expected
    print("✓ hp.collect with include filter works")


def test_collect_with_exclude():
    """Test hp.collect with exclude filter."""

    def config(hp: HP) -> dict:
        batch_size = hp.int(32, name="batch_size")
        learning_rate = hp.float(0.01, name="lr")
        model_name = hp.text("resnet", name="model")
        temp_var = "should be excluded"

        # Exclude specific variables
        return hp.collect(locals(), exclude=["temp_var"])

    result = instantiate(config)
    # temp_var should be excluded, hp is auto-excluded
    expected = {"batch_size": 32, "learning_rate": 0.01, "model_name": "resnet"}
    assert result == expected
    print("✓ hp.collect with exclude filter works")


def test_collect_excludes_private():
    """Test that hp.collect excludes private/dunder variables."""

    def config(hp: HP) -> dict:
        batch_size = hp.int(32, name="batch_size")
        _private_var = "should be excluded"
        __dunder_var = "should be excluded"

        return hp.collect(locals())

    result = instantiate(config)
    expected = {
        "batch_size": 32,
    }
    assert result == expected
    print("✓ hp.collect excludes private variables")


def test_manual_return_vs_collect():
    """Test that manual return and hp.collect produce equivalent results."""

    def manual_config(hp: HP) -> dict:
        batch_size = hp.int(32, name="batch_size")
        learning_rate = hp.float(0.01, name="lr")

        # Manual return
        return {"batch_size": batch_size, "learning_rate": learning_rate}

    def collect_config(hp: HP) -> dict:
        batch_size = hp.int(32, name="batch_size")
        learning_rate = hp.float(0.01, name="lr")

        # Using hp.collect
        return hp.collect(locals())

    manual_result = instantiate(manual_config)
    collect_result = instantiate(collect_config)

    assert manual_result == collect_result
    print("✓ Manual return and hp.collect produce equivalent results")


def main():
    print("Testing hp.collect helper functionality...")

    test_collect_basic()
    test_collect_with_include()
    test_collect_with_exclude()
    test_collect_excludes_private()
    test_manual_return_vs_collect()

    print("\n🎉 All hp.collect tests passed!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
