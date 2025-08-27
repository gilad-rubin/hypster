#!/usr/bin/env python3
"""Test configuration composition via hp.nest."""

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


def test_basic_nesting():
    """Test basic nested configuration."""

    def child(hp: HP) -> dict:
        x = hp.int(10, name="x")
        return {"x": x}

    def parent(hp: HP) -> dict:
        child_result = hp.nest(child, name="child")
        y = hp.int(20, name="y")
        return {"child": child_result, "y": y}

    result = instantiate(parent)
    assert result == {"child": {"x": 10}, "y": 20}

    # Override nested value with dot notation
    result = instantiate(parent, values={"child.x": 15})
    assert result == {"child": {"x": 15}, "y": 20}


def test_nest_with_args_kwargs():
    """Test passing args and kwargs to nested configs."""

    def child(hp: HP, multiplier: int, offset: int = 0) -> int:
        base = hp.int(5, name="base")
        return base * multiplier + offset

    def parent(hp: HP) -> dict:
        result1 = hp.nest(child, name="calc1", args=(2,))
        result2 = hp.nest(child, name="calc2", args=(3,), kwargs={"offset": 10})
        return {"calc1": result1, "calc2": result2}

    result = instantiate(parent)
    assert result == {"calc1": 10, "calc2": 25}  # 5*2=10, 5*3+10=25


def test_conditional_nesting():
    """Test conditional nesting based on parameter."""

    def model_a(hp: HP) -> dict:
        return {"type": "A", "param": hp.int(1, name="param")}

    def model_b(hp: HP) -> dict:
        return {"type": "B", "param": hp.float(2.0, name="param")}

    def config(hp: HP) -> dict:
        model_type = hp.select(["A", "B"], name="model_type", default="A")

        if model_type == "A":
            model_config = hp.nest(model_a, name="model")
        else:
            model_config = hp.nest(model_b, name="model")

        return {"model_type": model_type, "model_config": model_config}

    # Test default (A)
    result = instantiate(config)
    assert result == {"model_type": "A", "model_config": {"type": "A", "param": 1}}

    # Test override to B
    result = instantiate(config, values={"model_type": "B"})
    assert result == {"model_type": "B", "model_config": {"type": "B", "param": 2.0}}

    # Test nested parameter override
    result = instantiate(config, values={"model_type": "A", "model.param": 5})
    assert result == {"model_type": "A", "model_config": {"type": "A", "param": 5}}


def test_deep_nesting():
    """Test multiple levels of nesting."""

    def level3(hp: HP) -> dict:
        value = hp.int(3, name="value")
        return {"level": 3, "value": value}

    def level2(hp: HP) -> dict:
        value = hp.int(2, name="value")
        nested = hp.nest(level3, name="level3")
        return {"level": 2, "value": value, "nested": nested}

    def level1(hp: HP) -> dict:
        value = hp.int(1, name="value")
        nested = hp.nest(level2, name="level2")
        return {"level": 1, "value": value, "nested": nested}

    result = instantiate(level1)
    expected = {"level": 1, "value": 1, "nested": {"level": 2, "value": 2, "nested": {"level": 3, "value": 3}}}
    assert result == expected

    # Test deep nested parameter override
    result = instantiate(level1, values={"level2.level3.value": 10})
    expected["nested"]["nested"]["value"] = 10
    assert result == expected


def main():
    """Run all test functions."""
    print("Running nesting tests...")

    tests = [
        (test_basic_nesting, "test_basic_nesting"),
        (test_nest_with_args_kwargs, "test_nest_with_args_kwargs"),
        (test_conditional_nesting, "test_conditional_nesting"),
        (test_deep_nesting, "test_deep_nesting"),
    ]

    passed = 0
    total = len(tests)

    for test_func, test_name in tests:
        if run_test_function(test_func, test_name):
            passed += 1

    print(f"\nResults: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All nesting tests passed!")
        return 0
    else:
        print("❌ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
