"""Test edge cases and complex scenarios."""

from typing import Any, Dict

import pytest

from hypster import HP, instantiate


def test_deeply_nested_paths() -> None:
    """Test deeply nested configuration paths."""

    def level3(hp: HP) -> int:
        return hp.int(3, name="value")

    def level2(hp: HP) -> Dict[str, int]:
        return {"l3": hp.nest(level3, name="l3")}

    def level1(hp: HP) -> Dict[str, Dict[str, int]]:
        return {"l2": hp.nest(level2, name="l2")}

    result = instantiate(level1)
    assert result == {"l2": {"l3": 3}}

    # Deep override
    result = instantiate(level1, values={"l2.l3.value": 5})
    assert result == {"l2": {"l3": 5}}


def test_conditional_same_name_allowed() -> None:
    """Test that same parameter name in different conditional branches is allowed."""

    def config(hp: HP) -> Dict[str, Any]:
        mode = hp.select(["int", "float"], name="mode", default="int")

        # Same name 'value' used in different branches - this is OK
        if mode == "int":
            value = hp.int(10, name="value", min=0, max=100)
        else:
            value = hp.float(10.0, name="value", min=0.0, max=100.0)

        return {"mode": mode, "value": value}

    # Should work fine with int mode (default)
    result = instantiate(config)
    assert result == {"mode": "int", "value": 10}

    # Should work with float mode
    result = instantiate(config, values={"mode": "float"})
    assert result == {"mode": "float", "value": 10.0}

    # Override value in int mode
    result = instantiate(config, values={"value": 50})
    assert result == {"mode": "int", "value": 50}

    # Override value in float mode
    result = instantiate(config, values={"mode": "float", "value": 50.5})
    assert result == {"mode": "float", "value": 50.5}


def test_name_collision_detection() -> None:
    """Test detection of duplicate parameter names in same execution path."""

    def config(hp: HP) -> Dict[str, int]:
        # These create duplicate names in the same execution path
        x = hp.int(1, name="param")
        y = hp.int(2, name="param")  # Duplicate name - should error
        return {"x": x, "y": y}

    # Should raise error about duplicate parameter
    with pytest.raises(ValueError, match="Parameter 'param': has already been defined"):
        instantiate(config)

    # Another example with nested context
    def nested_collision_config(hp: HP) -> Dict[str, Any]:
        def child(hp: HP) -> int:
            return hp.int(5, name="value")

        # Parent defines 'nested.value'
        result1 = hp.nest(child, name="nested")
        # Parent tries to define same parameter path again
        result2 = hp.nest(child, name="nested")  # This would create 'nested.value' again

        return {"result1": result1, "result2": result2}

    with pytest.raises(ValueError, match="Parameter 'nested.value': has already been defined"):
        instantiate(nested_collision_config)


def test_reserved_prefix_collision() -> None:
    """Test that parent can't define params under child's prefix."""

    def child(hp: HP) -> Dict[str, int]:
        return {"x": hp.int(5, name="x")}

    def parent(hp: HP) -> Dict[str, Any]:
        # Parent tries to define under 'model.' prefix
        bad = hp.int(10, name="model.param")
        # Then tries to nest under 'model'
        nested = hp.nest(child, name="model")
        return {"bad": bad, "nested": nested}

    with pytest.raises(ValueError, match="prefix 'model' reserved"):
        instantiate(parent)
