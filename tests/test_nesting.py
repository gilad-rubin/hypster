"""Test configuration composition via hp.nest."""

from typing import Any, Dict

from hypster import HP, instantiate


def test_basic_nesting() -> None:
    """Test basic nested configuration."""

    def child(hp: HP) -> Dict[str, int]:
        x = hp.int(10, name="x")
        return {"x": x}

    def parent(hp: HP) -> Dict[str, Any]:
        child_result = hp.nest(child, name="child")
        y = hp.int(20, name="y")
        return {"child": child_result, "y": y}

    result = instantiate(parent)
    assert result == {"child": {"x": 10}, "y": 20}

    # Override nested value with dot notation
    result = instantiate(parent, values={"child.x": 15})
    assert result == {"child": {"x": 15}, "y": 20}


def test_nest_with_args_kwargs() -> None:
    """Test passing args and kwargs to nested configs."""

    def child(hp: HP, multiplier: int, offset: int = 0) -> int:
        base = hp.int(5, name="base")
        return base * multiplier + offset

    def parent(hp: HP) -> Dict[str, int]:
        result1 = hp.nest(child, name="calc1", args=(2,))
        result2 = hp.nest(child, name="calc2", args=(3,), kwargs={"offset": 10})
        return {"calc1": result1, "calc2": result2}

    result = instantiate(parent)
    assert result == {"calc1": 10, "calc2": 25}  # 5*2=10, 5*3+10=25


def test_conditional_nesting() -> None:
    """Test conditional nesting based on parameter."""

    def model_a(hp: HP) -> Dict[str, Any]:
        return {"type": "A", "param": hp.int(1, name="param")}

    def model_b(hp: HP) -> Dict[str, Any]:
        return {"type": "B", "param": hp.float(2.0, name="param")}

    def config(hp: HP) -> Dict[str, Any]:
        model_type = hp.select(["a", "b"], name="model_type", default="a")

        if model_type == "a":
            model = hp.nest(model_a, name="model")
        else:
            model = hp.nest(model_b, name="model")

        return {"model": model}

    result = instantiate(config)
    assert result == {"model": {"type": "A", "param": 1}}

    result = instantiate(config, values={"model_type": "b", "model.param": 3.0})
    assert result == {"model": {"type": "B", "param": 3.0}}


def test_nested_dict_override_precedence() -> None:
    """Test that nested dict values take precedence over dotted keys."""

    def child(hp: HP) -> Dict[str, int]:
        x = hp.int(10, name="x")
        y = hp.int(20, name="y")
        return {"x": x, "y": y}

    def parent(hp: HP) -> Dict[str, int]:
        return hp.nest(child, name="child")

    # Both dotted and nested dict notation - nested dict wins
    result = instantiate(
        parent,
        values={
            "child.x": 100,  # dotted notation
            "child": {"x": 200},  # nested dict notation
        },
    )
    assert result == {"x": 200, "y": 20}  # nested dict wins
