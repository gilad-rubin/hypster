"""Test configuration composition via hp.nest."""

from typing import Any, Dict

import pytest

from hypster import HP, instantiate, instantiate_with_params
from hypster.explore import explore


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
        result1 = hp.nest(child, name="calc1", multiplier=2)
        result2 = hp.nest(child, name="calc2", multiplier=3, offset=10)
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


def test_duplicate_dotted_and_nested_values_raise() -> None:
    """Test that one parameter path cannot be provided in both values forms."""

    def child(hp: HP) -> Dict[str, int]:
        x = hp.int(10, name="x")
        y = hp.int(20, name="y")
        return {"x": x, "y": y}

    def parent(hp: HP) -> Dict[str, int]:
        return hp.nest(child, name="child")

    with pytest.raises(ValueError) as exc_info:
        instantiate(
            parent,
            values={
                "child.x": 100,
                "child": {"x": 100},
            },
        )

    assert "Duplicate value for 'child.x'" in str(exc_info.value)
    assert "dotted key 'child.x'" in str(exc_info.value)
    assert "nested dict 'child' -> 'x'" in str(exc_info.value)
    assert "Use only one form" in str(exc_info.value)


def test_nested_values_keys_must_be_identifier_segments() -> None:
    def child(hp: HP) -> Dict[str, int]:
        return {"x": hp.int(10, name="x")}

    def parent(hp: HP) -> Dict[str, int]:
        return hp.nest(child, name="child")

    with pytest.raises(ValueError, match="nested values key"):
        instantiate(parent, values={"child": {"x.y": 100}})


def test_nested_scope_value_is_not_a_parameter_leaf() -> None:
    def child(hp: HP) -> Dict[str, int]:
        return {"x": hp.int(10, name="x")}

    def parent(hp: HP) -> Dict[str, int]:
        return hp.nest(child, name="child")

    with pytest.raises(ValueError, match="Unknown or unreachable parameters"):
        instantiate(parent, values={"child": 123})

    with pytest.raises(ValueError, match="Unknown or unreachable parameters"):
        instantiate_with_params(parent, values={"child": 123})

    with pytest.raises(ValueError, match="Unknown or unreachable parameters"):
        explore(parent, values={"child": 123}, return_schema=True)


def test_explicit_nested_values_raise_for_unknown_child_parameters() -> None:
    def child(hp: HP) -> Dict[str, int]:
        return {"x": hp.int(10, name="x")}

    def parent(hp: HP) -> Dict[str, int]:
        return hp.nest(child, name="child", values={"typo": 100})

    with pytest.raises(ValueError, match="Unknown or unreachable parameters"):
        instantiate(parent)

    with pytest.raises(ValueError, match="Unknown or unreachable parameters"):
        explore(parent, return_schema=True)


def test_nest_metadata_appears_on_group_node() -> None:
    """Metadata passed to hp.nest lands on the group ParameterInfo in the explored schema."""

    def child(hp: HP) -> Dict[str, int]:
        return {"x": hp.int(10, name="x")}

    def parent(hp: HP) -> Dict[str, int]:
        return hp.nest(child, name="child", metadata={"ui": True})

    info = explore(parent, return_schema=True)

    assert info is not None
    group = info.parameters[0]
    assert group.is_group()
    assert group.metadata == {"ui": True}


def test_nest_metadata_changes_the_explored_schema() -> None:
    """Falsifier: two different metadata dicts must produce two different schema outputs."""

    def child(hp: HP) -> Dict[str, int]:
        return {"x": hp.int(10, name="x")}

    def parent(hp: HP, nest_metadata: Dict[str, Any]) -> Dict[str, int]:
        return hp.nest(child, name="child", metadata=nest_metadata)

    schema_a = explore(parent, return_schema=True, nest_metadata={"ui": "wizard"})
    schema_b = explore(parent, return_schema=True, nest_metadata={"ui": "sidebar"})

    assert schema_a is not None and schema_b is not None
    group_a = schema_a.to_dict()["parameters"][0]
    group_b = schema_b.to_dict()["parameters"][0]
    assert group_a["metadata"] == {"ui": "wizard"}
    assert group_b["metadata"] == {"ui": "sidebar"}
    assert group_a["metadata"] != group_b["metadata"]


def test_nest_without_metadata_leaves_group_metadata_unset() -> None:
    def child(hp: HP) -> Dict[str, int]:
        return {"x": hp.int(10, name="x")}

    def parent(hp: HP) -> Dict[str, int]:
        return hp.nest(child, name="child")

    info = explore(parent, return_schema=True)

    assert info is not None
    group = info.parameters[0]
    assert group.is_group()
    assert group.metadata is None
    assert "metadata" not in info.to_dict()["parameters"][0]


def test_nest_metadata_must_be_json_compatible() -> None:
    def child(hp: HP) -> Dict[str, int]:
        return {"x": hp.int(10, name="x")}

    def parent(hp: HP) -> Dict[str, int]:
        return hp.nest(child, name="child", metadata={"editor": object()})

    with pytest.raises(ValueError, match=r"metadata\['editor'\] must be JSON-compatible"):
        explore(parent, return_schema=True)

    with pytest.raises(ValueError, match=r"metadata\['editor'\] must be JSON-compatible"):
        instantiate(parent)


def test_nest_metadata_does_not_affect_selected_params_or_replay() -> None:
    def child(hp: HP) -> Dict[str, int]:
        return {"x": hp.int(10, name="x")}

    def parent_plain(hp: HP) -> Dict[str, int]:
        return hp.nest(child, name="child")

    def parent_with_metadata(hp: HP) -> Dict[str, int]:
        return hp.nest(child, name="child", metadata={"ui": True})

    plain = instantiate_with_params(parent_plain)
    with_metadata = instantiate_with_params(parent_with_metadata)

    assert with_metadata.params == plain.params
    assert with_metadata.value == plain.value
    assert instantiate(parent_with_metadata, values=with_metadata.params) == with_metadata.value
