"""Test to verify what types are considered primitive for return_params."""

from hypster import HP, instantiate


def test_primitive_types():
    """Test different data types to verify primitive vs complex."""

    def config(hp: HP):
        # Test all types
        int_val = hp.select({"a": 42}, default="a", name="int_val")
        float_val = hp.select({"b": 3.14}, default="b", name="float_val")
        str_val = hp.select({"c": "hello"}, default="c", name="str_val")
        bool_val = hp.select({"d": True}, default="d", name="bool_val")
        none_val = hp.select({"e": None}, default="e", name="none_val")
        tuple_val = hp.select({"f": (1, 2)}, default="f", name="tuple_val")
        list_val = hp.select({"g": [1, 2, 3]}, default="g", name="list_val")
        dict_val = hp.select({"h": {"key": "value"}}, default="h", name="dict_val")

        return {
            "int": int_val,
            "float": float_val,
            "str": str_val,
            "bool": bool_val,
            "none": none_val,
            "tuple": tuple_val,
            "list": list_val,
            "dict": dict_val,
        }

    result = instantiate(config)
    print("Results:")
    for k, v in result.items():
        print(f"  {k}: {v} (type: {type(v).__name__})")

    # Verify the actual values returned
    assert result["int"] == 42
    assert result["float"] == 3.14
    assert result["str"] == "hello"
    assert result["bool"]
    assert result["none"] is None
    assert result["tuple"] == (1, 2)
    assert result["list"] == [1, 2, 3]
    assert result["dict"] == {"key": "value"}


def test_list_with_mixed_types():
    """Test list options with mixed primitive and complex types."""

    def config(hp: HP):
        # List with various types
        value = hp.select(["string", 42, 3.14, True, None, (1, 2)], default="string", options_only=False, name="mixed")
        return {"value": value}

    # Test each type
    test_cases = [
        ("string", "string", str),
        (42, 42, int),
        (3.14, 3.14, float),
        (True, True, bool),
        (None, None, type(None)),
        ((1, 2), (1, 2), tuple),
    ]

    for override, expected, expected_type in test_cases:
        result = instantiate(config, values={"mixed": override})
        print(f"Override with {override} ({type(override).__name__}): {result['value']}")
        assert result["value"] == expected
        assert isinstance(result["value"], expected_type)


if __name__ == "__main__":
    print("Testing primitive type recognition:")
    print("=" * 50)
    test_primitive_types()
    print("\n" + "=" * 50)
    print("Testing list with mixed types:")
    test_list_with_mixed_types()
    print("\nAll tests passed!")
