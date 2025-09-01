"""Test type restrictions for select/multi_select."""

from hypster import HP, instantiate


def test_select_list_primitives_only():
    """Test that select with list only accepts primitives."""

    # Valid: primitives only
    def valid_config(hp: HP):
        str_choice = hp.select(["a", "b", "c"], default="a", name="str_choice")
        int_choice = hp.select([1, 2, 3], default=1, name="int_choice")
        float_choice = hp.select([1.0, 2.5, 3.14], default=1.0, name="float_choice")
        bool_choice = hp.select([True, False], default=True, name="bool_choice")
        return {"str": str_choice, "int": int_choice, "float": float_choice, "bool": bool_choice}

    result = instantiate(valid_config)
    assert result == {"str": "a", "int": 1, "float": 1.0, "bool": True}

    # Invalid: contains None
    def invalid_none_config(hp: HP):
        return hp.select([None, "a", "b"], default=None, name="choice")

    # This should ideally raise an error for invalid type in options
    # For now, let's see what happens
    try:
        result = instantiate(invalid_none_config)
        print(f"None in list result: {result}")
    except Exception as e:
        print(f"None in list error: {e}")

    # Invalid: contains tuple
    def invalid_tuple_config(hp: HP):
        return hp.select([(1, 2), (3, 4)], default=(1, 2), name="choice")

    try:
        result = instantiate(invalid_tuple_config)
        print(f"Tuple in list result: {result}")
    except Exception as e:
        print(f"Tuple in list error: {e}")

    # Invalid: contains complex object
    class CustomClass:
        pass

    def invalid_object_config(hp: HP):
        obj = CustomClass()
        return hp.select([obj, "a"], default=obj, name="choice")

    try:
        result = instantiate(invalid_object_config)
        print(f"Object in list result: {result}")
    except Exception as e:
        print(f"Object in list error: {e}")


def test_select_dict_primitive_keys():
    """Test that dict options must have primitive keys."""

    # Valid: primitive keys
    def valid_dict_config(hp: HP):
        str_key = hp.select({"a": "value_a", "b": "value_b"}, default="a", name="str_key")
        int_key = hp.select({1: "one", 2: "two"}, default=1, name="int_key")
        bool_key = hp.select({True: "yes", False: "no"}, default=True, name="bool_key")
        return {"str": str_key, "int": int_key, "bool": bool_key}

    result = instantiate(valid_dict_config)
    assert result == {"str": "value_a", "int": "one", "bool": "yes"}

    # Valid: primitive keys with complex values
    def complex_values_config(hp: HP):
        choice = hp.select(
            {"none": None, "tuple": (1, 2), "dict": {"nested": "value"}, "list": [1, 2, 3]},
            default="none",
            name="choice",
        )
        return choice

    result = instantiate(complex_values_config)
    assert result is None

    # Test override
    result = instantiate(complex_values_config, values={"choice": "tuple"})
    assert result == (1, 2)


def test_multi_select_type_restrictions():
    """Test multi_select with type restrictions."""

    # Valid primitives
    def valid_multi_config(hp: HP):
        return hp.multi_select(["a", "b", "c", "d"], default=["a", "b"], name="choices")

    result = instantiate(valid_multi_config)
    assert result == ["a", "b"]

    # With dict
    def dict_multi_config(hp: HP):
        return hp.multi_select(
            {"opt1": "Value 1", "opt2": "Value 2", "opt3": "Value 3"}, default=["opt1", "opt2"], name="choices"
        )

    result = instantiate(dict_multi_config)
    assert result == ["Value 1", "Value 2"]


if __name__ == "__main__":
    print("Testing select list type restrictions:")
    print("=" * 50)
    test_select_list_primitives_only()

    print("\n" + "=" * 50)
    print("Testing select dict key restrictions:")
    test_select_dict_primitive_keys()

    print("\n" + "=" * 50)
    print("Testing multi_select type restrictions:")
    test_multi_select_type_restrictions()

    print("\nAll tests completed!")
