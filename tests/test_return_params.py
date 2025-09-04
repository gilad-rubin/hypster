"""Tests for the return_params feature."""

import warnings

from hypster import HP, instantiate


class TestBasicReturnParams:
    """Test basic return_params functionality."""

    def test_without_return_params(self):
        """Test default behavior without return_params."""

        def config(hp: HP):
            x = hp.int(5, name="x")
            y = hp.float(2.5, name="y")
            return {"sum": x + y}

        result = instantiate(config)
        assert result == {"sum": 7.5}
        assert not isinstance(result, tuple)

    def test_with_return_params_basic(self):
        """Test return_params returns tuple with SelectedParams."""

        def config(hp: HP):
            x = hp.int(5, name="x")
            y = hp.float(2.5, name="y")
            return {"sum": x + y}

        result, params = instantiate(config, return_params=True)
        assert result == {"sum": 7.5}
        assert params.get_flat() == {"x": 5, "y": 2.5}

    def test_tracks_all_params_not_just_returned(self):
        """Test that ALL hp calls are tracked, not just returned values."""

        def config(hp: HP):
            x = hp.int(10, name="x")
            y = hp.float(20.0, name="y")  # Not returned
            z = hp.bool(True, name="z")  # Not returned
            return {"value": x}

        result, params = instantiate(config, return_params=True)
        assert result == {"value": 10}
        assert params.get_flat() == {"x": 10, "y": 20.0, "z": True}

    def test_empty_config(self):
        """Test config with no parameters."""

        def config(hp: HP):
            return {"static": "value"}

        result, params = instantiate(config, return_params=True)
        assert result == {"static": "value"}
        assert params.get_flat() == {}
        assert params.get_nested() == {}


class TestParameterFormats:
    """Test different parameter format outputs."""

    def test_formats(self):
        """Test get_flat() returns dot-separated keys."""

        def config(hp: HP):
            def optimizer_config(hp: HP):
                lr = hp.float(0.001, name="lr")
                momentum = hp.float(0.9, name="momentum")
                return {"lr": lr, "momentum": momentum}

            batch_size = hp.int(32, name="batch_size")
            optimizer = hp.nest(optimizer_config, name="optimizer")
            return {"batch_size": batch_size, "optimizer": optimizer}

        result, params = instantiate(config, return_params=True)
        flat = params.get_flat()
        assert flat == {"batch_size": 32, "optimizer.lr": 0.001, "optimizer.momentum": 0.9}

        nested = params.get_nested()
        assert nested == {"batch_size": 32, "optimizer": {"lr": 0.001, "momentum": 0.9}}

    def test_deeply_nested(self):
        """Test multiple levels of nesting."""

        def config(hp: HP):
            def level2_config(hp: HP):
                def level3_config(hp: HP):
                    deep = hp.int(42, name="deep")
                    return deep

                mid = hp.float(3.14, name="mid")
                nested = hp.nest(level3_config, name="level3")
                return {"mid": mid, "nested": nested}

            top = hp.bool(True, name="top")
            nested = hp.nest(level2_config, name="level2")
            return {"top": top, "nested": nested}

        result, params = instantiate(config, return_params=True)

        flat = params.get_flat()
        assert flat == {"top": True, "level2.mid": 3.14, "level2.level3.deep": 42}

        nested = params.get_nested()
        assert nested == {"top": True, "level2": {"mid": 3.14, "level3": {"deep": 42}}}


class TestDictionaryOptions:
    """Test select/multi_select with dictionary options."""

    def test_select_with_dict_primitive_values(self):
        """Test hp.select with dict options and primitive values returns values in params."""

        def config(hp: HP):
            # Dictionary with primitive string values
            options = {"fast": "gpt-4o-mini", "smart": "gpt-4"}
            model = hp.select(options, default="fast", name="model")
            return {"selected_model": model}

        result, params = instantiate(config, return_params=True)
        assert result == {"selected_model": "gpt-4o-mini"}  # Mapped value
        assert params.get_flat() == {"model": "gpt-4o-mini"}  # Primitive value returned in params

    def test_select_dict_with_complex_values(self):
        """Test hp.select with dict containing complex values returns keys."""

        class ModelConfig:
            def __init__(self, name):
                self.name = name

        def config(hp: HP):
            # Dictionary with complex object values
            options = {"small": ModelConfig("gpt-3.5"), "large": ModelConfig("gpt-4")}
            choice = hp.select(options, default="small", name="model")
            return {"model": choice}

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result, params = instantiate(config, return_params=True)

            # Should warn about complex type
            assert len(w) == 1
            assert "model" in str(w[0].message)

        # Should return key for complex types
        assert params.get_flat() == {"model": "small"}  # Key, not the complex object

    def test_select_dict_mixed_value_types(self):
        """Test dict with mixed primitive and complex values."""

        class ComplexType:
            pass

        def config(hp: HP):
            options = {
                "str_val": "simple string",  # Primitive
                "int_val": 42,  # Primitive
                "complex": ComplexType(),  # Complex
                "none_val": None,  # None is complex
                "tuple_val": (1, 2),  # Tuple is complex
            }
            str_choice = hp.select(options, default="str_val", name="str_choice")
            int_choice = hp.select(options, default="int_val", name="int_choice")
            complex_choice = hp.select(options, default="complex", name="complex_choice")
            none_choice = hp.select(options, default="none_val", name="none_choice")
            tuple_choice = hp.select(options, default="tuple_val", name="tuple_choice")

            return {
                "str": str_choice,
                "int": int_choice,
                "complex": complex_choice,
                "none": none_choice,
                "tuple": tuple_choice,
            }

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result, params = instantiate(config, return_params=True)

            # Warning for complex types including None and tuple
            assert len(w) == 1
            assert "complex_choice" in str(w[0].message)
            assert "none_choice" in str(w[0].message)
            assert "tuple_choice" in str(w[0].message)

        assert params.get_flat() == {
            "str_choice": "simple string",  # Primitive value
            "int_choice": 42,  # Primitive value
            "complex_choice": "complex",  # Key for complex type
            "none_choice": "none_val",  # Key for None (complex)
            "tuple_choice": "tuple_val",  # Key for tuple (complex)
        }

    def test_multi_select_with_dict_primitive_values(self):
        """Test hp.multi_select with dict and primitive values."""

        def config(hp: HP):
            features = {"age": "Age in years", "income": "Annual income", "education": "Education level"}
            selected = hp.multi_select(features, default=["age", "income"], name="features")
            return {"selected_features": selected}

        result, params = instantiate(config, return_params=True)
        # Function returns the mapped values
        assert result == {"selected_features": ["Age in years", "Annual income"]}
        # Params contain the primitive values
        assert params.get_flat() == {"features": ["Age in years", "Annual income"]}

    def test_multi_select_with_dict_complex_values(self):
        """Test hp.multi_select with dict containing complex values."""

        class Feature:
            def __init__(self, name):
                self.name = name

        def config(hp: HP):
            features = {
                "f1": Feature("feature1"),
                "f2": Feature("feature2"),
                "f3": "simple_string",  # Mixed with primitive
            }
            selected = hp.multi_select(features, default=["f1", "f3"], name="features")
            return {"selected": selected}

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result, params = instantiate(config, return_params=True)

            # Warning for complex values
            assert len(w) == 1
            assert "features" in str(w[0].message)

        # Mixed: key for complex, value for primitive
        assert params.get_flat() == {"features": ["f1", "simple_string"]}

    def test_select_override_with_non_key_value(self):
        """Test overriding dict select with value not in keys."""

        def config(hp: HP):
            options = {"preset1": "value1", "preset2": "value2"}
            choice = hp.select(options, default="preset1", options_only=False, name="choice")
            return {"choice": choice}

        # Override with a custom primitive value (not a key)
        result, params = instantiate(config, values={"choice": "custom_string"}, return_params=True)
        assert result == {"choice": "custom_string"}
        assert params.get_flat() == {"choice": "custom_string"}  # Primitive as-is

        # Override with a custom complex value
        class CustomObj:
            pass

        custom = CustomObj()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result, params = instantiate(config, values={"choice": custom}, return_params=True)

            # Warning for non-primitive
            assert len(w) == 1
            assert "choice" in str(w[0].message)

        assert isinstance(params.get_flat()["choice"], str)  # Stringified

    def test_dict_with_tuple_values(self):
        """Test dict options containing tuple values (treated as complex)."""

        def config(hp: HP):
            ngram_ranges = {"unigram": (1, 1), "bigram": (1, 2), "trigram": (1, 3)}
            ngram = hp.select(ngram_ranges, default="bigram", name="ngram")
            return {"ngram_range": ngram}

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result, params = instantiate(config, return_params=True)

            # Should warn about tuple being complex
            assert len(w) == 1
            assert "ngram" in str(w[0].message)

        assert result == {"ngram_range": (1, 2)}
        # Tuples are complex, so return the key
        assert params.get_flat() == {"ngram": "bigram"}

    def test_dict_with_none_values(self):
        """Test dict options containing None values (treated as complex)."""

        def config(hp: HP):
            tokenizers = {"none": None, "basic": "basic_tokenizer"}
            tokenizer = hp.select(tokenizers, default="none", name="tokenizer")

            return {"tokenizer": tokenizer}

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result, params = instantiate(config, return_params=True)

            # Should warn about None being complex
            assert len(w) == 1
            assert "tokenizer" in str(w[0].message)

        assert result == {"tokenizer": None}
        # None is complex, so return the key
        assert params.get_flat() == {"tokenizer": "none"}


class TestNonPrimitiveHandling:
    """Test handling of non-primitive values."""

    class CustomClass:
        def __init__(self, value):
            self.value = value

        def __str__(self):
            return f"CustomClass({self.value})"

    def test_non_primitive_with_list_options(self):
        """Test non-primitive value with list options."""

        def config(hp: HP):
            # List options with options_only=False
            value = hp.select(["a", "b"], default="a", options_only=False, name="value")
            return {"value": value}

        custom = self.CustomClass(42)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result, params = instantiate(config, values={"value": custom}, return_params=True)

            # Check warning was issued
            assert len(w) == 1
            assert "value" in str(w[0].message)
            assert "not primitives" in str(w[0].message).lower()

        # Check value was converted to string
        assert isinstance(params.get_flat()["value"], str)
        assert "CustomClass(42)" in params.get_flat()["value"]

    def test_multiple_non_primitives_single_warning(self):
        """Test multiple non-primitive values generate single warning."""

        def config(hp: HP):
            val1 = hp.select(["a"], default="a", options_only=False, name="val1")
            val2 = hp.select(["b"], default="b", options_only=False, name="val2")
            return {"val1": val1, "val2": val2}

        custom1 = self.CustomClass(1)

        def custom2(x):
            return x  # Function is non-primitive

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result, params = instantiate(config, values={"val1": custom1, "val2": custom2}, return_params=True)

            # Single consolidated warning
            assert len(w) == 1
            warning_msg = str(w[0].message)
            assert "val1" in warning_msg
            assert "val2" in warning_msg

        # Both converted to strings
        assert isinstance(params.get_flat()["val1"], str)
        assert isinstance(params.get_flat()["val2"], str)

    def test_dict_select_no_warning_for_primitive_values(self):
        """Test dict select with primitive values doesn't trigger warnings."""

        def config(hp: HP):
            # Dictionary with all primitive values (excluding None)
            options = {
                "opt1": "string value",
                "opt2": 42,
                "opt3": 3.14,
                "opt4": True,
            }
            choice = hp.select(options, default="opt1", name="choice")
            return {"choice": choice}

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result, params = instantiate(config, return_params=True)

            # No warning for primitive values
            assert len(w) == 0

        assert params.get_flat() == {"choice": "string value"}  # Primitive value

    def test_none_treated_as_complex(self):
        """Test that None values are treated as complex types."""

        def config(hp: HP):
            # List options with None
            value = hp.select(["a", None], default=None, options_only=False, name="value")
            return {"value": value}

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result, params = instantiate(config, return_params=True)

            # Should warn about None
            assert len(w) == 1
            assert "value" in str(w[0].message)

        # None should be stringified
        assert params.get_flat() == {"value": "None"}

    def test_tuple_treated_as_complex(self):
        """Test that tuple values are treated as complex types."""

        def config(hp: HP):
            # List options with tuple
            value = hp.select(["a", (1, 2)], default=(1, 2), options_only=False, name="value")
            return {"value": value}

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result, params = instantiate(config, return_params=True)

            # Should warn about tuple
            assert len(w) == 1
            assert "value" in str(w[0].message)

        # Tuple should be stringified
        assert params.get_flat() == {"value": "(1, 2)"}


class TestConditionalParameters:
    """Test parameters in conditional flows."""

    def test_conditional_branch_only_executed_params(self):
        """Test only executed branch parameters are returned."""

        def config(hp: HP):
            mode = hp.select(["a", "b"], default="a", name="mode")

            if mode == "a":
                x = hp.int(10, name="x")
                return {"mode": mode, "value": x}
            else:
                y = hp.float(20.0, name="y")
                return {"mode": mode, "value": y}

        # Branch A
        result, params = instantiate(config, values={"mode": "a"}, return_params=True)
        assert params.get_flat() == {"mode": "a", "x": 10}
        assert "y" not in params.get_flat()

        # Branch B
        result, params = instantiate(config, values={"mode": "b"}, return_params=True)
        assert params.get_flat() == {"mode": "b", "y": 20.0}
        assert "x" not in params.get_flat()

    def test_nested_conditionals(self):
        """Test nested conditional flows."""

        def config(hp: HP):
            level1 = hp.select(["a", "b"], default="a", name="level1")

            if level1 == "a":
                level2 = hp.select(["x", "y"], default="x", name="level2")
                if level2 == "x":
                    val = hp.int(1, name="val_ax")
                else:
                    val = hp.int(2, name="val_ay")
            else:
                val = hp.int(3, name="val_b")

            return {"val": val}

        # Path a -> x
        result, params = instantiate(config, values={"level1": "a", "level2": "x"}, return_params=True)
        assert params.get_flat() == {"level1": "a", "level2": "x", "val_ax": 1}

        # Path a -> y
        result, params = instantiate(config, values={"level1": "a", "level2": "y"}, return_params=True)
        assert params.get_flat() == {"level1": "a", "level2": "y", "val_ay": 2}

        # Path b
        result, params = instantiate(config, values={"level1": "b"}, return_params=True)
        assert params.get_flat() == {"level1": "b", "val_b": 3}
        assert "level2" not in params.get_flat()


class TestParameterTypes:
    """Test all parameter types are handled correctly."""

    def test_all_primitive_types(self):
        """Test all primitive parameter types."""

        def config(hp: HP):
            i = hp.int(42, name="int_val")
            f = hp.float(3.14, name="float_val")
            b = hp.bool(True, name="bool_val")
            s = hp.text("hello", name="str_val")
            select = hp.select(["opt1", "opt2"], "opt1", name="select_val")
            multi = hp.multi_select(["a", "b", "c"], ["a", "b"], name="multi_val")

            return {"int": i, "float": f, "bool": b, "str": s, "select": select, "multi": multi}

        result, params = instantiate(config, return_params=True)

        flat = params.get_flat()
        assert flat == {
            "int_val": 42,
            "float_val": 3.14,
            "bool_val": True,
            "str_val": "hello",
            "select_val": "opt1",
            "multi_val": ["a", "b"],
        }

        # Verify types
        assert isinstance(flat["int_val"], int)
        assert isinstance(flat["float_val"], float)
        assert isinstance(flat["bool_val"], bool)
        assert isinstance(flat["str_val"], str)
        assert isinstance(flat["select_val"], str)
        assert isinstance(flat["multi_val"], list)
        assert all(isinstance(v, str) for v in flat["multi_val"])

    def test_numeric_edge_cases(self):
        """Test edge cases for numeric parameters."""

        def config(hp: HP):
            zero = hp.int(0, name="zero")
            negative_int = hp.int(-100, name="negative_int")
            negative_float = hp.float(-3.14, name="negative_float")
            large = hp.int(1_000_000_000, name="large")
            tiny = hp.float(1e-10, name="tiny")

            return {"sum": zero + negative_int + negative_float + large + tiny}

        result, params = instantiate(config, return_params=True)

        flat = params.get_flat()
        assert flat == {"zero": 0, "negative_int": -100, "negative_float": -3.14, "large": 1_000_000_000, "tiny": 1e-10}


class TestOverrides:
    """Test parameter overrides are reflected in returned params."""

    def test_simple_overrides(self):
        """Test overridden values are returned."""

        def config(hp: HP):
            x = hp.int(5, name="x")
            y = hp.float(2.5, name="y")
            return {"product": x * y}

        result, params = instantiate(config, values={"x": 10, "y": 3.0}, return_params=True)
        assert result == {"product": 30.0}
        assert params.get_flat() == {"x": 10, "y": 3.0}

    def test_partial_overrides(self):
        """Test partial overrides with some defaults."""

        def config(hp: HP):
            a = hp.int(1, name="a")
            b = hp.int(2, name="b")
            c = hp.int(3, name="c")
            return {"sum": a + b + c}

        result, params = instantiate(config, values={"b": 20}, return_params=True)
        assert result == {"sum": 24}
        assert params.get_flat() == {"a": 1, "b": 20, "c": 3}

    def test_nested_overrides(self):
        """Test overrides in nested configs."""

        def config(hp: HP):
            def child(hp: HP):
                x = hp.int(10, name="x")
                y = hp.int(20, name="y")
                return x + y

            base = hp.int(100, name="base")
            child_val = hp.nest(child, name="child")
            return {"result": base + child_val}

        result, params = instantiate(config, values={"base": 200, "child.x": 30}, return_params=True)
        assert result == {"result": 250}  # 200 + 30 + 20
        assert params.get_flat() == {"base": 200, "child.x": 30, "child.y": 20}


class TestMultiParameterListPreservation:
    """Test that multi_x parameters preserve their list values without conversion warnings."""

    def test_multi_int_preserves_list_no_warning(self):
        """Test that multi_int returns list without conversion warnings."""

        def config(hp: HP):
            test = hp.multi_int([1, 2, 3], name="test")
            return test

        result, params = instantiate(config, return_params=True)

        # The actual result should be the list
        assert result == [1, 2, 3]

        # get_flat() should return the list as-is without conversion
        flat_params = params.get_flat()
        assert flat_params["test"] == [1, 2, 3]
        assert isinstance(flat_params["test"], list)

        # No warnings should be generated for multi_int lists
        import warnings

        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter("always")
            params.get_flat()

        # Filter out any warnings that aren't about parameter conversion
        conversion_warnings = [w for w in warning_list if "converted to string" in str(w.message)]
        assert len(conversion_warnings) == 0

    def test_multi_float_preserves_list_no_warning(self):
        """Test that multi_float returns list without conversion warnings."""

        def config(hp: HP):
            test = hp.multi_float([1.1, 2.2, 3.3], name="test")
            return test

        result, params = instantiate(config, return_params=True)

        # The actual result should be the list
        assert result == [1.1, 2.2, 3.3]

        # get_flat() should return the list as-is without conversion
        flat_params = params.get_flat()
        assert flat_params["test"] == [1.1, 2.2, 3.3]
        assert isinstance(flat_params["test"], list)

        # No warnings should be generated for multi_float lists
        import warnings

        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter("always")
            params.get_flat()

        conversion_warnings = [w for w in warning_list if "converted to string" in str(w.message)]
        assert len(conversion_warnings) == 0

    def test_multi_text_preserves_list_no_warning(self):
        """Test that multi_text returns list without conversion warnings."""

        def config(hp: HP):
            test = hp.multi_text(["a", "b", "c"], name="test")
            return test

        result, params = instantiate(config, return_params=True)

        # The actual result should be the list
        assert result == ["a", "b", "c"]

        # get_flat() should return the list as-is without conversion
        flat_params = params.get_flat()
        assert flat_params["test"] == ["a", "b", "c"]
        assert isinstance(flat_params["test"], list)

        # No warnings should be generated for multi_text lists
        import warnings

        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter("always")
            params.get_flat()

        conversion_warnings = [w for w in warning_list if "converted to string" in str(w.message)]
        assert len(conversion_warnings) == 0

    def test_multi_bool_preserves_list_no_warning(self):
        """Test that multi_bool returns list without conversion warnings."""

        def config(hp: HP):
            test = hp.multi_bool([True, False, True], name="test")
            return test

        result, params = instantiate(config, return_params=True)

        # The actual result should be the list
        assert result == [True, False, True]

        # get_flat() should return the list as-is without conversion
        flat_params = params.get_flat()
        assert flat_params["test"] == [True, False, True]
        assert isinstance(flat_params["test"], list)

        # No warnings should be generated for multi_bool lists
        import warnings

        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter("always")
            params.get_flat()

        conversion_warnings = [w for w in warning_list if "converted to string" in str(w.message)]
        assert len(conversion_warnings) == 0

    def test_multi_select_preserves_list_no_warning(self):
        """Test that multi_select returns list without conversion warnings."""

        def config(hp: HP):
            test = hp.multi_select([1, 2, 3, 4], default=[1, 3], name="test")
            return test

        result, params = instantiate(config, return_params=True)

        # The actual result should be the list
        assert result == [1, 3]

        # get_flat() should return the list as-is without conversion
        flat_params = params.get_flat()
        assert flat_params["test"] == [1, 3]
        assert isinstance(flat_params["test"], list)

        # No warnings should be generated for multi_select lists with primitive values
        import warnings

        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter("always")
            params.get_flat()

        conversion_warnings = [w for w in warning_list if "converted to string" in str(w.message)]
        assert len(conversion_warnings) == 0

    def test_multi_select_with_complex_dict_still_warns(self):
        """Test that multi_select with complex dict values still uses key fallback and warns."""

        def config(hp: HP):
            complex_dict = {
                "option1": {"nested": "complex", "data": [1, 2, 3]},
                "option2": {"other": "complex", "stuff": {"more": "nesting"}},
            }
            selected = hp.multi_select(complex_dict, default=["option1"], name="test")
            return selected

        import warnings

        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter("always")
            result, params = instantiate(config, return_params=True)
            flat_params = params.get_flat()

        # The result should be the actual complex object
        assert result == [{"nested": "complex", "data": [1, 2, 3]}]

        # But the return_params should fall back to the key
        assert flat_params["test"] == ["option1"]

        # Should generate appropriate warning about complex values
        all_warnings = [str(w.message) for w in warning_list]
        complex_warnings = [w for w in all_warnings if "keys instead" in w]
        assert len(complex_warnings) > 0
