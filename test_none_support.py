#!/usr/bin/env python3
"""
Test script to verify None value support for hp.int, hp.float, hp.bool, and hp.select
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from hypster import HP, instantiate


def test_none_defaults():
    """Test that None can be used as default values"""
    print("Testing None as default values...")
    
    def config(hp: HP):
        result_int = hp.int(default=None, name='test_int')
        result_float = hp.float(default=None, name='test_float')
        result_bool = hp.bool(default=None, name='test_bool')
        result_select = hp.select(['a', 'b'], default=None, name='test_select')
        
        return {
            'test_int': result_int,
            'test_float': result_float,
            'test_bool': result_bool,
            'test_select': result_select
        }
    
    result = instantiate(config)
    expected = {'test_int': None, 'test_float': None, 'test_bool': None, 'test_select': None}
    assert result == expected, f"Expected {expected}, got {result}"
    print("✓ None defaults work correctly")


def test_none_values_during_instantiation():
    """Test that None can be passed as values during instantiation"""
    print("Testing None as values during instantiation...")
    
    def config(hp: HP):
        result_int = hp.int(default=7, name='test_int')
        result_float = hp.float(default=3.14, name='test_float')
        result_bool = hp.bool(default=True, name='test_bool')
        result_select = hp.select(['a', 'b'], default='a', name='test_select')
        
        return {
            'test_int': result_int,
            'test_float': result_float,
            'test_bool': result_bool,
            'test_select': result_select
        }
    
    values = {
        'test_int': None,
        'test_float': None,
        'test_bool': None,
        'test_select': None
    }
    
    result = instantiate(config, values=values)
    expected = {'test_int': None, 'test_float': None, 'test_bool': None, 'test_select': None}
    assert result == expected, f"Expected {expected}, got {result}"
    print("✓ None values during instantiation work correctly")


def test_specific_examples():
    """Test the specific examples from the problem statement"""
    print("Testing specific examples from problem statement...")
    
    # Example 1: hp.int(default=None, min=1, max=10, name="dimensions")
    def config1(hp: HP):
        dimensions = hp.int(default=None, min=1, max=10, name="dimensions")
        return {"dimensions": dimensions}
    
    result1 = instantiate(config1)
    assert result1 == {"dimensions": None}, f"Expected {{'dimensions': None}}, got {result1}"
    
    # Example 2: instantiate(conf, values={"dimensions": None})
    def config2(hp: HP):
        dimensions = hp.int(default=7, max=10, name="dimensions")
        return {"dimensions": dimensions}
    
    result2 = instantiate(config2, values={"dimensions": None})
    assert result2 == {"dimensions": None}, f"Expected {{'dimensions': None}}, got {result2}"
    
    print("✓ Specific examples work correctly")


def test_bounds_validation_with_none():
    """Test that bounds validation still works but skips None values"""
    print("Testing bounds validation with None values...")
    
    def config(hp: HP):
        value = hp.int(default=5, min=1, max=10, name='test_value')
        return {'test_value': value}
    
    # Test valid non-None value
    result = instantiate(config, values={'test_value': 7})
    assert result == {'test_value': 7}
    
    # Test None value (should bypass validation)
    result = instantiate(config, values={'test_value': None})
    assert result == {'test_value': None}
    
    # Test invalid non-None value (should still fail)
    try:
        instantiate(config, values={'test_value': 15})
        assert False, "Should have failed bounds validation"
    except Exception as e:
        assert "exceeds maximum bound" in str(e)
    
    print("✓ Bounds validation works correctly with None values")


def test_select_behavior():
    """Test select behavior with and without explicit None default"""
    print("Testing select behavior...")
    
    # Without explicit default (should use first option)
    def config1(hp: HP):
        result = hp.select(['a', 'b'], name='test_select')
        return {'test_select': result}
    
    result1 = instantiate(config1)
    assert result1 == {'test_select': 'a'}, f"Expected {{'test_select': 'a'}}, got {result1}"
    
    # With explicit None default (should return None)
    def config2(hp: HP):
        result = hp.select(['a', 'b'], default=None, name='test_select')
        return {'test_select': result}
    
    result2 = instantiate(config2)
    assert result2 == {'test_select': None}, f"Expected {{'test_select': None}}, got {result2}"
    
    print("✓ Select behavior is correct")


def test_backward_compatibility():
    """Test that existing behavior is preserved"""
    print("Testing backward compatibility...")
    
    def config(hp: HP):
        result_int = hp.int(default=42, name='test_int')
        result_float = hp.float(default=3.14, name='test_float')
        result_bool = hp.bool(default=True, name='test_bool')
        result_select = hp.select(['a', 'b'], default='b', name='test_select')
        
        return {
            'test_int': result_int,
            'test_float': result_float,
            'test_bool': result_bool,
            'test_select': result_select
        }
    
    # Test with defaults
    result = instantiate(config)
    expected = {'test_int': 42, 'test_float': 3.14, 'test_bool': True, 'test_select': 'b'}
    assert result == expected, f"Expected {expected}, got {result}"
    
    # Test with overrides
    values = {
        'test_int': 99,
        'test_float': 2.71,
        'test_bool': False,
        'test_select': 'a'
    }
    result = instantiate(config, values=values)
    assert result == values, f"Expected {values}, got {result}"
    
    print("✓ Backward compatibility maintained")


if __name__ == "__main__":
    test_none_defaults()
    test_none_values_during_instantiation()
    test_specific_examples()
    test_bounds_validation_with_none()
    test_select_behavior()
    test_backward_compatibility()
    print("\n🎉 All tests passed!")