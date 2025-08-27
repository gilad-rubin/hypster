#!/usr/bin/env python3
"""Debug nested parameter tracking."""

import os
import sys

# Add src to path to import hypster
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from hypster import HP, instantiate


def child(hp: HP) -> dict:
    print(f"Child namespace_stack: {hp.namespace_stack}")
    print(f"Child values: {hp.values}")

    # Debug the value lookup
    value, found = hp._get_value_for_param("x")
    print(f"Child looking for 'x': value={value}, found={found}")

    x = hp.int(10, name="x")
    print(f"Child called_params: {hp.called_params}")
    print(f"Child got x={x}")
    return {"x": x}


def parent(hp: HP) -> dict:
    print(f"Parent namespace_stack: {hp.namespace_stack}")
    print(f"Parent values: {hp.values}")
    child_result = hp.nest(child, name="child")
    print(f"After nesting, parent called_params: {hp.called_params}")
    y = hp.int(20, name="y")
    return {"child": child_result, "y": y}


print("=== Testing basic nesting ===")
result = instantiate(parent)
print(f"Result: {result}")

print("\n=== Testing with nested override ===")
result = instantiate(parent, values={"child.x": 15})
print(f"Result: {result}")
