#!/usr/bin/env python3
"""Debug deep nesting."""

import os
import sys

# Add src to path to import hypster
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from hypster import HP, instantiate


def level3(hp: HP) -> dict:
    print(f"Level3 namespace_stack: {hp.namespace_stack}")
    print(f"Level3 values: {hp.values}")
    value = hp.int(3, name="value")
    print(f"Level3 called_params: {hp.called_params}")
    return {"level": 3, "value": value}


def level2(hp: HP) -> dict:
    print(f"Level2 namespace_stack: {hp.namespace_stack}")
    print(f"Level2 values: {hp.values}")
    value = hp.int(2, name="value")
    print(f"Level2 called_params after local: {hp.called_params}")

    # Debug what will be passed to level3
    full_path = hp._get_full_param_path("level3")
    prefix = full_path + "."
    nested_values = {}
    for key, val in hp.values.items():
        if key.startswith(prefix):
            nested_key = key[len(prefix) :]
            nested_values[nested_key] = val
    print(f"Level2 will pass to level3: full_path={full_path}, prefix={prefix}, nested_values={nested_values}")

    nested = hp.nest(level3, name="level3")
    print(f"Level2 called_params after nesting: {hp.called_params}")
    print(f"Level2 called_params contents: {list(hp.called_params)}")
    return {"level": 2, "value": value, "nested": nested}


def level1(hp: HP) -> dict:
    print(f"Level1 namespace_stack: {hp.namespace_stack}")
    print(f"Level1 values: {hp.values}")
    value = hp.int(1, name="value")
    nested = hp.nest(level2, name="level2")
    print(f"Level1 called_params after nesting: {hp.called_params}")
    return {"level": 1, "value": value, "nested": nested}


print("=== Testing deep nesting with override ===")
result = instantiate(level1, values={"level2.level3.value": 10})
print(f"Result: {result}")
