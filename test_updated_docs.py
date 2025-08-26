#!/usr/bin/env python3
"""
Quick test to verify updated documentation examples work.
"""

from hypster import HP, config


def test_basic_example():
    """Test the basic example from getting-started docs."""

    @config
    def my_config(hp: HP):
        var = hp.select(["a", "b", "c"], default="a", name="var")
        num = hp.number(10, name="num")
        text = hp.text("Hey!", name="text")

    # Test instantiation
    result = my_config()
    print("✅ Basic example test passed")
    print(f"Result keys: {list(result.keys())}")
    assert "var" in result
    assert "num" in result
    assert "text" in result
    return result


def test_nested_example():
    """Test nested configuration example."""

    @config(register="test.llm")
    def llm_config(hp: HP):
        model = hp.select(["gpt-3.5-turbo", "gpt-4"], default="gpt-3.5-turbo", name="model")
        temperature = hp.number(0.7, min=0, max=1, name="temperature")

    @config
    def rag_config(hp: HP):
        llm = hp.nest("test.llm", name="llm")
        max_tokens = hp.int(1000, min=100, max=2000, name="max_tokens")

    # Test instantiation
    result = rag_config()
    print("✅ Nested example test passed")
    print(f"Result keys: {list(result.keys())}")
    assert "llm" in result
    assert "max_tokens" in result
    return result


if __name__ == "__main__":
    print("Testing updated documentation examples...")

    try:
        test_basic_example()
        print()

        test_nested_example()
        print()

        print("🎉 All updated documentation examples work correctly!")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise
