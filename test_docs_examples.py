#!/usr/bin/env python3
"""
Test script to verify that the documentation examples work correctly.
"""

from hypster import HP, config, registry


def test_registry_basic_usage():
    """Test basic registry usage from documentation."""

    @config(register="llm.claude")
    def claude_config(hp: HP):
        model = hp.select(["haiku", "sonnet"], default="haiku", name="model")
        temperature = hp.number(0.7, min=0, max=1, name="temperature")

    @config(register="llm.openai")
    def openai_config(hp: HP):
        model = hp.select(["gpt-3.5-turbo", "gpt-4"], default="gpt-3.5-turbo", name="model")
        temperature = hp.number(0.7, min=0, max=1, name="temperature")

    # Now use registry in nesting
    @config
    def rag_config(hp: HP):
        # Registry lookup - tries "llm.claude" first
        llm = hp.nest("llm.claude", name="llm")

        # QA-specific parameters
        max_tokens = hp.int(1000, min=100, max=2000, name="max_tokens")

    # Test instantiation
    result = rag_config()
    print("✅ Basic registry usage test passed")
    print(f"Result keys: {list(result.keys())}")

    # Test registry operations
    assert registry.contains("llm.claude")
    assert registry.contains("llm.openai")

    all_configs = registry.list()  # list all configs
    print(f"All configs: {all_configs}")

    llm_configs = registry.list("llm")  # list configs in namespace
    print(f"LLM configs: {llm_configs}")

    return result


def test_namespace_organization():
    """Test namespace organization from documentation."""

    # Clear registry for clean test
    registry.clear()

    # Embedding configurations
    @config(register="embeddings.openai")
    def openai_embeddings(hp: HP):
        model = hp.select(["text-embedding-ada-002"], default="text-embedding-ada-002", name="model")

    @config(register="embeddings.sentence_transformers")
    def st_embeddings(hp: HP):
        model = hp.select(["all-MiniLM-L6-v2"], default="all-MiniLM-L6-v2", name="model")

    # Retrieval configurations
    @config(register="retrieval.vector")
    def vector_retrieval(hp: HP):
        top_k = hp.int(5, min=1, max=20, name="top_k")

    @config(register="retrieval.hybrid")
    def hybrid_retrieval(hp: HP):
        alpha = hp.number(0.5, min=0, max=1, name="alpha")

    # Test namespace listing
    embedding_configs = registry.list("embeddings")
    retrieval_configs = registry.list("retrieval")

    print("✅ Namespace organization test passed")
    print(f"Embedding configs: {embedding_configs}")
    print(f"Retrieval configs: {retrieval_configs}")

    return embedding_configs, retrieval_configs


def test_adaptive_config():
    """Test adaptive configuration selection from documentation."""

    @config(register="llm.openai.test")
    def openai_test(hp: HP):
        model = hp.select(["gpt-3.5-turbo"], default="gpt-3.5-turbo", name="model")
        provider = "openai"

    @config(register="llm.anthropic.test")
    def anthropic_test(hp: HP):
        model = hp.select(["claude-3-haiku"], default="claude-3-haiku", name="model")
        provider = "anthropic"

    @config
    def adaptive_config(hp: HP):
        provider = hp.select(["openai", "anthropic"], default="openai", name="provider")

        # Dynamic registry lookup based on selection
        if provider == "openai":
            llm = hp.nest("llm.openai.test", name="llm")
        else:
            llm = hp.nest("llm.anthropic.test", name="llm")

    # Test with default (openai)
    result1 = adaptive_config()
    print("✅ Adaptive config test passed")
    print(f"Default result keys: {list(result1.keys())}")

    # Test with anthropic
    result2 = adaptive_config(values={"provider": "anthropic"})
    print(f"Anthropic result keys: {list(result2.keys())}")

    return result1, result2


if __name__ == "__main__":
    print("Testing documentation examples...")

    try:
        # Test basic registry usage
        test_registry_basic_usage()
        print()

        # Test namespace organization
        test_namespace_organization()
        print()

        # Test adaptive configuration
        test_adaptive_config()
        print()

        print("🎉 All documentation examples work correctly!")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise
