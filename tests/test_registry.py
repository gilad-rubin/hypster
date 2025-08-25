"""Tests for the Hypster configuration registry system."""

import pytest

from hypster import config, registry
from hypster.registry import Registry


def test_registry_creation():
    """Test that registry is properly initialized."""
    test_registry = Registry()
    assert len(test_registry) == 0
    assert test_registry.list() == []
    assert test_registry.get_namespaces() == []


def test_basic_registration():
    """Test basic configuration registration."""
    registry.clear()  # Clean slate

    @config(register="test.basic")
    def test_config(hp):
        value = hp.int(default=5, min=1, max=10, name="value")

    assert "test.basic" in registry
    assert registry.contains("test.basic")
    assert len(registry) == 1

    retrieved = registry.get("test.basic")
    assert retrieved.name == "test_config"


def test_namespace_registration():
    """Test namespace organization."""
    registry.clear()

    @config(register="models.bert_base")
    def bert_config(hp):
        size = hp.int(default=768, min=100, max=1000, name="size")

    @config(register="models.gpt")
    def gpt_config(hp):
        layers = hp.int(default=12, min=6, max=48, name="layers")

    @config(register="data.preprocessing")
    def preproc_config(hp):
        batch_size = hp.int(default=32, min=16, max=512, name="batch_size")

    # Test namespace listing
    assert len(registry.list()) == 3
    assert len(registry.list("models")) == 2
    assert len(registry.list("data")) == 1

    # Test namespace structure
    namespaces = registry.get_namespaces()
    assert "models" in namespaces
    assert "data" in namespaces


def test_auto_registration_with_name():
    """Test auto-registration with custom name and namespace."""
    registry.clear()

    @config(register=True, name="custom_bert", namespace="ml_models")
    def some_function(hp):
        param = hp.text("default", name="param")

    assert "ml_models.custom_bert" in registry
    retrieved = registry.get("ml_models.custom_bert")
    assert retrieved.name == "some_function"


def test_auto_registration_no_namespace():
    """Test auto-registration without namespace."""
    registry.clear()

    @config(register=True, name="simple")
    def simple_config(hp):
        value = hp.number(0.5, name="value")

    assert "simple" in registry
    retrieved = registry.get("simple")
    assert retrieved.name == "simple_config"


def test_duplicate_registration_prevention():
    """Test that duplicate registration raises error."""
    registry.clear()

    @config(register="duplicate.test")
    def first_config(hp):
        value1 = hp.int(default=5, min=1, max=10, name="value1")

    # This should raise an error
    with pytest.raises(ValueError, match="already registered"):

        @config(register="duplicate.test")
        def second_config(hp):
            value2 = hp.int(default=25, min=20, max=30, name="value2")


def test_override_registration():
    """Test registration with override=True."""
    registry.clear()

    @config(register="override.test")
    def first_config(hp):
        value1 = hp.int(default=5, min=1, max=10, name="value1")

    # This should work with override=True
    @config(register="override.test", override=True)
    def second_config(hp):
        value2 = hp.int(default=25, min=20, max=30, name="value2")

    retrieved = registry.get("override.test")
    assert retrieved.name == "second_config"


def test_registry_clearing():
    """Test registry clearing functionality."""
    registry.clear()

    @config(register="clear.test1")
    def config1(hp):
        pass

    @config(register="clear.test2")
    def config2(hp):
        pass

    @config(register="other.test")
    def config3(hp):
        pass

    assert len(registry) == 3

    # Clear specific namespace
    registry.clear("clear")
    assert len(registry) == 1
    assert "other.test" in registry

    # Clear all
    registry.clear()
    assert len(registry) == 0


def test_registry_contains():
    """Test the __contains__ method."""
    registry.clear()

    @config(register="contains.test")
    def test_config(hp):
        value = hp.bool(default=True, name="value")

    assert "contains.test" in registry
    assert "nonexistent.key" not in registry


def test_registry_key_error():
    """Test KeyError for non-existent keys."""
    registry.clear()

    with pytest.raises(KeyError, match="not found in registry"):
        registry.get("nonexistent.key")


def test_backward_compatibility():
    """Test that configs work without registration."""

    @config
    def unregistered_config(hp):
        value = hp.select(["a", "b", "c"], default="a", name="value")

    # Should work normally
    result = unregistered_config()
    assert "value" in result

    # Should not be in registry
    assert "unregistered_config" not in registry


def test_complex_namespace_hierarchy():
    """Test complex nested namespace structures."""
    registry.clear()

    configs = [
        "ml.models.transformers.bert.base",
        "ml.models.transformers.bert.large",
        "ml.models.cnn.resnet.50",
        "ml.models.cnn.resnet.101",
        "ml.data.loaders.text",
        "ml.data.loaders.image",
        "eval.metrics.accuracy",
        "eval.metrics.f1",
    ]

    for config_key in configs:

        @config(register=config_key)
        def dummy_config(hp):
            value = hp.int(default=5, min=1, max=10, name="value")

    # Test hierarchical listing
    assert len(registry.list("ml")) == 6
    assert len(registry.list("ml.models")) == 4
    assert len(registry.list("ml.models.transformers")) == 2
    assert len(registry.list("eval")) == 2

    # Test namespace structure
    namespaces = registry.get_namespaces()
    expected_namespaces = [
        "ml",
        "ml.models",
        "ml.models.transformers",
        "ml.models.transformers.bert",
        "ml.models.cnn",
        "ml.models.cnn.resnet",
        "ml.data",
        "ml.data.loaders",
        "eval",
        "eval.metrics",
    ]

    for ns in expected_namespaces:
        assert ns in namespaces


def test_registry_performance():
    """Test registry performance with many configurations."""
    registry.clear()

    # Register many configurations
    def create_perf_config(index):
        @config(register=f"perf.test.config_{index}")
        def perf_config(hp):
            value = hp.int(default=50, min=1, max=100, name="value")

        return perf_config

    for i in range(100):
        create_perf_config(i)

    assert len(registry) == 100

    # Test lookup performance (should be fast)
    for i in range(100):
        key = f"perf.test.config_{i}"
        assert key in registry
        config_obj = registry.get(key)
        assert config_obj is not None


if __name__ == "__main__":
    pytest.main([__file__])
