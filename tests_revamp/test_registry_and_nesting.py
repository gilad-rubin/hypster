"""
Tests for manual registry workflow and enhanced hp.nest resolution.
"""

import os
import tempfile

import pytest

from hypster import HP, config


def test_registry_manual_registration():
    """Test manual registry registration and retrieval"""
    from hypster import registry

    # Clear registry for clean test
    if hasattr(registry, "clear"):
        registry.clear()

    @config
    def tfidf_retriever(hp: HP):
        min_df = hp.int(2, name="min_df")
        max_features = hp.int(1000, name="max_features")

        return {"type": "tfidf", "min_df": min_df, "max_features": max_features}

    @config
    def model2vec_retriever(hp: HP):
        model_name = hp.text("all-MiniLM-L6-v2", name="model_name")
        dimensions = hp.int(384, name="dimensions")

        return {"type": "model2vec", "model_name": model_name, "dimensions": dimensions}

    # Register both configs
    registry.register(tfidf_retriever, "retriever/tfidf")
    registry.register(model2vec_retriever, "retriever/model2vec")

    # Test registry.list() reflects both entries
    registered = registry.list()
    assert "retriever/tfidf" in registered
    assert "retriever/model2vec" in registered

    # Test registry.get() returns correct objects
    retrieved_tfidf = registry.get("retriever/tfidf")
    retrieved_model2vec = registry.get("retriever/model2vec")

    assert retrieved_tfidf is tfidf_retriever
    assert retrieved_model2vec is model2vec_retriever

    # Test they work when called
    tfidf_result = retrieved_tfidf()
    model2vec_result = retrieved_model2vec()

    assert tfidf_result["type"] == "tfidf"
    assert model2vec_result["type"] == "model2vec"


def test_nest_by_alias():
    """Test hp.nest with registry alias resolution"""
    from hypster import registry

    # Clear registry for clean test
    if hasattr(registry, "clear"):
        registry.clear()

    @config
    def tfidf_retriever(hp: HP):
        min_df = hp.int(2, name="min_df")
        return {"type": "tfidf", "min_df": min_df}

    @config
    def model2vec_retriever(hp: HP):
        dimensions = hp.int(384, name="dimensions")
        return {"type": "model2vec", "dimensions": dimensions}

    # Register configs
    registry.register(tfidf_retriever, "retriever/tfidf")
    registry.register(model2vec_retriever, "retriever/model2vec")

    @config
    def parent_config(hp: HP):
        # Dynamic selection
        retriever_type = hp.select(["tfidf", "model2vec"], name="retriever.type")

        # Nest using alias
        retriever = hp.nest(f"retriever/{retriever_type}", name="retriever")

        return {"retriever_type": retriever_type, "retriever_config": retriever}

    # Test with default (first option)
    result = parent_config()
    assert result["retriever_type"] == "tfidf"
    assert result["retriever_config"]["type"] == "tfidf"

    # Test with override
    result = parent_config(values={"retriever.type": "model2vec"})
    assert result["retriever_type"] == "model2vec"
    assert result["retriever_config"]["type"] == "model2vec"


def test_nest_by_hypster_object():
    """Test hp.nest with direct Hypster object"""

    @config
    def child_config(hp: HP):
        value = hp.number(0.5, name="value")
        return {"child_value": value}

    @config
    def parent_config(hp: HP):
        # Nest using Hypster object directly
        child = hp.nest(child_config, name="child")
        parent_value = hp.number(1.0, name="parent_value")

        return {"parent": parent_value, "child": child}

    result = parent_config()
    assert result["parent"] == 1.0
    assert result["child"]["child_value"] == 0.5

    # Test with nested override
    result = parent_config(values={"parent_value": 2.0, "child.value": 0.8})
    assert result["parent"] == 2.0
    assert result["child"]["child_value"] == 0.8


def test_nest_by_file_path():
    """Test hp.nest with file path import"""
    # Create a temporary config file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        config_content = """
from hypster import config, HP

@config
def saved_config(hp: HP):
    param = hp.number(0.123, name="param")
    return {"saved_param": param}
"""
        f.write(config_content)
        temp_path = f.name

    try:

        @config
        def parent_config(hp: HP):
            # Nest using file path
            nested = hp.nest(temp_path, name="nested")
            return {"nested_result": nested}

        result = parent_config()
        assert "nested_result" in result
        assert result["nested_result"]["saved_param"] == 0.123

        # Test with override
        result = parent_config(values={"nested.param": 0.456})
        assert result["nested_result"]["saved_param"] == 0.456

    finally:
        # Clean up temp file
        os.unlink(temp_path)


def test_registry_unknown_alias_error():
    """Test that requesting unknown alias raises clear error"""
    from hypster import registry

    # Clear registry for clean test
    if hasattr(registry, "clear"):
        registry.clear()

    @config
    def parent_config(hp: HP):
        # Try to nest with unknown alias
        child = hp.nest("unknown/alias", name="child")
        return {"child": child}

    with pytest.raises(Exception) as exc_info:
        parent_config()

    error_msg = str(exc_info.value).lower()
    assert "unknown" in error_msg or "not found" in error_msg or "alias" in error_msg


def test_consistent_return_types():
    """Test that registry variants return consistent types"""
    from hypster import registry

    # Clear registry for clean test
    if hasattr(registry, "clear"):
        registry.clear()

    # Define a protocol-like interface
    class RetrieverProtocol:
        def retrieve(self, query: str):
            pass

    class TFIDFRetriever:
        def __init__(self, min_df: int):
            self.min_df = min_df

        def retrieve(self, query: str):
            return f"TFIDF search for '{query}' with min_df={self.min_df}"

    class Model2VecRetriever:
        def __init__(self, model_name: str):
            self.model_name = model_name

        def retrieve(self, query: str):
            return f"Model2Vec search for '{query}' with model={self.model_name}"

    @config
    def tfidf_config(hp: HP) -> TFIDFRetriever:
        min_df = hp.int(2, name="min_df")
        return TFIDFRetriever(min_df)

    @config
    def model2vec_config(hp: HP) -> Model2VecRetriever:
        model_name = hp.text("all-MiniLM-L6-v2", name="model_name")
        return Model2VecRetriever(model_name)

    # Register both
    registry.register(tfidf_config, "retriever/tfidf")
    registry.register(model2vec_config, "retriever/model2vec")

    @config
    def app_config(hp: HP):
        retriever_type = hp.select(["tfidf", "model2vec"], name="retriever_type")
        retriever = hp.nest(f"retriever/{retriever_type}", name="retriever")

        return {"retriever": retriever, "retriever_type": retriever_type}

    # Test both variants have the same interface
    result1 = app_config(values={"retriever_type": "tfidf"})
    result2 = app_config(values={"retriever_type": "model2vec"})

    retriever1 = result1["retriever"]
    retriever2 = result2["retriever"]

    # Both should have retrieve method
    assert hasattr(retriever1, "retrieve")
    assert hasattr(retriever2, "retrieve")

    # Both should work the same way
    query_result1 = retriever1.retrieve("test query")
    query_result2 = retriever2.retrieve("test query")

    assert "TFIDF search" in query_result1
    assert "Model2Vec search" in query_result2


def test_registry_list_empty():
    """Test registry.list() when empty"""
    from hypster import registry

    # Clear registry for clean test
    if hasattr(registry, "clear"):
        registry.clear()

    registered = registry.list()
    assert isinstance(registered, dict)
    # Should be empty or at least not contain our test entries
    assert "retriever/tfidf" not in registered


def test_nest_return_shape_passthrough():
    """Test that nested configs return exactly what the child returns"""

    @config
    def dict_child(hp: HP):
        value = hp.number(0.5, name="value")
        return {"result": value}

    class CustomClass:
        def __init__(self, value):
            self.value = value

    @config
    def object_child(hp: HP):
        value = hp.number(0.5, name="value")
        return CustomClass(value)

    @config
    def parent_config(hp: HP):
        dict_result = hp.nest(dict_child, name="dict_child")
        obj_result = hp.nest(object_child, name="obj_child")

        return {"dict_result": dict_result, "obj_result": obj_result}

    result = parent_config()

    # Dict child should return dict
    assert isinstance(result["dict_result"], dict)
    assert result["dict_result"]["result"] == 0.5

    # Object child should return custom object
    assert isinstance(result["obj_result"], CustomClass)
    assert result["obj_result"].value == 0.5
