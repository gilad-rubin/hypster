"""
Tests for return semantics - ensuring pass-through returns and typed-return patterns work.
"""

from dataclasses import dataclass
from typing import NamedTuple

import pytest

from hypster import HP, config


@dataclass
class ModelConfig:
    learning_rate: float
    epochs: int
    model_type: str


class TrainingResult(NamedTuple):
    accuracy: float
    loss: float
    model_name: str


class MockModel:
    """Mock model class for testing single object returns"""

    def __init__(self, model_type: str = "random_forest"):
        self.model_type = model_type

    def fit(self, X, y):
        return self

    def predict(self, X):
        return [1] * len(X)


def test_return_dict_basic():
    """Test returning a dict with explicit HP names"""

    @config
    def my_config(hp: HP):
        learning_rate = hp.number(0.01, name="learning_rate")
        model_type = hp.select(["rf", "svm"], name="model_type")

        return hp.collect(locals())

    result = my_config()
    assert isinstance(result, dict)
    assert "learning_rate" in result
    assert "model_type" in result
    assert result["learning_rate"] == 0.01
    assert result["model_type"] == "rf"


def test_return_dict_with_values_override():
    """Test that values overrides work with explicit names"""

    @config
    def my_config(hp: HP):
        learning_rate = hp.number(0.01, name="learning_rate")
        model_type = hp.select(["rf", "svm"], name="model_type")

        return hp.collect(locals())

    result = my_config(values={"learning_rate": 0.05, "model_type": "svm"})
    assert result["learning_rate"] == 0.05
    assert result["model_type"] == "svm"


def test_return_single_object():
    """Test returning a single object with proper type preservation"""

    @config
    def model_config(hp: HP) -> MockModel:
        model_type = hp.select(["random_forest", "svm"], name="model_type")
        return MockModel(model_type)

    result = model_config()
    assert isinstance(result, MockModel)
    assert hasattr(result, "fit")
    assert hasattr(result, "predict")
    assert result.model_type == "random_forest"


def test_return_single_object_with_override():
    """Test single object return with values override"""

    @config
    def model_config(hp: HP) -> MockModel:
        model_type = hp.select(["random_forest", "svm"], name="model_type")
        return MockModel(model_type)

    result = model_config(values={"model_type": "svm"})
    assert result.model_type == "svm"


def test_return_dataclass():
    """Test returning a dataclass with typed access"""

    @config
    def training_config(hp: HP) -> ModelConfig:
        learning_rate = hp.number(0.01, name="learning_rate")
        epochs = hp.int(10, name="epochs")
        model_type = hp.select(["rf", "svm"], name="model_type")

        return ModelConfig(learning_rate=learning_rate, epochs=epochs, model_type=model_type)

    result = training_config()
    assert isinstance(result, ModelConfig)
    assert result.learning_rate == 0.01
    assert result.epochs == 10
    assert result.model_type == "rf"


def test_return_namedtuple():
    """Test returning a NamedTuple with typed access"""

    @config
    def results_config(hp: HP) -> TrainingResult:
        accuracy = hp.number(0.95, name="accuracy")
        loss = hp.number(0.05, name="loss")
        model_name = hp.text("model_v1", name="model_name")

        return TrainingResult(accuracy=accuracy, loss=loss, model_name=model_name)

    result = results_config()
    assert isinstance(result, TrainingResult)
    assert result.accuracy == 0.95
    assert result.loss == 0.05
    assert result.model_name == "model_v1"


def test_no_return_raises_error():
    """Test that configs without return raise a clear error"""

    @config
    def bad_config(hp: HP):
        learning_rate = hp.number(0.01, name="learning_rate")
        # Missing return statement

    with pytest.raises(Exception) as exc_info:
        bad_config()

    # Should raise a clear error about missing return
    assert "return" in str(exc_info.value).lower()


def test_return_explicit_dict():
    """Test returning an explicit dict (not via hp.collect)"""

    @config
    def explicit_dict_config(hp: HP):
        learning_rate = hp.number(0.01, name="learning_rate")
        model_type = hp.select(["rf", "svm"], name="model_type")

        return {"lr": learning_rate, "model": model_type, "custom_field": "custom_value"}

    result = explicit_dict_config()
    assert isinstance(result, dict)
    assert result["lr"] == 0.01
    assert result["model"] == "rf"
    assert result["custom_field"] == "custom_value"


def test_no_runtime_filtering():
    """Test that whatever is returned is passed through as-is (no filtering)"""

    @config
    def unfiltered_config(hp: HP):
        learning_rate = hp.number(0.01, name="learning_rate")

        # Include some items that would have been filtered in old system
        def some_function():
            return "test"

        import os

        return {
            "learning_rate": learning_rate,
            "function": some_function,  # Functions would have been filtered before
            "module": os,  # Modules would have been filtered before
            "_private": "private_value",  # Private vars would have been filtered before
        }

    result = unfiltered_config()
    assert "learning_rate" in result
    assert "function" in result
    assert "module" in result
    assert "_private" in result
    assert callable(result["function"])
    assert result["_private"] == "private_value"
