"""Test different return types and pass-through semantics."""

from dataclasses import dataclass
from typing import Any, Dict

from hypster import HP, instantiate


def test_dict_return() -> None:
    """Test returning a plain dict."""

    def config(hp: HP) -> Dict[str, Any]:
        a = hp.int(1, name="a")
        b = hp.float(2.0, name="b")
        return {"a": a, "b": b}

    result = instantiate(config)
    assert result == {"a": 1, "b": 2.0}


def test_single_object_return() -> None:
    """Test returning a single object."""

    class Model:
        def __init__(self, n_estimators: int) -> None:
            self.n_estimators = n_estimators

    def config(hp: HP) -> Model:
        n = hp.int(100, name="n_estimators")
        return Model(n_estimators=n)

    result = instantiate(config, values={"n_estimators": 200})
    assert isinstance(result, Model)
    assert result.n_estimators == 200


def test_dataclass_return() -> None:
    """Test returning a dataclass."""

    @dataclass
    class Config:
        lr: float
        epochs: int

    def config(hp: HP) -> Config:
        return Config(lr=hp.float(0.1, name="lr"), epochs=hp.int(10, name="epochs"))

    result = instantiate(config, values={"lr": 0.05})
    assert isinstance(result, Config)
    assert result.lr == 0.05
    assert result.epochs == 10


def test_collect_helper() -> None:
    """Test hp.collect for filtering locals."""

    def config(hp: HP) -> Dict[str, Any]:
        a = hp.int(1, name="a")
        b = hp.int(2, name="b")
        _temp = "ignored"

        def helper_func(x):
            return x

        return hp.collect(locals(), exclude=["_temp", "helper_func"])

    result = instantiate(config)
    assert "a" in result
    assert "b" in result
    assert "_temp" not in result
    assert "helper_func" not in result
