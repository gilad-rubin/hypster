from __future__ import annotations

from typing import Any, Dict

import pytest

from hypster import HP, instantiate, instantiate_with_params


def test_instantiate_with_params_returns_value_and_replayable_selected_params() -> None:
    def openai(hp: HP) -> Dict[str, Any]:
        return {
            "model": hp.select(["gpt-4o-mini", "gpt-4.1"], name="model", default="gpt-4o-mini"),
            "temperature": hp.float(0.2, name="temperature", min=0.0, max=2.0),
        }

    def config(hp: HP) -> Dict[str, Any]:
        provider = hp.select(["gemini", "openai"], name="provider", default="gemini")
        if provider == "openai":
            llm = hp.nest(openai, name="openai")
        else:
            llm = {"model": hp.select(["flash-lite", "pro"], name="model", default="flash-lite")}
        return {"provider": provider, "llm": llm}

    output = instantiate_with_params(config, values={"provider": "openai", "openai.temperature": 0.7})

    assert output.value == {
        "provider": "openai",
        "llm": {"model": "gpt-4o-mini", "temperature": 0.7},
    }
    assert output.params == {
        "provider": "openai",
        "openai.model": "gpt-4o-mini",
        "openai.temperature": 0.7,
    }
    assert instantiate(config, values=output.params) == output.value


def test_instantiate_with_params_forwards_execution_kwargs_without_recording_them() -> None:
    def config(hp: HP, multiplier: int) -> Dict[str, int]:
        base = hp.int(10, name="base")
        return {"result": base * multiplier}

    output = instantiate_with_params(config, multiplier=4)

    assert output.value == {"result": 40}
    assert output.params == {"base": 10}


def test_instantiate_with_params_validates_on_unknown_before_execution() -> None:
    calls = []

    def config(hp: HP) -> Dict[str, int]:
        calls.append("executed")
        return {"x": hp.int(1, name="x")}

    with pytest.raises(ValueError, match="on_unknown must be one of"):
        instantiate_with_params(config, on_unknown="silent")

    assert calls == []
