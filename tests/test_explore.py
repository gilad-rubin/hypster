from __future__ import annotations

import json
from enum import Enum
from typing import Any, Dict

import pytest

import hypster
from hypster import HP
from hypster.explore import explore


def test_explore_is_exported_from_package() -> None:
    assert callable(getattr(hypster, "explore"))


def test_explore_prints_simple_tree(capsys) -> None:
    def config(hp: HP) -> Dict[str, Any]:
        output_mode = hp.select(["text", "structured"], name="output_mode", default="text")
        max_tokens = hp.int(100000, name="max_tokens", min=1000)
        system_prompt = hp.text("", name="system_prompt")
        return {
            "output_mode": output_mode,
            "max_tokens": max_tokens,
            "system_prompt": system_prompt,
        }

    result = explore(config)

    assert result is None
    assert (
        capsys.readouterr().out == "config\n"
        '├── output_mode: select = "text"  (options: ["text", "structured"])\n'
        "├── max_tokens: int = 100000  (min: 1000)\n"
        '└── system_prompt: text = ""\n'
    )


def test_explore_returns_schema_info_and_defaults() -> None:
    def config(hp: HP) -> Dict[str, Any]:
        batching = hp.select(["single", "split"], name="batching", default="split")
        temperature = hp.float(0.0, name="temperature", min=0.0, max=2.0)
        return {"batching": batching, "temperature": temperature}

    info = explore(config, return_info=True)

    assert info is not None
    assert info.defaults() == {
        "batching": "split",
        "temperature": 0.0,
    }
    assert info.to_dict() == {
        "name": "config",
        "parameters": [
            {
                "name": "batching",
                "path": "batching",
                "kind": "select",
                "default_value": "split",
                "selected_value": "split",
                "options": ["single", "split"],
                "minimum": None,
                "maximum": None,
                "children": [],
            },
            {
                "name": "temperature",
                "path": "temperature",
                "kind": "float",
                "default_value": 0.0,
                "selected_value": 0.0,
                "options": None,
                "minimum": 0.0,
                "maximum": 2.0,
                "children": [],
            },
        ],
    }


def test_explore_tracks_nested_defaults_with_prefixed_paths() -> None:
    def gemini(hp: HP) -> Dict[str, Any]:
        model = hp.select(["flash-lite", "pro"], name="model", default="flash-lite")
        return {"model": model}

    def query_llm(hp: HP) -> Dict[str, Any]:
        provider = hp.select(["gemini", "openai"], name="provider", default="gemini")
        if provider == "gemini":
            provider_config = hp.nest(gemini, name="gemini")
        else:
            provider_config = {"model": hp.select(["gpt-4o-mini"], name="model", default="gpt-4o-mini")}
        return {"provider": provider, "config": provider_config}

    def root(hp: HP) -> Dict[str, Any]:
        return {
            "query_llm": hp.nest(query_llm, name="query_llm"),
            "system_prompt": hp.text("", name="system_prompt"),
        }

    info = explore(root, return_info=True)

    assert info is not None
    assert info.defaults() == {
        "query_llm.provider": "gemini",
        "query_llm.gemini.model": "flash-lite",
        "system_prompt": "",
    }
    assert info.format_tree() == (
        "root\n"
        "├── query_llm\n"
        '│   ├── provider: select = "gemini"  (options: ["gemini", "openai"])\n'
        "│   └── gemini\n"
        '│       └── model: select = "flash-lite"  (options: ["flash-lite", "pro"])\n'
        '└── system_prompt: text = ""'
    )


def test_explore_values_select_a_different_branch() -> None:
    def gemini(hp: HP) -> Dict[str, Any]:
        return {"model": hp.select(["flash-lite", "pro"], name="model", default="flash-lite")}

    def openai(hp: HP) -> Dict[str, Any]:
        model = hp.select(["gpt-4o-mini", "gpt-4.1"], name="model", default="gpt-4o-mini")
        temperature = hp.float(0.2, name="temperature", min=0.0, max=2.0)
        return {"model": model, "temperature": temperature}

    def query_llm(hp: HP) -> Dict[str, Any]:
        provider = hp.select(["gemini", "openai"], name="provider", default="gemini")
        if provider == "gemini":
            provider_config = hp.nest(gemini, name="gemini")
        else:
            provider_config = hp.nest(openai, name="openai")
        return {"provider": provider, "config": provider_config}

    info = explore(
        query_llm,
        values={"provider": "openai", "openai.temperature": 0.7},
        return_info=True,
    )

    assert info is not None
    assert info.defaults() == {
        "provider": "gemini",
        "openai.model": "gpt-4o-mini",
        "openai.temperature": 0.2,
    }
    assert info.format_tree() == (
        "query_llm\n"
        '├── provider: select = "openai"  (options: ["gemini", "openai"])\n'
        "└── openai\n"
        '    ├── model: select = "gpt-4o-mini"  (options: ["gpt-4o-mini", "gpt-4.1"])\n'
        "    └── temperature: float = 0.7  (0.0-2.0)"
    )


def test_explore_warns_on_unknown_or_unreachable_values() -> None:
    def config(hp: HP) -> Dict[str, Any]:
        mode = hp.select(["a", "b"], name="mode", default="a")
        if mode == "a":
            hp.int(1, name="count")
        return {"mode": mode}

    with pytest.warns(UserWarning, match="Unknown or unreachable parameters"):
        explore(config, values={"missing": 123})

    with pytest.warns(UserWarning, match="Unknown or unreachable parameters"):
        explore(config, values={"mode": "b", "count": 9})


def test_explore_can_raise_on_unknown_values() -> None:
    def config(hp: HP) -> Dict[str, Any]:
        hp.int(5, name="batch_size")
        return {"batch_size": 5}

    with pytest.raises(ValueError, match="Unknown or unreachable parameters"):
        explore(config, values={"batchsize": 7}, on_unknown="raise")


def test_explore_to_dict_is_json_serializable() -> None:
    class Provider(Enum):
        OPENAI = "openai"
        GEMINI = "gemini"

    def config(hp: HP) -> Dict[str, Any]:
        provider = hp.select([Provider.OPENAI, Provider.GEMINI], name="provider", default=Provider.OPENAI)
        return {"provider": provider}

    info = explore(config, return_info=True)

    assert info is not None
    assert info.to_dict() == {
        "name": "config",
        "parameters": [
            {
                "name": "provider",
                "path": "provider",
                "kind": "select",
                "default_value": "openai",
                "selected_value": "openai",
                "options": ["openai", "gemini"],
                "minimum": None,
                "maximum": None,
                "children": [],
            }
        ],
    }
    assert json.dumps(info.to_dict())
