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


def test_explore_prints_unicode_text_without_ascii_escaping(capsys) -> None:
    def config(hp: HP) -> Dict[str, Any]:
        prompt = hp.text("שלום\n你好\nمرحبا", name="prompt")
        return {"prompt": prompt}

    result = explore(config)

    assert result is None
    assert capsys.readouterr().out == 'config\n└── prompt: text = "שלום\\n你好\\nمرحبا"\n'


def test_explore_returns_schema_info_and_defaults() -> None:
    def config(hp: HP) -> Dict[str, Any]:
        batching = hp.select(["single", "split"], name="batching", default="split")
        temperature = hp.float(0.0, name="temperature", min=0.0, max=2.0)
        return {"batching": batching, "temperature": temperature}

    info = explore(config, return_schema=True)

    assert info is not None
    assert info.defaults() == {
        "batching": "split",
        "temperature": 0.0,
    }
    assert info.to_dict() == {
        "name": "config",
        "display_label": "Config",
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
                "description": None,
                "display_label": "Batching",
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
                "description": None,
                "display_label": "Temperature",
                "children": [],
            },
        ],
    }


def test_explore_forwards_execution_kwargs_to_config() -> None:
    def config(hp: HP, default_batching: str) -> Dict[str, Any]:
        batching = hp.select(["single", "split"], name="batching", default=default_batching)
        return {"batching": batching}

    info = explore(config, return_schema=True, default_batching="single")

    assert info is not None
    assert info.defaults() == {"batching": "single"}


def test_explore_rejects_old_return_info_flag() -> None:
    calls = []

    def config(hp: HP, **execution_kwargs: object) -> Dict[str, int]:
        calls.append(execution_kwargs)
        return {"x": hp.int(1, name="x")}

    with pytest.raises(TypeError, match=r"explore\(\) reserves return_info=.*return_schema=True"):
        explore(config, return_info=True)

    assert calls == []


def test_explore_includes_descriptions_and_display_labels() -> None:
    def retrieval(hp: HP) -> Dict[str, Any]:
        top_k = hp.int(
            8,
            name="top_k",
            min=1,
            max=20,
            description="Number of documents to return.",
        )
        return {"top_k": top_k}

    def config(hp: HP) -> Dict[str, Any]:
        return {
            "retrieval": hp.nest(
                retrieval,
                name="retrieval",
                description="Retrieval settings used before generation.",
            )
        }

    info = explore(config, return_schema=True)

    assert info is not None
    schema = info.to_dict()
    group = schema["parameters"][0]
    top_k = group["children"][0]

    assert schema["display_label"] == "Config"
    assert group["display_label"] == "Retrieval"
    assert group["description"] == "Retrieval settings used before generation."
    assert top_k["display_label"] == "Top K"
    assert top_k["description"] == "Number of documents to return."


def test_explore_includes_metadata_when_provided_and_omits_when_absent() -> None:
    prompt_metadata = {
        "editor": "prompt",
        "audience": "domain_expert",
        "tags": ["prompt", "metadata"],
        "layout": {"rows": 12},
        "examples": ("short", "detailed"),
    }

    def config(hp: HP) -> Dict[str, Any]:
        metadata_prompt = hp.text(
            "Extract concise document metadata for retrieval indexing.",
            name="metadata_prompt",
            description="Prompt used before chunking.",
            metadata=prompt_metadata,
        )
        chunk_size = hp.int(512, name="chunk_size")
        empty_label = hp.text("none", name="empty_label", metadata={})
        return {"metadata_prompt": metadata_prompt, "chunk_size": chunk_size, "empty_label": empty_label}

    result = hypster.instantiate(config)
    info = explore(config, return_schema=True)

    assert result == {
        "metadata_prompt": "Extract concise document metadata for retrieval indexing.",
        "chunk_size": 512,
        "empty_label": "none",
    }
    assert info is not None
    schema = info.to_dict()
    prompt = schema["parameters"][0]
    chunk_size = schema["parameters"][1]
    empty_label = schema["parameters"][2]

    assert prompt["metadata"] == {
        "editor": "prompt",
        "audience": "domain_expert",
        "tags": ["prompt", "metadata"],
        "layout": {"rows": 12},
        "examples": ["short", "detailed"],
    }
    assert prompt["description"] == "Prompt used before chunking."
    assert "metadata" not in chunk_size
    assert "metadata" not in empty_label
    assert json.dumps(schema)


def test_metadata_is_available_on_all_parameter_primitives() -> None:
    def config(hp: HP) -> Dict[str, Any]:
        return {
            "count": hp.int(1, name="count", metadata={"primitive": "int"}),
            "temperature": hp.float(0.2, name="temperature", metadata={"primitive": "float"}),
            "prompt": hp.text("hello", name="prompt", metadata={"primitive": "text"}),
            "enabled": hp.bool(True, name="enabled", metadata={"primitive": "bool"}),
            "mode": hp.select(["fast", "safe"], name="mode", metadata={"primitive": "select"}),
            "layers": hp.multi_int([64, 32], name="layers", metadata={"primitive": "multi_int"}),
            "weights": hp.multi_float([0.2, 0.8], name="weights", metadata={"primitive": "multi_float"}),
            "labels": hp.multi_text(["a", "b"], name="labels", metadata={"primitive": "multi_text"}),
            "flags": hp.multi_bool([True, False], name="flags", metadata={"primitive": "multi_bool"}),
            "features": hp.multi_select(["cache", "trace"], name="features", metadata={"primitive": "multi_select"}),
        }

    info = explore(config, return_schema=True)

    assert info is not None
    assert {node["path"]: node["metadata"]["primitive"] for node in info.to_dict()["parameters"]} == {
        "count": "int",
        "temperature": "float",
        "prompt": "text",
        "enabled": "bool",
        "mode": "select",
        "layers": "multi_int",
        "weights": "multi_float",
        "labels": "multi_text",
        "flags": "multi_bool",
        "features": "multi_select",
    }


@pytest.mark.parametrize(
    ("metadata", "match"),
    [
        (["prompt"], "metadata must be a dictionary"),
        ({1: "prompt"}, "keys must be strings"),
        ({"editor": object()}, r"metadata\['editor'\] must be JSON-compatible"),
        ({"weight": float("inf")}, "finite float"),
    ],
)
def test_metadata_must_be_json_compatible(metadata: object, match: str) -> None:
    def config(hp: HP) -> str:
        return hp.text("hello", name="prompt", metadata=metadata)

    with pytest.raises(ValueError, match=match):
        explore(config, return_schema=True)
    with pytest.raises(ValueError, match=match):
        hypster.instantiate(config)


def test_metadata_rejects_excessive_nesting() -> None:
    nested: Any = "leaf"
    for _ in range(102):
        nested = [nested]

    def config(hp: HP) -> str:
        return hp.text("hello", name="prompt", metadata={"nested": nested})

    with pytest.raises(ValueError, match="maximum metadata nesting depth"):
        explore(config, return_schema=True)


def test_metadata_rejects_excessive_mapping_nesting() -> None:
    nested: Any = "leaf"
    for _ in range(60):
        nested = {"child": nested}

    def config(hp: HP) -> str:
        return hp.text("hello", name="prompt", metadata={"nested": nested})

    with pytest.raises(ValueError, match="maximum metadata nesting depth"):
        explore(config, return_schema=True)


def test_value_errors_are_reported_before_metadata_errors() -> None:
    def numeric(hp: HP) -> int:
        return hp.int(1, name="count", min=0, max=2, metadata={"bad": object()})

    def categorical(hp: HP) -> str:
        return hp.select(["a"], name="mode", options_only=True, metadata={"bad": object()})

    with pytest.raises(ValueError, match="exceeds maximum"):
        explore(numeric, values={"count": 99}, return_schema=True)
    with pytest.raises(ValueError, match="not in allowed options"):
        explore(categorical, values={"mode": "b"}, return_schema=True)


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

    info = explore(root, return_schema=True)

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
        return_schema=True,
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


def test_explore_raises_on_unknown_or_unreachable_values_by_default() -> None:
    def config(hp: HP) -> Dict[str, Any]:
        mode = hp.select(["a", "b"], name="mode", default="a")
        if mode == "a":
            hp.int(1, name="count")
        return {"mode": mode}

    with pytest.raises(ValueError, match="Unknown or unreachable parameters"):
        explore(config, values={"missing": 123})

    with pytest.raises(ValueError, match="Unknown or unreachable parameters"):
        explore(config, values={"mode": "b", "count": 9})

    with pytest.warns(UserWarning, match="Unknown or unreachable parameters"):
        explore(config, values={"missing": 123}, on_unknown="warn")


def test_explore_can_raise_on_unknown_values() -> None:
    def config(hp: HP) -> Dict[str, Any]:
        hp.int(5, name="batch_size")
        return {"batch_size": 5}

    with pytest.raises(ValueError, match="Unknown or unreachable parameters"):
        explore(config, values={"batchsize": 7}, on_unknown="raise")


def test_explore_validates_on_unknown_before_execution() -> None:
    calls = []

    def config(hp: HP) -> Dict[str, Any]:
        calls.append("executed")
        return {"x": hp.int(1, name="x")}

    with pytest.raises(ValueError, match="on_unknown must be one of"):
        explore(config, on_unknown="silent")

    assert calls == []


def test_explore_to_dict_is_json_serializable() -> None:
    class Provider(Enum):
        OPENAI = "openai"
        GEMINI = "gemini"

    def config(hp: HP) -> Dict[str, Any]:
        provider = hp.select(
            {"openai": Provider.OPENAI, "gemini": Provider.GEMINI},
            name="provider",
            default="openai",
        )
        return {"provider": provider}

    info = explore(config, return_schema=True)

    assert info is not None
    assert info.to_dict() == {
        "name": "config",
        "display_label": "Config",
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
                "description": None,
                "display_label": "Provider",
                "children": [],
            }
        ],
    }
    assert json.dumps(info.to_dict())
