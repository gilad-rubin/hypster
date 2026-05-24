# AI Workflows

Hypster is useful when an AI workflow needs to switch providers, retrieval modes, prompts, safety settings, or output formats while keeping a replayable record of the selected path.

## Provider Configs

{% code overflow="wrap" %}
```python
from dataclasses import dataclass
from hypster import HP, explore, instantiate_with_params

@dataclass
class ProviderConfig:
    provider: str
    model: str
    temperature: float
    max_tokens: int

def openai_config(hp: HP) -> ProviderConfig:
    return ProviderConfig(
        provider="openai",
        model=hp.select(
            ["gpt-5.4-mini", "gpt-5.5"],
            name="model",
            default="gpt-5.4-mini",
            options_only=True,
        ),
        temperature=hp.float(0.2, name="temperature", min=0.0, max=2.0),
        max_tokens=hp.int(1024, name="max_tokens", min=1, max=16_384),
    )

def gemini_config(hp: HP) -> ProviderConfig:
    return ProviderConfig(
        provider="gemini",
        model=hp.select(
            ["gemini-3.5-flash", "gemini-3.1-pro-preview"],
            name="model",
            default="gemini-3.5-flash",
            options_only=True,
        ),
        temperature=hp.float(0.3, name="temperature", min=0.0, max=2.0),
        max_tokens=hp.int(2048, name="max_tokens", min=1, max=16_384),
    )
```
{% endcode %}

## RAG And Output Settings

Use dict-backed `select` when the runtime value is complex. The parameter records the simple key, while your app receives the mapped object.

{% code overflow="wrap" %}
```python
def retrieval_config(hp: HP):
    retriever = hp.select(
        {
            "keyword": {"kind": "bm25", "index": "docs-v1"},
            "vector": {"kind": "dense", "index": "docs-embeddings-v3"},
            "hybrid": {"kind": "hybrid", "keyword_weight": 0.35},
        },
        name="retriever",
        default="hybrid",
        options_only=True,
    )
    return {
        "retriever": retriever,
        "top_k": hp.int(8, name="top_k", min=1, max=50),
        "rerank": hp.bool(True, name="rerank"),
    }

def output_config(hp: HP):
    return {
        "format": hp.select(["text", "json", "markdown"], name="format", default="text", options_only=True),
        "include_citations": hp.bool(True, name="include_citations"),
        "system_prompt": hp.text("Answer with concise, sourced reasoning.", name="system_prompt"),
    }
```
{% endcode %}

## Compose The Workflow

{% code overflow="wrap" %}
```python
def qa_workflow_config(hp: HP):
    provider = hp.select(["openai", "gemini"], name="provider", default="openai", options_only=True)

    if provider == "openai":
        llm = hp.nest(openai_config, name="openai")
    else:
        llm = hp.nest(gemini_config, name="gemini")

    return {
        "provider": provider,
        "llm": llm,
        "retrieval": hp.nest(retrieval_config, name="retrieval"),
        "output": hp.nest(output_config, name="output"),
    }
```
{% endcode %}

## Explore And Replay

{% code overflow="wrap" %}
```python
explore(
    qa_workflow_config,
    values={"provider": "gemini", "gemini.temperature": 0.1, "retrieval.top_k": 12},
)

run = instantiate_with_params(
    qa_workflow_config,
    values={
        "provider": "gemini",
        "gemini.model": "gemini-3.1-pro-preview",
        "gemini.temperature": 0.1,
        "retrieval.retriever": "vector",
        "output.format": "markdown",
    },
)

assert run.params["provider"] == "gemini"
assert run.params["gemini.model"] == "gemini-3.1-pro-preview"
assert run.params["retrieval.retriever"] == "vector"
assert run.value["retrieval"]["retriever"]["kind"] == "dense"
```
{% endcode %}

## Branch Safety

This raises by default because `openai.temperature` is unreachable when the `gemini` branch is selected:

{% code overflow="wrap" %}
```python
from hypster import instantiate

instantiate(
    qa_workflow_config,
    values={"provider": "gemini", "openai.temperature": 0.7},
)
```
{% endcode %}

Run `explore(qa_workflow_config, values={"provider": "gemini"})` before instantiation to inspect the reachable parameter paths for that branch.
