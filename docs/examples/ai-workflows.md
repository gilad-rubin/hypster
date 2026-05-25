# AI Workflows

Hypster is useful when an AI workflow needs to switch providers, retrieval modes, prompts, safety settings, or output formats while keeping a replayable record of the selected path.

## Provider Configs

{% code overflow="wrap" %}
```python
from hypster import HP, explore, instantiate_with_params
from my_app.llms import GeminiClient, OpenAIClient

def openai_config(hp: HP) -> OpenAIClient:
    model_name = hp.select(
        ["gpt-5.5-mini", "gpt-5.5"],
        name="model_name",
        default="gpt-5.5-mini",
        options_only=True,
    )
    temperature = hp.float(0.2, name="temperature", min=0.0, max=2.0)
    max_tokens = hp.int(1024, name="max_tokens", min=1, max=16_384)
    reasoning_effort = hp.select(
        [None, "low", "medium", "high"],
        name="reasoning_effort",
        default=None,
        allow_none=True,
        options_only=True,
    )
    return OpenAIClient(
        model=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        reasoning_effort=reasoning_effort,
    )

def gemini_config(hp: HP) -> GeminiClient:
    model_name = hp.select(
        ["gemini-3.5-flash", "gemini-3.1-pro-preview"],
        name="model_name",
        default="gemini-3.5-flash",
        options_only=True,
    )
    temperature = hp.float(0.3, name="temperature", min=0.0, max=2.0)
    max_tokens = hp.int(2048, name="max_tokens", min=1, max=16_384)
    return GeminiClient(model=model_name, temperature=temperature, max_tokens=max_tokens)

provider_options = {
    "openai": openai_config,
    "gemini": gemini_config,
}
```
{% endcode %}

These examples assume client and retriever constructors are local, cheap, and lazy. If construction opens network connections, loads indexes, or calls paid APIs, keep that work outside the config and build it after `instantiate()`.

## RAG And Output Settings

Use dict-backed `select` when the runtime value is complex. The parameter records the simple key, while your app receives the mapped object or selected config function.

{% code overflow="wrap" %}
```python
from my_app.rag import BM25Retriever, DenseRetriever, HybridRetriever
from my_app.rendering import JsonRenderer, MarkdownRenderer, TextRenderer

def keyword_retriever(hp: HP) -> BM25Retriever:
    index_name = hp.text("docs-v1", name="index_name")
    top_k = hp.int(8, name="top_k", min=1, max=50)
    return BM25Retriever(index=index_name, top_k=top_k)

def vector_retriever(hp: HP) -> DenseRetriever:
    index_name = hp.text("docs-embeddings-v3", name="index_name")
    top_k = hp.int(8, name="top_k", min=1, max=50)
    return DenseRetriever(index=index_name, top_k=top_k)

def hybrid_retriever(hp: HP) -> HybridRetriever:
    index_name = hp.text("docs-hybrid-v2", name="index_name")
    keyword_weight = hp.float(0.35, name="keyword_weight", min=0.0, max=1.0)
    top_k = hp.int(8, name="top_k", min=1, max=50)
    return HybridRetriever(index=index_name, keyword_weight=keyword_weight, top_k=top_k)

retriever_options = {
    "keyword": keyword_retriever,
    "vector": vector_retriever,
    "hybrid": hybrid_retriever,
}

def retrieval_config(hp: HP):
    selected_config = hp.select(retriever_options, name="retriever_kind", default="hybrid", options_only=True)
    return hp.nest(selected_config, name="retriever")

def output_config(hp: HP):
    renderer_cls = hp.select(
        {
            "text": TextRenderer,
            "json": JsonRenderer,
            "markdown": MarkdownRenderer,
        },
        name="format",
        default="text",
        options_only=True,
    )
    include_citations = hp.bool(True, name="include_citations")
    system_prompt = hp.text("Answer with concise, sourced reasoning.", name="system_prompt")
    return renderer_cls(
        include_citations=include_citations,
        system_prompt=system_prompt,
    )
```
{% endcode %}

## Compose The Workflow

{% code overflow="wrap" %}
```python
from my_app.workflows import QAWorkflow

def qa_workflow_config(hp: HP) -> QAWorkflow:
    selected_provider = hp.select(provider_options, name="provider", default="openai", options_only=True)
    llm = hp.nest(selected_provider, name="llm")
    retriever = hp.nest(retrieval_config, name="retrieval")
    output = hp.nest(output_config, name="output")

    return QAWorkflow(
        llm=llm,
        retriever=retriever,
        output=output,
    )
```
{% endcode %}

## Explore And Replay

{% code overflow="wrap" %}
```python
explore(
    qa_workflow_config,
    values={"provider": "gemini", "llm.temperature": 0.1, "retrieval.retriever.top_k": 12},
)

run = instantiate_with_params(
    qa_workflow_config,
    values={
        "provider": "gemini",
        "llm.model_name": "gemini-3.1-pro-preview",
        "llm.temperature": 0.1,
        "retrieval.retriever_kind": "vector",
        "output.format": "markdown",
    },
)

assert run.params["provider"] == "gemini"
assert run.params["llm.model_name"] == "gemini-3.1-pro-preview"
assert run.params["retrieval.retriever_kind"] == "vector"
assert run.value.retriever.index == "docs-embeddings-v3"
```
{% endcode %}

## Branch Safety

This raises by default because `llm.reasoning_effort` is only reachable when the `openai` branch is selected:

{% code overflow="wrap" %}
```python
from hypster import instantiate

instantiate(
    qa_workflow_config,
    values={"provider": "gemini", "llm.reasoning_effort": "high"},
)
```
{% endcode %}

Run `explore(qa_workflow_config, values={"provider": "gemini"})` before instantiation to inspect the reachable parameter paths for that branch.
