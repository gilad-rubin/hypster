# Haystack

Haystack pipelines often have swappable retrievers, rankers, prompts, and generation models. Hypster can select those pieces before you build or run the pipeline.

## Shape

```python
from hypster import HP, instantiate_with_params

def retrieval_config(hp: HP):
    return hp.select(
        {
            "keyword": {"type": "bm25", "index": "docs"},
            "vector": {"type": "embedding", "index": "docs-embeddings"},
            "hybrid": {"type": "hybrid", "keyword_weight": 0.35},
        },
        name="kind",
        default="hybrid",
        options_only=True,
    )

def haystack_pipeline_config(hp: HP):
    return {
        "retrieval": hp.nest(retrieval_config, name="retrieval"),
        "top_k": hp.int(8, name="top_k", min=1, max=50),
        "rerank": hp.bool(True, name="rerank"),
        "answer_style": hp.select(["brief", "sourced"], name="answer_style", default="sourced", options_only=True),
    }

run = instantiate_with_params(
    haystack_pipeline_config,
    values={"retrieval.kind": "vector", "top_k": 12},
)

# pipeline = build_haystack_pipeline(run.value)
# tracker.log_params(run.params)
```

## Build The Pipeline After Instantiation

Keep Hypster configs focused on replayable settings. Build Haystack components after `instantiate_with_params()` so exploration and UI generation do not open indexes, clients, or network connections.

```python
def build_haystack_pipeline(settings):
    retrieval = settings["retrieval"]

    if retrieval["type"] == "bm25":
        retriever = build_bm25_retriever(index=retrieval["index"], top_k=settings["top_k"])
    elif retrieval["type"] == "embedding":
        retriever = build_embedding_retriever(index=retrieval["index"], top_k=settings["top_k"])
    else:
        retriever = build_hybrid_retriever(
            keyword_weight=retrieval["keyword_weight"],
            top_k=settings["top_k"],
        )

    pipeline = make_pipeline(
        retriever=retriever,
        rerank=settings["rerank"],
        answer_style=settings["answer_style"],
    )
    return pipeline


run = instantiate_with_params(
    haystack_pipeline_config,
    values={"retrieval.kind": "hybrid", "answer_style": "sourced"},
)

pipeline = build_haystack_pipeline(run.value)
tracker.log_params(run.params)
```

If your components are cheap pure-Python objects, a config can return factories or initialized components. For retrievers, indexes, remote LLM clients, and pipelines that allocate resources, prefer returning lightweight settings and building the Haystack pipeline outside the config function.

Hypster does not ship a Haystack-specific adapter today. Use this pattern when you want replayable parameter selection around an existing Haystack builder.
