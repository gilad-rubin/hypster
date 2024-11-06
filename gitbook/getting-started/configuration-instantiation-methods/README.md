# Configuration Instantiation Methods

## Manual Instantiation

When working with configuration functions, you can directly specify parameter values using a dictionary:

```
pythonresults = modular_rag(
    values={
        "indexing.enrich_doc_w_llm": True,
        "indexing.llm.model": "gpt-4o-mini",
        "document_store_type": "qdrant",
        "retrieval.bm25_weight": 0.8,
        "embedder_type": "fastembed",
        "reranker.model": "tiny-bert-v2",
        "response.llm.model": "haiku",
        "indexing.splitter.split_length": 6,
        "reranker.top_k": 3,
    },
)
```

However, as configuration spaces become more complex, manual instantiation can become challenging. Consider this example:\
\[Add your toy example here]This becomes particularly difficult when:

* Dealing with nested configurations
* Managing multiple conditional parameters
* Ensuring parameter value validity

### Interactive Instantiation

To address these challenges, hypster provides a built-in Jupyter Notebook-based UI that allows you to:

* Interactively select valid configurations
* Visualize parameter dependencies
* Work directly within your IDE using ipywidgets
