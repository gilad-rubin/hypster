# Interactive Instantiation (UI)

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



<figure><img src="../.gitbook/assets/image (9).png" alt=""><figcaption></figcaption></figure>

Once you start getting used to creating configuration spaces, they can get somewhat complex. This makes it hard to track manually and instantiate them in a valid way.

Let's look at a simple example:&#x20;

In this toy example, if we want to instantiate `my_config` - we have to remember that `var2` can only be defined if `var == "a"`. This becomes much more difficult when we start using nested configurations and multiple conditions.

To address this challenge - hypster offers a built-in Jupyter Notebook based UI to interactively select valid configurations inside your IDE using `ipywidgets`.
