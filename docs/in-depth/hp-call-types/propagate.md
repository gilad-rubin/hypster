# Nested Configurations

Hypster enables hierarchical configuration management through the `hp.propagate()` method, allowing you to compose complex configurations from smaller, reusable components.

> For an in depth tutorial, please check out the article on Medium: [**Implementing Modular-RAG using Haystack and Hypster**](https://towardsdatascience.com/implementing-modular-rag-with-haystack-and-hypster-d2f0ecc88b8f)

## `propagate` Function Signature

```python
def propagate(
    config_func: Union[str, Path, "Hypster"],
    *,
    name: Optional[str] = None,
    final_vars: List[str] = [],
    exclude_vars: List[str] = [],
    values: Dict[str, Any] = {}
) -> Dict[str, Any]
```

#### Parameters

* `config_func`: Either a path to a saved configuration or a Hypster config object
* `name`: Optional name for the nested configuration (used in dot notation)
* `final_vars`: List of variables that cannot be modified by parent configs
* `exclude_vars`: List of variables to exclude from the configuration
* `values`: Dictionary of values to override in the nested configuration

## Steps for propagation

{% stepper %}
{% step %}
### Define a reusable config

```python
from hypster import config, HP

@config
def llm_config(hp: HP):
    model = hp.select({
        "haiku": "claude-3-haiku-20240307",
        "sonnet": "claude-3-sonnet-20240229"
    }, default="haiku")
    temperature = hp.number(0.7, min=0, max=1)
```
{% endstep %}

{% step %}
### Save it

```python
llm_config.save("configs/llm.py")
```
{% endstep %}

{% step %}
### Define a parent config and use `hp.propagate`

```python
@config
def qa_config(hp: HP):
    # propagate LLM configuration
    llm = hp.propagate("configs/llm.py")

    # Add QA-specific parameters
    max_context_length = hp.int(1000, min=100, max=2000)

    # Combine LLM and QA parameters
    qa_pipeline = QAPipeline(
        model=llm["model"],
        temperature=llm["temperature"],
        max_context_length=max_context_length
    )
```
{% endstep %}

{% step %}
### Instantiate using dot notation

```python
qa_config(values={
    "llm.model": "sonnet",
    "llm.temperature": 0.5,
    "max_context_length": 1500
})
```
{% endstep %}
{% endstepper %}

## Configuration Sources

`hp.propagate()` accepts two types of sources:

### Path to Configuration File

```python
llm = hp.propagate("configs/llm.py")
```

### Direct Configuration Object

```python
from hypster import load

# Load the configuration
llm_config = load("configs/llm.py")

# Use the loaded config
qa_config = hp.propagate(llm_config)
```

## Value Assignment

Values for nested configurations can be set using either dot notation or nested dictionaries:

```python
# Using dot notation
qa_config(values={
    "llm.model": "sonnet",
    "llm.temperature": 0.5,
    "max_context_length": 1500
})

# Using nested dictionary
qa_config(values={
    "llm": {
        "model": "sonnet",
        "temperature": 0.5
    },
    "max_context_length": 1500
})
```

## Hierarchical Propagations

Configurations can be nested multiple times to create modular, reusable components:

```python
@config
def indexing_config(hp: HP):
    # Reuse LLM config for document processing
    llm = hp.propagate("configs/llm.py")

    # Indexing-specific parameters
    embedding_dim = hp.int(512, min=128, max=1024)

    # Process documents with LLM
    enriched_docs = process_documents(
        llm=llm["model"],
        temperature=llm["temperature"],
        embedding_dim=embedding_dim
    )

@config
def rag_config(hp: HP):
    # Reuse indexing config (which includes LLM config)
    indexing = hp.propagate("configs/indexing.py")

    # Add retrieval configuration
    retrieval = hp.propagate("configs/retrieval.py")
```

## Passing Values to Nested Configs

Use the `values` parameter to pass dependent values to nested configuration values:

```python
retrieval = hp.propagate(
    "configs/retrieval.py",
    values={
        "embedding_dim": indexing["embedding_dim"],
        "top_k": 5
    }
)
```

`final_vars` and `exclude_vars` are also supported.

## Best Practices

1. **Modular Design**
   * Create small, focused configurations for specific components
   * Combine configurations only when there are clear dependencies
   * Keep configurations reusable across different use cases
2.  **Clear Naming**

    ```python
    # Use descriptive names for propagated configs
    llm = hp.propagate("configs/llm.py", name="llm")
    indexer = hp.propagate("configs/indexer.py", name="indexer")
    ```
3.  **Value Dependencies**

    ```python
    # Explicitly pass dependent values
    retriever = hp.propagate(
        "configs/retriever.py",
        values={"embedding_dim": embedder["embedding_dim"]}
    )
    ```
4.  **File Organization**

    ```python
    # Keep related configs in a dedicated directory
    configs/
    ├── llm.py
    ├── indexing.py
    ├── retrieval.py
    └── rag.py
    ```
