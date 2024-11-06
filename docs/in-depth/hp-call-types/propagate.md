# Configuration Propagation

Hypster enables hierarchical configuration management through the `hp.propagate()` method, allowing you to compose complex configurations from smaller, reusable components.

## Basic Propagation

#TODO this is an incomplete example. I want to show that the main config (rag, but the name can change) has its own logic, otherwise it's just a normal config. so either add something there or use two propagations... I think I prefer somethihg simple with 1 propagation to start with.

```python
from hypster import config, HP

@config
def llm_config(hp: HP):
    model = hp.select({
        "haiku": "claude-3-haiku-20240307",
        "sonnet": "claude-3-sonnet-20240229"
    }, default="haiku")
    temperature = hp.number(0.7, min=0, max=1)

# Save configuration for reuse
llm_config.save("configs/llm.py")

@config
def rag_config(hp: HP):
    # Load and propagate LLM configuration
    llm = hp.propagate("configs/llm.py")

    # Use propagated values
    generator = Generator(
        model=llm["model"],
        temperature=llm["temperature"]
    )
```

## Configuration Sources

`hp.propagate()` accepts two types of sources:
- Path to saved configuration: `hp.propagate("configs/llm.py")`
- Direct Hypster config object: `hp.propagate(llm_config)`
  #TODO: show that in the second case, it needs llm_config = hypster.load("configs/llm.py") and then pass that in.
  # since  hypster needs to import everything inside of the config - we need to show a simple example of that. def ... then from hypster import load,,,,

## Value Assignment

Values for nested configurations can be set using either dot notation or nested dictionaries:

```python
# Using dot notation
rag_config(values={
    "llm.model": "sonnet",
    "llm.temperature": 0.5
})

# Using nested dictionary
rag_config(values={
    "llm": {
        "model": "sonnet",
        "temperature": 0.5
    }
})
```

## multiple propagations
#TODO: here make the complexity about the propagation, not match etc... create another config and then build the modular rag.
# you can show how the indexing pipeline config uses llm_config. nesting dolls... just like in the modular_rag example.

```python
@config
def modular_rag(hp: HP):
    # Base indexing configuration
    indexing = hp.propagate("configs/indexing.py")

    # Conditional embedder configuration
    embedder_type = hp.select(["fastembed", "jina"], default="fastembed")
    match embedder_type:
        case "fastembed":
            embedder = hp.propagate("configs/fast_embed.py")
        case "jina":
            embedder = hp.propagate(
                "configs/jina_embed.py",
                values={"late_chunking": True}
            )

    # Retrieval with dependent parameters
    retrieval = hp.propagate(
        "configs/retrieval.py",
        values={"embedding_dim": embedder["embedding_dim"]}
    )
```
#TODO: add a section about the nested values, final_vars, exclude_vars, etc. and why use them.
in the modular example I had to pass a value of the emebedding size to the document store.

values = {...} overrides these values that are in the propagate(...).

## Best Practices
#todo this needs a lot of expansion.

1. prefer small, modular, reusable configs (like "llm", "retrieval", "indexing", etc). merge them into larger configs only when strictly necessary.
2.
3. **Modular Design**
   ```python
   # Split configurations into logical components
   llm_config.save("configs/llm.py")
   indexer_config.save("configs/indexer.py")
   retriever_config.save("configs/retriever.py")
   ```

I'll help improve each section based on the TODOs:

# Configuration Propagation

Hypster enables hierarchical configuration management through the `hp.propagate()` method, allowing you to compose complex configurations from smaller, reusable components.

## Basic Propagation

```python
from hypster import config, HP

@config
def llm_config(hp: HP):
    model = hp.select({
        "haiku": "claude-3-haiku-20240307",
        "sonnet": "claude-3-sonnet-20240229"
    }, default="haiku")
    temperature = hp.number(0.7, min=0, max=1)

# Save configuration for reuse
llm_config.save("configs/llm.py")

@config
def qa_config(hp: HP):
    # Load and propagate LLM configuration
    llm = hp.propagate("configs/llm.py")

    # Add QA-specific parameters
    max_tokens = hp.int(1000, min=100, max=2000)

    # Combine LLM and QA parameters
    qa_pipeline = QAPipeline(
        model=llm["model"],
        temperature=llm["temperature"],
        max_tokens=max_tokens
    )
```

## Configuration Sources

`hp.propagate()` accepts two types of sources:

### Path to Configuration File
```python
llm = hp.propagate("configs/llm.py")
```

### Direct Configuration Object
```python
def qa_config(hp: HP):
    from hypster import load # all imports need to be inside the function

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

## Multiple Propagations

Configurations can be nested multiple times. Notice how the indexing config uses the LLM config.

```python
@config
def indexing_config(hp: HP):
    # Reuse LLM config for document processing
    llm = hp.propagate("configs/llm.py")

    # Indexing-specific parameters
    chunk_size = hp.int(512, min=128, max=1024)

    # Process documents with LLM
    enriched_docs = process_documents(
        llm=llm["model"],
        temperature=llm["temperature"],
        chunk_size=chunk_size
    )

@config
def rag_config(hp: HP):
    # Reuse indexing config (which includes LLM config)
    indexing = hp.propagate("configs/indexing.py")

    # Add retrieval configuration
    retrieval = hp.propagate(
        "configs/retrieval.py",
        values={"embedding_dim": indexing["embedding_dim"]}
    )

    pipeline = Pipeline([indexing, retrieval])
```

## Advanced Configuration Options

### Passing Values to Nested Configs
Use the `values` parameter to pass values to nested configurations:

```python
retrieval = hp.propagate(
    "configs/retrieval.py",
    values={
        "embedding_dim": indexing["embedding_dim"],
    }
)
```
nested `final_vars` and `exclude_vars` are also supported.

## Best Practices

1. **Modular Design**
   - Create small, focused configurations for specific components (like "llm", "retrieval", "indexing", etc).
   - Combine configurations only when there are clear dependencies.
   - Keep configurations reusable across different use cases.

2. **File Organization**
   ```python
   # Keep related configs in a dedicated directory
   configs/
   ├── llm.py
   ├── indexing.py
   ├── retrieval.py
   └── rag.py
   ```
