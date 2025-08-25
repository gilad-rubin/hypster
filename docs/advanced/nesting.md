# Advanced Configuration Nesting & Registry System

Hypster provides powerful configuration composition capabilities through an enhanced nesting system that supports multiple resolution strategies, a global configuration registry, and seamless integration with file-based and module-based configurations.

{% hint style="info" %}
**Note**: All examples in this document require the `name` parameter for HP method calls. For brevity, some examples may omit this parameter, but in practice, you must include `name="parameter_name"` for all `hp.select()`, `hp.number()`, `hp.int()`, etc. calls.
{% endhint %}

## Configuration Registry

The configuration registry provides a global, namespace-organized system for managing and accessing configurations across your application.

### Basic Registry Usage

```python
from hypster import config, HP, registry

@config(register="llm.claude")
def claude_config(hp: HP):
    model = hp.select(["haiku", "sonnet"], default="haiku")
    temperature = hp.number(0.7, min=0, max=1)
    return {"model": model, "temperature": temperature}

@config(register="llm.openai")
def openai_config(hp: HP):
    model = hp.select(["gpt-3.5-turbo", "gpt-4"], default="gpt-3.5-turbo")
    temperature = hp.number(0.7, min=0, max=1)
    return {"model": model, "temperature": temperature}

# Now use registry in nesting
@config
def rag_config(hp: HP):
    # Registry lookup - tries "llm.claude" first
    llm = hp.nest("llm.claude")

    # QA-specific parameters
    max_tokens = hp.int(1000, min=100, max=2000)

    return {
        "llm": llm,
        "max_tokens": max_tokens
    }
```

### Namespace Organization

The registry supports hierarchical namespaces using dot notation:

```python
# Embedding configurations
@config(register="embeddings.openai")
def openai_embeddings(hp: HP): ...

@config(register="embeddings.sentence_transformers")
def st_embeddings(hp: HP): ...

# Retrieval configurations
@config(register="retrieval.vector")
def vector_retrieval(hp: HP): ...

@config(register="retrieval.hybrid")
def hybrid_retrieval(hp: HP): ...

# List available configurations by namespace
print(registry.list_configs("embeddings"))  # ['openai', 'sentence_transformers']
print(registry.list_configs("retrieval"))   # ['vector', 'hybrid']
```

## Enhanced Nesting Resolution

The `hp.nest()` method now supports multiple resolution strategies with the following priority order:

1. **Registry lookup** - Check the global configuration registry
2. **File loading** - Load from file path (`.py`, `.json`, `.yaml`)
3. **Module loading** - Import from Python module with optional object specification

### Resolution Examples

```python
@config
def main_config(hp: HP):
    # 1. Registry lookup (highest priority)
    llm = hp.nest("llm.claude")

    # 2. File loading with specific object
    embeddings = hp.nest("configs/embeddings.py:openai_config")

    # 3. Module loading
    retriever = hp.nest("my_package.configs.retrieval")

    # 4. Module with specific object
    reranker = hp.nest("my_package.models:cohere_reranker")
```

## File and Module Loading

### File Loading with Object Specification

Load specific configuration objects from Python files:

```python
# configs/models.py contains multiple configs
@config
def claude_config(hp: HP): ...

@config
def openai_config(hp: HP): ...

# Load specific config from file
claude_llm = hp.nest("configs/models.py:claude_config")
openai_llm = hp.nest("configs/models.py:openai_config")
```

### Module Loading

Import configurations directly from Python modules:

```python
# From installed package
transformers_config = hp.nest("transformers_configs.base")

# From local module with specific object
custom_config = hp.nest("my_app.configs.models:custom_transformer")
```

### External Import Preservation

When saving configurations that use external imports, Hypster automatically preserves the import statements:

```python
@config
def ml_config(hp: HP):
    from sklearn.ensemble import RandomForestClassifier
    from transformers import AutoTokenizer

    model_type = hp.select(["rf", "transformer"], default="rf")

    if model_type == "rf":
        return RandomForestClassifier(n_estimators=hp.int(100, min=10, max=500))
    else:
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        return {"tokenizer": tokenizer}

# When saved, imports are preserved in the output file
ml_config.save("configs/ml_model.py")
```

## Advanced Nesting Patterns

### Conditional Nesting

```python
@config
def adaptive_config(hp: HP):
    provider = hp.select(["openai", "anthropic", "local"], default="openai")

    # Dynamic registry lookup based on selection
    if provider == "openai":
        llm = hp.nest("llm.openai")
    elif provider == "anthropic":
        llm = hp.nest("llm.claude")
    else:
        llm = hp.nest("configs/local_models.py:llama_config")

    return {"provider": provider, "llm": llm}
```

### Multi-Level Hierarchical Nesting

```python
@config
def rag_pipeline(hp: HP):
    # Each component can be swapped independently
    embeddings = hp.nest("embeddings.sentence_transformers")
    retrieval = hp.nest("retrieval.hybrid")
    llm = hp.nest("llm.claude")

    # Reranking is optional
    use_reranking = hp.bool(True)
    if use_reranking:
        reranker = hp.nest("reranking.cohere")
    else:
        reranker = None

    return {
        "embeddings": embeddings,
        "retrieval": retrieval,
        "llm": llm,
        "reranker": reranker
    }
```

### Value Propagation Between Nested Configs

```python
@config
def coordinated_config(hp: HP):
    # Base configuration
    base = hp.nest("base.standard")

    # Propagate values to dependent configs
    embeddings = hp.nest(
        "embeddings.openai",
        values={"dimensions": base["embedding_dim"]}
    )

    retrieval = hp.nest(
        "retrieval.vector",
        values={
            "embedding_dim": base["embedding_dim"],
            "top_k": base["max_results"]
        }
    )

    return {
        "base": base,
        "embeddings": embeddings,
        "retrieval": retrieval
    }
```

## Registry Management

### Programmatic Registry Operations

```python
from hypster import registry

# Check if config exists
if registry.contains("llm.claude"):
    llm = hp.nest("llm.claude")

# List all configurations
all_configs = registry.list_configs()
print(f"Available configs: {all_configs}")

# List by namespace
llm_configs = registry.list_configs("llm")
print(f"LLM configs: {llm_configs}")

# Clear registry (useful for testing)
registry.clear()
```

### Runtime Registration

```python
# Register configurations at runtime
def create_dynamic_config():
    @config
    def dynamic_llm(hp: HP):
        return {"model": "dynamic-model"}

    registry.register("llm.dynamic", dynamic_llm)
    return dynamic_llm

# Use the dynamically registered config
dynamic_config = create_dynamic_config()
llm = hp.nest("llm.dynamic")
```

## Best Practices

### Namespace Design

```python
# Use hierarchical namespaces for organization
@config(register="ml.models.classification.random_forest")
def rf_config(hp: HP): ...

@config(register="ml.models.classification.svm")
def svm_config(hp: HP): ...

@config(register="ml.preprocessing.scaling.standard")
def standard_scaler_config(hp: HP): ...
```

### Configuration Discovery

```python
@config
def auto_discover_config(hp: HP):
    # Dynamically discover available models
    available_models = registry.list_configs("ml.models.classification")

    model_type = hp.select(available_models, default=available_models[0])
    model = hp.nest(f"ml.models.classification.{model_type}")

    return {"model_type": model_type, "model": model}
```

### Fallback Strategies

```python
@config
def robust_config(hp: HP):
    model_preference = hp.select(["premium", "standard", "basic"], default="standard")

    # Try multiple sources with fallbacks
    try:
        if model_preference == "premium":
            llm = hp.nest("llm.gpt4")  # Registry first
        else:
            llm = hp.nest("llm.claude")
    except Exception:
        # Fallback to file-based config
        llm = hp.nest("configs/fallback_models.py:basic_llm")

    return {"llm": llm}
```

This enhanced nesting system provides maximum flexibility for configuration management while maintaining clear resolution priorities and excellent debugging capabilities through comprehensive error messages.
