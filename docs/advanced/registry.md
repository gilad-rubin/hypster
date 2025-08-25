# Configuration Registry

The Hypster Configuration Registry provides a global, namespace-organized system for managing and discovering configurations across your application. It enables powerful composition patterns and runtime configuration management.

{% hint style="info" %}
**Note**: All examples in this document require the `name` parameter for HP method calls. For brevity, some examples may omit this parameter, but in practice, you must include `name="parameter_name"` for all `hp.select()`, `hp.number()`, `hp.int()`, etc. calls.
{% endhint %}

## Overview

The registry is a singleton that maintains a hierarchical namespace of configuration functions, allowing you to:

- **Register** configurations with meaningful names
- **Discover** available configurations by namespace
- **Compose** complex configurations from registered components
- **Manage** configuration lifecycle programmatically

## Basic Usage

### Registration

Register configurations using the `register` parameter in the `@config` decorator:

```python
from hypster import config, HP

@config(register="llm.openai")
def openai_config(hp: HP):
    model = hp.select(["gpt-3.5-turbo", "gpt-4"], default="gpt-3.5-turbo")
    temperature = hp.number(0.7, min=0, max=1)
    max_tokens = hp.int(1000, min=100, max=4000)

    return {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens
    }

@config(register="llm.anthropic")
def anthropic_config(hp: HP):
    model = hp.select(["claude-3-haiku", "claude-3-sonnet"], default="claude-3-haiku")
    temperature = hp.number(0.7, min=0, max=1)
    max_tokens = hp.int(1000, min=100, max=4000)

    return {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens
    }
```

### Using Registered Configurations

Access registered configurations through `hp.nest()`:

```python
@config
def rag_config(hp: HP):
    # Use registered configuration
    llm = hp.nest("llm.openai")

    # Add RAG-specific parameters
    retrieval_top_k = hp.int(5, min=1, max=20)
    chunk_size = hp.int(512, min=128, max=1024)

    return {
        "llm": llm,
        "retrieval_top_k": retrieval_top_k,
        "chunk_size": chunk_size
    }
```

## Namespace Organization

### Hierarchical Namespaces

Use dot notation to create organized, hierarchical namespaces:

```python
# Model configurations
@config(register="models.llm.openai")
def openai_llm(hp: HP): ...

@config(register="models.llm.anthropic")
def anthropic_llm(hp: HP): ...

@config(register="models.embedding.openai")
def openai_embedding(hp: HP): ...

@config(register="models.embedding.sentence_transformers")
def sentence_transformers_embedding(hp: HP): ...

# Pipeline configurations
@config(register="pipelines.rag.basic")
def basic_rag(hp: HP): ...

@config(register="pipelines.rag.advanced")
def advanced_rag(hp: HP): ...

# Data processing configurations
@config(register="processing.text.standard")
def standard_text_processing(hp: HP): ...

@config(register="processing.text.advanced")
def advanced_text_processing(hp: HP): ...
```

### Discovery and Listing

Discover available configurations by namespace:

```python
from hypster import registry

# List all configurations
all_configs = registry.list()
print(f"All configs: {all_configs}")

# List by specific namespace
llm_configs = registry.list("models.llm")
print(f"LLM configs: {llm_configs}")  # ['models.llm.openai', 'models.llm.anthropic']

embedding_configs = registry.list("models.embedding")
print(f"Embedding configs: {embedding_configs}")  # ['models.embedding.openai', 'models.embedding.sentence_transformers']

# List all model configurations
model_configs = registry.list("models")
print(f"Model configs: {model_configs}")  # ['models.llm.openai', 'models.llm.anthropic', 'models.embedding.openai', ...]
```

## Runtime Registry Management

### Programmatic Registration

Register configurations at runtime for dynamic scenarios:

```python
from hypster import registry

def create_custom_llm_config(model_name: str, default_temp: float = 0.7):
    @config
    def custom_llm(hp: HP):
        temperature = hp.number(default_temp, min=0, max=1)
        max_tokens = hp.int(1000, min=100, max=4000)

        return {
            "model": model_name,
            "temperature": temperature,
            "max_tokens": max_tokens
        }

    # Register with dynamic name
    registry.register(f"llm.custom.{model_name}", custom_llm)
    return custom_llm

# Create and register custom configurations
gpt4_config = create_custom_llm_config("gpt-4", 0.3)
llama_config = create_custom_llm_config("llama-2-70b", 0.8)

# Now use them in other configurations
@config
def multi_model_config(hp: HP):
    model_choice = hp.select(["gpt-4", "llama-2-70b"], default="gpt-4")
    llm = hp.nest(f"llm.custom.{model_choice}")
    return {"llm": llm}
```

### Registry Queries

Check configuration availability and manage registry state:

```python
# Check if configuration exists
if registry.contains("llm.openai"):
    print("OpenAI LLM config is available")

# Get configuration function directly
if registry.contains("llm.anthropic"):
    anthropic_func = registry.get("llm.anthropic")
    # Use directly if needed
    result = anthropic_func()

# Clear registry (useful for testing)
registry.clear()

# Or clear specific namespace
registry.clear("models.llm")  # Note: This would need to be implemented
```

## Advanced Patterns

### Configuration Factories

Create configuration factories that register multiple related configurations:

```python
def register_llm_provider(provider_name: str, models: list, default_model: str):
    """Register a complete LLM provider configuration set."""

    @config(register=f"llm.{provider_name}")
    def provider_config(hp: HP):
        model = hp.select(models, default=default_model)
        temperature = hp.number(0.7, min=0, max=1)
        max_tokens = hp.int(1000, min=100, max=4000)

        return {
            "provider": provider_name,
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens
        }

    # Register individual model configs
    for model in models:
        @config(register=f"llm.{provider_name}.{model}")
        def model_config(hp: HP, model_name=model):
            temperature = hp.number(0.7, min=0, max=1)
            max_tokens = hp.int(1000, min=100, max=4000)

            return {
                "provider": provider_name,
                "model": model_name,
                "temperature": temperature,
                "max_tokens": max_tokens
            }

# Register multiple providers
register_llm_provider("openai", ["gpt-3.5-turbo", "gpt-4"], "gpt-3.5-turbo")
register_llm_provider("anthropic", ["claude-3-haiku", "claude-3-sonnet"], "claude-3-haiku")
register_llm_provider("cohere", ["command", "command-light"], "command")
```

### Dynamic Configuration Selection

Build configurations that adapt based on available registry entries:

```python
@config
def adaptive_rag_config(hp: HP):
    # Discover available LLM providers
    available_llms = registry.list("llm")

    # Let user choose from available options
    llm_choice = hp.select(available_llms, default=available_llms[0])
    llm = hp.nest(f"{llm_choice}")

    # Similarly for embeddings
    available_embeddings = registry.list("models.embedding")
    embedding_choice = hp.select(available_embeddings, default=available_embeddings[0])
    embeddings = hp.nest(f"{embedding_choice}")

    return {
        "llm": llm,
        "embeddings": embeddings,
        "chunk_size": hp.int(512, min=128, max=1024)
    }
```

### Configuration Inheritance

Create base configurations and extend them:

```python
@config(register="base.llm")
def base_llm_config(hp: HP):
    temperature = hp.number(0.7, min=0, max=1)
    max_tokens = hp.int(1000, min=100, max=4000)

    return {
        "temperature": temperature,
        "max_tokens": max_tokens
    }

@config(register="llm.openai.extended")
def openai_extended_config(hp: HP):
    # Inherit base configuration
    base = hp.nest("base.llm")

    # Add OpenAI-specific parameters
    model = hp.select(["gpt-3.5-turbo", "gpt-4"], default="gpt-3.5-turbo")
    frequency_penalty = hp.number(0.0, min=-2.0, max=2.0)

    return {
        **base,
        "model": model,
        "frequency_penalty": frequency_penalty,
        "provider": "openai"
    }
```

## Best Practices

### Namespace Design

1. **Use Clear Hierarchies**: Organize by functionality, then by provider/type
   ```python
   # Good
   "models.llm.openai"
   "models.embedding.sentence_transformers"
   "pipelines.rag.basic"

   # Avoid
   "openai_llm"
   "rag_pipeline_v2"
   ```

2. **Consistent Naming**: Use lowercase with underscores for multi-word components
   ```python
   # Good
   "models.text_generation.hugging_face"
   "processing.data_cleaning.advanced"

   # Avoid
   "models.TextGeneration.HuggingFace"
   "processing.DataCleaning.Advanced"
   ```

### Configuration Design

1. **Atomic Configurations**: Keep individual configurations focused and reusable
   ```python
   # Good - focused and reusable
   @config(register="models.llm.openai")
   def openai_llm(hp: HP):
       model = hp.select(["gpt-3.5-turbo", "gpt-4"])
       temperature = hp.number(0.7, min=0, max=1)
       return {"model": model, "temperature": temperature}

   # Avoid - too complex and coupled
   @config(register="complete.rag.system")
   def complete_rag(hp: HP):
       # Too many responsibilities in one config
       pass
   ```

2. **Composition Over Inheritance**: Prefer composing configurations rather than deep inheritance
   ```python
   @config(register="pipelines.rag.modular")
   def modular_rag(hp: HP):
       llm = hp.nest("models.llm.openai")
       embeddings = hp.nest("models.embedding.openai")
       retrieval = hp.nest("retrieval.vector.faiss")

       return {
           "llm": llm,
           "embeddings": embeddings,
           "retrieval": retrieval
       }
   ```

### Error Handling

1. **Graceful Fallbacks**: Handle missing configurations gracefully
   ```python
   @config
   def robust_config(hp: HP):
       try:
           llm = hp.nest("llm.premium")
       except Exception:
           # Fallback to standard configuration
           llm = hp.nest("llm.standard")

       return {"llm": llm}
   ```

2. **Validation**: Validate registry state in critical applications
   ```python
   def validate_required_configs():
       required_configs = [
           "models.llm.openai",
           "models.embedding.openai",
           "retrieval.vector.basic"
       ]

       missing = [cfg for cfg in required_configs if not registry.contains(cfg)]
       if missing:
           raise ValueError(f"Missing required configurations: {missing}")

   # Call before critical operations
   validate_required_configs()
   ```

The configuration registry enables powerful, flexible configuration management while maintaining clarity and discoverability. Use it to build modular, reusable configuration systems that can evolve with your application's needs.
