# Usage Example - LLMs

This tutorial demonstrates how to use Hypster with LiteLLM for managing different LLM configurations. We'll create a simple example showing how to switch between models and adjust generation parameters.

## Configurable LLM

```python
import os
import llm
from hypster import HP, config


@config
def llm_config(hp: HP):
    models = {"gpt-4o-mini": "gpt-4o-mini", "gpt-4o": "gpt-4o"}
    model_name = hp.select(models, default="gpt-4o-mini")
    temperature = hp.number_input(0.0, min=0.0, max=1.0)
    max_tokens = hp.int_input(256, max=2048)


def generate(prompt: str, 
             model_name: str, 
             temperature: float, 
             max_tokens: int) -> str:
    model = llm.get_model(model_name)
    return model.prompt(prompt, temperature=temperature, max_tokens=max_tokens)


os.environ["OPENAI_API_KEY"] = "..."

# Create configurations for different use cases
final_vars = ["model_name", "temperature", "max_tokens"]
default_config = llm_config(final_vars=final_vars)
creative_config = llm_config(values={"model_name": "gpt-4o", 
                                     "temperature": 1.0}, 
                             final_vars=final_vars)

# Example prompts
prompt1 = "Explain what machine learning is in 5 words."
prompt2 = "Write a haiku about AI in 17 syllables."
# Generate responses with different configurations
print("Default Configuration (Balanced):")
print(generate(prompt1, **default_config))
print("Creative Configuration (Higher Temperature):")
print(generate(prompt2, **creative_config))
```

This example demonstrates:

1. Simple model configuration with Hypster
2. Easy model switching using `llm`
3. Adjustable generation parameters (temperature, max\_tokens)
4. Different configurations for different use cases
