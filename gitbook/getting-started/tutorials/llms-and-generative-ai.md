# LLMs & Generative AI Tutorial

This tutorial demonstrates how to use Hypster with LiteLLM for managing different LLM configurations. We'll create a simple example showing how to switch between models and adjust generation parameters.

## Complete Example: Configurable LLM

```python
from hypster import config, HP
from litellm import completion
from typing import Dict
import os

@config
def llm_config(hp: HP):
    anthropic_models = {"haiku": "claude-3-5-haiku-latest", "sonnet": "claude-3-5-sonnet-latest"}
    openai_models = {"gpt-4o-mini": "gpt-4o-mini", "gpt-4o": "gpt-4o", "gpt-4o-latest": "gpt-4o-2024-08-06"}
    model_options = {**anthropic_models, **openai_models}

    model = hp.select(model_options, default="gpt-4o-mini")
    temperature = hp.number_input(0.0, min=0.0, max=1.0)
    max_tokens = hp.int(256)

def generate(prompt: str, llm_config: Dict) -> str:
    """Generate text using the configured LLM."""
    messages = [{"role": "user", "content": prompt}]
    response = completion(messages=messages, **llm_config)
    return response.choices[0].message.content

# Example usage
# Set your API keys
os.environ["OPENAI_API_KEY"] = "your-openai-key"
os.environ["ANTHROPIC_API_KEY"] = "your-anthropic-key"

# Create configurations for different use cases
final_vars=["model", "temperature", "max_tokens"]
default_config = llm_config(final_vars=final_vars)
creative_config = llm_config(values={"temperature": 1.0, "max_tokens": 1024}, final_vars=final_vars)

# Example prompts
prompt1 = "Explain what machine learning is."
prompt2 = "Write a creative story about a time-traveling scientist."

# Generate responses with different configurations
print("Default Configuration (Balanced):")
print(generate(prompt1, default_config))
print("\nCreative Configuration (Higher Temperature):")
print(generate(prompt2, creative_config))
```

This example demonstrates:
1. Simple model configuration with Hypster
2. Easy model switching using LiteLLM
3. Adjustable generation parameters (temperature, max_tokens)
4. Different configurations for different use cases
