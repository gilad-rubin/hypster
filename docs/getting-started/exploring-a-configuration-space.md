# 🔎 Exploring a Configuration Space

Use `explore(func, values=...)` to inspect which parameters a configuration exposes before you instantiate it.

* **See the shape first** - Print a tree of parameters, defaults, bounds, and options.
* **Follow real branches** - Pass `values=` to inspect the branch that would execute for a specific choice.
* **Use the same override format** - `explore()` accepts the same dotted keys you already use with `instantiate()`.
* **Get structured metadata** - Return a `ConfigSchema` object for programmatic tooling.

## Print the parameter tree

```python
from hypster import HP, explore


def openai(hp: HP):
    model = hp.select(["gpt-4o-mini", "gpt-4.1"], name="model", default="gpt-4o-mini")
    temperature = hp.float(0.2, name="temperature", min=0.0, max=2.0)
    return {"model": model, "temperature": temperature}


def gemini(hp: HP):
    model = hp.select(["flash-lite", "pro"], name="model", default="flash-lite")
    thinking_level = hp.select(
        [None, "minimal", "elevated"],
        name="thinking_level",
        default=None,
    )
    return {"model": model, "thinking_level": thinking_level}


def query_llm(hp: HP):
    provider = hp.select(["gemini", "openai"], name="provider", default="gemini")

    if provider == "gemini":
        provider_config = hp.nest(gemini, name="gemini")
    else:
        provider_config = hp.nest(openai, name="openai")

    return {"provider": provider, "config": provider_config}


def query_graph_config(hp: HP):
    output_mode = hp.select(["text", "structured"], name="output_mode", default="text")
    max_tokens = hp.int(100000, name="max_tokens", min=1000)
    query_model = hp.nest(query_llm, name="query_llm")
    system_prompt = hp.text("", name="system_prompt")
    return {
        "output_mode": output_mode,
        "max_tokens": max_tokens,
        "query_llm": query_model,
        "system_prompt": system_prompt,
    }


explore(query_graph_config)
```

Output:

```text
query_graph_config
├── output_mode: select = "text"  (options: ["text", "structured"])
├── max_tokens: int = 100000  (min: 1000)
├── query_llm
│   ├── provider: select = "gemini"  (options: ["gemini", "openai"])
│   └── gemini
│       ├── model: select = "flash-lite"  (options: ["flash-lite", "pro"])
│       └── thinking_level: select = None  (options: [None, "minimal", "elevated"])
└── system_prompt: text = ""
```

Use this when you want to answer questions like:

* Which parameters exist on this branch?
* What is the default value?
* Which options are valid?
* Which knobs are nested under a child config?

## Explore a different conditional branch

Pass the same dotted overrides you would use with `instantiate()`:

```python
explore(
    query_graph_config,
    values={"query_llm.provider": "openai", "query_llm.openai.temperature": 0.7},
)
```

Output:

```text
query_graph_config
├── output_mode: select = "text"  (options: ["text", "structured"])
├── max_tokens: int = 100000  (min: 1000)
├── query_llm
│   ├── provider: select = "openai"  (options: ["gemini", "openai"])
│   └── openai
│       ├── model: select = "gpt-4o-mini"  (options: ["gpt-4o-mini", "gpt-4.1"])
│       └── temperature: float = 0.7  (0.0-2.0)
└── system_prompt: text = ""
```

This is useful for conditional configs where different branches expose different parameters.

## Get structured metadata

Use `return_info=True` when you want to inspect the schema in code:

```python
info = explore(query_graph_config, return_info=True)

info.defaults()
# {
#     "output_mode": "text",
#     "max_tokens": 100000,
#     "query_llm.provider": "gemini",
#     "query_llm.gemini.model": "flash-lite",
#     "query_llm.gemini.thinking_level": None,
#     "system_prompt": "",
# }

info.to_dict()
# JSON-serializable nested structure
```

By default, `explore()` also warns when `values=` contains unknown names or overrides for a branch that was not reached. Pass `on_unknown="raise"` to make that strict, or `on_unknown="ignore"` to silence it.

## When to use `explore()` vs `instantiate()`

Use `explore()` to understand the configuration space.

Call `instantiate()` when you need the actual object, dict, or workflow your config returns.

```python
from hypster import instantiate

cfg = instantiate(
    query_graph_config,
    values={"query_llm.provider": "openai", "query_llm.openai.model": "gpt-4.1"},
)
```

## Notes

* `explore()` only shows the branch that executes for the provided `values`.
* Nested configs appear as groups, and parameter paths use dotted names such as `query_llm.provider`.
* The `values=` argument follows the same override rules documented in [Values & Overrides](../in-depth/values-and-overrides.md).
