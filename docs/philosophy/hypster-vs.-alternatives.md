# Hypster vs. Alternatives

Hypster is not trying to replace every configuration tool. It is focused on Python workflows where configuration is executable, conditional, nested, and part of a reproducibility record.

## When Hypster Fits

Use Hypster when:

* parameter choices change which code path runs
* a config needs reusable child configs
* you want to explore or render the active parameter schema
* you need replayable selected params, including defaults
* the same config should support manual runs, UI runs, and HPO

## Compared With Static Config Files

YAML, TOML, and JSON are excellent for static settings. Hypster is better when the configuration space itself contains logic:

{% code overflow="wrap" %}
```python
if provider == "openai":
    llm = hp.nest(openai_config, name="openai")
else:
    llm = hp.nest(gemini_config, name="gemini")
```
{% endcode %}

The active branch determines which parameters exist for a run.

## Compared With Hydra

Hydra is powerful for hierarchical file-based composition and command-line overrides. Hypster is smaller and plain-Python-first:

* no config file format required
* no decorator required
* no global registry required
* config functions return normal Python values
* branch exploration and HPO use the same function

Hydra is likely a better fit if your project already depends on file-based config groups and large CLI-driven sweeps.

## Compared With Pydantic Settings

Pydantic is excellent for validating known fields. Hypster is focused on discovering which fields are active at runtime and replaying the selected parameter paths. You can still return Pydantic models from a Hypster config if that is useful in your application.

## Compared With Optuna Directly

Optuna's define-by-run API is excellent for optimization. Hypster lets you use a similar shape for normal configuration, schema exploration, experiment tracking, and UI generation, then hand the same config to Optuna when you want HPO.
