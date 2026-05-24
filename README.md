<p align="center">
  <img src="https://raw.githubusercontent.com/gilad-rubin/hypster/master/assets/hypster_with_text.png" alt="Hypster Logo" width="600"/>
</p>

<div align="center">
  <div>
    <a href="https://gilad-rubin.gitbook.io/hypster"><strong>Docs</strong></a> ·
    <a href="https://github.com/gilad-rubin/hypster/issues/new?template=bug_report.md"><strong>Report Bug</strong></a> ·
    <a href="https://github.com/gilad-rubin/hypster/issues/new?template=feature_request.md"><strong>Feature Request</strong></a> ·
    <a href="https://github.com/gilad-rubin/hypster/blob/master/CHANGELOG.md"><strong>Changelog</strong></a>
  </div>
</div>

</p>

<p align="center">
  <a href="https://deepwiki.com/gilad-rubin/hypster" style="text-decoration:none;display:inline-block">
    <img src="https://img.shields.io/badge/chat%20with%20our%20AI%20docs-%E2%86%92-72A1FF?style=for-the-badge&logo=readthedocs&logoColor=white"
         alt="chat with our AI docs" width="220">
  </a>

</p>
<p align="center">
  <a href="https://github.com/gilad-rubin/hypster/actions/workflows/ci.yml"><img src="https://github.com/gilad-rubin/hypster/actions/workflows/ci.yml/badge.svg" alt="CI"/></a>
  <a href="https://codecov.io/gh/gilad-rubin/hypster"><img src="https://codecov.io/gh/gilad-rubin/hypster/graph/badge.svg" alt="codecov"/></a>
  <a href="https://pypi.org/project/hypster/"><img src="https://img.shields.io/pypi/v/hypster.svg" alt="PyPI version"/></a>
  <a href="https://pypi.org/project/hypster/"><img src="https://img.shields.io/pypi/pyversions/hypster.svg" alt="Python versions"/></a>
  <a href="https://deepwiki.com/gilad-rubin/hypster"><img src="https://deepwiki.com/badge.svg" alt="DeepWiki"/></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT"/></a>
  <a href="https://codspeed.io/gilad-rubin/hypster"><img src="https://img.shields.io/endpoint?url=https://codspeed.io/badge.json" alt="CodSpeed"/></a>
</p>

<p align="center">
  <em>
    Hypster is a lightweight configuration framework for managing and <b>optimizing AI & ML workflows</b>
  </em>
</p>

## Key Features

- 🐍 **Pure Python, Not a DSL**: Use normal functions, `if` statements, loops, lists, helper functions, imports, and runtime objects
- 🪆 **Hierarchical, Conditional Configurations**: Support for nested and swappable configurations
- 📐 **Type Safety**: Built-in type hints and validation
- 🧪 **Hyperparameter Optimization Built-In**: Native, first-class optuna support

Hypster configs are ordinary Python functions rather than a separate configuration language. That keeps them flexible and readable, but it also means Hypster discovers the available parameters by executing the function. A config usually returns the initialized object your application needs. Keep config functions fast and side-effect-free: avoid paid API calls, network calls, file writes, training, database access, and costly initialization in code paths used by `explore()`, HPO, or interactive UIs.

## Installation

You can install Hypster using uv:

```bash
uv add hypster
# optional notebook visualization UI
uv add 'hypster[viz]'
# optional HPO backend
uv add 'hypster[optuna]'
```

Or using pip:

```bash
pip install hypster
# optional notebook visualization UI
pip install 'hypster[viz]'
# optional HPO backend
pip install 'hypster[optuna]'
```

## Quick Start

Define a configuration function and instantiate it with overrides:

```python
from hypster import HP, explore, instantiate_with_params
from my_app.llms import GeminiClient, OpenAIClient


def openai_config(hp: HP) -> OpenAIClient:
    model = hp.select(["gpt-5", "gpt-5-mini"], name="model", default="gpt-5-mini", options_only=True)
    temperature = hp.float(0.7, name="temperature", min=0.0, max=1.0)
    max_tokens = hp.int(256, name="max_tokens", min=1, max=4096)
    return OpenAIClient(model=model, temperature=temperature, max_tokens=max_tokens)


def gemini_config(hp: HP) -> GeminiClient:
    model = hp.select(
        ["gemini-3.5-flash", "gemini-3.1-pro"],
        name="model",
        default="gemini-3.5-flash",
        options_only=True,
    )
    temperature = hp.float(0.3, name="temperature", min=0.0, max=1.0)
    return GeminiClient(model=model, temperature=temperature)


llm_options = {
    "openai": openai_config,
    "gemini": gemini_config,
}

# Use a named options dict for swappable components. The params log the
# simple key, while the config receives the selected child config function.

def llm_config(hp: HP):
    selected_config = hp.select(llm_options, name="provider", default="openai", options_only=True)
    return hp.nest(selected_config, name="llm")

explore(llm_config)
# llm_config
# ├── provider: select = "openai"  (options: ["openai", "gemini"])
# └── llm
#     ├── model: select = "gpt-5-mini"  (options: ["gpt-5", "gpt-5-mini"])
#     ├── temperature: float = 0.7  (0.0-1.0)
#     └── max_tokens: int = 256  (1-4096)

run = instantiate_with_params(
    llm_config,
    values={"provider": "gemini", "llm.temperature": 0.1},
)

response = run.value.invoke("How's your day going?")
assert run.params["provider"] == "gemini"
```

Use `explore(..., values=...)` to inspect a specific conditional branch before you instantiate it, or `explore(..., return_info=True)` to get a JSON-serializable schema object.

## HPO with Optuna

```python
import optuna
from hypster import HP, instantiate
from hypster.hpo.optuna import suggest_values
from hypster.hpo.types import HpoFloat, HpoInt
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def logistic_config(hp: HP) -> LogisticRegression:
    C = hp.float(1.0, name="C", min=1e-4, max=10.0, hpo_spec=HpoFloat(scale="log"))
    return LogisticRegression(C=C, max_iter=1000)


def forest_config(hp: HP) -> RandomForestClassifier:
    n_estimators = hp.int(
        200,
        name="n_estimators",
        min=50,
        max=1000,
        hpo_spec=HpoInt(step=50),
    )
    return RandomForestClassifier(n_estimators=n_estimators, random_state=42)


model_options = {"logistic": logistic_config, "forest": forest_config}


def model_cfg(hp: HP) -> ClassifierMixin:
    model_config = hp.select(model_options, name="model_family", default="forest", options_only=True)
    return hp.nest(model_config, name="model")


def objective(trial: optuna.Trial) -> float:
    values = suggest_values(trial, config=model_cfg)
    model = instantiate(model_cfg, values=values)
    return train_and_score(model)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)
```

## Inspiration

Hypster draws inspiration from Meta's [hydra](https://github.com/facebookresearch/hydra) and [hydra-zen](https://github.com/mit-ll-responsible-ai/hydra-zen) framework.
The API design is influenced by [Optuna's](https://github.com/optuna/optuna) "define-by-run" API.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
