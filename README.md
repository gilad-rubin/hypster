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

> ⚠️ Hypster is in active development and not yet battle-tested in production.
> If you’re gaining value and want to promote it to production, please reach out!

## Key Features

- 🐍 **Pure Python, Not a DSL**: Use normal functions, `if` statements, loops, lists, helper functions, imports, dataclasses, and runtime objects
- 🪆 **Hierarchical, Conditional Configurations**: Support for nested and swappable configurations
- 📐 **Type Safety**: Built-in type hints and validation
- 🧪 **Hyperparameter Optimization Built-In**: Native, first-class optuna support

Hypster configs are ordinary Python functions rather than a separate configuration language. That keeps them flexible and readable, but it also means Hypster discovers the available parameters by executing the function. Keep config functions fast and side-effect-free: avoid paid API calls, network calls, file writes, training, database access, and costly initialization in code paths used by `explore()`, HPO, or interactive UIs.

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
```

## Quick Start

Define a configuration function and instantiate it with overrides:

```python
from dataclasses import dataclass
from hypster import HP, explore, instantiate_with_params

@dataclass
class LLMSettings:
    provider: str
    temperature: float
    max_tokens: int

def llm_config(hp: HP) -> LLMSettings:
    provider = hp.select(["openai", "gemini"], name="provider", default="openai", options_only=True)
    temperature = hp.float(0.7, name="temperature", min=0.0, max=1.0)
    max_tokens = hp.int(256, name="max_tokens", min=1, max=4096)
    return LLMSettings(provider=provider, temperature=temperature, max_tokens=max_tokens)

explore(llm_config)
# llm_config
# ├── provider: select = "openai"  (options: ["openai", "gemini"])
# ├── temperature: float = 0.7  (0.0-1.0)
# └── max_tokens: int = 256  (1-4096)

run = instantiate_with_params(llm_config, values={"provider": "gemini", "temperature": 0.3})

assert run.value == LLMSettings(provider="gemini", temperature=0.3, max_tokens=256)
assert run.params == {"provider": "gemini", "temperature": 0.3, "max_tokens": 256}
```

Use `explore(..., values=...)` to inspect a specific conditional branch before you instantiate it, or `explore(..., return_info=True)` to get a JSON-serializable schema object.

## HPO with Optuna

```python
import optuna
from hypster import HP, instantiate
from hypster.hpo.optuna import suggest_values
from hypster.hpo.types import HpoFloat, HpoInt


def model_cfg(hp: HP):
    family = hp.select(["linear", "forest"], name="family", default="forest", options_only=True)
    if family == "linear":
        return {
            "family": family,
            "alpha": hp.float(0.1, name="alpha", min=1e-4, max=10.0, hpo_spec=HpoFloat(scale="log")),
        }
    return {
        "family": family,
        "n_estimators": hp.int(200, name="n_estimators", min=50, max=1000, hpo_spec=HpoInt(step=50)),
    }


def objective(trial: optuna.Trial) -> float:
    values = suggest_values(trial, config=model_cfg)
    cfg = instantiate(model_cfg, values=values)
    return train_and_score(cfg)

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
