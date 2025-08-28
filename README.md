<p align="center">
  <img src="https://raw.githubusercontent.com/gilad-rubin/hypster/master/assets/hypster_with_text.png" alt="Hypster Logo" width="600"/>
</p>

<div align="center">
  <div>
    <a href="https://gilad-rubin.gitbook.io/hypster"><strong>Docs</strong></a> ·
    <a href="https://github.com/gilad-rubin/hypster/issues/new?template=bug_report.md"><strong>Report Bug</strong></a> ·
    <a href="https://github.com/gilad-rubin/hypster/issues/new?template=feature_request.md"><strong>Feature Request</strong></a> ·
    <a href="https://github.com/gilad-rubin/hypster/blob/hypster-v2/CHANGELOG.md"><strong>Changelog</strong></a>
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
    Hypster is a lightweight configuration framework for <b>optimizing AI & ML workflows</b>
  </em>
</p>

> ⚠️ Hypster is in active development and not yet battle-tested in production.
> If you’re gaining value and want to promote it to production, please reach out!

## Key Features

- 🐍 **Pythonic API**: Intuitive & minimal syntax that feels natural to Python developers
- 🪆 **Hierarchical, Conditional Configurations**: Support for nested and swappable configurations
- 📐 **Type Safety**: Built-in type hints and validation
- 🧪 **Hyperparameter Optimization Built-In**: Native, first-class optuna support

## Installation

You can install Hypster using uv:

```bash
uv add hypster
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
from hypster import HP, instantiate
from llm import LLM

def llm_config(hp: HP):
    model_name = hp.select(["gpt-4o-mini", "gpt-4o"], name="model_name")
    temperature = hp.float(0.7, name="temperature", min=0.0, max=1.0)
    max_tokens = hp.int(256, name="max_tokens", min=1, max=4096)
    llm = LLM(model_name=model_name, temperature=temperature, max_tokens=max_tokens)
    return llm

llm = instantiate(llm_config, values={"model_name": "gpt-4o-mini", "temperature": 0.3})
llm.invoke("How's your day going?")
```

## HPO with Optuna

```python
import optuna
from hypster.hpo.types import HpoInt, HpoFloat, HpoCategorical
from hypster.hpo.optuna import suggest_values


def objective(trial: optuna.Trial) -> float:
    values = suggest_values(trial, config=model_cfg)
    model = instantiate(model_cfg, values=values)
    X, y = make_classification(
        n_samples=400, n_features=20, n_informative=10, random_state=42
    )
    return cross_val_score(model, X, y, cv=3, n_jobs=-1).mean()

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
