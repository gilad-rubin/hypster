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

- 🐍 **Pythonic API**: Intuitive & minimal syntax that feels natural to Python developers
- 🪆 **Hierarchical, Conditional Configurations**: Support for nested and swappable configurations
- 📐 **Type Safety**: Built-in type hints and validation
- 🧪 **Hyperparameter Optimization Built-In**: Native, first-class optuna support
- 🧩 **Rules As Config**: Define replayable WHEN/THEN rule lists with built-in notebook controls

## Installation

```bash
uv add hypster
# optional notebook visualization UI
uv add 'hypster[viz]'
# optional HPO backend
uv add 'hypster[optuna]'
```

See the [installation guide](https://gilad-rubin.gitbook.io/hypster/getting-started/installation) for pip, optional extras, and development setup.

## Quick Start

Define a configuration function and instantiate it with overrides:

```python
from hypster import HP, explore, instantiate
from my_app.llms import LLMClient


def llm_config(hp: HP) -> LLMClient:
    model_name = hp.select(["gpt-5.5", "gpt-5.5-mini"], name="model_name")
    temperature = hp.float(0.7, name="temperature", min=0.0, max=1.0)
    max_tokens = hp.int(256, name="max_tokens", min=1, max=4096)

    return LLMClient(
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
    )

explore(llm_config)
# llm_config
# ├── model_name: select = "gpt-5.5"  (options: ["gpt-5.5", "gpt-5.5-mini"])
# ├── temperature: float = 0.7  (0.0-1.0)
# └── max_tokens: int = 256  (1-4096)

llm = instantiate(
    llm_config,
    values={"model_name": "gpt-5.5", "temperature": 0.2},
)

response = llm.invoke("How's your day going?")
```

Use `explore(..., values=...)` to inspect a specific conditional branch before you instantiate it, or `explore(..., return_schema=True)` to get a JSON-serializable schema object.

## AI-readable docs

GitBook publishes an agent-friendly docs index at [llms.txt](https://gilad-rubin.gitbook.io/hypster/llms.txt) and a full Markdown export at [llms-full.txt](https://gilad-rubin.gitbook.io/hypster/llms-full.txt).

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
    values = suggest_values(trial, model_cfg)
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
