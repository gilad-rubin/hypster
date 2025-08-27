<p align="center">
  <picture>
    <source srcset="assets/hypster_white_text.png" media="(prefers-color-scheme: dark)">
    <img src="assets/hypster_with_text.png" alt="Hypster Logo" width="600"/>
  </picture>
</p>


<div align="center">
  <div>
    <a href="https://gilad-rubin.gitbook.io/hypster"><strong>Docs</strong></a> ·
    <a href="https://github.com/gilad-rubin/hypster/issues/new?template=bug_report.md"><strong>Report Bug</strong></a> ·
    <a href="https://github.com/gilad-rubin/hypster/issues/new?template=feature_request.md"><strong>Feature Request</strong></a> ·
    <a href="https://github.com/gilad-rubin/hypster/blob/hypster-v2/CHANGELOG.md"><strong>Changelog</strong></a> ·
    <a href="https://github.com/gilad-rubin/hypster/issues?q=is%3Aopen+label%3Aroadmap"><strong>Roadmap</strong></a>
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
  <a href="https://github.com/gilad-rubin/hypster/actions/workflows/ci.yml?query=branch%3Ahypster-v2"><img src="https://github.com/gilad-rubin/hypster/actions/workflows/ci.yml/badge.svg?branch=hypster-v2" alt="CI"/></a>
  <a href="https://codecov.io/gh/gilad-rubin/hypster"><img src="https://codecov.io/gh/gilad-rubin/hypster/branch/hypster-v2/graph/badge.svg" alt="codecov"/></a>
  <a href="https://pypi.org/project/hypster/"><img src="https://img.shields.io/pypi/v/hypster.svg" alt="PyPI version"/></a>
  <a href="https://pypi.org/project/hypster/"><img src="https://img.shields.io/pypi/pyversions/hypster.svg" alt="Python versions"/></a>
  <a href="https://gilad-rubin.gitbook.io/hypster"><img src="https://img.shields.io/badge/docs-gitbook-blue" alt="Docs"/></a>
  <a href="https://deepwiki.com/gilad-rubin/hypster"><img src="https://deepwiki.com/badge.svg" alt="DeepWiki"/></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT"/></a>
  <a href="https://codspeed.io/gilad-rubin/hypster"><img src="https://img.shields.io/endpoint?url=https://codspeed.io/badge.json" alt="CodSpeed"/></a>
</p>

<p align="center">
  <em>
    Hypster is a Pythonic framework for defining <b>conditional & hierarchical configuration spaces</b> to build and
    optimize <b>AI/ML workflows</b>. It enables <b>swappable components</b>, <b>safe overrides</b>, and is
    <b>HPO‑ready</b> out of the box.
  </em>
</p>

> [!WARNING]
>
> Hypster is in active development and not yet battle-tested in production.
>
> If you’re gaining value and want to promote it to production, please reach out!

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


def model_cfg(hp: HP):
    kind = hp.select(["rf", "lr"], name="kind")
    if kind == "rf":
        n = hp.int(100, name="n_estimators", min=50, max=300)
        d = hp.float(10.0, name="max_depth", min=2.0, max=30.0)
        return {"model": ("rf", n, d)}
    C = hp.float(1.0, name="C", min=1e-5, max=10.0)
    solver = hp.select(["lbfgs", "saga"], name="solver")
    return {"model": ("lr", C, solver)}

cfg = instantiate(model_cfg, values={"kind": "rf", "n_estimators": 200, "max_depth": 12.5})
```

### HPO with Optuna (optional)

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
