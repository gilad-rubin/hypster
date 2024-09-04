<p align="center">
  <img src="assets/hypster_with_text.png" alt="Hypster Logo" width="600"/>
</p>

</p>
<p align="center">
  <span style="font-size: 18px;">
    <a href="https://gilad-rubin.github.io/hypster/">[Documentation]</a> |
    <a href="#installation">[Installation]</a> |
    <a href="#quick-start">[Quick Start]</a>
  </span>
</p>

**`Hypster`** is a lightweight configuration system for AI & Machine Learning projects.
It offers minimal, intuitive pythonic syntax, supporting hierarchical and swappable configurations.

## Installation

You can install Hypster using pip:

```bash
pip install hypster
```

## Quick Start

Here's a simple example of how to use Hypster:

```python
from hypster import HP, config

@config
def my_config(hp: HP):
    chunking_strategy = hp.select(['paragraph', 'semantic', 'fixed'], default='paragraph')

    llm_model = hp.select({'haiku': 'claude-3-haiku-20240307',
                           'sonnet': 'claude-3-5-sonnet-20240620',
                           'gpt-4o-mini': 'gpt-4o-mini'}, default='gpt-4o-mini')

    llm_config = {'temperature': hp.number_input(0),
                  'max_tokens': hp.number_input(64)}

    system_prompt = hp.text_input('You are a helpful assistant. Answer with one word only')
```

Now we can instantiate the configs with our selections and overrides:

```python
results = my_config(final_vars=["chunking_strategy", "llm_config", "llm_model"],
                    selections={"llm_model" : "haiku"},
                    overrides={"llm_config.temperature" : 0.5})
```

## Inspiration

Hypster draws inspiration from Meta's [hydra](https://github.com/facebookresearch/hydra) and [hydra-zen](https://github.com/mit-ll-responsible-ai/hydra-zen) framework.
The API design is influenced by [Optuna's](https://github.com/optuna/optuna) "define-by-run" API.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
