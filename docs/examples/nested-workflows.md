# Nested Workflows

Use `hp.nest()` to split a large workflow into smaller config functions. Nested config values receive dotted parameter paths such as `trainer.optimizer.settings.learning_rate`.

## A Deeply Nested Training Workflow

{% code overflow="wrap" %}
```python
from hypster import HP, explore, instantiate
from my_app.training import Adam, Optimizer, SGD, Trainer, TrainingExperiment

def adam_config(hp: HP) -> Adam:
    learning_rate = hp.float(0.001, name="learning_rate", min=1e-6, max=1.0)
    beta1 = hp.float(0.9, name="beta1", min=0.0, max=0.999)
    return Adam(learning_rate=learning_rate, beta1=beta1)

def sgd_config(hp: HP) -> SGD:
    learning_rate = hp.float(0.01, name="learning_rate", min=1e-6, max=1.0)
    momentum = hp.float(0.9, name="momentum", min=0.0, max=1.0)
    return SGD(learning_rate=learning_rate, momentum=momentum)

optimizer_options = {
    "adam": adam_config,
    "sgd": sgd_config,
}

def optimizer_config(hp: HP) -> Optimizer:
    selected_config = hp.select(optimizer_options, name="algorithm", default="adam", options_only=True)
    return hp.nest(selected_config, name="settings")

def trainer_config(hp: HP, epochs_default: int = 10) -> Trainer:
    epochs = hp.int(epochs_default, name="epochs", min=1, max=1000)
    batch_size = hp.int(64, name="batch_size", min=1, max=2048)
    optimizer = hp.nest(optimizer_config, name="optimizer")
    return Trainer(epochs=epochs, batch_size=batch_size, optimizer=optimizer)

def experiment_config(hp: HP) -> TrainingExperiment:
    dataset = hp.select(["mnist", "cifar10"], name="dataset", default="mnist", options_only=True)
    trainer = hp.nest(trainer_config, name="trainer", kwargs={"epochs_default": 20})
    return TrainingExperiment(dataset=dataset, trainer=trainer)
```
{% endcode %}

The `optimizer_options` dict is deliberately separate from `optimizer_config()`. For nested workflows, that keeps the parent function small while making the set of selectable components obvious.

## Override Nested Values

{% code overflow="wrap" %}
```python
cfg = instantiate(
    experiment_config,
    values={
        "dataset": "cifar10",
        "trainer.epochs": 50,
        "trainer.optimizer.algorithm": "sgd",
        "trainer.optimizer.settings.momentum": 0.95,
    },
)

assert isinstance(cfg.trainer.optimizer, SGD)
assert cfg.trainer.optimizer.momentum == 0.95
```
{% endcode %}

The same values can be expressed as nested dictionaries:

{% code overflow="wrap" %}
```python
cfg = instantiate(
    experiment_config,
    values={
        "dataset": "cifar10",
        "trainer": {
            "epochs": 50,
            "optimizer": {
                "algorithm": "sgd",
                "settings": {"momentum": 0.95},
            },
        },
    },
)
```
{% endcode %}

Do not provide both spellings for the same final parameter path. Hypster raises duplicate-path errors so logs and replays stay unambiguous.

## Explore A Nested Branch

{% code overflow="wrap" %}
```python
explore(
    experiment_config,
    values={"trainer.optimizer.algorithm": "sgd"},
)
```
{% endcode %}

The printed tree includes only the active optimizer branch, so you can see that `momentum` is reachable and `beta1` is not.
