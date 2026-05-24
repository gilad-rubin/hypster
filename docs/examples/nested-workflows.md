# Nested Workflows

Use `hp.nest()` to split a large workflow into smaller config functions. Nested config values receive dotted parameter paths such as `trainer.optimizer.learning_rate`.

## A Deeply Nested Training Workflow

```python
from hypster import HP, explore, instantiate

def optimizer_config(hp: HP):
    algorithm = hp.select(["adam", "sgd"], name="algorithm", default="adam", options_only=True)

    if algorithm == "adam":
        return {
            "algorithm": algorithm,
            "learning_rate": hp.float(0.001, name="learning_rate", min=1e-6, max=1.0),
            "beta1": hp.float(0.9, name="beta1", min=0.0, max=0.999),
        }

    return {
        "algorithm": algorithm,
        "learning_rate": hp.float(0.01, name="learning_rate", min=1e-6, max=1.0),
        "momentum": hp.float(0.9, name="momentum", min=0.0, max=1.0),
    }

def trainer_config(hp: HP, epochs_default: int = 10):
    return {
        "epochs": hp.int(epochs_default, name="epochs", min=1, max=1000),
        "batch_size": hp.int(64, name="batch_size", min=1, max=2048),
        "optimizer": hp.nest(optimizer_config, name="optimizer"),
    }

def experiment_config(hp: HP):
    dataset = hp.select(["mnist", "cifar10"], name="dataset", default="mnist", options_only=True)
    trainer = hp.nest(trainer_config, name="trainer", kwargs={"epochs_default": 20})
    return {"dataset": dataset, "trainer": trainer}
```

## Override Nested Values

```python
cfg = instantiate(
    experiment_config,
    values={
        "dataset": "cifar10",
        "trainer.epochs": 50,
        "trainer.optimizer.algorithm": "sgd",
        "trainer.optimizer.momentum": 0.95,
    },
)

assert cfg["trainer"]["optimizer"]["algorithm"] == "sgd"
assert cfg["trainer"]["optimizer"]["momentum"] == 0.95
```

The same values can be expressed as nested dictionaries:

```python
cfg = instantiate(
    experiment_config,
    values={
        "dataset": "cifar10",
        "trainer": {
            "epochs": 50,
            "optimizer": {
                "algorithm": "sgd",
                "momentum": 0.95,
            },
        },
    },
)
```

Do not provide both spellings for the same final parameter path. Hypster raises duplicate-path errors so logs and replays stay unambiguous.

## Explore A Nested Branch

```python
explore(
    experiment_config,
    values={"trainer.optimizer.algorithm": "sgd"},
)
```

The printed tree includes only the active optimizer branch, so you can see that `momentum` is reachable and `beta1` is not.
