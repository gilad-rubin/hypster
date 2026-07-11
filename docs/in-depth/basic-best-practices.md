# Best Practices

These practices keep Hypster configs easy to explore, optimize, log, and replay.

## Embrace Pure Python

Hypster configs are ordinary Python functions rather than a DSL. Use `if` statements, loops, local variables, lists, helpers, imports, and typed return values when they make the config clearer.

The implication is that Hypster discovers the available parameters by running your function. Design config functions so they can be run repeatedly by `explore()`, HPO, and interactive UIs without causing side effects or surprising costs.

## Return Typed Runtime Objects

A strong Hypster pattern is to make each config function a typed factory for the object the caller needs:

{% code overflow="wrap" %}
```python
from hypster import HP
from sklearn.ensemble import RandomForestClassifier

def classifier_config(hp: HP) -> RandomForestClassifier:
    n_estimators = hp.int(200, name="n_estimators", min=10, max=1000)
    max_depth = hp.int(None, name="max_depth", allow_none=True)

    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
    )
```
{% endcode %}

Use a return type annotation for config functions whenever the output is a meaningful object. It makes the config easier to read, test, and compose with `hp.nest()`.

## Keep Config Functions Side-Effect-Light

`explore()`, HPO, and UI builders execute your config function to discover parameters. Interactive UIs may rerun it on every value change. Initializing cheap in-memory runtime objects is a good fit for config functions; effects and expensive work should stay outside the config body:

* train the model after `instantiate()`
* make paid API or network calls after `instantiate()`
* write files or database rows after `instantiate()`
* load indexes, large datasets, or heavyweight clients after `instantiate()`
* defer costly resource construction when exploratory safety matters

Use this boundary when deciding what a config should return:

| Return from the config | Usually safe during `explore()`? | Notes |
| --- | --- | --- |
| Enums, paths, mappings your runtime actually consumes, small Python objects | Yes | Good for UI generation, experiment tracking, and replay. |
| In-memory model estimators or pipeline objects | Usually | Good when construction is cheap and does not open files, sockets, or remote handles. |
| SDK clients, database handles, loaded indexes, network retrievers | Usually no | Return lightweight settings or factories, then build these after `instantiate()`. |
| Training jobs, writes, API calls, migrations | No | Run these outside the config function. |

## Name Everything Explicitly

Every `hp.*` call needs a stable `name=`. Names become the keys in `values=`, `explore()` output, and `instantiate_with_params().params`.

{% code overflow="wrap" %}
```python
hp.float(0.001, name="learning_rate")
```
{% endcode %}

Use Python identifier-style names:

* Good: `learning_rate`, `max_depth`, `retriever_kind`
* Avoid: `learning-rate`, `model.lr`, `max depth`

Let `hp.nest()` create dotted paths.

## Use Branches For Real Runtime Decisions

Branch when downstream structure changes:

{% code overflow="wrap" %}
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from hypster import HP

def model_config(hp: HP):
    family = hp.select(["linear", "forest"], name="family", default="forest", options_only=True)

    if family == "linear":
        C = hp.float(1.0, name="C", min=1e-4, max=100.0)
        return LogisticRegression(C=C, max_iter=1000)

    n_estimators = hp.int(200, name="n_estimators", min=10, max=1000)
    max_depth = hp.int(None, name="max_depth", min=1, max=100, allow_none=True)
    return RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
```
{% endcode %}

Avoid carrying irrelevant parameters for inactive branches. Branch-aware configs make experiment logs cleaner and HPO search spaces smaller.

Use inline branches for small, local differences. When the branch chooses between reusable components, prefer composition with `hp.nest()` and a dict-backed `select` over a long `if`/`elif` chain. The key stays replayable, each component keeps its parameters local, and interactive UIs can render the selected child config as a contained group.

## Prefer Dict-Backed Selects For Swappable Components

Select keys should be simple and replayable. For swappable runtime components, map those keys to config functions and then nest the selected function:

{% code overflow="wrap" %}
```python
from my_app.tokenizers import SimpleTokenizer, Tokenizer, WordPieceTokenizer

def simple_tokenizer_config(hp: HP) -> SimpleTokenizer:
    lowercase = hp.bool(True, name="lowercase")
    return SimpleTokenizer(lowercase=lowercase)

def wordpiece_tokenizer_config(hp: HP) -> WordPieceTokenizer:
    vocab_path = hp.text("vocab.txt", name="vocab_path")
    return WordPieceTokenizer(vocab_path=vocab_path)

tokenizer_options = {
    "simple": simple_tokenizer_config,
    "wordpiece": wordpiece_tokenizer_config,
}

def tokenizer_config(hp: HP) -> Tokenizer:
    selected_config = hp.select(tokenizer_options, name="tokenizer", default="wordpiece", options_only=True)
    return hp.nest(selected_config, name="settings")
```
{% endcode %}

This keeps `params={"tokenizer": "wordpiece", "settings.vocab_path": "vocab.txt"}` while your app receives the selected tokenizer object.

Keep the options mapping in a named variable such as `tokenizer_options`, `model_options`, or `retriever_options`. That keeps the parent config readable, especially when the mapping is long, and makes it easy to reuse the same option set in HPO, interactive UIs, tests, and nested configs.

For a tiny branch with one or two scalar differences, an `if` statement is fine. Once each branch has its own parameters or returns a different runtime type, split the branches into child config functions and choose between them with a dict-backed select.

## Turn On `options_only=True` For Enums

By default, `select` allows custom scalar values outside the listed options. Use `options_only=True` when the option list is closed:

{% code overflow="wrap" %}
```python
provider = hp.select(["openai", "gemini"], name="provider", default="openai", options_only=True)
```
{% endcode %}

## Use `allow_none=True` Deliberately

`None` is a real value, not an unspecified value. Mark it explicitly:

{% code overflow="wrap" %}
```python
max_depth = hp.int(None, name="max_depth", min=1, max=100, allow_none=True)
```
{% endcode %}

For nullable choices, you can put `None` directly in the options:

{% code overflow="wrap" %}
```python
tokenizer = hp.select([None, "basic"], name="tokenizer", default=None, allow_none=True)
```
{% endcode %}

## Use Numeric Coercion Deliberately

Hypster safely coerces common numeric inputs by default. Integral floats can be used for integer parameters, and integers can be used for float parameters:

{% code overflow="wrap" %}
```python
def config(hp: HP):
    return {
        "epochs": hp.int(10, name="epochs"),
        "lr": hp.float(0.1, name="lr"),
    }

instantiate(config, values={"epochs": 20.0, "lr": 1})
# => {"epochs": 20, "lr": 1.0}
```
{% endcode %}

Use `strict=True` when the input type itself matters:

{% code overflow="wrap" %}
```python
def strict_config(hp: HP):
    return {
        "epochs": hp.int(10, name="epochs", strict=True),
        "lr": hp.float(0.1, name="lr", strict=True),
    }
```
{% endcode %}

`True` and `False` are rejected by numeric parameters. Use `hp.bool()` for boolean choices.

## Capture Params For Anything You May Replay

Use `instantiate_with_params()` for experiments, UI submissions, scheduled jobs, and production runs:

{% code overflow="wrap" %}
```python
run = instantiate_with_params(config, values={"learning_rate": 0.01})
# tracker.log_params(run.params)
```
{% endcode %}

The params include defaults as well as explicit overrides, so later replay does not depend on changing defaults.

When after-the-fact params are not enough, pass `tracker=` to observe every parameter event live during the run — see [Public API](../reference/api.md).

## Explore Before Instantiating Conditional Values

When overriding a branch, inspect it first:

{% code overflow="wrap" %}
```python
explore(config, values={"provider": "gemini"})
```
{% endcode %}

This prevents stale values from inactive branches from leaking into logs.

## Keep Return Values Narrow

Return what the caller needs. A small return surface makes configs easier to test and less likely to couple unrelated workflow stages.

{% code overflow="wrap" %}
```python
def training_config(hp: HP) -> TrainingRunner:
    model = hp.nest(model_config, name="model")
    optimizer = hp.nest(optimizer_config, name="optimizer")
    return TrainingRunner(model=model, optimizer=optimizer)
```
{% endcode %}

Use `hp.collect(locals(), include=[...])` when the caller genuinely wants a mapping and that makes the return explicit and concise.
