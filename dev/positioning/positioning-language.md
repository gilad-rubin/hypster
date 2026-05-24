# Positioning Language

This file is a scratchpad for public copy.

## Favorite Core Sentence

Hypster is a hierarchical, modular configuration management framework for
Python, built for complex AI and ML workflows.

## Slightly Warmer Version

Hypster helps you manage complex AI and ML configuration with plain Python
functions, nested reusable configs, branch-aware exploration, and
optimization-ready parameters.

## README Opening Candidate

Hypster is a hierarchical, modular configuration management framework for
Python, built for complex AI and ML workflows. Define the valid ways your system
can run using plain Python functions, compose reusable configs for nested
components, explore the active configuration tree, and instantiate or optimize
one concrete setup when you need it.

## First Benefit Bullets

- Define the valid ways your workflow can run using simple Python functions.
- Compose nested configs for models, prompts, retrievers, environments, and app
  modes.
- Explore active branches, defaults, bounds, and options before you instantiate
  anything.
- Reuse the same config for manual runs, notebooks, experiments, and HPO.
- Capture selected params for replay, observability, and experiment tracking.
- Keep complex AI applications configurable without duplicating logic across
  files and scripts.

## Tagline Options

- Hierarchical configuration management for complex AI workflows.
- Modular configuration management in plain Python.
- Define, explore, compose, and optimize AI workflow configurations.
- Build configurable AI systems from reusable Python configs.
- Turn one hard-coded AI pipeline into a family of valid workflows.
- Configuration management for AI systems that keep changing.
- Pythonic configuration for nested, conditional AI workflows.

## Plain Explanations For "Configuration Space"

Use "configuration space" after one of these:

- the valid ways your system can run,
- the set of supported workflow variants,
- the family of possible pipelines,
- the choices, defaults, bounds, branches, and nested components your workflow
  supports.

Good sentence:

Hypster lets you define the valid ways your system can run. That definition is
your configuration space: the choices, defaults, bounds, branches, and nested
components that can produce a concrete workflow.

## Words That Feel Right

- hierarchical,
- modular,
- nested,
- composable,
- branch-aware,
- explore,
- instantiate,
- selected params,
- replayable,
- valid configurations,
- concrete workflow,
- Python functions,
- normal Python control flow,
- configurable systems,
- workflow variants,
- reusable components,
- decision space.

## Words To Use Carefully

- configuration space: good, but explain it.
- framework: okay, but pair it with concrete benefits.
- HPO: useful but too narrow as the headline.
- hyper-workflow: interesting for philosophy pages, probably too abstract for
  first README line.
- superposition: fun in articles, probably not README-first.
- no-code: possible platform angle, not the core dev README message.

## Phrases To Avoid

- "Not a DSL" or other negative framing.
- "Magic" without showing the mechanism.
- "Simple" without acknowledging real AI systems are complex.
- "Just YAML but better."
- "Best configuration framework" style claims.

## Positive Differentiation

Instead of saying:

- "Unlike YAML..."

Say:

- "Use Python control flow for branch-specific configuration."
- "Build real objects and workflows directly from selected values."
- "Compose child configs under parent configs using dotted paths."

Instead of saying:

- "Unlike HPO tools..."

Say:

- "Use the same config for manual runs and automated optimization."
- "Tune workflow structure and parameter values together."

Instead of saying:

- "Unlike experiment trackers..."

Say:

- "Emit replayable selected params for your experiment tracker."
- "Attach complete config metadata to metrics, traces, and artifacts."

## Short README Section Sketch

```md
## Why Hypster?

Real AI systems rarely have one stable configuration. They have model choices,
provider choices, retrieval modes, prompt variants, evaluation settings,
environment modes, and product-specific branches.

Hypster lets those choices live in Python as a reusable, inspectable
configuration tree. Explore the tree, instantiate one concrete setup, log the
selected params, or hand the same config to an HPO backend.
```
