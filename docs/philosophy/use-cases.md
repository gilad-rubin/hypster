---
icon: list-check
---

# Use Cases

Hypster is a good fit when configuration is conditional, nested, or part of an experiment record.

## Machine Learning

* Choose model families such as `linear`, `forest`, `boosted`, or `neural`.
* Keep family-specific parameters on their active branch.
* Use `hpo_spec=` to make the same config searchable by Optuna.
* Log `instantiate_with_params().params` with metrics.

See [Machine Learning](../examples/machine-learning.md).

## Data Processing

* Configure ingestion paths, delimiters, schemas, cleaning rules, and export formats.
* Represent environment choices such as `sample` vs `full`.
* Replay a pipeline run from selected params.

See [Data Processing](../examples/data-processing.md).

## AI Workflows

* Switch providers, models, prompts, retrieval strategies, and output modes.
* Use dict-backed selects for complex provider or retriever objects.
* Explore the active branch before rendering a UI or submitting a job.

See [AI Workflows](../examples/ai-workflows.md).

## Internal Tools And UIs

* Generate form controls from `explore(..., return_schema=True)`.
* Submit UI state as `values=`.
* Recompute the schema when a branch-selecting value changes.

See [Interactive UI From Schema](../examples/interactive-ui.md).

## Production Replay

* Store selected params next to a versioned config function.
* Keep `on_unknown="raise"` so stale payloads fail visibly.
* Smoke-test production parameter payloads in CI.
