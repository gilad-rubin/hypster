# Reproducibility And Observability

This is the positioning story around `instantiate_with_params`, experiment
tracking, and replay.

## Current Status

`instantiate_with_params` exists on the `codex/reproducible-instantiation-params`
branch, not in the current `master` checkout I inspected.

On that branch:

- `instantiate_with_params(func, values=...)` returns an `InstantiationOutput`.
- `InstantiationOutput.value` is the same runtime value the normal config would
  return.
- `InstantiationOutput.params` is a dictionary of selected parameter values.
- `params` includes reachable defaults, not only user-provided overrides.
- Passing `run.params` back into `instantiate(func, values=run.params)` replays
  the same selected configuration.
- Unknown/unreachable values raise by default on that branch, because `values`
  is treated as a reproducibility surface.

## Why This Matters

Most observability and experiment tools want key-value metadata:

- MLflow params.
- W&B config.
- Langfuse trace/session metadata.
- OpenTelemetry span attributes.
- Custom run records in a database.

Hypster can produce those values from the same config function that builds the
runtime object or workflow.

## User Benefit

Without this:

- A run might log only explicit overrides.
- Defaults can be lost.
- Branch-specific values can be forgotten.
- A logged run may be hard to replay.
- Unknown values may be silently accepted by a wrapper layer.

With this:

- Every reached `hp.*` parameter can be logged.
- The logged params are replayable.
- Defaults become observable.
- Config metadata is attached to metrics, traces, artifacts, and evaluation
  records.
- Strict unknown-value handling protects logged runs from accidental typos or
  stale branch params.

## Suggested Language

README bullet:

- Capture a replayable params record for each run, including defaults, for
  MLflow, W&B, Langfuse, or your own observability stack.

Short section:

Hypster can return a selected-params sidecar when you instantiate a config. Log
that sidecar with your experiment tracker, attach it to traces, or pass it back
into `instantiate()` to replay the same configuration later.

More technical but still positioning:

`instantiate_with_params()` turns configuration into structured run metadata:
the concrete value your application needs plus the selected parameter dictionary
your observability stack needs.

## Example Shape

This is intentionally illustrative. Public docs should update it to the current
API before publishing.

```python
run = instantiate_with_params(rag_config, values={
    "retrieval.top_k_pages": 50,
    "generation.llm.model": "gpt-5.2",
})

graph = run.value
params = run.params

mlflow.log_params(params)
```

## Observability Integrations To Mention

- MLflow: experiment params, metrics, artifacts, traces.
- W&B: run config and sweeps.
- Langfuse: trace/session metadata for LLM workflows.
- OpenTelemetry: span attributes for config-relevant execution.
- Custom data warehouse: store params beside evaluation outcomes.

## Product Framing

Reproducibility is not only "save config to disk." For Hypster, it means:

- the active branch is known,
- every reached parameter has a selected value,
- defaults are captured,
- unknown values are rejected when strictness matters,
- the run can be replayed,
- the configuration can be compared against other runs.
