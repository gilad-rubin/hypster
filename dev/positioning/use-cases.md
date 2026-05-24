# Use Cases

This page collects use cases and scenarios. It is intentionally broad and raw.

## AI And ML Workflow Modes

- Development vs production.
- Local vs remote.
- Fast/cheap vs accurate/expensive.
- Test data vs real data.
- Mock components vs live services.
- Offline evaluation vs online serving.
- Batch evaluation vs single request execution.
- Different country, customer, tenant, population, or domain variants.

Hypster angle:

Use one config function to expose modes and branch-specific values, then
instantiate the concrete mode needed for a run.

## RAG Systems

Good fit because RAG has many naturally configurable parts:

- Ingestion/conversion provider.
- Chunking strategy and chunk size.
- Embedding provider/model.
- Vector database or document store.
- Retriever type: dense, sparse, hybrid, semantic.
- Join strategy and weighting.
- Reranker on/off and reranker provider.
- Prompt template and prompt building mode.
- Generation model, temperature, reasoning effort, max tokens.
- Diagram/image handling.
- Citation mode.

Hypster angle:

Represent the RAG system as a family of valid pipelines rather than one
hard-coded pipeline. Use nested configs for indexing, retrieval, generation,
validation, and evaluation.

## Modular RAG

From the Modular RAG article:

- Build indexing, retrieval, and generation as separate configurable modules.
- Use nesting to compose them into a parent RAG config.
- Allow a parent config to pass dependent values into child configs, such as an
  embedding dimension or diagram mode.
- Use config values to change pipeline topology, not only scalar parameters.

Hypster angle:

RAG is not one pipeline. It is a design space of valid pipelines, and Hypster
lets that design space live in Python.

## Graph And Pipeline Execution

Use cases:

- Build a workflow graph from selected config values.
- Bind configured dependencies into nodes.
- Nest sub-workflow configs under parent workflow configs.
- Select optional nodes or branches based on config values.
- Reuse one graph definition with many valid component combinations.
- Keep graph execution separate from configuration selection.

Hypster angle:

Hypster is the configuration layer, not the execution layer. It chooses the
concrete graph, components, and bound dependencies; a graph, DAG, pipeline, or
agent runtime executes the selected workflow.

## Hyperparameter Optimization

Use cases:

- Model selection.
- Retrieval top-k values.
- Chunk size and overlap.
- Reranking provider and model.
- Prompt template variants.
- LLM temperature, max tokens, and reasoning effort.
- Evaluation judge settings.
- End-to-end pipeline choices where structure and numeric values interact.

Hypster angle:

Use one config function for manual instantiation and automatic search. The
config's branch logic determines which parameters exist for a trial.

## Experiment Tracking And Observability

Use cases:

- Log selected params to MLflow/W&B/Langfuse.
- Replay a run using recorded params.
- Store defaults as well as explicit overrides.
- Link evaluation metrics and traces to the configuration that produced them.
- Compare branches across evaluation runs.

Hypster angle:

`instantiate_with_params` can produce a replayable params sidecar for the
observability layer, while returning the same runtime value the app needs.

## Production A/B Testing

Use cases:

- Route a percentage of traffic to different model providers.
- Run different retrieval strategies for different query types.
- Assign high-recall/high-cost configs to enterprise users and cheaper configs
  to low-risk paths.
- Test prompt variants per request.
- Switch between multimodal and text-only generation.

Hypster angle:

An API request can specify a config, or a router can choose one. The same
execution code receives a concrete workflow.

## Evaluation Pipelines

Use cases:

- Retrieval recall evaluation.
- LLM-as-judge generation scoring.
- Batch evaluation over query sets.
- Holdout/validation split selection.
- Optional MLflow logging.
- Optional artifact construction and image extraction.
- Prompt improvement loops.

Hypster angle:

Evaluation is itself a configurable workflow. Optional logging and artifacts can
be branch-specific nodes, not separate duplicated pipelines.

## Developer Tools And CLIs

Use cases:

- `params` command that lists configurable values.
- `run` command that accepts overrides.
- `info` command that instantiates a graph and reports inputs/outputs.
- Local exploration of conditional branches.

Hypster angle:

The config function becomes the source for both runtime construction and tool
introspection.

## Notebook Workflows

Use cases:

- Data scientists select models, retrieval options, and evaluation settings
  interactively.
- Branch-specific widgets appear only when relevant.
- Results update as parameter values change.

Hypster angle:

Interactive configuration is the natural bridge between Python code and
exploratory model/system work.

## AI Agent Tool Use

Use cases:

- Agent chooses a RAG variant for a query.
- Agent selects cheap vs accurate mode based on budget.
- Agent runs a batch experiment overnight.
- Agent inspects schema before deciding values.
- Agent logs params and metrics for follow-up reasoning.

Hypster angle:

Expose workflow variants as a typed, inspectable parameter tree rather than
asking the agent to modify code or invent JSON from memory.

## No-Code Or Low-Code Control Planes

Use cases:

- Build a UI from config schema.
- Let operators choose workflow settings safely.
- Clone/import/export project configurations.
- Compare configurations side by side.
- Restrict available options by tenant or role.

Hypster angle:

Hypster can be the Python-side source of truth underneath a UI or control plane.
