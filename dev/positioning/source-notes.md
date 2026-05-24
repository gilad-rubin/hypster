# Source Notes

This file maps positioning ideas back to sources.

## Current Hypster Docs

Local paths inspected:

- `README.md`
- `docs/README.md`
- `docs/SUMMARY.md`
- `docs/getting-started/defining-a-configuration-space.md`
- `docs/getting-started/exploring-a-configuration-space.md`
- `docs/getting-started/interactive-instantiation-ui.md`
- `docs/in-depth/nest.md`
- `docs/in-depth/basic-best-practices.md`
- `docs/in-depth/performing-hyperparameter-optimization.md`
- placeholder reproducibility pages under `docs/reproducibility/`

Extracted ideas:

- Hypster is currently described as a lightweight config framework for AI/ML.
- Strong docs concepts: config functions, `explore`, `instantiate`, nested
  configs, values/overrides, HPO, Pythonic best practices.
- Current docs already support "regular Python function" language.
- `explore` is a major differentiator because it prints and returns structured
  parameter metadata.
- `hp.nest` is central to the hierarchical/modular story.

Open issues before public positioning:

- Reproducibility pages are placeholders.
- Interactive docs say the UI was removed, but source/tests include
  `interactive_explore`.
- Some philosophy pages are placeholders.

## "Introducing HyPSTER" Article

URL:

https://medium.com/@giladrubin/introducing-hypster-a-pythonic-framework-for-managing-configurations-to-build-highly-optimized-ai-5ee004dbd6a5

Extracted ideas:

- AI workflows balance speed, cost, and performance.
- Projects need multiple modes: dev/prod, local/remote, different workflow
  types, and specialized sub-workflows.
- Hypster is for managing those modes while optimizing each context.
- Benefits include swappable/hierarchical configurations, Pythonic authoring,
  nested/conditional logic, DRY config design, UI-centric exploration, and HPO.
- The article emphasizes the "superposition" idea: one function defines many
  possible concrete configurations.

Legacy API notes:

- Uses `@config`, `hp.number`, automatic naming, `final_vars`, `save`, and
  `load`, which should not be copied directly into current public docs without
  updating.

## "Don't Build One AI Pipeline. Build 100s Instead."

URL:

https://pub.towardsai.net/dont-build-one-ai-pipeline-build-100s-instead-9c905569e9d6

Extracted ideas:

- A single hard-coded AI pipeline causes rigidity, slow iteration, lost
  knowledge, unintentional overfitting, and missed configurations.
- Version control and experiment tracking often preserve only tested snapshots,
  not the broader space of valid alternatives.
- The proposed shift is from one pipeline to a space of potential pipelines.
- Configuration functions contain hyperparameters, conditional dependencies,
  and hierarchical/swappable configurations.
- Benefits: improved code quality, knowledge accumulation, manageable nested
  configurations, accelerated development, optimized pipelines, flexibility, and
  future-proofing.
- Strong positioning phrase to adapt: creation of a pipeline is driven by
  configuration, not hard-coded logic.

Public copy translation:

- "Turn a hard-coded pipeline into a family of valid workflows."
- "Keep alternatives alive as configurable choices."
- "Optimize whole workflow structure, not just model hyperparameters."

## "Implementing Modular RAG With Haystack And Hypster"

URL:

https://medium.com/data-science/implementing-modular-rag-with-haystack-and-hypster-d2f0ecc88b8f

Extracted ideas:

- Modular RAG is a natural showcase because different modules can be swapped,
  nested, and connected conditionally.
- Hypster config can alter pipeline topology: enrichment on/off, retriever
  choices, reranker branches, provider choices, and nested LLM configs.
- Dot notation makes nested overrides understandable.
- The article frames this as building a codebase that accommodates multiple
  potential workflows.
- Benefits named: HPO, diverse scenario-specific configs, agentic tool use, and
  production A/B testing.

Public copy translation:

- "RAG is not one pipeline; it is a family of valid retrieval/generation
  workflows."
- "Use nested configs for indexing, retrieval, generation, and evaluation."
- "Let query type, customer segment, or product mode choose a concrete workflow."

## Reproducible Instantiation Branch

Branch inspected:

- `codex/reproducible-instantiation-params`

Files inspected via `git show`/`git grep`:

- `src/hypster/core.py`
- `src/hypster/__init__.py`
- `docs/getting-started/instantiating-a-configuration-space.md`
- `docs/adr/0001-strict-unknown-values.md`
- `tests/test_instantiate_with_params.py`

Extracted ideas:

- `instantiate_with_params` returns `InstantiationOutput(value, params)`.
- `params` contains selected reachable params, including defaults.
- `params` can replay the run via `instantiate(..., values=run.params)`.
- Strict unknown handling is part of reproducibility because stale or unreachable
  values should not be silently logged.
- This API is ideal for MLflow, W&B, Langfuse, OpenTelemetry, or custom
  observability metadata.

Public copy caveat:

- Do not put this in public README as current API until the branch lands on the
  release branch.

## Architecture Learning: Hypster Plus Execution Graphs

Extracted ideas:

- Hypster is strongest as the configuration layer beside a graph, DAG, pipeline,
  or workflow execution layer.
- Parent workflow configs can nest child workflow configs.
- Config choices can bind dependencies into graph nodes.
- Config values can select optional nodes, graph branches, providers, prompts,
  and runtime components.
- Config functions can act as factories for components and subgraphs.
- Introspection can power CLI, notebook, or UI surfaces that show configurable
  parameters before a workflow is run.
- Evaluation and observability can be branch-specific workflow behavior, not
  separate duplicated pipelines.

Public copy translation:

- "Hypster configures the family of possible workflows; your execution layer
  runs the selected workflow."
- "Use config functions as factories for components, dependencies, and
  sub-workflows."
- "Let the configuration tree mirror your workflow architecture."
- "Keep configuration selection separate from runtime execution."
