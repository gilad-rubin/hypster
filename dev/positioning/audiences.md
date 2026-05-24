# Audiences

Hypster should feel technical, useful, and unsentimental. Different readers
care about different benefits.

## Python Developers

Pain:

- Configuration logic gets duplicated across files and scripts.
- Complex branches are painful to represent in static config formats.
- Production code and experiment code drift apart.
- Adding a new provider/component requires touching too many places.

Benefit:

- Use ordinary Python functions for configuration.
- Compose reusable configs instead of copying setup code.
- Keep conditional logic close to the objects and workflows it configures.
- Refactor configuration with the same tools used for code.

Good copy:

- "Use normal Python control flow to manage complex configuration."
- "Define config modules once, then compose them into larger workflows."

## Data Scientists

Pain:

- Notebooks accumulate ad hoc parameters.
- Experiment settings are hard to replay.
- Valuable alternatives disappear after the best run is found.
- Manual trial-and-error slows down iteration.

Benefit:

- Explore options interactively.
- Capture selected params for each run.
- Move from a one-off notebook to a reusable configuration function.
- Reuse the same config for manual runs and HPO.

Good copy:

- "Make experiments replayable without rewriting your workflow."
- "Keep the decision space alive after a run finishes."

## AI Engineers

Pain:

- RAG/agent systems have many moving parts.
- Pipeline structure changes based on provider, mode, query type, cost, latency,
  and evaluation results.
- A single hard-coded pipeline becomes brittle quickly.

Benefit:

- Model a family of valid workflows.
- Compose configs for retrievers, prompts, LLMs, embedders, vector stores, and
  evaluators.
- Test new combinations without rewriting core execution logic.
- Route different requests to different valid configurations.

Good copy:

- "Build configurable AI systems from reusable parts."
- "Treat your RAG pipeline as a design space, not a single hard-coded path."

## ML Engineers And MLOps

Pain:

- HPO often focuses only on model parameters.
- Experiment trackers receive incomplete or inconsistent config metadata.
- Defaults are not always logged.
- Unknown/unreachable config values can silently corrupt reproducibility.

Benefit:

- Optimize whole workflows, including preprocessing, retrieval, prompts, and
  model choices.
- Log a complete selected-params sidecar.
- Replay the exact parameter set.
- Fail loudly on unknown or unreachable values when strictness matters.

Good copy:

- "Log the concrete parameter set behind each run."
- "Optimize workflow structure and hyperparameters from the same config."

## Platform Teams

Pain:

- Internal users need flexibility without editing production code.
- Multiple teams need different variants of the same workflow.
- UI/control-plane surfaces need a reliable source of configurable options.

Benefit:

- Generate parameter UIs from config schemas.
- Centralize valid options and defaults.
- Keep custom variants bounded and inspectable.
- Support per-request configuration in APIs.

Good copy:

- "Expose safe workflow variants without handing users the codebase."
- "Use the config tree as the contract between platform and application code."

## Product Teams

Pain:

- AI features need A/B tests, customer-specific behavior, and fast iteration.
- Quality, latency, and cost tradeoffs change by context.
- Product changes often force code changes in the pipeline layer.

Benefit:

- Define supported modes explicitly.
- Route users or requests to different configurations.
- Compare variants against the same evaluation harness.
- Add new product modes as config branches.

Good copy:

- "Ship multiple valid modes of the same AI system."
- "Adapt quality, latency, and cost tradeoffs by scenario."

## AI Agents

Pain:

- Agents need to operate inside explicit boundaries.
- Asking an agent to edit code to change behavior is risky.
- Agents need introspection before selecting a workflow.

Benefit:

- The config schema becomes an action surface.
- The agent can inspect options, defaults, and bounds.
- The agent can select values, run, observe metrics, and replay.
- Human developers still own the valid choices.

Good copy:

- "Give agents a bounded way to choose workflow variants."
- "Let agents configure systems without rewriting them."
