# Hypster Positioning Notes

This folder is a raw strategy workspace for Hypster's differentiation, benefits,
audiences, use cases, and product language.

It is intentionally not a user guide. The public docs explain how to use
Hypster. This space explains why Hypster matters, who it helps, where it fits,
and how to talk about it without turning the README into sales copy.

## Core Direction

Hypster is a hierarchical, modular configuration management framework for
Python, built for complex AI and ML workflows.

Supporting version:

Hypster helps you define the valid ways your system can run using plain Python
functions, compose reusable configs for nested components, explore the active
configuration tree, and instantiate or optimize one concrete setup when you
need it.

## Messaging Principles

- Lead with a concrete developer problem, not a category phrase.
- Explain "configuration space" only after grounding it in "valid ways your
  system can run."
- Use positive framing: "plain Python functions", "normal control flow",
  "nested reusable configs", "explorable configuration tree."
- Keep the README technical and useful. The proof should be in a tiny example,
  then a compact benefit list.
- Avoid positioning Hypster only as HPO. HPO is one major benefit, but the
  bigger idea is making complex systems configurable, inspectable, reusable,
  observable, and optimizable.
- Treat AI agents as a real audience: a Hypster config gives an agent a bounded,
  typed, inspectable action surface for choosing workflow variants.

## Files

- [benefits.md](benefits.md) - benefit inventory and payoffs.
- [use-cases.md](use-cases.md) - product, AI/ML, RAG, agent, and platform use
  cases.
- [audiences.md](audiences.md) - who Hypster is for and what each group gets.
- [reproducibility-observability.md](reproducibility-observability.md) -
  `instantiate_with_params`, replay, and logging story.
- [positioning-language.md](positioning-language.md) - taglines, README bullets,
  and words to use or avoid.
- [source-notes.md](source-notes.md) - source map from articles, current docs,
  branches, and architecture lessons.

## Current Product Truths To Check Before Public Copy

- The current checkout exposes `instantiate`, `explore`, `interactive_explore`,
  and `apply_vscode_theme`.
- `instantiate_with_params` exists on the `codex/reproducible-instantiation-params`
  branch, not in the current `master` checkout.
- Current docs still have placeholder pages under `docs/reproducibility/`.
- Current docs mention that interactive UI was removed, while the current source
  includes `interactive_explore`. Public copy should reconcile this before
  making interactive UI a headline feature.
- Older articles use legacy API names such as `@config`, `hp.number`, automatic
  naming, `final_vars`, `save`, and `load`. Reuse the ideas, but translate
  examples into the current API before publishing.
