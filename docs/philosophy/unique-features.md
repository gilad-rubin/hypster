---
icon: alicorn
---

# Unique Features

## Define-By-Run Configuration

Hypster config functions are normal Python rather than a DSL. Branches, loops, helper functions, lists, imports, and object construction work the way Python developers expect.

The price of that flexibility is honest execution: Hypster discovers parameters by running the config function. Keep discovery paths fast and side-effect-free so exploration, UI rendering, and HPO can rerun them safely.

## Branch-Aware Exploration

`explore(config, values=...)` traces the same branch that `instantiate(config, values=...)` would run. That makes it useful for UIs, HPO, schema exports, and debugging stale values.

## Replayable Params Sidecar

`instantiate_with_params()` returns both the runtime value and the selected parameter dictionary. Defaults are included, so replay does not depend on future default changes.

## Nested Composition

`hp.nest()` gives reusable child configs their own dotted parameter paths without requiring a global registry or decorator.

## Dict-Backed Selects

Hypster separates the logged key from the runtime value. This lets you log `"large"` while returning an object, dictionary, tuple, or factory.

For swappable components, the idiomatic shape is a named options dictionary from simple keys to config functions, followed by `hp.nest()` on the selected function. That keeps logs stable, UI options simple, and the parent config readable even when the option set grows.

## Strict Unknown Values By Default

Unknown and unreachable values raise by default. This is stricter than many config systems, but it protects experiment logs and production replays from silently accepting stale parameters.
