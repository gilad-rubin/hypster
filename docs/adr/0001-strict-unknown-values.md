# Strict Unknown Values

Hypster treats `values` as a reproducibility surface, so unknown or unreachable overrides should fail loudly by default instead of being silently logged or replayed incorrectly. We changed the default unknown-parameter policy for `instantiate`, `instantiate_with_params`, and `explore` to `raise`, while keeping `warn` and `ignore` available for callers who intentionally want softer behavior.

## Consequences

- Nested-dict `values` must participate in unknown/unreachable checking as their equivalent dotted parameter paths.
- A parameter path specified more than once through mixed dotted and nested forms always raises as malformed input, even when the duplicate values are equal.
- Unknown/unreachable errors should guide users to `explore(config, values=...)` to inspect the active branch, but `instantiate` must not execute extra branch exploration automatically.
- This is stricter than the previous default, but it better matches logging and replay workflows where ignored overrides are usually bugs.
