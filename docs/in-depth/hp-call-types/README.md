# HP Call Types

`HP` methods define the public parameters in a config function. Each call records a parameter path, validates overrides, and can be explored or replayed.

The examples in this reference sometimes return small dictionaries to keep the call behavior visible. In application code, prefer returning the initialized runtime object unless the mapping itself is the object your caller needs.

## Scalar Calls

| Call | Use for | Notes |
| --- | --- | --- |
| `hp.int` | Integer parameters | Accepts integral floats by default, optional bounds, optional strict mode, optional `allow_none=True`. |
| `hp.float` | Floating-point parameters | Accepts integer values by default, optional bounds, optional strict mode, optional `allow_none=True`. |
| `hp.text` | Strings | Use for prompts, paths, IDs, and labels. |
| `hp.bool` | Booleans | Requires actual `True` or `False`, not string values. |
| `hp.select` | One categorical choice | Supports list options or dict-backed key-to-value mapping. |

## Multi-Value Calls

| Call | Use for | Notes |
| --- | --- | --- |
| `hp.multi_int` | List of integers | Elements use the same safe coercion and strict-mode behavior as `hp.int`. |
| `hp.multi_float` | List of floats | Elements use the same safe coercion and strict-mode behavior as `hp.float`. |
| `hp.multi_text` | List of strings | Useful for columns, tags, stop sequences, and feature names. |
| `hp.multi_bool` | List of booleans | Useful when each position has meaning. |
| `hp.multi_select` | List of categorical choices | Supports nullable choices with `allow_none=True`. |

Nullable elements are not supported for `multi_int`, `multi_float`, `multi_text`, or `multi_bool`. Use `multi_select(..., allow_none=True)` for nullable categorical lists.

## Composition Calls

| Call | Use for |
| --- | --- |
| `hp.nest` | Run another config function under a named scope. |
| `hp.collect` | Collect selected local variables into a returned dictionary. |

## Shared Rules

* `name=` is required for every `hp.*` parameter call.
* Names must be valid Python identifiers and cannot contain dots, spaces, or hyphens.
* `values=` may use dotted paths such as `optimizer.learning_rate`.
* Unknown or unreachable values raise by default.
* Dict-backed `select` is the right way to return complex objects while logging simple keys.
* Numeric parameters reject `True` and `False` even though Python treats `bool` as a subclass of `int`.

See [Public API](../../reference/api.md) for exact signatures.
