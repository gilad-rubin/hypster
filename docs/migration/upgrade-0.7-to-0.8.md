# Upgrade From 0.7 To 0.8

Hypster 0.8 removes an accidental integration namespace and renames one schema metadata key. Update those two surfaces before upgrading; ordinary config functions, `explore()`, `instantiate()`, rules, and replayable values keep their 0.7 shapes.

Install the target release and check the version:

```bash
uv add "hypster==0.8.*"
uv run python -c "import hypster; print(hypster.__version__)"
```

## Import HPO APIs From `hypster.hpo`

The removed `hypster.integrations` package used a wildcard export and exposed implementation details as if they were public. Import the HPO API from its owning package.

Before:

```python
from hypster.integrations import HpoInt, suggest_values
```

After:

<!-- test: exec -->
```python
from hypster.hpo import HpoInt, TrialValueProvider, suggest_values

assert HpoInt(step=2).step == 2
assert callable(suggest_values)
assert TrialValueProvider.__name__ == "TrialValueProvider"
```

You can also import `suggest_values` from `hypster.hpo.optuna`. No compatibility alias remains at `hypster.integrations`.

## Read Schema Definitions From `schema_fields`

Metadata emitted by `hp.schema()` now uses `schema_fields`. The `field_specs` key remains the vocabulary for the conditions accepted by `hp.rules()`; the two concepts are intentionally distinct.

Before:

```python
fields = parameter["metadata"]["field_specs"]
```

After:

<!-- test: exec -->
```python
from hypster import HP, SchemaField, explore


def extraction_config(hp: HP):
    return hp.schema(
        default=[SchemaField(key="invoice_number", value_type="text", required=True)],
        name="fields",
    )


schema = explore(extraction_config, return_schema=True)
metadata = schema.to_dict()["parameters"][0]["metadata"]

assert metadata["schema_fields"][0]["key"] == "invoice_number"
assert metadata["schema_fields"][0]["required"] is True
assert "field_specs" not in metadata
```

## Check HPO Behavior That Previously Failed

The 0.8 Optuna adapter validates trial suggestions through the same path as explicit values. Integer trials default to a step of `1`; log-scale integer trials enforce Optuna's step rule. Parameter kinds without an Optuna mapping, including `hp.bool`, `hp.text`, and `multi_*`, keep their validated defaults and are omitted from `suggest_values()`'s returned dictionary.

If a search relied on a raw Optuna error or on an unsupported kind appearing in returned suggestions, update the search space explicitly. See [Optuna HPO API](../reference/optuna-hpo.md) for the current mapping and error cases.

## Verify The Upgrade

Run your config tests and search for the two removed 0.7 forms:

```bash
rg -n "hypster\.integrations|\[\"field_specs\"\]" .
uv run pytest
```

An occurrence of `field_specs` can still be correct when it reads `hp.rules()` metadata.

For all user-visible 0.8 changes, see the [changelog](../../CHANGELOG.md).
