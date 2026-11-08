# Schema

`hp.schema()` selects a list of typed field definitions — `SchemaField` objects — as a normal Hypster configuration value. It is useful when *which fields to produce* should itself be configuration: extraction targets for a structured-LLM call, a document metadata contract, or any form-like definition an end user should be able to edit in a UI and have logged and replayed with the rest of the run.

{% code overflow="wrap" %}
```python
from hypster import HP, instantiate
from hypster.schema_field import SchemaField


def extraction_config(hp: HP) -> list[SchemaField]:
    return hp.schema(
        name="extraction_fields",
        default=[
            SchemaField(key="invoice_number", value_type="text", label="Invoice Number"),
            SchemaField(key="total_amount", value_type="number", unit="USD"),
            SchemaField(key="status", value_type="enum", possible_values=["paid", "unpaid", "overdue"]),
        ],
    )


fields = instantiate(extraction_config)
```
{% endcode %}

`fields` contains real `SchemaField` objects your config function (or the component it constructs) can use directly. The selected params are JSON-safe dicts, so schema values override, log, and replay exactly like every other Hypster value.

## SchemaField

{% code overflow="wrap" %}
```python
@dataclass(frozen=True)
class SchemaField:
    key: str                              # machine name, e.g. "invoice_number"
    value_type: str                       # "text" | "enum" | "number" | "date"
    description: str = ""                 # extraction instruction / help text
    label: str = ""                       # human display label
    multi_valued: bool = False            # list of value_type instead of a scalar
    possible_values: list[str] | None = None   # only for value_type="enum"
    unit: str | None = None               # e.g. "USD", "days"
    required: bool = False                # mandatory for consumers that gate on it
    may_be_empty: bool = False            # an explicit empty value is one of its values
```
{% endcode %}

Round-trip with `.to_dict()` / `SchemaField.from_dict(data)`.

`required` lives on the field itself so renaming a field can never orphan its requiredness in a parallel key list. It is deliberately **not** emitted by `.to_json_schema()`: JSON Schema expresses requiredness as a `required` array on the parent object, so the code assembling the full schema reads the flag and builds that array.

## Empty values and `check`

`may_be_empty` says whether "nothing" is a value the field may hold. An explicit empty value is `None`, a blank string or an empty list (`hypster.schema_field.is_empty`). It is apart from `required`: required says a value must be given, `may_be_empty` says whether an empty one counts.

`field.check(value)` returns why a value is not one of the field's values, naming the field, or `None` when it is:

{% code overflow="wrap" %}
```python
district = SchemaField(key="district", value_type="enum", possible_values=["North", "South"])
district.check(None)      # "district: may not be empty"
district.check("East")    # "district: 'East' is not one of North, South"
district.check("North")   # None

SchemaField(key="district", value_type="enum", possible_values=["North"], may_be_empty=True).check(None)  # None
```
{% endcode %}

A number is an `int` or a finite `float` (never a `bool`); a date is a `datetime.date` or a `YYYY-MM-DD` string; a multi-valued field takes a list of such values. Whether a value must be given at all, and what fills one left out, is the caller's.

## Feeding a structured-output LLM call

`.to_json_schema()` converts one field into a JSON Schema property definition, ready to assemble into a structured-output request:

{% code overflow="wrap" %}
```python
SchemaField(key="total_amount", value_type="number", unit="USD").to_json_schema()
# {'type': 'number'}

SchemaField(key="status", value_type="enum", possible_values=["paid", "unpaid"]).to_json_schema()
# {'type': 'string', 'enum': ['paid', 'unpaid']}

SchemaField(key="tags", value_type="text", multi_valued=True).to_json_schema()
# {'type': 'array', 'items': {'type': 'string'}}
```
{% endcode %}

`"date"` maps to `{"type": "string", "format": "date"}`; a set `description` is included on the property. A field that may be empty admits `null`: `{"anyOf": [<property>, {"type": "null"}]}`.

## Rendering a schema editor in a UI

`explore(..., return_schema=True)` records the parameter with `kind="schema"` and `metadata` carrying `schema_fields` — the `to_dict()` form of every field — which is exactly what a "define what to extract" form needs to render controls and round-trip user edits back through `values=`:

{% code overflow="wrap" %}
```python
from hypster import explore

schema = explore(extraction_config, return_schema=True)
schema.parameters[0].kind                              # "schema"
schema.parameters[0].metadata["schema_fields"][0]["key"]  # "invoice_number"
```
{% endcode %}

## Not the same as `field` / `FieldSpec`

`SchemaField` describes an **output field** — what an extractor or LLM should produce. The `hypster.field` constructors (see [Rules](rules.md)) build **`FieldSpec`** objects — the condition/payload vocabulary for `hp.rules()`. They share no code path; a rules condition never uses `SchemaField`, and a schema definition never uses `FieldSpec`.
