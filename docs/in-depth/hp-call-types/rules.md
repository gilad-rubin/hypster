# Rules

`hp.rules()` selects a list of declarative WHEN/THEN rules as a normal Hypster configuration value. It is useful when a config needs user-authored policy, prompt additions, routing conditions, or lightweight business rules that should be logged and replayed with the rest of the run.

Rules are built into Hypster. You can construct fields and rules with `hypster.field`, `Rule`, `Leaf`, `And`, `Or`, and `Not`; `hyperrules` is not required to define or serialize them.

{% code overflow="wrap" %}
```python
from hypster import HP, Leaf, Rule, field, instantiate_with_params


audience = field.select(["clinical", "operations"], name="audience")
document_tag = field.multi_select(["drug_leaflet", "formulary"], name="document_tag")
prompt = field.text(name="prompt", multiline=True)


def prompt_config(hp: HP) -> list[Rule[str]]:
    return hp.rules(
        when=[audience, document_tag],
        then=prompt,
        name="prompt_rules",
        default=[
            audience.eq("clinical").then("Prefer protocol language."),
            Rule(
                when=Leaf("document_tag", "in", ["formulary"]),
                then="Use formulary wording.",
                name="formulary",
            ),
        ],
    )


run = instantiate_with_params(prompt_config)
```
{% endcode %}

`run.value` contains `Rule` objects:

{% code overflow="wrap" %}
```python
assert isinstance(run.value[0], Rule)
assert run.value[0].then == "Prefer protocol language."
```
{% endcode %}

`run.params` contains JSON-friendly dictionaries that can be logged and replayed:

{% code overflow="wrap" %}
```python
assert run.params == {
    "prompt_rules": [
        {
            "when": {"field": "audience", "operator": "=", "value": "clinical"},
            "then": "Prefer protocol language.",
        },
        {
            "when": {"field": "document_tag", "operator": "in", "value": ["formulary"]},
            "then": "Use formulary wording.",
            "name": "formulary",
        },
    ]
}
```
{% endcode %}

## Field Specs

Use `field.*` constructors to declare the condition vocabulary and payload shape:

| Constructor | Use for |
| --- | --- |
| `field.select(options, name=...)` | Single categorical condition field. |
| `field.multi_select(options, name=...)` | Multi-value categorical condition field. |
| `field.bool(name=...)` | Boolean condition or payload field. |
| `field.multi_bool(name=...)` | Multi-value boolean field. |
| `field.text(name=..., multiline=False)` | Text payloads, including prompt blocks. |
| `field.multi_text(name=...)` | List-of-text payload field. |
| `field.int(name=...)` / `field.float(name=...)` | Numeric condition or payload fields. |
| `field.multi_int(name=...)` / `field.multi_float(name=...)` | List-of-numeric payload fields. |

Field specs are schema declarations, not selected params. They appear in `explore(..., return_schema=True)` under the rules parameter metadata so UIs can render the rule builder.

## FieldSpec Methods And Default Operators

Every `field.*` constructor returns a `FieldSpec` with a fixed `type`. That `type` picks a default `operators` tuple, and each `FieldSpec` method builds a `Leaf` using one specific operator string. The method name and the operator string it writes are not always the same word:

| Method | Leaf operator built | Available on default operators for |
| --- | --- | --- |
| `.eq(value)` | `"="` | `select`, `int`, `float`, `text` |
| `.neq(value)` | `"!="` | `select`, `int`, `float` |
| `.is_in(values)` | `"in"` | `select`, `multi_select`, `multi_bool`, `multi_int`, `multi_float`, `multi_text` |
| `.not_in(values)` | `"not_in"` | `select`, `multi_select` |
| `.is_true()` | `"is"` with value `True` | `bool` |
| `.is_false()` | `"is"` with value `False` | `bool` |
| `.gt(value)` | `">"` | `int`, `float` |
| `.lt(value)` | `"<"` | `int`, `float` |
| `.contains(value)` | `"contains"` | `text`, `multi_int`, `multi_float`, `multi_text` |

Default operators per type (`FieldSpec.operators`, populated from each `field.*` constructor):

| Type | Default `operators` |
| --- | --- |
| `select` | `=`, `!=`, `in`, `not_in` |
| `multi_select` | `in`, `not_in` |
| `bool` | `is` |
| `multi_bool` | `in` |
| `int` | `>`, `<`, `=`, `!=` |
| `multi_int` | `in`, `contains` |
| `float` | `>`, `<`, `=`, `!=` |
| `multi_float` | `in`, `contains` |
| `text` | `=`, `contains` |
| `multi_text` | `in`, `contains` |

`.is_true()` and `.is_false()` both build an `"is"` leaf, differing only in the boolean value. There is no `.neq_in()`; the negated-membership method is `.not_in(values)`.

`FieldSpec` methods do not check the `operators` tuple before building a `Leaf`. Calling a method whose operator is outside the type's own default list (for example `.gt(...)` on a `field.text(...)`) still returns a `Leaf` with that operator string instead of raising:

```python
from hypster import field

label = field.text(name="label")
label.gt("x")
# Leaf(field='label', operator='>', value='x')
```

## Composite Payloads

Pass a list of named field specs to `then=` when each rule needs a structured payload:

{% code overflow="wrap" %}
```python
def prompt_config(hp: HP) -> list[Rule[dict[str, object]]]:
    return hp.rules(
        when=[audience, document_tag],
        then=[
            field.text(name="prompt", multiline=True),
            field.bool(name="use_citations"),
        ],
        name="prompt_rules",
        default=[
            Rule(
                when=audience.eq("clinical"),
                then={"prompt": "Prefer protocol language.", "use_citations": True},
            )
        ],
    )
```
{% endcode %}

Every `then` field spec must have a `name=`. Hypster raises if `then=` is a string, an unnamed field spec, or a list containing non-field specs.

## Schema And Notebook UI

Rules appear in schemas with `kind="rules"`. The selected value is a JSON-friendly list, and metadata includes the fields needed by renderers:

{% code overflow="wrap" %}
```python
from hypster import explore

schema = explore(prompt_config, return_schema=True)
rules_param = schema.to_dict()["parameters"][0]

assert rules_param["kind"] == "rules"
assert rules_param["metadata"]["field_specs"][0]["name"] == "audience"
assert rules_param["metadata"]["then_specs"][0]["name"] == "prompt"
```
{% endcode %}

The built-in Jupyter UI from `interact()` renders rules parameters from this same schema metadata. It supports existing rules, adding and removing rules, editing flat condition lists, and editing typed THEN payload fields. Deeply nested boolean expressions still round-trip as values and display as summaries; edit those in Python or a custom UI if you need full tree editing.
