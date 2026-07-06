"""Tests for hp.schema() and SchemaField."""

import json

import pytest

from hypster import SchemaField, instantiate_with_params
from hypster.explore import explore


def _make_fields():
    return [
        SchemaField(key="invoice_number", value_type="text", label="Invoice Number"),
        SchemaField(key="total_amount", value_type="number", unit="USD"),
        SchemaField(
            key="status",
            value_type="enum",
            possible_values=["paid", "unpaid", "overdue"],
            description="Payment status of the invoice.",
        ),
        SchemaField(key="due_date", value_type="date"),
    ]


# --- hp.schema() values ---


def test_schema_returns_list_of_schema_field_objects():
    def config(hp):
        return hp.schema(name="extraction_fields", default=_make_fields())

    from hypster.core import instantiate

    result = instantiate(config)
    assert isinstance(result, list)
    assert len(result) == 4
    assert all(isinstance(f, SchemaField) for f in result)
    assert result[0].key == "invoice_number"
    assert result[2].possible_values == ["paid", "unpaid", "overdue"]


def test_schema_empty_default():
    def config(hp):
        return hp.schema(name="extraction_fields")

    from hypster.core import instantiate

    result = instantiate(config)
    assert result == []


def test_schema_override_changes_constructed_value():
    """Overriding via values= must change the constructed value, not just recorded params."""

    def config(hp):
        return hp.schema(name="fields", default=[SchemaField(key="a", value_type="text")])

    from hypster.core import instantiate

    out = instantiate(config, values={"fields": [{"key": "b", "value_type": "enum", "possible_values": ["x"]}]})
    assert out[0].key == "b" and out[0].possible_values == ["x"]
    assert isinstance(out[0], SchemaField)


def test_schema_rejects_non_field_items():
    def config(hp):
        return hp.schema(name="fields", default=[42])

    from hypster.core import instantiate

    with pytest.raises(ValueError, match="expected a SchemaField or dict"):
        instantiate(config)


# --- params logging and replay ---


def test_schema_params_are_json_safe_and_replay_byte_identically():
    def config(hp):
        return hp.schema(name="extraction_fields", default=_make_fields())

    run = instantiate_with_params(config)
    serialized = json.dumps(run.params, sort_keys=True)

    replay = instantiate_with_params(config, values=json.loads(serialized))
    assert replay.value == run.value
    assert json.dumps(replay.params, sort_keys=True) == serialized


# --- explore() schema ---


def test_schema_explore_kind_is_schema():
    def config(hp):
        hp.schema(name="extraction_fields", default=_make_fields())

    schema = explore(config, return_schema=True)
    assert schema is not None
    param = schema.parameters[0]
    assert param.kind == "schema"


def test_schema_explore_metadata_has_field_specs():
    def config(hp):
        hp.schema(name="extraction_fields", default=_make_fields())

    schema = explore(config, return_schema=True)
    assert schema is not None
    param = schema.parameters[0]
    metadata = param.metadata
    assert metadata is not None
    assert "field_specs" in metadata
    assert len(metadata["field_specs"]) == 4
    assert metadata["field_specs"][0]["key"] == "invoice_number"
    assert metadata["field_specs"][2]["possible_values"] == ["paid", "unpaid", "overdue"]


# --- SchemaField.to_json_schema() ---


def test_to_json_schema_text():
    assert SchemaField(key="title", value_type="text").to_json_schema() == {"type": "string"}


def test_to_json_schema_enum():
    prop = SchemaField(key="status", value_type="enum", possible_values=["paid", "unpaid"]).to_json_schema()
    assert prop == {"type": "string", "enum": ["paid", "unpaid"]}


def test_to_json_schema_number():
    assert SchemaField(key="total", value_type="number").to_json_schema() == {"type": "number"}


def test_to_json_schema_date():
    assert SchemaField(key="due", value_type="date").to_json_schema() == {"type": "string", "format": "date"}


def test_to_json_schema_multi_valued_wraps_in_array():
    prop = SchemaField(key="tags", value_type="text", multi_valued=True).to_json_schema()
    assert prop == {"type": "array", "items": {"type": "string"}}


def test_to_json_schema_includes_description():
    prop = SchemaField(key="dose", value_type="number", description="Dose in mg.").to_json_schema()
    assert prop == {"type": "number", "description": "Dose in mg."}


# --- SchemaField dict round-trip ---


def test_schema_field_dict_round_trip():
    original = SchemaField(
        key="status",
        value_type="enum",
        description="Payment status.",
        label="Status",
        multi_valued=True,
        possible_values=["paid", "unpaid"],
        unit="n/a",
    )
    assert SchemaField.from_dict(original.to_dict()) == original


def test_schema_field_to_dict_omits_defaults():
    assert SchemaField(key="title", value_type="text").to_dict() == {"key": "title", "value_type": "text"}


# --- SchemaField.required (PRD 0027: requiredness lives ON the field) ---


def test_schema_field_required_defaults_false():
    assert SchemaField(key="title", value_type="text").required is False


def test_schema_field_required_round_trips_through_dict():
    field = SchemaField(key="station", value_type="enum", possible_values=["a", "b"], required=True)
    assert field.to_dict()["required"] is True
    assert SchemaField.from_dict(field.to_dict()) == field
    # And the default stays out of the dict, so existing pins are byte-stable.
    assert "required" not in SchemaField(key="tags", value_type="text").to_dict()
