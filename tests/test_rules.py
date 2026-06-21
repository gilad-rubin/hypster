"""Tests for hp.rules() and hp.text(multiline=)."""

import pytest
from hyperrules import And, Leaf, Rule, field

from hypster.explore import explore


def _make_fields():
    return [
        field.multi_select(["drug_leaflet", "formulary"], name="document_tag"),
        field.select(["NICU", "ER", "ICU"], name="document_station"),
        field.bool(name="vision_page_presence"),
    ]


THEN_PROMPT = field.text(name="prompt", multiline=True)


def _sample_rules():
    return [
        Rule(
            when=And(Leaf("document_tag", "in", ["drug_leaflet"])),
            then="Drug dosing instructions",
            name="drug_leaflet",
        ),
    ]


# --- hp.text(multiline=) ---


def test_text_multiline_records_metadata():
    def config(hp):
        hp.text("", name="system_prompt", multiline=True)

    schema = explore(config, return_schema=True)
    param = schema.parameters[0]
    assert param.metadata == {"multiline": True}


def test_text_without_multiline_no_metadata():
    def config(hp):
        hp.text("", name="title")

    schema = explore(config, return_schema=True)
    param = schema.parameters[0]
    assert param.metadata is None


def test_text_multiline_value_returned():
    def config(hp):
        return hp.text("default", name="prompt", multiline=True)

    from hypster.core import instantiate

    result = instantiate(config)
    assert result == "default"

    result_override = instantiate(config, values={"prompt": "custom"})
    assert result_override == "custom"


# --- hp.rules() ---


def test_rules_returns_list_of_rule_objects():
    def config(hp):
        return hp.rules(when=_make_fields(), then=THEN_PROMPT, name="additions", default=_sample_rules())

    from hypster.core import instantiate

    result = instantiate(config)
    assert isinstance(result, list)
    assert len(result) == 1
    assert isinstance(result[0], Rule)
    assert result[0].name == "drug_leaflet"


def test_rules_schema_has_kind_rules():
    def config(hp):
        hp.rules(when=_make_fields(), then=THEN_PROMPT, name="additions")

    schema = explore(config, return_schema=True)
    param = schema.parameters[0]
    assert param.kind == "rules"


def test_rules_schema_metadata_has_field_specs():
    def config(hp):
        hp.rules(when=_make_fields(), then=THEN_PROMPT, name="additions")

    schema = explore(config, return_schema=True)
    param = schema.parameters[0]
    assert "field_specs" in param.metadata
    assert len(param.metadata["field_specs"]) == 3
    assert param.metadata["field_specs"][0]["name"] == "document_tag"
    assert param.metadata["field_specs"][0]["type"] == "multi_select"


def test_rules_schema_metadata_has_then_specs():
    def config(hp):
        hp.rules(when=_make_fields(), then=field.text(name="prompt", multiline=True), name="additions")

    schema = explore(config, return_schema=True)
    param = schema.parameters[0]
    assert "then_specs" in param.metadata
    assert param.metadata["then_specs"][0]["type"] == "text"
    assert param.metadata["then_specs"][0]["name"] == "prompt"
    assert param.metadata["then_specs"][0]["multiline"] is True


def test_rules_schema_metadata_has_combinators():
    def config(hp):
        hp.rules(when=_make_fields(), then=THEN_PROMPT, name="additions", combinators=["and", "or", "not"])

    schema = explore(config, return_schema=True)
    param = schema.parameters[0]
    assert param.metadata["combinators"] == ["and", "or", "not"]


def test_rules_default_combinators_is_and():
    def config(hp):
        hp.rules(when=_make_fields(), then=THEN_PROMPT, name="additions")

    schema = explore(config, return_schema=True)
    param = schema.parameters[0]
    assert param.metadata["combinators"] == ["and"]


def test_rules_then_requires_field_spec():
    with pytest.raises(TypeError, match="must be a FieldSpec"):

        def config(hp):
            hp.rules(when=_make_fields(), then="text", name="additions")

        explore(config, return_schema=True)


def test_rules_then_requires_name():
    with pytest.raises(ValueError, match="must have a name"):

        def config(hp):
            hp.rules(when=_make_fields(), then=field.text(multiline=True), name="additions")

        explore(config, return_schema=True)


def test_rules_then_multi_field_specs():
    def config(hp):
        hp.rules(
            when=_make_fields(),
            then=[
                field.text(name="prompt", multiline=True),
                field.bool(name="use_citations"),
            ],
            name="additions",
        )

    schema = explore(config, return_schema=True)
    param = schema.parameters[0]
    assert len(param.metadata["then_specs"]) == 2
    assert param.metadata["then_specs"][0]["name"] == "prompt"
    assert param.metadata["then_specs"][0]["type"] == "text"
    assert param.metadata["then_specs"][1]["name"] == "use_citations"
    assert param.metadata["then_specs"][1]["type"] == "bool"


def test_rules_then_list_requires_names():
    with pytest.raises(ValueError, match="must have a name"):

        def config(hp):
            hp.rules(
                when=_make_fields(),
                then=[field.text(multiline=True), field.bool(name="flag")],
                name="additions",
            )

        explore(config, return_schema=True)


def test_rules_then_select_field_spec():
    def config(hp):
        hp.rules(
            when=_make_fields(),
            then=field.select(["high", "medium", "low"], name="priority"),
            name="additions",
        )

    schema = explore(config, return_schema=True)
    param = schema.parameters[0]
    assert param.metadata["then_specs"][0]["type"] == "select"
    assert param.metadata["then_specs"][0]["name"] == "priority"
    assert param.metadata["then_specs"][0]["options"] == ["high", "medium", "low"]


def test_rules_then_float_field_spec():
    def config(hp):
        hp.rules(when=_make_fields(), then=field.float(name="score"), name="scores")

    schema = explore(config, return_schema=True)
    param = schema.parameters[0]
    assert param.metadata["then_specs"][0]["type"] == "float"
    assert param.metadata["then_specs"][0]["name"] == "score"


def test_rules_empty_default():
    def config(hp):
        return hp.rules(when=_make_fields(), then=THEN_PROMPT, name="additions")

    from hypster.core import instantiate

    result = instantiate(config)
    assert result == []


def test_rules_override_with_values():
    default_rules = _sample_rules()

    def config(hp):
        return hp.rules(when=_make_fields(), then=THEN_PROMPT, name="additions", default=default_rules)

    from hypster.core import instantiate

    new_rules = [{"when": {"field": "document_station", "operator": "=", "value": "ER"}, "then": "ER protocol"}]
    result = instantiate(config, values={"additions": new_rules})
    assert len(result) == 1
    assert isinstance(result[0], Rule)
    assert result[0].then == "ER protocol"


def test_rules_selected_value_is_serializable():
    def config(hp):
        hp.rules(when=_make_fields(), then=THEN_PROMPT, name="additions", default=_sample_rules())

    schema = explore(config, return_schema=True)
    param = schema.parameters[0]
    assert isinstance(param.selected_value, list)
    assert isinstance(param.selected_value[0], dict)
    assert "when" in param.selected_value[0]
    assert "then" in param.selected_value[0]


def test_rules_in_experiment_config_pattern():
    """The real panda use case — rules alongside model/prompt config."""

    def experiment_config(hp):
        model = hp.select(["gpt-4", "gpt-4o"], name="model")
        temperature = hp.float(0.7, name="temperature", min=0.0, max=2.0)
        system_prompt = hp.text("You are helpful.", name="system_prompt", multiline=True)
        rules = hp.rules(when=_make_fields(), then=THEN_PROMPT, name="prompt_additions", default=_sample_rules())
        return {"model": model, "temperature": temperature, "system_prompt": system_prompt, "rules": rules}

    schema = explore(experiment_config, return_schema=True)
    kinds = {p.name: p.kind for p in schema.parameters}
    assert kinds == {"model": "select", "temperature": "float", "system_prompt": "text", "prompt_additions": "rules"}

    from hypster.core import instantiate

    result = instantiate(experiment_config)
    assert result["model"] == "gpt-4"
    assert len(result["rules"]) == 1
    assert isinstance(result["rules"][0], Rule)
