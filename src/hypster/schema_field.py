from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

_ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")


def is_empty(value: Any) -> bool:
    """An explicit empty value: ``None``, a blank string, or an empty list."""
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    return isinstance(value, (list, tuple)) and not value


@dataclass(frozen=True)
class SchemaField:
    """A single extraction field definition for structured LLM output schemas.

    Shared spec with literature_agent. Mirrors the TypeScript SchemaField type.
    """

    key: str
    value_type: str  # "text" | "enum" | "number" | "date"
    description: str = ""
    label: str = ""
    multi_valued: bool = False
    possible_values: list[str] | None = None
    unit: str | None = None
    # Requiredness lives ON the field (PRD 0027 in superposition): a consumer
    # that gates on required fields reads this flag, so renaming a field can
    # never orphan its requiredness in a parallel key list. NOT emitted by
    # ``to_json_schema`` — JSON Schema expresses required as an array on the
    # PARENT object, so schema assemblers read the flag and build that array.
    required: bool = False
    # Whether an explicit empty value (``is_empty``) is one of this field's
    # values. Apart from ``required``: required says a value must be given;
    # may_be_empty says whether "nothing" is a value the field may hold.
    may_be_empty: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.may_be_empty, bool):
            raise ValueError(f"{self.key}: may_be_empty must be true or false, got {self.may_be_empty!r}")

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"key": self.key, "value_type": self.value_type}
        if self.description:
            result["description"] = self.description
        if self.label:
            result["label"] = self.label
        if self.multi_valued:
            result["multi_valued"] = True
        if self.possible_values is not None:
            result["possible_values"] = list(self.possible_values)
        if self.unit is not None:
            result["unit"] = self.unit
        if self.required:
            result["required"] = True
        if self.may_be_empty:
            result["may_be_empty"] = True
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SchemaField:
        return cls(
            key=data["key"],
            value_type=data["value_type"],
            description=data.get("description", ""),
            label=data.get("label", ""),
            multi_valued=data.get("multi_valued", False),
            possible_values=data.get("possible_values"),
            unit=data.get("unit"),
            required=data.get("required", False),
            may_be_empty=data.get("may_be_empty", False),
        )

    def check(self, value: Any) -> str | None:
        """Why ``value`` is not a value of this field, or ``None`` when it is.

        An explicit empty value (``is_empty``) is valid only where the field
        may be empty. Any other value must fit ``value_type`` (a list of them
        when ``multi_valued``) and, for an enum with ``possible_values``, be
        one of them. Whether a value must be given at all is the caller's.
        """
        if is_empty(value):
            return None if self.may_be_empty else f"{self.key}: may not be empty"
        if not self.multi_valued:
            return self._check_one(value)
        if not isinstance(value, (list, tuple)):
            return f"{self.key}: expected a list, got {value!r}"
        return next((problem for item in value if (problem := self._check_one(item))), None)

    def _check_one(self, value: Any) -> str | None:
        if self.value_type == "number":
            fits = isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
            return None if fits else f"{self.key}: expected a number, got {value!r}"
        if self.value_type == "date":
            return None if _is_date(value) else f"{self.key}: expected a date (YYYY-MM-DD), got {value!r}"
        if not isinstance(value, str) or not value.strip():
            return f"{self.key}: expected text, got {value!r}"
        if self.value_type == "enum" and self.possible_values is not None and value not in self.possible_values:
            return f"{self.key}: {value!r} is not one of {', '.join(self.possible_values)}"
        return None

    def to_json_schema(self) -> dict[str, Any]:
        """Convert to a JSON Schema property definition for structured LLM output."""
        if self.value_type == "enum":
            base: dict[str, Any] = {"type": "string"}
            if self.possible_values:
                base["enum"] = list(self.possible_values)
        elif self.value_type == "number":
            base = {"type": "number"}
        elif self.value_type == "date":
            base = {"type": "string", "format": "date"}
        else:
            base = {"type": "string"}

        if self.multi_valued:
            prop: dict[str, Any] = {"type": "array", "items": base}
        else:
            prop = base

        if self.may_be_empty:
            prop = {"anyOf": [prop, {"type": "null"}]}

        if self.description:
            prop["description"] = self.description

        return prop


def _is_date(value: Any) -> bool:
    if isinstance(value, date):
        return not isinstance(value, datetime)
    if not isinstance(value, str) or not _ISO_DATE.fullmatch(value):
        return False
    try:
        date.fromisoformat(value)
    except ValueError:
        return False
    return True


def coerce_schema_fields(raw: list) -> list[SchemaField]:
    """Coerce a list of SchemaField objects or dicts into SchemaField objects."""
    result = []
    for i, item in enumerate(raw):
        if isinstance(item, SchemaField):
            result.append(item)
        elif isinstance(item, dict):
            result.append(SchemaField.from_dict(item))
        else:
            raise ValueError(f"schema[{i}]: expected a SchemaField or dict, got {type(item).__name__}")
    return result


def schema_fields_to_jsonable(fields: list) -> list:
    return [f.to_dict() if isinstance(f, SchemaField) else f for f in fields]
