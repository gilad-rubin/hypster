from __future__ import annotations

from dataclasses import dataclass
from typing import Any


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
        )

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

        if self.description:
            prop["description"] = self.description

        return prop
