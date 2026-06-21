from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from hypster.rules import Leaf

DEFAULT_OPERATORS: dict[str, list[str]] = {
    "select": ["=", "!=", "in", "not_in"],
    "multi_select": ["in", "not_in"],
    "bool": ["is"],
    "multi_bool": ["in"],
    "int": [">", "<", "=", "!="],
    "multi_int": ["in", "contains"],
    "float": [">", "<", "=", "!="],
    "multi_float": ["in", "contains"],
    "text": ["=", "contains"],
    "multi_text": ["in", "contains"],
}


@dataclass(frozen=True)
class FieldSpec:
    type: str
    name: str | None = None
    description: str | None = None
    options: tuple[Any, ...] | None = None
    multiline: bool = False
    operators: tuple[str, ...] = ()

    def eq(self, value: Any) -> Leaf:
        return Leaf(self._require_name(), "=", value)

    def neq(self, value: Any) -> Leaf:
        return Leaf(self._require_name(), "!=", value)

    def is_in(self, values: list | tuple) -> Leaf:
        return Leaf(self._require_name(), "in", list(values))

    def not_in(self, values: list | tuple) -> Leaf:
        return Leaf(self._require_name(), "not_in", list(values))

    def is_true(self) -> Leaf:
        return Leaf(self._require_name(), "is", True)

    def is_false(self) -> Leaf:
        return Leaf(self._require_name(), "is", False)

    def gt(self, value: Any) -> Leaf:
        return Leaf(self._require_name(), ">", value)

    def lt(self, value: Any) -> Leaf:
        return Leaf(self._require_name(), "<", value)

    def contains(self, value: Any) -> Leaf:
        return Leaf(self._require_name(), "contains", value)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"type": self.type}
        if self.name is not None:
            result["name"] = self.name
        if self.description is not None:
            result["description"] = self.description
        if self.options is not None:
            result["options"] = list(self.options)
        if self.multiline:
            result["multiline"] = True
        result["operators"] = list(self.operators)
        return result

    def _require_name(self) -> str:
        if self.name is None:
            raise ValueError("FieldSpec must have a name to build conditions")
        return self.name


def _make_spec(
    type_name: str, *, name: str | None, description: str | None, options: tuple | None = None, multiline: bool = False
) -> FieldSpec:
    ops = tuple(DEFAULT_OPERATORS.get(type_name, []))
    return FieldSpec(
        type=type_name, name=name, description=description, options=options, multiline=multiline, operators=ops
    )
