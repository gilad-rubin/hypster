from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generic, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class Leaf:
    field: str
    operator: str
    value: Any

    def __and__(self, other: Leaf | Group) -> Group:
        return And(self, other)

    def __or__(self, other: Leaf | Group) -> Group:
        return Or(self, other)

    def __invert__(self) -> Group:
        return Not(self)

    def then(self, payload: Any) -> Rule:
        return Rule(when=self, then=payload)

    def to_dict(self) -> dict[str, Any]:
        return {"field": self.field, "operator": self.operator, "value": self.value}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Leaf:
        return cls(field=data["field"], operator=data["operator"], value=data["value"])


@dataclass(frozen=True)
class Group:
    combinator: str
    conditions: tuple[Leaf | Group, ...]

    def __and__(self, other: Leaf | Group) -> Group:
        return And(self, other)

    def __or__(self, other: Leaf | Group) -> Group:
        return Or(self, other)

    def __invert__(self) -> Group:
        return Not(self)

    def then(self, payload: Any) -> Rule:
        return Rule(when=self, then=payload)

    def to_dict(self) -> dict[str, Any]:
        return {
            "combinator": self.combinator,
            "conditions": [c.to_dict() for c in self.conditions],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Group:
        if "combinator" not in data or not isinstance(data["combinator"], str):
            raise ValueError("group node must include a string 'combinator'")
        if "conditions" not in data:
            raise ValueError("group node is missing a 'conditions' key")
        if not isinstance(data["conditions"], list):
            raise ValueError("group node 'conditions' must be a list")
        return cls(
            combinator=data["combinator"],
            conditions=tuple(_node_from_dict(c) for c in data["conditions"]),
        )


Condition = Leaf | Group


def And(*conditions: Condition) -> Group:
    return Group(combinator="and", conditions=conditions)


def Or(*conditions: Condition) -> Group:
    return Group(combinator="or", conditions=conditions)


def Not(condition: Condition) -> Group:
    if not isinstance(condition, (Leaf, Group)):
        raise TypeError(f"Not() takes a single condition, got {type(condition).__name__}")
    return Group(combinator="not", conditions=(condition,))


@dataclass(frozen=True)
class Rule(Generic[T]):
    when: Condition
    then: T
    name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"when": self.when.to_dict(), "then": self.then}
        if self.name is not None:
            result["name"] = self.name
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Rule:
        return cls(
            when=_node_from_dict(data["when"]),
            then=data["then"],
            name=data.get("name"),
        )


def _node_from_dict(data: dict[str, Any]) -> Condition:
    if "combinator" in data:
        return Group.from_dict(data)
    return Leaf.from_dict(data)


def coerce_rules(raw: list) -> list[Rule]:
    """Coerce a list of Rule objects or rule dicts into Rule objects."""
    result = []
    for i, item in enumerate(raw):
        if isinstance(item, Rule):
            result.append(item)
        elif isinstance(item, dict):
            result.append(Rule.from_dict(item))
        else:
            raise ValueError(f"rules[{i}]: expected a Rule or dict, got {type(item).__name__}")
    return result


def rules_to_jsonable(rules: list) -> list:
    return [r.to_dict() if isinstance(r, Rule) else r for r in rules]
