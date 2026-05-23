from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import anywidget
import traitlets

from .session import InteractiveResult


class HypsterInteractWidget(anywidget.AnyWidget):
    _esm = Path(__file__).parent / "interact.js"
    _css = Path(__file__).parent / "interact.css"

    snapshot = traitlets.Dict().tag(sync=True)
    action = traitlets.Dict().tag(sync=True)

    def __init__(self, result: InteractiveResult[Any]):
        super().__init__(snapshot=result.snapshot)
        self._result = result
        self.observe(self._handle_action, names="action")

    def _handle_action(self, change: Dict[str, Any]) -> None:
        action = change.get("new")
        if not action:
            return
        self._result.dispatch(_decode_action(action))


def _decode_action(action: Dict[str, Any]) -> Dict[str, Any]:
    if action.get("type") != "set_value" or "encoded_value" not in action:
        return action

    decoded = dict(action)
    decoded["value"] = _decode_value(decoded.pop("encoded_value"))
    return decoded


def _decode_value(encoded_value: Any) -> Any:
    if not isinstance(encoded_value, dict):
        return encoded_value

    kind = encoded_value.get("kind")
    raw_value = encoded_value.get("value")
    if raw_value is None:
        return None

    if kind == "int":
        if raw_value == "":
            return None
        try:
            return int(raw_value)
        except (TypeError, ValueError):
            return raw_value

    if kind == "float":
        if raw_value == "":
            return None
        try:
            return float(raw_value)
        except (TypeError, ValueError):
            return raw_value

    if isinstance(kind, str) and kind.startswith("multi_"):
        if raw_value == "":
            return []
        try:
            value = json.loads(raw_value)
        except (TypeError, json.JSONDecodeError):
            return raw_value
        return value if isinstance(value, list) else raw_value

    return raw_value
