from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import anywidget
import traitlets

from .session import InteractiveResult


class HypsterInteractWidget(anywidget.AnyWidget):
    _esm = Path(__file__).parent / "interact.js"
    _css = Path(__file__).parent / "interact.css"

    snapshot = traitlets.Dict().tag(sync=True)
    action = traitlets.Dict(default_value={}).tag(sync=True)

    def __init__(self, result: InteractiveResult[Any]):
        super().__init__(snapshot=result.snapshot)
        self._result = result
        self.observe(self._handle_action, names="action")

    def _handle_action(self, change: Dict[str, Any]) -> None:
        action = change.get("new")
        if not action:
            return
        self._result.dispatch(action)
