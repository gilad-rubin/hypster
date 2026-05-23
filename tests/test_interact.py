from __future__ import annotations

from typing import Any, Dict

import pytest

from hypster import HP, instantiate_with_params, interact


def test_interact_returns_live_result_matching_instantiate_with_params() -> None:
    def openai(hp: HP) -> Dict[str, Any]:
        return {
            "model": hp.select(["gpt-4o-mini", "gpt-4.1"], name="model", default="gpt-4o-mini"),
            "temperature": hp.float(0.2, name="temperature", min=0.0, max=2.0),
        }

    def config(hp: HP) -> Dict[str, Any]:
        provider = hp.select(["gemini", "openai"], name="provider", default="gemini")
        if provider == "openai":
            llm = hp.nest(openai, name="openai")
        else:
            llm = {"model": hp.select(["flash-lite", "pro"], name="model", default="flash-lite")}
        return {"provider": provider, "llm": llm}

    result = interact(config, values={"provider": "openai", "openai.temperature": 0.7})
    expected = instantiate_with_params(config, values={"provider": "openai", "openai.temperature": 0.7})

    assert result.value == expected.value
    assert result.params == expected.params

    params = result.params
    params["provider"] = "gemini"

    assert result.params == expected.params


def test_interact_action_updates_value_and_params() -> None:
    def openai(hp: HP) -> Dict[str, Any]:
        return {
            "model": hp.select(["gpt-4o-mini", "gpt-4.1"], name="model", default="gpt-4o-mini"),
            "temperature": hp.float(0.2, name="temperature", min=0.0, max=2.0),
        }

    def config(hp: HP) -> Dict[str, Any]:
        provider = hp.select(["gemini", "openai"], name="provider", default="gemini")
        if provider == "openai":
            llm = hp.nest(openai, name="openai")
        else:
            llm = {"model": hp.select(["flash-lite", "pro"], name="model", default="flash-lite")}
        return {"provider": provider, "llm": llm}

    result = interact(config)

    snapshot = result.dispatch({"type": "set_value", "path": "provider", "value": "openai"})

    assert result.value == {
        "provider": "openai",
        "llm": {"model": "gpt-4o-mini", "temperature": 0.2},
    }
    assert result.params == {
        "provider": "openai",
        "openai.model": "gpt-4o-mini",
        "openai.temperature": 0.2,
    }
    assert snapshot["draft_values"] == result.params


def test_interact_remembers_latest_compatible_branch_choice() -> None:
    def config(hp: HP) -> Dict[str, str]:
        provider = hp.select(["openai", "gemini"], name="provider", default="openai")
        if provider == "gemini":
            model = hp.select(["flash-lite", "pro"], name="model", default="flash-lite")
        else:
            model = hp.select(["gpt-4o-mini", "gpt-4.1"], name="model", default="gpt-4o-mini")
        return {"provider": provider, "model": model}

    result = interact(config)

    result.dispatch({"type": "set_value", "path": "provider", "value": "gemini"})
    result.dispatch({"type": "set_value", "path": "model", "value": "pro"})
    result.dispatch({"type": "set_value", "path": "provider", "value": "openai"})
    result.dispatch({"type": "set_value", "path": "provider", "value": "gemini"})

    assert result.value == {"provider": "gemini", "model": "pro"}
    assert result.params == {"provider": "gemini", "model": "pro"}


def test_interact_reset_restores_baseline_and_clears_later_branch_memory() -> None:
    def config(hp: HP) -> Dict[str, str]:
        provider = hp.select(["openai", "gemini"], name="provider", default="openai")
        if provider == "gemini":
            model = hp.select(["flash-lite", "pro"], name="model", default="flash-lite")
        else:
            model = hp.select(["gpt-4o-mini", "gpt-4.1"], name="model", default="gpt-4o-mini")
        return {"provider": provider, "model": model}

    result = interact(config)

    result.dispatch({"type": "set_value", "path": "provider", "value": "gemini"})
    result.dispatch({"type": "set_value", "path": "model", "value": "pro"})
    result.dispatch({"type": "set_value", "path": "provider", "value": "openai"})

    snapshot = result.dispatch({"type": "reset"})

    assert result.value == {"provider": "openai", "model": "gpt-4o-mini"}
    assert snapshot["selected_params"] == {"provider": "openai", "model": "gpt-4o-mini"}

    result.dispatch({"type": "set_value", "path": "provider", "value": "gemini"})

    assert result.value == {"provider": "gemini", "model": "flash-lite"}


def test_interact_manual_apply_keeps_value_on_last_applied_state_until_apply() -> None:
    def config(hp: HP) -> Dict[str, int]:
        return {"count": hp.int(1, name="count", min=1, max=5)}

    result = interact(config, auto_apply=False)

    snapshot = result.dispatch({"type": "set_value", "path": "count", "value": 3})

    assert result.value == {"count": 1}
    assert result.params == {"count": 1}
    assert snapshot["draft_values"] == {"count": 3}
    assert snapshot["applied_values"] == {"count": 1}
    assert snapshot["status"] == "pending"

    snapshot = result.dispatch({"type": "apply"})

    assert result.value == {"count": 3}
    assert result.params == {"count": 3}
    assert snapshot["status"] == "applied"


def test_interact_immediate_apply_exposes_errors_until_fixed() -> None:
    def config(hp: HP) -> Dict[str, int]:
        return {"count": hp.int(1, name="count", min=1, max=5)}

    result = interact(config)

    snapshot = result.dispatch({"type": "set_value", "path": "count", "value": 9})

    assert snapshot["status"] == "error"
    assert snapshot["error"]["kind"] == "exploration"
    assert "exceeds maximum bound" in snapshot["error"]["message"]
    with pytest.raises(RuntimeError, match="exceeds maximum bound"):
        result.value
    with pytest.raises(RuntimeError, match="exceeds maximum bound"):
        result.params

    snapshot = result.dispatch({"type": "set_value", "path": "count", "value": 4})

    assert snapshot["status"] == "applied"
    assert snapshot["error"] is None
    assert result.value == {"count": 4}
    assert result.params == {"count": 4}


def test_interact_widget_view_dispatches_actions_to_same_session() -> None:
    def config(hp: HP) -> Dict[str, int]:
        return {"count": hp.int(1, name="count", min=1, max=5)}

    result = interact(config)
    widget = result.interact()

    widget.action = {"id": "test-action", "type": "set_value", "path": "count", "value": 2}

    assert widget.snapshot["selected_params"] == {"count": 2}
    assert result.value == {"count": 2}


def test_interact_multiple_widget_views_share_session_updates() -> None:
    def config(hp: HP) -> Dict[str, int]:
        return {"count": hp.int(1, name="count", min=1, max=5)}

    result = interact(config)
    first = result.interact()
    second = result.interact()

    first.action = {"id": "test-action", "type": "set_value", "path": "count", "value": 4}

    assert first.snapshot["selected_params"] == {"count": 4}
    assert second.snapshot["selected_params"] == {"count": 4}
    assert result.value == {"count": 4}
