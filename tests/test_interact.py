from __future__ import annotations

from typing import Any, Dict

import pytest

from hypster import HP, instantiate_with_params, interact


def test_interact_requires_viz_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    def config(hp: HP) -> Dict[str, int]:
        return {"count": hp.int(1, name="count", min=1, max=5)}

    monkeypatch.setattr("hypster.interactive.session.importlib.util.find_spec", lambda name: None)

    with pytest.raises(RuntimeError, match=r"hypster\[viz\]"):
        interact(config)


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


def test_interact_separates_same_path_values_by_branch_context() -> None:
    def config(hp: HP) -> Dict[str, Any]:
        mode = hp.select(["a", "b"], name="mode", default="a")
        if mode == "a":
            n = hp.int(0, name="n", min=0, max=10)
        else:
            n = hp.int(0, name="n", min=0, max=10)
        return {"mode": mode, "n": n}

    result = interact(config)

    result.dispatch({"type": "set_value", "path": "n", "value": 7})
    result.dispatch({"type": "set_value", "path": "mode", "value": "b"})

    assert result.params == {"mode": "b", "n": 0}

    result.dispatch({"type": "set_value", "path": "n", "value": 3})
    result.dispatch({"type": "set_value", "path": "mode", "value": "a"})

    assert result.params == {"mode": "a", "n": 7}

    result.dispatch({"type": "set_value", "path": "mode", "value": "b"})

    assert result.params == {"mode": "b", "n": 3}


def test_interact_updates_same_path_numeric_bounds_by_branch_context() -> None:
    def config(hp: HP) -> Dict[str, int]:
        a = hp.select([2, 3], name="a", default=3)
        if a == 2:
            n = hp.int(0, name="n", min=0, max=3)
        else:
            n = hp.int(0, name="n", min=0, max=10)
        return {"a": a, "n": n}

    result = interact(config)

    result.dispatch({"type": "set_value", "path": "n", "value": 8})
    snapshot = result.dispatch({"type": "set_value", "path": "a", "value": 2})
    n_parameter = snapshot["schema"]["parameters"][1]

    assert snapshot["status"] == "applied"
    assert n_parameter["minimum"] == 0
    assert n_parameter["maximum"] == 3
    assert result.params == {"a": 2, "n": 0}

    result.dispatch({"type": "set_value", "path": "n", "value": 2})
    result.dispatch({"type": "set_value", "path": "a", "value": 3})

    assert result.params == {"a": 3, "n": 8}

    result.dispatch({"type": "set_value", "path": "a", "value": 2})

    assert result.params == {"a": 2, "n": 2}


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


def test_interact_manual_apply_preserves_sibling_drafts_after_exploration_error() -> None:
    def config(hp: HP) -> Dict[str, int]:
        return {
            "count": hp.int(1, name="count", min=1, max=5),
            "width": hp.int(2, name="width", min=1, max=5),
        }

    result = interact(config, auto_apply=False)

    result.dispatch({"type": "set_value", "path": "width", "value": 4})
    snapshot = result.dispatch({"type": "set_value", "path": "count", "value": 9})

    assert snapshot["status"] == "draft_error"
    assert snapshot["draft_values"] == {"count": 9, "width": 4}
    assert result.value == {"count": 1, "width": 2}


def test_interact_immediate_apply_exposes_errors_until_fixed() -> None:
    def config(hp: HP) -> Dict[str, int]:
        return {"count": hp.int(1, name="count", min=1, max=5)}

    result = interact(config)

    snapshot = result.dispatch({"type": "set_value", "path": "count", "value": 9})

    assert snapshot["status"] == "error"
    assert snapshot["error"]["kind"] == "exploration"
    assert "exceeds maximum bound" in snapshot["error"]["message"]
    assert snapshot["draft_values"] == {"count": 9}
    assert snapshot["applied_values"] == {"count": 1}
    assert snapshot["selected_params"] is None
    with pytest.raises(RuntimeError, match="exceeds maximum bound"):
        result.value
    with pytest.raises(RuntimeError, match="exceeds maximum bound"):
        result.params

    snapshot = result.dispatch({"type": "set_value", "path": "count", "value": 4})

    assert snapshot["status"] == "applied"
    assert snapshot["error"] is None
    assert result.value == {"count": 4}
    assert result.params == {"count": 4}


def test_interact_reset_reports_errors_without_stale_value() -> None:
    state = {"fail": False}

    def config(hp: HP) -> Dict[str, int]:
        count = hp.int(1, name="count", min=1, max=5)
        if state["fail"]:
            raise RuntimeError("reset failed")
        return {"count": count}

    result = interact(config)
    result.dispatch({"type": "set_value", "path": "count", "value": 3})

    state["fail"] = True
    snapshot = result.dispatch({"type": "reset"})

    assert snapshot["status"] == "error"
    assert snapshot["error"] == {"kind": "exploration", "message": "reset failed"}
    assert snapshot["selected_params"] is None
    with pytest.raises(RuntimeError, match="reset failed"):
        result.value


def test_interact_branch_memory_rejects_incompatible_multi_value_history() -> None:
    def config(hp: HP) -> Dict[str, Any]:
        mode = hp.select(["a", "b"], name="mode", default="a")
        if mode == "a":
            values = hp.multi_int([1], name="values", min=0, max=5)
            return {"mode": mode, "values": values}
        return {"mode": mode, "enabled": hp.bool(True, name="enabled")}

    result = interact(config)

    snapshot = result.dispatch({"type": "set_value", "path": "values", "value": [99]})

    assert snapshot["status"] == "error"

    result.dispatch({"type": "set_value", "path": "mode", "value": "b"})
    snapshot = result.dispatch({"type": "set_value", "path": "mode", "value": "a"})

    assert snapshot["status"] == "applied"
    assert result.value == {"mode": "a", "values": [1]}
    assert result.params == {"mode": "a", "values": [1]}


def test_interact_widget_view_dispatches_actions_to_same_session() -> None:
    def config(hp: HP) -> Dict[str, int]:
        return {"count": hp.int(1, name="count", min=1, max=5)}

    result = interact(config)
    widget = result.interact()

    widget.action = {"id": "test-action", "type": "set_value", "path": "count", "value": 2}

    assert widget.snapshot["selected_params"] == {"count": 2}
    assert result.value == {"count": 2}


def test_interact_widget_decodes_numeric_transport_values() -> None:
    def child(hp: HP) -> Dict[str, float]:
        return {"temperature": hp.float(0.5, name="temperature", min=0.0, max=2.0)}

    def config(hp: HP) -> Dict[str, Any]:
        return {"child": hp.nest(child, name="child")}

    result = interact(config)
    widget = result.interact()

    widget.action = {
        "id": "test-action",
        "type": "set_value",
        "path": "child.temperature",
        "encoded_value": {"kind": "float", "value": "1"},
    }

    assert result.params == {"child.temperature": 1.0}
    assert isinstance(result.params["child.temperature"], float)


def test_interact_widget_decodes_empty_multi_value_transport_as_empty_list() -> None:
    def config(hp: HP) -> Dict[str, list[int]]:
        return {"values": hp.multi_int([1, 2], name="values", min=0, max=5)}

    result = interact(config)
    widget = result.interact()

    widget.action = {
        "id": "test-action",
        "type": "set_value",
        "path": "values",
        "encoded_value": {"kind": "multi_int", "value": ""},
    }

    assert result.params == {"values": []}


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


def test_interact_widget_display_includes_vscode_background_shim(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[Any, Dict[str, Any]]] = []

    def fake_display(obj: Any = None, **kwargs: Any) -> None:
        calls.append((obj, kwargs))

    def config(hp: HP) -> Dict[str, int]:
        return {"count": hp.int(1, name="count", min=1, max=5)}

    monkeypatch.setattr("IPython.display.display", fake_display)

    widget = interact(config).interact()
    widget._ipython_display_()

    assert len(calls) == 2
    assert "cell-output-ipywidget-background" in calls[0][0].data
    assert "vscode-cell-output" in calls[0][0].data
    assert calls[1][1]["raw"] is True
    assert "application/vnd.jupyter.widget-view+json" in calls[1][0]
