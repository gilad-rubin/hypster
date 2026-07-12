from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from hypster import HP
from hypster.interactive.session import INTERACTIVE_PROTOCOL_VERSION, InteractiveSession

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "protocol-v1.json"


def branch_config(hp: HP) -> dict[str, Any]:
    mode = hp.select(["local", "remote"], name="mode", default="local")
    if mode == "remote":
        temperature = hp.float(0.25, name="temperature", min=0.0, max=2.0)
        return {"mode": mode, "temperature": temperature}
    return {"mode": mode}


def protocol_fixture() -> dict[str, Any]:
    session: InteractiveSession[dict[str, Any]] = InteractiveSession(branch_config)
    actions = [
        {
            "protocol_version": INTERACTIVE_PROTOCOL_VERSION,
            "type": "set_value",
            "path": "mode",
            "value": "remote",
        },
        {
            "protocol_version": INTERACTIVE_PROTOCOL_VERSION,
            "type": "set_value",
            "path": "temperature",
            "value": 1.25,
        },
    ]
    initial_snapshot = session.snapshot
    branch_snapshot = session.dispatch(actions[0])
    final_snapshot = session.dispatch(actions[1])
    return {
        "protocol_version": INTERACTIVE_PROTOCOL_VERSION,
        "initial_snapshot": initial_snapshot,
        "actions": actions,
        "branch_snapshot": branch_snapshot,
        "final_snapshot": final_snapshot,
        "final_params": session.params,
    }


def serialized_fixture() -> str:
    return json.dumps(protocol_fixture(), indent=2, sort_keys=True) + "\n"


def serve_protocol() -> None:
    session: InteractiveSession[dict[str, Any]] = InteractiveSession(branch_config)
    print(json.dumps(session.snapshot, sort_keys=True), flush=True)
    for line in sys.stdin:
        action = json.loads(line)
        print(json.dumps(session.dispatch(action), sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--serve", action="store_true")
    args = parser.parse_args()

    if args.serve:
        serve_protocol()
        return

    expected = serialized_fixture()

    if args.check:
        if not FIXTURE_PATH.exists() or FIXTURE_PATH.read_text() != expected:
            raise SystemExit(f"{FIXTURE_PATH} is stale; regenerate it with {Path(__file__).name}")
        return

    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE_PATH.write_text(expected)


if __name__ == "__main__":
    main()
