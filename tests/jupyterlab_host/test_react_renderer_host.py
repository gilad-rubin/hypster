from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Iterator
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from playwright.sync_api import sync_playwright  # noqa: E402

from hypster import HP  # noqa: E402
from hypster.interactive.session import InteractiveSession  # noqa: E402

REACT = ROOT / "packages" / "react"


def _remote_config(hp: HP) -> dict[str, float]:
    return {"temperature": hp.float(0.5, name="temperature")}


def _host_config(hp: HP) -> dict[str, Any]:
    mode = hp.select(["local", "remote"], name="mode", default="local")
    result: dict[str, Any] = {"mode": mode}
    if mode == "remote":
        result["remote"] = hp.nest(_remote_config, name="remote")
    return result


class _Bridge:
    def __init__(self, bundle: Path) -> None:
        self.bundle = bundle
        self.execution_id = str(uuid4())
        self.sequence = 0
        self.actions: list[dict[str, Any]] = []
        self.session = InteractiveSession(_host_config)
        self.lock = threading.Lock()

    def snapshot(self, *, v2: bool = False) -> dict[str, Any]:
        snapshot = self.session.snapshot
        if v2:
            snapshot["protocol_version"] = 2
        snapshot["bridge_execution_id"] = self.execution_id
        snapshot["bridge_sequence"] = self.sequence
        return snapshot

    def dispatch(self, execution_id: str, action: dict[str, Any]) -> dict[str, Any]:
        if execution_id != self.execution_id:
            raise PermissionError("stale host execution marker")
        with self.lock:
            self.actions.append(action)
            self.session.dispatch(action)
            self.sequence += 1
            return self.snapshot()


def _handler(bridge: _Bridge) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path.startswith("/snapshot"):
                self._json(bridge.snapshot(v2="v2=1" in self.path))
                return
            if self.path == "/app.js":
                content = bridge.bundle.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/javascript")
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)
                return
            if self.path.startswith("/"):
                body = (
                    b'<!doctype html><html><body><main id="root"></main>'
                    b'<script type="module" src="/app.js"></script></body></html>'
                )
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            self.send_error(404)

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/action":
                self.send_error(404)
                return
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length))
            try:
                self._json(bridge.dispatch(payload["execution_id"], payload["action"]))
            except PermissionError as error:
                self._json({"error": str(error)}, status=409)

        def _json(self, value: Any, *, status: int = 200) -> None:
            body = json.dumps(value).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, _format: str, *args: Any) -> None:
            del args

    return Handler


@contextmanager
def _server(bridge: _Bridge) -> Iterator[str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _handler(bridge))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        assert not thread.is_alive(), "React host HTTP server did not stop"


def test_production_react_renderer_round_trips_through_python() -> None:
    artifact_dir = Path(os.environ.get("HYPSTER_REACT_HOST_ARTIFACT_DIR", ROOT / "react-host-evidence"))
    artifact_dir.mkdir(parents=True, exist_ok=True)

    subprocess.run(["npm", "run", "build"], cwd=REACT, check=True)
    chromium_version = "not-started"
    with TemporaryDirectory(prefix="hypster-react-host-") as temporary:
        bundle = Path(temporary) / "app.js"
        subprocess.run(
            ["npx", "esbuild", "test/host-entry.tsx", "--bundle", "--format=esm", f"--outfile={bundle}"],
            cwd=REACT,
            check=True,
        )
        bridge = _Bridge(bundle)
        with _server(bridge) as url:
            stale = Request(
                f"{url}/action",
                data=json.dumps(
                    {"execution_id": "old-execution", "action": {"protocol_version": 1, "type": "reset"}}
                ).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                urlopen(stale)
            except HTTPError as error:
                assert error.code == 409
            else:
                raise AssertionError("stale execution marker was accepted")

            with sync_playwright() as playwright:
                browser = playwright.chromium.launch()
                chromium_version = browser.version
                page = browser.new_page()
                page.goto(url)
                page.locator("#root[data-renderer='@hypster/react']").wait_for()
                assert page.locator(".hypster-renderer").count() == 1
                assert page.locator(".hypster-widget").count() == 0
                page.get_by_label("Mode").select_option(label="remote")
                page.get_by_label("Temperature").fill("1.25")
                page.wait_for_function("window.__hypsterHost.sequence === 2")
                assert bridge.session.params == {"mode": "remote", "remote.temperature": 1.25}
                assert page.locator("#root").get_attribute("data-execution-id") == bridge.execution_id
                page.screenshot(path=artifact_dir / "react-round-trip.png")

                v2 = browser.new_page()
                v2.goto(f"{url}/?v2=1")
                assert "version mismatch" in v2.get_by_role("alert").inner_text()
                assert v2.locator("select,input,textarea,button").count() == 0
                assert len(bridge.actions) == 2
                v2.screenshot(path=artifact_dir / "react-v2-mismatch.png")
                browser.close()

        evidence = {
            "renderer": "@hypster/react",
            "versions": {
                "python": sys.version.split()[0],
                "node": subprocess.check_output(["node", "--version"], text=True).strip(),
                "npm": subprocess.check_output(["npm", "--version"], text=True).strip(),
                "playwright": version("playwright"),
                "chromium": chromium_version,
            },
            "execution_id": bridge.execution_id,
            "sequence": bridge.sequence,
            "actions": bridge.actions,
            "python_oracle": bridge.session.params,
            "process_cleanup": "http server stopped",
        }
        (artifact_dir / "evidence.json").write_text(json.dumps(evidence, indent=2, sort_keys=True))
