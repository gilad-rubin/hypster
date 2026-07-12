from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from playwright.sync_api import Page, sync_playwright
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

HARNESS_ROOT = Path(__file__).resolve().parent
NOTEBOOK = HARNESS_ROOT / "branch_round_trip.ipynb"
KERNEL_LOCK = HARNESS_ROOT / "kernel-requirements.lock"
SERVER_TIMEOUT_SECONDS = 60
SERVER_PORT_RETRIES = 5
COMM_TIMEOUT_MS = 30_000


def required_path(name: str) -> Path:
    raw_value = os.environ.get(name)
    assert raw_value, f"{name} must name an explicit absolute path"
    path = Path(raw_value)
    assert path.is_absolute(), f"{name} must be absolute: {path}"
    return path


def run_checked(command: list[str], *, cwd: Path) -> str:
    completed = subprocess.run(command, cwd=cwd, check=False, text=True, capture_output=True)
    assert completed.returncode == 0, (
        f"command failed ({completed.returncode}): {command!r}\n"
        f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )
    return completed.stdout.strip()


def available_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def wait_for_server(url: str, token: str, process: subprocess.Popen[str]) -> None:
    deadline = time.monotonic() + SERVER_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        assert process.poll() is None, f"JupyterLab exited before readiness with code {process.returncode}"
        try:
            with urllib.request.urlopen(f"{url}/api/status?token={token}", timeout=1) as response:
                if response.status == 200:
                    return
        except OSError:
            time.sleep(0.25)
    raise AssertionError(
        f"JupyterLab did not become ready within {SERVER_TIMEOUT_SECONDS}s; process return code: {process.poll()}"
    )


def wait_for_server_info(runtime_dir: Path, token: str, process: subprocess.Popen[str]) -> str:
    deadline = time.monotonic() + SERVER_TIMEOUT_SECONDS
    info_file = runtime_dir / f"jpserver-{process.pid}.json"
    while time.monotonic() < deadline:
        assert process.poll() is None, f"JupyterLab exited before publishing server info: {process.returncode}"
        try:
            info = json.loads(info_file.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            time.sleep(0.25)
            continue

        assert info["pid"] == process.pid, f"server info PID {info['pid']} does not match launched PID {process.pid}"
        assert info["token"] == token, "server info token does not match the configured host-test token"
        url = str(info["url"]).rstrip("/")
        parsed_url = urlsplit(url)
        assert parsed_url.scheme in {"http", "https"}, f"server info published an invalid URL: {url}"
        assert parsed_url.hostname in {"127.0.0.1", "localhost"}, f"server info published a non-local URL: {url}"
        assert parsed_url.port == info["port"], f"server info URL and port disagree: {info}"
        return url

    raise AssertionError(f"JupyterLab did not publish {info_file.name} within {SERVER_TIMEOUT_SECONDS}s")


def wait_for_kernel(url: str, token: str, process: subprocess.Popen[str]) -> None:
    deadline = time.monotonic() + SERVER_TIMEOUT_SECONDS
    request = urllib.request.Request(
        f"{url}/api/sessions",
        headers={"Authorization": f"token {token}"},
    )
    while time.monotonic() < deadline:
        assert process.poll() is None, f"JupyterLab exited while the kernel was connecting: {process.returncode}"
        try:
            with urllib.request.urlopen(request, timeout=1) as response:
                sessions = json.load(response)
            for session in sessions:
                if session["path"] == NOTEBOOK.name and session["kernel"]["execution_state"] == "idle":
                    return
        except OSError:
            pass
        time.sleep(0.25)
    raise AssertionError(f"notebook kernel did not become idle within {SERVER_TIMEOUT_SECONDS}s")


def execute_cell(page: Page, index: int, marker: str) -> str:
    cell = page.locator(".jp-Notebook-cell").nth(index)
    cell.locator(".jp-Cell-inputArea").click()
    page.keyboard.press("Control+Enter")
    output = cell.locator(".jp-OutputArea")
    try:
        output.get_by_text(re.compile(re.escape(marker))).wait_for(state="visible", timeout=COMM_TIMEOUT_MS)
    except PlaywrightTimeoutError as error:
        raise AssertionError(
            f"cell {index} did not emit {marker!r}; kernel output was:\n{output.inner_text()}"
        ) from error
    error_output = output.locator("[data-mime-type='application/vnd.jupyter.error']")
    assert error_output.count() == 0, f"cell {index} produced a kernel error: {output.inner_text()}"
    return output.inner_text()


def assert_widget_has_no_error(page: Page) -> None:
    errors = page.locator(".hypster-status-error, .hypster-status-draft_error")
    assert errors.count() == 0, f"widget surfaced an error: {errors.all_inner_texts()}"


def notebook_model(url: str, token: str) -> dict[str, Any]:
    request = urllib.request.Request(
        f"{url}/api/contents/{NOTEBOOK.name}?content=1",
        headers={"Authorization": f"token {token}"},
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.load(response)["content"]


def assert_saved_cells_succeeded(model: dict[str, Any]) -> None:
    cells = model["cells"]
    assert len(cells) == 2, f"expected exactly two cells, got {len(cells)}"
    for index, cell in enumerate(cells):
        assert cell["execution_count"] is not None, f"cell {index} was not executed"
        errors = [output for output in cell["outputs"] if output["output_type"] == "error"]
        assert not errors, f"cell {index} saved kernel errors: {errors}"


def wait_for_saved_notebook(url: str, token: str) -> dict[str, Any]:
    deadline = time.monotonic() + 10
    last_model: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        last_model = notebook_model(url, token)
        cells = last_model["cells"]
        if len(cells) == 2 and all(cell["execution_count"] is not None for cell in cells):
            return last_model
        time.sleep(0.1)
    raise AssertionError(f"executed notebook was not saved within 10s: {last_model}")


def terminate_process_group(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        process.wait()
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        process.wait(timeout=10)
        return
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=5)


def test_real_jupyterlab_round_trip() -> None:
    wheel = required_path("HYPSTER_HOST_WHEEL").resolve(strict=True)
    artifact_dir = required_path("HYPSTER_HOST_ARTIFACT_DIR")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    assert wheel.suffix == ".whl", f"expected a built wheel, got {wheel}"

    with tempfile.TemporaryDirectory(prefix="hypster-jupyterlab-host-") as temp_raw:
        temp = Path(temp_raw)
        notebook_root = temp / "notebooks"
        notebook_root.mkdir()
        shutil.copy2(NOTEBOOK, notebook_root / NOTEBOOK.name)

        kernel_env = temp / "kernel-env"
        run_checked(["uv", "venv", "--python", "3.13", str(kernel_env)], cwd=temp)
        kernel_python = kernel_env / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        install_output = run_checked(
            [
                "uv",
                "pip",
                "install",
                "--python",
                str(kernel_python),
                "--requirement",
                str(KERNEL_LOCK),
                f"{wheel}[viz]",
            ],
            cwd=temp,
        )
        (artifact_dir / "kernel-install.txt").write_text(install_output + "\n")
        run_checked(["uv", "pip", "check", "--python", str(kernel_python)], cwd=temp)

        jupyter_prefix = temp / "jupyter-prefix"
        run_checked(
            [
                str(kernel_python),
                "-m",
                "ipykernel",
                "install",
                "--prefix",
                str(jupyter_prefix),
                "--name",
                "hypster-host",
                "--display-name",
                "Hypster host test",
            ],
            cwd=temp,
        )
        jupyter_path = jupyter_prefix / "share" / "jupyter"

        token = "hypster-host-test"
        port = available_port()
        server_log_path = artifact_dir / "jupyterlab.log"
        server_log = server_log_path.open("w")
        runtime_dir = temp / "jupyter-runtime"
        server_env = os.environ.copy()
        server_env.pop("PYTHONPATH", None)
        server_env["JUPYTER_PATH"] = str(jupyter_path)
        server_env["JUPYTER_CONFIG_DIR"] = str(temp / "jupyter-config")
        server_env["JUPYTER_DATA_DIR"] = str(temp / "jupyter-data")
        server_env["JUPYTER_RUNTIME_DIR"] = str(runtime_dir)
        server_env["JUPYTERLAB_SETTINGS_DIR"] = str(temp / "jupyterlab-settings")
        server_env["JUPYTERLAB_WORKSPACES_DIR"] = str(temp / "jupyterlab-workspaces")
        server = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "jupyterlab",
                "--no-browser",
                f"--ServerApp.root_dir={notebook_root}",
                f"--ServerApp.port={port}",
                f"--ServerApp.port_retries={SERVER_PORT_RETRIES}",
                f"--IdentityProvider.token={token}",
                "--ServerApp.allow_remote_access=false",
            ],
            cwd=notebook_root,
            env=server_env,
            stdout=server_log,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )

        browser_errors: list[str] = []
        page_errors: list[str] = []
        kernel_packages = json.loads(
            run_checked(
                [
                    str(kernel_python),
                    "-c",
                    (
                        "import importlib.metadata as m, json, sys; "
                        "names=['hypster','anywidget','ipywidgets','jupyterlab-widgets','ipykernel']; "
                        "print(json.dumps({'python':sys.version,'executable':sys.executable,"
                        "'packages':{name:m.version(name) for name in names}}))"
                    ),
                ],
                cwd=temp,
            )
        )
        evidence: dict[str, Any] = {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "uv": run_checked(["uv", "--version"], cwd=temp),
            "wheel": wheel.name,
            "wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
            "host_packages": {
                name: importlib.metadata.version(name)
                for name in ["anywidget", "jupyterlab", "jupyterlab-widgets", "playwright", "pytest"]
            },
            "kernel": kernel_packages,
            "kernel_lock": KERNEL_LOCK.name,
        }
        (artifact_dir / "versions.json").write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")

        try:
            server_url = wait_for_server_info(runtime_dir, token, server)
            evidence["jupyter_server_pid"] = server.pid
            evidence["jupyter_server_url"] = server_url
            wait_for_server(server_url, token, server)
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch()
                evidence["chromium"] = browser.version
                context = browser.new_context()
                page = context.new_page()
                page.on(
                    "console", lambda message: browser_errors.append(message.text) if message.type == "error" else None
                )
                page.on("pageerror", lambda error: page_errors.append(f"{error}\n{error.stack}"))

                try:
                    page.goto(
                        f"{server_url}/lab/tree/{NOTEBOOK.name}?token={token}",
                        wait_until="domcontentloaded",
                        timeout=COMM_TIMEOUT_MS,
                    )
                    page.locator(".jp-NotebookPanel").wait_for(state="visible", timeout=COMM_TIMEOUT_MS)
                    wait_for_kernel(server_url, token, server)

                    # 1. The creation cell executes in the real kernel.
                    creation_output = execute_cell(page, 0, "HYPSTER_IMPORT_PATH=")
                    import_match = re.search(r"HYPSTER_IMPORT_PATH=(.+)", creation_output)
                    assert import_match, f"missing import-path proof in output: {creation_output}"
                    evidence["hypster_import_path"] = import_match.group(1).strip()

                    # 2. The current renderer creates a real widget root.
                    root = page.locator(".hypster-widget")
                    root.wait_for(state="visible", timeout=COMM_TIMEOUT_MS)
                    assert root.count() == 1, f"expected one rendered widget, got {root.count()}"
                    assert_widget_has_no_error(page)

                    # 3. Real branch and numeric DOM events change the live controls.
                    numeric = page.locator("input[data-path='remote.temperature'][data-kind='float']")
                    assert numeric.count() == 0, "dependent numeric control was reachable before the branch action"
                    before_branch_html = root.inner_html()
                    page.locator(".hypster-choice[data-path='mode'] .hypster-choice-trigger").click()
                    page.locator("[data-hypster-choice-option][data-path='mode']", has_text="remote").click()

                    # 4. The dependent control appears only after replacement snapshot DOM arrives.
                    numeric.wait_for(state="visible", timeout=COMM_TIMEOUT_MS)
                    assert numeric.input_value() == "0.25", (
                        "branch round trip did not publish the expected dependent default"
                    )
                    after_branch_html = root.inner_html()
                    assert after_branch_html != before_branch_html, "branch action did not publish replacement DOM"
                    assert_widget_has_no_error(page)

                    old_numeric = numeric.element_handle()
                    assert old_numeric is not None, "numeric control disappeared before its DOM event"
                    evidence["numeric_dom_before"] = numeric.input_value()
                    old_numeric.evaluate(
                        """element => {
                            element.value = '1.25';
                            element.dispatchEvent(new Event('change', { bubbles: true }));
                        }"""
                    )
                    page.wait_for_function("element => !element.isConnected", arg=old_numeric, timeout=COMM_TIMEOUT_MS)
                    replacement_numeric = page.locator("input[data-path='remote.temperature'][data-kind='float']")
                    replacement_numeric.wait_for(state="visible", timeout=COMM_TIMEOUT_MS)
                    assert replacement_numeric.input_value() == "1.25", (
                        "numeric comm round trip did not publish the chosen value"
                    )
                    assert not old_numeric.is_visible(), "numeric action did not detach the prior DOM control"
                    evidence["numeric_dom_after"] = replacement_numeric.input_value()
                    evidence["numeric_dom_replaced"] = True
                    assert_widget_has_no_error(page)

                    # 5. The notebook's Python oracle verifies the exact selected params.
                    verification_output = execute_cell(page, 1, "HYPSTER_PARAMS_VERIFIED=")
                    assert "'remote.temperature': 1.25" in verification_output, verification_output
                    (artifact_dir / "notebook-output.txt").write_text(f"{creation_output}\n\n{verification_output}\n")

                    page.keyboard.press("ControlOrMeta+S")
                    assert_saved_cells_succeeded(wait_for_saved_notebook(server_url, token))

                    # 6. Browser, page, protocol/widget, kernel, and missing-comm failures are loud.
                    assert not browser_errors, f"error-level browser console entries: {browser_errors}"
                    assert not page_errors, f"browser page errors: {page_errors}"
                except Exception:
                    page.screenshot(path=artifact_dir / "failure.png", full_page=True)
                    raise
                finally:
                    context.close()
                    browser.close()
        except PlaywrightTimeoutError as error:
            raise AssertionError(f"real-host comm or UI timeout: {error}") from error
        finally:
            evidence["browser_console_errors"] = browser_errors
            evidence["page_errors"] = page_errors
            (artifact_dir / "versions.json").write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
            terminate_process_group(server)
            server_log.close()
