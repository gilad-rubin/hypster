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
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

from playwright.sync_api import Locator, Page, sync_playwright
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

HARNESS_ROOT = Path(__file__).resolve().parent
NOTEBOOK = HARNESS_ROOT / "branch_round_trip.ipynb"
KERNEL_LOCK = HARNESS_ROOT / "kernel-requirements.lock"
SERVER_TIMEOUT_SECONDS = 60
SERVER_PORT_RETRIES = 5
COMM_TIMEOUT_MS = 30_000
BASIC_CELL_MARKERS = (
    "create-auto",
    "verify-auto",
)
JUPYTERLAB_CELL_MARKERS = BASIC_CELL_MARKERS + (
    "create-manual",
    "verify-manual-staged",
    "verify-manual-applied",
    "verify-manual-reset",
    "create-protocol",
    "inject-protocol-mismatch",
    "verify-protocol-unchanged",
    "recover-protocol",
)


@dataclass(frozen=True)
class HostConfig:
    name: str
    server_module: str
    route: str
    browser: Literal["chromium", "firefox"]
    required_cell_markers: tuple[str, ...]
    extended: bool


JUPYTERLAB = HostConfig(
    name="jupyterlab",
    server_module="jupyterlab",
    route="lab/tree",
    browser="chromium",
    required_cell_markers=JUPYTERLAB_CELL_MARKERS,
    extended=True,
)
NOTEBOOK7 = HostConfig(
    name="notebook7",
    server_module="notebook",
    route="tree",
    browser="firefox",
    required_cell_markers=BASIC_CELL_MARKERS,
    extended=False,
)


@dataclass(frozen=True)
class CellExecution:
    output: str
    count: int
    identity: str


@dataclass(frozen=True)
class OracleEvidence:
    count: int
    identity: str


@dataclass(frozen=True)
class WidgetAppearance:
    theme: str
    color: str
    background: str
    contrast: float


@dataclass(frozen=True)
class ThemeEvidence:
    dark: WidgetAppearance
    light: WidgetAppearance


@dataclass(frozen=True)
class BasicAutoEvidence:
    numeric_dom_before: str
    numeric_dom_after: str
    numeric_dom_replaced: bool
    oracle: OracleEvidence
    transcript: tuple[str, ...]


@dataclass(frozen=True)
class BranchMemoryEvidence:
    branch_memory_dom_restored: str
    branch_memory_python: str
    oracle: OracleEvidence
    transcript: tuple[str, ...]


@dataclass(frozen=True)
class ManualEvidence:
    staged_dom: str
    staged_python: str
    applied_python: str
    invalid_dom_error: str
    invalid_python_unchanged: str
    reset_dom: str
    reset_python: str
    applied_oracles: tuple[OracleEvidence, ...]
    transcript: tuple[str, ...]


@dataclass(frozen=True)
class ProtocolEvidence:
    error_dom: str
    action_emitted: bool
    python_unchanged: str
    recovered_dom: str
    oracles: tuple[OracleEvidence, ...]
    transcript: tuple[str, ...]


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


def wait_for_server(url: str, token: str, process: subprocess.Popen[str], host_name: str) -> None:
    deadline = time.monotonic() + SERVER_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        assert process.poll() is None, f"{host_name} exited before readiness with code {process.returncode}"
        try:
            with urllib.request.urlopen(f"{url}/api/status?token={token}", timeout=1) as response:
                if response.status == 200:
                    return
        except OSError:
            time.sleep(0.25)
    raise AssertionError(
        f"{host_name} did not become ready within {SERVER_TIMEOUT_SECONDS}s; process return code: {process.poll()}"
    )


def wait_for_server_info(
    runtime_dir: Path,
    token: str,
    process: subprocess.Popen[str],
    host_name: str,
) -> str:
    deadline = time.monotonic() + SERVER_TIMEOUT_SECONDS
    info_file = runtime_dir / f"jpserver-{process.pid}.json"
    while time.monotonic() < deadline:
        assert process.poll() is None, f"{host_name} exited before publishing server info: {process.returncode}"
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

    raise AssertionError(f"{host_name} did not publish {info_file.name} within {SERVER_TIMEOUT_SECONDS}s")


def wait_for_kernel(url: str, token: str, process: subprocess.Popen[str], host_name: str) -> None:
    deadline = time.monotonic() + SERVER_TIMEOUT_SECONDS
    request = urllib.request.Request(
        f"{url}/api/sessions",
        headers={"Authorization": f"token {token}"},
    )
    while time.monotonic() < deadline:
        assert process.poll() is None, f"{host_name} exited while the kernel was connecting: {process.returncode}"
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


def notebook_cell(page: Page, marker: str) -> Locator:
    matches: list[Locator] = []
    cells = page.locator(".jp-Notebook-cell")
    for index in range(cells.count()):
        cell = cells.nth(index)
        if f"HYPSTER_HOST_CELL:{marker}" in cell.locator(".jp-Cell-inputArea").inner_text():
            matches.append(cell)
    assert len(matches) == 1, f"expected one notebook cell marked {marker!r}, got {len(matches)}"
    return matches[0]


def execute_cell(page: Page, cell_marker: str, output_marker: str) -> CellExecution:
    cell = notebook_cell(page, cell_marker)
    prompt = cell.locator(".jp-InputPrompt")
    before_prompt = prompt.inner_text()
    output = cell.locator(".jp-OutputArea")
    before_output = output.inner_text()
    cell.locator(".jp-Cell-inputArea").click()
    page.keyboard.press("Control+Enter")

    deadline = time.monotonic() + COMM_TIMEOUT_MS / 1000
    count_match: re.Match[str] | None = None
    while time.monotonic() < deadline:
        current_prompt = prompt.inner_text()
        candidate = re.search(r"\[(\d+)\]:", current_prompt)
        if current_prompt != before_prompt and candidate is not None:
            count_match = candidate
            break
        page.wait_for_timeout(25)
    assert count_match is not None, (
        f"cell {cell_marker!r} did not publish a new execution count; before was {before_prompt!r}, "
        f"after was {prompt.inner_text()!r}"
    )

    try:
        output.get_by_text(re.compile(re.escape(output_marker))).wait_for(state="visible", timeout=COMM_TIMEOUT_MS)
    except PlaywrightTimeoutError as error:
        raise AssertionError(
            f"cell {cell_marker!r} did not emit {output_marker!r}; kernel output was:\n{output.inner_text()}"
        ) from error
    current_output = output.inner_text()
    identity_match = re.search(r"EXECUTION_ID=([0-9a-f]{32})", current_output)
    assert identity_match is not None, f"cell {cell_marker!r} output has no execution identity: {current_output}"
    identity = identity_match.group(1)
    assert current_output != before_output, f"cell {cell_marker!r} retained stale output"
    assert identity not in before_output, f"cell {cell_marker!r} reused stale execution identity {identity}"
    error_output = output.locator("[data-mime-type='application/vnd.jupyter.error']")
    assert error_output.count() == 0, f"cell {cell_marker!r} produced a kernel error: {current_output}"
    return CellExecution(output=current_output, count=int(count_match.group(1)), identity=identity)


def assert_widget_has_no_error(page: Page) -> None:
    errors = page.locator(".hypster-status-error, .hypster-status-draft_error")
    assert errors.count() == 0, f"widget surfaced an error: {errors.all_inner_texts()}"


def widget_appearance(widget: Locator) -> WidgetAppearance:
    raw = widget.evaluate(
        """element => {
            const rgb = value => (value.match(/[0-9.]+/g) || []).slice(0, 3).map(Number);
            const luminance = value => {
                const channels = rgb(value).map(channel => {
                    const normalized = channel / 255;
                    return normalized <= 0.04045
                        ? normalized / 12.92
                        : ((normalized + 0.055) / 1.055) ** 2.4;
                });
                return 0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2];
            };
            const style = getComputedStyle(element);
            const foreground = luminance(style.color);
            const background = luminance(style.backgroundColor);
            return {
                theme: element.dataset.hypsterTheme,
                color: style.color,
                background: style.backgroundColor,
                contrast: (Math.max(foreground, background) + 0.05) /
                    (Math.min(foreground, background) + 0.05),
            };
        }"""
    )
    assert isinstance(raw, dict), f"widget appearance must be an object, got {raw!r}"
    theme = raw.get("theme")
    color = raw.get("color")
    background = raw.get("background")
    contrast = raw.get("contrast")
    assert isinstance(theme, str), f"widget theme must be text, got {theme!r}"
    assert isinstance(color, str), f"widget color must be text, got {color!r}"
    assert isinstance(background, str), f"widget background must be text, got {background!r}"
    assert isinstance(contrast, (int, float)) and not isinstance(contrast, bool), (
        f"widget contrast must be numeric, got {contrast!r}"
    )
    return WidgetAppearance(
        theme=theme,
        color=color,
        background=background,
        contrast=float(contrast),
    )


def switch_jupyterlab_theme(page: Page, theme: str) -> None:
    page.locator(".lm-MenuBar-itemLabel", has_text="Settings").click()
    page.locator(".lm-Menu-itemLabel").filter(has_text=re.compile(r"^Theme$")).filter(visible=True).hover()
    page.locator(".lm-Menu-itemLabel").filter(has_text=re.compile(f"^{re.escape(theme)}$")).filter(visible=True).click()


def notebook_model(url: str, token: str) -> dict[str, Any]:
    request = urllib.request.Request(
        f"{url}/api/contents/{NOTEBOOK.name}?content=1",
        headers={"Authorization": f"token {token}"},
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.load(response)["content"]


def assert_saved_cells_succeeded(model: dict[str, Any], required_markers: tuple[str, ...]) -> None:
    cells = model["cells"]
    for marker in required_markers:
        matching = [cell for cell in cells if f"HYPSTER_HOST_CELL:{marker}" in "".join(cell["source"])]
        assert len(matching) == 1, f"expected one saved cell marked {marker!r}, got {len(matching)}"
        cell = matching[0]
        assert cell["execution_count"] is not None, f"cell {marker!r} was not executed"
        errors = [output for output in cell["outputs"] if output["output_type"] == "error"]
        assert not errors, f"cell {marker!r} saved kernel errors: {errors}"


def wait_for_saved_notebook(url: str, token: str, required_markers: tuple[str, ...]) -> dict[str, Any]:
    deadline = time.monotonic() + 10
    last_model: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        last_model = notebook_model(url, token)
        cells = last_model["cells"]
        executed_markers = {
            marker
            for marker in required_markers
            if any(
                f"HYPSTER_HOST_CELL:{marker}" in "".join(cell["source"]) and cell["execution_count"] is not None
                for cell in cells
            )
        }
        if executed_markers == set(required_markers):
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


def exercise_theme(page: Page, widget: Locator) -> ThemeEvidence:
    widget_handle = widget.element_handle()
    assert widget_handle is not None, "rendered widget root disappeared"

    switch_jupyterlab_theme(page, "JupyterLab Dark")
    page.wait_for_function(
        "element => element.dataset.hypsterTheme === 'dark'",
        arg=widget_handle,
        timeout=COMM_TIMEOUT_MS,
    )
    dark = widget_appearance(widget)
    assert dark.contrast >= 4.5, dark

    switch_jupyterlab_theme(page, "JupyterLab Light")
    page.wait_for_function(
        "element => element.dataset.hypsterTheme === 'light'",
        arg=widget_handle,
        timeout=COMM_TIMEOUT_MS,
    )
    light = widget_appearance(widget)
    assert light.contrast >= 4.5, light
    assert light.background != dark.background, (light, dark)
    return ThemeEvidence(dark=dark, light=light)


def exercise_basic_auto_mode(
    page: Page,
    widget: Locator,
) -> BasicAutoEvidence:
    numeric = widget.locator("input[data-path='remote.temperature'][data-kind='float']")
    assert numeric.count() == 0, "dependent numeric control was reachable before the branch action"
    before_branch_html = widget.inner_html()
    widget.locator(".hypster-choice[data-path='mode'] .hypster-choice-trigger").click()
    widget.locator("[data-hypster-choice-option][data-path='mode']", has_text="remote").click()

    numeric.wait_for(state="visible", timeout=COMM_TIMEOUT_MS)
    assert numeric.input_value() == "0.25", "branch round trip did not publish the expected dependent default"
    assert widget.inner_html() != before_branch_html, "branch action did not publish replacement DOM"
    assert_widget_has_no_error(page)

    old_numeric = numeric.element_handle()
    assert old_numeric is not None, "numeric control disappeared before its DOM event"
    numeric_dom_before = numeric.input_value()
    numeric.fill("1.25")
    numeric.press("Tab")
    page.wait_for_function("element => !element.isConnected", arg=old_numeric, timeout=COMM_TIMEOUT_MS)
    replacement_numeric = widget.locator("input[data-path='remote.temperature'][data-kind='float']")
    replacement_numeric.wait_for(state="visible", timeout=COMM_TIMEOUT_MS)
    assert replacement_numeric.input_value() == "1.25", "numeric comm round trip lost the chosen value"
    assert not old_numeric.is_visible(), "numeric action did not detach the prior DOM control"
    numeric_dom_after = replacement_numeric.input_value()
    assert_widget_has_no_error(page)

    verification = execute_cell(page, "verify-auto", "HYPSTER_PARAMS_VERIFIED=")
    assert "'remote.temperature': 1.25" in verification.output, verification.output
    return BasicAutoEvidence(
        numeric_dom_before=numeric_dom_before,
        numeric_dom_after=numeric_dom_after,
        numeric_dom_replaced=True,
        oracle=OracleEvidence(count=verification.count, identity=verification.identity),
        transcript=(verification.output,),
    )


def exercise_branch_memory(page: Page, widget: Locator) -> BranchMemoryEvidence:
    numeric = widget.locator("input[data-path='remote.temperature'][data-kind='float']")
    numeric.wait_for(state="visible", timeout=COMM_TIMEOUT_MS)
    assert numeric.input_value() == "1.25", "Branch Choice Memory did not start from the chosen value"

    widget.locator(".hypster-choice[data-path='mode'] .hypster-choice-trigger").click()
    widget.locator("[data-hypster-choice-option][data-path='mode']", has_text="local").click()
    numeric.wait_for(state="detached", timeout=COMM_TIMEOUT_MS)
    widget.locator(".hypster-choice[data-path='mode'] .hypster-choice-trigger").click()
    widget.locator("[data-hypster-choice-option][data-path='mode']", has_text="remote").click()
    restored_numeric = widget.locator("input[data-path='remote.temperature'][data-kind='float']")
    restored_numeric.wait_for(state="visible", timeout=COMM_TIMEOUT_MS)
    assert restored_numeric.input_value() == "1.25", "Branch Choice Memory lost the compatible numeric choice"
    branch_memory_dom_restored = restored_numeric.input_value()

    verification = execute_cell(page, "verify-auto", "HYPSTER_PARAMS_VERIFIED=")
    assert "'remote.temperature': 1.25" in verification.output, verification.output
    return BranchMemoryEvidence(
        branch_memory_dom_restored=branch_memory_dom_restored,
        branch_memory_python=verification.output,
        oracle=OracleEvidence(count=verification.count, identity=verification.identity),
        transcript=(verification.output,),
    )


def exercise_manual_mode(page: Page) -> ManualEvidence:
    manual_ready = execute_cell(page, "create-manual", "HYPSTER_MANUAL_READY")
    transcript = [manual_ready.output]
    widget = notebook_cell(page, "create-manual").locator(".hypster-widget")
    widget.wait_for(state="visible", timeout=COMM_TIMEOUT_MS)
    widget.locator(".hypster-choice[data-path='mode'] .hypster-choice-trigger").click()
    widget.locator("[data-hypster-choice-option][data-path='mode']", has_text="remote").click()
    numeric = widget.locator("input[data-path='remote.temperature'][data-kind='float']")
    numeric.wait_for(state="visible", timeout=COMM_TIMEOUT_MS)
    numeric.fill("1.25")
    numeric.press("Tab")
    widget.locator(".hypster-status-pending").wait_for(state="visible", timeout=COMM_TIMEOUT_MS)
    assert numeric.input_value() == "1.25"

    staged = execute_cell(page, "verify-manual-staged", "HYPSTER_MANUAL_STAGED_VERIFIED=")
    assert "{'mode': 'local'}" in staged.output, staged.output
    staged_dom = f"remote.temperature={numeric.input_value()}"
    transcript.append(staged.output)

    widget.get_by_role("button", name="Apply", exact=True).click()
    widget.locator(".hypster-status-applied").wait_for(state="visible", timeout=COMM_TIMEOUT_MS)
    assert numeric.input_value() == "1.25"
    applied = execute_cell(page, "verify-manual-applied", "HYPSTER_MANUAL_APPLIED_VERIFIED=")
    assert "'remote.temperature': 1.25" in applied.output, applied.output
    transcript.append(applied.output)

    numeric = widget.locator("input[data-path='remote.temperature'][data-kind='float']")
    numeric.fill("")
    numeric.press("Tab")
    visible_error = widget.locator(".hypster-status-draft_error")
    visible_error.wait_for(state="visible", timeout=COMM_TIMEOUT_MS)
    assert visible_error.inner_text().startswith("Error:"), visible_error.inner_text()
    assert numeric.input_value() == ""
    assert widget.get_by_role("button", name="Apply", exact=True).is_disabled()
    invalid = execute_cell(page, "verify-manual-applied", "HYPSTER_MANUAL_APPLIED_VERIFIED=")
    assert "'remote.temperature': 1.25" in invalid.output, invalid.output
    invalid_dom_error = visible_error.inner_text()
    transcript.append(invalid.output)

    numeric = widget.locator("input[data-path='remote.temperature'][data-kind='float']")
    numeric.fill("1.5")
    numeric.press("Tab")
    widget.locator(".hypster-status-pending").wait_for(state="visible", timeout=COMM_TIMEOUT_MS)
    assert numeric.input_value() == "1.5"
    pending = execute_cell(page, "verify-manual-applied", "HYPSTER_MANUAL_APPLIED_VERIFIED=")
    assert "'remote.temperature': 1.25" in pending.output, pending.output
    assert applied.count < invalid.count < pending.count
    assert len({applied.identity, invalid.identity, pending.identity}) == 3
    transcript.append(pending.output)

    widget.get_by_role("button", name="Reset", exact=True).click()
    widget.locator(".hypster-status-applied").wait_for(state="visible", timeout=COMM_TIMEOUT_MS)
    assert widget.locator("input[data-path='remote.temperature'][data-kind='float']").count() == 0
    mode = widget.locator(".hypster-choice[data-path='mode'] .hypster-choice-value").inner_text()
    assert mode == "local"
    reset = execute_cell(page, "verify-manual-reset", "HYPSTER_MANUAL_RESET_VERIFIED=")
    assert "{'mode': 'local'}" in reset.output, reset.output
    transcript.append(reset.output)
    return ManualEvidence(
        staged_dom=staged_dom,
        staged_python=staged.output,
        applied_python=applied.output,
        invalid_dom_error=invalid_dom_error,
        invalid_python_unchanged=invalid.output,
        reset_dom="mode=local; remote.temperature absent",
        reset_python=reset.output,
        applied_oracles=tuple(
            OracleEvidence(count=execution.count, identity=execution.identity)
            for execution in (applied, invalid, pending)
        ),
        transcript=tuple(transcript),
    )


def exercise_protocol_guard(page: Page) -> ProtocolEvidence:
    ready = execute_cell(page, "create-protocol", "HYPSTER_PROTOCOL_READY")
    transcript = [ready.output]
    widget = notebook_cell(page, "create-protocol").locator(".hypster-widget")
    widget.wait_for(state="visible", timeout=COMM_TIMEOUT_MS)
    assert_widget_has_no_error(page)

    mismatch = execute_cell(page, "inject-protocol-mismatch", "HYPSTER_PROTOCOL_MISMATCH_SENT=2")
    transcript.append(mismatch.output)
    protocol_error = widget.locator(".hypster-status-protocol_error[role='alert']")
    protocol_error.wait_for(state="visible", timeout=COMM_TIMEOUT_MS)
    error_text = protocol_error.inner_text()
    assert "expected 1, received 2" in error_text, error_text
    assert widget.get_attribute("data-status") == "protocol_error"
    assert widget.locator(".hypster-field, .hypster-actions, input, select, textarea, button").count() == 0

    unchanged = execute_cell(page, "verify-protocol-unchanged", "HYPSTER_PROTOCOL_UNCHANGED=")
    assert "{'mode': 'local'}" in unchanged.output, unchanged.output
    assert "ACTION={}" in unchanged.output, unchanged.output
    transcript.append(unchanged.output)

    recovered = execute_cell(page, "recover-protocol", "HYPSTER_PROTOCOL_RECOVERED=1")
    transcript.append(recovered.output)
    protocol_error.wait_for(state="detached", timeout=COMM_TIMEOUT_MS)
    widget.locator(".hypster-status-applied").wait_for(state="visible", timeout=COMM_TIMEOUT_MS)
    assert widget.locator(".hypster-choice[data-path='mode'] .hypster-choice-value").inner_text() == "local"
    recovered_oracle = execute_cell(page, "verify-protocol-unchanged", "HYPSTER_PROTOCOL_UNCHANGED=")
    assert unchanged.count < recovered_oracle.count
    assert unchanged.identity != recovered_oracle.identity
    assert "{'mode': 'local'}" in recovered_oracle.output, recovered_oracle.output
    assert "ACTION={}" in recovered_oracle.output, recovered_oracle.output
    transcript.append(recovered_oracle.output)
    return ProtocolEvidence(
        error_dom=error_text,
        action_emitted=False,
        python_unchanged=unchanged.output,
        recovered_dom="mode=local; controls restored",
        oracles=tuple(
            OracleEvidence(count=execution.count, identity=execution.identity)
            for execution in (unchanged, recovered_oracle)
        ),
        transcript=tuple(transcript),
    )


def run_real_host_round_trip(host: HostConfig) -> None:
    wheel = required_path("HYPSTER_HOST_WHEEL").resolve(strict=True)
    artifact_dir = required_path("HYPSTER_HOST_ARTIFACT_DIR")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    assert wheel.suffix == ".whl", f"expected a built wheel, got {wheel}"

    with tempfile.TemporaryDirectory(prefix=f"hypster-{host.name}-host-") as temp_raw:
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
        server_log_path = artifact_dir / f"{host.name}.log"
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
                host.server_module,
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
            "host": asdict(host),
            "host_packages": {
                name: importlib.metadata.version(name)
                for name in [
                    "anywidget",
                    "jupyter-server",
                    "jupyterlab",
                    "jupyterlab-widgets",
                    "notebook",
                    "playwright",
                    "pytest",
                ]
            },
            "kernel": kernel_packages,
            "kernel_lock": KERNEL_LOCK.name,
        }
        (artifact_dir / "versions.json").write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")

        try:
            server_url = wait_for_server_info(runtime_dir, token, server, host.name)
            evidence["jupyter_server_pid"] = server.pid
            evidence["jupyter_server_url"] = server_url
            wait_for_server(server_url, token, server, host.name)
            with sync_playwright() as playwright:
                browser_type = playwright.chromium if host.browser == "chromium" else playwright.firefox
                browser = browser_type.launch()
                evidence["browser"] = {"engine": host.browser, "version": browser.version}
                context = browser.new_context()
                page = context.new_page()
                page.on(
                    "console", lambda message: browser_errors.append(message.text) if message.type == "error" else None
                )
                page.on("pageerror", lambda error: page_errors.append(f"{error}\n{error.stack}"))

                try:
                    page.goto(
                        f"{server_url}/{host.route}/{NOTEBOOK.name}?token={token}",
                        wait_until="domcontentloaded",
                        timeout=COMM_TIMEOUT_MS,
                    )
                    page.locator(".jp-NotebookPanel").wait_for(state="visible", timeout=COMM_TIMEOUT_MS)
                    wait_for_kernel(server_url, token, server, host.name)
                    page.wait_for_timeout(1_000)
                    evidence["page_errors_before_widget_execution"] = list(page_errors)

                    creation = execute_cell(page, "create-auto", "HYPSTER_IMPORT_PATH=")
                    import_match = re.search(r"HYPSTER_IMPORT_PATH=([^;]+)", creation.output)
                    assert import_match, f"missing import-path proof in output: {creation.output}"
                    evidence["hypster_import_path"] = import_match.group(1).strip()
                    evidence["creation_oracle"] = {"count": creation.count, "identity": creation.identity}
                    transcript = [creation.output]

                    auto_widget = notebook_cell(page, "create-auto").locator(".hypster-widget")
                    auto_widget.wait_for(state="visible", timeout=COMM_TIMEOUT_MS)
                    assert auto_widget.count() == 1, f"expected one rendered widget, got {auto_widget.count()}"
                    assert_widget_has_no_error(page)

                    basic_evidence = exercise_basic_auto_mode(page, auto_widget)
                    evidence["numeric_dom_before"] = basic_evidence.numeric_dom_before
                    evidence["numeric_dom_after"] = basic_evidence.numeric_dom_after
                    evidence["numeric_dom_replaced"] = basic_evidence.numeric_dom_replaced
                    evidence["auto_oracle"] = asdict(basic_evidence.oracle)
                    evidence["page_errors_after_basic_round_trip"] = list(page_errors)
                    transcript.extend(basic_evidence.transcript)

                    if host.extended:
                        theme_evidence = exercise_theme(page, auto_widget)
                        branch_memory_evidence = exercise_branch_memory(page, auto_widget)
                        manual_evidence = exercise_manual_mode(page)
                        protocol_evidence = exercise_protocol_guard(page)

                        evidence["theme_dark"] = asdict(theme_evidence.dark)
                        evidence["theme_light"] = asdict(theme_evidence.light)
                        evidence["branch_memory_dom_restored"] = branch_memory_evidence.branch_memory_dom_restored
                        evidence["branch_memory_python"] = branch_memory_evidence.branch_memory_python
                        evidence["branch_memory_oracle"] = asdict(branch_memory_evidence.oracle)
                        evidence["manual_staged_dom"] = manual_evidence.staged_dom
                        evidence["manual_staged_python"] = manual_evidence.staged_python
                        evidence["manual_applied_python"] = manual_evidence.applied_python
                        evidence["invalid_dom_error"] = manual_evidence.invalid_dom_error
                        evidence["invalid_python_unchanged"] = manual_evidence.invalid_python_unchanged
                        evidence["manual_reset_dom"] = manual_evidence.reset_dom
                        evidence["manual_reset_python"] = manual_evidence.reset_python
                        evidence["manual_applied_oracles"] = [
                            asdict(oracle) for oracle in manual_evidence.applied_oracles
                        ]
                        evidence["protocol_error_dom"] = protocol_evidence.error_dom
                        evidence["protocol_action_emitted"] = protocol_evidence.action_emitted
                        evidence["protocol_python_unchanged"] = protocol_evidence.python_unchanged
                        evidence["protocol_recovered_dom"] = protocol_evidence.recovered_dom
                        evidence["protocol_oracles"] = [asdict(oracle) for oracle in protocol_evidence.oracles]

                        transcript.extend(branch_memory_evidence.transcript)
                        transcript.extend(manual_evidence.transcript)
                        transcript.extend(protocol_evidence.transcript)

                    (artifact_dir / "notebook-output.txt").write_text("\n\n".join(transcript) + "\n")

                    page.keyboard.press("ControlOrMeta+S")
                    saved_notebook = wait_for_saved_notebook(server_url, token, host.required_cell_markers)
                    assert_saved_cells_succeeded(saved_notebook, host.required_cell_markers)

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


def test_real_jupyterlab_round_trip() -> None:
    run_real_host_round_trip(JUPYTERLAB)


def test_real_notebook7_round_trip() -> None:
    run_real_host_round_trip(NOTEBOOK7)
