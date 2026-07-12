import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { once } from "node:events";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { createInterface } from "node:readline";
import { afterEach, test } from "node:test";

import { JSDOM } from "jsdom";

const dom = new JSDOM("<!doctype html><html><body></body></html>", {
  url: "http://localhost/",
});
globalThis.window = dom.window;
globalThis.document = dom.window.document;
Object.defineProperty(globalThis, "navigator", {
  configurable: true,
  value: dom.window.navigator,
});
globalThis.HTMLElement = dom.window.HTMLElement;
globalThis.HTMLInputElement = dom.window.HTMLInputElement;
globalThis.HTMLSelectElement = dom.window.HTMLSelectElement;
globalThis.HTMLTextAreaElement = dom.window.HTMLTextAreaElement;
globalThis.IS_REACT_ACT_ENVIRONMENT = true;

const React = await import("react");
const { cleanup, fireEvent, render, screen } = await import("@testing-library/react");
const { HypsterRenderer } = await import("../dist/index.js");
const protocolFixture = JSON.parse(
  readFileSync(new URL("./fixtures/protocol-v1.json", import.meta.url), "utf8"),
);

afterEach(cleanup);

function parameter(kind, path, selectedValue, extra = {}) {
  const name = path.split(".").at(-1);
  return {
    name,
    path,
    kind,
    default_value: selectedValue,
    selected_value: selectedValue,
    options: null,
    minimum: null,
    maximum: null,
    description: null,
    display_label: name.replaceAll("_", " "),
    children: [],
    ...extra,
  };
}

function snapshot(parameters, draftValues, extra = {}) {
  return {
    protocol_version: 1,
    schema: {
      name: "demo",
      display_label: "Demo",
      parameters,
    },
    draft_values: draftValues,
    applied_values: draftValues,
    selected_params: draftValues,
    mode: { auto_apply: true },
    status: "applied",
    error: null,
    ...extra,
  };
}

function renderer(currentSnapshot, onAction = () => {}) {
  return React.createElement(HypsterRenderer, {
    snapshot: currentSnapshot,
    onAction,
  });
}

async function withTimeout(promise, message) {
  let timeout;
  try {
    return await Promise.race([
      promise,
      new Promise((_, reject) => {
        timeout = setTimeout(() => reject(new Error(message)), 5_000);
      }),
    ]);
  } finally {
    clearTimeout(timeout);
  }
}

async function terminateProcess(process, exited) {
  if (process.exitCode !== null || process.signalCode !== null) {
    await exited;
    return;
  }

  process.kill("SIGTERM");
  try {
    await withTimeout(exited, "Python protocol process ignored SIGTERM for 5s");
  } catch {
    process.kill("SIGKILL");
    await withTimeout(exited, "Python protocol process did not exit after SIGKILL within 5s");
  }
}

test("renders every current snapshot parameter kind from snake_case data", () => {
  const schemaFields = [
    { key: "invoice_number", value_type: "text", label: "Invoice number" },
  ];
  const rules = [
    {
      when: { field: "audience", operator: "=", value: "clinical" },
      then: "Use clinical language",
    },
  ];
  const parameters = [
    parameter("select", "mode", "local", { options: ["local", "remote"] }),
    parameter("multi_select", "tags", ["fast"], { options: ["fast", "safe"] }),
    parameter("bool", "enabled", true),
    parameter("int", "count", 2, { minimum: 1, maximum: 5 }),
    parameter("float", "temperature", 0.5, { minimum: 0, maximum: 1 }),
    parameter("text", "title", "hello"),
    parameter("multi_bool", "flags", [true, false]),
    parameter("multi_int", "depths", [1, 2]),
    parameter("multi_float", "weights", [0.2, 0.8]),
    parameter("multi_text", "labels", ["a", "b"]),
    parameter("rules", "routing_rules", rules, {
      metadata: {
        field_specs: [
          { type: "select", name: "audience", options: ["clinical"], operators: ["="] },
        ],
        then_specs: [{ type: "text", name: "prompt", operators: ["=", "contains"] }],
        combinators: ["and"],
      },
    }),
    parameter("schema", "extraction_fields", schemaFields, {
      metadata: { schema_fields: schemaFields },
    }),
    parameter("group", "advanced", null, {
      display_label: "Advanced",
      children: [parameter("text", "advanced.note", "nested")],
    }),
  ];
  const draftValues = Object.fromEntries(
    parameters
      .flatMap((item) => (item.kind === "group" ? item.children : [item]))
      .map((item) => [item.path, item.selected_value]),
  );

  render(renderer(snapshot(parameters, draftValues)));

  assert.equal(screen.getByRole("combobox", { name: "mode" }).value, "0");
  assert.deepEqual(
    Array.from(screen.getByRole("listbox", { name: "tags" }).selectedOptions, (option) => option.textContent),
    ["fast"],
  );
  assert.equal(screen.getByRole("checkbox", { name: "enabled" }).checked, true);
  assert.equal(screen.getByRole("spinbutton", { name: "count" }).value, "2");
  assert.equal(screen.getByRole("spinbutton", { name: "temperature" }).value, "0.5");
  assert.equal(screen.getByRole("textbox", { name: "title" }).value, "hello");
  assert.equal(screen.getByRole("textbox", { name: "flags" }).value, "[\n  true,\n  false\n]");
  assert.equal(screen.getByRole("textbox", { name: "depths" }).value, "[\n  1,\n  2\n]");
  assert.equal(screen.getByRole("textbox", { name: "weights" }).value, "[\n  0.2,\n  0.8\n]");
  assert.equal(screen.getByRole("textbox", { name: "labels" }).value, '[\n  "a",\n  "b"\n]');
  assert.match(screen.getByText(/Condition fields:/).textContent, /audience/);
  assert.match(screen.getByText(/Schema fields:/).textContent, /Invoice number/);
  assert.equal(screen.getByRole("group", { name: "Advanced" }) !== null, true);
  assert.equal(screen.getByRole("textbox", { name: "note" }).value, "nested");
});

test("emits exact Protocol V1 set, apply, and reset action payloads", () => {
  const actions = [];
  const parameters = [parameter("int", "count", 2)];
  const current = snapshot(parameters, { count: 2 }, { mode: { auto_apply: false }, status: "pending" });
  render(renderer(current, (action) => actions.push(action)));

  fireEvent.change(screen.getByRole("spinbutton", { name: "count" }), { target: { value: "4" } });
  fireEvent.change(screen.getByRole("spinbutton", { name: "count" }), { target: { value: "" } });
  fireEvent.click(screen.getByRole("button", { name: "Apply" }));
  fireEvent.click(screen.getByRole("button", { name: "Reset" }));

  assert.deepEqual(actions, [
    { protocol_version: 1, type: "set_value", path: "count", value: 4 },
    { protocol_version: 1, type: "set_value", path: "count", value: null },
    { protocol_version: 1, type: "apply" },
    { protocol_version: 1, type: "reset" },
  ]);
});

test("does not change the rendered value until a replacement snapshot arrives", () => {
  const actions = [];
  const count = parameter("int", "count", 2);
  const view = render(renderer(snapshot([count], { count: 2 }), (action) => actions.push(action)));
  const input = screen.getByRole("spinbutton", { name: "count" });

  fireEvent.change(input, { target: { value: "7" } });

  assert.deepEqual(actions, [{ protocol_version: 1, type: "set_value", path: "count", value: 7 }]);
  assert.equal(input.value, "2");

  view.rerender(renderer(snapshot([count], { count: 7 }), (action) => actions.push(action)));
  assert.equal(screen.getByRole("spinbutton", { name: "count" }).value, "7");
});

test("adds a dependent control only when the replacement snapshot contains it", () => {
  const actions = [];
  const mode = parameter("select", "mode", "local", { options: ["local", "remote"] });
  const view = render(renderer(snapshot([mode], { mode: "local" }), (action) => actions.push(action)));

  fireEvent.change(screen.getByRole("combobox", { name: "mode" }), { target: { value: "1" } });

  assert.deepEqual(actions, [
    { protocol_version: 1, type: "set_value", path: "mode", value: "remote" },
  ]);
  assert.equal(screen.queryByRole("textbox", { name: "remote token" }), null);

  const token = parameter("text", "remote_token", "secret", { display_label: "remote token" });
  view.rerender(
    renderer(snapshot([mode, token], { mode: "remote", remote_token: "secret" }), (action) => actions.push(action)),
  );
  assert.equal(screen.getByRole("textbox", { name: "remote token" }).value, "secret");
});

test("consumes Python-generated Protocol V1 snapshots and actions", () => {
  const actions = [];
  const view = render(renderer(protocolFixture.initial_snapshot, (action) => actions.push(action)));

  fireEvent.change(screen.getByRole("combobox", { name: "Mode" }), { target: { value: "1" } });
  assert.deepEqual(actions, [protocolFixture.actions[0]]);

  view.rerender(renderer(protocolFixture.branch_snapshot, (action) => actions.push(action)));
  fireEvent.change(screen.getByRole("spinbutton", { name: "Temperature" }), { target: { value: "1.25" } });
  assert.deepEqual(actions, protocolFixture.actions);

  view.rerender(renderer(protocolFixture.final_snapshot, (action) => actions.push(action)));
  assert.equal(screen.getByRole("spinbutton", { name: "Temperature" }).value, "1.25");
  assert.deepEqual(protocolFixture.final_params, { mode: "remote", temperature: 1.25 });
});

test("completes a live Python to React to Python round trip", async (context) => {
  const reactRoot = fileURLToPath(new URL("..", import.meta.url));
  const python = spawn("uv", ["run", "python", "test/python_protocol.py", "--serve"], {
    cwd: reactRoot,
    stdio: ["pipe", "pipe", "pipe"],
  });
  const lines = createInterface({ input: python.stdout });
  const snapshots = lines[Symbol.asyncIterator]();
  const exited = once(python, "exit");
  let stderr = "";
  python.stderr.on("data", (chunk) => {
    stderr += chunk.toString();
  });
  context.after(async () => {
    if (!python.stdin.destroyed) python.stdin.end();
    lines.close();
    await terminateProcess(python, exited);
  });

  const initialLine = await withTimeout(snapshots.next(), "Python did not emit the initial snapshot within 5s");
  assert.equal(initialLine.done, false, stderr);
  const initialSnapshot = JSON.parse(initialLine.value);
  const actions = [];
  const view = render(renderer(initialSnapshot, (action) => actions.push(action)));

  fireEvent.change(screen.getByRole("combobox", { name: "Mode" }), { target: { value: "1" } });
  python.stdin.write(`${JSON.stringify(actions.at(-1))}\n`);
  const branchLine = await withTimeout(snapshots.next(), "Python did not emit the branch snapshot within 5s");
  assert.equal(branchLine.done, false, stderr);
  view.rerender(renderer(JSON.parse(branchLine.value), (action) => actions.push(action)));

  fireEvent.change(screen.getByRole("spinbutton", { name: "Temperature" }), { target: { value: "1.25" } });
  python.stdin.write(`${JSON.stringify(actions.at(-1))}\n`);
  const finalLine = await withTimeout(snapshots.next(), "Python did not emit the final snapshot within 5s");
  assert.equal(finalLine.done, false, stderr);
  const finalSnapshot = JSON.parse(finalLine.value);
  view.rerender(renderer(finalSnapshot, (action) => actions.push(action)));

  assert.deepEqual(finalSnapshot.selected_params, { mode: "remote", temperature: 1.25 });
  assert.equal(screen.getByRole("spinbutton", { name: "Temperature" }).value, "1.25");

  python.stdin.end();
  const [exitCode, signal] = await withTimeout(exited, "Python protocol process did not exit within 5s");
  assert.equal(exitCode, 0, `Python exited via ${signal ?? "code"}: ${stderr}`);
});

test("renders snapshot errors and missing authoritative values visibly", () => {
  const count = parameter("int", "count", 2);
  const current = snapshot([count], {}, {
    status: "draft_error",
    error: { kind: "exploration", message: "count must be positive" },
  });
  render(renderer(current));

  assert.match(screen.getAllByRole("alert")[0].textContent, /exploration: count must be positive/);
  assert.match(screen.getAllByRole("alert")[1].textContent, /missing draft_values\["count"\]/);
});

test("renders missing or mismatched protocol versions visibly and exposes no action surfaces", () => {
  const count = parameter("int", "count", 2);

  for (const invalidSnapshot of [
    { ...snapshot([count], { count: 2 }), protocol_version: 2 },
    Object.fromEntries(
      Object.entries(snapshot([count], { count: 2 })).filter(([key]) => key !== "protocol_version"),
    ),
  ]) {
    const actions = [];
    const view = render(renderer(invalidSnapshot, (action) => actions.push(action)));

    assert.match(screen.getByRole("alert").textContent, /protocol version/i);
    assert.equal(screen.queryByRole("spinbutton", { name: "count" }), null);
    assert.equal(screen.queryByRole("button", { name: "Apply" }), null);
    assert.equal(screen.queryByRole("button", { name: "Reset" }), null);
    assert.deepEqual(actions, []);

    view.unmount();
  }
});
