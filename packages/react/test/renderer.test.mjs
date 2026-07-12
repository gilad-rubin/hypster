import assert from "node:assert/strict";
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
  fireEvent.click(screen.getByRole("button", { name: "Apply" }));
  fireEvent.click(screen.getByRole("button", { name: "Reset" }));

  assert.deepEqual(actions, [
    { protocol_version: 1, type: "set_value", path: "count", value: 4 },
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
