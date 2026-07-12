import assert from "node:assert/strict";
import { test } from "node:test";

import { JSDOM } from "jsdom";

const dom = new JSDOM("<!doctype html><html><body></body></html>");
globalThis.document = dom.window.document;
globalThis.Element = dom.window.Element;
globalThis.HTMLElement = dom.window.HTMLElement;
globalThis.HTMLInputElement = dom.window.HTMLInputElement;
globalThis.HTMLSelectElement = dom.window.HTMLSelectElement;
globalThis.HTMLTextAreaElement = dom.window.HTMLTextAreaElement;

const { default: notebookRenderer } = await import("../../../src/hypster/interactive/interact.js");

test("notebook renderer exposes no action surface for missing or mismatched snapshots", () => {
  for (const snapshot of [{ protocol_version: 2 }, {}]) {
    const actions = [];
    const model = {
      get(name) {
        assert.equal(name, "snapshot");
        return snapshot;
      },
      set(name, action) {
        if (name === "action") actions.push(action);
      },
      save_changes() {},
      on() {},
      off() {},
    };
    const element = document.createElement("div");
    const cleanup = notebookRenderer.render({ model, el: element });

    assert.match(element.querySelector('[role="alert"]').textContent, /protocol version mismatch/i);
    assert.equal(element.querySelector("button, input, select, textarea, [data-path]"), null);
    element.click();
    assert.deepEqual(actions, []);

    cleanup();
  }
});
