"use strict";

const assert = require("node:assert/strict");
const {
  classifyCommandResult,
  classifyMissingCreationMarker,
  selectKernelStrategy,
} = require("../creation-gate.cjs");

const fulfilled = { status: "fulfilled" };
const neverExecuted = {
  selectorCommand: fulfilled,
  creationCommand: fulfilled,
  executionSummary: null,
  outputCount: 0,
  rawOutput: "",
};

for (const [name, falsifier] of Object.entries({
  transientEmptyState: {},
  selectorRejected: { selectorCommand: { status: "rejected", error: "unknown controller" } },
  selectorTimedOut: { selectorCommand: { status: "timeout" } },
  missingKernelspec: {
    creationCommand: { status: "rejected", error: "No kernel named hypster-host" },
  },
  creationTimedOut: { creationCommand: { status: "timeout" } },
  executionFailed: { executionSummary: { success: false } },
  executionCompleted: { executionSummary: { success: true } },
  missingKernelspecOutput: {
    executionSummary: { success: false },
    outputCount: 1,
    rawOutput: "Error: No kernel named hypster-host",
  },
  unrelatedErrorOutput: { outputCount: 1, rawOutput: "ImportError: unrelated dependency" },
  nonTextOutput: { outputCount: 1 },
})) {
  const result = classifyMissingCreationMarker({ ...neverExecuted, ...falsifier });
  assert.equal(result.accepted, false, `${name} must not be an accepted gate outcome`);
  assert.equal(result.outcome, "runtime_failure", `${name} must be red`);
}

assert.deepEqual(classifyCommandResult("verification execution", fulfilled), { ok: true });
for (const status of ["rejected", "timeout"]) {
  assert.deepEqual(classifyCommandResult("verification execution", { status }), {
    ok: false,
    outcome: "runtime_failure",
    reason: `verification execution command ${status}`,
  });
}

assert.equal(
  selectKernelStrategy({ openNotebookExported: true, commandRegistered: false }),
  "ms-toolsai.jupyter.exports.openNotebook",
  "the supported export must bypass the observed missing command",
);
assert.equal(
  selectKernelStrategy({ openNotebookExported: false, commandRegistered: true }),
  "notebook.selectKernel",
);
assert.equal(
  selectKernelStrategy({ openNotebookExported: false, commandRegistered: false }),
  "kernel_selection_gate_failure",
);

console.log(
  "host classifier rejects races and prefers the supported Jupyter export",
);
