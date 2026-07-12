"use strict";

const assert = require("node:assert/strict");
const { classifyMissingCreationMarker } = require("../creation-gate.cjs");

const fulfilled = { status: "fulfilled" };
const neverExecuted = {
  selectorCommand: fulfilled,
  creationCommand: fulfilled,
  executionSummary: null,
  outputCount: 0,
  rawOutput: "",
};

assert.deepEqual(classifyMissingCreationMarker(neverExecuted), {
  accepted: true,
  outcome: "kernel_selection_gate_failure",
  reason: "Kernel selection and execution commands fulfilled, but the creation cell never executed.",
});

for (const [name, falsifier] of Object.entries({
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

console.log("creation-gate classifier rejects command, execution, and output falsifiers");
