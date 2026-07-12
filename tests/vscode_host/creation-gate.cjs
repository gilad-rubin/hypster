"use strict";

function classifyMissingCreationMarker({
  selectorCommand,
  creationCommand,
  executionSummary,
  outputCount,
  rawOutput,
}) {
  const runtimeEvidence = [];
  if (selectorCommand.status !== "fulfilled") {
    runtimeEvidence.push(`selector command ${selectorCommand.status}`);
  }
  if (creationCommand.status !== "fulfilled") {
    runtimeEvidence.push(`creation command ${creationCommand.status}`);
  }
  if (executionSummary !== null) {
    runtimeEvidence.push("creation cell has an execution summary");
  }
  if (outputCount !== 0 || rawOutput.length !== 0) {
    runtimeEvidence.push(`creation cell has ${outputCount} output(s)`);
  }

  if (runtimeEvidence.length === 0) {
    return {
      accepted: true,
      outcome: "kernel_selection_gate_failure",
      reason:
        "Kernel selection and execution commands fulfilled, but the creation cell never executed.",
    };
  }
  return {
    accepted: false,
    outcome: "runtime_failure",
    reason: `Creation marker missing after runtime activity: ${runtimeEvidence.join("; ")}.`,
  };
}

module.exports = { classifyMissingCreationMarker };
