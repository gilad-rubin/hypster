"use strict";

const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const vscode = require("vscode");
const { classifyMissingCreationMarker } = require("../creation-gate.cjs");
const pins = require("../pins.cjs");

const decoder = new TextDecoder();

function requiredPath(name) {
  const value = process.env[name];
  if (!value || !path.isAbsolute(value)) {
    throw new Error(`${name} must be an explicit absolute path`);
  }
  return value;
}

function delay(milliseconds) {
  return new Promise((resolve) => setTimeout(resolve, milliseconds));
}

async function settleWithin(promise, milliseconds) {
  let timer;
  try {
    return await Promise.race([
      promise.then(
        (value) => ({ status: "fulfilled", value }),
        (error) => ({
          status: "rejected",
          error: error instanceof Error ? `${error.message}\n${error.stack}` : String(error),
        }),
      ),
      new Promise((resolve) => {
        timer = setTimeout(() => resolve({ status: "timeout", milliseconds }), milliseconds);
      }),
    ]);
  } finally {
    clearTimeout(timer);
  }
}

function outputText(cell) {
  return cell.outputs
    .flatMap((output) => output.items)
    .map((item) => {
      try {
        return decoder.decode(item.data);
      } catch (error) {
        return `[decode error for ${item.mime}: ${error}]`;
      }
    })
    .join("\n");
}

async function waitForMarker(cell, marker, milliseconds) {
  const deadline = Date.now() + milliseconds;
  while (Date.now() < deadline) {
    const text = outputText(cell);
    if (text.includes(marker)) {
      return text;
    }
    await delay(100);
  }
  throw new Error(
    `cell ${cell.index} did not emit ${JSON.stringify(marker)} within ${milliseconds}ms; ` +
      `outputs were:\n${outputText(cell)}`,
  );
}

function cellBySourceMarker(document, marker) {
  const matches = document.getCells().filter((cell) => cell.document.getText().includes(marker));
  if (matches.length !== 1) {
    throw new Error(`expected one cell containing ${JSON.stringify(marker)}, got ${matches.length}`);
  }
  return matches[0];
}

function executionSummary(cell) {
  const summary = cell.executionSummary;
  if (!summary) {
    return null;
  }
  return {
    executionOrder: summary.executionOrder ?? null,
    success: summary.success ?? null,
    timing: summary.timing
      ? {
          startTime: summary.timing.startTime,
          endTime: summary.timing.endTime,
        }
      : null,
  };
}

function recordCreationState(roundTrip, cell) {
  roundTrip.creationExecutionSummary = executionSummary(cell);
  roundTrip.creationOutputCount = cell.outputs.length;
  roundTrip.creationRawOutput = outputText(cell);
}

function classifyRecordedCreationState(evidence) {
  return classifyMissingCreationMarker({
    selectorCommand: evidence.selector.commandResult,
    creationCommand: evidence.roundTrip.creationCommand,
    executionSummary: evidence.roundTrip.creationExecutionSummary,
    outputCount: evidence.roundTrip.creationOutputCount,
    rawOutput: evidence.roundTrip.creationRawOutput,
  });
}

function extensionVersions() {
  return Object.fromEntries(
    Object.entries(pins.extensions).map(([id, expected]) => {
      const extension = vscode.extensions.getExtension(id);
      if (!extension) {
        throw new Error(`required extension ${id}@${expected} is not installed`);
      }
      const actual = extension.packageJSON.version;
      if (actual !== expected) {
        throw new Error(`required extension ${id} expected ${expected}, got ${actual}`);
      }
      return [id, actual];
    }),
  );
}

async function run() {
  const notebookPath = requiredPath("HYPSTER_NOTEBOOK");
  const artifactDir = requiredPath("HYPSTER_VSCODE_ARTIFACT_DIR");
  fs.mkdirSync(artifactDir, { recursive: true });

  const evidence = {
    platform: {
      arch: os.arch(),
      platform: os.platform(),
      release: os.release(),
    },
    processVersions: process.versions,
    extensionHostEnvironmentKeys: Object.keys(process.env).sort(),
    vscode: {
      expected: pins.vscode,
      actual: vscode.version,
      electron: process.versions.electron,
      chromium: process.versions.chrome,
    },
    extensions: extensionVersions(),
    notebookPath,
    notebooksApiKeys: Object.keys(vscode.notebooks).sort(),
    rendererBoundary: {
      id: "hypster-vscode-host-probe",
      extends: "jupyter-ipywidget-renderer",
      messaging: "NotebookRendererMessaging",
    },
    selector: {
      command: "notebook.selectKernel",
      documentedReturn: "void",
      attemptedArgument: {
        id: "hypster-host",
        extension: "ms-toolsai.jupyter",
      },
    },
    roundTrip: { attempted: false },
  };

  if (vscode.version !== pins.vscode) {
    throw new Error(`expected VS Code ${pins.vscode}, got ${vscode.version}`);
  }
  if (process.platform !== "linux") {
    throw new Error(`physical witness requires Ubuntu/Linux, got ${process.platform}`);
  }
  if (!process.versions.electron || !process.versions.chrome) {
    throw new Error("Electron or Chromium runtime version is missing from the physical witness");
  }

  let editor;
  try {
    const python = vscode.extensions.getExtension("ms-python.python");
    const jupyter = vscode.extensions.getExtension("ms-toolsai.jupyter");
    await python.activate();
    await jupyter.activate();

    const commands = await vscode.commands.getCommands(true);
    evidence.selector.commandRegistered = commands.includes("notebook.selectKernel");
    if (!evidence.selector.commandRegistered) {
      evidence.outcome = "kernel_selection_gate_failure";
      evidence.reason =
        "VS Code did not register the documented notebook.selectKernel command after Jupyter activation.";
      console.log("HYPSTER_VSCODE_KERNEL_SELECTION_GATE_REPRODUCED");
      return;
    }
    const forbiddenDiscoveryKeys = ["controllers", "getControllers", "getNotebookControllers"];
    evidence.selector.publicControllerDiscoveryKeys = forbiddenDiscoveryKeys.filter((key) =>
      Object.prototype.hasOwnProperty.call(vscode.notebooks, key),
    );
    if (evidence.selector.publicControllerDiscoveryKeys.length) {
      throw new Error(
        `public controller discovery unexpectedly appeared: ${evidence.selector.publicControllerDiscoveryKeys}`,
      );
    }

    const document = await vscode.workspace.openNotebookDocument(vscode.Uri.file(notebookPath));
    const creationCell = cellBySourceMarker(document, "HYPSTER_IMPORT_PATH=");
    const verificationCell = cellBySourceMarker(document, "HYPSTER_PARAMS_VERIFIED=");
    evidence.fixture = {
      cellCount: document.cellCount,
      creationCellIndex: creationCell.index,
      verificationCellIndex: verificationCell.index,
    };
    editor = await vscode.window.showNotebookDocument(document);
    await delay(2_000);

    evidence.selector.commandResult = await settleWithin(
      vscode.commands.executeCommand("notebook.selectKernel", {
        notebookEditor: editor,
        id: "hypster-host",
        extension: "ms-toolsai.jupyter",
      }),
      10_000,
    );

    evidence.roundTrip.attempted = true;
    evidence.roundTrip.creationCommand = await settleWithin(
      vscode.commands.executeCommand("notebook.cell.execute", {
        ranges: [{ start: creationCell.index, end: creationCell.index + 1 }],
        document: document.uri,
      }),
      30_000,
    );
    if (
      evidence.selector.commandResult.status !== "fulfilled" ||
      evidence.roundTrip.creationCommand.status !== "fulfilled"
    ) {
      recordCreationState(evidence.roundTrip, creationCell);
      const classification = classifyRecordedCreationState(evidence);
      evidence.roundTrip.creationGateClassification = classification;
      throw new Error(classification.reason);
    }

    try {
      evidence.roundTrip.creationOutput = await waitForMarker(
        creationCell,
        "HYPSTER_IMPORT_PATH=",
        20_000,
      );
      recordCreationState(evidence.roundTrip, creationCell);
    } catch (error) {
      evidence.roundTrip.creationMarkerError =
        error instanceof Error ? `${error.message}\n${error.stack}` : String(error);
      recordCreationState(evidence.roundTrip, creationCell);
      const classification = classifyRecordedCreationState(evidence);
      evidence.roundTrip.creationGateClassification = classification;
      evidence.roundTrip.completed = false;
      if (!classification.accepted) {
        throw new Error(classification.reason);
      }
      evidence.outcome = classification.outcome;
      evidence.reason = classification.reason;
      console.log("HYPSTER_VSCODE_KERNEL_SELECTION_GATE_REPRODUCED");
      return;
    }

    const probeExtension = vscode.extensions.getExtension("hypster.hypster-vscode-host-spike");
    if (!probeExtension) {
      throw new Error("test renderer extension is not registered");
    }
    const probe = await probeExtension.activate();
    evidence.roundTrip.renderer = await probe.exercise(editor);

    evidence.roundTrip.verificationCommand = await settleWithin(
      vscode.commands.executeCommand("notebook.cell.execute", {
        ranges: [{ start: verificationCell.index, end: verificationCell.index + 1 }],
        document: document.uri,
      }),
      30_000,
    );
    evidence.roundTrip.verificationOutput = await waitForMarker(
      verificationCell,
      "HYPSTER_PARAMS_VERIFIED=",
      20_000,
    );
    if (!evidence.roundTrip.verificationOutput.includes("'remote.temperature': 1.25")) {
      throw new Error(`Python oracle did not contain the falsifier value: ${evidence.roundTrip.verificationOutput}`);
    }
    evidence.roundTrip.completed = true;
    evidence.outcome = "basic_scenario_green";
    console.log("HYPSTER_VSCODE_BASIC_SCENARIO_GREEN");
  } catch (error) {
    evidence.outcome = "runtime_failure";
    evidence.roundTrip.completed = false;
    evidence.error = error instanceof Error ? `${error.message}\n${error.stack}` : String(error);
    throw error;
  } finally {
    fs.writeFileSync(
      path.join(artifactDir, "spike-result.json"),
      `${JSON.stringify(evidence, null, 2)}\n`,
    );
    if (editor) {
      await vscode.commands.executeCommand("workbench.action.closeActiveEditor");
    }
  }
  console.log(JSON.stringify(evidence, null, 2));
}

module.exports = { run };
