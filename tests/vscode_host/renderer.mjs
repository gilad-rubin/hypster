import { detectWidgetRootTransition } from "./renderer-witness.mjs";

const EXERCISE_TIMEOUT_MS = 20_000;
const DIAGNOSTIC_TEXT_LIMIT = 20_000;
const BASE_RENDERER_ID = "jupyter-ipywidget-renderer";
const browserEvents = [];
const moduleLoadedAt = new Date().toISOString();
const moduleLoadedPerformance = performance.now();

function diagnosticValue(value) {
  if (value instanceof Error) {
    return {
      name: value.name,
      message: value.message,
      stack: value.stack ?? null,
    };
  }
  if (typeof value === "string" || value == null) {
    return value;
  }
  try {
    return JSON.parse(JSON.stringify(value));
  } catch {
    try {
      return String(value);
    } catch {
      return "<unserializable>";
    }
  }
}

function recordBrowserEvent(event) {
  browserEvents.push({
    millisecondsSinceModuleLoad: Math.round(performance.now() - moduleLoadedPerformance),
    ...event,
  });
  if (browserEvents.length > 100) {
    browserEvents.shift();
  }
}

window.addEventListener(
  "error",
  (event) => {
    if (event.target && event.target !== window) {
      recordBrowserEvent({
        kind: "resource-error",
        tagName: event.target.tagName ?? null,
        id: event.target.id ?? null,
        className: event.target.className ?? null,
        src: event.target.src ?? event.target.href ?? null,
      });
      return;
    }
    recordBrowserEvent({
      kind: "window-error",
      message: event.message ?? null,
      filename: event.filename ?? null,
      line: event.lineno ?? null,
      column: event.colno ?? null,
      error: diagnosticValue(event.error),
    });
  },
  true,
);
window.addEventListener("unhandledrejection", (event) => {
  recordBrowserEvent({
    kind: "unhandled-rejection",
    reason: diagnosticValue(event.reason),
  });
});

for (const channel of ["error", "warn"]) {
  const original = console[channel].bind(console);
  console[channel] = (...values) => {
    recordBrowserEvent({
      kind: `console-${channel}`,
      values: values.map(diagnosticValue),
    });
    original(...values);
  };
}

function delay(milliseconds) {
  return new Promise((resolve) => setTimeout(resolve, milliseconds));
}

async function waitFor(description, predicate, deadline) {
  while (Date.now() < deadline) {
    const value = predicate();
    if (value) {
      return value;
    }
    await delay(50);
  }
  throw new Error(`${description} did not appear within ${EXERCISE_TIMEOUT_MS}ms`);
}

function objectKeys(value) {
  return value && (typeof value === "object" || typeof value === "function")
    ? Object.keys(value).sort()
    : [];
}

function limitedText(value) {
  if (!value) {
    return "";
  }
  return value.length <= DIAGNOSTIC_TEXT_LIMIT
    ? value
    : `${value.slice(0, DIAGNOSTIC_TEXT_LIMIT)}<truncated>`;
}

function amdDiagnostics() {
  const requireJs = globalThis.requirejs;
  const context = requireJs?.s?.contexts?._;
  const definedModules = objectKeys(context?.defined);
  const registryModules = objectKeys(context?.registry);
  const fetchedUrls = objectKeys(context?.urlFetched);
  const relevant = (values) =>
    values.filter(
      (value) => value.includes("anywidget") || value.includes("jupyter-widgets"),
    );
  let anywidgetDefined = null;
  let anywidgetSpecified = null;
  try {
    anywidgetDefined =
      typeof requireJs?.defined === "function" ? requireJs.defined("anywidget") : null;
    anywidgetSpecified =
      typeof requireJs?.specified === "function" ? requireJs.specified("anywidget") : null;
  } catch (error) {
    return {
      requireJsType: typeof requireJs,
      inspectionError: diagnosticValue(error),
    };
  }
  return {
    requireJsType: typeof requireJs,
    anywidgetDefined,
    anywidgetSpecified,
    relevantDefinedModules: relevant(definedModules),
    relevantRegistryModules: relevant(registryModules),
    relevantFetchedUrls: relevant(fetchedUrls),
    definedModuleCount: definedModules.length,
    registryModuleCount: registryModules.length,
  };
}

function elementSummaries(selector) {
  return [...document.querySelectorAll(selector)].slice(0, 20).map((element) => ({
    tagName: element.tagName,
    id: element.id || null,
    className: typeof element.className === "string" ? element.className : null,
    text: limitedText(element.textContent?.trim() ?? "").slice(0, 500),
    html: limitedText(element.outerHTML).slice(0, 2_000),
  }));
}

function rendererDiagnostics(context = {}) {
  const baseRendererContext = globalThis.jupyter_vscode_rendererContext;
  const ipywidgetsKernel = globalThis.ipywidgetsKernel;
  return {
    moduleLoadedAt,
    capturedAt: new Date().toISOString(),
    document: {
      readyState: document.readyState,
      visibilityState: document.visibilityState,
      hasFocus: document.hasFocus(),
      locationProtocol: location.protocol,
      viewport: { width: innerWidth, height: innerHeight },
      bodyChildCount: document.body?.children.length ?? null,
      bodyText: limitedText(document.body?.innerText ?? ""),
      bodyHtml: limitedText(document.body?.innerHTML ?? ""),
      outputElements: elementSummaries(
        ".cell-output-ipywidget-background, .widget-subarea, .vscode-cell-output, [data-output-id]",
      ),
      hypsterWidgetCount: document.querySelectorAll(".hypster-widget").length,
      scripts: [...document.scripts].map((script) => script.src || "<inline>").slice(0, 100),
      contentSecurityPolicy:
        document.querySelector("meta[http-equiv='Content-Security-Policy']")?.content ?? null,
    },
    amd: amdDiagnostics(),
    inheritedRenderer: {
      baseRendererContextPresent: Boolean(baseRendererContext),
      baseRendererContextKeys: objectKeys(baseRendererContext),
      ipywidgetsKernelPresent: Boolean(ipywidgetsKernel),
      ipywidgetsKernelKeys: objectKeys(ipywidgetsKernel),
      widgets7Present: Boolean(globalThis.vscIPyWidgets7),
      widgets8Present: Boolean(globalThis.vscIPyWidgets8),
    },
    browserEvents: [...browserEvents],
    ...context,
  };
}

function safeRendererDiagnostics(context) {
  try {
    return rendererDiagnostics(context);
  } catch (error) {
    return {
      collectionError: diagnosticValue(error),
      browserEvents: [...browserEvents],
      ...context,
    };
  }
}

async function probeBlobModuleImport(deadline) {
  let url;
  try {
    url = URL.createObjectURL(
      new Blob(["export default 'hypster-blob-module-ok';"], { type: "text/javascript" }),
    );
    const importResult = import(url).then(
      (module) => ({ kind: "loaded", module }),
      (error) => ({ kind: "error", error }),
    );
    const remaining = Math.max(0, Math.min(2_000, deadline - Date.now()));
    const result = await Promise.race([
      importResult,
      delay(remaining).then(() => ({ kind: "timeout" })),
    ]);
    if (result.kind === "loaded") {
      return { ok: result.module.default === "hypster-blob-module-ok" };
    }
    if (result.kind === "error") {
      return { ok: false, error: diagnosticValue(result.error) };
    }
    return { ok: false, timeoutMilliseconds: remaining };
  } catch (error) {
    return { ok: false, probeError: diagnosticValue(error) };
  } finally {
    if (url) {
      try {
        URL.revokeObjectURL(url);
      } catch {
        // The capability result above is more useful than a cleanup-only error.
      }
    }
  }
}

async function exerciseWidget(deadline) {
  const root = await waitFor(
    ".hypster-widget",
    () => document.querySelector(".hypster-widget"),
    deadline,
  );
  const prematureNumeric = document.querySelector(
    "input[data-path='remote.temperature'][data-kind='float']",
  );
  if (prematureNumeric) {
    throw new Error("dependent numeric control existed before the branch action");
  }

  const trigger = root.querySelector(
    ".hypster-choice[data-path='mode'] .hypster-choice-trigger",
  );
  if (!trigger) {
    throw new Error("mode branch trigger was not rendered");
  }
  trigger.click();

  const remoteOption = await waitFor(
    "remote branch option",
    () =>
      [...document.querySelectorAll("[data-hypster-choice-option][data-path='mode']")].find(
        (element) => element.textContent?.trim() === "remote",
      ),
    deadline,
  );
  const branchRoot = document.querySelector(".hypster-widget");
  if (!branchRoot?.isConnected) {
    throw new Error("connected widget root disappeared before the branch action");
  }
  const beforeBranch = branchRoot.innerHTML;
  remoteOption.click();

  const rootTransition = await waitFor(
    "replacement renderer state",
    () =>
      detectWidgetRootTransition(
        branchRoot,
        beforeBranch,
        document.querySelector(".hypster-widget"),
      ),
    deadline,
  );
  const currentRoot = rootTransition.currentRoot;
  const numeric = await waitFor(
    "dependent numeric control",
    () => currentRoot.querySelector("input[data-path='remote.temperature'][data-kind='float']"),
    deadline,
  );
  if (numeric.value !== "0.25") {
    throw new Error(`dependent numeric default was ${numeric.value}, expected 0.25`);
  }
  numeric.value = "1.25";
  numeric.dispatchEvent(new Event("change", { bubbles: true }));
  await waitFor("detached prior numeric control", () => !numeric.isConnected, deadline);
  const replacement = await waitFor(
    "replacement numeric control",
    () => {
      const candidate = document.querySelector(
        "input[data-path='remote.temperature'][data-kind='float']",
      );
      return candidate && candidate !== numeric ? candidate : undefined;
    },
    deadline,
  );
  if (replacement.value !== "1.25") {
    throw new Error(`replacement numeric value was ${replacement.value}, expected 1.25`);
  }

  const widgetErrors = [...document.querySelectorAll(
    ".hypster-status-error, .hypster-status-draft_error",
  )].map((element) => element.textContent);
  if (widgetErrors.length) {
    throw new Error(`widget surfaced errors: ${JSON.stringify(widgetErrors)}`);
  }

  return {
    branch: "remote",
    branchRootTransition: rootTransition.kind,
    branchRootReplaced: rootTransition.kind === "replaced",
    priorBranchRootDetached: !branchRoot.isConnected,
    numericBefore: "0.25",
    numericAfter: replacement.value,
    numericNodeReplaced: !numeric.isConnected,
  };
}

export async function activate(context) {
  if (!context.onDidReceiveMessage || !context.postMessage) {
    throw new Error("NotebookRendererMessaging is unavailable");
  }
  const baseRenderer = await context.getRenderer(BASE_RENDERER_ID);
  if (!baseRenderer || typeof baseRenderer.renderOutputItem !== "function") {
    throw new Error(`extended renderer could not resolve ${BASE_RENDERER_ID}`);
  }
  context.postMessage({
    kind: "hypster-probe-ready",
    evidence: safeRendererDiagnostics({
      phase: "activate",
      baseRendererId: BASE_RENDERER_ID,
      baseRendererApiKeys: objectKeys(baseRenderer),
    }),
  });
  context.onDidReceiveMessage(async (message) => {
    if (!message || message.kind !== "hypster-probe-exercise") {
      return;
    }
    const deadline = Date.now() + EXERCISE_TIMEOUT_MS;
    const diagnosticContext = {
      creationCellIndex: message.creationCellIndex ?? null,
      hostCreationCellVisible: message.creationCellVisible ?? null,
      blobModuleImport: await probeBlobModuleImport(deadline),
    };
    try {
      const evidence = await exerciseWidget(deadline);
      context.postMessage({
        kind: "hypster-probe-result",
        requestId: message.requestId,
        ok: true,
        evidence: {
          ...evidence,
          diagnostics: safeRendererDiagnostics(diagnosticContext),
        },
      });
    } catch (error) {
      context.postMessage({
        kind: "hypster-probe-result",
        requestId: message.requestId,
        ok: false,
        error: error instanceof Error ? `${error.message}\n${error.stack}` : String(error),
        diagnostics: safeRendererDiagnostics(diagnosticContext),
      });
    }
  });
  return {};
}
