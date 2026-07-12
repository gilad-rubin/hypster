const COMM_TIMEOUT_MS = 30_000;

function delay(milliseconds) {
  return new Promise((resolve) => setTimeout(resolve, milliseconds));
}

async function waitFor(description, predicate) {
  const deadline = Date.now() + COMM_TIMEOUT_MS;
  while (Date.now() < deadline) {
    const value = predicate();
    if (value) {
      return value;
    }
    await delay(50);
  }
  throw new Error(`${description} did not appear within ${COMM_TIMEOUT_MS}ms`);
}

async function exerciseWidget() {
  const root = await waitFor(".hypster-widget", () => document.querySelector(".hypster-widget"));
  const prematureNumeric = document.querySelector(
    "input[data-path='remote.temperature'][data-kind='float']",
  );
  if (prematureNumeric) {
    throw new Error("dependent numeric control existed before the branch action");
  }

  const beforeBranch = root.innerHTML;
  const trigger = root.querySelector(
    ".hypster-choice[data-path='mode'] .hypster-choice-trigger",
  );
  if (!trigger) {
    throw new Error("mode branch trigger was not rendered");
  }
  trigger.click();

  const remoteOption = await waitFor("remote branch option", () =>
    [...document.querySelectorAll("[data-hypster-choice-option][data-path='mode']")].find(
      (element) => element.textContent?.trim() === "remote",
    ),
  );
  remoteOption.click();

  const numeric = await waitFor("dependent numeric control", () =>
    document.querySelector("input[data-path='remote.temperature'][data-kind='float']"),
  );
  if (numeric.value !== "0.25") {
    throw new Error(`dependent numeric default was ${numeric.value}, expected 0.25`);
  }
  if (root.innerHTML === beforeBranch) {
    throw new Error("branch action did not publish replacement renderer state");
  }

  numeric.value = "1.25";
  numeric.dispatchEvent(new Event("change", { bubbles: true }));
  await waitFor("detached prior numeric control", () => !numeric.isConnected);
  const replacement = await waitFor("replacement numeric control", () => {
    const candidate = document.querySelector(
      "input[data-path='remote.temperature'][data-kind='float']",
    );
    return candidate && candidate !== numeric ? candidate : undefined;
  });
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
    numericBefore: "0.25",
    numericAfter: replacement.value,
    numericNodeReplaced: !numeric.isConnected,
  };
}

export function activate(context) {
  if (!context.onDidReceiveMessage || !context.postMessage) {
    throw new Error("NotebookRendererMessaging is unavailable");
  }
  context.onDidReceiveMessage(async (message) => {
    if (!message || message.kind !== "hypster-probe-exercise") {
      return;
    }
    try {
      const evidence = await exerciseWidget();
      context.postMessage({
        kind: "hypster-probe-result",
        requestId: message.requestId,
        ok: true,
        evidence,
      });
    } catch (error) {
      context.postMessage({
        kind: "hypster-probe-result",
        requestId: message.requestId,
        ok: false,
        error: error instanceof Error ? `${error.message}\n${error.stack}` : String(error),
      });
    }
  });
  return {};
}
