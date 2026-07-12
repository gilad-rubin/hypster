const NUMERIC_SELECTOR = "input[data-path='remote.temperature'][data-kind='float']";
const MODE_VALUE_SELECTOR = ".hypster-choice[data-path='mode'] .hypster-choice-value";

export function findPublishedRemoteState(documentRoot) {
  const numeric = documentRoot.querySelector(NUMERIC_SELECTOR);
  if (!numeric?.isConnected) {
    return undefined;
  }
  const currentRoot = numeric.closest(".hypster-widget");
  if (!currentRoot?.isConnected) {
    return undefined;
  }
  const mode = currentRoot.querySelector(MODE_VALUE_SELECTOR)?.textContent?.trim();
  return mode === "remote" ? { currentRoot, numeric } : undefined;
}

export function detectWidgetRootTransition(previousRoot, previousHtml, currentRoot) {
  if (!currentRoot?.isConnected) {
    return undefined;
  }
  if (currentRoot !== previousRoot) {
    if (previousRoot.isConnected) {
      return undefined;
    }
    return { kind: "replaced", currentRoot };
  }
  if (currentRoot.innerHTML !== previousHtml) {
    return { kind: "updated-in-place", currentRoot };
  }
  return undefined;
}
