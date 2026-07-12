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
