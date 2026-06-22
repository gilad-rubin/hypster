import type {
  AnyConfigValue,
  ConfigNode,
  ConfigValue,
  ConfigValues,
} from "./types.js";

// ---------------------------------------------------------------------------
// Schema reconciliation — keep values whose paths still exist, fill defaults
// for new paths, check branch memory before falling back to schema defaults.
// ---------------------------------------------------------------------------

export function valuesReconcile(
  oldValues: ConfigValues,
  newSchema: ConfigNode[],
  memory?: ConfigValues,
): ConfigValues {
  const next: ConfigValues = {};
  const visit = (nodes: ConfigNode[]) => {
    for (const node of nodes) {
      if (node.kind === "group") {
        visit(node.children);
        continue;
      }
      const existing = oldValues[node.path];
      if (existing !== undefined) {
        next[node.path] = existing;
      } else {
        const remembered = memory?.[node.path];
        if (remembered !== undefined) {
          next[node.path] = remembered;
        } else {
          const fallback = node.selectedValue ?? node.defaultValue;
          if (fallback !== null) next[node.path] = fallback;
        }
      }
    }
  };
  visit(newSchema);
  return next;
}

export function findConfigNode(
  nodes: ConfigNode[],
  path: string,
): ConfigNode | null {
  for (const node of nodes) {
    if (node.path === path) return node;
    const child = findConfigNode(node.children, path);
    if (child) return child;
  }
  return null;
}

// ---------------------------------------------------------------------------
// Node presentation helpers
// ---------------------------------------------------------------------------

export function nodeLabel(node: ConfigNode): string {
  return node.displayLabel || node.name;
}

export function currentValue(
  node: ConfigNode,
  values: ConfigValues,
): AnyConfigValue | null {
  const value = values[node.path];
  if (value !== undefined) return value;
  return node.selectedValue ?? node.defaultValue;
}

// ---------------------------------------------------------------------------
// Edited-state detection
// ---------------------------------------------------------------------------

export function isEdited(
  node: ConfigNode,
  values: ConfigValues,
  baseline: ConfigValues,
): boolean {
  const value = values[node.path];
  const base = baseline[node.path];
  if (value === undefined || base === undefined) return false;
  if (node.kind === "rules") return JSON.stringify(value) !== JSON.stringify(base);
  return value !== base;
}

export function anyLeafEdited(
  node: ConfigNode,
  values: ConfigValues,
  baseline: ConfigValues,
): boolean {
  if (node.kind !== "group") return isEdited(node, values, baseline);
  return node.children.some((child) => anyLeafEdited(child, values, baseline));
}

// ---------------------------------------------------------------------------
// Summary chips for collapsed groups
// ---------------------------------------------------------------------------

export function leafSummaries(
  node: ConfigNode,
  values: ConfigValues,
): string[] {
  const summaries: string[] = [];
  const visit = (child: ConfigNode) => {
    if (child.kind === "group") {
      child.children.forEach(visit);
      return;
    }
    if (child.kind === "rules") {
      const rv = currentValue(child, values);
      const count = Array.isArray(rv) ? rv.length : 0;
      if (count > 0) summaries.push(`${count} rules`);
      return;
    }
    const value = currentValue(child, values) as ConfigValue | null;
    if (value === null || value === "" || value === false) return;
    const text =
      value === true
        ? nodeLabel(child)
        : typeof value === "number"
          ? `${nodeLabel(child)}: ${value}`
          : String(value);
    if (text.length > 24) return;
    summaries.push(text);
  };
  node.children.forEach(visit);
  return summaries;
}

// ---------------------------------------------------------------------------
// Field type inference & numeric helpers
// ---------------------------------------------------------------------------

export function isMultilineText(
  _node: ConfigNode,
  value: ConfigValue | null,
): boolean {
  const text = String(value ?? "");
  return text.includes("\n") || text.length > 80;
}

export function clamp(
  value: number,
  minimum: number | null,
  maximum: number | null,
): number {
  let next = value;
  if (minimum !== null && next < minimum) next = minimum;
  if (maximum !== null && next > maximum) next = maximum;
  return next;
}

export function countLeaves(node: ConfigNode): number {
  return node.children.reduce(
    (total, child) =>
      total + (child.kind === "group" ? countLeaves(child) : 1),
    0,
  );
}
