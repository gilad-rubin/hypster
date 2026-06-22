import type {
  ConditionNode,
  FieldSpec,
  Group,
  Leaf,
  RuleSchema,
  RulesFieldSpec,
  RulesMetadata,
} from "./types.js";

// ---------------------------------------------------------------------------
// Condition tree helpers
// ---------------------------------------------------------------------------

export function isGroup(node: ConditionNode): node is Group {
  return (node as Group).combinator !== undefined;
}

export function emptyGroup(combinator: Group["combinator"] = "and"): Group {
  return { combinator, rules: [] };
}

export function fieldById(
  schema: RuleSchema,
  id: string,
): FieldSpec | undefined {
  return schema.fields.find((f) => f.id === id);
}

export function defaultValueFor(field: FieldSpec): unknown {
  if (field.type === "bool") return true;
  if (field.type === "multi_enum") return [];
  if (field.type === "number") return 0;
  return field.options?.[0] ?? "";
}

export function newLeaf(schema: RuleSchema): Leaf {
  const field = schema.fields[0];
  if (!field) return { field: "", operator: "=", value: "" };
  const operator = (schema.operators[field.type] ?? ["="])[0] ?? "=";
  return { field: field.id, operator, value: defaultValueFor(field) };
}

export function shouldShowGroupCombinator(group: Group): boolean {
  return group.rules.length >= 2;
}

// ---------------------------------------------------------------------------
// RulesMetadata → RuleSchema conversion
// ---------------------------------------------------------------------------

const FIELD_TYPE_MAP: Record<string, string> = {
  select: "enum",
  multi_select: "multi_enum",
  bool: "bool",
  int: "number",
  float: "number",
  text: "string",
  multi_bool: "bool",
  multi_int: "number",
  multi_float: "number",
  multi_text: "string",
};

export function toRuleSchema(meta: RulesMetadata): RuleSchema {
  const fields = meta.field_specs.map((spec: RulesFieldSpec) => ({
    id: spec.name,
    label: spec.description || spec.name,
    type: (FIELD_TYPE_MAP[spec.type] || "string") as FieldSpec["type"],
    options: spec.options,
  }));
  const operators: Record<string, string[]> = {};
  for (const spec of meta.field_specs) {
    const mappedType = FIELD_TYPE_MAP[spec.type] || "string";
    operators[spec.name] = spec.operators;
    if (!operators[mappedType]) operators[mappedType] = spec.operators;
  }
  const combinators = meta.combinators.length > 0 ? meta.combinators : undefined;
  return { fields, operators, combinators };
}

// ---------------------------------------------------------------------------
// Condition ↔ serialized-dict conversion
// ---------------------------------------------------------------------------

export function conditionToGroup(when: Record<string, unknown>): Group {
  if (!when || Object.keys(when).length === 0) return emptyGroup();
  if ("field" in when) {
    return {
      combinator: "and",
      rules: [
        {
          field: when.field as string,
          operator: when.operator as string,
          value: when.value,
        },
      ],
    };
  }
  const combinator = (when.combinator as string) || "and";
  const children = (when.conditions ||
    when.rules ||
    []) as Array<Record<string, unknown>>;
  return {
    combinator: combinator as Group["combinator"],
    rules: children.map((child) => {
      if ("combinator" in child) return conditionToGroup(child);
      return {
        field: child.field as string,
        operator: child.operator as string,
        value: child.value,
      };
    }),
  };
}

export function groupToCondition(
  group: Group,
): Record<string, unknown> {
  return {
    combinator: group.combinator,
    conditions: group.rules.map((child) => {
      if ("combinator" in child)
        return groupToCondition(child as Group);
      return {
        field: child.field,
        operator: child.operator,
        value: child.value,
      };
    }),
  };
}

// ---------------------------------------------------------------------------
// Summary helpers (collapsed rule display)
// ---------------------------------------------------------------------------

export function summarizeThen(
  then: unknown,
  specs: RulesMetadata["then_specs"],
): string {
  if (!then || typeof then !== "object" || Array.isArray(then))
    return "(empty)";
  const obj = then as Record<string, unknown>;
  const parts = specs.map((spec) => {
    const name = spec.name ?? spec.type;
    const val = obj[name];
    if (spec.type === "bool")
      return `${name} = ${val ? "true" : "false"}`;
    const s = String(val ?? "");
    if (!s) return `${name} = (empty)`;
    return `${name} = "${s.length > 40 ? s.slice(0, 40) + "..." : s}"`;
  });
  return parts.join("  ·  ");
}

export function summarizeWhen(when: Record<string, unknown>): string {
  if ("field" in when) {
    return `${when.field} ${when.operator} ${JSON.stringify(when.value)}`;
  }
  const children = (when.conditions ||
    when.rules ||
    []) as Array<Record<string, unknown>>;
  if (children.length === 0) return "Always";
  const combinator = (when.combinator as string) || "and";
  if (combinator === "not") {
    return `NOT (${children.map((c) => summarizeWhen(c)).join(", ")})`;
  }
  const parts = children.map((child) => {
    if ("combinator" in child) return `(${summarizeWhen(child)})`;
    return `${child.field} ${child.operator} ${JSON.stringify(child.value)}`;
  });
  return parts.join(combinator === "or" ? " OR " : " AND ");
}

// ---------------------------------------------------------------------------
// Then-value helpers (single vs multi then-spec)
// ---------------------------------------------------------------------------

export function defaultThenValue(meta: RulesMetadata): unknown {
  const multiThen = meta.then_specs.length > 1;
  if (!multiThen) {
    const spec = meta.then_specs[0];
    if (!spec) return "";
    if (spec.type === "bool") return false;
    if (spec.type === "select") return spec.options?.[0] ?? "";
    return "";
  }
  const obj: Record<string, unknown> = {};
  for (const spec of meta.then_specs) {
    if (spec.type === "bool") obj[spec.name ?? spec.type] = false;
    else if (spec.type === "select")
      obj[spec.name ?? spec.type] = spec.options?.[0] ?? "";
    else obj[spec.name ?? spec.type] = "";
  }
  return obj;
}

export function getThenField(
  then: unknown,
  fieldName: string,
): unknown {
  if (then && typeof then === "object" && !Array.isArray(then)) {
    return (then as Record<string, unknown>)[fieldName];
  }
  return undefined;
}

export function singleThenValue(
  then: unknown,
  meta: RulesMetadata,
): unknown {
  if (
    meta.then_specs.length <= 1 &&
    then &&
    typeof then === "object" &&
    !Array.isArray(then)
  ) {
    const key = meta.then_specs[0]?.name;
    return key ? (then as Record<string, unknown>)[key] : then;
  }
  return then;
}

export function setThenField(
  then: unknown,
  fieldName: string,
  value: unknown,
): unknown {
  const current =
    then && typeof then === "object" && !Array.isArray(then)
      ? (then as Record<string, unknown>)
      : {};
  return { ...current, [fieldName]: value };
}
