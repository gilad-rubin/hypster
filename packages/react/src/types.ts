// ---------------------------------------------------------------------------
// Config schema types — mirror the JSON returned by hypster's explore().
// ---------------------------------------------------------------------------

export type ConfigNodeKind = "select" | "int" | "float" | "text" | "bool" | "group" | "rules" | "schema";

export type ConfigValue = string | number | boolean;

export type RuleValue = {
  when: Record<string, unknown>;
  then: unknown;
  name?: string;
};

export type SchemaFieldValue = {
  key: string;
  value_type: "text" | "enum" | "number" | "date";
  description?: string;
  label?: string;
  multi_valued?: boolean;
  possible_values?: string[];
  unit?: string;
  required?: boolean;
};

export type SchemaMetadata = {
  schema_fields: SchemaFieldValue[];
};

export type AnyConfigValue = ConfigValue | RuleValue[] | SchemaFieldValue[];

export type ConfigValues = Record<string, AnyConfigValue>;

export type RulesFieldSpec = {
  name: string;
  type: string;
  options?: string[];
  operators: string[];
  description?: string;
  multiline?: boolean;
};

export type RulesMetadata = {
  field_specs: RulesFieldSpec[];
  then_specs: { name?: string; type: string; multiline?: boolean; options?: string[] }[];
  combinators: string[];
};

export type ConfigNode = {
  name: string;
  path: string;
  kind: ConfigNodeKind;
  defaultValue: AnyConfigValue | null;
  selectedValue: AnyConfigValue | null;
  options: ConfigValue[] | null;
  minimum: number | null;
  maximum: number | null;
  description?: string | null;
  displayLabel?: string | null;
  metadata?: RulesMetadata | Record<string, unknown> | null;
  children: ConfigNode[];
};

export type ExperimentSchema = {
  kind: string;
  parameters: ConfigNode[];
};

// ---------------------------------------------------------------------------
// Condition / rules types — the nestable condition tree grammar.
// ---------------------------------------------------------------------------

export type FieldType = "enum" | "multi_enum" | "bool" | "number" | "string";

export interface FieldSpec {
  id: string;
  label: string;
  type: FieldType;
  options?: string[];
}

export interface RuleSchema {
  fields: FieldSpec[];
  operators: Record<string, string[]>;
  combinators?: string[];
}

export interface Leaf {
  field: string;
  operator: string;
  value: unknown;
}

export interface Group {
  combinator: "and" | "or" | "not";
  rules: ConditionNode[];
}

export type ConditionNode = Leaf | Group;
