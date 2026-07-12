export type InteractiveScalar = string | number | boolean | null;

export type InteractiveValue =
  | InteractiveScalar
  | readonly InteractiveValue[]
  | { readonly [key: string]: InteractiveValue };

export type InteractiveValues = Readonly<Record<string, InteractiveValue>>;

export type FieldSpec = {
  readonly type: string;
  readonly name?: string;
  readonly description?: string;
  readonly options?: readonly InteractiveValue[];
  readonly multiline?: boolean;
  readonly operators: readonly string[];
};

export type RulesMetadata = {
  readonly field_specs: readonly FieldSpec[];
  readonly then_specs: readonly FieldSpec[];
  readonly combinators: readonly string[];
  readonly [key: string]: InteractiveValue;
};

export type SchemaField = {
  readonly key: string;
  readonly value_type: "text" | "enum" | "number" | "date";
  readonly description?: string;
  readonly label?: string;
  readonly multi_valued?: boolean;
  readonly possible_values?: readonly string[];
  readonly unit?: string;
  readonly required?: boolean;
};

export type SchemaMetadata = {
  readonly schema_fields: readonly SchemaField[];
  readonly [key: string]: InteractiveValue;
};

export type InteractiveParameterKind =
  | "select"
  | "multi_select"
  | "bool"
  | "int"
  | "float"
  | "text"
  | "multi_bool"
  | "multi_int"
  | "multi_float"
  | "multi_text"
  | "group"
  | "rules"
  | "schema";

type ParameterBase = {
  readonly name: string;
  readonly path: string;
  readonly default_value: InteractiveValue;
  readonly selected_value: InteractiveValue;
  readonly options: readonly InteractiveValue[] | null;
  readonly minimum: number | null;
  readonly maximum: number | null;
  readonly description: string | null;
  readonly display_label: string;
  readonly children: readonly InteractiveParameter[];
};

export type InteractiveParameter =
  | (ParameterBase & {
      readonly kind: Exclude<InteractiveParameterKind, "group" | "rules" | "schema">;
      readonly metadata?: Readonly<Record<string, InteractiveValue>>;
    })
  | (ParameterBase & {
      readonly kind: "group";
      readonly metadata?: Readonly<Record<string, InteractiveValue>>;
    })
  | (ParameterBase & {
      readonly kind: "rules";
      readonly metadata: RulesMetadata;
    })
  | (ParameterBase & {
      readonly kind: "schema";
      readonly metadata: SchemaMetadata;
    });

export type InteractiveSchema = {
  readonly name: string;
  readonly display_label: string;
  readonly parameters: readonly InteractiveParameter[];
};

export type InteractiveStatus = "applied" | "pending" | "draft_error" | "error";

export type InteractiveError = {
  readonly kind: string;
  readonly message: string;
};

export type InteractiveSnapshot = {
  readonly protocol_version: 1;
  readonly schema: InteractiveSchema | null;
  readonly draft_values: InteractiveValues;
  readonly applied_values: InteractiveValues;
  readonly selected_params: InteractiveValues | null;
  readonly mode: { readonly auto_apply: boolean };
  readonly status: InteractiveStatus;
  readonly error: InteractiveError | null;
};

export type InteractiveAction =
  | {
      readonly protocol_version: 1;
      readonly type: "set_value";
      readonly path: string;
      readonly value: InteractiveValue;
    }
  | { readonly protocol_version: 1; readonly type: "apply" }
  | { readonly protocol_version: 1; readonly type: "reset" };
