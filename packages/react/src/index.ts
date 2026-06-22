// Types
export type {
  AnyConfigValue,
  ConditionNode,
  ConfigNode,
  ConfigNodeKind,
  ConfigValue,
  ConfigValues,
  ExperimentSchema,
  FieldSpec,
  FieldType,
  Group,
  Leaf,
  RuleSchema,
  RuleValue,
  RulesFieldSpec,
  RulesMetadata,
} from "./types.js";

// Config utilities
export {
  anyLeafEdited,
  clamp,
  countLeaves,
  currentValue,
  findConfigNode,
  isEdited,
  isMultilineText,
  leafSummaries,
  nodeLabel,
  valuesReconcile,
} from "./config.js";

// Rules utilities
export {
  conditionToGroup,
  defaultThenValue,
  defaultValueFor,
  emptyGroup,
  fieldById,
  getThenField,
  groupToCondition,
  isGroup,
  newLeaf,
  setThenField,
  shouldShowGroupCombinator,
  singleThenValue,
  summarizeThen,
  summarizeWhen,
  toRuleSchema,
} from "./rules.js";

// Hooks
export { useConfigSchema } from "./use-config-schema.js";
export type {
  ConfigSchemaFetcher,
  UseConfigSchemaOptions,
  UseConfigSchemaReturn,
} from "./use-config-schema.js";

export { useRulesField } from "./use-rules-field.js";
export type {
  UseRulesFieldOptions,
  UseRulesFieldReturn,
} from "./use-rules-field.js";
