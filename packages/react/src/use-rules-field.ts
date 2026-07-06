import { useState } from "react";

import {
  conditionToGroup,
  defaultThenValue,
  groupToCondition,
  newLeaf,
  setThenField,
  toRuleSchema,
} from "./rules.js";
import type { Group, RuleSchema, RuleValue, RulesMetadata } from "./types.js";

export interface UseRulesFieldOptions {
  rules: RuleValue[];
  meta: RulesMetadata;
  onChange: (rules: RuleValue[]) => void;
}

export interface UseRulesFieldReturn {
  editingIndex: number | null;
  setEditingIndex: (i: number | null) => void;
  ruleSchema: RuleSchema;
  multiThen: boolean;
  updateRule: (index: number, updated: RuleValue) => void;
  removeRule: (index: number) => void;
  addRule: () => void;
  ruleWhen: (index: number) => Group;
  setRuleName: (index: number, name: string) => void;
  setRuleWhen: (index: number, when: Group) => void;
  setRuleThen: (index: number, then: unknown) => void;
  setRuleThenField: (index: number, fieldName: string, value: unknown) => void;
}

export function useRulesField({
  rules,
  meta,
  onChange,
}: UseRulesFieldOptions): UseRulesFieldReturn {
  const [editingIndex, setEditingIndex] = useState<number | null>(
    rules.length > 0 ? 0 : null,
  );
  const ruleSchema = toRuleSchema(meta);
  const multiThen = meta.then_specs.length > 1;

  const updateRule = (index: number, updated: RuleValue) => {
    onChange(rules.map((r, i) => (i === index ? updated : r)));
  };

  const removeRule = (index: number) => {
    onChange(rules.filter((_, i) => i !== index));
    if (editingIndex === index) setEditingIndex(null);
    else if (editingIndex !== null && editingIndex > index)
      setEditingIndex(editingIndex - 1);
  };

  const addRule = () => {
    const leaf = newLeaf(ruleSchema);
    const defaultCombinator = (ruleSchema.combinators?.[0] ?? "and") as Group["combinator"];
    const newRuleVal: RuleValue = {
      when: groupToCondition({ combinator: defaultCombinator, rules: [leaf] }),
      then: defaultThenValue(meta),
      name: `rule_${rules.length + 1}`,
    };
    const next = [...rules, newRuleVal];
    onChange(next);
    setEditingIndex(next.length - 1);
  };

  const ruleWhen = (index: number): Group =>
    conditionToGroup(rules[index]?.when ?? {});

  const setRuleName = (index: number, name: string) => {
    const rule = rules[index];
    if (!rule) return;
    onChange(rules.map((r, i) => (i === index ? { ...r, name } : r)));
  };

  const setRuleWhen = (index: number, when: Group) => {
    const rule = rules[index];
    if (!rule) return;
    onChange(
      rules.map((r, i) =>
        i === index ? { ...r, when: groupToCondition(when) } : r,
      ),
    );
  };

  const setRuleThen = (index: number, then: unknown) => {
    const rule = rules[index];
    if (!rule) return;
    onChange(rules.map((r, i) => (i === index ? { ...r, then } : r)));
  };

  const setRuleThenField = (
    index: number,
    fieldName: string,
    value: unknown,
  ) => {
    const rule = rules[index];
    if (!rule) return;
    onChange(
      rules.map((r, i) =>
        i === index
          ? { ...r, then: setThenField(r.then, fieldName, value) }
          : r,
      ),
    );
  };

  return {
    editingIndex,
    setEditingIndex,
    ruleSchema,
    multiThen,
    updateRule,
    removeRule,
    addRule,
    ruleWhen,
    setRuleName,
    setRuleWhen,
    setRuleThen,
    setRuleThenField,
  };
}
