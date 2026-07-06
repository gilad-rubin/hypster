import { useState } from "react";
import type { SchemaFieldValue } from "./types.js";

export interface UseSchemaFieldOptions {
  fields: SchemaFieldValue[];
  onChange: (fields: SchemaFieldValue[]) => void;
}

export interface UseSchemaFieldReturn {
  editingIndex: number | null;
  setEditingIndex: (i: number | null) => void;
  addField: () => void;
  updateField: (index: number, updated: SchemaFieldValue) => void;
  removeField: (index: number) => void;
  moveField: (from: number, to: number) => void;
}

const DEFAULT_FIELD: SchemaFieldValue = {
  key: "",
  value_type: "text",
  description: "",
  multi_valued: false,
};

export function useSchemaField({
  fields,
  onChange,
}: UseSchemaFieldOptions): UseSchemaFieldReturn {
  const [editingIndex, setEditingIndex] = useState<number | null>(
    fields.length > 0 ? 0 : null
  );

  const addField = () => {
    const next = [
      ...fields,
      { ...DEFAULT_FIELD, key: `field_${fields.length + 1}` },
    ];
    onChange(next);
    setEditingIndex(next.length - 1);
  };

  const updateField = (index: number, updated: SchemaFieldValue) => {
    onChange(fields.map((f, i) => (i === index ? updated : f)));
  };

  const removeField = (index: number) => {
    onChange(fields.filter((_, i) => i !== index));
    if (editingIndex === index) setEditingIndex(null);
    else if (editingIndex !== null && editingIndex > index)
      setEditingIndex(editingIndex - 1);
  };

  const moveField = (from: number, to: number) => {
    if (from === to) return;
    const next = [...fields];
    const moved = next.splice(from, 1)[0];
    if (!moved) return;
    next.splice(to, 0, moved);
    onChange(next);
    if (editingIndex === from) setEditingIndex(to);
  };

  return {
    editingIndex,
    setEditingIndex,
    addField,
    updateField,
    removeField,
    moveField,
  };
}
