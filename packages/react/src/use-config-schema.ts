import { useCallback, useEffect, useRef, useState } from "react";

import { valuesReconcile } from "./config.js";
import type { AnyConfigValue, ConfigNode, ConfigValues } from "./types.js";

export type ConfigSchemaFetcher = (
  kind: string,
  values: ConfigValues,
) => Promise<{ parameters: ConfigNode[] }>;

export interface UseConfigSchemaOptions {
  fetchSchema: ConfigSchemaFetcher;
  kind: string;
  debounceMs?: number;
}

export interface UseConfigSchemaReturn {
  schema: ConfigNode[];
  values: ConfigValues;
  baseline: ConfigValues;
  loading: boolean;
  error: string | null;
  handleChange: (path: string, value: AnyConfigValue) => void;
}

export function useConfigSchema({
  fetchSchema,
  kind,
  debounceMs = 300,
}: UseConfigSchemaOptions): UseConfigSchemaReturn {
  const [schema, setSchema] = useState<ConfigNode[]>([]);
  const [values, setValues] = useState<ConfigValues>({});
  const [baseline, setBaseline] = useState<ConfigValues>({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const timer = useRef<ReturnType<typeof setTimeout>>(undefined);
  const memory = useRef<ConfigValues>({});
  const fetchRef = useRef(fetchSchema);
  fetchRef.current = fetchSchema;
  const requestId = useRef(0);

  useEffect(() => {
    clearTimeout(timer.current);
    const id = ++requestId.current;
    memory.current = {};
    const t = setTimeout(() => {
      setLoading(true);
      setError(null);
      fetchRef.current(kind, {})
        .then((next) => {
          if (id !== requestId.current) return;
          const seeded = valuesReconcile({}, next.parameters);
          setSchema(next.parameters);
          setBaseline(seeded);
          setValues(seeded);
        })
        .catch((err: unknown) => {
          if (id !== requestId.current) return;
          setError(err instanceof Error ? err.message : String(err));
        })
        .finally(() => {
          if (id === requestId.current) setLoading(false);
        });
    }, 0);
    return () => {
      requestId.current++;
      clearTimeout(t);
    };
  }, [kind]);

  const refreshSchema = useCallback(
    (nextValues: ConfigValues) => {
      clearTimeout(timer.current);
      const id = ++requestId.current;
      timer.current = setTimeout(async () => {
        setLoading(true);
        try {
          const next = await fetchRef.current(kind, nextValues);
          if (id !== requestId.current) return;
          setSchema(next.parameters);
          setValues((current) =>
            valuesReconcile(current, next.parameters, memory.current),
          );
          setError(null);
        } catch (err: unknown) {
          if (id !== requestId.current) return;
          setError(err instanceof Error ? err.message : String(err));
        } finally {
          if (id === requestId.current) setLoading(false);
        }
      }, debounceMs);
    },
    [kind, debounceMs],
  );

  const handleChange = useCallback(
    (path: string, value: AnyConfigValue) => {
      memory.current[path] = value;
      const next = { ...values, [path]: value };
      setValues(next);
      refreshSchema(next);
    },
    [values, refreshSchema],
  );

  return { schema, values, baseline, loading, error, handleChange };
}
