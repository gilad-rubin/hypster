import type { ChangeEvent, ReactNode } from "react";

import type {
  InteractiveAction,
  InteractiveParameter,
  InteractiveSnapshot,
  InteractiveValue,
} from "./types.js";

export type HypsterRendererProps = {
  readonly snapshot: InteractiveSnapshot;
  readonly onAction: (action: InteractiveAction) => void;
};

function encoded(value: InteractiveValue): string {
  return JSON.stringify(value);
}

function snapshotValue(
  snapshot: InteractiveSnapshot,
  path: string,
): { readonly found: boolean; readonly value: InteractiveValue } {
  return {
    found: Object.prototype.hasOwnProperty.call(snapshot.draft_values, path),
    value: snapshot.draft_values[path] ?? null,
  };
}

function JsonControl({
  label,
  path,
  value,
  onAction,
  context,
}: {
  readonly label: string;
  readonly path: string;
  readonly value: InteractiveValue;
  readonly onAction: (action: InteractiveAction) => void;
  readonly context?: ReactNode;
}) {
  const emit = (event: ChangeEvent<HTMLTextAreaElement>) => {
    try {
      const parsed = JSON.parse(event.currentTarget.value) as InteractiveValue;
      event.currentTarget.setCustomValidity("");
      onAction({ protocol_version: 1, type: "set_value", path, value: parsed });
    } catch {
      event.currentTarget.setCustomValidity("Enter valid JSON.");
      event.currentTarget.reportValidity();
    }
  };

  return (
    <div className="hypster-json-control">
      {context}
      <textarea
        aria-label={label}
        rows={5}
        value={JSON.stringify(value, null, 2)}
        onChange={emit}
      />
    </div>
  );
}

function ParameterControl({
  parameter,
  value,
  onAction,
}: {
  readonly parameter: InteractiveParameter;
  readonly value: InteractiveValue;
  readonly onAction: (action: InteractiveAction) => void;
}) {
  const emit = (nextValue: InteractiveValue) => {
    onAction({
      protocol_version: 1,
      type: "set_value",
      path: parameter.path,
      value: nextValue,
    });
  };

  if (parameter.kind === "select") {
    const options = parameter.options ?? [];
    return (
      <select
        aria-label={parameter.display_label}
        value={String(options.findIndex((option) => encoded(option) === encoded(value)))}
        onChange={(event) => {
          const option = options[Number(event.currentTarget.value)];
          if (option !== undefined) emit(option);
        }}
      >
        {options.map((option, index) => (
          <option key={`${index}-${encoded(option)}`} value={index}>
            {String(option)}
          </option>
        ))}
      </select>
    );
  }

  if (parameter.kind === "multi_select") {
    const options = parameter.options ?? [];
    const selected = Array.isArray(value) ? value : [];
    return (
      <select
        aria-label={parameter.display_label}
        multiple
        value={options
          .map((option, index) => (selected.some((item) => encoded(item) === encoded(option)) ? String(index) : null))
          .filter((index): index is string => index !== null)}
        onChange={(event) =>
          emit(
            Array.from(event.currentTarget.selectedOptions, (option) => options[Number(option.value)]).filter(
              (option): option is InteractiveValue => option !== undefined,
            ),
          )
        }
      >
        {options.map((option, index) => (
          <option key={`${index}-${encoded(option)}`} value={index}>
            {String(option)}
          </option>
        ))}
      </select>
    );
  }

  if (parameter.kind === "bool") {
    return (
      <input
        aria-label={parameter.display_label}
        type="checkbox"
        checked={value === true}
        onChange={(event) => emit(event.currentTarget.checked)}
      />
    );
  }

  if (parameter.kind === "int" || parameter.kind === "float") {
    return (
      <input
        aria-label={parameter.display_label}
        type="number"
        min={parameter.minimum ?? undefined}
        max={parameter.maximum ?? undefined}
        step={parameter.kind === "float" ? "any" : 1}
        value={typeof value === "number" ? value : String(value ?? "")}
        onChange={(event) => emit(event.currentTarget.value === "" ? "" : Number(event.currentTarget.value))}
      />
    );
  }

  if (parameter.kind === "text") {
    const multiline = parameter.metadata?.["multiline"] === true;
    return multiline ? (
      <textarea
        aria-label={parameter.display_label}
        rows={4}
        value={String(value ?? "")}
        onChange={(event) => emit(event.currentTarget.value)}
      />
    ) : (
      <input
        aria-label={parameter.display_label}
        type="text"
        value={String(value ?? "")}
        onChange={(event) => emit(event.currentTarget.value)}
      />
    );
  }

  if (
    parameter.kind === "multi_bool" ||
    parameter.kind === "multi_int" ||
    parameter.kind === "multi_float" ||
    parameter.kind === "multi_text"
  ) {
    return (
      <JsonControl
        label={parameter.display_label}
        path={parameter.path}
        value={value}
        onAction={onAction}
      />
    );
  }

  if (parameter.kind === "rules") {
    return (
      <JsonControl
        label={parameter.display_label}
        path={parameter.path}
        value={value}
        onAction={onAction}
        context={
          <div className="hypster-rules-vocabulary">
            Condition fields: {parameter.metadata.field_specs.map((spec) => spec.name ?? spec.type).join(", ")}
          </div>
        }
      />
    );
  }

  if (parameter.kind === "schema") {
    return (
      <JsonControl
        label={parameter.display_label}
        path={parameter.path}
        value={value}
        onAction={onAction}
        context={
          <div className="hypster-schema-fields">
            Schema fields: {parameter.metadata.schema_fields.map((field) => field.label || field.key).join(", ")}
          </div>
        }
      />
    );
  }

  return null;
}

function ParameterField({
  parameter,
  snapshot,
  onAction,
}: {
  readonly parameter: InteractiveParameter;
  readonly snapshot: InteractiveSnapshot;
  readonly onAction: (action: InteractiveAction) => void;
}) {
  if (parameter.kind === "group") {
    return (
      <fieldset className="hypster-group">
        <legend>{parameter.display_label}</legend>
        {parameter.description ? <p>{parameter.description}</p> : null}
        {parameter.children.map((child) => (
          <ParameterField key={child.path} parameter={child} snapshot={snapshot} onAction={onAction} />
        ))}
      </fieldset>
    );
  }

  const current = snapshotValue(snapshot, parameter.path);
  if (!current.found) {
    return (
      <div role="alert">
        Snapshot is missing draft_values[{JSON.stringify(parameter.path)}].
      </div>
    );
  }

  return (
    <label className="hypster-field">
      <span>{parameter.display_label}</span>
      {parameter.description ? <span>{parameter.description}</span> : null}
      <ParameterControl parameter={parameter} value={current.value} onAction={onAction} />
    </label>
  );
}

export function HypsterRenderer({ snapshot, onAction }: HypsterRendererProps) {
  return (
    <section className="hypster-renderer" data-status={snapshot.status}>
      {snapshot.error ? (
        <div role="alert">
          <strong>{snapshot.error.kind}</strong>: {snapshot.error.message}
        </div>
      ) : null}

      {snapshot.schema ? (
        <>
          <h2>{snapshot.schema.display_label}</h2>
          {snapshot.schema.parameters.map((parameter) => (
            <ParameterField
              key={parameter.path}
              parameter={parameter}
              snapshot={snapshot}
              onAction={onAction}
            />
          ))}
          <div className="hypster-actions">
            {!snapshot.mode.auto_apply ? (
              <button type="button" onClick={() => onAction({ protocol_version: 1, type: "apply" })}>
                Apply
              </button>
            ) : null}
            <button type="button" onClick={() => onAction({ protocol_version: 1, type: "reset" })}>
              Reset
            </button>
          </div>
          <output aria-label="Configuration status">{snapshot.status}</output>
        </>
      ) : snapshot.error ? null : (
        <div role="alert">Interactive snapshot has no schema.</div>
      )}
    </section>
  );
}
