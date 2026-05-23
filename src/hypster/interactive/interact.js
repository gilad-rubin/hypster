function controlValue(input) {
  if (input instanceof HTMLSelectElement && input.multiple) {
    return Array.from(input.selectedOptions, (option) => option.value);
  }
  if (input.type === "checkbox") {
    return input.checked;
  }
  if (input.dataset.kind === "int") {
    return Number.parseInt(input.value, 10);
  }
  if (input.dataset.kind === "float") {
    return Number.parseFloat(input.value);
  }
  return input.value;
}

function send(model, action) {
  model.set("action", { id: crypto.randomUUID(), ...action });
  model.save_changes();
}

function labelFor(parameter) {
  return parameter.display_label || parameter.name;
}

function renderParameter(model, parameter) {
  if (parameter.kind === "group") {
    const group = document.createElement("fieldset");
    group.className = "hypster-group";

    const legend = document.createElement("legend");
    legend.textContent = labelFor(parameter);
    group.append(legend);

    if (parameter.description) {
      const description = document.createElement("p");
      description.className = "hypster-description";
      description.textContent = parameter.description;
      group.append(description);
    }

    for (const child of parameter.children || []) {
      group.append(renderParameter(model, child));
    }
    return group;
  }

  const field = document.createElement("label");
  field.className = "hypster-field";

  const header = document.createElement("span");
  header.className = "hypster-field-header";
  header.textContent = labelFor(parameter);
  field.append(header);

  if (parameter.description) {
    const description = document.createElement("span");
    description.className = "hypster-description";
    description.textContent = parameter.description;
    field.append(description);
  }

  field.append(renderControl(model, parameter));
  return field;
}

function renderControl(model, parameter) {
  if (parameter.kind === "select" || parameter.kind === "multi_select") {
    const select = document.createElement("select");
    select.dataset.path = parameter.path;
    select.dataset.kind = parameter.kind;
    select.multiple = parameter.kind === "multi_select";
    for (const optionValue of parameter.options || []) {
      const option = document.createElement("option");
      option.value = optionValue;
      option.textContent = String(optionValue);
      option.selected = Array.isArray(parameter.selected_value)
        ? parameter.selected_value.includes(optionValue)
        : optionValue === parameter.selected_value;
      select.append(option);
    }
    return select;
  }

  if (parameter.kind === "bool") {
    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.checked = Boolean(parameter.selected_value);
    checkbox.dataset.path = parameter.path;
    checkbox.dataset.kind = parameter.kind;
    return checkbox;
  }

  const input = document.createElement("input");
  input.dataset.path = parameter.path;
  input.dataset.kind = parameter.kind;
  input.value = parameter.selected_value ?? "";

  if (parameter.kind === "int" || parameter.kind === "float") {
    input.type = "number";
    if (parameter.minimum != null) input.min = parameter.minimum;
    if (parameter.maximum != null) input.max = parameter.maximum;
    if (parameter.kind === "float") input.step = "any";
    return input;
  }

  input.type = "text";
  return input;
}

function renderStatus(snapshot) {
  const status = document.createElement("div");
  status.className = `hypster-status hypster-status-${snapshot.status}`;
  status.textContent = snapshot.status === "pending" ? "Pending changes" : "Up to date";

  if (snapshot.error) {
    status.textContent = snapshot.error.message;
  }

  return status;
}

function render(model, el) {
  const snapshot = model.get("snapshot");
  el.replaceChildren();

  const root = document.createElement("div");
  root.className = "hypster-widget";

  const title = document.createElement("div");
  title.className = "hypster-title";
  title.textContent = snapshot.schema?.name || "Hypster";
  root.append(title);
  root.append(renderStatus(snapshot));

  for (const parameter of snapshot.schema?.parameters || []) {
    root.append(renderParameter(model, parameter));
  }

  const actions = document.createElement("div");
  actions.className = "hypster-actions";

  const reset = document.createElement("button");
  reset.type = "button";
  reset.textContent = "Reset";
  reset.addEventListener("click", () => send(model, { type: "reset" }));
  actions.append(reset);

  if (snapshot.mode?.auto_apply === false) {
    const apply = document.createElement("button");
    apply.type = "button";
    apply.textContent = "Apply";
    apply.disabled = snapshot.status === "draft_error";
    apply.addEventListener("click", () => send(model, { type: "apply" }));
    actions.append(apply);
  }

  root.append(actions);
  el.append(root);
}

export default {
  render({ model, el }) {
    el.addEventListener("change", (event) => {
      const target = event.target;
      if (!(target instanceof HTMLInputElement || target instanceof HTMLSelectElement)) return;
      if (!target.dataset.path) return;

      send(model, {
        type: "set_value",
        path: target.dataset.path,
        value: controlValue(target),
      });
    });

    model.on("change:snapshot", () => render(model, el));
    render(model, el);
  },
};
