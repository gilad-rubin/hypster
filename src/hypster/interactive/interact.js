function hasOwn(object, key) {
  return Object.prototype.hasOwnProperty.call(object || {}, key);
}

function valuesEqual(left, right) {
  return JSON.stringify(left) === JSON.stringify(right);
}

function optionValue(option) {
  if (option.dataset.value == null) return option.value;
  return JSON.parse(option.dataset.value);
}

function encodedValue(value) {
  return JSON.stringify(value);
}

function decodedValue(element) {
  if (element.dataset.value == null) return element.value;
  return JSON.parse(element.dataset.value);
}

function valueFor(snapshot, parameter) {
  if (hasOwn(snapshot.draft_values, parameter.path)) {
    return snapshot.draft_values[parameter.path];
  }
  return parameter.selected_value;
}

function encodedControlValue(input) {
  const kind = input.dataset.kind;
  if (
    kind === "int" ||
    kind === "float" ||
    ["multi_int", "multi_float", "multi_text", "multi_bool"].includes(kind)
  ) {
    return { encoded_value: { kind, value: input.value } };
  }
  return { value: controlValue(input) };
}

function actionId() {
  return globalThis.crypto?.randomUUID?.() || `${Date.now()}-${Math.random().toString(36).slice(2)}`;
}

function controlValue(input) {
  if (input instanceof HTMLSelectElement && input.multiple) {
    return Array.from(input.selectedOptions, optionValue);
  }
  if (input instanceof HTMLSelectElement) {
    const option = input.selectedOptions[0];
    return option ? optionValue(option) : null;
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
  model.set("action", { id: actionId(), ...action });
  model.save_changes();
}

function labelFor(parameter) {
  return parameter.display_label || parameter.name;
}

function closeChoiceMenus(root) {
  for (const choice of root.querySelectorAll(".hypster-choice-open")) {
    choice.classList.remove("hypster-choice-open");
    choice.querySelector(".hypster-choice-trigger")?.setAttribute("aria-expanded", "false");
    choice.querySelector(".hypster-choice-menu")?.setAttribute("hidden", "");
  }
}

function themeDocuments(baseDocument) {
  const documents = [];
  if (baseDocument) documents.push(baseDocument);

  try {
    const parentDocument = globalThis.parent?.document;
    if (parentDocument && !documents.includes(parentDocument)) documents.push(parentDocument);
  } catch {
    // Cross-origin notebook hosts can block parent access.
  }

  return documents;
}

function themeFromDocument(doc) {
  const body = doc.body;
  const html = doc.documentElement;
  if (!body || !html) return null;

  const vscodeThemeKind = body.getAttribute("data-vscode-theme-kind") || html.getAttribute("data-vscode-theme-kind");
  if (vscodeThemeKind) return vscodeThemeKind.includes("light") ? "light" : "dark";

  const classNames = `${body.className || ""} ${html.className || ""}`.toLowerCase();
  if (classNames.includes("vscode-high-contrast-light") || classNames.includes("vscode-light")) {
    return "light";
  }
  if (classNames.includes("vscode-high-contrast") || classNames.includes("vscode-dark")) {
    return "dark";
  }

  const jupyterThemeLight = body.dataset.jpThemeLight ?? html.dataset.jpThemeLight;
  if (jupyterThemeLight === "true") return "light";
  if (jupyterThemeLight === "false") return "dark";
  if (classNames.includes("jp-mod-light") || classNames.includes("jp-mod-theme-light")) return "light";
  if (classNames.includes("jp-mod-dark") || classNames.includes("jp-mod-theme-dark")) return "dark";

  const dataTheme = body.dataset.theme || html.dataset.theme || body.dataset.mode || html.dataset.mode;
  if (dataTheme === "light" || dataTheme === "dark") return dataTheme;

  return null;
}

function detectHostTheme(baseDocument) {
  for (const doc of themeDocuments(baseDocument)) {
    const theme = themeFromDocument(doc);
    if (theme) return theme;
  }

  return globalThis.matchMedia?.("(prefers-color-scheme: light)").matches ? "light" : "dark";
}

function applyHostTheme(el) {
  const root = el.querySelector(".hypster-widget");
  if (!root) return;

  const theme = detectHostTheme(el.ownerDocument || document);
  root.classList.toggle("hypster-theme-light", theme === "light");
  root.classList.toggle("hypster-theme-dark", theme === "dark");
  root.dataset.hypsterTheme = theme;
}

function watchHostTheme(el) {
  const apply = () => applyHostTheme(el);
  const observers = [];
  const watchedAttributes = ["class", "data-vscode-theme-kind", "data-jp-theme-light", "data-theme", "data-mode"];

  for (const doc of themeDocuments(el.ownerDocument || document)) {
    try {
      const Observer = doc.defaultView?.MutationObserver || globalThis.MutationObserver;
      if (!Observer) continue;
      const observer = new Observer(apply);
      observer.observe(doc.body, { attributes: true, attributeFilter: watchedAttributes });
      observer.observe(doc.documentElement, { attributes: true, attributeFilter: watchedAttributes });
      observers.push(observer);
    } catch {
      // Some hosts expose a partial or inaccessible parent document.
    }
  }

  const mediaQuery = globalThis.matchMedia?.("(prefers-color-scheme: dark)");
  mediaQuery?.addEventListener?.("change", apply);
  apply();

  return () => {
    for (const observer of observers) observer.disconnect();
    mediaQuery?.removeEventListener?.("change", apply);
  };
}

function renderParameter(model, snapshot, parameter) {
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
      group.append(renderParameter(model, snapshot, child));
    }
    return group;
  }

  const field = document.createElement("div");
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

  field.append(renderControl(snapshot, parameter));
  return field;
}

function renderControl(snapshot, parameter) {
  const currentValue = valueFor(snapshot, parameter);

  if (parameter.kind === "select") {
    return renderChoiceControl(parameter, currentValue);
  }

  if (parameter.kind === "multi_select") {
    const select = document.createElement("select");
    select.dataset.path = parameter.path;
    select.dataset.kind = parameter.kind;
    select.multiple = parameter.kind === "multi_select";
    for (const optionValue of parameter.options || []) {
      const option = document.createElement("option");
      option.value = String(select.options.length);
      option.dataset.value = encodedValue(optionValue);
      option.textContent = String(optionValue);
      option.selected = Array.isArray(currentValue)
        ? currentValue.some((selectedValue) => valuesEqual(selectedValue, optionValue))
        : valuesEqual(optionValue, currentValue);
      select.append(option);
    }
    return select;
  }

  if (parameter.kind === "bool") {
    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.checked = Boolean(currentValue);
    checkbox.dataset.path = parameter.path;
    checkbox.dataset.kind = parameter.kind;
    return checkbox;
  }

  const input = document.createElement("input");
  input.dataset.path = parameter.path;
  input.dataset.kind = parameter.kind;
  input.value = parameter.kind.startsWith("multi_") ? JSON.stringify(currentValue ?? []) : (currentValue ?? "");

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

function renderChoiceControl(parameter, currentValue) {
  const choice = document.createElement("div");
  choice.className = "hypster-choice";
  choice.dataset.path = parameter.path;
  choice.dataset.kind = parameter.kind;

  const trigger = document.createElement("button");
  trigger.type = "button";
  trigger.className = "hypster-choice-trigger";
  trigger.dataset.hypsterChoiceTrigger = "";
  trigger.setAttribute("aria-haspopup", "listbox");
  trigger.setAttribute("aria-expanded", "false");

  const value = document.createElement("span");
  value.className = "hypster-choice-value";
  value.textContent = String(currentValue ?? "");
  trigger.append(value);

  const arrow = document.createElement("span");
  arrow.className = "hypster-choice-arrow";
  arrow.setAttribute("aria-hidden", "true");
  trigger.append(arrow);

  choice.append(trigger);

  const menu = document.createElement("div");
  menu.className = "hypster-choice-menu";
  menu.setAttribute("role", "listbox");
  menu.setAttribute("hidden", "");

  for (const optionValue of parameter.options || []) {
    const option = document.createElement("button");
    option.type = "button";
    option.className = "hypster-choice-option";
    option.dataset.hypsterChoiceOption = "";
    option.dataset.path = parameter.path;
    option.dataset.value = encodedValue(optionValue);
    option.setAttribute("role", "option");
    option.setAttribute("aria-selected", valuesEqual(optionValue, currentValue) ? "true" : "false");
    option.textContent = String(optionValue);
    menu.append(option);
  }

  choice.append(menu);
  return choice;
}

function renderStatus(snapshot) {
  const status = document.createElement("div");
  status.className = `hypster-status hypster-status-${snapshot.status}`;
  status.textContent = snapshot.status === "pending" ? "Pending changes" : "Applied";

  if (snapshot.error) {
    status.textContent = `Error: ${snapshot.error.message}`;
    status.title = snapshot.error.message;
  }

  return status;
}

function render(model, el) {
  const snapshot = model.get("snapshot");
  el.replaceChildren();

  const root = document.createElement("div");
  const theme = detectHostTheme(el.ownerDocument || document);
  root.className = `hypster-widget hypster-theme-${theme}`;
  root.dataset.hypsterTheme = theme;

  const header = document.createElement("div");
  header.className = "hypster-header";

  const title = document.createElement("div");
  title.className = "hypster-title";
  title.textContent = snapshot.schema?.display_label || snapshot.schema?.name || "Hypster";
  header.append(title);
  header.append(renderStatus(snapshot));
  root.append(header);

  for (const parameter of snapshot.schema?.parameters || []) {
    root.append(renderParameter(model, snapshot, parameter));
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
    el.addEventListener("click", (event) => {
      const target = event.target;
      if (!(target instanceof Element)) return;

      const option = target.closest("[data-hypster-choice-option]");
      if (option instanceof HTMLElement && option.dataset.path) {
        closeChoiceMenus(el);
        send(model, {
          type: "set_value",
          path: option.dataset.path,
          value: decodedValue(option),
        });
        return;
      }

      const trigger = target.closest("[data-hypster-choice-trigger]");
      if (trigger instanceof HTMLElement) {
        const choice = trigger.closest(".hypster-choice");
        if (!(choice instanceof HTMLElement)) return;
        const menu = choice.querySelector(".hypster-choice-menu");
        const isOpen = choice.classList.contains("hypster-choice-open");
        closeChoiceMenus(el);
        if (!isOpen) {
          choice.classList.add("hypster-choice-open");
          trigger.setAttribute("aria-expanded", "true");
          menu?.removeAttribute("hidden");
        }
        return;
      }

      closeChoiceMenus(el);
    });

    el.addEventListener("change", (event) => {
      const target = event.target;
      if (!(target instanceof HTMLInputElement || target instanceof HTMLSelectElement)) return;
      if (!target.dataset.path) return;

      send(model, {
        type: "set_value",
        path: target.dataset.path,
        ...encodedControlValue(target),
      });
    });

    const rerender = () => render(model, el);
    model.on("change:snapshot", rerender);
    render(model, el);
    const cleanupTheme = watchHostTheme(el);

    return () => {
      cleanupTheme();
      model.off?.("change:snapshot", rerender);
    };
  },
};
