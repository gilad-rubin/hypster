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
  try {
    const parentDocument = globalThis.parent?.document;
    if (parentDocument) documents.push(parentDocument);
  } catch {
    // Cross-origin notebook hosts can block parent access.
  }
  if (baseDocument && !documents.includes(baseDocument)) documents.push(baseDocument);
  return documents;
}

function themeFromColor(value) {
  const raw = String(value || "").trim();
  if (!raw || raw === "transparent" || raw === "rgba(0, 0, 0, 0)") return null;

  const rgb = raw.match(/rgba?\(([^)]+)\)/i);
  if (rgb) {
    const parts = (rgb[1].match(/[\d.]+/g) || []).map(Number);
    if (parts.length >= 3 && parts.slice(0, 3).every(Number.isFinite)) {
      if (parts.length >= 4 && parts[3] < 0.1) return null;
      const luminance = 0.299 * parts[0] + 0.587 * parts[1] + 0.114 * parts[2];
      return { background: raw, luminance, theme: luminance > 150 ? "light" : "dark" };
    }
  }

  const hex = raw.match(/^#([0-9a-f]{3}|[0-9a-f]{6})$/i);
  if (hex) {
    const digits =
      hex[1].length === 3
        ? hex[1].split("").map((digit) => digit + digit).join("")
        : hex[1];
    const red = Number.parseInt(digits.slice(0, 2), 16);
    const green = Number.parseInt(digits.slice(2, 4), 16);
    const blue = Number.parseInt(digits.slice(4, 6), 16);
    const luminance = 0.299 * red + 0.587 * green + 0.114 * blue;
    return { background: raw, luminance, theme: luminance > 150 ? "light" : "dark" };
  }

  return null;
}

function fallbackBackground(theme) {
  return theme === "light" ? "#ffffff" : "#111111";
}

function themeState(theme, parsedColor = null) {
  const safeTheme = theme === "light" ? "light" : "dark";
  let background = parsedColor?.background || fallbackBackground(safeTheme);

  if (typeof parsedColor?.luminance === "number") {
    const backgroundLooksLight = parsedColor.luminance > 150;
    if ((safeTheme === "dark" && backgroundLooksLight) || (safeTheme === "light" && !backgroundLooksLight)) {
      background = fallbackBackground(safeTheme);
    }
  }

  return {
    background,
    luminance: parsedColor?.luminance ?? null,
    theme: safeTheme,
  };
}

function themeFromDocument(doc) {
  const body = doc.body;
  const html = doc.documentElement;
  if (!body || !html) return null;

  const view = doc.defaultView || globalThis;
  const rootStyle = view.getComputedStyle?.(html);
  const bodyStyle = view.getComputedStyle?.(body);
  const backgroundCandidates = [
    rootStyle?.getPropertyValue("--vscode-editor-background"),
    rootStyle?.getPropertyValue("--jp-layout-color0"),
    rootStyle?.getPropertyValue("--jp-layout-color1"),
    bodyStyle?.backgroundColor,
    rootStyle?.backgroundColor,
  ];
  const parsedBackground = backgroundCandidates.map(themeFromColor).find(Boolean);

  const vscodeThemeKind = body.getAttribute("data-vscode-theme-kind") || html.getAttribute("data-vscode-theme-kind");
  if (vscodeThemeKind) return themeState(vscodeThemeKind.includes("light") ? "light" : "dark", parsedBackground);

  const classNames = `${body.className || ""} ${html.className || ""}`.toLowerCase();
  if (classNames.includes("vscode-high-contrast-light") || classNames.includes("vscode-light")) {
    return themeState("light", parsedBackground);
  }
  if (classNames.includes("vscode-high-contrast") || classNames.includes("vscode-dark")) {
    return themeState("dark", parsedBackground);
  }

  const jupyterThemeLight = body.dataset.jpThemeLight ?? html.dataset.jpThemeLight;
  if (jupyterThemeLight === "true") return themeState("light", parsedBackground);
  if (jupyterThemeLight === "false") return themeState("dark", parsedBackground);
  if (classNames.includes("jp-mod-light")) return themeState("light", parsedBackground);
  if (classNames.includes("jp-mod-dark")) return themeState("dark", parsedBackground);

  const dataTheme = body.dataset.theme || html.dataset.theme || body.dataset.mode || html.dataset.mode;
  if (dataTheme === "light" || dataTheme === "dark") return themeState(dataTheme, parsedBackground);

  const colorScheme = rootStyle?.getPropertyValue("color-scheme").trim();
  const colorSchemes = colorScheme?.split(/\s+/) || [];
  if (colorSchemes.includes("dark") && !colorSchemes.includes("light")) return themeState("dark", parsedBackground);
  if (colorSchemes.includes("light") && !colorSchemes.includes("dark")) return themeState("light", parsedBackground);

  if (parsedBackground) return themeState(parsedBackground.theme, parsedBackground);

  return null;
}

function detectHostTheme(baseDocument) {
  for (const doc of themeDocuments(baseDocument)) {
    const state = themeFromDocument(doc);
    if (state) return state;
  }

  const preferredTheme = globalThis.matchMedia?.("(prefers-color-scheme: light)").matches ? "light" : "dark";
  return themeState(preferredTheme);
}

function applyHostTheme(el) {
  const state = detectHostTheme(el.ownerDocument || document);
  const root = el.querySelector(".hypster-widget");
  if (root) {
    root.classList.toggle("hypster-theme-light", state.theme === "light");
    root.classList.toggle("hypster-theme-dark", state.theme === "dark");
    root.dataset.hypsterTheme = state.theme;
    root.style.setProperty("--hypster-background", state.background);
  }
  paintOutputSurface(el, state);
}

function outputSurfaceElements(el) {
  const elements = [el];
  let current = el.parentElement;

  for (let depth = 0; current && current !== document.body && current !== document.documentElement && depth < 6; depth += 1) {
    elements.push(current);
    const className = String(current.className || "");
    if (/cell-output|outputarea|output-area|jp-outputarea|widget-subarea|widget-area|vscode-cell-output/i.test(className)) {
      break;
    }
    current = current.parentElement;
  }

  return elements;
}

function rememberStyle(el, element, property) {
  if (!el.__hypsterStyleRecords) el.__hypsterStyleRecords = new Map();
  let record = el.__hypsterStyleRecords.get(element);
  if (!record) {
    record = new Map();
    el.__hypsterStyleRecords.set(element, record);
  }
  if (!record.has(property)) {
    record.set(property, {
      priority: element.style.getPropertyPriority(property),
      value: element.style.getPropertyValue(property),
    });
  }
}

function setTrackedStyle(el, element, property, value, priority = "") {
  rememberStyle(el, element, property);
  element.style.setProperty(property, value, priority);
}

function paintOutputSurface(el, state) {
  for (const element of outputSurfaceElements(el)) {
    setTrackedStyle(el, element, "background-color", state.background, "important");
    setTrackedStyle(el, element, "color-scheme", state.theme);
  }

  setTrackedStyle(el, el, "display", "block");
  setTrackedStyle(el, el, "width", "100%");
}

function restoreOutputSurface(el) {
  for (const [element, record] of el.__hypsterStyleRecords || new Map()) {
    for (const [property, previous] of record) {
      element.style.setProperty(property, previous.value, previous.priority);
    }
  }
  delete el.__hypsterStyleRecords;
}

function watchHostTheme(el) {
  const apply = () => applyHostTheme(el);
  const observers = [];
  const watchedAttributes = ["class", "data-vscode-theme-kind", "data-jp-theme-light", "data-theme", "data-mode", "style"];

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
  root.className = `hypster-widget hypster-theme-${theme.theme}`;
  root.dataset.hypsterTheme = theme.theme;
  root.style.setProperty("--hypster-background", theme.background);

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
  paintOutputSurface(el, theme);
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
      restoreOutputSurface(el);
      model.off?.("change:snapshot", rerender);
    };
  },
};
