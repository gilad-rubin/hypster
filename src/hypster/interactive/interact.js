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

function cloneValue(value) {
  if (value == null) return value;
  return JSON.parse(JSON.stringify(value));
}

function namedSpecs(specs) {
  return (specs || []).filter((spec) => spec?.name);
}

function editableThenSpecs(thenSpecs) {
  const specs = thenSpecs || [];
  return specs.length === 1 ? specs : namedSpecs(specs);
}

function specByName(specs, name) {
  return namedSpecs(specs).find((spec) => spec.name === name) || namedSpecs(specs)[0] || null;
}

function defaultOperator(spec) {
  return spec?.operators?.[0] || "=";
}

function defaultSpecValue(spec) {
  if (!spec) return "";
  if (spec.type === "bool") return false;
  if (spec.type === "int" || spec.type === "float") return 0;
  if (spec.type?.startsWith("multi_")) return [];
  if ((spec.type === "select" || spec.type === "multi_select") && Array.isArray(spec.options)) {
    return spec.type === "multi_select" ? [] : (spec.options[0] ?? "");
  }
  return "";
}

function defaultThenValue(thenSpecs) {
  const specs = editableThenSpecs(thenSpecs);
  if (specs.length > 1) {
    return Object.fromEntries(specs.map((spec) => [spec.name, defaultSpecValue(spec)]));
  }
  return defaultSpecValue(specs[0]);
}

function defaultCondition(fieldSpecs) {
  const spec = namedSpecs(fieldSpecs)[0];
  return {
    field: spec?.name || "",
    operator: defaultOperator(spec),
    value: defaultSpecValue(spec),
  };
}

function editableWhen(when) {
  if (!when) return { combinator: "and", conditions: [] };
  if (when.field) return { combinator: "and", conditions: [when] };
  const conditions = when.conditions || [];
  if (
    (when.combinator === "and" || when.combinator === "or") &&
    Array.isArray(conditions) &&
    conditions.every((condition) => condition?.field)
  ) {
    return { combinator: when.combinator, conditions };
  }
  return null;
}

function conditionInputSpec(spec, operator) {
  if (spec?.type === "select" && (operator === "in" || operator === "not_in")) {
    return { ...spec, type: "multi_select" };
  }
  return spec;
}

function conditionValueForOperator(spec, operator, currentValue) {
  const inputSpec = conditionInputSpec(spec, operator);
  if (inputSpec?.type === "multi_select") {
    if (Array.isArray(currentValue)) return currentValue;
    return currentValue == null || currentValue === "" ? [] : [currentValue];
  }
  if (Array.isArray(currentValue)) return currentValue[0] ?? defaultSpecValue(inputSpec);
  return currentValue ?? defaultSpecValue(inputSpec);
}

function specValueFromControl(input, spec) {
  if (input instanceof HTMLSelectElement && input.multiple) {
    return Array.from(input.selectedOptions, optionValue);
  }
  if (input instanceof HTMLSelectElement) {
    const option = input.selectedOptions[0];
    return option ? optionValue(option) : null;
  }
  if (input.type === "checkbox") return input.checked;
  if (spec?.type === "int" || spec?.type === "float") {
    const value = spec.type === "int" ? Number.parseInt(input.value, 10) : Number.parseFloat(input.value);
    return Number.isFinite(value) ? value : defaultSpecValue(spec);
  }
  if (spec?.type?.startsWith("multi_")) {
    if (!input.value.trim()) return [];
    let values;
    try {
      values = JSON.parse(input.value);
    } catch {
      values = input.value
        .split(",")
        .map((part) => part.trim())
        .filter(Boolean);
    }
    const items = Array.isArray(values) ? values : [values];
    if (spec.type === "multi_int") return items.map((item) => Number.parseInt(item, 10)).filter(Number.isFinite);
    if (spec.type === "multi_float") return items.map((item) => Number.parseFloat(item)).filter(Number.isFinite);
    if (spec.type === "multi_bool") return items.map((item) => item === true || String(item).toLowerCase() === "true");
    return items;
  }
  return input.value;
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

function findParameter(parameters, path) {
  for (const p of parameters || []) {
    if (p.path === path) return p;
    if (p.children) {
      const found = findParameter(p.children, path);
      if (found) return found;
    }
  }
  return null;
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

  if (parameter.kind === "text" && parameter.metadata?.multiline) {
    const textarea = document.createElement("textarea");
    textarea.dataset.path = parameter.path;
    textarea.dataset.kind = parameter.kind;
    textarea.value = currentValue ?? "";
    textarea.rows = 4;
    textarea.className = "hypster-textarea";
    return textarea;
  }

  if (parameter.kind === "rules") {
    return renderRulesControl(snapshot, parameter);
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

function renderRulesControl(snapshot, parameter) {
  const meta = parameter.metadata || {};
  const fieldSpecs = meta.field_specs || [];
  const thenSpecs = meta.then_specs || [];
  const rules = valueFor(snapshot, parameter) || [];

  const container = document.createElement("div");
  container.className = "hypster-rules";
  container.dataset.path = parameter.path;
  container.dataset.kind = "rules";

  for (let i = 0; i < rules.length; i++) {
    const rule = rules[i];
    container.append(renderRuleCard(parameter, fieldSpecs, thenSpecs, rule, i));
  }

  const addBtn = document.createElement("button");
  addBtn.type = "button";
  addBtn.className = "hypster-rules-add";
  addBtn.textContent = "+ Add rule";
  addBtn.dataset.hypsterRulesAdd = "";
  addBtn.dataset.path = parameter.path;
  container.append(addBtn);

  return container;
}

function renderRuleCard(parameter, fieldSpecs, thenSpecs, rule, index) {
  const card = document.createElement("div");
  card.className = "hypster-rule-card";

  const header = document.createElement("div");
  header.className = "hypster-rule-header";
  const title = document.createElement("span");
  title.className = "hypster-rule-title";
  title.textContent = rule.name || `Rule ${index + 1}`;
  header.append(title);

  const removeBtn = document.createElement("button");
  removeBtn.type = "button";
  removeBtn.className = "hypster-rule-remove";
  removeBtn.textContent = "×";
  removeBtn.setAttribute("aria-label", `Remove rule ${index + 1}`);
  removeBtn.dataset.hypsterRuleRemove = "";
  removeBtn.dataset.path = parameter.path;
  removeBtn.dataset.ruleIndex = String(index);
  header.append(removeBtn);
  card.append(header);

  const whenLabel = document.createElement("div");
  whenLabel.className = "hypster-rule-label";
  whenLabel.textContent = "WHEN";
  card.append(whenLabel);

  const when = editableWhen(rule.when);
  if (when) {
    card.append(renderConditionEditor(parameter, fieldSpecs, index, when));
  } else {
    const whenSummary = document.createElement("div");
    whenSummary.className = "hypster-rule-when";
    whenSummary.textContent = summarizeCondition(rule.when);
    card.append(whenSummary);
  }

  const thenLabel = document.createElement("div");
  thenLabel.className = "hypster-rule-label";
  thenLabel.textContent = "THEN";
  card.append(thenLabel);

  card.append(renderThenEditor(parameter, thenSpecs, rule, index));

  return card;
}

function renderConditionEditor(parameter, fieldSpecs, ruleIndex, when) {
  const editor = document.createElement("div");
  editor.className = "hypster-rule-conditions";

  if (when.conditions.length === 0) {
    const empty = document.createElement("div");
    empty.className = "hypster-rule-empty";
    empty.textContent = "Always applies";
    editor.append(empty);
  }

  for (let conditionIndex = 0; conditionIndex < when.conditions.length; conditionIndex++) {
    editor.append(renderConditionRow(parameter, fieldSpecs, ruleIndex, conditionIndex, when.conditions[conditionIndex]));
  }

  if (namedSpecs(fieldSpecs).length > 0) {
    const addCondition = document.createElement("button");
    addCondition.type = "button";
    addCondition.className = "hypster-rule-condition-add";
    addCondition.textContent = "+ Condition";
    addCondition.dataset.hypsterRuleConditionAdd = "";
    addCondition.dataset.path = parameter.path;
    addCondition.dataset.ruleIndex = String(ruleIndex);
    editor.append(addCondition);
  }

  return editor;
}

function renderConditionRow(parameter, fieldSpecs, ruleIndex, conditionIndex, condition) {
  const row = document.createElement("div");
  row.className = "hypster-rule-condition-row";

  const spec = specByName(fieldSpecs, condition.field);

  const fieldSelect = document.createElement("select");
  fieldSelect.className = "hypster-rule-condition-field";
  fieldSelect.dataset.hypsterRuleConditionField = "";
  fieldSelect.dataset.path = parameter.path;
  fieldSelect.dataset.ruleIndex = String(ruleIndex);
  fieldSelect.dataset.conditionIndex = String(conditionIndex);
  for (const fieldSpec of namedSpecs(fieldSpecs)) {
    const option = document.createElement("option");
    option.value = fieldSpec.name;
    option.textContent = fieldSpec.description || fieldSpec.name;
    option.selected = fieldSpec.name === condition.field;
    fieldSelect.append(option);
  }
  row.append(fieldSelect);

  const operatorSelect = document.createElement("select");
  operatorSelect.className = "hypster-rule-condition-operator";
  operatorSelect.dataset.hypsterRuleConditionOperator = "";
  operatorSelect.dataset.path = parameter.path;
  operatorSelect.dataset.ruleIndex = String(ruleIndex);
  operatorSelect.dataset.conditionIndex = String(conditionIndex);
  for (const operator of spec?.operators || ["="]) {
    const option = document.createElement("option");
    option.value = operator;
    option.textContent = operator;
    option.selected = operator === condition.operator;
    operatorSelect.append(option);
  }
  row.append(operatorSelect);

  const valueSpec = conditionInputSpec(spec, condition.operator);
  row.append(
    renderSpecInput(valueSpec, conditionValueForOperator(spec, condition.operator, condition.value), {
      className: "hypster-rule-condition-value",
      hypsterRuleConditionValue: "",
      path: parameter.path,
      ruleIndex: String(ruleIndex),
      conditionIndex: String(conditionIndex),
    })
  );

  const remove = document.createElement("button");
  remove.type = "button";
  remove.className = "hypster-rule-condition-remove";
  remove.textContent = "×";
  remove.setAttribute("aria-label", `Remove condition ${conditionIndex + 1}`);
  remove.dataset.hypsterRuleConditionRemove = "";
  remove.dataset.path = parameter.path;
  remove.dataset.ruleIndex = String(ruleIndex);
  remove.dataset.conditionIndex = String(conditionIndex);
  row.append(remove);

  return row;
}

function renderThenEditor(parameter, thenSpecs, rule, ruleIndex) {
  const editor = document.createElement("div");
  editor.className = "hypster-rule-then";

  const specs = editableThenSpecs(thenSpecs);
  if (specs.length === 0) {
    editor.textContent = typeof rule.then === "string" ? rule.then : JSON.stringify(rule.then);
    return editor;
  }

  for (const spec of specs) {
    const field = document.createElement("label");
    field.className = "hypster-rule-then-field";

    const label = document.createElement("span");
    label.textContent = spec.description || spec.name || "Value";
    field.append(label);

    const value = specs.length > 1 && rule.then && typeof rule.then === "object" && !Array.isArray(rule.then)
      ? rule.then[spec.name]
      : rule.then;
    field.append(
      renderSpecInput(spec, value ?? defaultSpecValue(spec), {
        className: "hypster-rule-then-value",
        hypsterRuleThenValue: "",
        path: parameter.path,
        ruleIndex: String(ruleIndex),
        thenName: spec.name,
      })
    );
    editor.append(field);
  }

  return editor;
}

function renderSpecInput(spec, value, dataset) {
  const className = dataset.className;
  const attrs = { ...dataset };
  delete attrs.className;

  let input;
  if (spec?.type === "select" || spec?.type === "multi_select") {
    input = document.createElement("select");
    input.multiple = spec.type === "multi_select";
    for (const optionValue of spec.options || []) {
      const option = document.createElement("option");
      option.value = String(input.options.length);
      option.dataset.value = encodedValue(optionValue);
      option.textContent = String(optionValue);
      option.selected = Array.isArray(value)
        ? value.some((selectedValue) => valuesEqual(selectedValue, optionValue))
        : valuesEqual(optionValue, value);
      input.append(option);
    }
  } else if (spec?.type === "bool") {
    input = document.createElement("input");
    input.type = "checkbox";
    input.checked = Boolean(value);
  } else if (spec?.type === "text" && spec.multiline) {
    input = document.createElement("textarea");
    input.value = value ?? "";
    input.rows = 3;
  } else {
    input = document.createElement("input");
    if (spec?.type === "int" || spec?.type === "float") {
      input.type = "number";
      if (spec.type === "float") input.step = "any";
      input.value = value ?? 0;
    } else {
      input.type = "text";
      input.value = spec?.type?.startsWith("multi_") ? JSON.stringify(value ?? []) : (value ?? "");
    }
  }

  input.className = className;
  input.dataset.kind = spec?.type || "text";
  for (const [key, attrValue] of Object.entries(attrs)) {
    if (attrValue == null) continue;
    input.dataset[key] = attrValue;
  }
  return input;
}

function summarizeCondition(when) {
  if (!when) return "(always)";
  if (when.field) {
    return `${when.field} ${when.operator} ${JSON.stringify(when.value)}`;
  }
  if (when.combinator && when.conditions) {
    const parts = when.conditions.map(summarizeCondition);
    const sep = ` ${when.combinator.toUpperCase()} `;
    return parts.join(sep);
  }
  return JSON.stringify(when);
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

function updateRule(model, path, rawRuleIndex, updater) {
  const snapshot = model.get("snapshot");
  const param = findParameter(snapshot.schema?.parameters, path);
  if (!param) return;

  const ruleIndex = parseInt(rawRuleIndex, 10);
  const rules = (valueFor(snapshot, param) || []).map(cloneValue);
  const currentRule = rules[ruleIndex] || {
    when: { combinator: "and", conditions: [] },
    then: defaultThenValue(param.metadata?.then_specs || []),
  };
  rules[ruleIndex] = updater(currentRule, param);
  send(model, { type: "set_value", path, value: rules });
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

      const ruleRemove = target.closest("[data-hypster-rule-remove]");
      if (ruleRemove instanceof HTMLElement && ruleRemove.dataset.path) {
        const snapshot = model.get("snapshot");
        const path = ruleRemove.dataset.path;
        const index = parseInt(ruleRemove.dataset.ruleIndex, 10);
        const param = findParameter(snapshot.schema?.parameters, path);
        if (!param) return;
        const rules = valueFor(snapshot, param) || [];
        const updated = rules.filter((_, i) => i !== index);
        send(model, { type: "set_value", path, value: updated });
        return;
      }

      const ruleAdd = target.closest("[data-hypster-rules-add]");
      if (ruleAdd instanceof HTMLElement && ruleAdd.dataset.path) {
        const snapshot = model.get("snapshot");
        const path = ruleAdd.dataset.path;
        const param = findParameter(snapshot.schema?.parameters, path);
        if (!param) return;
        const rules = valueFor(snapshot, param) || [];
        const thenSpecs = param.metadata?.then_specs || [];
        const newRule = { when: { combinator: "and", conditions: [] }, then: defaultThenValue(thenSpecs) };
        send(model, { type: "set_value", path, value: [...rules, newRule] });
        return;
      }

      const conditionAdd = target.closest("[data-hypster-rule-condition-add]");
      if (conditionAdd instanceof HTMLElement && conditionAdd.dataset.path) {
        updateRule(model, conditionAdd.dataset.path, conditionAdd.dataset.ruleIndex, (rule, param) => {
          if (namedSpecs(param.metadata?.field_specs || []).length === 0) return rule;
          const when = editableWhen(rule.when) || { combinator: "and", conditions: [] };
          return {
            ...rule,
            when: {
              combinator: when.combinator,
              conditions: [...when.conditions, defaultCondition(param.metadata?.field_specs || [])],
            },
          };
        });
        return;
      }

      const conditionRemove = target.closest("[data-hypster-rule-condition-remove]");
      if (conditionRemove instanceof HTMLElement && conditionRemove.dataset.path) {
        updateRule(model, conditionRemove.dataset.path, conditionRemove.dataset.ruleIndex, (rule) => {
          const when = editableWhen(rule.when) || { combinator: "and", conditions: [] };
          const conditionIndex = parseInt(conditionRemove.dataset.conditionIndex, 10);
          return {
            ...rule,
            when: {
              combinator: when.combinator,
              conditions: when.conditions.filter((_, i) => i !== conditionIndex),
            },
          };
        });
        return;
      }

      closeChoiceMenus(el);
    });

    el.addEventListener("change", (event) => {
      const target = event.target;
      if (!(target instanceof HTMLInputElement || target instanceof HTMLSelectElement || target instanceof HTMLTextAreaElement)) return;
      if (!target.dataset.path) return;

      if (target.dataset.hypsterRuleConditionField != null) {
        updateRule(model, target.dataset.path, target.dataset.ruleIndex, (rule, param) => {
          const when = editableWhen(rule.when) || { combinator: "and", conditions: [] };
          const conditionIndex = parseInt(target.dataset.conditionIndex, 10);
          const spec = specByName(param.metadata?.field_specs || [], target.value);
          const conditions = when.conditions.map((condition, i) =>
            i === conditionIndex
              ? { field: spec?.name || "", operator: defaultOperator(spec), value: defaultSpecValue(spec) }
              : condition
          );
          return { ...rule, when: { combinator: when.combinator, conditions } };
        });
        return;
      }

      if (target.dataset.hypsterRuleConditionOperator != null) {
        updateRule(model, target.dataset.path, target.dataset.ruleIndex, (rule, param) => {
          const when = editableWhen(rule.when) || { combinator: "and", conditions: [] };
          const conditionIndex = parseInt(target.dataset.conditionIndex, 10);
          const current = when.conditions[conditionIndex];
          const spec = specByName(param.metadata?.field_specs || [], current?.field);
          const conditions = when.conditions.map((condition, i) =>
            i === conditionIndex
              ? {
                  ...condition,
                  operator: target.value,
                  value: conditionValueForOperator(spec, target.value, condition.value),
                }
              : condition
          );
          return { ...rule, when: { combinator: when.combinator, conditions } };
        });
        return;
      }

      if (target.dataset.hypsterRuleConditionValue != null) {
        updateRule(model, target.dataset.path, target.dataset.ruleIndex, (rule, param) => {
          const when = editableWhen(rule.when) || { combinator: "and", conditions: [] };
          const conditionIndex = parseInt(target.dataset.conditionIndex, 10);
          const current = when.conditions[conditionIndex];
          const spec = specByName(param.metadata?.field_specs || [], current?.field);
          const value = specValueFromControl(target, conditionInputSpec(spec, current?.operator));
          const conditions = when.conditions.map((condition, i) =>
            i === conditionIndex ? { ...condition, value } : condition
          );
          return { ...rule, when: { combinator: when.combinator, conditions } };
        });
        return;
      }

      if (target.dataset.hypsterRuleThenValue != null) {
        updateRule(model, target.dataset.path, target.dataset.ruleIndex, (rule, param) => {
          const thenSpecs = editableThenSpecs(param.metadata?.then_specs || []);
          const spec = thenSpecs.length > 1 ? specByName(thenSpecs, target.dataset.thenName) : thenSpecs[0];
          const value = specValueFromControl(target, spec);
          if (thenSpecs.length > 1) {
            const then = rule.then && typeof rule.then === "object" && !Array.isArray(rule.then) ? { ...rule.then } : {};
            then[target.dataset.thenName] = value;
            return { ...rule, then };
          }
          return { ...rule, then: value };
        });
        return;
      }

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
