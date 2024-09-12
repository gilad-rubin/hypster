# File: magic.py

from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List

import ipywidgets as widgets
import pandas as pd
from IPython.core.magic import Magics, cell_magic, magics_class
from IPython.core.magic_arguments import argument, magic_arguments, parse_argstring
from IPython.display import HTML, clear_output, display

from .core import Hypster, find_hp_function_body_and_name
from .hp import HP


def custom_load(module_source: str, inject_names=True) -> Hypster:
    module_source = module_source.replace("@config", "").replace("@hypster.config", "")
    namespace = {"HP": HP}
    exec(module_source, namespace)
    result = find_hp_function_body_and_name(module_source)

    if result is None:
        raise ValueError("No configuration function found in the module")

    func_name, config_body = result
    func = namespace.get(func_name)

    if func is None:
        raise ValueError(f"Could not find the function {func_name} in the loaded module")

    return Hypster(func_name, config_body, namespace, inject_names)


def get_available_options(combinations, selected_params):
    available_options = OrderedDict()
    all_params = list(OrderedDict.fromkeys(key for comb in combinations for key in comb.keys()))

    for param in all_params:
        options = set()
        for comb in combinations:
            if all(comb.get(k) == v for k, v in selected_params.items() if k < param):
                if param in comb:
                    options.add(comb[param])
        if options:
            available_options[param] = sorted(list(options))

    return available_options


class InteractiveHypster:
    def __init__(
        self,
        hp_config: Hypster,
        shell,
        selections: Dict[str, Any] = None,
        overrides: Dict[str, Any] = None,
        final_vars: List[str] = None,
        results_var: str = None,
        instantiate_on_first: bool = False,
        instantiate_on_change: bool = False,
    ):
        self.hp_config = hp_config
        self.shell = shell
        self.combinations = hp_config.get_combinations()
        self.defaults = self.get_defaults()
        self.selections = selections or {}
        self.overrides = overrides or {}
        self.final_vars = final_vars or []
        self.results_var = results_var
        self.instantiate_on_first = instantiate_on_first
        self.instantiate_on_change = instantiate_on_change
        self.selected_params = OrderedDict()
        self.widgets = OrderedDict()
        self.output = widgets.Output()
        self.widgets_container = widgets.VBox([])
        self.create_widgets()
        if self.instantiate_on_first:
            self.instantiate(None)

    def get_defaults(self):
        defaults = OrderedDict()
        for comb in self.combinations:
            for param, value in comb.items():
                if param not in defaults:
                    defaults[param] = value
        return defaults

    def create_widgets(self):
        available_options = get_available_options(self.combinations, self.selected_params)
        for param, options in available_options.items():
            self.create_widget_for_param(param, options)
        self.update_widgets_display()

    def create_widget_for_param(self, param, options, value=None):
        disabled = False

        if param in self.overrides:
            value = self.overrides[param]
            disabled = True
        elif param in self.selections:
            value = self.selections[param]
        elif value is None:
            value = self.defaults.get(param, options[0] if options else None)

        if value not in options:
            options.append(value)

        widget = widgets.Dropdown(
            options=options,
            value=value,
            description=f"Select {param}",
            style={'description_width': 'initial'},
            #layout=widgets.Layout(width='auto'),
            disabled=disabled,
        )
        widget.observe(self.on_change, names="value")

        self.widgets[param] = widget
        self.selected_params[param] = value

    def on_change(self, change):
        with self.output:
            clear_output()
            param = change["owner"].description.split()[-1]
            self.selected_params[param] = change["new"]

            # Store current selections
            current_selections = {k: v for k, v in self.selected_params.items()}

            # Remove all subsequent parameters
            keys_to_remove = [key for key in list(self.widgets.keys()) if key > param]
            for key in keys_to_remove:
                del self.widgets[key]
                if key in self.selected_params:
                    del self.selected_params[key]

            # Recreate subsequent widgets
            available_options = get_available_options(self.combinations, self.selected_params)
            for next_param, options in available_options.items():
                if next_param not in self.widgets:
                    if next_param in current_selections and current_selections[next_param] in options:
                        # Preserve the user's previous selection if it's still valid
                        self.create_widget_for_param(next_param, options, current_selections[next_param])
                    else:
                        self.create_widget_for_param(next_param, options)

            self.update_widgets_display()
            self.display_valid_combinations()

            if self.instantiate_on_change:
                self.instantiate(None)

    def update_widgets_display(self):
        self.widgets_container.children = list(self.widgets.values())

    def display_valid_combinations(self):
        valid_combinations = [
            comb for comb in self.combinations if all(comb.get(k) == v for k, v in self.selected_params.items())
        ]
        df = pd.DataFrame(valid_combinations)
        display(df)

    def instantiate(self, button):
        with self.output:
            clear_output()
            results = self.hp_config(
                final_vars=self.final_vars, selections=self.selected_params, overrides=self.overrides
            )
            if self.results_var:
                self.shell.user_ns[self.results_var] = results
            else:
                self.shell.user_ns.update(results)

    def display(self):
        instantiate_button = widgets.Button(description="Instantiate")
        instantiate_button.on_click(self.instantiate)

        if self.instantiate_on_change:
            button_layout = widgets.Layout(display="none")
        else:
            button_layout = widgets.Layout()

        instantiate_button.layout = button_layout

        display(widgets.VBox([self.widgets_container, instantiate_button, self.output]))


@magics_class
class HypsterMagics(Magics):
    def __init__(self, shell):
        super().__init__(shell)
        self._first_run = True

    @magic_arguments()
    @argument("config_name", help="Name for the config module")
    @argument("-s", "--selections", help="Variable name containing selections dict")
    @argument("-o", "--overrides", help="Variable name containing overrides dict")
    @argument("-f", "--final_vars", help="Comma-separated list of final variables")
    @argument("-w", "--write_to_file", help="Write cell content to a file")
    @argument("-r", "--results", help="Variable name to store the results dictionary")
    @argument(
        "-i",
        "--instantiate",
        choices=["first", "change", "button", "first,change"],
        help="Instantiation behavior: 'first' (on first run), 'change' (on parameter change), "
        "'button' (manual only), or 'first,change' (on first run and parameter change)",
    )
    @cell_magic
    def hypster(self, line, cell):
        css = """
        <style>
        .cell-output-ipywidget-background {
           background-color: transparent !important;
        }
        :root {
            --jp-widgets-color: var(--vscode-editor-foreground);
            --jp-widgets-font-size: var(--vscode-editor-font-size);
        }
        </style>
        """
        display(HTML(css))

        args = parse_argstring(self.hypster, line)
        hp_config = custom_load(cell)

        selections = self.shell.user_ns.get(args.selections, {}) if args.selections else {}
        overrides = self.shell.user_ns.get(args.overrides, {}) if args.overrides else {}
        final_vars = args.final_vars.split(",") if args.final_vars else None
        results_var = args.results

        instantiate_options = args.instantiate.split(",") if args.instantiate else ["button"]
        instantiate_on_first = "first" in instantiate_options
        instantiate_on_change = "change" in instantiate_options

        interactive_hypster = InteractiveHypster(
            hp_config,
            self.shell,
            selections,
            overrides,
            final_vars,
            results_var,
            instantiate_on_first=instantiate_on_first,
            instantiate_on_change=instantiate_on_change,
        )
        interactive_hypster.display()

        self.shell.user_ns[args.config_name] = hp_config

        if args.write_to_file:
            Path(args.write_to_file).write_text(cell)

        return None


def load_ipython_extension(ipython):
    ipython.register_magics(HypsterMagics)
