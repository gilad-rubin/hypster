import asyncio
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import ipywidgets as widgets
from IPython.display import HTML, display

from .handler import (
    BooleanComponent,
    ComponentBase,
    FloatComponent,
    IntComponent,
    NumericComponentBase,
    SelectComponent,
    TextComponent,
    UIComponent,
    UIHandler,
    create_ui_handler,
)

logger = logging.getLogger("hypster.ui.ipywidgets")


class IPySelectComponent(UIComponent):
    def __init__(self, component: SelectComponent, on_change: Callable):
        self.on_change = on_change
        super().__init__(component)

    def _create_widget(self) -> widgets.Widget:
        logger.debug(f"Creating select widget for {self.component.id}")
        widget_cls = widgets.SelectMultiple if not self.component.single_value else widgets.Dropdown
        widget = widget_cls(
            options=self.component.options,
            value=self.component.value if not self.component.single_value else self.component.value,
            description=self.component.label,
            style={"description_width": "initial"},
            layout=widgets.Layout(background_color="transparent"),
        )
        widget.observe(
            lambda change: self.on_change(self.component.id, change["new"], delay=not self.component.single_value),
            names="value",
        )
        return widget

    def _update_widget(self) -> None:
        logger.debug(f"Updating select widget {self.component.id}")
        self.widget.options = self.component.options
        self.widget.value = self.component.value
        self.widget.description = self.component.label


class IPyNumericComponent(UIComponent):
    def __init__(self, component: NumericComponentBase, on_change: Callable):
        self.on_change = on_change
        super().__init__(component)

    def _create_widget(self) -> widgets.Widget:
        logger.debug(f"Creating numeric widget for {self.component.id}")
        widget_cls = widgets.IntSlider if isinstance(self.component.min_step, int) else widgets.FloatSlider

        config = self.component.widget_config
        widget = widget_cls(
            **config,
            style={"description_width": "initial"},
            layout=widgets.Layout(background_color="transparent"),
            continuous_update=False,
        )
        widget.observe(
            lambda change: self.on_change(
                self.component.id, change["new"], delay=not isinstance(self.component.min_step, int)
            ),
            names="value",
        )
        return widget

    def _update_widget(self) -> None:
        logger.debug(f"Updating numeric widget {self.component.id}")
        logger.debug(f"Current widget config: {self.component.widget_config}")

        config = self.component.widget_config
        for key, value in config.items():
            logger.debug(f"Setting {key} = {value}")
            setattr(self.widget, key, value)

        logger.debug(f"Widget update complete for {self.component.id}")


class IPyTextComponent(UIComponent):
    def __init__(self, component: TextComponent, on_change: Callable):
        self.on_change = on_change
        super().__init__(component)

    def _create_widget(self) -> widgets.Widget:
        widget = widgets.Text(
            value=str(self.component.value),  # Convert to string
            description=self.component.label,
            style={"description_width": "initial"},
            layout=widgets.Layout(background_color="transparent"),
            continuous_update=False,
        )
        widget.observe(
            lambda change: self.on_change(self.component.id, change["new"], delay=True),  # Add delay
            names="value",
        )
        return widget

    def _update_widget(self) -> None:
        self.widget.value = str(self.component.value)  # Convert to string
        self.widget.description = self.component.label


class IPyBooleanComponent(UIComponent):
    def __init__(self, component: BooleanComponent, on_change: Callable):
        self.on_change = on_change
        super().__init__(component)

    def _create_widget(self) -> widgets.Widget:
        if not self.component.single_value:
            container = widgets.VBox([])
            self._update_multi_widget(container)
            return container
        return self._create_single_widget()

    def _create_single_widget(self) -> widgets.Widget:
        return widgets.Checkbox(
            value=self.component.value,
            description=self.component.label,
            style={"description_width": "initial"},
        )

    def _update_multi_widget(self, container: widgets.VBox) -> None:
        widgets_list = []
        for i, value in enumerate(self.component.value):
            widget = self._create_single_widget()
            widget.value = value
            widget.description = f"{self.component.label}[{i}]"
            widget.observe(
                lambda change, idx=i: self.on_change(
                    self.component.id, [v if j != idx else change["new"] for j, v in enumerate(self.component.value)]
                ),
                names="value",
            )
            widgets_list.append(widget)
        container.children = widgets_list

    def _update_widget(self) -> None:
        if not self.component.single_value:
            self._update_multi_widget(self.widget)
        else:
            self.widget.value = self.component.value
            self.widget.description = self.component.label


class IPyMultiValueComponent(UIComponent):
    def __init__(self, component: Union[IntComponent, FloatComponent, TextComponent], on_change: Callable):
        self.on_change = on_change
        super().__init__(component)

    def _create_widget(self) -> widgets.Widget:
        # Calculate initial height based on number of values (minimum 2 rows)
        num_rows = max(len(self.component.value), 2)
        # Reduced base height and row height
        height = f"{25 + (num_rows * 15)}px"  # Base height + row height

        widget = widgets.Textarea(
            value=self._format_value(self.component.value),
            description=f"{self.component.label}",
            style={"description_width": "initial"},
            layout=widgets.Layout(
                height=height,
                background_color="transparent",
                min_height="52px",  # Minimum height for 2 rows
            ),
            continuous_update=False,
        )
        widget.observe(lambda change: self._handle_value_change(change["new"]), names="value")
        return widget

    def _format_value(self, value: Union[List[int], List[float], List[str]]) -> str:
        return "\n".join(str(v) for v in value)

    def _parse_value(self, text: str) -> List[Any]:
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        if isinstance(self.component, IntComponent):
            return [int(line) for line in lines]
        elif isinstance(self.component, FloatComponent):
            return [float(line) for line in lines]
        return lines  # For TextComponent

    def _handle_value_change(self, text: str) -> None:
        try:
            new_value = self._parse_value(text)
            self.on_change(self.component.id, new_value, delay=True)
        except (ValueError, TypeError) as e:
            logger.warning(f"Invalid input: {e}")

    def _update_widget(self) -> None:
        self.widget.value = self._format_value(self.component.value)
        self.widget.description = self.component.label

        # Update height based on current number of values
        num_rows = max(len(self.component.value), 2)
        self.widget.layout.height = f"{25 + (num_rows * 15)}px"  # Using same calculation as in create


class IPyWidgetsUI:
    """IPyWidgets-based UI implementation."""

    def __init__(self):
        self.ui_handler: Optional[UIHandler] = None
        self.ui_components: Dict[str, UIComponent] = {}
        self.container = widgets.VBox([], layout=widgets.Layout(background_color="transparent"))
        self._update_timer: Optional[asyncio.Future] = None
        self.update_delay = 0.1  # seconds
        logger.debug("Initialized IPyWidgetsUI")

    async def _delayed_update(self, component_id: str, new_value: Any) -> None:
        """Handle component updates with delay."""
        if self._update_timer:
            self._update_timer.cancel()

        await asyncio.sleep(self.update_delay)
        self._handle_change_impl(component_id, new_value)

    def _handle_change(self, component_id: str, new_value: Any, delay: bool = False) -> None:
        """Schedule an update, with optional delay."""
        if not self.ui_handler:
            return

        logger.debug(f"Scheduling update for {component_id}: {new_value}")
        if delay:
            asyncio.create_task(self._delayed_update(component_id, new_value))
        else:
            self._handle_change_impl(component_id, new_value)

    def _handle_change_impl(self, component_id: str, new_value: Any) -> None:
        """Handle component changes and update UI."""
        if not self.ui_handler:
            return

        # Log to regular output
        logger.debug(f"Handling change for {component_id}: {new_value}")

        # Get affected components from handler
        affected_components = self.ui_handler.update_component(component_id, new_value)

        # Update affected UI components
        for comp_id, component in affected_components.items():
            if comp_id in self.ui_components:
                logger.debug(f"Updating UI component: {comp_id}")
                # Remove output context wrapper
                self.ui_components[comp_id].update(component)
            else:
                logger.debug(f"Creating new UI component: {comp_id}")
                self.ui_components[comp_id] = self._create_ui_component(component)

        # Rebuild widget list in correct order
        widgets_list = []
        for comp_id, component in self.ui_handler.components.items():
            if comp_id in self.ui_components:
                widgets_list.append(self.ui_components[comp_id].render())

        self.container.children = widgets_list

    def _create_ui_component(self, component: ComponentBase) -> UIComponent:
        """Create a UI component with theme-aware styling."""
        if isinstance(component, SelectComponent):
            return IPySelectComponent(component, on_change=self._handle_change)
        elif isinstance(component, (IntComponent, FloatComponent, TextComponent)):
            if not component.single_value:
                return IPyMultiValueComponent(component, on_change=self._handle_change)
            elif isinstance(component, (IntComponent, FloatComponent)):
                return IPyNumericComponent(component, on_change=self._handle_change)
            else:
                return IPyTextComponent(component, on_change=self._handle_change)
        elif isinstance(component, BooleanComponent):
            return IPyBooleanComponent(component, on_change=self._handle_change)
        else:
            raise ValueError(f"Unsupported component type: {type(component)}")

    def _update_display(self) -> None:
        """Update the display with current components."""
        if not self.ui_handler:
            logger.warning("No UI handler set")
            return

        logger.debug("Updating display")
        widgets_list = []

        # Get components in order
        ordered_components = self.ui_handler.get_ordered_components()
        logger.debug(f"Ordered components: {[comp.id for comp in ordered_components]}")

        if not ordered_components:
            logger.warning("No components found in UI handler")
            return

        current_component_ids = {comp.id for comp in ordered_components}

        # Remove obsolete UI components
        for comp_id in list(self.ui_components.keys()):
            if comp_id not in current_component_ids:
                logger.debug(f"Removing obsolete component: {comp_id}")
                self.ui_components.pop(comp_id)

        # Update or create UI components in order
        for component in ordered_components:
            logger.debug(f"Processing component: {component.id} ({component.parameter_type})")
            if component.id not in self.ui_components:
                logger.debug(f"Creating new UI component for: {component.id}")
                self.ui_components[component.id] = self._create_ui_component(component)
            widgets_list.append(self.ui_components[component.id].render())

        self.container.children = widgets_list

    def display(self, config_func: Callable) -> None:
        """Display interactive UI for the given config function."""
        # self.ui_handler = create_ui_handler(config_func)
        self._update_display()
        display(self.container)


def create_interactive_config(config_func: Callable) -> Tuple[UIHandler, "IPyWidgetsUI"]:
    """Create and display an interactive configuration interface."""
    logger.debug("Creating interactive config")

    # First run the config to ensure we have initial values
    config_func()

    # Create handler and UI
    handler = create_ui_handler(config_func)
    ui = IPyWidgetsUI()
    ui.ui_handler = handler

    # Verify components were created
    logger.debug(f"Created components: {list(handler.components.keys())}")

    # Update and display UI
    ui._update_display()
    display(ui.container)

    return handler, ui


def apply_vscode_theme():
    """Apply VS Code-aware theming to ipywidgets"""
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
