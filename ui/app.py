import os
import sys
from typing import Any, Dict, List

import streamlit as st
from hypster import Builder, ConfigNode, HypsterDriver, visualize_config_tree

project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)  # TODO: fix this?
sys.path.insert(0, project_root)


def get_suffix(name: str) -> str:
    return name.split("__")[-1]


def render_config_node(node: ConfigNode, path: str = "", state: Dict[str, str] = {}):
    if node.type == "root":
        for child in node.children.values():
            render_config_node(
                child, f"{path}.{child.name}" if path else child.name, state
            )
    elif node.type == "Select":
        options = [get_suffix(name) for name in node.children.keys()]
        full_options = list(node.children.keys())
        if full_options:
            index = full_options.index(state.get(node.name, full_options[0]))
            selected = st.selectbox(
                f"Select {get_suffix(node.name)}",
                options,
                index=index,
                key=f"{path}.{node.name}",
            )
            selected_full_name = next(
                name for name in node.children.keys() if name.endswith(f"__{selected}")
            )
            state[node.name] = selected_full_name
            render_config_node(
                node.children[selected_full_name],
                f"{path}.{selected_full_name}" if path else selected_full_name,
                state,
            )
        else:
            st.warning(f"No options available for {get_suffix(node.name)}")
    elif node.children:
        st.subheader(get_suffix(node.name))
        for child in node.children.values():
            render_config_node(
                child, f"{path}.{child.name}" if path else child.name, state
            )
    else:
        value = node.value if node.value is not None else ""
        shared_text = " [SHARED]" if node.is_shared else ""
        new_value = st.text_input(
            f"{get_suffix(node.name)}{shared_text}",
            value=str(value),
            key=f"{path}.{node.name}",
        )
        if str(new_value) != str(value):
            node.value = (
                type(node.value)(new_value) if node.value is not None else new_value
            )


def collect_config(node: ConfigNode, state: Dict[str, str]) -> Dict[str, Any]:
    if node.type == "Select":
        selected_option = state.get(node.name, list(node.children.keys())[0])
        selected_option_short = selected_option.split("__")[-1]
        return {
            selected_option_short: collect_config(node.children[selected_option], state)
        }
    elif not node.children:
        return node.value
    else:
        return {
            child.name: collect_config(child, state) for child in node.children.values()
        }


def main(driver: HypsterDriver, final_vars: List[str]):
    st.set_page_config(page_title="Hypster Configuration", layout="wide")
    st.title("Hypster Configuration")

    filtered_root = driver.filter_config(final_vars)

    state = {}
    for var in final_vars:
        with st.expander(var, expanded=True):
            render_config_node(filtered_root.children[var], var, state)

    if st.button("Save and Instantiate Configuration"):
        config = collect_config(filtered_root, state)
        st.subheader("Saved Configuration:")
        st.json(config)

        instantiated = driver.instantiate(final_vars)
        st.subheader("Instantiated Objects:")
        for key, value in instantiated.items():
            st.write(f"{key}: {type(value).__name__} object")
            for attr, attr_value in vars(value).items():
                st.write(f"  {attr}: {attr_value}")

        st.success("Configuration saved and instantiated!")

    st.subheader("Configuration Tree Visualization:")
    st.text(visualize_config_tree(filtered_root))


if __name__ == "__main__":
    from examples import configs

    builder = Builder().with_modules(configs)
    driver = builder.build()
    final_vars = ["cache_manager", "llm_driver"]

    main(driver, final_vars)
