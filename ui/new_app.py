from collections import OrderedDict

import pandas as pd
import streamlit as st


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


def filter_combinations(combinations, selected_params):
    return [comb for comb in combinations if all(comb.get(k) == v for k, v in selected_params.items() if k in comb)]


def select_initial_parameters(combinations, defaults):
    selected = OrderedDict()
    for key, default_value in defaults.items():
        available_options = get_available_options(combinations, selected)
        if key in available_options and default_value in available_options[key]:
            selected[key] = default_value
        elif key in available_options:
            selected[key] = available_options[key][0]
        else:
            break
    return selected


def main():
    st.title("Interactive Parameter Selector")

    valid_combinations = [
        {"param1": "A", "param2": "X"},
        # {"param1": "A", "param2": "X", "param3": "2"},
        {"param1": "A", "param2": "Y", "param3": "1"},
        {"param1": "B", "param2": "X", "param3": "2"},
        {"param1": "B", "param2": "Y", "param3": "1"},
        {"param1": "B", "param2": "Y", "param3": "3"},
    ]

    defaults = {"param1": "A", "param2": "Y", "param3": "2"}

    if "selected_params" not in st.session_state:
        st.session_state.selected_params = select_initial_parameters(valid_combinations, defaults)

    available_options = get_available_options(valid_combinations, st.session_state.selected_params)

    new_selection = OrderedDict()
    for param, options in available_options.items():
        current_value = st.session_state.selected_params.get(param, options[0] if options else None)
        new_value = st.selectbox(
            f"Select {param}", options, index=options.index(current_value) if current_value in options else 0
        )
        new_selection[param] = new_value

        # Update selected_params and rerun if a change is made
        if new_value != st.session_state.selected_params.get(param):
            st.session_state.selected_params = new_selection
            st.rerun()

    valid_combinations = filter_combinations(valid_combinations, st.session_state.selected_params)
    st.write("Valid Combinations with Current Selection:")
    st.write(pd.DataFrame(valid_combinations))


if __name__ == "__main__":
    main()
