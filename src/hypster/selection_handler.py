from collections import OrderedDict


class SelectionHandler:
    def __init__(self, hp_config):
        self.hp_config = hp_config
        self.combinations = hp_config.get_combinations()
        self.defaults = hp_config.get_defaults()
        self.current_options = OrderedDict()
        self.selected_params = OrderedDict()
        self.filtered_combinations = []

    def initialize(self):
        self.filtered_combinations = self.combinations.copy()
        self._run_process(current_param_name=None)

    def _run_process(self, current_param_name=None):
        param_name = self._get_next_param_name(current_param_name)
        while param_name is not None:
            self._process_param(param_name)
            param_name = self._get_next_param_name(param_name)

    def _get_next_param_name(self, current_param_name=None):
        seen = False
        for combination in self.filtered_combinations:
            for param_name in combination:
                if current_param_name is None:
                    return param_name
                elif param_name == current_param_name:
                    seen = True
                elif seen:
                    return param_name
            seen = False
        return None

    def _process_param(self, param_name):
        self.current_options[param_name] = self._collect_options(param_name, self.filtered_combinations)
        self.selected_params[param_name] = self._select_value(
            param_name,
            self.defaults.get(param_name, None),
            self.current_options[param_name],
        )
        filtered_combinations = self._filter_combinations(
            param_name, self.filtered_combinations, self.selected_params[param_name]
        )
        self.filtered_combinations = self._remove_dups(filtered_combinations)
        # TODO: see if we can remove this (redundant)
        self.selected_params = self._update_params_with_filtered_combinations(
            self.selected_params.copy(), self.filtered_combinations.copy()
        )
        self.current_options = self._update_params_with_filtered_combinations(
            self.current_options.copy(), self.filtered_combinations.copy()
        )

    def _update_params_with_filtered_combinations(self, selected_params, filtered_combinations):
        final_selected_params = selected_params.copy()
        to_remove = []
        to_update = {}
        for param_name in final_selected_params.keys():
            if not isinstance(final_selected_params[param_name], dict):
                if all(param_name not in combination for combination in filtered_combinations):
                    to_remove.append(param_name)
            else:
                if param_name in selected_params:
                    selected_params = self._update_params_with_filtered_combinations(
                        selected_params[param_name].copy(),
                        [comb[param_name] for comb in filtered_combinations if param_name in comb],
                    )
                    if len(selected_params) == 0:
                        to_remove.append(param_name)
                    else:
                        to_update[param_name] = selected_params
        result = {k: v for k, v in final_selected_params.items() if k not in to_remove}
        result.update(to_update)
        return result

    def _remove_dups(self, filtered_combinations):
        unique_combinations = []
        seen = set()

        for combination in filtered_combinations:
            combination_hash = self._hash_combination(combination)
            if combination_hash not in seen:
                seen.add(combination_hash)
                unique_combinations.append(combination)

        return unique_combinations

    def _hash_combination(self, combination):
        hash_parts = []
        for key, value in combination.items():
            if isinstance(value, dict):
                hash_parts.append(f"{key}:{self._hash_combination(value)}")
            elif isinstance(value, list):
                hash_parts.append(f"{key}:{tuple(sorted(value))}")
            else:
                hash_parts.append(f"{key}:{value}")
        return tuple(sorted(hash_parts))

    def _process_nested_dict(self, param_name):
        for sub_param in self.combinations[param_name]:
            full_param_name = f"{param_name}.{sub_param}"
            self._process_param(full_param_name)

    def _collect_options(self, param_name, combinations):
        if isinstance(combinations[0][param_name], dict):
            sub_combinations = []
            for combination in combinations:
                if param_name in combination:
                    sub_combinations.append(combination[param_name])

            comb_dct = OrderedDict()
            for combination in sub_combinations:
                for key in combination.keys():
                    comb_dct[key] = self._collect_options(key, sub_combinations)
            return comb_dct

        options_set = set()
        for combination in combinations:
            param = combination[param_name]
            if isinstance(param, list):
                options_set.update(set(param))
            else:
                options_set.add(param)
        return options_set

    def _select_value(self, param_name, param_defaults, current_param_options):
        if isinstance(current_param_options, dict):
            selected_values = OrderedDict()
            for key in current_param_options.keys():
                selected_values[key] = self._select_value(
                    key,
                    param_defaults.get(key, None) if param_defaults else None,
                    current_param_options[key],
                )
            return selected_values
        if param_defaults:
            for default in reversed(param_defaults):
                if isinstance(default, list):
                    if set(default).issubset(current_param_options):
                        return default
                elif default in current_param_options:
                    return default
        # if no default is found, set the first option as default
        return next(iter(current_param_options))

    def _filter_combinations(self, param_name, current_combinations, selected_param_value):
        if isinstance(selected_param_value, dict):
            sub_combinations = current_combinations.copy()
            for key in selected_param_value.keys():
                if any(key in combination[param_name] for combination in sub_combinations):
                    filtered_combinations = []
                    for combination in sub_combinations:
                        result = self._filter_combinations(key, [combination[param_name]], selected_param_value[key])
                        if result:
                            filtered_combinations.append(combination)
                    sub_combinations = [comb for comb in sub_combinations if comb in filtered_combinations]
            return sub_combinations
        else:
            filtered_combinations = [
                comb for comb in current_combinations if self._compare_values(comb[param_name], selected_param_value)
            ]
            return filtered_combinations

    def _compare_values(self, value1, value2):
        if isinstance(value1, list) and isinstance(value2, list):
            return set(value1) == set(value2)
        else:
            return value1 == value2

    def convert_param(self, param_name, selected_value):
        keys = param_name.split(".")
        nested_value = selected_value

        if len(keys) == 1:
            return keys[0], nested_value

        for key in reversed(keys[1:]):
            nested_value = {key: nested_value}

        return keys[0], nested_value

    def update_param(self, param_name, selected_value):
        # this handles dot notation for updating "propagate" calls
        updated_param_name, updated_value = self.convert_param(param_name, selected_value)
        # print(f"Updating {updated_param_name} with {updated_value}")
        self.defaults[updated_param_name] = self._update_defaults(
            updated_param_name,
            self.defaults.get(updated_param_name, None),
            updated_value,
        )
        self.current_options = OrderedDict()
        self.selected_params = OrderedDict()
        self.filtered_combinations = self.combinations.copy()
        self._run_process()

    def _update_defaults(self, param_name, param_defaults, selected_value):
        if isinstance(selected_value, dict):
            if param_defaults is None:
                defaults = {}
            else:
                defaults = param_defaults.copy()

            for key in selected_value.keys():
                defaults[key] = self._update_defaults(key, param_defaults.get(key, None), selected_value[key])
            return defaults

        if param_defaults is None:
            defaults = []
        else:
            defaults = param_defaults.copy()

        defaults.append(selected_value)
        return defaults

    def get_current_state(self):
        return {
            "selected_params": self.selected_params,
            "current_options": self.current_options,
            "filtered_combinations": self.filtered_combinations,
        }
