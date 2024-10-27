import re
from typing import Any, Dict, List


class HPCallGenerator:
    def generate_code(self) -> str:
        raise NotImplementedError

    def generate_assertion(self) -> str:
        raise NotImplementedError


class SelectGenerator(HPCallGenerator):
    def generate_code(self) -> str:
        return "var = hp.select(['option1', 'option2'], name='var', default='option1')"

    def generate_assertion(self) -> str:
        return "assert result['var'] == 'option1'"


class MultiSelectGenerator(HPCallGenerator):
    def generate_code(self) -> str:
        return "var = hp.multi_select(['option1', 'option2', 'option3'], name='var', default=['option1', 'option2'])"

    def generate_assertion(self) -> str:
        return "assert result['var'] == ['option1', 'option2']"


class NumberInputGenerator(HPCallGenerator):
    def generate_code(self) -> str:
        return "var = hp.number_input(name='var', default=10)"

    def generate_assertion(self) -> str:
        return "assert result['var'] == 10"

    # Add this method to indicate this type doesn't support selections
    def supports_selections(self) -> bool:
        return False


class NamingStrategy:
    def apply(self, code: str) -> str:
        raise NotImplementedError


class ImplicitNaming(NamingStrategy):
    def apply(self, code: str) -> str:
        return code


class ExplicitNaming(NamingStrategy):
    def apply(self, code: str) -> str:
        return code.replace("name='var'", "name='explicit_name'")


class StructureGenerator:
    def apply(self, code: str) -> str:
        raise NotImplementedError


class FlatStructure(StructureGenerator):
    def apply(self, code: str) -> str:
        return f"""@config
def config_func(hp: HP):
    {code}"""


class NestedStructure(StructureGenerator):
    def apply(self, code: str) -> str:
        return f"""@config
def nested_func(hp: HP):
    {code}


nested_func.save("tests/helper_configs/nested_func.py")


@config
def config_func(hp: HP):
    result = hp.propagate("tests/helper_configs/nested_func.py")"""


class LogicGenerator:
    def apply(self, code: str) -> str:
        raise NotImplementedError


class NoConditionals(LogicGenerator):
    def apply(self, code: str) -> str:
        return code


class SimpleConditionals(LogicGenerator):
    def apply(self, code: str) -> str:
        # Extract the function body (everything after the first line containing config_func)
        lines = code.split("\n")
        header = next(line for line in lines if "config_func" in line)
        body = "\n".join(line for line in lines if "config_func" not in line and "@config" not in line)

        condition = "if hp.select(['condition1', 'condition2'], name='condition', default='condition1') == 'condition1'"
        true_branch = body.strip().replace("\n", "\n        ")
        false_branch = true_branch.replace("'option1'", "'option3'").replace("default='option1'", "default='option3'")

        return f"""@config
def config_func(hp: HP):
    {condition}:
        {true_branch}
    else:
        {false_branch}"""

    def get_extra_assertions(self) -> List[str]:
        return [
            "result = config_func(selections={'condition': 'condition1'})",
            "result = config_func(selections={'condition': 'condition2'})",
        ]


class DefaultsGenerator:
    def apply(self, code: str) -> str:
        raise NotImplementedError


class WithDefaults(DefaultsGenerator):
    def apply(self, code: str) -> str:
        return code


class WithoutDefaults(DefaultsGenerator):
    def apply(self, code: str) -> str:
        # Handle both list and non-list defaults
        code = re.sub(r",\s*default=\[[^\]]+\]", "", code)
        code = re.sub(r",\s*default=[^,)]+", "", code)
        return code


class TestCaseGenerator:
    def __init__(
        self,
        hp_call: HPCallGenerator,
        naming: NamingStrategy,
        structure: StructureGenerator,
        logic: LogicGenerator,
        defaults: DefaultsGenerator,
    ):
        self.hp_call = hp_call
        self.naming = naming
        self.structure = structure
        self.logic = logic
        self.defaults = defaults

    def generate_source_code(self) -> str:
        base_code = self.hp_call.generate_code()
        named_code = self.naming.apply(base_code)
        structured_code = self.structure.apply(named_code)
        logic_applied_code = self.logic.apply(structured_code)
        final_code = self.defaults.apply(logic_applied_code)

        return final_code

    def generate_assertions(self) -> List[str]:
        assertions = []
        var_name = "var"

        # For nested structure, we need to prefix with "result."
        if isinstance(self.structure, NestedStructure):
            var_path = f"result.{var_name}"
            result_access = f"result['result']['{var_name}']"
        else:
            var_path = var_name
            result_access = f"result['{var_name}']"

        # Special case: NumberInputGenerator without defaults should raise an error
        if isinstance(self.hp_call, NumberInputGenerator) and isinstance(self.defaults, WithoutDefaults):
            assertions.append("with pytest.raises(ValueError, match='number_input must have a default value'):")
            assertions.append("    config_func()")
            return assertions

        if isinstance(self.defaults, WithoutDefaults):
            assertions.append("with pytest.raises(ValueError):")
            assertions.append("    config_func()")
        else:
            assertions.append("result = config_func()")
            if isinstance(self.structure, NestedStructure):
                assertions.append(
                    f"assert {result_access} == {self.hp_call.generate_assertion().split('==')[1].strip()}"
                )
            else:
                assertions.append(self.hp_call.generate_assertion().replace("'var'", f"'{var_name}'"))

        # Only add selections test for generators that support it
        if not isinstance(self.hp_call, NumberInputGenerator):
            if isinstance(self.logic, SimpleConditionals) and isinstance(self.defaults, WithoutDefaults):
                assertions.append(
                    f"result = config_func(selections={{'condition': 'condition1', '{var_path}': 'option2'}})"
                )
            else:
                assertions.append(f"result = config_func(selections={{'{var_path}': 'option2'}})")
            assertions.append(f"assert {result_access} == 'option2'")

        # Customize override value based on generator type
        override_value = "5" if isinstance(self.hp_call, NumberInputGenerator) else "'option2'"
        if isinstance(self.logic, SimpleConditionals) and isinstance(self.defaults, WithoutDefaults):
            assertions.append(
                f"result = config_func(overrides={{'condition': 'condition1', '{var_path}': {override_value}}})"
            )
        else:
            assertions.append(f"result = config_func(overrides={{'{var_path}': {override_value}}})")
        assertions.append(f"assert {result_access} == {override_value}")

        return assertions


def generate_test_case(hp_call: str, naming: str, structure: str, logic: str, defaults: str) -> Dict[str, Any]:
    hp_call_map = {
        "select": SelectGenerator(),
        "multi_select": MultiSelectGenerator(),
        "number_input": NumberInputGenerator(),
    }
    naming_map = {"implicit": ImplicitNaming(), "explicit": ExplicitNaming()}
    structure_map = {"flat": FlatStructure(), "nested": NestedStructure()}
    logic_map = {"no_conditionals": NoConditionals(), "simple_conditionals": SimpleConditionals()}
    defaults_map = {"with_defaults": WithDefaults(), "without_defaults": WithoutDefaults()}

    generator = TestCaseGenerator(
        hp_call_map[hp_call], naming_map[naming], structure_map[structure], logic_map[logic], defaults_map[defaults]
    )

    return {"source_code": generator.generate_source_code(), "assertions": generator.generate_assertions()}


def write_tests_to_file(all_test_cases):
    with open("tests/test_generated_configs.py", "w") as f:
        # Write import statements at the top of the file
        f.write("import pytest\n")
        f.write("from hypster import HP, config\n\n")

        # Iterate over each test case
        for i, test_case in enumerate(all_test_cases):
            f.write(f"def test_case_{i}():\n")

            # Split the source code into lines
            source_lines = test_case["source_code"].split("\n")
            for line in source_lines:
                if line.strip() == "":
                    f.write("\n")
                else:
                    # All lines inside the test function should be indented by 4 spaces
                    f.write(f"    {line}\n")

            f.write("\n")

            # Write assertions with proper indentation (4 spaces)
            for assertion in test_case["assertions"]:
                f.write(f"    {assertion}\n")
            f.write("\n")


if __name__ == "__main__":
    import os

    # Ensure the tests directory exists
    os.makedirs("tests", exist_ok=True)

    # Generate all test cases
    all_test_cases = []
    for hp_call in ["select", "multi_select", "number_input"]:
        for naming in ["implicit"]:
            for structure in ["flat", "nested"]:
                for logic in ["no_conditionals", "simple_conditionals"]:
                    # Skip the combination of nested structure and conditionals
                    if structure == "nested" and logic == "simple_conditionals":
                        continue

                    for defaults in ["with_defaults", "without_defaults"]:
                        # Skip number_input without defaults
                        if hp_call == "number_input" and defaults == "without_defaults":
                            continue

                        test_case = generate_test_case(hp_call, naming, structure, logic, defaults)
                        all_test_cases.append(test_case)

    # Write tests to file
    write_tests_to_file(all_test_cases)

    print("Tests have been written to 'tests/test_generated_configs.py'. Run them using pytest.")
