from hypster import HP


def nested_config(hp: HP):
    nested_param = hp.select(["a", "b"], default="a")
    nested_number = hp.number_input(default=1.0)
