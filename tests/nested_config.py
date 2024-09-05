from hypster import HP

def nested_config(hp: HP):
    nested_param = hp.select(["a", "b"], default="a")