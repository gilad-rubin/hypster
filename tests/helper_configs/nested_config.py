from hypster import HP


def nested_config(hp: HP):
    optimizer = hp.select(["adam", "sgd"], default="adam")
    lr = hp.number_input(0.001)
