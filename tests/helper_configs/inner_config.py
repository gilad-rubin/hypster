from hypster import HP


def inner_config(hp: HP):
    activation = hp.select(["relu", "tanh"], default="relu")
    dropout = hp.select([0.1, 0.2], default=0.1)
