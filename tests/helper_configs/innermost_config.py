from hypster import HP


def innermost_config(hp: HP):
    dropout = hp.select([0.1, 0.2], default=0.1)
