from hypster import HP

def config_func(hp: HP):
    var1 = hp.select(["a", "b"], default="a")
    var2 = hp.number_input(10)