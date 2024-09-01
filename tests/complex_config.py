import os

from hypster import HP


class TestClass:
    def __init__(self, hello: str):
        self.hello = hello


def func(a: int = 5):
    return a


def nested_config(hp: HP):
    b = func(6)
    c = TestClass("hey")
    cwd = os.getcwd()
    nested_param = hp.select(["a", "b"], default="a")
