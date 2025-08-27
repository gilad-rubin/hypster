import pytest

from hypster import HP, instantiate


@pytest.mark.benchmark
def test_bench_int_select_nest_end_to_end():
    def child(hp: HP):
        return hp.int(5, name="x", min=0, max=100)

    def cfg(hp: HP):
        mode = hp.select(["a", "b"], name="mode", default="a")
        val = hp.nest(child, name="child")
        return {"mode": mode, "val": val}

    for _ in range(50):
        out = instantiate(cfg, values={"mode": "b", "child.x": 42})
        assert out["val"] == 42


def test_bench_multi_operations(benchmark):
    def cfg(hp: HP):
        a = hp.multi_int([1, 2, 3, 4], name="a", min=0, max=100)
        b = hp.multi_float([1.0, 2.0, 3.0, 4.0], name="b", min=0.0, max=100.0)
        sel = hp.multi_select(["x", "y", "z"], name="sel", default=["x"])
        return a, b, sel

    def run():
        return instantiate(cfg, values={"a": [9, 8, 7, 6], "b": [0.5, 1.5, 2.5, 3.5]})

    a, b, sel = benchmark(run)
    assert a and b and sel
