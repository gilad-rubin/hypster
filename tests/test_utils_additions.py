from hypster.utils import flatten_dict, unflatten_dict


def test_flatten_unflatten_roundtrip():
    nested = {"a": {"b": 1, "c": {"d": 2}}, "x": 3}
    flat = flatten_dict(nested)
    assert flat == {"a.b": 1, "a.c.d": 2, "x": 3}
    assert unflatten_dict(flat) == nested
