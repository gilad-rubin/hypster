import pytest

from hypster import HP


def test_bool_input_basic():
    hp = HP(final_vars=[], selections={}, overrides={})
    result = hp.bool_input(default=True, name="test_bool")
    assert result is True
    assert hp.snapshot["test_bool"] is True


def test_bool_input_with_override():
    hp = HP(final_vars=[], selections={}, overrides={"test_bool": False})
    result = hp.bool_input(default=True, name="test_bool")
    assert result is False
    assert hp.snapshot["test_bool"] is False


def test_bool_input_invalid_default():
    hp = HP(final_vars=[], selections={}, overrides={})
    with pytest.raises(TypeError):
        hp.bool_input(default="not a bool", name="test_bool")


def test_multi_bool_basic():
    hp = HP(final_vars=[], selections={}, overrides={})
    result = hp.multi_bool(default=[True, False], name="test_multi_bool")
    assert result == [True, False]
    assert hp.snapshot["test_multi_bool"] == [True, False]


def test_multi_bool_with_override():
    hp = HP(final_vars=[], selections={}, overrides={"test_multi_bool": [False, False]})
    result = hp.multi_bool(default=[True, True], name="test_multi_bool")
    assert result == [False, False]
    assert hp.snapshot["test_multi_bool"] == [False, False]


def test_multi_bool_invalid_default():
    hp = HP(final_vars=[], selections={}, overrides={})
    with pytest.raises(TypeError):
        hp.multi_bool(default=[True, "not a bool"], name="test_multi_bool")


def test_bool_input_explore_mode():
    hp = HP(final_vars=[], selections={}, overrides={}, explore_mode=True)
    result = hp.bool_input(default=True, name="test_bool")
    assert result is True


def test_multi_bool_explore_mode():
    hp = HP(final_vars=[], selections={}, overrides={}, explore_mode=True)
    result = hp.multi_bool(default=[True, False], name="test_multi_bool")
    assert result == [True, False]
