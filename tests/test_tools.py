import pytest

from agentlab.tools import ToolRegistry


@pytest.fixture
def registry():
    reg = ToolRegistry()

    @reg.register("add two numbers", {"a": "number", "b": "number"})
    def add(a, b):
        return a + b

    @reg.register("fail loudly", {})
    def boom():
        raise RuntimeError("kaput")

    return reg


def test_register_and_call(registry):
    assert registry.call("add", {"a": 2, "b": 3}) == "5"


def test_duplicate_name_rejected(registry):
    with pytest.raises(ValueError):
        registry.register("another adder", {"a": "number"}, name="add")(lambda a: a)


def test_unknown_tool_returns_error_text(registry):
    assert "unknown tool" in registry.call("multiply", {"a": 1})


def test_missing_argument_reported(registry):
    assert "missing arguments" in registry.call("add", {"a": 1})


def test_unexpected_argument_reported(registry):
    assert "unexpected arguments" in registry.call("add", {"a": 1, "b": 2, "c": 3})


def test_non_dict_arguments_rejected(registry):
    assert "must be an object" in registry.call("add", [1, 2])


def test_tool_exception_becomes_observation(registry):
    assert "kaput" in registry.call("boom", {})


def test_describe_lists_tools(registry):
    text = registry.describe()
    assert "add(a: number, b: number): add two numbers" in text
    assert "boom" in text
