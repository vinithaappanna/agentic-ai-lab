import json

import pytest

from agentlab.agent import Agent
from agentlab.llm import MockLLM
from agentlab.tools import ToolRegistry


def reg_with_counter(calls):
    reg = ToolRegistry()

    @reg.register("look up", {"q": "string"})
    def lookup(q):
        calls.append(q)
        return "nothing"

    return reg


def call(q):
    return json.dumps({"thought": "t", "action": "lookup", "action_input": {"q": q}})


def test_identical_repeated_calls_stop_the_run():
    calls = []
    llm = MockLLM([call("x")] * 10)
    result = Agent(llm, reg_with_counter(calls), max_steps=8).run("go")
    assert result.stopped_reason == "loop_detected"
    assert result.answer is None
    assert len(calls) == 3  # stopped after 3 identical calls, not 8


def test_varied_calls_do_not_trip_the_guard():
    calls = []
    llm = MockLLM([call("a"), call("b"), call("a"), call("b"),
                   '{"thought": "d", "final": "ok"}'])
    result = Agent(llm, reg_with_counter(calls)).run("go")
    assert result.stopped_reason == "final"


def test_repeat_count_resets_after_a_different_call():
    calls = []
    llm = MockLLM([call("a"), call("a"), call("b"), call("a"), call("a"),
                   '{"thought": "d", "final": "ok"}'])
    assert Agent(llm, reg_with_counter(calls)).run("go").answer == "ok"


def test_max_repeats_is_configurable_and_validated():
    calls = []
    result = Agent(MockLLM([call("x")] * 5), reg_with_counter(calls), max_repeats=2).run("go")
    assert result.stopped_reason == "loop_detected" and len(calls) == 2
    with pytest.raises(ValueError):
        Agent(MockLLM([]), reg_with_counter([]), max_repeats=1)
