from agentlab.agent import Agent, extract_json
from agentlab.llm import MockLLM
from agentlab.tools import ToolRegistry


def make_registry():
    reg = ToolRegistry()

    @reg.register("add two numbers", {"a": "number", "b": "number"})
    def add(a, b):
        return a + b

    return reg


def test_extract_json_tolerates_surrounding_prose():
    assert extract_json('Sure! {"final": "42"} hope that helps') == {"final": "42"}
    assert extract_json("no json here") is None
    assert extract_json("[1, 2, 3]") is None  # only objects count


def test_agent_runs_tool_then_answers():
    llm = MockLLM([
        '{"thought": "need math", "action": "add", "action_input": {"a": 20, "b": 22}}',
        '{"thought": "have it", "final": "The answer is 42"}',
    ])
    result = Agent(llm, make_registry()).run("what is 20 + 22?")
    assert result.answer == "The answer is 42"
    assert result.stopped_reason == "final"
    assert len(result.steps) == 1
    assert result.steps[0].observation == "42"
    # the observation was fed back to the model
    assert "observation: 42" in llm.calls[1][-1]["content"]


def test_agent_recovers_from_invalid_json():
    llm = MockLLM([
        "let me think out loud",
        '{"thought": "retry", "final": "done"}',
    ])
    result = Agent(llm, make_registry()).run("hi")
    assert result.answer == "done"
    assert "not valid JSON" in llm.calls[1][-1]["content"]


def test_agent_survives_unknown_tool():
    llm = MockLLM([
        '{"thought": "x", "action": "teleport", "action_input": {}}',
        '{"thought": "x", "final": "fallback"}',
    ])
    result = Agent(llm, make_registry()).run("go")
    assert result.steps[0].observation.startswith("error: unknown tool")
    assert result.answer == "fallback"


def test_agent_stops_at_max_steps():
    llm = MockLLM([
        '{"thought": "loop", "action": "add", "action_input": {"a": 1, "b": 1}}',
    ] * 10)
    result = Agent(llm, make_registry(), max_steps=3).run("keep adding")
    assert result.stopped_reason == "max_steps"
    assert result.answer is None
    assert len(result.steps) == 3


def test_agent_requires_positive_max_steps():
    import pytest
    with pytest.raises(ValueError):
        Agent(MockLLM([]), make_registry(), max_steps=0)


def test_mock_llm_fallback_terminates_loops():
    result = Agent(MockLLM([]), make_registry()).run("anything")
    assert result.answer == "no scripted answer"
