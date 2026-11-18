import json

import pytest

from agentlab.agent import Agent
from agentlab.llm import MockLLM
from agentlab.planner import PlanError, make_plan, plan_and_execute
from agentlab.tools import ToolRegistry


def plan_llm(steps):
    return MockLLM([json.dumps({"plan": steps})])


def agent_with(responses):
    reg = ToolRegistry()

    @reg.register("add", {"a": "number", "b": "number"})
    def add(a, b):
        return a + b

    return Agent(MockLLM(responses), reg, max_steps=3)


def final(text):
    return json.dumps({"thought": "t", "final": text})


def test_make_plan_validates():
    assert make_plan(plan_llm([" a ", "b"]), "task") == ["a", "b"]
    for bad in ([], "not a list", ["ok", ""], ["a"] * 6):
        with pytest.raises(PlanError):
            make_plan(plan_llm(bad), "task")
    with pytest.raises(PlanError):
        make_plan(MockLLM(["no json at all"]), "task")


def test_executes_steps_in_order_with_context():
    agent = agent_with([final("2"), final("done: 2 + 2 = 4")])
    out = plan_and_execute(plan_llm(["find x", "double it"]), agent, "task")
    assert out.completed and out.answer == "done: 2 + 2 = 4"
    assert len(out.results) == 2
    second_prompt = agent.llm.calls[1][0]["content"]
    assert "Results so far" in second_prompt and "find x -> 2" in second_prompt


def test_stops_at_first_failed_subtask():
    stuck = json.dumps({"thought": "t", "action": "add", "action_input": {"a": 1, "b": 1}})
    agent = agent_with([stuck, stuck, stuck, final("never reached")])
    out = plan_and_execute(plan_llm(["step one", "step two"]), agent, "task")
    assert out.completed is False and out.answer is None
    assert len(out.results) == 1  # step two never ran
