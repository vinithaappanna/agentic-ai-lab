import json

from agentlab.agent import Agent
from agentlab.llm import MockLLM
from agentlab.tools import ToolRegistry


def build(calls):
    reg = ToolRegistry()

    @reg.register("refund (limited)", {"id": "string"}, max_calls=2)
    def refund(id):
        calls.append(id)
        return f"refunded {id}"

    @reg.register("lookup", {"id": "string"})
    def lookup(id):
        return "ok"

    return reg


def act(name, i):
    return json.dumps({"thought": "t", "action": name, "action_input": {"id": i}})


def test_budget_blocks_calls_after_limit():
    calls = []
    reg = build(calls)
    assert reg.call("refund", {"id": "1"}) == "refunded 1"
    assert reg.call("refund", {"id": "2"}) == "refunded 2"
    out = reg.call("refund", {"id": "3"})
    assert "budget of 2" in out and calls == ["1", "2"]


def test_invalid_calls_do_not_spend_budget():
    reg = build([])
    for _ in range(5):
        reg.call("refund", {})  # missing args
    assert reg.call("refund", {"id": "x"}) == "refunded x"


def test_unlimited_tools_unaffected():
    reg = build([])
    assert all(reg.call("lookup", {"id": str(i)}) == "ok" for i in range(20))


def test_budget_resets_each_agent_run():
    calls = []
    reg = build(calls)
    script = [act("refund", "a"), act("refund", "b"), act("refund", "c"),
              '{"thought": "d", "final": "done"}']
    first = Agent(MockLLM(list(script)), reg).run("go")
    assert "budget" in first.steps[2].observation
    second = Agent(MockLLM([act("refund", "z"), '{"thought": "d", "final": "ok"}']), reg).run("go")
    assert second.steps[0].observation == "refunded z"  # fresh run, fresh budget
    assert calls == ["a", "b", "z"]
