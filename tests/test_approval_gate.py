"""Approval gate: sensitive tools run only when the agent's approver allows."""
import json

from agentlab.agent import Agent
from agentlab.llm import MockLLM
from agentlab.tools import ToolRegistry


def make_registry(calls):
    reg = ToolRegistry()

    @reg.register("read-only lookup", {"claim_id": "string"})
    def get_claim(claim_id):
        return {"claim_id": claim_id, "amount": 1200}

    @reg.register("flag a claim for review (side effect)",
                  {"claim_id": "string", "reason": "string"}, sensitive=True)
    def flag_review(claim_id, reason):
        calls.append((claim_id, reason))
        return {"claim_id": claim_id, "status": "flagged"}

    return reg


def reply(action, action_input):
    return json.dumps({"thought": "t", "action": action, "action_input": action_input})


def test_sensitive_tool_runs_when_approver_allows():
    calls = []
    seen = []

    def approver(name, args):
        seen.append((name, args))
        return True

    llm = MockLLM([
        reply("flag_review", {"claim_id": "CLM-1", "reason": "over max"}),
        '{"thought": "done", "final": "flagged"}',
    ])
    result = Agent(llm, make_registry(calls), approver=approver).run("triage")
    assert calls == [("CLM-1", "over max")]  # side effect happened
    assert seen == [("flag_review", {"claim_id": "CLM-1", "reason": "over max"})]
    assert result.steps[0].blocked is False
    assert "flagged" in result.steps[0].observation
    assert result.answer == "flagged"


def test_sensitive_tool_denied_without_approver_fails_closed():
    calls = []
    llm = MockLLM([
        reply("flag_review", {"claim_id": "CLM-1", "reason": "over max"}),
        '{"thought": "cannot flag", "final": "left for a human"}',
    ])
    result = Agent(llm, make_registry(calls)).run("triage")  # no approver
    assert calls == []  # side effect never happened
    assert result.steps[0].blocked is True
    assert "approval was denied" in result.steps[0].observation
    assert result.answer == "left for a human"


def test_sensitive_tool_denied_when_approver_says_no():
    calls = []
    llm = MockLLM([
        reply("flag_review", {"claim_id": "CLM-9", "reason": "unsure"}),
        '{"thought": "ok", "final": "skipped"}',
    ])
    result = Agent(llm, make_registry(calls), approver=lambda n, a: False).run("triage")
    assert calls == []
    assert result.steps[0].blocked is True


def test_approver_error_denies_instead_of_crashing():
    calls = []

    def broken_approver(name, args):
        raise RuntimeError("review UI unreachable")

    llm = MockLLM([
        reply("flag_review", {"claim_id": "CLM-1", "reason": "x"}),
        '{"thought": "ok", "final": "stopped"}',
    ])
    result = Agent(llm, make_registry(calls), approver=broken_approver).run("triage")
    assert calls == []  # fail closed, not fail open
    assert result.steps[0].blocked is True
    assert result.answer == "stopped"


def test_non_sensitive_tools_ignore_the_gate():
    calls = []
    llm = MockLLM([
        reply("get_claim", {"claim_id": "CLM-1"}),
        '{"thought": "have it", "final": "amount is 1200"}',
    ])
    # No approver at all: read-only tools still work.
    result = Agent(llm, make_registry(calls)).run("look up CLM-1")
    assert result.steps[0].blocked is False
    assert "1200" in result.steps[0].observation
    assert result.answer == "amount is 1200"


def test_approver_can_auto_approve_by_rule():
    """A rule-based approver stands in for a human reviewer in demos."""
    calls = []

    def auto_approve_clear_cases(name, args):
        # e.g. only auto-approve flags that cite a concrete policy breach
        return "exceeds annual max" in args.get("reason", "")

    llm = MockLLM([
        reply("flag_review", {"claim_id": "CLM-1", "reason": "unsure"}),
        reply("flag_review", {"claim_id": "CLM-2", "reason": "amount exceeds annual max"}),
        '{"thought": "done", "final": "one flagged, one left for review"}',
    ])
    result = Agent(llm, make_registry(calls), approver=auto_approve_clear_cases).run("triage")
    assert calls == [("CLM-2", "amount exceeds annual max")]
    assert [s.blocked for s in result.steps] == [True, False]
