"""Tests for the synthetic expense-audit example."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

from expense_audit_agent import build_registry, scripted_llm

from agentlab.agent import Agent


def test_tools_report_unknown_ids():
    reg = build_registry()
    assert "no such report" in reg.call("get_report", {"report_id": "NOPE"})
    assert "unknown category" in reg.call("get_limit", {"category": "NOPE"})
    assert "unknown currency" in reg.call("to_usd", {"amount": 1, "currency": "XXX"})
    assert "no such report" in reg.call(
        "decide", {"report_id": "NOPE", "decision": "approved", "note": "x"})


def test_currency_conversion():
    reg = build_registry()
    assert "324.0" in reg.call("to_usd", {"amount": 300.0, "currency": "EUR"})


def test_decision_validation():
    reports = {"E-9": {"employee": "T", "category": "meals", "amount": 5.0,
                       "currency": "USD", "status": "submitted"}}
    reg = build_registry(reports=reports)
    assert "must be approved or escalated" in reg.call(
        "decide", {"report_id": "E-9", "decision": "maybe", "note": "x"})
    reg.call("decide", {"report_id": "E-9", "decision": "approved", "note": "ok"})
    assert reports["E-9"]["status"] == "approved"


def test_full_audit_run_approves_in_limit_report():
    reports = {"EXP-2": {"employee": "K. Tan", "category": "travel",
                         "amount": 300.0, "currency": "EUR",
                         "status": "submitted"}}
    reg = build_registry(reports=reports)
    result = Agent(scripted_llm(), reg).run("Audit expense report EXP-2")
    assert result.stopped_reason == "final"
    assert "approved" in result.answer
    assert reports["EXP-2"]["status"] == "approved"
    actions = [s.action for s in result.steps]
    assert actions == ["get_report", "get_limit", "to_usd", "decide"]
