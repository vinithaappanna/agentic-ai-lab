"""Tests for the synthetic claims-triage example."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

from claims_triage_agent import build_registry, reviewer_approver, scripted_llm

from agentlab.agent import Agent


def test_tools_report_unknown_ids():
    reg = build_registry()
    assert "no such claim" in reg.call("get_claim", {"claim_id": "NOPE"})
    assert "no such policy" in reg.call("check_policy", {"policy_id": "NOPE"})
    assert "no such claim" in reg.call("flag_review", {"claim_id": "NOPE", "reason": "x"})


def test_coverage_lookup():
    reg = build_registry()
    assert "'covered': True" in reg.call("check_coverage", {"code": "99213"})
    assert "'covered': False" in reg.call("check_coverage", {"code": "00000"})


def test_flag_review_updates_claim_status():
    claims = {"C-9": {"policy_id": "POL-100", "amount": 10, "code": "99213",
                      "status": "submitted"}}
    reg = build_registry(claims=claims)
    out = reg.call("flag_review", {"claim_id": "C-9", "reason": "test"})
    assert "flagged" in out
    assert claims["C-9"]["status"] == "flagged"


def test_full_triage_run_flags_outlier_claim():
    claims = {"CLM-3": {"policy_id": "POL-100", "amount": 99999,
                        "code": "99213", "status": "submitted"}}
    reg = build_registry(claims=claims)
    result = Agent(scripted_llm(), reg, approver=reviewer_approver).run("Triage claim CLM-3")
    assert result.stopped_reason == "final"
    assert "flagged" in result.answer
    assert claims["CLM-3"]["status"] == "flagged"
    # the agent used all four tools in order, and the sensitive one was approved
    actions = [s.action for s in result.steps]
    assert actions == ["get_claim", "check_policy", "check_coverage", "flag_review"]
    assert result.steps[-1].blocked is False


def test_triage_run_without_approver_cannot_flag():
    """Fail closed: with nobody to approve it, flag_review never runs."""
    claims = {"CLM-3": {"policy_id": "POL-100", "amount": 99999,
                        "code": "99213", "status": "submitted"}}
    reg = build_registry(claims=claims)
    result = Agent(scripted_llm(), reg).run("Triage claim CLM-3")  # no approver
    assert claims["CLM-3"]["status"] == "submitted"  # unchanged
    assert result.steps[-1].blocked is True
    assert "approval was denied" in result.steps[-1].observation
