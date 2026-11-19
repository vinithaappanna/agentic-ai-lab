import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

import guardrail_evals as g

from agentlab.evals import Case, run_case, run_cases


def test_all_guardrail_cases_pass_and_only_valid_call_is_recorded():
    recorded = []
    results = run_cases(g.make_agent_factory(recorded), g.CASES)
    assert all(r.passed for r in results), [r for r in results if not r.passed]
    assert recorded == [("CLM-1", 100)]      # three blocked calls recorded nothing


def test_suite_detects_a_weakened_policy(monkeypatch):
    monkeypatch.setattr(g, "LIMIT", 10_000.0)   # someone raises the cap
    results = run_cases(g.make_agent_factory([]), g.CASES)
    failed = [r.name for r in results if not r.passed]
    assert failed == ["over the limit is blocked"]


def test_call_budget_blocks_second_recommendation():
    recorded = []
    twice = Case("two adjustments", "t",
                 [g.step("recommend_adjustment", claim_id="C", amount=10, reason="contract rate"),
                  g.step("recommend_adjustment", claim_id="C", amount=20, reason="contract rate"), g.DONE])
    run_case(g.make_agent_factory(recorded), twice)
    assert recorded == [("C", 10)]
