#!/usr/bin/env python3
"""Synthetic healthcare claims-triage agent.

A small agent that triages an insurance claim using three tools over a fully
synthetic dataset - no real patient or member data anywhere. With a real LLM
you would swap MockLLM for a model adapter; the tools and triage logic stay.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agentlab import Agent, MockLLM, ToolRegistry

# Synthetic reference data. Every record is invented.
POLICIES = {
    "POL-100": {"holder": "J. Doe", "active": True, "annual_max": 5000},
    "POL-200": {"holder": "A. Roe", "active": False, "annual_max": 2500},
}
CLAIMS = {
    "CLM-1": {"policy_id": "POL-100", "amount": 1200, "code": "99213",
              "status": "submitted"},
    "CLM-2": {"policy_id": "POL-200", "amount": 900, "code": "99214",
              "status": "submitted"},
    "CLM-3": {"policy_id": "POL-100", "amount": 99999, "code": "99213",
              "status": "submitted"},
}
COVERED_CODES = {"99213": "office visit, established patient",
                 "99214": "office visit, moderate complexity"}


def build_registry(claims=None, policies=None) -> ToolRegistry:
    claims = claims if claims is not None else CLAIMS
    policies = policies if policies is not None else POLICIES
    tools = ToolRegistry()

    @tools.register("look up a claim by id", {"claim_id": "string"})
    def get_claim(claim_id):
        claim = claims.get(claim_id)
        return claim if claim else {"error": f"no such claim {claim_id}"}

    @tools.register("check whether a policy is active", {"policy_id": "string"})
    def check_policy(policy_id):
        pol = policies.get(policy_id)
        if not pol:
            return {"error": f"no such policy {policy_id}"}
        return {"active": pol["active"], "annual_max": pol["annual_max"]}

    @tools.register("check whether a procedure code is covered", {"code": "string"})
    def check_coverage(code):
        desc = COVERED_CODES.get(code)
        return {"covered": desc is not None, "description": desc or "unknown code"}

    @tools.register("flag a claim for manual review",
                    {"claim_id": "string", "reason": "string"})
    def flag_review(claim_id, reason):
        if claim_id not in claims:
            return {"error": f"no such claim {claim_id}"}
        claims[claim_id]["status"] = "flagged"
        return {"claim_id": claim_id, "status": "flagged", "reason": reason}

    return tools


def scripted_llm() -> MockLLM:
    return MockLLM([
        '{"thought": "start with the claim", "action": "get_claim", '
        '"action_input": {"claim_id": "CLM-3"}}',
        '{"thought": "check the policy", "action": "check_policy", '
        '"action_input": {"policy_id": "POL-100"}}',
        '{"thought": "check the code", "action": "check_coverage", '
        '"action_input": {"code": "99213"}}',
        '{"thought": "amount far exceeds the annual max; flag it", '
        '"action": "flag_review", "action_input": {"claim_id": "CLM-3", '
        '"reason": "amount 99999 exceeds annual max 5000"}}',
        '{"thought": "done", "final": "CLM-3 flagged for manual review: '
        'amount 99999 exceeds annual max 5000"}',
    ])


if __name__ == "__main__":
    registry = build_registry()
    result = Agent(scripted_llm(), registry).run("Triage claim CLM-3")
    for i, step in enumerate(result.steps, 1):
        print(f"step {i}: {step.action}{step.action_input} -> {step.observation}")
    print("answer:", result.answer)
