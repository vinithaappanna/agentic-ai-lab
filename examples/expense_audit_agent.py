#!/usr/bin/env python3
"""Synthetic expense-audit agent.

A small agent that audits employee expense reports against a synthetic
policy: fetch a report, look up the category limit, convert currency when
needed, and approve or escalate. All data is invented - no real financial
records. Swap MockLLM for a real model adapter and the loop stays the same.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agentlab import Agent, MockLLM, ToolRegistry

# Synthetic reference data. Every record is invented.
REPORTS = {
    "EXP-1": {"employee": "E. Rivera", "category": "meals", "amount": 84.0,
              "currency": "USD", "status": "submitted"},
    "EXP-2": {"employee": "K. Tan", "category": "travel", "amount": 300.0,
              "currency": "EUR", "status": "submitted"},
    "EXP-3": {"employee": "S. Patel", "category": "lodging", "amount": 610.0,
              "currency": "USD", "status": "submitted"},
}
CATEGORY_LIMITS_USD = {"meals": 100.0, "travel": 500.0, "lodging": 400.0}
FX_TO_USD = {"USD": 1.0, "EUR": 1.08, "GBP": 1.27}


def build_registry(reports=None, limits=None, fx=None) -> ToolRegistry:
    reports = reports if reports is not None else REPORTS
    limits = limits if limits is not None else CATEGORY_LIMITS_USD
    fx = fx if fx is not None else FX_TO_USD
    tools = ToolRegistry()

    @tools.register("look up an expense report by id", {"report_id": "string"})
    def get_report(report_id):
        rep = reports.get(report_id)
        return rep if rep else {"error": f"no such report {report_id}"}

    @tools.register("get the USD limit for an expense category",
                    {"category": "string"})
    def get_limit(category):
        limit = limits.get(category)
        if limit is None:
            return {"error": f"unknown category {category}"}
        return {"category": category, "limit_usd": limit}

    @tools.register("convert an amount to USD",
                    {"amount": "number", "currency": "string"})
    def to_usd(amount, currency):
        rate = fx.get(currency)
        if rate is None:
            return {"error": f"unknown currency {currency}"}
        return {"amount_usd": round(amount * rate, 2), "rate": rate}

    @tools.register("record the audit decision",
                    {"report_id": "string", "decision": "string", "note": "string"})
    def decide(report_id, decision, note):
        if report_id not in reports:
            return {"error": f"no such report {report_id}"}
        if decision not in ("approved", "escalated"):
            return {"error": "decision must be approved or escalated"}
        reports[report_id]["status"] = decision
        return {"report_id": report_id, "status": decision, "note": note}

    return tools


def scripted_llm() -> MockLLM:
    return MockLLM([
        '{"thought": "fetch the report", "action": "get_report", '
        '"action_input": {"report_id": "EXP-2"}}',
        '{"thought": "need the travel limit", "action": "get_limit", '
        '"action_input": {"category": "travel"}}',
        '{"thought": "amount is EUR, convert", "action": "to_usd", '
        '"action_input": {"amount": 300.0, "currency": "EUR"}}',
        '{"thought": "324 USD is under the 500 USD limit", '
        '"action": "decide", "action_input": {"report_id": "EXP-2", '
        '"decision": "approved", "note": "324.00 USD within 500 USD travel limit"}}',
        '{"thought": "done", "final": "EXP-2 approved: 324.00 USD is within '
        'the 500 USD travel limit"}',
    ])


if __name__ == "__main__":
    registry = build_registry()
    result = Agent(scripted_llm(), registry).run("Audit expense report EXP-2")
    for i, step in enumerate(result.steps, 1):
        print(f"step {i}: {step.action}{step.action_input} -> {step.observation}")
    print("answer:", result.answer)
