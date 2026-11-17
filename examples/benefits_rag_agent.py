#!/usr/bin/env python3
"""Retrieval-as-a-tool demo: a synthetic health-plan benefits Q&A agent.

The agent searches a small invented policy FAQ and answers only from what it
retrieved. All plan text is made up and illustrative - not real coverage
rules or advice.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agentlab import Agent, MockLLM, ToolRegistry
from agentlab.retrieval import KeywordRetriever

FAQ = {
    "deductible": "The synthetic Silver plan has a 1500 dollar annual deductible; "
                  "preventive visits are covered before the deductible.",
    "prior-auth": "Imaging such as MRI and CT scans requires prior authorization; "
                  "requests are answered within 5 business days.",
    "out-of-network": "Out-of-network care is reimbursed at 60 percent of the allowed amount "
                      "after the deductible.",
    "appeals": "A denied claim can be appealed within 180 days by submitting the denial "
               "letter and supporting notes.",
}


def build_registry(retriever=None) -> ToolRegistry:
    retriever = retriever or KeywordRetriever(FAQ)
    tools = ToolRegistry()

    @tools.register("search the plan FAQ", {"query": "string"})
    def search_faq(query):
        hits = retriever.search(query, k=2)
        return hits or "no matching FAQ entries"

    return tools


def scripted_llm() -> MockLLM:
    return MockLLM([
        '{"thought": "look up MRI rules", "action": "search_faq", '
        '"action_input": {"query": "does an MRI need prior authorization"}}',
        '{"thought": "answer from the retrieved entry", "final": '
        '"Yes: MRI scans need prior authorization, answered within 5 business days '
        '(source: prior-auth)."}',
    ])


if __name__ == "__main__":
    result = Agent(scripted_llm(), build_registry()).run("Does an MRI need prior auth?")
    for i, step in enumerate(result.steps, 1):
        print(f"step {i}: {step.action}{step.action_input} -> {step.observation}")
    print("answer:", result.answer)
