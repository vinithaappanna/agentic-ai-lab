#!/usr/bin/env python3
"""Run a tiny calculator agent end to end with a scripted model.

With a real LLM you would swap MockLLM for an adapter that calls your model;
the agent loop, tools and validation stay exactly the same.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agentlab import Agent, MockLLM, ToolRegistry

tools = ToolRegistry()


@tools.register("add two numbers", {"a": "number", "b": "number"})
def add(a, b):
    return a + b


@tools.register("multiply two numbers", {"a": "number", "b": "number"})
def multiply(a, b):
    return a * b


llm = MockLLM([
    '{"thought": "multiply first", "action": "multiply", "action_input": {"a": 6, "b": 7}}',
    '{"thought": "now add", "action": "add", "action_input": {"a": 42, "b": 0}}',
    '{"thought": "done", "final": "6 * 7 + 0 = 42"}',
])

result = Agent(llm, tools).run("What is 6 * 7 + 0?")
for i, step in enumerate(result.steps, 1):
    print(f"step {i}: {step.action}{step.action_input} -> {step.observation}")
print("answer:", result.answer)
