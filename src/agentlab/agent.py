"""A small ReAct-style agent loop.

Each turn the model must answer with a single JSON object:

  {"thought": "...", "action": "tool_name", "action_input": {...}}
or
  {"thought": "...", "final": "the answer"}

The loop runs the tool, feeds the observation back, and stops on "final"
or when max_steps is reached.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from .tools import ToolRegistry

SYSTEM_PROMPT = """You are an agent that solves tasks step by step using tools.

Available tools:
{tools}

Reply with exactly one JSON object per turn, no prose around it:
- to use a tool: {{"thought": "why", "action": "tool_name", "action_input": {{"arg": "value"}}}}
- when done:      {{"thought": "why", "final": "your answer"}}
"""


@dataclass
class Step:
    thought: str
    action: str
    action_input: dict
    observation: str


@dataclass
class AgentResult:
    answer: str | None
    steps: list[Step] = field(default_factory=list)
    stopped_reason: str = "final"  # "final" or "max_steps"


def extract_json(text: str) -> dict | None:
    """Pull the first JSON object out of a model reply, tolerating prose around it."""
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        parsed = json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        break
                    return parsed if isinstance(parsed, dict) else None
        start = text.find("{", start + 1)
    return None


class Agent:
    def __init__(self, llm, tools: ToolRegistry, max_steps: int = 8):
        if max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        self.llm = llm
        self.tools = tools
        self.max_steps = max_steps

    def run(self, task: str) -> AgentResult:
        messages = [{"role": "user", "content": task}]
        steps: list[Step] = []
        system = SYSTEM_PROMPT.format(tools=self.tools.describe() or "(no tools)")

        for _ in range(self.max_steps):
            raw = self.llm.complete(system, messages)
            messages.append({"role": "assistant", "content": raw})
            parsed = extract_json(raw)
            if parsed is None:
                messages.append({"role": "user", "content":
                    "observation: your reply was not valid JSON; respond with one JSON object"})
                continue
            if "final" in parsed:
                return AgentResult(answer=str(parsed["final"]), steps=steps,
                                   stopped_reason="final")
            action = str(parsed.get("action", ""))
            action_input = parsed.get("action_input", {})
            observation = self.tools.call(action, action_input)
            steps.append(Step(
                thought=str(parsed.get("thought", "")),
                action=action,
                action_input=action_input if isinstance(action_input, dict) else {},
                observation=observation,
            ))
            messages.append({"role": "user", "content": f"observation: {observation}"})

        return AgentResult(answer=None, steps=steps, stopped_reason="max_steps")
