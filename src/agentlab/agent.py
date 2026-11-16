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
from typing import Callable

from .tools import ToolRegistry

# An approver decides whether a sensitive tool call may run. It receives the
# tool name and the exact arguments the model proposed, and returns True to
# allow the call. In production this is where a human reviews the action;
# in tests and demos it can be a simple rule.
Approver = Callable[[str, dict], bool]

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
    blocked: bool = False  # True when approval for a sensitive tool was denied


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
    def __init__(self, llm, tools: ToolRegistry, max_steps: int = 8,
                 approver: Approver | None = None):
        if max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        self.llm = llm
        self.tools = tools
        self.max_steps = max_steps
        # Human-in-the-loop hook for tools registered with sensitive=True.
        # Fail closed: with no approver, sensitive tools never run.
        self.approver = approver

    def _approved(self, name: str, arguments: dict) -> bool:
        """Decide whether a sensitive tool call may execute.

        Fails closed: no approver, or an approver that errors, means denied -
        a sensitive action must never run because the safety check broke.
        """
        if self.approver is None:
            return False
        if not isinstance(arguments, dict):
            return False
        try:
            return bool(self.approver(name, arguments))
        except Exception:
            return False

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
            tool = self.tools.get(action)
            blocked = False
            if tool is not None and tool.sensitive and not self._approved(action, action_input):
                blocked = True
                observation = (
                    f"error: tool {action!r} is sensitive and approval was denied; "
                    "do not retry the same call - continue without it or finish"
                )
            else:
                observation = self.tools.call(action, action_input)
            steps.append(Step(
                thought=str(parsed.get("thought", "")),
                action=action,
                action_input=action_input if isinstance(action_input, dict) else {},
                observation=observation,
                blocked=blocked,
            ))
            messages.append({"role": "user", "content": f"observation: {observation}"})

        return AgentResult(answer=None, steps=steps, stopped_reason="max_steps")
