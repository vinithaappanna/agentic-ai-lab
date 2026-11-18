"""Audit-trail export for agent runs.

Every run produces a JSONL trace: one line per step (thought, action,
arguments, observation, whether the approval gate blocked it) plus a final
summary line. Regulated domains (healthcare, finance) need this record of
exactly what the agent did and what a human approved or denied.
"""
from __future__ import annotations

import json
from pathlib import Path

from .agent import AgentResult
from .redact import redact_value


def to_dict(result: AgentResult, redact: bool = False) -> dict:
    """A run as a plain dict: JSON-safe, ready to serialize or assert on.

    redact=True masks SSNs, emails and phone numbers in every text field.
    """
    payload = {
        "answer": result.answer,
        "stopped_reason": result.stopped_reason,
        "steps": [
            {
                "thought": step.thought,
                "action": step.action,
                "action_input": step.action_input,
                "observation": step.observation,
                "blocked": step.blocked,
            }
            for step in result.steps
        ],
    }
    return redact_value(payload) if redact else payload


def to_jsonl(result: AgentResult, redact: bool = False) -> str:
    """One JSON object per step, then a summary line."""
    lines = [json.dumps({"type": "step", "index": i, **step})
             for i, step in enumerate(to_dict(result, redact)["steps"])]
    answer = redact_value(result.answer) if redact else result.answer
    lines.append(json.dumps({
        "type": "summary",
        "answer": answer,
        "stopped_reason": result.stopped_reason,
        "step_count": len(result.steps),
        "blocked_count": sum(1 for s in result.steps if s.blocked),
    }))
    return "\n".join(lines) + "\n"


def write_jsonl(result: AgentResult, path, redact: bool = False) -> Path:
    """Write the trace to `path`; returns the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(to_jsonl(result, redact), encoding="utf-8")
    return path
