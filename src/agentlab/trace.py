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


def stats(result: AgentResult) -> dict:
    """Per-run usage summary for dashboards and alerts.

    Each step is counted once, as 'blocked' (approval denied), 'errors'
    (the tool or validation returned an error observation) or 'ok'. Counts
    are per tool, plus totals. Only the observation prefix 'error' is used, so
    tools that return their own error-shaped data count as ok.
    """
    per_tool: dict[str, dict[str, int]] = {}
    for step in result.steps:
        row = per_tool.setdefault(step.action or "(none)", {"ok": 0, "errors": 0, "blocked": 0})
        if step.blocked:
            row["blocked"] += 1
        elif step.observation.startswith("error"):
            row["errors"] += 1
        else:
            row["ok"] += 1
    totals = {k: sum(r[k] for r in per_tool.values()) for k in ("ok", "errors", "blocked")}
    return {"steps": len(result.steps), "stopped_reason": result.stopped_reason,
            "per_tool": dict(sorted(per_tool.items())), **totals}


def audit_summary(jsonl_text: str) -> dict:
    """Read a trace written by to_jsonl back and summarize it for a reviewer:
    steps, how many the approval gate blocked, which actions ran and which
    were blocked, and how the run stopped. Malformed lines are counted, not
    fatal, so one bad line does not hide the rest of the audit trail."""
    steps = blocked = bad = 0
    actions: dict[str, int] = {}
    blocked_actions: dict[str, int] = {}
    stopped = None
    for line in jsonl_text.splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            bad += 1
            continue
        if row.get("type") == "step":
            steps += 1
            name = row.get("action") or "(none)"
            actions[name] = actions.get(name, 0) + 1
            if row.get("blocked"):
                blocked += 1
                blocked_actions[name] = blocked_actions.get(name, 0) + 1
        elif row.get("type") == "summary":
            stopped = row.get("stopped_reason")
    return {"steps": steps, "blocked": blocked, "actions": dict(sorted(actions.items())),
            "blocked_actions": dict(sorted(blocked_actions.items())),
            "stopped_reason": stopped, "malformed_lines": bad}
