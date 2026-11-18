"""Planner-executor pattern on top of the same agent loop.

A planner call turns a task into a short list of sub-tasks (JSON), then a
fresh Agent run executes each one with the earlier results as context. The
plan is validated: bad JSON, an empty plan, or too many steps fails loudly
instead of running something unplanned.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .agent import Agent, AgentResult, extract_json

PLANNER_PROMPT = (
    'Break the task into at most {max_steps} short, ordered sub-tasks. Reply with one '
    'JSON object: {{"plan": ["sub-task 1", "sub-task 2"]}}'
)


class PlanError(ValueError):
    """The planner did not return a usable plan."""


@dataclass
class PlanResult:
    plan: list[str]
    results: list[AgentResult] = field(default_factory=list)
    completed: bool = True  # False when a sub-task did not finish with a final answer

    @property
    def answer(self) -> str | None:
        return self.results[-1].answer if self.results and self.completed else None


def make_plan(llm, task: str, max_steps: int = 5) -> list[str]:
    raw = llm.complete(PLANNER_PROMPT.format(max_steps=max_steps),
                       [{"role": "user", "content": task}])
    parsed = extract_json(raw)
    steps = parsed.get("plan") if parsed else None
    if not isinstance(steps, list) or not steps or not all(isinstance(s, str) and s.strip() for s in steps):
        raise PlanError("planner did not return a non-empty list of sub-tasks")
    if len(steps) > max_steps:
        raise PlanError(f"plan has {len(steps)} steps, limit is {max_steps}")
    return [s.strip() for s in steps]


def plan_and_execute(planner_llm, agent: Agent, task: str, max_steps: int = 5) -> PlanResult:
    """Plan, then run each sub-task through `agent`, stopping at the first
    sub-task that fails to reach a final answer."""
    plan = make_plan(planner_llm, task, max_steps)
    out = PlanResult(plan=plan)
    context: list[str] = []
    for i, sub in enumerate(plan, 1):
        prompt = sub if not context else f"{sub}\n\nResults so far:\n" + "\n".join(context)
        result = agent.run(prompt)
        out.results.append(result)
        if result.answer is None:
            out.completed = False
            break
        context.append(f"{i}. {sub} -> {result.answer}")
    return out
