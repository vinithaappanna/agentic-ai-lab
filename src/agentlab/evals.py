"""A tiny regression harness for agent behavior.

A Case pairs a task and a scripted model with expectations: the final answer
should contain some text, certain tools must (or must not) be called, and
sensitive calls may be required to have been blocked. Because the model is
scripted, cases run offline and fail when the agent loop, tools or approval
policy change behavior. This tests your wiring, not a real model's quality.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from .llm import MockLLM


@dataclass
class Case:
    name: str
    task: str
    script: list
    answer_contains: str | None = None
    must_call: tuple[str, ...] = ()
    must_not_call: tuple[str, ...] = ()
    must_block: tuple[str, ...] = ()       # tools whose calls must all have been blocked
    stopped_reason: str | None = None


@dataclass
class CaseResult:
    name: str
    passed: bool
    failures: list[str] = field(default_factory=list)


def run_case(make_agent: Callable, case: Case) -> CaseResult:
    """make_agent(llm) -> Agent. A fresh agent per case; exceptions are failures."""
    failures: list[str] = []
    try:
        result = make_agent(MockLLM(list(case.script))).run(case.task)
    except Exception as exc:  # a crashing agent is a failing case, not a crashed suite
        return CaseResult(case.name, False, [f"raised {type(exc).__name__}: {exc}"])
    called = [s.action for s in result.steps if not s.blocked]
    blocked = [s.action for s in result.steps if s.blocked]
    if case.answer_contains is not None and case.answer_contains.lower() not in result.answer.lower():
        failures.append(f"answer {result.answer!r} lacks {case.answer_contains!r}")
    for t in case.must_call:
        if t not in called:
            failures.append(f"expected call to {t!r}")
    for t in case.must_not_call:
        if t in called:
            failures.append(f"{t!r} must not run")
    for t in case.must_block:
        if t not in blocked:
            failures.append(f"expected {t!r} to be blocked")
        if t in called:
            failures.append(f"{t!r} ran despite must_block")
    if case.stopped_reason is not None and result.stopped_reason != case.stopped_reason:
        failures.append(f"stopped_reason {result.stopped_reason!r} != {case.stopped_reason!r}")
    return CaseResult(case.name, not failures, failures)


def run_cases(make_agent: Callable, cases: list[Case]) -> list[CaseResult]:
    return [run_case(make_agent, c) for c in cases]


def summarize(results: list[CaseResult]) -> str:
    lines = [f"{'PASS' if r.passed else 'FAIL'}  {r.name}" + "".join(f"\n        - {f}" for f in r.failures)
             for r in results]
    ok = sum(r.passed for r in results)
    lines.append(f"{ok}/{len(results)} cases passed")
    return "\n".join(lines)
