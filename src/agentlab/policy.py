"""Composable, declarative approvers for sensitive tool calls.

Each rule is a callable (tool_name, arguments) -> bool. `all_of` allows a call
only if every rule allows it; `any_of` if at least one does. Everything fails
closed: a rule that raises, or a missing/non-numeric argument, means deny.
Rules are illustrative building blocks, not compliance logic. Use synthetic
data only.
"""
from __future__ import annotations

from typing import Callable

Rule = Callable[[str, dict], bool]


class UnsupportedComposition(RuntimeError):
    """A stateful rule (one that spends a budget) was called outside
    all_of / any_of / audited. Raised instead of guessing, because a wrapper
    that cannot tell "this rule allowed" from "the whole decision allowed"
    could spend budget on a call that is then denied. It is never swallowed
    by all_of / any_of (which otherwise treat a raising rule as deny)."""


def _plan(rule: Rule, name: str, args: dict):
    """Evaluate a rule without side effects. Returns a list of commit
    callables when the rule allows (empty for stateless rules) or None when it
    denies. Stateful rules and the composites expose `_plan`; plain rules are
    called directly. A plain rule that raises denies."""
    planner = getattr(rule, "_plan", None)
    if planner is not None:
        return planner(name, args)
    try:
        return [] if rule(name, args) else None
    except UnsupportedComposition:
        raise
    except Exception:
        return None


def _with_plan(planner: Callable) -> Callable[[Rule], Rule]:
    def attach(fn: Rule) -> Rule:
        fn._plan = planner          # type: ignore[attr-defined]
        return fn
    return attach


def _run_plan(planner: Callable, name: str, args: dict) -> bool:
    commits = planner(name, args)
    if commits is None:
        return False
    for c in commits:               # budgets are spent only now, after the whole decision allowed
        c()
    return True


def allow_tools(*names: str) -> Rule:
    """Allow only the named tools."""
    allowed = set(names)
    return lambda name, args: name in allowed


def max_amount(field: str, limit: float, tools: tuple[str, ...] | None = None) -> Rule:
    """Allow calls whose numeric `field` is at most `limit`. Applies to the
    named tools only (others pass); a missing or non-numeric value is denied."""
    def rule(name: str, args: dict) -> bool:
        if tools is not None and name not in tools:
            return True
        value = args.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return False
        return value <= limit
    return rule


def require_text(field: str, needle: str) -> Rule:
    """Allow calls whose string `field` contains `needle` (case-insensitive),
    e.g. a flag reason must cite a policy breach."""
    return lambda name, args: needle.lower() in str(args.get(field, "")).lower()


def all_of(*rules: Rule) -> Rule:
    def planner(name: str, args: dict):
        commits: list = []
        for r in rules:
            sub = _plan(r, name, args)
            if sub is None:
                return None          # one denial denies the call and spends nothing
            commits.extend(sub)
        return commits

    return _with_plan(planner)(lambda name, args: _run_plan(planner, name, args))


def any_of(*rules: Rule) -> Rule:
    def planner(name: str, args: dict):
        for r in rules:
            sub = _plan(r, name, args)
            if sub is not None:
                return sub           # only the winning branch spends budget
        return None

    return _with_plan(planner)(lambda name, args: _run_plan(planner, name, args))


def audited(rule: Rule, log: list) -> Rule:
    """Wrap a rule so each decision is appended to `log` as a dict. Inside a
    composite, the logged decision is this rule's own, made before the whole
    composite has decided."""
    def planner(name: str, args: dict):
        sub = _plan(rule, name, args)
        log.append({"tool": name, "arguments": dict(args), "allowed": sub is not None})
        return sub

    return _with_plan(planner)(lambda name, args: _run_plan(planner, name, args))


def cumulative_cap(field: str, limit: float, tools: tuple[str, ...] | None = None) -> Rule:
    """Stateful rule: allow a call only while the running total of numeric
    `field` across approved calls stays within `limit` (a per-run budget on
    top of a per-call `max_amount`). Create one per run. Budget is spent only
    when the WHOLE decision allows, so placement inside all_of / any_of /
    audited does not matter: a call denied by another rule, an any_of branch
    that loses, or a failure after the cap consumes nothing. Missing,
    non-numeric or negative values are denied. Calling the rule directly (or
    through a wrapper that is not all_of / any_of / audited) raises
    UnsupportedComposition rather than risking a wrong spend. Running total:
    `rule.state["spent"]`. Illustrative; nothing here moves money."""
    state = {"spent": 0.0}

    def planner(name: str, args: dict):
        if tools is not None and name not in tools:
            return []
        value = args.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
            return None
        if state["spent"] + value > limit:
            return None

        def commit(v=value):
            state["spent"] += v
        return [commit]

    def rule(name: str, args: dict) -> bool:
        raise UnsupportedComposition(
            "cumulative_cap must be used inside all_of / any_of / audited; "
            "wrap it as all_of(cumulative_cap(...)) to use it on its own")

    rule.state = state              # type: ignore[attr-defined]
    return _with_plan(planner)(rule)


def require_fields(*fields: str, tools: tuple[str, ...] | None = None) -> Rule:
    """Deny calls missing any of the named arguments, or with an empty string
    value. Useful for regulated tools that must always carry an id or a reason,
    e.g. require_fields("claim_id", "reason"). Applies to the named tools only
    (others pass)."""
    def rule(name: str, args: dict) -> bool:
        if tools is not None and name not in tools:
            return True
        return all(args.get(f) not in (None, "") for f in fields)
    return rule


def max_calls(limit: int, tools: tuple[str, ...] | None = None) -> Rule:
    """Stateful rule: allow at most `limit` approved calls per run (a call
    count, complementing the amount-based `cumulative_cap`). Create one per
    run. Like cumulative_cap it is composition-safe: a call is counted only
    when the WHOLE decision allows, and calling it outside all_of / any_of /
    audited raises UnsupportedComposition. Applies to the named tools only
    (others pass and are not counted). Count: `rule.state["calls"]`."""
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
        raise ValueError("limit must be a non-negative integer")
    state = {"calls": 0}

    def planner(name: str, args: dict):
        if tools is not None and name not in tools:
            return []
        if state["calls"] + 1 > limit:
            return None

        def commit():
            state["calls"] += 1
        return [commit]

    def rule(name: str, args: dict) -> bool:
        raise UnsupportedComposition(
            "max_calls must be used inside all_of / any_of / audited; "
            "wrap it as all_of(max_calls(...)) to use it on its own")

    rule.state = state              # type: ignore[attr-defined]
    return _with_plan(planner)(rule)


def forbid_pattern(field: str, pattern: str, tools: tuple[str, ...] | None = None) -> Rule:
    """Deny calls whose string `field` matches the regex `pattern` anywhere,
    e.g. keep an SSN-shaped value out of a free-text note:
    forbid_pattern("note", r"\\b\\d{3}-\\d{2}-\\d{4}\\b"). Applies to the named tools only
    (others pass). A missing field passes (use require_fields to demand it);
    a non-string value is denied, since it cannot be checked. Illustrative
    guardrail, not DLP."""
    import re
    rx = re.compile(pattern)

    def rule(name: str, args: dict) -> bool:
        if tools is not None and name not in tools:
            return True
        if field not in args:
            return True
        value = args[field]
        return isinstance(value, str) and rx.search(value) is None
    return rule
