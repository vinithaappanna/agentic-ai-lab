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
    def rule(name: str, args: dict) -> bool:
        try:
            return all(r(name, args) for r in rules)
        except Exception:
            return False
    return rule


def any_of(*rules: Rule) -> Rule:
    def rule(name: str, args: dict) -> bool:
        for r in rules:
            try:
                if r(name, args):
                    return True
            except Exception:
                continue
        return False
    return rule


def audited(rule: Rule, log: list) -> Rule:
    """Wrap a rule so each decision is appended to `log` as a dict."""
    def wrapped(name: str, args: dict) -> bool:
        decision = bool(rule(name, args))
        log.append({"tool": name, "arguments": dict(args), "allowed": decision})
        return decision
    return wrapped
