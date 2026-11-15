"""Tool definitions and a registry with argument validation."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict  # {param_name: "type-name"}; all listed params are required
    func: Callable


@dataclass
class ToolRegistry:
    _tools: dict = field(default_factory=dict)

    def register(self, description: str, parameters: dict | None = None, name: str | None = None):
        """Decorator: @registry.register("what it does", {"x": "number"})"""
        def wrap(func: Callable) -> Callable:
            tool_name = name or func.__name__
            if tool_name in self._tools:
                raise ValueError(f"tool {tool_name!r} is already registered")
            self._tools[tool_name] = Tool(tool_name, description, parameters or {}, func)
            return func
        return wrap

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def describe(self) -> str:
        """Human-readable tool list for the system prompt."""
        lines = []
        for tool in self._tools.values():
            params = ", ".join(f"{k}: {v}" for k, v in tool.parameters.items()) or "none"
            lines.append(f"- {tool.name}({params}): {tool.description}")
        return "\n".join(lines)

    def call(self, name: str, arguments: dict) -> str:
        """Validate and invoke a tool, returning a string observation.

        Errors come back as text instead of raising, so the model can see and
        recover from its own mistakes.
        """
        tool = self._tools.get(name)
        if tool is None:
            return f"error: unknown tool {name!r}"
        if not isinstance(arguments, dict):
            return "error: action_input must be an object"
        expected = set(tool.parameters)
        got = set(arguments)
        missing = expected - got
        extra = got - expected
        if missing:
            return f"error: missing arguments {sorted(missing)}"
        if extra:
            return f"error: unexpected arguments {sorted(extra)}"
        try:
            return str(tool.func(**arguments))
        except Exception as exc:  # the model needs to see tool failures too
            return f"error: {type(exc).__name__}: {exc}"
