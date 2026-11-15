"""LLM interface used by the agent loop.

Any object with `complete(system: str, messages: list[dict]) -> str` works,
so swapping the mock for a real model (OpenAI, Anthropic, a local model)
means writing one thin adapter, not touching the agent.
"""
from __future__ import annotations

from typing import Callable, Protocol, Sequence, Union

Message = dict
Response = Union[str, Callable[[Sequence[Message]], str]]


class LLM(Protocol):
    def complete(self, system: str, messages: list[Message]) -> str: ...


class MockLLM:
    """A scripted LLM for tests and examples.

    Responses are returned in order. A response can be a string or a callable
    that receives the message list and returns a string. When the script runs
    out, the mock replies with a final answer so loops still terminate.
    """

    def __init__(self, responses: Sequence[Response]):
        self._responses = list(responses)
        self.calls: list[list[Message]] = []

    def complete(self, system: str, messages: list[Message]) -> str:
        self.calls.append(list(messages))
        if not self._responses:
            return '{"thought": "script exhausted", "final": "no scripted answer"}'
        response = self._responses.pop(0)
        return response(messages) if callable(response) else response
