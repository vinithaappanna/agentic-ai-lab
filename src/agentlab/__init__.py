"""agentlab: a tiny, readable toolkit for building tool-using AI agents."""

from .agent import Agent, AgentResult, Step
from .llm import MockLLM
from .tools import Tool, ToolRegistry

__version__ = "0.1.0"
__all__ = ["Agent", "AgentResult", "Step", "MockLLM", "Tool", "ToolRegistry"]
