# Agentic AI Lab

A tiny, dependency-free toolkit for building tool-using AI agents in Python.
The whole core is three small modules you can read in one sitting, with a test
suite that runs offline against a scripted mock model.

## Why

Agent frameworks hide the interesting parts. This lab keeps them visible:

- **`agentlab.agent`** - a ReAct-style loop: the model replies with one JSON
  object per turn, either calling a tool or giving a final answer. Tool results
  go back in as observations until the task is done or `max_steps` is hit.
- **`agentlab.tools`** - a registry that validates arguments and turns tool
  errors into observations, so the model can recover from its own mistakes.
- **`agentlab.llm`** - a one-method protocol (`complete(system, messages)`),
  plus `MockLLM`, a scripted stand-in that makes agent behavior fully testable
  without API keys or network.

## Quickstart

```bash
python examples/calculator_agent.py   # watch a scripted agent solve 6 * 7 + 0
python -m pytest                      # run the offline test suite
```

## Using a real model

Write one adapter:

```python
class OpenAILLM:
    def complete(self, system, messages):
        ...  # call your model, return its text

agent = Agent(OpenAILLM(), tools)
```

Nothing else changes - the loop, validation and error handling are
model-agnostic.

## Roadmap

- Planner-executor pattern on top of the same loop
- Retry/backoff and structured-output helpers in the real-model adapter
- A small RAG example (retriever as just another tool)
