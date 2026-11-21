# Agentic AI Lab

![tests](https://github.com/vinithaappanna/agentic-ai-lab/actions/workflows/tests.yml/badge.svg)

![How the agent loop works](docs/agent-loop.png)

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

## Approval gate for sensitive tools

Tools that change the world (flag a claim, issue a refund) can be registered
with `sensitive=True`. A sensitive call only runs when the agent's `approver`
- a callable taking `(tool_name, arguments)` - allows that exact call:

```python
def reviewer(name, args):
    ...  # ask a human, or apply a policy rule; return True to allow

agent = Agent(llm, tools, approver=reviewer)
```

The gate **fails closed**: with no approver, or an approver that errors, the
call is denied and the model gets an observation telling it to move on.
Denied calls show up in `result.steps` with `blocked=True`, so every run is
auditable. The claims-triage example shows the pattern end to end.

## Declarative approval policies

`agentlab.policy` builds approvers from small rules instead of hand-written
functions: `allow_tools`, `max_amount("amount", 500)`, `require_text`, combined
with `all_of` / `any_of`, and `audited(rule, log)` to record every decision.
Rules fail closed (missing or non-numeric values, or a rule that raises, deny).
Illustrative building blocks, not compliance logic.

```python
policy = all_of(allow_tools("refund"), max_amount("amount", 100))
agent = Agent(llm, tools, approver=policy)
```

`cumulative_cap("amount", 500)` adds a per-run running-total budget (one instance per run,
placed last in `all_of` so only otherwise-allowed calls count; denied calls add nothing).

## Behavior regression cases

`agentlab.evals` runs `Case`s (task, scripted model, expectations such as
`answer_contains`, `must_call`, `must_not_call`, `must_block`, `call_order`, `stopped_reason`)
against a fresh agent and prints a PASS/FAIL summary. It tests your loop, tools
and approval policy offline; it does not measure a real model's quality.

## Retrieval as a tool

`agentlab.retrieval.KeywordRetriever` is a dependency-free retriever; wrap its
`search` in a tool and the agent can ground answers in documents. See
`examples/benefits_rag_agent.py` (synthetic plan FAQ, illustrative text only).

`KeywordRetriever(docs, weighting="idf")` weights matched terms by rarity, so a rare
term like "deductible" outranks several common ones.

## Planner-executor

`agentlab.planner.plan_and_execute(planner_llm, agent, task)` asks a planner for
a validated list of sub-tasks, then runs each through the agent with earlier
results as context. A bad plan raises `PlanError`; a sub-task that doesn't
finish stops the run and sets `completed=False`. `retries=N` re-runs an unfinished
sub-task up to N more times (`PlanResult.attempts` records runs used); a
`loop_detected` stop is never retried.

## Per-run call budgets

`@tools.register(..., max_calls=2)` caps how often a tool can run in one agent
run (counters reset at the start of each `Agent.run`). Over-budget calls return
an error observation instead of executing - a cheap safety net for refunds,
flags or paid API calls. Invalid calls do not spend budget.

## Loop guard

If the model issues the identical tool call `max_repeats` times in a row
(default 3), the run stops with `stopped_reason="loop_detected"` instead of
burning steps or repeating a side effect.

## Audit trail

`agentlab.trace` exports any run as JSONL - one line per step (thought,
action, arguments, observation, blocked flag) plus a summary line:

```python
from agentlab import trace
trace.write_jsonl(result, "traces/run.jsonl")
```

`trace.stats(result)` gives per-tool counts of ok, error and blocked steps for
dashboards or alerts. `trace.audit_summary(jsonl_text)` reads a saved JSONL trace
back for a reviewer: step count, blocked steps, actions run and blocked, stop reason.

Pass `redact=True` to mask SSNs, emails and phone numbers (`agentlab.redact`,
pattern-based safety net, not a compliance guarantee; use synthetic data only).
Opt-in `HEALTH_PATTERNS` (member id, MRN, date of birth; illustrative formats) go
in through `extra=`.

Regulated workflows (claims, refunds) get a durable record of what the
agent did and what the approver allowed or denied.

## Quickstart

```bash
python examples/calculator_agent.py   # watch a scripted agent solve 6 * 7 + 0
python examples/claims_triage_agent.py  # synthetic claims triage + approval gate
python examples/expense_audit_agent.py  # synthetic expense auditing (finance)
python examples/benefits_rag_agent.py  # retrieval-as-a-tool, synthetic benefits FAQ
python examples/guardrail_evals.py     # policy + eval cases: guardrail regression suite
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

- Retry/backoff and structured-output helpers in the real-model adapter
