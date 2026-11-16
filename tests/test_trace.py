"""Audit-trail export: structure, blocked flags, and file output."""
import json

from agentlab.agent import Agent
from agentlab.llm import MockLLM
from agentlab.tools import ToolRegistry
from agentlab.trace import to_dict, to_jsonl, write_jsonl


def run_with_one_blocked_step():
    reg = ToolRegistry()

    @reg.register("look something up", {"q": "string"})
    def search(q):
        return f"results for {q}"

    @reg.register("do something regulated", {"x": "string"}, sensitive=True)
    def act(x):
        return f"did {x}"

    llm = MockLLM([
        '{"thought": "research", "action": "search", "action_input": {"q": "policy"}}',
        '{"thought": "act", "action": "act", "action_input": {"x": "refund"}}',
        '{"thought": "wrap up", "final": "done, one action needed approval"}',
    ])
    return Agent(llm, reg).run("audit me")  # no approver -> sensitive call denied


def test_to_dict_is_json_safe_and_marks_blocked_steps():
    result = run_with_one_blocked_step()
    payload = json.loads(json.dumps(to_dict(result)))  # must round-trip
    assert payload["answer"] == "done, one action needed approval"
    assert payload["stopped_reason"] == "final"
    assert [s["blocked"] for s in payload["steps"]] == [False, True]
    assert payload["steps"][1]["action"] == "act"
    assert "approval was denied" in payload["steps"][1]["observation"]


def test_to_jsonl_has_a_line_per_step_plus_summary():
    result = run_with_one_blocked_step()
    rows = [json.loads(line) for line in to_jsonl(result).splitlines()]
    assert [r["type"] for r in rows] == ["step", "step", "summary"]
    assert rows[0]["index"] == 0 and rows[1]["index"] == 1
    assert rows[-1]["step_count"] == 2
    assert rows[-1]["blocked_count"] == 1


def test_write_jsonl_creates_parent_dirs(tmp_path):
    result = run_with_one_blocked_step()
    out = write_jsonl(result, tmp_path / "traces" / "run.jsonl")
    rows = out.read_text(encoding="utf-8").splitlines()
    assert len(rows) == 3
    assert json.loads(rows[-1])["answer"] == "done, one action needed approval"
