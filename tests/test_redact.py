import json

from agentlab.agent import Agent
from agentlab.llm import MockLLM
from agentlab.redact import redact_text, redact_value
from agentlab.tools import ToolRegistry
from agentlab.trace import to_dict, to_jsonl, write_jsonl


def test_redacts_ssn_email_phone():
    out = redact_text("SSN 123-45-6789, mail a.b+c@ex.co.uk, call (555) 123-4567 or 555-123-4567")
    assert "123-45-6789" not in out and "ex.co.uk" not in out and "4567" not in out
    assert out.count("[PHONE]") == 2 and "[SSN]" in out and "[EMAIL]" in out


def test_leaves_ordinary_numbers_alone():
    text = "claim CLM-61117 paid 1500.25 on 2026-11-18, code 99213"
    assert redact_text(text) == text


def test_extra_patterns_and_nested_values():
    out = redact_value({"a": ["MBR-0042", {"b": "ok", "n": 5}]}, {"MEMBER": r"MBR-\d+"})
    assert out == {"a": ["[MEMBER]", {"b": "ok", "n": 5}]}


def run():
    reg = ToolRegistry()

    @reg.register("lookup", {"q": "string"})
    def lookup(q):
        return "member jane@example.com, SSN 123-45-6789"

    llm = MockLLM([
        '{"thought": "email jane@example.com", "action": "lookup", "action_input": {"q": "555-123-4567"}}',
        '{"thought": "d", "final": "reached jane@example.com"}',
    ])
    return Agent(llm, reg).run("go")


def test_trace_redaction_is_opt_in_and_covers_all_fields():
    result = run()
    assert "jane@example.com" in to_jsonl(result)  # default unchanged
    red = to_jsonl(result, redact=True)
    for secret in ("jane@example.com", "123-45-6789", "555-123-4567"):
        assert secret not in red
    rows = [json.loads(l) for l in red.splitlines()]
    assert rows[-1]["answer"] == "reached [EMAIL]"
    assert to_dict(result, redact=True)["steps"][0]["action_input"] == {"q": "[PHONE]"}
    assert result.steps[0].observation.startswith("member jane@")  # original untouched


def test_write_jsonl_redact(tmp_path):
    out = write_jsonl(run(), tmp_path / "t.jsonl", redact=True)
    assert "jane@example.com" not in out.read_text()


def test_health_patterns_mask_ids_and_dates_only_when_requested():
    from agentlab.redact import HEALTH_PATTERNS, redact_text, redact_value
    text = "Member MBR-0042 (MRN: 1234567) born 3/14/1985, call 555-123-4567."
    default = redact_text(text)
    assert "MBR-0042" in default and "[PHONE]" in default            # health patterns are opt-in
    out = redact_text(text, extra=HEALTH_PATTERNS)
    assert out == "Member [MEMBER_ID] ([MRN]) born [DOB], call [PHONE]."
    nested = redact_value({"notes": ["see MBR-100", {"dob": "12/31/2001"}]}, HEALTH_PATTERNS)
    assert nested == {"notes": ["see [MEMBER_ID]", {"dob": "[DOB]"}]}


def test_health_patterns_do_not_overmatch():
    from agentlab.redact import HEALTH_PATTERNS, redact_text
    safe = "Claim CLM-00042 on 13/45/2020 and MBR-12 and MRN 12 and version 3/14/1785"
    assert redact_text(safe, extra=HEALTH_PATTERNS) == safe          # too short, invalid month/day, year out of range
