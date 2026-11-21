import json

from agentlab import Agent, MockLLM, ToolRegistry
from agentlab.policy import all_of, allow_tools, any_of, audited, max_amount, require_text


def test_max_amount_boundaries_and_scope():
    r = max_amount("amount", 500, tools=("refund",))
    assert r("refund", {"amount": 500}) is True
    assert r("refund", {"amount": 500.01}) is False
    assert r("refund", {}) is False                    # missing: deny
    assert r("refund", {"amount": "10"}) is False      # non-numeric: deny
    assert r("refund", {"amount": True}) is False      # bool is not a number here
    assert r("other_tool", {"amount": 10**9}) is True  # out of scope


def test_all_of_any_of_and_fail_closed():
    ok = lambda n, a: True
    no = lambda n, a: False
    boom = lambda n, a: 1 / 0
    assert all_of(ok, ok)("t", {}) and not all_of(ok, no)("t", {})
    assert not all_of(ok, boom)("t", {})               # raising rule denies
    assert any_of(no, ok)("t", {}) and not any_of(no, no)("t", {})
    assert any_of(boom, ok)("t", {})                   # broken rule skipped, other allows
    assert not any_of(boom)("t", {})


def test_allow_tools_and_require_text():
    assert allow_tools("flag")("flag", {}) and not allow_tools("flag")("refund", {})
    r = require_text("reason", "exceeds annual max")
    assert r("flag", {"reason": "Claim EXCEEDS ANNUAL MAX of plan"})
    assert not r("flag", {"reason": "seems odd"}) and not r("flag", {})


def test_policy_drives_agent_gate_and_audit_log():
    tools = ToolRegistry()
    done = []

    @tools.register("refund a synthetic customer", {"amount": "number"}, sensitive=True)
    def refund(amount):
        done.append(amount)
        return {"refunded": amount}

    log = []
    policy = audited(all_of(allow_tools("refund"), max_amount("amount", 100)), log)
    steps = [json.dumps({"thought": "small", "action": "refund", "action_input": {"amount": 40}}),
             json.dumps({"thought": "big", "action": "refund", "action_input": {"amount": 4000}}),
             json.dumps({"thought": "done", "final": "ok"})]
    result = Agent(MockLLM(steps), tools, approver=policy).run("refund please")
    assert done == [40]                                # the 4000 call never ran
    assert [d["allowed"] for d in log] == [True, False]
    assert any(s.blocked for s in result.steps)


def test_cumulative_cap_tracks_running_total_and_counts_only_allowed_calls():
    from agentlab.policy import cumulative_cap
    cap = cumulative_cap("amount", 100, tools=("refund",))
    assert cap("refund", {"amount": 60}) is True
    assert cap("refund", {"amount": 50}) is False              # 110 > 100: denied, adds nothing
    assert cap.state["spent"] == 60
    assert cap("refund", {"amount": 40}) is True                # exactly at the cap
    assert cap("refund", {"amount": 0.01}) is False
    assert cap("lookup", {"amount": 999}) is True               # out of scope: passes, not counted
    for bad in ({}, {"amount": "5"}, {"amount": True}, {"amount": -5}):
        assert cap("refund", bad) is False
    assert cap.state["spent"] == 100


def test_cumulative_cap_last_in_all_of_ignores_calls_other_rules_deny():
    from agentlab.policy import cumulative_cap
    cap = cumulative_cap("amount", 100)
    pol = all_of(allow_tools("refund"), max_amount("amount", 80), cap)
    assert pol("refund", {"amount": 90}) is False               # per-call cap denies first
    assert pol("other", {"amount": 10}) is False                # tool not allowed
    assert cap.state["spent"] == 0
    assert pol("refund", {"amount": 70}) is True
    assert pol("refund", {"amount": 40}) is False               # would reach 110
    assert cap.state["spent"] == 70
