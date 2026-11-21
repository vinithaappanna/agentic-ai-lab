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
    pol = all_of(cap)
    assert pol("refund", {"amount": 60}) is True
    assert pol("refund", {"amount": 50}) is False              # 110 > 100: denied, adds nothing
    assert cap.state["spent"] == 60
    assert pol("refund", {"amount": 40}) is True                # exactly at the cap
    assert pol("refund", {"amount": 0.01}) is False
    assert pol("lookup", {"amount": 999}) is True               # out of scope: passes, not counted
    for bad in ({}, {"amount": "5"}, {"amount": True}, {"amount": -5}):
        assert pol("refund", bad) is False
    assert cap.state["spent"] == 100


def test_cap_spends_nothing_when_any_other_rule_denies_wherever_it_sits():
    from agentlab.policy import cumulative_cap
    for order in ("first", "middle", "last"):
        cap = cumulative_cap("amount", 100)
        others = [allow_tools("refund"), max_amount("amount", 80)]
        rules = {"first": [cap] + others, "middle": [others[0], cap, others[1]],
                 "last": others + [cap]}[order]
        pol = all_of(*rules)
        assert pol("refund", {"amount": 90}) is False           # per-call cap denies
        assert pol("other", {"amount": 10}) is False            # tool not allowed
        assert cap.state["spent"] == 0, order
        assert pol("refund", {"amount": 70}) is True
        assert pol("refund", {"amount": 40}) is False           # would reach 110
        assert cap.state["spent"] == 70, order


def test_cap_ignores_failures_after_it_and_losing_any_of_branches():
    from agentlab.policy import cumulative_cap
    def boom(name, args):
        raise ValueError("rule bug")
    cap = cumulative_cap("amount", 100)
    assert all_of(cap, boom)("refund", {"amount": 10}) is False     # raising rule after the cap denies
    assert cap.state["spent"] == 0
    # any_of: first branch has the cap but a later all_of member denies; second branch wins
    cap2 = cumulative_cap("amount", 100)
    pol = any_of(all_of(cap2, allow_tools("never")), allow_tools("refund"))
    assert pol("refund", {"amount": 30}) is True
    assert cap2.state["spent"] == 0                                 # losing branch spent nothing
    # winning branch spends exactly once
    cap3 = cumulative_cap("amount", 100)
    assert any_of(allow_tools("x"), all_of(cap3, allow_tools("refund")))("refund", {"amount": 30}) is True
    assert cap3.state["spent"] == 30
    # nested + audited: still all-or-nothing, and the audit log records the cap's own decision
    log: list = []
    cap4 = cumulative_cap("amount", 50)
    pol4 = all_of(audited(cap4, log), allow_tools("refund"))
    assert pol4("other", {"amount": 10}) is False and cap4.state["spent"] == 0
    assert pol4("refund", {"amount": 10}) is True and cap4.state["spent"] == 10


def test_cap_called_directly_or_via_foreign_wrapper_fails_clearly():
    import pytest
    from agentlab.policy import UnsupportedComposition, cumulative_cap
    cap = cumulative_cap("amount", 100)
    with pytest.raises(UnsupportedComposition):
        cap("refund", {"amount": 10})
    foreign = lambda name, args: cap(name, args)                    # not plan-aware
    with pytest.raises(UnsupportedComposition):
        all_of(foreign)("refund", {"amount": 10})                   # not swallowed as a plain deny
    with pytest.raises(UnsupportedComposition):
        any_of(foreign, allow_tools("refund"))("refund", {"amount": 10})
    assert cap.state["spent"] == 0
