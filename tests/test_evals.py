import json

from agentlab import Agent, ToolRegistry
from agentlab.evals import Case, run_case, run_cases, summarize
from agentlab.policy import all_of, allow_tools, max_amount


def call(action, **inp):
    return json.dumps({"thought": "t", "action": action, "action_input": inp})


FINAL = json.dumps({"thought": "done", "final": "refund processed"})


def make_agent(llm):
    tools = ToolRegistry()

    @tools.register("look up balance", {"acct": "string"})
    def balance(acct):
        return {"balance": 10}

    @tools.register("refund", {"amount": "number"}, sensitive=True)
    def refund(amount):
        return {"ok": True}

    return Agent(llm, tools, approver=all_of(allow_tools("refund"), max_amount("amount", 100)))


def test_must_block_is_strict_when_the_tool_also_ran():
    case = Case("small refund ok, big refund blocked", "refund",
                [call("balance", acct="A"), call("refund", amount=50), call("refund", amount=9000), FINAL],
                answer_contains="processed", must_call=("balance", "refund"),
                must_block=("refund",), stopped_reason="final")
    # refund(50) runs and refund(9000) is blocked, so must_block fails: the harness is strict
    res = run_case(make_agent, case)
    assert not res.passed and any("ran despite must_block" in f for f in res.failures)


def test_clean_pass_and_each_failure_kind():
    ok = Case("big refund blocked", "refund", [call("refund", amount=9000), FINAL],
              answer_contains="PROCESSED", must_block=("refund",), must_not_call=("balance",),
              stopped_reason="final")
    assert run_case(make_agent, ok).passed
    bad = Case("wrong", "refund", [call("refund", amount=9000), FINAL],
               answer_contains="xyz", must_call=("balance",), must_not_call=("refund",),
               stopped_reason="max_steps")
    res = run_case(make_agent, bad)
    assert not res.passed
    assert any("lacks" in f for f in res.failures) and any("expected call" in f for f in res.failures)
    assert any("stopped_reason" in f for f in res.failures)
    assert not any("must not run" in f for f in res.failures)   # blocked calls are not "called"


def test_crashing_agent_is_a_failure_not_an_exception():
    def broken(llm):
        raise RuntimeError("boom")
    res = run_case(broken, Case("x", "t", []))
    assert not res.passed and "RuntimeError" in res.failures[0]


def test_summary_counts():
    cases = [Case("a", "t", [FINAL], answer_contains="refund"), Case("b", "t", [FINAL], answer_contains="nope")]
    out = summarize(run_cases(make_agent, cases))
    assert "PASS  a" in out and "FAIL  b" in out and "1/2 cases passed" in out


def test_call_order_passes_in_order_and_fails_out_of_order():
    ok = Case("lookup then refund", "refund",
              [call("balance", acct="A"), call("refund", amount=50), FINAL],
              call_order=("balance", "refund"))
    assert run_case(make_agent, ok).passed
    bad = Case("refund before lookup", "refund",
               [call("refund", amount=50), call("balance", acct="A"), FINAL],
               call_order=("balance", "refund"))
    res = run_case(make_agent, bad)
    assert not res.passed and any("in order" in f for f in res.failures)
    # a blocked call does not count as having run
    blocked = Case("blocked refund", "refund",
                   [call("balance", acct="A"), call("refund", amount=9000), FINAL],
                   call_order=("balance", "refund"))
    assert not run_case(make_agent, blocked).passed


def test_to_json_totals_and_failures_round_trip():
    import json as _json
    from agentlab.evals import to_json
    ok = Case("ok", "refund", [call("refund", amount=9000), FINAL], must_block=("refund",))
    bad = Case("bad", "refund", [call("balance", acct="A"), FINAL], must_call=("refund",))
    out = _json.loads(to_json(run_cases(make_agent, [ok, bad])))
    assert (out["total"], out["passed"], out["failed"]) == (2, 1, 1)
    assert out["cases"][0] == {"name": "ok", "passed": True, "failures": []}
    assert out["cases"][1]["passed"] is False and "refund" in out["cases"][1]["failures"][0]
    assert _json.loads(to_json([])) == {"total": 0, "passed": 0, "failed": 0, "cases": []}


def test_max_steps_fails_when_the_run_is_too_long():
    ok = Case("quick", "refund", [call("refund", amount=9000), FINAL], max_steps=2)
    assert run_case(make_agent, ok).passed
    # the final answer is not a step, so this run takes 2 steps
    slow = Case("slow", "refund", [call("balance", acct="A"), call("refund", amount=9000), FINAL],
                max_steps=1)
    res = run_case(make_agent, slow)
    assert not res.passed and any("2 steps, over the limit of 1" in f for f in res.failures)
    edge = Case("edge", "refund", [call("refund", amount=9000), FINAL], max_steps=1)
    assert run_case(make_agent, edge).passed        # exactly at the limit passes


def test_max_tool_calls_counts_only_executed_calls():
    script = [call("balance", acct="A"), call("balance", acct="B"), call("refund", amount=9000), FINAL]
    ok = Case("two lookups allowed", "x", script, max_tool_calls={"balance": 2, "refund": 0})
    assert run_case(make_agent, ok).passed          # blocked refund is not an executed call
    tight = Case("one lookup allowed", "x", script, max_tool_calls={"balance": 1})
    res = run_case(make_agent, tight)
    assert not res.passed and any("'balance' ran 2 times, over the limit of 1" in f for f in res.failures)


def test_answer_excludes_catches_leaks_case_insensitively():
    leak = json.dumps({"thought": "d", "final": "Member MBR-0042 refund processed"})
    script = [leak]
    bad = Case("no member id in answer", "x", script, answer_excludes=("mbr-0042", "ssn"))
    res = run_case(make_agent, bad)
    assert not res.passed and any("forbidden text 'mbr-0042'" in f for f in res.failures)
    assert not any("'ssn'" in f for f in res.failures)
    ok = Case("clean answer", "x", [FINAL], answer_excludes=("mbr-0042",))
    assert run_case(make_agent, ok).passed


def test_must_call_with_matches_argument_subset_on_executed_calls_only():
    script = [call("refund", amount=50), call("refund", amount=9000), FINAL]
    assert run_case(make_agent, Case("a", "x", script, must_call_with={"refund": {"amount": 50}})).passed
    res = run_case(make_agent, Case("b", "x", script, must_call_with={"refund": {"amount": 9000}}))
    assert not res.passed and "amount" in res.failures[0]          # 9000 was blocked, so it did not execute
    res = run_case(make_agent, Case("c", "x", script, must_call_with={"balance": {"acct": "A"}}))
    assert not res.passed
    assert run_case(make_agent, Case("d", "x", script, must_call_with={"refund": {}})).passed   # any executed call


def test_must_not_call_with_flags_executed_calls_only():
    script = [call("refund", amount=50), call("refund", amount=9000), FINAL]
    res = run_case(make_agent, Case("a", "x", script, must_not_call_with={"refund": {"amount": 50}}))
    assert not res.passed and "forbidden arguments" in res.failures[0]
    # the 9000 refund was blocked by policy, so it never executed and does not trip the check
    assert run_case(make_agent, Case("b", "x", script, must_not_call_with={"refund": {"amount": 9000}})).passed
    assert run_case(make_agent, Case("c", "x", script, must_not_call_with={"balance": {}})).passed
    assert not run_case(make_agent, Case("d", "x", script, must_not_call_with={"refund": {}})).passed   # empty = any executed call
