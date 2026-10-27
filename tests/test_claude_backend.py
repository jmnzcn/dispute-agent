"""The Claude backend's request shape, checked offline with a fake client (no key, no cost)."""
import copy
from types import SimpleNamespace as NS

from dispute_agent.agent import run_case
from dispute_agent.backends import Claude
from dispute_agent.evals import truth_table


class FakeMessages:
    def __init__(self, script):
        self.script, self.requests = script, []

    def create(self, **kw):
        self.requests.append(copy.deepcopy(kw))   # the agent keeps appending to the same list
        blocks = self.script(len(self.requests), kw)
        usage = NS(input_tokens=100, output_tokens=50, cache_read_input_tokens=1000, cache_creation_input_tokens=200)
        return NS(content=blocks, usage=usage, stop_reason="tool_use")


def test_tools_results_and_cache_breakpoints(world):
    r = truth_table().query("status == 'NOT_CLAIMABLE'").iloc[0]

    def script(n, kw):
        if n == 1:
            return [NS(type="text", text="Reading the deduction."),
                    NS(type="tool_use", id="t1", name="get_deduction", input={"deduction_number": r.ref})]
        return [NS(type="tool_use", id="t2", name="submit_decision", input={
            "status": "NOT_CLAIMABLE", "category": r.category, "recoverable_amount": "0",
            "evidence": [f"DED:{r.ref}"], "proposed_match": "", "rationale": "x", "letter": ""})]

    fake = FakeMessages(script)
    ep = run_case({"vendor_code": r.vendor_code, "ref": r.ref},
                  Claude(model="claude-opus-5-5", client=NS(messages=fake)), world)
    first, second = fake.requests
    assert first["model"] == "claude-opus-5-5" and all(t["strict"] for t in first["tools"])
    assert first["system"][0]["cache_control"] == {"type": "ephemeral"}
    # The tool result goes back with the tool_use id, and only the newest turn carries a breakpoint
    result = second["messages"][-1]["content"][0]
    assert result["type"] == "tool_result" and result["tool_use_id"] == "t1"
    assert result["cache_control"] == {"type": "ephemeral"}
    assert all("cache_control" not in b for m in second["messages"][:-1] if m["role"] == "user"
               and isinstance(m["content"], list) for b in m["content"])
    assert ep.decision["status"] == "NOT_CLAIMABLE"
    # 2 turns x (100 in + 200 cache write at 1.25x + 1000 cache read + 50 out), Opus 5.5 prices
    assert abs(ep.cost_usd - 2 * (100 * 4 + 200 * 5 + 1000 * 0.2 + 50 * 20) / 1e6) < 1e-9
