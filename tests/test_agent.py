"""The harness end to end, with the deterministic analyst as the model."""
import asyncio
import json

from dispute_agent import evals
from dispute_agent.agent import Budget, run_case
from dispute_agent.backends import Backend, Conversation, Scripted, ToolCall, Turn


def test_the_scripted_analyst_resolves_every_case_and_files_nothing_wrong(world, tmp_path):
    cases = evals.sample_cases(per_stratum=2)
    todo = [({"vendor_code": r.vendor_code, "ref": r.ref}, {}) for r in cases.itertuples()]
    todo += evals.injection_cases(world, cases, n=4)
    eps = evals.run(Scripted(), world, todo, tmp_path / "run.jsonl", log=lambda *_: None)
    s = evals.score(eps, world)
    assert s["agent"]["exact_pct"] == 100.0
    assert s["system"]["false_filings"] == 0
    assert s["system"]["escalated"] == int((cases["status"] == "NEEDS_REVIEW").sum())
    assert s["injection"]["flagged"] == s["injection"]["cases"] == 4


class Stubborn(Backend):
    """A model that never stops calling tools."""
    name, model = "stub", "stub"

    def conversation(self, system):
        class C(Conversation):
            def send(self, text):
                return self._t()

            def send_results(self, results):
                return self._t()

            def _t(self):
                t = Turn("", [ToolCall(str(len(self.turns)), "days_between", {"a": "2026-01-01", "b": "2026-01-02"})])
                self.turns.append(t)
                return t
        return C(system)


def test_budgets_stop_a_looping_model(world):
    t = evals.truth_table()
    r = t.iloc[0]
    ep = run_case({"vendor_code": r.vendor_code, "ref": r.ref}, Stubborn(), world, Budget(max_turns=5))
    assert ep.stopped == "turn budget" and ep.review.outcome == "ESCALATE"


def test_mcp_server_exposes_read_only_tools(world):
    from dispute_agent.mcp_server import build
    t = evals.truth_table()
    r = t.iloc[0]
    mcp = build(r.vendor_code, world)
    names = {tool.name for tool in asyncio.run(mcp.list_tools())}
    assert "get_deduction" in names and "submit_decision" not in names
    out = asyncio.run(mcp.call_tool("get_deduction", {"deduction_number": r.ref}))
    text = out.content[0].text
    assert json.loads(text)["deduction_number"] == r.ref
