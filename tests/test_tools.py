import json

import pytest

from dispute_agent.evals import truth_table
from dispute_agent.tools import ToolSet, calculate


@pytest.fixture(scope="module")
def case():
    t = truth_table()
    r = t[t["status"] == "CLAIMABLE"].iloc[0]
    return r.vendor_code, r.ref


def test_tools_are_scoped_to_the_case_vendor(world, case):
    vendor, ref = case
    other = next(v for v in world.tables["deductions"]["vendor_code"].unique() if v != vendor)
    assert "error" not in json.loads(ToolSet(world, vendor).call("get_deduction", {"deduction_number": ref}).result)
    # The same deduction number, asked by a tool set bound to another vendor, does not exist
    out = json.loads(ToolSet(world, other).call("get_deduction", {"deduction_number": ref}).result)
    assert out == {"error": "no such deduction for this vendor"}


def test_every_document_has_a_doc_id_and_notes(world, case):
    vendor, ref = case
    t = ToolSet(world, vendor)
    ded = json.loads(t.call("get_deduction", {"deduction_number": ref}).result)
    assert ded["doc_id"] == f"DED:{ded['deduction_number']}" and ded["notes"] == ""
    assert t.retrieved == {ded["doc_id"]}


def test_submit_needs_an_investigation_and_seen_evidence(world, case):
    vendor, ref = case
    t = ToolSet(world, vendor)
    decision = {"status": "CLAIMABLE", "category": "SHORTAGE", "recoverable_amount": "1.00",
                "evidence": [f"DED:{ref}"], "proposed_match": "", "rationale": "", "letter": ""}
    assert "investigate first" in t.call("submit_decision", decision).result
    t.call("get_deduction", {"deduction_number": ref})
    bad = {**decision, "evidence": ["POD:INVENTED"]}
    assert "never returned" in t.call("submit_decision", bad).result
    assert t.decision is None
    t.call("submit_decision", {**decision, "evidence": [f"DED:{ref}"]})
    assert t.decision["status"] == "CLAIMABLE"


def test_bad_arguments_come_back_as_errors_not_crashes(world, case):
    t = ToolSet(world, case[0])
    c = t.call("get_invoice", {"wrong": 1})
    assert c.is_error and "bad arguments" in c.result
    assert t.call("no_such_tool", {}).is_error


@pytest.mark.parametrize("expr,value", [("125.40 - 20", 105.4), ("(3 + 4) * 2", 14), ("1,250.50 / 2", 625.25)])
def test_calculator(expr, value):
    assert calculate(expr) == value


@pytest.mark.parametrize("expr", ["__import__('os').system('x')", "open('f')", "a + 1", "2 ** 999999"])
def test_calculator_only_does_arithmetic(expr):
    with pytest.raises(ValueError):
        calculate(expr)
