"""Whatever the model says, a wrong claim is not filed."""
import json

import pytest

from dispute_agent.evals import truth_table
from dispute_agent.guardrails import letter_problems, review
from dispute_agent.policy import judge
from dispute_agent.tools import ToolSet


def investigated(world, vendor, ref) -> ToolSet:
    """A tool set that has retrieved every document the rule check relies on."""
    t = ToolSet(world, vendor)
    v = judge(world, vendor, ref)
    ded = json.loads(t.call("get_deduction", {"deduction_number": ref}).result)
    t.call("find_similar_deductions", {"deduction_number": ref})
    t.call("get_reversals", {"deduction_number": ref})
    if v.matched and ded["type"] != "CHARGEBACK" and v.category != "DUPLICATE_DEDUCTION":
        po = json.loads(t.call("get_invoice", {"invoice_number": v.matched}).result)["purchase_order"]
    else:
        po = v.matched
    if po:
        t.call("get_purchase_order", {"purchase_order": po})
        for s in json.loads(t.call("get_shipments", {"purchase_order": po}).result)[:1]:
            t.call("get_proof_of_delivery", {"bill_of_lading": s["bill_of_lading"]})
    return t


def pick(status):
    t = truth_table()
    r = t[(t["status"] == status) & (t["category"] == "SHORTAGE")].iloc[0]
    return r.vendor_code, r.ref


def decide(world, t, vendor, ref, **over):
    v = judge(world, vendor, ref)
    d = {"status": v.status, "category": v.category, "recoverable_amount": f"{v.recoverable_cents / 100:.2f}",
         "evidence": v.evidence, "proposed_match": "", "rationale": "test",
         "letter": f"We dispute deduction {ref} for {v.recoverable_cents / 100:.2f}."}
    d.update(over)
    t.decision = d
    return review(t, {"vendor_code": vendor, "ref": ref}, world)


def test_a_correct_claim_is_filed(world):
    vendor, ref = pick("CLAIMABLE")
    t = investigated(world, vendor, ref)
    assert decide(world, t, vendor, ref).outcome == "FILE"


def test_claiming_a_valid_deduction_is_escalated_not_filed(world):
    vendor, ref = pick("NOT_CLAIMABLE")
    t = investigated(world, vendor, ref)
    ded = world.deduction(vendor, ref)
    r = decide(world, t, vendor, ref, status="CLAIMABLE", recoverable_amount=f"{ded['amount_cents'] / 100:.2f}",
               letter=f"We dispute deduction {ref} for {ded['amount_cents'] / 100:.2f}.")
    assert r.outcome == "ESCALATE" and "disagree" in r.why


def test_a_wrong_amount_is_escalated(world):
    vendor, ref = pick("CLAIMABLE")
    t = investigated(world, vendor, ref)
    v = judge(world, vendor, ref)
    r = decide(world, t, vendor, ref, recoverable_amount=f"{(v.recoverable_cents + 500) / 100:.2f}")
    assert r.outcome == "ESCALATE"


def test_citing_documents_never_retrieved_is_escalated(world):
    vendor, ref = pick("CLAIMABLE")
    t = ToolSet(world, vendor)
    t.call("get_deduction", {"deduction_number": ref})
    r = decide(world, t, vendor, ref)
    assert r.outcome == "ESCALATE" and "never saw" in r.why


def test_a_letter_with_an_invented_fact_is_escalated(world):
    vendor, ref = pick("CLAIMABLE")
    t = investigated(world, vendor, ref)
    v = judge(world, vendor, ref)
    r = decide(world, t, vendor, ref, letter=f"We dispute deduction {ref} for {v.recoverable_cents / 100:.2f}. "
                                            f"See proof of delivery POD-ZZ81234 dated 2026-02-30.")
    assert r.outcome == "ESCALATE" and "letter" in r.why


def test_ambiguous_references_always_go_to_a_person(world):
    t = truth_table()
    r = t[t["status"] == "NEEDS_REVIEW"].iloc[0]
    ts = investigated(world, r.vendor_code, r.ref)
    out = decide(world, ts, r.vendor_code, r.ref, proposed_match="INV-1 has the ASIN")
    assert out.outcome == "ESCALATE"


@pytest.mark.parametrize("letter,problem", [
    ("We dispute deduction SC1 for 10.00.", None),
    ("We dispute deduction SC1 for 12.00.", "amount not in the records: 12.00"),
    ("We dispute deduction SC1 for 10.00, delivered 2026-01-02.", "date not in the records: 2026-01-02"),
    ("We dispute deduction SC1 for 10.00 per POD-ABC123.", "document number not in the records: POD-ABC123"),
    ("We dispute it for 10.00.", "deduction number missing"),
])
def test_letter_check(letter, problem):
    probs = letter_problems(letter, '{"deduction_number": "SC1", "amount": "10.00 USD"}', "SC1", 1000)
    assert (problem in probs) if problem else not probs
