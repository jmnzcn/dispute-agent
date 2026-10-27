"""The rule check behind the guardrails reproduces the ground truth on every deduction.

That also proves the tools: the policy only sees the world through the same lookups the
agent uses, so a wrong record from any tool would show up here as a mismatch.
"""
import pytest

from dispute_agent.evals import truth_table
from dispute_agent.policy import judge


@pytest.fixture(scope="module")
def truth():
    return truth_table()


def test_every_deduction_matches_the_ground_truth(world, truth):
    wrong = []
    for r in truth.itertuples():
        v = judge(world, r.vendor_code, r.ref)
        got = (v.status, v.category, v.recoverable_usd_cents, v.expired_usd_cents)
        want = (r.status, r.category, r.recoverable_usd_cents, r.expired_usd_cents)
        if got != want:
            wrong.append((r.vendor_code, r.ref, want, got))
    assert not wrong, f"{len(wrong)} of {len(truth)} differ, e.g. {wrong[:3]}"


def test_verdicts_cite_the_documents_they_rest_on(world, truth):
    for r in truth[truth["status"] == "CLAIMABLE"].head(50).itertuples():
        v = judge(world, r.vendor_code, r.ref)
        kinds = {e.split(":")[0] for e in v.evidence}
        need = {"SHORTAGE": {"DED", "INV", "SHP", "POD"}, "PRICE_CLAIM": {"DED", "INV", "PO"},
                "CHARGEBACK_ASN": {"DED", "SHP"}, "CHARGEBACK_LATE_SHIPMENT": {"DED", "PO", "SHP"},
                "DUPLICATE_DEDUCTION": {"DED"}}[v.category]
        assert need <= kinds, (r.ref, v.category, v.evidence)
