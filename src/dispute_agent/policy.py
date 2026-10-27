"""The recovery policy as code: what a careful analyst can prove about one deduction.

This is the independent check behind every decision the agent makes. It reaches its own
verdict with the same read-only lookups the agent has, and lists the documents that verdict
rests on. The agent's job is the investigation and the letter; money only moves when the
agent and this check agree (see guardrails.py).

`tests/test_policy.py` holds it to vendor-recovery-engine's ground truth on every deduction
of the bundled world, which also proves the tools return the right records.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from dispute_agent.world import AS_OF, DEDUCTION_WINDOW_DAYS, DUPLICATE_WINDOW_DAYS, World

CATEGORY = {("SHORTAGE", ""): "SHORTAGE", ("PRICE_CLAIM", ""): "PRICE_CLAIM",
            ("CHARGEBACK", "LATE_SHIPMENT"): "CHARGEBACK_LATE_SHIPMENT", ("CHARGEBACK", "ASN_LATE"): "CHARGEBACK_ASN"}


@dataclass
class Verdict:
    status: str
    category: str
    recoverable_cents: int            # in the deduction's currency
    recoverable_usd_cents: int
    expired_usd_cents: int
    reason: str
    evidence: list[str] = field(default_factory=list)   # doc_ids the verdict rests on
    matched: str | None = None                           # invoice or PO the reference resolved to
    candidates: list[str] = field(default_factory=list)  # when the reference is ambiguous


def category_of(ded: dict) -> str:
    return CATEGORY.get((ded["type"], ded["reason_code"] if ded["type"] == "CHARGEBACK" else ""), "UNKNOWN")


def _evidence_status(w: World, v: str, ded: dict, force: str | None = None
                     ) -> tuple[str, str, list[str], str | None, list[str]]:
    """Status before reversals and expiry: (status, reason, evidence, matched, candidates).

    `force` takes the reference as meaning that record (to grade a proposed match)."""
    kind = "purchase_order" if ded["type"] == "CHARGEBACK" else "invoice"
    found = w.search(v, ded["reference_number"], kind)
    if force is not None:
        found = {"ambiguous": False, "candidates": [force]}
    if found["ambiguous"]:
        return "NEEDS_REVIEW", "Reference matches more than one record.", [], None, found["candidates"]
    if not found["candidates"]:
        return "INSUFFICIENT_EVIDENCE", "Reference not found in the records.", [], None, []
    match = found["candidates"][0]

    if kind == "invoice":
        inv = w.invoice(v, match)
        line = next((ln for ln in inv["lines"] if ln["asin"] == ded["asin"]), None)
        if line is None:
            return "INSUFFICIENT_EVIDENCE", f"ASIN not on invoice {match}.", [inv["doc_id"]], match, []
        if ded["type"] == "SHORTAGE":
            ships = w.shipments(v, inv["purchase_order"])
            if not ships:
                return "INSUFFICIENT_EVIDENCE", "No shipment for the invoice's PO.", [inv["doc_id"]], match, []
            ship = ships[0]
            pod = w.proof_of_delivery(v, ship["bill_of_lading"])
            if pod is None:
                return ("INSUFFICIENT_EVIDENCE", "No carrier proof of delivery.", [inv["doc_id"], ship["doc_id"]],
                        match, [])
            docs = [inv["doc_id"], ship["doc_id"], pod["doc_id"]]
            if pod["cartons_delivered"] >= ship["cartons_shipped"]:
                return "CLAIMABLE", "Every carton delivered; the shortage is not supported.", docs, match, []
            return "NOT_CLAIMABLE", "Cartons missing at delivery; the shortage is valid.", docs, match, []
        po = w.purchase_order(v, inv["purchase_order"])
        po_line = None if po is None else next((ln for ln in po["lines"] if ln["asin"] == ded["asin"]), None)
        if po_line is None:
            return "INSUFFICIENT_EVIDENCE", "PO line for the ASIN not found.", [inv["doc_id"]], match, []
        docs = [inv["doc_id"], po["doc_id"]]
        if line["unit_cost_cents"] > po_line["unit_cost_cents"]:
            return "NOT_CLAIMABLE", "Invoiced above the PO cost; the price claim is valid.", docs, match, []
        return "CLAIMABLE", "Invoiced at the PO cost; the price claim is not supported.", docs, match, []

    # Evidence is what decides the case: the ship window lives on the PO, the ASN date on the shipment
    po = w.purchase_order(v, match)
    ships = w.shipments(v, match)
    if not ships:
        return "INSUFFICIENT_EVIDENCE", "No shipment for the PO.", [], match, []
    ship = ships[0]
    if ded["reason_code"] == "LATE_SHIPMENT":
        ok = ship["shipped_date"] <= po["ship_window_end"]
        return (("CLAIMABLE", "Shipped inside the ship window.") if ok else
                ("NOT_CLAIMABLE", "Shipped after the ship window.")) + ([po["doc_id"], ship["doc_id"]], match, [])
    ok = ship["asn_sent_date"] <= ship["shipped_date"]
    return (("CLAIMABLE", "ASN sent no later than the ship date.") if ok else
            ("NOT_CLAIMABLE", "ASN sent after the ship date.")) + ([ship["doc_id"]], match, [])


def judge(w: World, vendor: str, deduction_number: str, match: str | None = None) -> Verdict:
    """The verdict on one deduction. `match` forces what an ambiguous reference means."""
    ded = w.deduction(vendor, deduction_number)
    if ded is None:
        raise KeyError(deduction_number)
    category = category_of(ded)

    # Duplicates first: the same deduction applied again within the window
    group = [{"deduction_number": ded["deduction_number"], "date": ded["date"], "doc_id": ded["doc_id"]},
             *w.similar_deductions(vendor, deduction_number)]
    first = min(group, key=lambda g: (g["date"], g["deduction_number"]))
    dup_days = (pd.Timestamp(ded["date"]) - pd.Timestamp(first["date"])).days
    if first["deduction_number"] != ded["deduction_number"] and dup_days <= DUPLICATE_WINDOW_DAYS:
        status, reason, evidence, matched, cands = (
            "CLAIMABLE", f"Duplicate of {first['deduction_number']}.", [first["doc_id"]],
            first["deduction_number"], [])
        category = "DUPLICATE_DEDUCTION"
    else:
        status, reason, evidence, matched, cands = _evidence_status(w, vendor, ded, match)
    evidence = [ded["doc_id"], *evidence]

    recoverable = 0
    if status == "CLAIMABLE":
        revs = w.reversals(vendor, ded["deduction_number"])
        back = sum(r["amount_cents"] for r in revs)
        evidence += [r["doc_id"] for r in revs]
        recoverable = max(ded["amount_cents"] - back, 0)
        if back > 0 and recoverable == 0:
            status, reason = "ALREADY_RECOVERED", "Amazon already reversed it in full."
    usd = w.to_usd_cents(recoverable, ded["currency"], ded["date"]) if recoverable else 0
    expired_usd = 0
    deadline = pd.Timestamp(ded["date"]) + pd.Timedelta(days=DEDUCTION_WINDOW_DAYS)
    if status == "CLAIMABLE" and deadline < pd.Timestamp(AS_OF):
        status, reason = "EXPIRED", f"Past the dispute window ({deadline:%Y-%m-%d}). " + reason
        expired_usd, usd, recoverable = usd, 0, 0
    return Verdict(status, category, recoverable, usd, expired_usd, reason, evidence, matched, cands)
