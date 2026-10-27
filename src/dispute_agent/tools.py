"""The agent's tools: read-only lookups scoped to one vendor, plus the final decision.

Least privilege by construction: a ToolSet is bound to the vendor of the case, so the
model cannot read another vendor's records whatever it asks for. Every call is recorded
(name, arguments, result, documents returned), which is what the guardrails audit.
"""
from __future__ import annotations

import ast
import json
import operator
import time
from dataclasses import dataclass, field
from datetime import date

from dispute_agent.world import World

STATUSES = ["CLAIMABLE", "NOT_CLAIMABLE", "INSUFFICIENT_EVIDENCE", "ALREADY_RECOVERED", "EXPIRED", "NEEDS_REVIEW"]
CATEGORIES = ["SHORTAGE", "PRICE_CLAIM", "CHARGEBACK_LATE_SHIPMENT", "CHARGEBACK_ASN", "DUPLICATE_DEDUCTION"]


def _schema(props: dict, required: list[str]) -> dict:
    return {"type": "object", "properties": props, "required": required, "additionalProperties": False}


_S = {"type": "string"}
SPECS = [
    {"name": "get_deduction", "description": "The deduction being investigated, or another one by number.",
     "input_schema": _schema({"deduction_number": _S}, ["deduction_number"])},
    {"name": "search_records",
     "description": "Find the invoice or purchase order a loosely written reference points to. Returns the "
                    "candidates and the tier that found them (exact, canonical, suffix, fuzzy). If 'ambiguous' "
                    "is true, more than one record matches and none may be assumed.",
     "input_schema": _schema({"reference": _S, "kind": {"type": "string", "enum": ["invoice", "purchase_order"]}},
                             ["reference", "kind"])},
    {"name": "get_invoice", "description": "An invoice with its lines (ASIN, quantity, unit cost) and its PO.",
     "input_schema": _schema({"invoice_number": _S}, ["invoice_number"])},
    {"name": "get_purchase_order",
     "description": "A purchase order: lines with agreed unit cost, and the ship window end date.",
     "input_schema": _schema({"purchase_order": _S}, ["purchase_order"])},
    {"name": "get_shipments",
     "description": "Shipments for a PO: shipped date, ASN sent date, bill of lading, cartons shipped.",
     "input_schema": _schema({"purchase_order": _S}, ["purchase_order"])},
    {"name": "get_proof_of_delivery",
     "description": "The carrier's proof of delivery for a bill of lading: cartons delivered and date.",
     "input_schema": _schema({"bill_of_lading": _S}, ["bill_of_lading"])},
    {"name": "get_reversals", "description": "Amounts Amazon has already paid back on a deduction.",
     "input_schema": _schema({"deduction_number": _S}, ["deduction_number"])},
    {"name": "find_similar_deductions",
     "description": "Other deductions with the same type, reason, reference, ASIN, quantity and amount, "
                    "oldest first (possible duplicates).",
     "input_schema": _schema({"deduction_number": _S}, ["deduction_number"])},
    {"name": "days_between", "description": "Days from date a to date b (YYYY-MM-DD); negative if b is earlier.",
     "input_schema": _schema({"a": _S, "b": _S}, ["a", "b"])},
    {"name": "calculator", "description": "Evaluate arithmetic: + - * / and parentheses, e.g. '125.40 - 20'.",
     "input_schema": _schema({"expression": _S}, ["expression"])},
    {"name": "submit_decision",
     "description": "Final answer for the case. Call exactly once, when the investigation is complete.",
     "input_schema": _schema({
         "status": {"type": "string", "enum": STATUSES},
         "category": {"type": "string", "enum": CATEGORIES},
         "recoverable_amount": {"type": "string",
                                "description": "Amount to recover in the deduction's currency, e.g. '125.40'; "
                                               "'0' unless status is CLAIMABLE"},
         "evidence": {"type": "array", "items": {"type": "string"},
                      "description": "doc_id of every document the decision relies on"},
         "proposed_match": {"type": "string",
                            "description": "For NEEDS_REVIEW: the record you believe the reference means, and why"},
         "rationale": {"type": "string", "description": "One or two sentences a reviewer can check"},
         "letter": {"type": "string",
                    "description": "If CLAIMABLE: the dispute letter to Amazon, under 160 words, using only facts "
                                   "from the documents; otherwise empty"},
     }, ["status", "category", "recoverable_amount", "evidence", "proposed_match", "rationale", "letter"])},
]
TOOL_NAMES = [s["name"] for s in SPECS]

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.USub: operator.neg}


def calculate(expression: str) -> float:
    """Arithmetic only: no names, calls or attributes reach eval."""
    def ev(n):
        if isinstance(n, ast.Constant) and isinstance(n.value, int | float):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.operand))
        raise ValueError("only numbers and + - * / ( ) are allowed")
    return round(ev(ast.parse(expression.replace(",", ""), mode="eval").body), 6)


@dataclass
class Call:
    name: str
    args: dict
    result: str
    is_error: bool
    doc_ids: list[str]
    ms: float


@dataclass
class ToolSet:
    world: World
    vendor: str
    calls: list[Call] = field(default_factory=list)
    decision: dict | None = None

    @property
    def retrieved(self) -> set[str]:
        return {d for c in self.calls for d in c.doc_ids}

    def _run(self, name: str, a: dict):
        w, v = self.world, self.vendor
        match name:
            case "get_deduction":
                return w.deduction(v, a["deduction_number"]) or {"error": "no such deduction for this vendor"}
            case "search_records":
                return w.search(v, a["reference"], a["kind"])
            case "get_invoice":
                return w.invoice(v, a["invoice_number"]) or {"error": "no such invoice for this vendor"}
            case "get_purchase_order":
                return w.purchase_order(v, a["purchase_order"]) or {"error": "no such purchase order"}
            case "get_shipments":
                return w.shipments(v, a["purchase_order"])
            case "get_proof_of_delivery":
                return w.proof_of_delivery(v, a["bill_of_lading"]) or {"error": "no proof of delivery on file"}
            case "get_reversals":
                return w.reversals(v, a["deduction_number"])
            case "find_similar_deductions":
                return w.similar_deductions(v, a["deduction_number"])
            case "days_between":
                return {"days": (date.fromisoformat(a["b"][:10]) - date.fromisoformat(a["a"][:10])).days}
            case "calculator":
                return {"result": calculate(a["expression"])}
            case "submit_decision":
                missing = [k for k in ("status", "category", "recoverable_amount", "evidence") if k not in a]
                if missing:
                    return {"error": f"missing fields: {missing}"}
                # Preconditions the model can act on; the guardrails still review whatever gets through
                if not any(c.name == "get_deduction" and not c.is_error for c in self.calls):
                    return {"error": "investigate first: call get_deduction, then follow the procedure"}
                evidence = a["evidence"] if isinstance(a["evidence"], list) else [a["evidence"]]
                unseen = [e for e in evidence if e not in self.retrieved]
                if unseen:
                    return {"error": f"evidence must be doc_id values returned by the tools; never returned: {unseen}"}
                self.decision = {k: a.get(k, "") for k in ("status", "category", "recoverable_amount", "evidence",
                                                         "proposed_match", "rationale", "letter")}
                return {"ok": True, "message": "Decision recorded. It will be checked before anything is filed."}
        return {"error": f"unknown tool {name}"}

    def call(self, name: str, args: dict | None) -> Call:
        t0 = time.perf_counter()
        args = args if isinstance(args, dict) else {}
        try:
            out = self._run(name, args)
            err = isinstance(out, dict) and "error" in out
        except (KeyError, ValueError, TypeError, SyntaxError) as exc:  # bad arguments from the model
            out, err = {"error": f"bad arguments: {exc}"}, True
        docs = [x["doc_id"] for x in (out if isinstance(out, list) else [out]) if isinstance(x, dict) and "doc_id" in x]
        c = Call(name, args, json.dumps(out, default=str), err, docs, (time.perf_counter() - t0) * 1000)
        self.calls.append(c)
        return c
