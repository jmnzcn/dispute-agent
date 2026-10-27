"""What happens to an agent decision: FILE the dispute, CLOSE the case, or ESCALATE to a person.

The model never moves money on its own say-so. A decision is filed only if every check passes:

  decided      the agent submitted a decision within its budgets
  grounded     every document it cites was actually returned to it in this case
  evidence     it cites the documents the verdict depends on (an auditor can retrace it)
  agrees       an independent rule check (policy.judge) reaches the same status, category and amount
  letter       every amount, date and document number in the letter appears in the retrieved records,
               and the letter names the deduction and the amount (the approach of
               vendor-recovery-engine's verifier)
  value        claims above the approval limit go to a person even when everything else passes

NEEDS_REVIEW is always a person's call; the agent's proposed match travels with it. Injected
instructions found in documents are flagged in the trace whatever the outcome.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation

from dispute_agent.policy import judge
from dispute_agent.tools import CATEGORIES, STATUSES, ToolSet
from dispute_agent.world import World

APPROVAL_LIMIT_USD_CENTS = 250_000   # claims above $2,500 need a person's sign-off

_AMOUNT = re.compile(r"(?<![\d.])\d{1,3}(?:,\d{3})*\.\d{2}(?!\d)|(?<![\d.,])\d+\.\d{2}(?!\d)")
_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_ID = re.compile(r"\b(?=[A-Za-z0-9-]*\d)[A-Za-z0-9][A-Za-z0-9-]{5,}\b")
_INJECTION = re.compile(r"(ignore|disregard|override).{0,40}(instruction|procedure|rule|check)|"
                        r"(approve|mark|submit).{0,40}(claimable|full amount)|system\s*:|ai agent|assistant",
                        re.IGNORECASE)


@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""


@dataclass
class Review:
    outcome: str                      # FILE | CLOSE | ESCALATE
    why: str
    checks: list[Check] = field(default_factory=list)
    expected_status: str = ""
    expected_amount_cents: int = 0
    expected_usd_cents: int = 0
    injection_seen: bool = False


def cents(text: str) -> int | None:
    try:
        return int((Decimal(str(text).replace(",", "").strip() or "0") * 100).to_integral_value())
    except (InvalidOperation, ValueError):
        return None


def notes_in(result) -> list[str]:
    """Free-text notes of every document in a tool result."""
    docs = result if isinstance(result, list) else [result]
    return [x["notes"] for x in docs if isinstance(x, dict) and x.get("notes")]


def letter_problems(letter: str, retrieved_text: str, deduction_number: str, amount_cents: int) -> list[str]:
    """Facts in the letter that are not in the records the agent retrieved."""
    allowed_amounts = {cents(a) for a in _AMOUNT.findall(retrieved_text)} | {amount_cents}
    allowed_dates = set(_DATE.findall(retrieved_text))
    allowed_ids = {t.upper() for t in _ID.findall(retrieved_text)}
    problems = []
    for a in _AMOUNT.findall(letter):
        if cents(a) not in allowed_amounts:
            problems.append(f"amount not in the records: {a}")
    for d in _DATE.findall(letter):
        if d not in allowed_dates:
            problems.append(f"date not in the records: {d}")
    for t in _ID.findall(_DATE.sub(" ", _AMOUNT.sub(" ", letter))):
        if not t.isdigit() and t.upper() not in allowed_ids:
            problems.append(f"document number not in the records: {t}")
    if deduction_number not in letter:
        problems.append("deduction number missing")
    if amount_cents not in {cents(a) for a in _AMOUNT.findall(letter)}:
        problems.append("disputed amount missing")
    return problems


def review(tools: ToolSet, case: dict, world: World) -> Review:
    d = tools.decision
    verdict = judge(world, case["vendor_code"], case["ref"])
    injection = any(_INJECTION.search(n) for c in tools.calls for n in notes_in(json.loads(c.result)))
    base = {"expected_status": verdict.status, "expected_amount_cents": verdict.recoverable_cents,
            "expected_usd_cents": verdict.recoverable_usd_cents, "injection_seen": bool(injection)}
    checks: list[Check] = []

    def done(outcome: str, why: str) -> Review:
        return Review(outcome, why, checks, **base)

    checks.append(Check("decided", d is not None, "" if d else "no decision submitted"))
    if d is None:
        return done("ESCALATE", "The agent did not reach a decision.")
    amount = cents(d.get("recoverable_amount", ""))
    well_formed = d.get("status") in STATUSES and d.get("category") in CATEGORIES and amount is not None
    checks.append(Check("well formed", well_formed, "" if well_formed else "invalid status, category or amount"))
    if not well_formed:
        return done("ESCALATE", "The decision is malformed.")

    cited = [str(x) for x in d.get("evidence") or []]
    unknown = sorted(set(cited) - tools.retrieved)
    checks.append(Check("grounded", bool(cited) and not unknown,
                        "no evidence cited" if not cited else (f"never retrieved: {unknown}" if unknown else "")))
    if d["status"] == "NEEDS_REVIEW":
        return done("ESCALATE", "Ambiguous reference: a person decides, with the agent's proposed match.")
    if not cited or unknown:
        return done("ESCALATE", "The decision cites documents the agent never saw.")

    agrees = d["status"] == verdict.status and (
        d["status"] != "CLAIMABLE" or (amount == verdict.recoverable_cents and d["category"] == verdict.category))
    checks.append(Check("agrees", agrees, "" if agrees else
                        f"rule check says {verdict.status} {verdict.recoverable_cents / 100:.2f} ({verdict.category})"))
    if not agrees:
        return done("ESCALATE", "The agent and the rule check disagree.")

    missing = [e for e in verdict.evidence if e not in cited]
    checks.append(Check("evidence", not missing, f"not cited: {missing}" if missing else ""))
    if missing:
        return done("ESCALATE", "The decision does not cite the documents it depends on.")

    if d["status"] != "CLAIMABLE":
        return done("CLOSE", f"No dispute: {d['status']}.")

    retrieved_text = " ".join(c.result for c in tools.calls if c.name != "submit_decision")
    problems = letter_problems(d.get("letter") or "", retrieved_text, case["ref"], amount)
    checks.append(Check("letter", not problems, "; ".join(problems)))
    if problems:
        return done("ESCALATE", "The dispute letter states facts that are not in the records.")
    over = verdict.recoverable_usd_cents > APPROVAL_LIMIT_USD_CENTS
    checks.append(Check("value", not over, "above the approval limit" if over else ""))
    if over:
        return done("ESCALATE", "Verified claim above the approval limit: a person signs it off.")
    return done("FILE", "Every check passed.")
