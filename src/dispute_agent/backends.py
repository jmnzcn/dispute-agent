"""Model backends behind one small interface, so the same agent loop runs on any of them.

  claude    Claude through the Anthropic SDK (needs ANTHROPIC_API_KEY)
  ollama    a local model through Ollama's chat API (no key, no data leaves the machine)
  scripted  a deterministic analyst that follows the procedure with the same tools: the
            control group that proves the harness, tools and guardrails before any model runs
"""
from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field

import httpx

from dispute_agent.tools import SPECS

# $ per million tokens: input, output, cache read (Claude API list prices; cache writes cost 1.25x input)
PRICES = {"claude-opus-5-5": (4.0, 20.0, 0.20), "claude-sonnet-5-5": (2.0, 10.0, 0.20),
          "claude-haiku-5-5": (0.10, 0.50, 0.01)}


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict


@dataclass
class Turn:
    text: str
    calls: list[ToolCall]
    input_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0
    stop: str = ""
    cost_usd: float = 0.0


@dataclass
class Conversation:
    """One case's dialogue with a model. Subclasses keep the history in their own format."""
    system: str
    turns: list[Turn] = field(default_factory=list)

    def send(self, user_text: str) -> Turn:
        raise NotImplementedError

    def send_results(self, results: list[tuple[ToolCall, str, bool]]) -> Turn:
        raise NotImplementedError


class Backend:
    name = "base"
    model = ""

    def conversation(self, system: str) -> Conversation:
        raise NotImplementedError


# ---------------------------------------------------------------------------------- Claude
class ClaudeConversation(Conversation):
    def __init__(self, system: str, client, model: str, effort: str):
        super().__init__(system)
        self.client, self.model, self.effort = client, model, effort
        self.messages: list[dict] = []

    def _call(self) -> Turn:
        t0 = time.perf_counter()
        # Cache breakpoints: tools + procedure (identical for every case), and the conversation so far
        last = self.messages[-1]
        if isinstance(last["content"], str):
            last["content"] = [{"type": "text", "text": last["content"]}]
        for m in self.messages:
            if m["role"] == "user" and isinstance(m["content"], list):
                for block in m["content"]:
                    block.pop("cache_control", None)
        last["content"][-1]["cache_control"] = {"type": "ephemeral"}
        r = self.client.messages.create(
            model=self.model, max_tokens=16000,
            system=[{"type": "text", "text": self.system, "cache_control": {"type": "ephemeral"}}],
            tools=[{**s, "strict": True} for s in SPECS], messages=self.messages,
            output_config={"effort": self.effort},
        )
        self.messages.append({"role": "assistant", "content": r.content})   # thinking blocks kept intact
        text = "".join(b.text for b in r.content if b.type == "text")
        calls = [ToolCall(b.id, b.name, dict(b.input)) for b in r.content if b.type == "tool_use"]
        u = r.usage
        read, write = u.cache_read_input_tokens or 0, u.cache_creation_input_tokens or 0
        p_in, p_out, p_read = PRICES.get(self.model, (0.0, 0.0, 0.0))
        cost = (u.input_tokens * p_in + write * p_in * 1.25 + read * p_read + u.output_tokens * p_out) / 1e6
        turn = Turn(text, calls, u.input_tokens + read + write, u.output_tokens, time.perf_counter() - t0,
                    r.stop_reason or "", cost)
        self.turns.append(turn)
        return turn

    def send(self, user_text: str) -> Turn:
        self.messages.append({"role": "user", "content": user_text})
        return self._call()

    def send_results(self, results):
        self.messages.append({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": c.id, "content": out, "is_error": err} for c, out, err in results]})
        return self._call()


def load_dotenv(path) -> None:
    """KEY=value lines from a local, git-ignored .env into the environment (existing values win)."""
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            key, sep, value = line.strip().partition("=")
            if sep and key and not key.startswith("#"):
                os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


class Claude(Backend):
    name = "claude"

    def __init__(self, model: str | None = None, effort: str = "medium", client=None):
        if client is None:
            import anthropic  # optional dependency: pip install dispute-agent[claude]

            from dispute_agent.world import ROOT
            load_dotenv(ROOT / ".env")
            client = anthropic.Anthropic(max_retries=4)
        self.client = client
        self.model = model or os.environ.get("DISPUTE_AGENT_MODEL", "claude-opus-5-5")
        self.effort = effort

    def conversation(self, system: str) -> Conversation:
        return ClaudeConversation(system, self.client, self.model, self.effort)


# ---------------------------------------------------------------------------------- Ollama
def _ollama_tools() -> list[dict]:
    return [{"type": "function", "function": {"name": s["name"], "description": s["description"],
                                              "parameters": s["input_schema"]}} for s in SPECS]


class OllamaConversation(Conversation):
    def __init__(self, system: str, http: httpx.Client, model: str, options: dict):
        super().__init__(system)
        self.http, self.model, self.options = http, model, options
        self.messages: list[dict] = [{"role": "system", "content": system}]

    def _call(self) -> Turn:
        t0 = time.perf_counter()
        r = self.http.post("/api/chat", json={"model": self.model, "messages": self.messages, "stream": False,
                                              "tools": _ollama_tools(), "options": self.options})
        r.raise_for_status()
        data = r.json()
        msg = data["message"]
        self.messages.append(msg)
        calls = []
        for c in msg.get("tool_calls") or []:
            args = c["function"].get("arguments") or {}
            if isinstance(args, str):  # some models return a JSON string
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            calls.append(ToolCall(c.get("id") or uuid.uuid4().hex[:12], c["function"]["name"], args))
        turn = Turn(msg.get("content") or "", calls, data.get("prompt_eval_count", 0), data.get("eval_count", 0),
                    time.perf_counter() - t0, data.get("done_reason", ""))
        self.turns.append(turn)
        return turn

    def send(self, user_text: str) -> Turn:
        self.messages.append({"role": "user", "content": user_text})
        return self._call()

    def send_results(self, results):
        for c, out, _err in results:
            self.messages.append({"role": "tool", "content": out, "tool_name": c.name})
        return self._call()


class Ollama(Backend):
    name = "ollama"

    def __init__(self, model: str = "llama3.2:3b", host: str | None = None, num_ctx: int = 8192):
        self.model = model
        self.http = httpx.Client(base_url=host or os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
                                 timeout=600)
        self.options = {"temperature": 0, "seed": 7, "num_ctx": num_ctx}

    def conversation(self, system: str) -> Conversation:
        return OllamaConversation(system, self.http, self.model, self.options)


# ---------------------------------------------------------------------------------- scripted
class ScriptedConversation(Conversation):
    """Follows the operating procedure step by step, reading each tool result like a model would.

    It sees only what the tools return. It is not the guardrail's policy check: it never calls
    policy.judge, and the agent loop, budgets, tracing and guardrails treat it like any model.
    """

    def __init__(self, system: str, case: dict):
        super().__init__(system)
        self.case = case
        self.gen = self._plan()
        self.last: list[str] = []

    def _ask(self, name: str, **args):
        out = yield [ToolCall(uuid.uuid4().hex[:12], name, args)]
        return json.loads(out[0])

    def _plan(self):
        no = self.case["ref"]
        ded = yield from self._ask("get_deduction", deduction_number=no)
        docs = [ded["doc_id"]]
        category = {"SHORTAGE": "SHORTAGE", "PRICE_CLAIM": "PRICE_CLAIM"}.get(
            ded["type"], "CHARGEBACK_LATE_SHIPMENT" if ded["reason_code"] == "LATE_SHIPMENT" else "CHARGEBACK_ASN")
        status, proposed, note = None, "", ""
        similar = yield from self._ask("find_similar_deductions", deduction_number=no)
        group = sorted([(ded["date"], ded["deduction_number"], ded["doc_id"])]
                       + [(s["date"], s["deduction_number"], s["doc_id"]) for s in similar])
        if group[0][1] != ded["deduction_number"]:
            gap = yield from self._ask("days_between", a=group[0][0], b=ded["date"])
            if gap["days"] <= 120:
                status, category = "CLAIMABLE", "DUPLICATE_DEDUCTION"
                docs.append(group[0][2])
        if status is None:
            kind = "purchase_order" if ded["type"] == "CHARGEBACK" else "invoice"
            found = yield from self._ask("search_records", reference=ded["reference_number"], kind=kind)
            if found["ambiguous"]:
                status = "NEEDS_REVIEW"   # the procedure as code lists candidates; it does not propose one
                note = f" Candidates: {', '.join(found['candidates'])}."
            elif not found["candidates"]:
                status = "INSUFFICIENT_EVIDENCE"
            elif kind == "invoice":
                inv = yield from self._ask("get_invoice", invoice_number=found["candidates"][0])
                docs.append(inv["doc_id"])
                line = next((ln for ln in inv["lines"] if ln["asin"] == ded["asin"]), None)
                if line is None:
                    status = "INSUFFICIENT_EVIDENCE"
                elif ded["type"] == "SHORTAGE":
                    ships = yield from self._ask("get_shipments", purchase_order=inv["purchase_order"])
                    if not ships:
                        status = "INSUFFICIENT_EVIDENCE"
                    else:
                        docs.append(ships[0]["doc_id"])
                        pod = yield from self._ask("get_proof_of_delivery", bill_of_lading=ships[0]["bill_of_lading"])
                        if "error" in pod:
                            status = "INSUFFICIENT_EVIDENCE"
                        else:
                            docs.append(pod["doc_id"])
                            full = pod["cartons_delivered"] >= ships[0]["cartons_shipped"]
                            status = "CLAIMABLE" if full else "NOT_CLAIMABLE"
                else:
                    po = yield from self._ask("get_purchase_order", purchase_order=inv["purchase_order"])
                    po_line = None if "error" in po else next(
                        (ln for ln in po["lines"] if ln["asin"] == ded["asin"]), None)
                    if po_line is None:
                        status = "INSUFFICIENT_EVIDENCE"
                    else:
                        docs.append(po["doc_id"])
                        over = line["unit_cost_cents"] > po_line["unit_cost_cents"]
                        status = "NOT_CLAIMABLE" if over else "CLAIMABLE"
            else:
                po = yield from self._ask("get_purchase_order", purchase_order=found["candidates"][0])
                ships = yield from self._ask("get_shipments", purchase_order=found["candidates"][0])
                docs.append(po["doc_id"])
                if not ships:
                    status = "INSUFFICIENT_EVIDENCE"
                else:
                    s = ships[0]
                    docs.append(s["doc_id"])
                    ok = (s["shipped_date"] <= po["ship_window_end"] if ded["reason_code"] == "LATE_SHIPMENT"
                          else s["asn_sent_date"] <= s["shipped_date"])
                    status = "CLAIMABLE" if ok else "NOT_CLAIMABLE"
        amount = 0
        if status == "CLAIMABLE":
            revs = yield from self._ask("get_reversals", deduction_number=no)
            back = sum(r["amount_cents"] for r in revs)
            docs += [r["doc_id"] for r in revs]
            amount = max(ded["amount_cents"] - back, 0)
            if back and not amount:
                status = "ALREADY_RECOVERED"
        if status == "CLAIMABLE":
            age = yield from self._ask("days_between", a=ded["date"], b="2026-07-31")
            if age["days"] > 365:
                status, amount = "EXPIRED", 0
        letter = ""
        if status == "CLAIMABLE":
            letter = (f"Hello Vendor Central team,\n\nWe dispute deduction {ded['deduction_number']} "
                      f"({category.replace('_', ' ').lower()}) for {amount / 100:.2f} {ded['currency']}. "
                      f"Our records ({', '.join(d.split(':', 1)[1] for d in docs[1:])}) show the deduction is "
                      f"not supported. Please reverse {amount / 100:.2f} {ded['currency']}.\n\nThank you,\n"
                      f"Vendor {self.case['vendor_code']}")
        yield [ToolCall(uuid.uuid4().hex[:12], "submit_decision", {
            "status": status, "category": category, "recoverable_amount": f"{amount / 100:.2f}",
            "evidence": docs, "proposed_match": proposed, "rationale": f"Procedure result: {status}.{note}",
            "letter": letter})]

    def _next(self, value) -> Turn:
        try:
            calls = self.gen.send(value) if value is not None else next(self.gen)
        except StopIteration:
            calls = []
        t = Turn("", calls)
        self.turns.append(t)
        return t

    def send(self, user_text: str) -> Turn:
        return self._next(None)

    def send_results(self, results):
        return self._next([out for _c, out, _e in results])


class Scripted(Backend):
    name = "scripted"
    model = "procedure"

    def __init__(self):
        self.case: dict = {}

    def conversation(self, system: str) -> Conversation:
        return ScriptedConversation(system, self.case)


def make(name: str, model: str | None = None) -> Backend:
    if name == "claude":
        return Claude(model)
    if name == "ollama":
        return Ollama(model or "llama3.2:3b")
    if name == "scripted":
        return Scripted()
    raise ValueError(f"unknown backend {name}")
