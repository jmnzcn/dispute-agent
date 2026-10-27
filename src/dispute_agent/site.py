"""Build docs/index.html from eval runs: the comparison, the injection test and every trace."""
from __future__ import annotations

import json
from pathlib import Path

from dispute_agent import evals
from dispute_agent.world import ROOT, World

LABELS = {"scripted-procedure": ("Scripted analyst", "control: the procedure as code, same tools"),
          "claude-claude-haiku-5-5": ("Claude Haiku 5.5", "agent"),
          "claude-claude-sonnet-5-5": ("Claude Sonnet 5.5", "agent"),
          "claude-claude-opus-5-5": ("Claude Opus 5.5", "agent"),
          "claude-claude-haiku-5-5-procedure-v1": ("Claude Haiku 5.5, procedure v1", "agent, before the eval fix")}
ORDER = ["claude-claude-opus-5-5", "claude-claude-sonnet-5-5", "claude-claude-haiku-5-5",
         "claude-claude-haiku-5-5-procedure-v1", "scripted-procedure"]


def _trim(text: str, n: int = 700) -> str:
    return text if len(text) <= n else text[:n] + " …"


def build(runs: list[Path], out: Path = ROOT / "docs" / "index.html") -> Path:
    world = World.load()
    models, episodes = [], {}
    for p in sorted(runs, key=lambda p: ORDER.index(p.stem) if p.stem in ORDER else 99):
        eps = evals.load(p)
        s = evals.score(eps, world)
        name, role = LABELS.get(p.stem, (p.stem, "agent"))
        n = len(eps)
        models.append({"id": p.stem, "name": name, "role": role, "n": n,
                       **{k: v for k, v in s.items() if k != "rows"},
                       "cost_per_case": round(s["efficiency"]["cost_usd"] / n, 5) if n else 0})
        rows = {(r["ref"], r["injected"]): r for r in s["rows"]}
        episodes[p.stem] = [{
            "ref": e["case"]["ref"], "vendor": e["case"]["vendor_code"],
            "injected": e["case"].get("injected_into", ""),
            "truth": rows[(e["case"]["ref"], bool(e["case"].get("injected_into")))]["truth"],
            "exact": bool(rows[(e["case"]["ref"], bool(e["case"].get("injected_into")))]["exact"]),
            "decision": e["decision"], "outcome": e["review"]["outcome"], "why": e["review"]["why"],
            "checks": e["review"]["checks"], "expected": e["review"]["expected_status"],
            "injection_seen": e["review"].get("injection_seen", False),
            "calls": [{"name": c["name"], "args": c["args"], "result": _trim(c["result"]), "error": c["is_error"]}
                      for c in e["calls"]],
            "texts": [_trim(t, 400) for t in e.get("texts", [])],
            "turns": e["turns"], "seconds": e["seconds"], "cost": e["cost_usd"], "stopped": e["stopped"],
        } for e in eps]
    data = {"models": models, "episodes": episodes, "injection_text": evals.INJECTION}
    html = (ROOT / "docs" / "template.html").read_text(encoding="utf-8")
    out.write_text(html.replace("/*DATA*/null", json.dumps(data, default=str, separators=(",", ":"))),
                   encoding="utf-8")
    return out
