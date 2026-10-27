"""dispute-agent: investigate deductions with an AI agent, under guardrails, and measure it."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import typer

from dispute_agent import evals
from dispute_agent.agent import Budget, run_case
from dispute_agent.backends import make
from dispute_agent.world import ROOT, World

app = typer.Typer(add_completion=False, no_args_is_help=True)
RUNS = ROOT / "runs"


@app.command()
def case(vendor: str, deduction: str, backend: str = "ollama", model: str = typer.Option(None)) -> None:
    """Run the agent on one deduction and print the decision, the review and the trace."""
    ep = run_case({"vendor_code": vendor, "ref": deduction}, make(backend, model), World.load())
    for c in ep.calls:
        typer.echo(f"  {c['name']}({json.dumps(c['args'])}) -> {c['result'][:140]}")
    typer.echo(json.dumps(ep.decision, indent=2))
    typer.echo(f"{ep.review.outcome}: {ep.review.why}")
    for chk in ep.review.checks:
        typer.echo(f"  [{'ok' if chk.ok else 'FAIL'}] {chk.name} {chk.detail}")


@app.command()
def cases(out: Path = ROOT / "data" / "cases.csv", per_stratum: int = 5) -> None:
    """Write the stratified case set."""
    s = evals.sample_cases(per_stratum)
    s.to_csv(out, index=False)
    typer.echo(f"{len(s)} cases -> {out}\n{s.groupby(['category', 'status']).size().to_string()}")


@app.command()
def run(backend: str = "ollama", model: str = typer.Option(None), cases_file: Path = ROOT / "data" / "cases.csv",
        injection: int = 20, limit: int = 0, out: Path = typer.Option(None), max_turns: int = 16,
        procedure: int = 2) -> None:
    """Run the eval (resumable): every case, then the injection suite."""
    b = make(backend, model)
    world = World.load()
    cs = pd.read_csv(cases_file, dtype=str)
    if limit:
        cs = cs.head(limit)
    todo = [({"vendor_code": r.vendor_code, "ref": r.ref}, {}) for r in cs.itertuples()]
    todo += evals.injection_cases(world, cs, injection) if injection else []
    tag = "" if procedure == 2 else f"-procedure-v{procedure}"
    out = out or RUNS / f"{backend}-{(model or b.model).replace(':', '-').replace('/', '-')}{tag}.jsonl"
    evals.run(b, world, todo, out, Budget(max_turns=max_turns, procedure=procedure))
    typer.echo(f"episodes -> {out}")


@app.command()
def score(runs: list[Path], out: Path = typer.Option(None)) -> None:
    """Score one or more run files against the ground truth."""
    world = World.load()
    report = {}
    for p in runs:
        s = evals.score(evals.load(p), world)
        report[p.stem] = s
        typer.echo(f"\n{p.stem}\n" + json.dumps({k: v for k, v in s.items() if k != "rows"}, indent=2))
    if out:
        out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")


@app.command()
def site(runs: list[Path]) -> None:
    """Build docs/index.html from run files."""
    from dispute_agent.site import build
    typer.echo(f"-> {build(runs)}")


@app.command("serve-mcp")
def serve_mcp(vendor: str) -> None:
    """Serve the vendor-scoped, read-only tools over MCP (stdio)."""
    from dispute_agent.mcp_server import build
    build(vendor).run()


if __name__ == "__main__":
    app()
