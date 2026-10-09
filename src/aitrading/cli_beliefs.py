"""Learning commands: the belief ledger and the research loop.

* ``aitrading research ["theme"]``   propose -> judge -> backtest -> learn rounds, then a holdout exam.
* ``aitrading beliefs list``         what the platform believes about every hypothesis tested in a market.
* ``aitrading beliefs show ID``      one hypothesis: statement, evidence, forecasts, belief.
* ``aitrading beliefs judge "idea"`` the judge's view of an idea BEFORE testing it (related experience).
* ``aitrading beliefs lessons``      per-concept lessons and the experience digest the agents read.
* ``aitrading beliefs calibration``  how good each judge's pre-registered forecasts were.
* ``aitrading beliefs export FILE``  point-in-time JSONL to train a judge model of your own.
* ``aitrading beliefs forget ID``    remove a hypothesis and its evidence.

``backtest``, ``discover`` / ``ideas try`` and ``strategy run`` record their results in the ledger of
their market (the data provider) unless ``--no-learn`` is given; see docs/LEARNING_LOOP.md.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from aitrading.cli import (
    EXIT_OK,
    CLIError,
    UsageError,
    _emit,
    _iso_date,
    _note,
    default_as_of,
    default_provider_name,
    make_llm,
    resolve_engine,
)

JUDGES = ("bayes", "claude", "open")


# --------------------------------------------------------------------------------------------
# Shared helpers (also used by cli_lab)
# --------------------------------------------------------------------------------------------


def no_learn_option() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--no-learn", action="store_true",
                   help="do not record this run in the belief ledger (no learning from it)")
    return p


def learn_options() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(add_help=False, parents=[no_learn_option()])
    p.add_argument("--judge", choices=JUDGES, default="bayes",
                   help="who forecasts a hypothesis before it is tested: 'bayes' (the belief engine, default), "
                        "'claude' (Claude reads the experience log; adds a literature prior), 'open' (your own judge "
                        "model behind an OpenAI-compatible endpoint: --judge-url, --judge-model)")
    p.add_argument("--judge-url", default=None, metavar="URL", help="OpenAI-compatible base URL for --judge open, e.g. http://localhost:8000/v1")
    p.add_argument("--judge-model", default=None, metavar="NAME", help="served model name for --judge open")
    return p


def ledger_for(market: str) -> Any:
    from aitrading.beliefs.ledger import BeliefLedger

    return BeliefLedger(market)


def make_judge(args: argparse.Namespace, ledger: Any, llm: Any = None) -> Any:
    from aitrading.beliefs.judge import BayesianJudge, LLMJudge, OpenAICompatibleJudge

    kind = getattr(args, "judge", "bayes") or "bayes"
    if kind == "claude":
        if llm is None:
            llm = make_llm(getattr(args, "model", None), getattr(args, "effort", "high"))
        return LLMJudge(llm, ledger, effort="medium")
    if kind == "open":
        if not getattr(args, "judge_url", None) or not getattr(args, "judge_model", None):
            raise UsageError("--judge open needs --judge-url (e.g. http://localhost:8000/v1) and --judge-model")
        return OpenAICompatibleJudge(ledger, base_url=args.judge_url, model=args.judge_model)
    return BayesianJudge(ledger)


def make_learner(args: argparse.Namespace, market: str, llm: Any = None) -> Any:
    """A :class:`~aitrading.beliefs.learner.Learner` for ``market``, or None with --no-learn."""
    if getattr(args, "no_learn", False):
        return None
    from aitrading.beliefs.learner import Learner

    ledger = ledger_for(market)
    return Learner(ledger, make_judge(args, ledger, llm))


def update_lines(update: Any) -> list[str]:
    """What a run taught, for the terminal."""
    b, a = update.before, update.after
    head = "Learning:" if b.already_tested == 0 else "Learning (this hypothesis was tested before):"
    lines = [
        f"{head} [{update.hypothesis_id}] {update.statement}",
        f"  Before the run, {b.judge} predicted Sharpe {b.mean:+.2f} ± {b.sd:.2f} (P(>0) = {b.p_positive:.0%}; "
        f"{b.verdict.replace('_', ' ')}; {b.related_tested} related hypothesis(es) tested).",
        f"  Observed {update.observed_sharpe:+.2f} (surprise z = {update.surprise_z:+.1f}). Belief now: net Sharpe "
        f"{a.mean:+.2f} ± {a.sd:.2f}, P(>0) = {a.p_positive:.0%}, {a.status} ({a.n_evidence} run(s), "
        f"{a.oos_years:.1f}y out of sample).",
    ]
    for n in b.neighbors[:3]:
        lines.append(f"  Related: [{n.hypothesis_id}] {n.statement[:90]} (similarity {n.similarity:+.2f}): "
                     f"{n.mean:+.2f} ± {n.sd:.2f}")
    return lines


def report_learning(holder: Any) -> None:
    """Print the belief update (or the ledger error) of an IdeaLab result / IdeaLab."""
    upd = getattr(holder, "belief", None) if hasattr(holder, "belief") else getattr(holder, "last_belief", None)
    err = getattr(holder, "belief_error", None) if hasattr(holder, "belief_error") else getattr(holder, "last_belief_error", None)
    if err:
        _note(f"warning: {err} (the backtest itself is fine)")
    if upd is not None:
        _emit("\n".join(update_lines(upd)) + "\n")


def record_paper(market: str, strategy_name: str, slug: str, spec: Any, summary: dict[str, Any]) -> None:
    """Record a paper account's track record as out-of-sample evidence (a failure is only a warning)."""
    try:
        ledger = ledger_for(market)
        ev = ledger.record_paper(spec, slug, summary)
    except Exception as exc:  # noqa: BLE001 - learning must never break simulated trading
        _note(f"warning: could not record '{strategy_name}' in the belief ledger ({type(exc).__name__}: {exc})")
        return
    if ev is None:
        return
    from aitrading.beliefs.judge import BayesianJudge

    b = BayesianJudge(ledger).model().belief(ev.hypothesis_id)
    _note(f"learning: {strategy_name}'s simulated track record ({ev.n_periods} months, Sharpe {ev.sharpe:+.2f}) is "
          f"out-of-sample evidence; belief now {b.mean:+.2f} ± {b.sd:.2f} (P(>0) = {b.p_positive:.0%}, {b.status})")


def _market(args: argparse.Namespace) -> str:
    return getattr(args, "market", None) or getattr(args, "provider", None) or default_provider_name()


def _fmt(x: Any, nd: int = 2, sign: bool = True) -> str:
    if x is None:
        return "n/a"
    return f"{float(x):+.{nd}f}" if sign else f"{float(x):.{nd}f}"


# --------------------------------------------------------------------------------------------
# research
# --------------------------------------------------------------------------------------------


def _cmd_research(args: argparse.Namespace) -> int:
    from aitrading import cli_lab
    from aitrading.beliefs.loop import HeuristicGenerator, LLMGenerator, LoopConfig, ResearchLoop

    theme = " ".join(args.theme or []).strip() or None
    engine = resolve_engine(args) if args.generator in ("auto", "claude") or args.judge == "claude" else "offline"
    llm = make_llm(args.model, args.effort) if engine == "claude" else None
    if args.generator == "claude" and llm is None:
        raise UsageError("--generator claude needs Claude (set ANTHROPIC_API_KEY, or use --generator heuristic)")
    provider = cli_lab._lab_provider(args)
    end = args.end or default_as_of(args.provider, provider)
    market = _market(args)
    ledger = ledger_for(market)
    judge = make_judge(args, ledger, llm)
    if args.generator == "heuristic" or llm is None:
        generator: Any = HeuristicGenerator()
    else:
        from aitrading.strategy.nl import StrategyTranslator

        generator = LLMGenerator(llm, StrategyTranslator(llm, effort=args.effort), effort=args.effort,
                                 on_error=lambda m: _note(f"  {m}"))
    cfg = LoopConfig(rounds=args.rounds, per_round=args.per_round, candidates_per_round=args.candidates,
                     holdout_years=args.holdout_years, research_years=args.research_years or None,
                     holdout_top=args.holdout_top, end=end)
    loop = ResearchLoop(cli_lab._runner(provider), ledger, judge=judge, generator=generator, config=cfg,
                        progress=lambda m: _note(m))
    report = loop.run(theme)
    md = report.to_markdown()
    _emit(md)
    folder = Path(getattr(args, "out", None) or "aitrading_output") / "research" / datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    try:
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "report.md").write_text(md, encoding="utf-8")
        (folder / "report.json").write_text(report.to_json(), encoding="utf-8")
        _note(f"research report: {folder / 'report.md'}; ledger: {ledger.path}")
    except OSError as exc:
        _note(f"warning: could not write the research report under {folder} ({exc}); the ledger is up to date")
    return EXIT_OK


# --------------------------------------------------------------------------------------------
# beliefs
# --------------------------------------------------------------------------------------------


def _cmd_beliefs(args: argparse.Namespace) -> int:
    from aitrading.beliefs.judge import BayesianJudge

    market = _market(args)
    ledger = ledger_for(market)
    action = args.action
    if action == "list":
        model = BayesianJudge(ledger).model()
        beliefs = model.beliefs()
        if args.status:
            beliefs = [b for b in beliefs if b.status == args.status]
        if not beliefs:
            _emit(f"No tested hypotheses in market '{market}' yet. Backtest an idea, or run: aitrading research\n")
            return EXIT_OK
        beliefs.sort(key=lambda b: -b.mean)
        h = model.hyper
        lines = [f"# Beliefs - market '{market}' ({len(beliefs)} hypotheses; learned prior for a new idea: gross Sharpe "
                 f"{h.m0:+.2f} ± {h.s0:.2f})", "",
                 "| Id | Hypothesis | Belief (net Sharpe) | Gross / costs | P(>0) | Runs | Years | Out-of-sample | Status |",
                 "|---|---|---:|---:|---:|---:|---:|---|---|"]
        for b in beliefs[: args.limit]:
            oos = f"{b.oos_sharpe:+.2f} / {b.oos_years:.1f}y" if b.oos_sharpe is not None else "-"
            stmt = b.statement.replace("|", "/")
            lines.append(f"| {b.hypothesis_id} | {stmt[:100]} | {b.mean:+.2f} ± {b.sd:.2f} | {b.gross_mean:+.2f} / {b.cost:.2f} | {b.p_positive:.0%} | "
                         f"{b.n_evidence} | {b.years:.1f} | {oos} | {b.status} |")
        _emit("\n".join(lines) + "\n")
        return EXIT_OK
    if action == "show":
        hyp = ledger.hypothesis(args.target or "")
        if hyp is None:
            raise CLIError(f"no hypothesis '{args.target}' in market '{market}' (see: aitrading beliefs list)")
        model = BayesianJudge(ledger).model()
        b = model.belief(hyp.hypothesis_id)
        lines = [f"# [{hyp.hypothesis_id}] {hyp.statement}", ""]
        if hyp.mechanism:
            lines += [f"Mechanism: {hyp.mechanism}", ""]
        lines += [f"Concept: {hyp.concept}; tested as: {', '.join(hyp.names) or '-'}; source: {hyp.source}",
                  f"Belief: net Sharpe {b.mean:+.2f} ± {b.sd:.2f} (gross {b.gross_mean:+.2f}, cost drag {b.cost:.2f}), "
                  f"P(>0) = {b.p_positive:.0%}, P(>{b.hurdle:g}) = {b.p_hurdle:.0%}, {b.status}", ""]
        if hyp.llm_prior is not None:
            lp = hyp.llm_prior
            lines += [f"Literature prior ({lp.model}): {lp.mean:+.2f} ± {lp.sd:.2f}. {lp.reasoning}", ""]
        lines += ["| Evidence | Kind | Window | Sharpe | ± (sampling) | Out of sample | Caveats |", "|---|---|---|---:|---:|---|---|"]
        for e in ledger.evidence_for(hyp.hypothesis_id):
            lines.append(f"| {e.evidence_id[:48]} | {e.kind} | {e.start} to {e.end} | {e.sharpe:+.2f} | {e.se:.2f} | "
                         f"{'yes' if e.out_of_sample else 'no'} | {'; '.join(c[:60] for c in e.caveats)} |")
        preds = [p for p in ledger.predictions() if p.hypothesis_id == hyp.hypothesis_id]
        if preds:
            lines += ["", "| Forecast made | Judge | For | Predicted | Outcome |", "|---|---|---|---:|---:|"]
            for p in preds:
                ev = ledger.data.evidence.get(p.target_evidence_id or "")
                lines.append(f"| {p.made_at:%Y-%m-%d %H:%M} | {p.judge} | {p.purpose} | {p.mean:+.2f} ± {p.sd:.2f} | "
                             f"{_fmt(ev.sharpe) if ev else 'pending'} |")
        _emit("\n".join(lines) + "\n")
        return EXIT_OK
    if action == "judge":
        idea = " ".join(args.rest or []).strip() or (args.target or "")
        if args.target and args.rest:
            idea = f"{args.target} {idea}"
        if not idea:
            raise UsageError('give an idea, e.g. aitrading beliefs judge "5-day momentum"')
        llm = make_llm(args.model, args.effort) if resolve_engine(args) == "claude" else None
        if llm is not None:
            from aitrading.strategy.nl import StrategyTranslator

            spec = StrategyTranslator(llm, effort=args.effort).translate(idea).spec
        else:
            from aitrading.strategy.nl import HeuristicStrategyTranslator

            spec = HeuristicStrategyTranslator().translate(idea).spec
        a = make_judge(args, ledger, llm).assess(spec)
        _emit(f"# [{a.hypothesis_id}] {a.statement}\n\nJudge: {a.judge}\n\n{a.critique}\n")
        return EXIT_OK
    if action == "lessons":
        from aitrading.beliefs.digest import experience_digest

        _emit(experience_digest(BayesianJudge(ledger).model(), max_items=args.limit) + "\n")
        return EXIT_OK
    if action == "calibration":
        from aitrading.beliefs.calibration import calibration_report

        rows = calibration_report(ledger)
        if not rows:
            _emit("No resolved forecasts yet: every backtest with learning on adds one.\n")
            return EXIT_OK
        lines = [f"# Judge calibration - market '{market}'", "",
                 "Brier skill > 0 and log-score gain > 0 mean the judge beat a coin flip / a naive prior; an 80% "
                 "interval should contain about 80% of outcomes. Out-of-sample rows (holdout, paper) are the ones "
                 "that matter.", "",
                 "| Judge | Target | N | Brier | Brier skill | Log score | Gain vs naive | 80% coverage | Mean z |",
                 "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
        for r in rows:
            lines.append(f"| {r.judge} | {r.target_kind} | {r.n} | {r.brier:.3f} | {r.brier_skill:+.2f} | {r.log_score:.2f} | "
                         f"{r.log_score_gain:+.2f} | {r.coverage_80:.0%} | {r.mean_z:+.2f} |")
        _emit("\n".join(lines) + "\n")
        return EXIT_OK
    if action == "export":
        from aitrading.beliefs.dataset import build_examples, write_jsonl

        if not args.target:
            raise UsageError("aitrading beliefs export needs an output FILE (e.g. judge_data.jsonl)")
        kinds = tuple(k.strip() for k in args.kinds.split(",") if k.strip())
        ex = build_examples(ledger, kinds=kinds, test_from=args.test_from, llm_cutoff=args.llm_cutoff)
        path = write_jsonl(ex, args.target)
        n_test = sum(1 for e in ex if e["split"] == "test")
        _emit(f"wrote {len(ex)} example(s) ({n_test} test) to {path}\n")
        return EXIT_OK
    if action == "forget":
        if not args.target:
            raise UsageError("aitrading beliefs forget needs a hypothesis ID")
        try:
            n = ledger.forget(args.target)
        except KeyError:
            raise CLIError(f"no hypothesis '{args.target}' in market '{market}'") from None
        _emit(f"removed {args.target} and {n} evidence record(s)\n")
        return EXIT_OK
    raise UsageError(f"unknown action {action}")  # pragma: no cover - argparse restricts it


# --------------------------------------------------------------------------------------------
# Parser registration
# --------------------------------------------------------------------------------------------


def register(sub: Any, *, base: argparse.ArgumentParser, engine: argparse.ArgumentParser, data: argparse.ArgumentParser) -> None:
    out = argparse.ArgumentParser(add_help=False)
    out.add_argument("--out", default="aitrading_output", metavar="DIR", help="output folder (default: aitrading_output)")
    judge = learn_options()

    rs = sub.add_parser("research", parents=[base, engine, data, out, judge],
                        help="propose hypotheses, judge them against experience, test, learn; then a holdout exam",
                        description="The learning research loop (see docs/LEARNING_LOOP.md).")
    rs.add_argument("theme", nargs="*", help='optional theme, e.g. "short-term momentum and reversal"')
    rs.add_argument("--rounds", type=int, default=3, help="propose / test rounds (default 3)")
    rs.add_argument("--per-round", type=int, default=3, help="backtests per round (default 3)")
    rs.add_argument("--candidates", type=int, default=6, help="hypotheses the generator proposes per round (default 6)")
    rs.add_argument("--holdout-years", type=float, default=2.0, help="final years kept for the out-of-sample exam (default 2)")
    rs.add_argument("--research-years", type=float, default=8.0, help="years before the holdout to research on (0 = all; default 8)")
    rs.add_argument("--holdout-top", type=int, default=2, help="best hypotheses that sit the holdout exam (default 2)")
    rs.add_argument("--end", type=_iso_date, default=None, metavar="YYYY-MM-DD", help="last date of data (default: latest)")
    rs.add_argument("--generator", choices=("auto", "heuristic", "claude"), default="auto",
                    help="who proposes hypotheses: Claude when available (auto), or the offline generator")
    rs.add_argument("--market", default=None, help="ledger name (default: the --provider)")
    rs.set_defaults(func=_cmd_research)

    bl = sub.add_parser("beliefs", parents=[base, engine, judge],
                        help="the belief ledger: list / show / judge / lessons / calibration / export / forget",
                        description="What the platform has learned from every backtest, holdout and paper record.")
    bl.add_argument("action", choices=("list", "show", "judge", "lessons", "calibration", "export", "forget"))
    bl.add_argument("target", nargs="?", default=None, help="hypothesis ID (show / forget), idea (judge), file (export)")
    bl.add_argument("rest", nargs="*", help=argparse.SUPPRESS)
    bl.add_argument("--market", default=None, help=f"ledger name = data provider (default: {default_provider_name()})")
    bl.add_argument("--status", choices=("supported", "refuted", "uncertain"), default=None, help="list: filter by status")
    bl.add_argument("--limit", type=int, default=50)
    bl.add_argument("--kinds", default="backtest,holdout,paper", help="export: evidence kinds used as targets")
    bl.add_argument("--test-from", type=_iso_date, default=None, metavar="YYYY-MM-DD",
                    help="export: targets whose data start on/after this date go to the test split")
    bl.add_argument("--llm-cutoff", type=_iso_date, default=None, metavar="YYYY-MM-DD",
                    help="export: the base model's knowledge cutoff (flags targets free of pretraining look-ahead)")
    bl.set_defaults(func=_cmd_beliefs)
