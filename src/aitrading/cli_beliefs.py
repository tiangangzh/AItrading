"""Learning commands: the belief ledger and the research loop.

* ``aitrading research ["theme"]``   propose -> judge -> backtest -> learn rounds, then a holdout exam.
* ``aitrading beliefs list``         what the platform believes about every hypothesis tested in a market.
* ``aitrading beliefs show ID``      one hypothesis: statement, evidence, forecasts, belief.
* ``aitrading beliefs judge "idea"`` the judge's view of an idea BEFORE testing it (related experience).
* ``aitrading beliefs lessons``      per-concept lessons and the experience digest the agents read.
* ``aitrading beliefs calibration``  how good each judge's pre-registered forecasts were.
* ``aitrading beliefs export FILE``  point-in-time JSONL to train a judge model of your own.
* ``aitrading beliefs forget ID``    remove a hypothesis and its evidence.
* ``aitrading research --rl``        the same loop as reinforcement learning through the prompt: an LLM policy
                                     improved by paired tests, the Bayesian critic as reward, a holdout TD error.
* ``aitrading beliefs rl ...``       the RL policy and episodes: show / history / episodes / brief / reset /
                                     rollback VERSION / export FILE (actor training data).

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

JUDGES = ("bayes", "llm", "claude", "open")  # "claude" = "llm": the engine --llm selects


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
                        "'llm' or 'claude' (the LLM engine --llm selects - Claude or an open model - reads the "
                        "experience log and adds a literature prior), 'open' (a judge model you trained, behind an "
                        "OpenAI-compatible endpoint: --judge-url, --judge-model)")
    p.add_argument("--judge-url", default=None, metavar="URL", help="OpenAI-compatible base URL for --judge open, e.g. http://localhost:8000/v1")
    p.add_argument("--judge-model", default=None, metavar="NAME", help="served model name for --judge open")
    return p


def ledger_for(market: str) -> Any:
    from aitrading.beliefs.ledger import BeliefLedger

    return BeliefLedger(market)


def make_judge(args: argparse.Namespace, ledger: Any, llm: Any = None) -> Any:
    from aitrading.beliefs.judge import BayesianJudge, LLMJudge, OpenAICompatibleJudge

    kind = getattr(args, "judge", "bayes") or "bayes"
    if kind in ("llm", "claude"):
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
    if getattr(args, "rl", False):
        return _cmd_research_rl(args, theme)
    wants_llm = args.generator in ("auto", "llm", "claude") or args.judge in ("llm", "claude")
    engine = resolve_engine(args) if wants_llm else "offline"
    llm = make_llm(args.model, args.effort) if engine == "claude" else None
    if args.generator in ("llm", "claude") and llm is None:
        raise UsageError(f"--generator {args.generator} needs an LLM engine (ANTHROPIC_API_KEY for Claude, or --llm openai "
                         "--llm-base-url URL --model NAME for an open-source / OpenAI-compatible model), or use "
                         "--generator heuristic")
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


def _cmd_research_rl(args: argparse.Namespace, theme: str | None) -> int:
    from aitrading import cli_lab
    from aitrading.beliefs.rl.actor import HeuristicActor, PromptPolicyActor
    from aitrading.beliefs.rl.loop import RLConfig, RLResearchLoop
    from aitrading.beliefs.rl.reflect import Reflector

    wants_llm = args.generator in ("auto", "llm", "claude")
    engine = resolve_engine(args) if wants_llm else "offline"
    llm = make_llm(args.model, args.effort) if engine == "claude" else None
    if args.generator in ("llm", "claude") and llm is None:
        raise UsageError(f"--generator {args.generator} needs an LLM engine (ANTHROPIC_API_KEY for Claude, or --llm openai "
                         "--llm-base-url URL --model NAME), or use --generator heuristic")
    provider = cli_lab._lab_provider(args)
    end = args.end or default_as_of(args.provider, provider)
    ledger = ledger_for(_market(args))
    actor: Any = PromptPolicyActor(llm, effort=args.effort, proposals_per_call=args.proposals_per_call) if llm else HeuristicActor()
    reflector = Reflector(llm) if llm is not None and not args.no_reflect else None
    cfg = RLConfig(rounds=args.rounds, per_round=args.per_round, candidates_per_round=max(args.candidates, 2),
                   holdout_years=args.holdout_years, research_years=args.research_years or None, holdout_top=args.holdout_top,
                   end=end, costs_bps=args.costs_bps if args.costs_bps is not None else 10.0,
                   max_llm_calls=args.max_llm_calls)
    loop = RLResearchLoop(cli_lab._runner(provider), ledger, actor=actor, reflector=reflector, config=cfg,
                          progress=lambda m: _note(m))
    if args.walk_forward:
        from datetime import date as _date

        try:
            ends = [_date.fromisoformat(x.strip()) for x in args.walk_forward.split(",") if x.strip()]
            reports = loop.run_walk_forward(theme, ends)
        except ValueError as exc:
            raise UsageError(f"--walk-forward: {exc}") from exc
    else:
        reports = [loop.run(theme)]
    md = "\n".join(r.to_markdown() for r in reports)
    _emit(md)
    folder = Path(getattr(args, "out", None) or "aitrading_output") / "research" / datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    try:
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "report.md").write_text(md, encoding="utf-8")
        (folder / "report.json").write_text("[" + ",".join(r.to_json() for r in reports) + "]", encoding="utf-8")
        _note(f"research report: {folder / 'report.md'}; ledger: {ledger.path}")
    except OSError as exc:
        _note(f"warning: could not write the research report under {folder} ({exc}); the ledger is up to date")
    return EXIT_OK


def _cmd_beliefs_rl(args: argparse.Namespace, ledger: Any) -> int:
    from datetime import date as _date

    from aitrading.beliefs.rl.episodes import EpisodeStore
    from aitrading.beliefs.rl.policy import PolicyStore

    sub = (args.target or "show").lower()
    store = PolicyStore(ledger)
    if sub == "show":
        st = store.state(None)
        lines = [f"# RL policy - market '{ledger.market}'", "", f"Champion: {st.champion}"
                 + (f"; challenger {st.challenger} (SPRT {st.sprt.wins} wins, {st.sprt.losses} losses, LLR {st.sprt.llr:+.2f})"
                    if st.challenger else "; no challenger") + f"; {len(st.tabu)} tabu text(s).", ""]
        for vid in [st.champion] + ([st.challenger] if st.challenger else []):
            v = store.version(vid)
            lines.append(f"## {vid} ({v.author}; parent {v.parent or '-'}; horizon {v.info_horizon or '-'})")
            lines += [f"- [{d.directive_id}] {d.kind}: {d.text}" + (f" (cites {', '.join(d.cites)})" if d.cites else "")
                      for d in v.directives]
            zs = st.terminal_z.get(vid)
            if zs:
                lines.append(f"Exam TD errors: n {len(zs)}, mean {sum(zs) / len(zs):+.2f}")
            lines.append("")
        lines += [f"- {h}" for h in st.history]
        _emit("\n".join(lines) + "\n")
        return EXIT_OK
    if sub == "history":
        evs = store.load().events
        lines = ["| Seq | Kind | Horizon | Champion | Challenger / version | Outcome | z | Reasons |", "|---:|---|---|---|---|---|---:|---|"]
        for e in evs[-args.limit:]:
            lines.append(f"| {e.seq} | {e.kind} | {e.info_horizon} | {e.champion or ''} | {e.challenger or e.version or ''} | "
                         f"{e.outcome or ''} | {'' if e.z is None else f'{e.z:+.2f}'} | {', '.join(e.reasons)} |")
        _emit("\n".join(lines) + "\n")
        return EXIT_OK
    if sub == "episodes":
        rows = EpisodeStore(ledger).index()
        if not rows:
            _emit("No RL episodes yet. Run: aitrading research --rl\n")
            return EXIT_OK
        lines = ["| Episode | Status | Holdout | Tested | Finalists | Exam utility |", "|---|---|---|---:|---:|---:|"]
        for r in rows[-args.limit:]:
            lines.append(f"| {r.episode_id} | {r.status} | {r.holdout_window[0]}..{r.holdout_window[1]} | {r.n_tested} | "
                         f"{r.n_finalists} | {'' if r.r_oos is None else f'{r.r_oos:+.2f}'} |")
        _emit("\n".join(lines) + "\n")
        return EXIT_OK
    if sub == "brief":
        from aitrading.beliefs.rl.brief import brief_for_agent

        _emit(brief_for_agent(ledger, _date.today()) + "\n")
        return EXIT_OK
    if sub == "reset":
        store.reset(at_horizon=_date.min)
        store.save()
        _emit("the RL policy is back to the seed policy (learned versions are kept for the record)\n")
        return EXIT_OK
    if sub == "rollback":
        vid = (args.rest or [None])[0]
        if not vid:
            raise UsageError("aitrading beliefs rl rollback needs a VERSION (see: aitrading beliefs rl history)")
        try:
            v = store.version(vid)
        except KeyError:
            raise CLIError(f"no policy version '{vid}'") from None
        store.rollback(vid, at_horizon=v.info_horizon or _date.min)
        store.save()
        _emit(f"champion set to {vid} (for research runs whose holdout starts after {v.info_horizon or 'any date'})\n")
        return EXIT_OK
    if sub == "export":
        import json as _json

        from aitrading.beliefs.rl.export import actor_examples, dpo_pairs, grpo_prompts, kto_rows, sft_rows

        target = (args.rest or [None])[0]
        if not target:
            raise UsageError("aitrading beliefs rl export needs an output FILE")
        if not args.t_split:
            raise UsageError("aitrading beliefs rl export needs --t-split YYYY-MM-DD (train labels before it, eval after)")
        rows, dropped = actor_examples(ledger, EpisodeStore(ledger), t_split=args.t_split, llm_cutoff=args.llm_cutoff)
        fmt = args.format
        data = {"all": rows, "sft": sft_rows(rows), "dpo": dpo_pairs(rows), "kto": kto_rows(rows), "grpo": grpo_prompts(rows)}[fmt]
        path = Path(target)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            for r in data:
                fh.write(_json.dumps(r, ensure_ascii=False, default=str) + "\n")
        reasons: dict[str, int] = {}
        for d in dropped:
            reasons[d["reason"]] = reasons.get(d["reason"], 0) + 1
        _emit(f"wrote {len(data)} {fmt} row(s) to {path} from {len(rows)} logged proposal(s)"
              + (f"; dropped: {', '.join(f'{k} {v}' for k, v in sorted(reasons.items()))}" if reasons else "") + "\n")
        return EXIT_OK
    raise UsageError(f"unknown rl action '{sub}' (show, history, episodes, brief, reset, rollback, export)")


# --------------------------------------------------------------------------------------------
# beliefs
# --------------------------------------------------------------------------------------------


def _cmd_beliefs(args: argparse.Namespace) -> int:
    from aitrading.beliefs.judge import BayesianJudge

    market = _market(args)
    ledger = ledger_for(market)
    action = args.action
    if action == "rl":
        return _cmd_beliefs_rl(args, ledger)
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
    rs.add_argument("--generator", choices=("auto", "heuristic", "llm", "claude"), default="auto",
                    help="who proposes hypotheses: the LLM engine when one is configured (auto; 'llm' / 'claude' require "
                         "it - Claude or, with --llm openai, an open model), or the offline generator")
    rs.add_argument("--market", default=None, help="ledger name (default: the --provider)")
    rl = rs.add_argument_group("reinforcement learning (--rl)")
    rl.add_argument("--rl", action="store_true",
                    help="run the loop as RL through the prompt: the LLM's research policy text is improved by paired "
                         "tests, the Bayesian critic values every proposal, a holdout exam gives the TD error")
    rl.add_argument("--no-reflect", action="store_true", help="--rl: no reflection call (the policy text does not change)")
    rl.add_argument("--walk-forward", default=None, metavar="END1,END2,...",
                    help="--rl: one episode per end date, with chronological non-overlapping holdouts")
    rl.add_argument("--max-llm-calls", type=int, default=None, help="--rl: hard cap on LLM calls per episode")
    rl.add_argument("--costs-bps", type=float, default=None, help="--rl: one-way costs set on every proposal (default 10)")
    rl.add_argument("--proposals-per-call", type=int, default=None,
                    help="--rl: 1 = one proposal per LLM call (the format for small or fine-tuned open models)")
    rs.set_defaults(func=_cmd_research)

    bl = sub.add_parser("beliefs", parents=[base, engine, judge],
                        help="the belief ledger: list / show / judge / lessons / calibration / export / forget",
                        description="What the platform has learned from every backtest, holdout and paper record.")
    bl.add_argument("action", choices=("list", "show", "judge", "lessons", "calibration", "export", "forget", "rl"))
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
    bl.add_argument("--format", choices=("all", "sft", "dpo", "kto", "grpo"), default="all",
                    help="rl export: which actor-training rows to write")
    bl.add_argument("--t-split", type=_iso_date, default=None, metavar="YYYY-MM-DD",
                    help="rl export: train rows' labels end before this date; eval rows start 182 days after it")
    bl.set_defaults(func=_cmd_beliefs)
