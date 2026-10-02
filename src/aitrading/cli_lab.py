"""Idea-lab commands for the CLI: backtest ideas, run simulated strategies, discover ideas, dashboard.

* ``aitrading backtest "idea"``      idea -> StrategySpec -> point-in-time backtest -> verdict + HTML.
* ``aitrading library``              the built-in idea templates (factor models, anomalies, timing rules).
* ``aitrading strategy ...``         save / list / run (simulated trading) / status / reset / delete.
* ``aitrading discover``             find ideas (arXiv, feeds, Claude web search, a URL or PDF), ask the
                                     trader idea by idea, backtest + replicate the accepted ones.
* ``aitrading ideas ...``            the idea inbox: list / show / try / reject / later.
* ``aitrading dashboard``            index page linking every report, strategy and idea.

Simulated trading only: nothing here places orders with a broker.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Callable

from aitrading.cli import (
    EXIT_ERROR,
    EXIT_OK,
    CLIError,
    UsageError,
    _emit,
    _iso_date,
    _note,
    _provider_from_args,
    _should_open,
    default_as_of,
    make_llm,
    resolve_engine,
)

Prompt = Callable[[str], str]


# --------------------------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------------------------


def _out_dir(args: argparse.Namespace) -> Path:
    return Path(getattr(args, "out", None) or "aitrading_output")


SYNTHETIC_HISTORY_START = date(2012, 1, 2)  # backtests on the simulated market need a long history


def _lab_provider(args: argparse.Namespace) -> Any:
    """Provider for backtest-style commands; the simulated market gets a long history."""
    if getattr(args, "provider", None) == "synthetic" and not getattr(args, "tickers", None) and not getattr(args, "universe_file", None):
        from aitrading.cli import make_provider

        return make_provider("synthetic", start=SYNTHETIC_HISTORY_START)
    return _provider_from_args(args)


def _llm_or_none(args: argparse.Namespace) -> Any:
    return make_llm(args.model, args.effort) if resolve_engine(args) == "claude" else None


def _lab(args: argparse.Namespace, provider: Any, llm: Any) -> Any:
    from aitrading.idealab import IdeaLab
    from aitrading.strategy.interpret import BacktestInterpreter, HeuristicInterpreter
    from aitrading.strategy.nl import HeuristicStrategyTranslator, StrategyTranslator

    translator = StrategyTranslator(llm, effort=args.effort) if llm is not None else HeuristicStrategyTranslator()
    interpreter = BacktestInterpreter(llm, effort=args.effort) if llm is not None else HeuristicInterpreter()
    return IdeaLab(provider, translator=translator, interpreter=interpreter, out_dir=_out_dir(args) / "backtests")


def _apply_overrides(spec: Any, args: argparse.Namespace) -> Any:
    update: dict[str, Any] = {}
    if getattr(args, "start", None):
        update["start"] = args.start
    if getattr(args, "end", None):
        update["end"] = args.end
    if getattr(args, "rebalance", None):
        update["rebalance"] = args.rebalance
    if getattr(args, "costs_bps", None) is not None:
        update["costs_bps"] = args.costs_bps
    return spec.model_copy(update=update) if update else spec


def _fmt(x: Any, unit: str = "", digits: int = 2) -> str:
    if x is None:
        return "n/a"
    try:
        return f"{float(x):.{digits}f}{unit}"
    except (TypeError, ValueError):
        return str(x)


def _summary_lines(result: Any) -> list[str]:
    st = result.stats.get("strategy")
    reg = result.regression
    interp = result.interpretation
    lines = [f"Idea: {result.idea}", f"Period: {result.start} to {result.end} ({result.rebalance} rebalance), data: {result.provider}"]
    if st is not None:
        lines.append(
            f"Strategy: CAGR {_fmt(st.cagr_pct, '%')} | Sharpe {_fmt(st.sharpe)} | max drawdown {_fmt(st.max_drawdown_pct, '%')}"
            f" | volatility {_fmt(st.volatility_pct, '%')}"
        )
    bench = result.stats.get("benchmark")
    if bench is not None:
        lines.append(f"Benchmark: CAGR {_fmt(bench.cagr_pct, '%')} | Sharpe {_fmt(bench.sharpe)}")
    if reg is not None:
        lines.append(f"{reg.model.upper()} alpha: {_fmt(reg.alpha_annual_pct, '% a year')} (t = {_fmt(reg.alpha_t_stat)}), R^2 {_fmt(reg.r_squared)}")
    if result.quantiles is not None:
        q = result.quantiles
        lines.append(f"Quantile spread: {_fmt(q.spread_annual_pct, '% a year')}, monotonicity {_fmt(q.monotonicity)}, IC {_fmt(q.ic_mean, '', 3)} (t = {_fmt(q.ic_t_stat)})")
    for chk in result.factor_checks:
        lines.append(
            f"Factor {chk.factor}: premium {_fmt(chk.annual_premium_constructed_pct, '%')} (official {_fmt(chk.annual_premium_official_pct, '%')}),"
            f" correlation with official {_fmt(chk.correlation_with_official)}"
        )
    if interp is not None:
        summary = interp.summary
        if summary.lower().startswith("verdict:"):
            summary = summary.split("-", 1)[-1].strip() if " - " in summary else summary[8:].strip()
        lines.append(f"Verdict: {interp.verdict.upper()} - {summary}")
    if result.warnings:
        lines.append(f"{len(result.warnings)} warning(s) in the report (survivorship, data coverage, costs ...).")
    return lines


def _open(path: Path | None, args: argparse.Namespace) -> None:
    if path is not None and _should_open(args):
        from aitrading.report.dashboard import open_in_browser

        open_in_browser(path)


def _store() -> Any:
    from aitrading.trading.store import StrategyStore

    return StrategyStore()


def _inbox() -> Any:
    from aitrading.discovery.inbox import IdeaInbox

    return IdeaInbox()


# --------------------------------------------------------------------------------------------
# backtest / library
# --------------------------------------------------------------------------------------------


def _cmd_backtest(args: argparse.Namespace) -> int:
    from aitrading.strategy.spec import StrategySpec

    idea = " ".join(args.idea or []).strip()
    if not idea and not args.spec_file:
        raise UsageError("give an idea (e.g. aitrading backtest \"Fama-French 3 factor model\") or --spec-file")
    provider = _lab_provider(args)
    llm = _llm_or_none(args)
    lab = _lab(args, provider, llm)
    spec = None
    if args.spec_file:
        try:
            spec = StrategySpec.model_validate_json(Path(args.spec_file).read_text(encoding="utf-8-sig"))
        except (OSError, ValueError) as exc:
            raise UsageError(f"cannot read --spec-file: {exc}") from exc
    elif any(getattr(args, k, None) not in (None, False) for k in ("start", "end", "rebalance", "costs_bps")):
        spec = _apply_overrides(lab.translator.translate(idea).spec, args)
    out = lab.run(idea or None, spec=spec)
    result = out.result
    if args.format == "json":
        _emit(result.model_dump_json(indent=2) + "\n")
    else:
        _emit("\n".join(_summary_lines(result)) + "\n")
    if out.html_path is not None:
        _note(f"report: {out.html_path}")
    if args.save:
        from aitrading.strategy.library import TEMPLATES

        path = _store().save(
            args.save, out.spec, result, overwrite=args.overwrite, idea=idea or out.spec.idea,
            template=out.spec.name if out.spec.name in TEMPLATES else None,
        )
        _note(f"saved strategy '{args.save}' to {path}; simulate it with: aitrading strategy run {args.save}")
    _open(out.html_path, args)
    return EXIT_OK


def _cmd_library(args: argparse.Namespace) -> int:
    from aitrading.strategy.library import TEMPLATES

    lines = ["# Built-in idea library", "", "| Key | Idea | Kind | References |", "|---|---|---|---|"]
    for key, t in TEMPLATES.items():
        spec = t.build()
        refs = "; ".join(t.references[:2])
        lines.append(f"| `{key}` | {t.title} | {spec.kind} | {refs} |")
    lines += ["", 'Backtest one with e.g. `aitrading backtest "' + next(iter(TEMPLATES.values())).title + '"`.']
    _emit("\n".join(lines) + "\n")
    return EXIT_OK


# --------------------------------------------------------------------------------------------
# strategy (simulated trading)
# --------------------------------------------------------------------------------------------


def _runner(provider: Any) -> Any:
    from aitrading.backtest.runner import StrategyRunner

    return StrategyRunner(provider)


def _strategy_page(store: Any, name: str, out_root: Path) -> Path:
    from aitrading.report.html import render_strategy_page
    from aitrading.trading.paper import PaperAccount

    spec, backtest, _meta = store.load(name)
    account = PaperAccount(store, name)
    try:
        paper = account.summary(backtest=backtest)
    except Exception:  # noqa: BLE001 - never started: show the backtest only
        paper = None
    path = out_root / "strategies" / f"{store.path(name).name}.html"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_strategy_page(name, spec, backtest, paper), encoding="utf-8")
    return path


def _cmd_strategy(args: argparse.Namespace) -> int:
    from aitrading.trading.paper import PaperAccount

    store = _store()
    action = args.action
    if action == "list":
        rows = store.list()
        if not rows:
            _emit("No saved strategies. Save one with: aitrading backtest \"idea\" --save NAME\n")
            return EXIT_OK
        lines = ["| Strategy | Idea | Kind | Backtest Sharpe | Verdict | Simulated | Updated |", "|---|---|---|---:|---|---|---|"]
        for r in rows:
            bt = r.get("backtest") or {}
            lines.append(
                f"| {r.get('name')} | {str(r.get('idea') or '')[:50]} | {r.get('kind') or ''} | {_fmt(bt.get('sharpe'))} | "
                f"{bt.get('verdict') or ''} | {'yes' if r.get('paper_trading') else 'no'} | {r.get('updated') or ''} |"
            )
        _emit("\n".join(lines) + "\n")
        return EXIT_OK
    if not args.name:
        raise UsageError(f"aitrading strategy {action} needs a strategy NAME")
    name = args.name
    if action == "delete":
        store.delete(name)
        _emit(f"deleted strategy '{name}'\n")
        return EXIT_OK
    if action == "reset":
        archived = PaperAccount(store, name).reset(args.capital)
        _emit(f"reset the simulated account of '{name}'" + (f" (old ledger archived at {archived})" if archived else "") + "\n")
        return EXIT_OK
    if action == "run":
        spec, backtest, _meta = store.load(name)
        provider = _lab_provider(args)
        as_of = args.as_of or default_as_of(args.provider, provider)
        account = PaperAccount(store, name, initial_capital=args.capital) if args.capital else PaperAccount(store, name)
        report = account.rebalance(_runner(provider), spec, as_of, force=args.force)
        if report.skipped_reason:
            _emit(f"{name} {as_of}: no trades ({report.skipped_reason}); NAV {report.nav_after:,.2f}\n")
        else:
            lines = [f"{name} {as_of}: {len(report.orders)} simulated order(s); NAV {report.nav_before:,.2f} -> {report.nav_after:,.2f}"]
            for o in report.orders[:40]:
                lines.append(f"  {o.side:5s} {o.ticker:8s} {o.shares:12,.2f} @ {o.price:10,.2f}  cost {o.cost:,.2f}")
            if len(report.orders) > 40:
                lines.append(f"  ... {len(report.orders) - 40} more (see the strategy page)")
            _emit("\n".join(lines) + "\n")
        for w in report.warnings:
            _note(f"warning: {w}")
    page = _strategy_page(store, name, _out_dir(args))
    if action == "status":
        account = PaperAccount(store, name)
        try:
            s = account.summary(backtest=store.load_backtest(name))
            _emit(
                f"{name}: NAV {s.get('nav', 0):,.2f}, since start {_fmt(s.get('since_start_return_pct'), '%')}"
                f" (backtest expected {_fmt(s.get('backtest_expected_return_pct'), '%')}), last run {s.get('last_run')}\n"
            )
        except Exception as exc:  # noqa: BLE001
            _emit(f"{name}: not started yet ({exc}). Start it with: aitrading strategy run {name}\n")
    _note(f"strategy page: {page}")
    _open(page, args)
    return EXIT_OK


# --------------------------------------------------------------------------------------------
# discover / ideas
# --------------------------------------------------------------------------------------------


def _ask(prompt: str, choices: str, default: str, ask: Prompt) -> str:
    """Prompt until one of ``choices`` (single letters) is given; EOF / empty -> default."""
    while True:
        try:
            answer = ask(prompt).strip().lower()
        except EOFError:
            return default
        if not answer:
            return default
        if answer[0] in choices:
            return answer[0]
        print(f"  please answer one of: {', '.join(choices)}", file=sys.stderr)


def _idea_card(c: Any) -> str:
    from aitrading.discovery.rank import security_flags

    e = c.extraction
    claims = []
    if e.reported_sharpe is not None:
        claims.append(f"Sharpe {e.reported_sharpe:g}")
    if e.reported_t_stat is not None:
        claims.append(f"t-stat {e.reported_t_stat:g}")
    if e.reported_annual_return_pct is not None:
        claims.append(f"{e.reported_annual_return_pct:g}% a year")
    quotes_ok = sum(q.status == "verified" for q in c.quote_checks)
    lines = [
        "-" * 78,
        f"[{c.idea_id}] {e.title}   (score {c.score:.2f}, {c.novelty.replace('_', ' ')})",
        f"Source: {c.source.title} - {c.source.url}" + (f" ({c.source.published})" if c.source.published else ""),
        f"Summary: {e.summary}",
        f"Claim: {e.claimed_effect}" + (f" [reported: {', '.join(claims)}]" if claims else ""),
        f"Testable here: {e.testability.replace('_', ' ')}" + (f" - missing: {', '.join(e.missing_data)}" if e.missing_data else ""),
        f"Evidence quotes verified against the source: {quotes_ok}/{len(c.quote_checks)}",
    ]
    if e.proposed_strategy_idea:
        lines.append(f"Would test: {e.proposed_strategy_idea}")
    flags = security_flags(c)
    if flags:
        lines.append("SECURITY: the source contains instruction-like text aimed at AI systems: " + "; ".join(flags[:3]))
    return "\n".join(lines)


def _gather_documents(args: argparse.Namespace, llm: Any) -> list[Any]:
    from aitrading.discovery.sources import ArxivSource, FeedSource, fetch_url, load_pdf

    docs: list[Any] = []
    since = args.since or (date.today() - timedelta(days=args.days))
    if args.url:
        for u in args.url:
            docs.append(fetch_url(u))
    if args.pdf:
        for p in args.pdf:
            docs.append(load_pdf(p))
    if args.url or args.pdf:
        return docs
    sources = {s.strip().lower() for s in args.sources.split(",") if s.strip()}
    if "arxiv" in sources:
        arxiv = ArxivSource()
        queries = args.query or None
        found = arxiv.search_many(queries, max_results_per_query=args.max, since=since) if queries else arxiv.search_many(max_results_per_query=args.max, since=since)
        docs += found
        for w in arxiv.warnings:
            _note(f"arxiv: {w}")
    if "feeds" in sources:
        feeds = FeedSource()
        docs += feeds.fetch(max_items_per_feed=args.max, since=since)
        for w in feeds.warnings:
            _note(f"feeds: {w}")
    if "web" in sources:
        if llm is None:
            _note("web search needs Claude (set ANTHROPIC_API_KEY); skipping the 'web' source")
        else:
            from aitrading.discovery.websearch import ClaudeWebSearchSource

            brief = " ".join(args.query) if args.query else "recent systematic US equity strategies and anomalies"
            web = ClaudeWebSearchSource(model=getattr(llm, "model", None) or "claude-opus-5-5", effort=args.effort)
            docs += web.discover(brief, max_ideas=args.max)
            for w in web.warnings:
                _note(f"web: {w}")
    return docs


def _try_idea(c: Any, args: argparse.Namespace, *, lab: Any, inbox: Any, ask: Prompt) -> None:
    """Translate an accepted idea, backtest it, run the replication suite, record and show the result."""
    from aitrading.discovery.rank import is_security_flagged

    if is_security_flagged(c):
        if _ask("  This source contains instruction-like text. Test it anyway? [y/N] ", "yn", "n", ask) != "y":
            inbox.set_status(c.idea_id, "deferred", note="security flag: trader declined to test")
            return
    text = c.extraction.proposed_strategy_idea or c.extraction.title
    if not text:
        inbox.set_status(c.idea_id, "failed", note="no testable strategy description")
        _note("  nothing testable was extracted from this idea")
        return
    inbox.set_status(c.idea_id, "accepted")
    try:
        translation = lab.translator.translate(text)
        spec = translation.spec
        inbox.attach_spec(c.idea_id, spec)
        print(f"  Testing: {spec.name} ({spec.kind}, {spec.rebalance}) ...", file=sys.stderr)
        result, replication, html_path = lab.replicate_idea(c, spec)
    except Exception as exc:  # noqa: BLE001 - one failed idea must not stop the session
        inbox.set_status(c.idea_id, "failed", note=f"{type(exc).__name__}: {exc}")
        _note(f"  could not test this idea: {type(exc).__name__}: {exc}")
        return
    inbox.attach_result(c.idea_id, result.run_id, replication)
    print("\n".join("  " + line for line in _summary_lines(result)))
    print(f"  Replication: {replication.verdict.replace('_', ' ').upper()} - {replication.summary}")
    for chk in replication.checks:
        mark = {True: "pass", False: "FAIL", None: "n/a "}[chk.passed]
        print(f"    [{mark}] {chk.name}: Sharpe {_fmt(chk.sharpe)}  {chk.note}")
    if html_path is not None:
        print(f"  Report: {html_path}")
    if _ask("  Save it as a strategy for simulated trading? [y/N] ", "yn", "n", ask) == "y":
        name = (spec.name or c.idea_id)[:40]
        _store().save(name, spec, result, overwrite=True, idea=c.extraction.title, notes=f"from idea {c.idea_id}: {c.source.url}")
        inbox.set_status(c.idea_id, "saved")
        print(f"  Saved as '{name}'. Simulate it with: aitrading strategy run {name}")


def _review_loop(cands: list[Any], args: argparse.Namespace, *, lab: Any, inbox: Any, ask: Prompt) -> None:
    for c in cands:
        print(_idea_card(c))
        if c.extraction.testability == "not_testable":
            print("  (not testable with the data this platform has; kept in the inbox for reference)")
            continue
        choice = _ask("Try this strategy? [y]es / [n]o / [l]ater / [q]uit: ", "ynlq", "l", ask)
        if choice == "q":
            break
        if choice == "n":
            inbox.set_status(c.idea_id, "rejected")
        elif choice == "l":
            inbox.set_status(c.idea_id, "deferred")
        else:
            _try_idea(c, args, lab=lab, inbox=inbox, ask=ask)


def _cmd_discover(args: argparse.Namespace, ask: Prompt = input) -> int:
    from aitrading.discovery.extract import HeuristicIdeaExtractor, IdeaExtractor

    llm = _llm_or_none(args)
    docs = _gather_documents(args, llm)
    if not docs:
        _emit("No documents found. Try --query, --days, --sources arxiv,feeds,web, --url or --pdf.\n")
        return EXIT_OK
    extractor = IdeaExtractor(llm, effort=args.effort) if llm is not None else HeuristicIdeaExtractor()
    candidates = []
    for d in docs:
        try:
            c = extractor.extract(d)
        except Exception as exc:  # noqa: BLE001
            _note(f"skipped {d.url}: {type(exc).__name__}: {exc}")
            continue
        if c.extraction.is_trading_idea:
            candidates.append(c)
    inbox = _inbox()
    added = inbox.add(candidates)
    _note(f"{len(docs)} document(s) read, {len(candidates)} trading idea(s), {len(added)} new in the inbox")
    new = [c for c in inbox.list(status="new")][: args.review]
    if not new:
        _emit("No new ideas to review. See them all with: aitrading ideas list\n")
        return EXIT_OK
    if args.no_interactive or not sys.stdin.isatty() and ask is input:
        for c in new:
            _emit(_idea_card(c) + "\n")
        _emit("Review them later with: aitrading ideas try <idea_id>\n")
        return EXIT_OK
    provider = _lab_provider(args)
    _review_loop(new, args, lab=_lab(args, provider, llm), inbox=inbox, ask=ask)
    return EXIT_OK


def _cmd_ideas(args: argparse.Namespace, ask: Prompt = input) -> int:
    inbox = _inbox()
    if args.action == "list":
        items = inbox.list(status=args.status) if args.status else inbox.list()
        if not items:
            _emit("The idea inbox is empty. Find ideas with: aitrading discover\n")
            return EXIT_OK
        lines = ["| Id | Status | Score | Testable | Idea | Source |", "|---|---|---:|---|---|---|"]
        for c in items[: args.limit]:
            lines.append(
                f"| {c.idea_id} | {c.status} | {c.score:.2f} | {c.extraction.testability.replace('_', ' ')} | "
                f"{c.extraction.title[:50]} | {c.source.source_name or c.source.source_type} |"
            )
        _emit("\n".join(lines) + "\n")
        return EXIT_OK
    if not args.idea_id:
        raise UsageError(f"aitrading ideas {args.action} needs an IDEA_ID (see: aitrading ideas list)")
    c = inbox.get(args.idea_id)
    if c is None:
        raise CLIError(f"no idea with id '{args.idea_id}' in the inbox")
    if args.action == "show":
        _emit(_idea_card(c) + "\n")
        return EXIT_OK
    if args.action == "reject":
        inbox.set_status(c.idea_id, "rejected")
        _emit(f"rejected {c.idea_id}\n")
        return EXIT_OK
    if args.action == "later":
        inbox.set_status(c.idea_id, "deferred")
        _emit(f"deferred {c.idea_id}\n")
        return EXIT_OK
    # try
    llm = _llm_or_none(args)
    provider = _lab_provider(args)
    print(_idea_card(c))
    _try_idea(c, args, lab=_lab(args, provider, llm), inbox=inbox, ask=ask)
    return EXIT_OK


# --------------------------------------------------------------------------------------------
# dashboard
# --------------------------------------------------------------------------------------------


def _cmd_dashboard(args: argparse.Namespace) -> int:
    from aitrading.report.dashboard import write_dashboard

    out = _out_dir(args)
    path = write_dashboard(out, runs_dir=out)
    _emit(f"dashboard: {path}\n")
    _open(path, args)
    return EXIT_OK


# --------------------------------------------------------------------------------------------
# Parser registration
# --------------------------------------------------------------------------------------------


def register(sub: Any, *, base: argparse.ArgumentParser, engine: argparse.ArgumentParser, data: argparse.ArgumentParser) -> None:
    """Add the idea-lab subcommands to the main CLI parser."""
    out = argparse.ArgumentParser(add_help=False)
    out.add_argument("--out", default="aitrading_output", metavar="DIR", help="output folder (default: aitrading_output)")
    browser = out.add_mutually_exclusive_group()
    browser.add_argument("--open", dest="open_browser", action="store_true", default=None, help="open the HTML page")
    browser.add_argument("--no-open", dest="open_browser", action="store_false", help="do not open the HTML page")

    bt = sub.add_parser("backtest", parents=[base, engine, data, out], help="backtest a research idea",
                        description="Translate an idea into a strategy, backtest it point-in-time and judge the result.")
    bt.add_argument("idea", nargs="*", help='the idea, e.g. "Fama-French 3 factor model" or "12-1 momentum deciles"')
    bt.add_argument("--spec-file", metavar="JSON", help="backtest a StrategySpec JSON file instead of translating an idea")
    bt.add_argument("--start", type=_iso_date, default=None, metavar="YYYY-MM-DD", help="backtest start")
    bt.add_argument("--end", type=_iso_date, default=None, metavar="YYYY-MM-DD", help="backtest end")
    bt.add_argument("--rebalance", choices=("daily", "weekly", "monthly", "quarterly", "annual"), default=None)
    bt.add_argument("--costs-bps", type=float, default=None, metavar="BPS", help="one-way transaction cost")
    bt.add_argument("--save", metavar="NAME", default=None, help="save the strategy for simulated trading")
    bt.add_argument("--overwrite", action="store_true", help="replace a saved strategy with the same name")
    bt.add_argument("--format", choices=("text", "json"), default="text")
    bt.set_defaults(func=_cmd_backtest)

    lib = sub.add_parser("library", parents=[base], help="list the built-in idea library")
    lib.set_defaults(func=_cmd_library)

    st = sub.add_parser("strategy", parents=[base, data, out], help="saved strategies and simulated trading",
                        description="Simulated (paper) trading of saved strategies. No broker is involved.")
    st.add_argument("action", choices=("list", "run", "status", "reset", "delete"))
    st.add_argument("name", nargs="?", default=None)
    st.add_argument("--force", action="store_true", help="rebalance even if today is not a scheduled rebalance day")
    st.add_argument("--capital", type=float, default=None, help="initial capital for a new or reset account (default 100,000)")
    st.set_defaults(func=_cmd_strategy)

    disc = sub.add_parser("discover", parents=[base, engine, data, out], help="find strategy ideas in papers and on the web",
                          description="Find ideas (arXiv, your feeds, Claude web search, or a URL/PDF), verify their quotes, "
                                      "then ask idea by idea whether to backtest and stress-test them.")
    disc.add_argument("--query", action="append", default=None, help="search terms (repeatable)")
    disc.add_argument("--sources", default="arxiv,feeds,web", help="comma list of arxiv, feeds, web (default: all)")
    disc.add_argument("--url", action="append", default=None, help="read this page or PDF link (repeatable)")
    disc.add_argument("--pdf", action="append", default=None, help="read this local PDF (repeatable; needs pypdf)")
    disc.add_argument("--since", type=_iso_date, default=None, metavar="YYYY-MM-DD", help="only documents published since")
    disc.add_argument("--days", type=int, default=365, help="look back this many days when --since is not given (default 365)")
    disc.add_argument("--max", type=int, default=15, help="max documents per query / feed (default 15)")
    disc.add_argument("--review", type=int, default=10, help="max new ideas to review in this session (default 10)")
    disc.add_argument("--no-interactive", action="store_true", help="only list new ideas; do not prompt")
    disc.set_defaults(func=_cmd_discover)

    ideas = sub.add_parser("ideas", parents=[base, engine, data, out], help="the idea inbox",
                           description="List, inspect, try, reject or defer ideas found by 'aitrading discover'.")
    ideas.add_argument("action", choices=("list", "show", "try", "reject", "later"))
    ideas.add_argument("idea_id", nargs="?", default=None)
    ideas.add_argument("--status", default=None, help="filter list by status (new, tested, saved, ...)")
    ideas.add_argument("--limit", type=int, default=50)
    ideas.set_defaults(func=_cmd_ideas)

    dash = sub.add_parser("dashboard", parents=[base, out], help="write and open the index page of all reports")
    dash.set_defaults(func=_cmd_dashboard)


__all__ = ["register", "EXIT_ERROR"]
