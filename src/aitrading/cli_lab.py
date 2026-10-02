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
import json
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

from aitrading.cli import (
    EXIT_ERROR,
    EXIT_OK,
    PROVIDERS,
    CLIError,
    UsageError,
    _emit,
    _iso_date,
    _note,
    _provider_from_args,
    _should_open,
    data_options,
    default_as_of,
    default_provider_name,
    explicit_tickers,
    make_llm,
    read_text_file,
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
        from aitrading.report.prose import interpretation_unverified_numbers

        unverified = interpretation_unverified_numbers(result)
        if unverified:
            lines.append(f"UNVERIFIED: the verdict text states {', '.join(unverified)}, which are not in the backtest results;"
                         " rely on the statistics above.")
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
            spec = StrategySpec.model_validate_json(read_text_file(args.spec_file))
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

        store = _store()
        path = store.save(
            args.save, out.spec, result, overwrite=args.overwrite, idea=idea or out.spec.idea,
            template=out.spec.name if out.spec.name in TEMPLATES else None,
        )
        _record_backtest_data(store, args.save, args.provider, explicit_tickers(args))
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


# The data a saved strategy was backtested on, and the data its paper account trades on, kept next to
# the strategy (``<strategy dir>/run_settings.json``) so `strategy run` uses the same market:
# {"backtest": {"provider", "tickers"}, "paper": {"provider", "tickers", "started"}}.
RUN_SETTINGS_FILE = "run_settings.json"


def _load_run_settings(store: Any, name: str) -> dict[str, Any]:
    try:
        data = json.loads((store.path(name) / RUN_SETTINGS_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _save_run_settings(store: Any, name: str, settings: dict[str, Any]) -> None:
    from aitrading.trading.store import write_json_atomic

    write_json_atomic(store.path(name) / RUN_SETTINGS_FILE, settings)


def _record_backtest_data(store: Any, name: str, provider: str | None, tickers: list[str] | None) -> None:
    settings = _load_run_settings(store, name)
    settings["backtest"] = {"provider": provider, "tickers": tickers}
    _save_run_settings(store, name, settings)


def _data_label(provider: str | None, tickers: list[str] | None) -> str:
    if not tickers:
        return f"provider '{provider}' (its default universe)"
    shown = ",".join(tickers[:8]) + (f",... ({len(tickers)} tickers)" if len(tickers) > 8 else "")
    return f"provider '{provider}' with --tickers {shown}"


def _same_universe(a: list[str] | None, b: list[str] | None) -> bool:
    return sorted(a or []) == sorted(b or [])


def _strategy_run_data(store: Any, name: str, meta: dict[str, Any], args: argparse.Namespace, started: date | None) -> tuple[str, list[str] | None, bool]:
    """(provider, explicit tickers or None, whether the account's recorded data applies) for `strategy run`.

    Without --provider / --tickers the run uses the data the paper account has been trading on, else
    the data the strategy was backtested on. Once the account has started, a different provider or
    universe is refused: one ledger must not mix markets.
    """
    settings = _load_run_settings(store, name)
    paper = settings.get("paper") if isinstance(settings.get("paper"), dict) else None
    if paper is not None and (started is None or paper.get("started") != started.isoformat()):
        paper = None  # the account was reset since
    backtest = settings.get("backtest") if isinstance(settings.get("backtest"), dict) else {}
    saved = paper or backtest
    saved_provider = saved.get("provider") or (meta.get("backtest") or {}).get("provider")
    if saved_provider not in PROVIDERS:
        saved_provider = None
    requested_tickers = explicit_tickers(args)
    provider = args.provider or saved_provider or default_provider_name()
    if requested_tickers is not None:
        tickers = requested_tickers
    else:
        tickers = (saved.get("tickers") or None) if provider == saved_provider else None
    if paper is not None:
        if provider != paper.get("provider") or not _same_universe(tickers, paper.get("tickers")):
            raise CLIError(
                f"strategy '{name}' has been paper-trading on {_data_label(paper.get('provider'), paper.get('tickers'))} "
                f"since {paper.get('started')}; this run would use {_data_label(provider, tickers)}, which would mix "
                f"markets in one ledger. Run it without --provider / --tickers, or start the account over with: "
                f"aitrading strategy reset {name}"
            )
    elif saved_provider and (provider != saved_provider or (requested_tickers is not None and not _same_universe(tickers, saved.get("tickers")))):
        _note(f"warning: '{name}' was backtested on {_data_label(saved_provider, saved.get('tickers'))}; simulating it on "
              f"{_data_label(provider, tickers)} means 'strategy status' compares it with a backtest of other data")
    if args.provider is None and saved_provider and provider != default_provider_name():
        _note(f"data: {_data_label(provider, tickers)}, as {'paper-traded' if paper else 'backtested'} so far")
    if started is not None and paper is None:
        _note(f"note: the data the paper account of '{name}' started on was not recorded (older version); "
              f"it is recorded as {_data_label(provider, tickers)} from this run on")
    return provider, tickers, paper is not None


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
        spec, backtest, meta = store.load(name)
        account = PaperAccount(store, name, initial_capital=args.capital) if args.capital else PaperAccount(store, name)
        started = _account_started(account)
        provider_name, tickers, recorded = _strategy_run_data(store, name, meta, args, started)
        run_args = argparse.Namespace(**{**vars(args), "provider": provider_name, "universe_file": None,
                                         "tickers": ",".join(tickers) if tickers else None})
        provider = _lab_provider(run_args)
        as_of = args.as_of or default_as_of(provider_name, provider)
        report = account.rebalance(_runner(provider), spec, as_of, force=args.force)
        now_started = _account_started(account)
        if not recorded and now_started is not None:
            settings = _load_run_settings(store, name)
            settings["paper"] = {"provider": provider_name, "tickers": tickers, "started": now_started.isoformat()}
            _save_run_settings(store, name, settings)
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
    page = _write_strategy_page(store, name, args)
    if action == "status":
        account = PaperAccount(store, name)
        try:
            s = account.summary(backtest=store.load_backtest(name))
            _emit(
                f"{name}: NAV {s.get('nav', 0):,.2f}, since start {_fmt(s.get('since_start_return_pct'), '%')}"
                f" (backtest expected {_fmt(s.get('backtest_expected_return_pct'), '%')}), last run {s.get('last_run')}\n"
            )
            for n in s.get("backtest_comparison_notes") or []:
                _emit(f"  note: {n}\n")
        except Exception as exc:  # noqa: BLE001
            _emit(f"{name}: not started yet ({exc}). Start it with: aitrading strategy run {name}\n")
    if page is not None:
        _note(f"strategy page: {page}")
        _open(page, args)
    return EXIT_OK


def _account_started(account: Any) -> date | None:
    try:
        return account.started
    except Exception:  # noqa: BLE001 - no ledger yet, or unreadable: rebalance reports the latter
        return None


def _write_strategy_page(store: Any, name: str, args: argparse.Namespace) -> Path | None:
    """The strategy page under --out; a failure to write it is a warning, not an error, because the
    simulated trades are already recorded (e.g. Task Scheduler starts in C:\\Windows\\System32,
    where the default relative ./aitrading_output cannot be created)."""
    out = _out_dir(args)
    try:
        return _strategy_page(store, name, out)
    except OSError as exc:
        if out.is_absolute() or getattr(args, "out", None) not in (None, "aitrading_output"):
            _note(f"warning: could not write the strategy page under {out} ({exc}); the ledger is up to date")
            return None
        fallback = Path.home() / "aitrading_output"
        try:
            page = _strategy_page(store, name, fallback)
        except OSError as exc2:
            _note(f"warning: could not write the strategy page ({exc}; {exc2}); the ledger is up to date")
            return None
        _note(f"warning: cannot write to {Path.cwd() / out} ({type(exc).__name__}); wrote the page under {fallback} instead "
              "(pass --out FOLDER, or set the scheduled task's 'Start in' folder)")
        return page


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


def _sources_config() -> Any:
    from aitrading.discovery.sources import load_sources_config

    try:
        return load_sources_config()
    except (OSError, ValueError) as exc:
        raise UsageError(f"cannot use the sources config (~/.aitrading/sources.json or $AITRADING_SOURCES): {exc}") from exc


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
    cfg = _sources_config()
    web = None
    if "web" in sources and llm is not None:
        # built first so a domain-list mistake in sources.json stops the run before anything is fetched
        from aitrading.discovery.websearch import ClaudeWebSearchSource

        try:
            web = ClaudeWebSearchSource(model=getattr(llm, "model", None) or "claude-opus-5-5", effort=args.effort,
                                        allowed_domains=cfg.web_allowed_domains, blocked_domains=cfg.web_blocked_domains)
        except ValueError as exc:
            raise UsageError(f"sources config: {exc}") from exc
    if "arxiv" in sources:
        try:
            arxiv = ArxivSource(categories=cfg.arxiv_categories) if cfg.arxiv_categories else ArxivSource()
        except ValueError as exc:
            raise UsageError(f"sources config arxiv_categories: {exc}") from exc
        queries = args.query or cfg.arxiv_queries or None
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
        if web is None:
            _note("web search needs Claude (set ANTHROPIC_API_KEY); skipping the 'web' source")
        else:
            brief = " ".join(args.query) if args.query else "recent systematic US equity strategies and anomalies"
            if web.allowed_domains or web.blocked_domains:
                which = "only " + ", ".join(web.allowed_domains) if web.allowed_domains else "not " + ", ".join(web.blocked_domains)
                _note(f"web: searching {which} (sources.json)")
            try:
                docs += web.discover(brief, max_ideas=args.max)
            except Exception as exc:  # noqa: BLE001 - e.g. web search not enabled, rate limit, refusal: keep the other sources
                _note(f"web: skipped ({type(exc).__name__}: {exc}); continuing with the other sources")
            for w in web.warnings:
                _note(f"web: {w}")
    return docs


# Documents Claude read in an earlier run that were not trading ideas (they never reach the inbox), so a
# daily `discover` does not pay to read them again: {"documents": {doc_key: {"url", "title", "seen_at"}}}.
SEEN_FILE = "seen_documents.json"
MAX_SEEN = 5000


def _seen_path(inbox: Any) -> Path:
    from aitrading.discovery.inbox import default_inbox_path

    base = getattr(inbox, "path", None)
    return Path(base if base is not None else default_inbox_path()).with_name(SEEN_FILE)


def _load_seen(inbox: Any) -> dict[str, Any]:
    try:
        data = json.loads(_seen_path(inbox).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    docs = data.get("documents") if isinstance(data, dict) else None
    return docs if isinstance(docs, dict) else {}


def _save_seen(inbox: Any, seen: dict[str, Any]) -> None:
    from aitrading.trading.store import write_json_atomic

    items = sorted(seen.items(), key=lambda kv: str((kv[1] or {}).get("seen_at") or ""))[-MAX_SEEN:]
    try:
        write_json_atomic(_seen_path(inbox), {"schema_version": 1, "documents": dict(items)})
    except OSError as exc:  # a cache: never fail the run over it
        _note(f"could not update {_seen_path(inbox)}: {exc}")


def _known_documents(inbox: Any) -> tuple[dict[str, str], dict[str, str]]:
    """(doc_key -> idea id, canonical URL -> idea id) of every idea in the inbox and every pruned one."""
    from aitrading.discovery.rank import canonical_url

    keys: dict[str, str] = {}
    urls: dict[str, str] = {}
    for c in (inbox.all() if hasattr(inbox, "all") else []):
        keys[c.idea_id] = c.idea_id
        u = canonical_url(c.source.url)
        if u:
            urls.setdefault(u, c.idea_id)
    try:  # pruned ideas leave tombstones; inbox.add would drop them again after paying for the extraction
        tombstones = inbox._load().get("pruned") or []
    except Exception:  # noqa: BLE001 - a fake / older inbox: in-inbox ideas are still skipped
        tombstones = []
    for t in tombstones:
        if isinstance(t, dict) and t.get("idea_id"):
            keys.setdefault(str(t["idea_id"]), f"{t['idea_id']} (pruned)")
            u = canonical_url(str(t.get("url") or ""))
            if u:
                urls.setdefault(u, f"{t['idea_id']} (pruned)")
    return keys, urls


def _new_documents(docs: list[Any], inbox: Any, seen: dict[str, Any], *, explicit: bool) -> list[Any]:
    """The documents not read before: not in the inbox (by key or canonical URL), not pruned from it
    and not already judged 'not a trading idea' by Claude. Notes what was skipped."""
    from aitrading.discovery.rank import canonical_url

    keys, urls = _known_documents(inbox)
    fresh, skipped = [], 0
    for d in docs:
        hit = keys.get(d.doc_key) or urls.get(canonical_url(d.url) or "")
        if hit is None and d.doc_key in seen:
            hit = "not a trading idea"
        if hit is None:
            fresh.append(d)
            continue
        skipped += 1
        if explicit:
            _note(f"already read: {d.url} -> {hit} (see: aitrading ideas show ID; --reextract reads it again)")
    if skipped and not explicit:
        _note(f"{skipped} document(s) were read in an earlier run and are skipped (--reextract reads them again)")
    return fresh


def _save_idea(c: Any, spec: Any, result: Any, args: argparse.Namespace) -> str:
    """Save an accepted idea as a strategy without replacing one that exists: the spec's name (often
    a library template key), else that name plus the idea id. Returns the name used."""
    from aitrading.trading.store import StrategyExistsError

    store = _store()
    base = (spec.name or c.idea_id)[:40]
    names = [base, f"{base[:27]}-{c.idea_id}"] + [f"{base[:24]}-{c.idea_id}-{i}" for i in range(2, 100)]
    for name in names:
        try:
            store.save(name, spec, result, overwrite=False, idea=c.extraction.title, notes=f"from idea {c.idea_id}: {c.source.url}")
        except StrategyExistsError:
            continue
        _record_backtest_data(store, name, getattr(args, "provider", None), explicit_tickers(args))
        if name != base:
            _emit(f"  A strategy named '{base}' already exists; this idea is saved as '{name}' instead.\n")
        return name
    raise CLIError(f"could not find a free strategy name for idea {c.idea_id}; delete old ones with: aitrading strategy delete NAME")


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
        _note(f"  Testing: {spec.name} ({spec.kind}, {spec.rebalance}) ...")
        result, replication, html_path = lab.replicate_idea(c, spec)
    except Exception as exc:  # noqa: BLE001 - one failed idea must not stop the session
        inbox.set_status(c.idea_id, "failed", note=f"{type(exc).__name__}: {exc}")
        _note(f"  could not test this idea: {type(exc).__name__}: {exc}")
        return
    inbox.attach_result(c.idea_id, result.run_id, replication)
    lines = ["  " + line for line in _summary_lines(result)]
    lines.append(f"  Replication: {replication.verdict.replace('_', ' ').upper()} - {replication.summary}")
    for chk in replication.checks:
        mark = {True: "pass", False: "FAIL", None: "n/a "}[chk.passed]
        lines.append(f"    [{mark}] {chk.name}: Sharpe {_fmt(chk.sharpe)}  {chk.note}")
    if html_path is not None:
        lines.append(f"  Report: {html_path}")
    _emit("\n".join(lines) + "\n")
    if _ask("  Save it as a strategy for simulated trading? [y/N] ", "yn", "n", ask) == "y":
        name = _save_idea(c, spec, result, args)
        inbox.set_status(c.idea_id, "saved")
        _emit(f"  Saved as '{name}'. Simulate it with: aitrading strategy run {name}\n")


def _review_loop(cands: list[Any], args: argparse.Namespace, *, lab: Any, inbox: Any, ask: Prompt) -> None:
    for c in cands:
        _emit(_idea_card(c) + "\n")
        if c.extraction.testability == "not_testable":
            _emit("  (not testable with the data this platform has; kept in the inbox for reference)\n")
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
    inbox = _inbox()
    seen = _load_seen(inbox)
    todo = docs if args.reextract else _new_documents(docs, inbox, seen, explicit=bool(args.url or args.pdf))
    extractor = IdeaExtractor(llm, effort=args.effort) if llm is not None else HeuristicIdeaExtractor()
    candidates = []
    seen_changed = False
    for d in todo:
        try:
            c = extractor.extract(d)
        except Exception as exc:  # noqa: BLE001
            _note(f"skipped {d.url}: {type(exc).__name__}: {exc}")
            continue
        if c.extraction.is_trading_idea:
            candidates.append(c)
            seen_changed |= seen.pop(d.doc_key, None) is not None
        elif llm is not None:  # only Claude's verdict is remembered: the offline extractor costs nothing to rerun
            seen[d.doc_key] = {"url": d.url, "title": d.title[:200], "seen_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
            seen_changed = True
    if seen_changed:
        _save_seen(inbox, seen)
    added = inbox.add(candidates)
    _note(f"{len(docs)} document(s) found, {len(todo)} read, {len(candidates)} trading idea(s), {len(added)} new in the inbox")
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
    _emit(_idea_card(c) + "\n")
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

    st = sub.add_parser("strategy", parents=[base, data_options(None), out], help="saved strategies and simulated trading",
                        description="Simulated (paper) trading of saved strategies. No broker is involved. 'run' uses the "
                                    "data (provider and universe) the strategy was backtested and is paper-trading on, "
                                    "unless --provider / --tickers say otherwise.")
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
    disc.add_argument("--reextract", action="store_true",
                      help="read documents an earlier run already read again (by default they are skipped, which saves Claude calls)")
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
