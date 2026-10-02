"""Command-line interface: ``aitrading run | screen | spec | catalog | demo``.

Commands
--------
* ``run``     observation -> ScreenSpec -> screen -> ranked candidates -> grounded explanations.
  Prints the Markdown report (``--format md``, default) or the ``PipelineResult`` JSON to stdout and
  the run directory to stderr. The run directory (``--out``, default ``aitrading_output``) holds the
  pipeline's ``result.json`` / ``spec.json`` / ``features.csv`` / ``documents.json`` plus
  ``report.md`` and ``report.html`` (opened in the browser when run from a terminal).
* ``screen``  the same without explanations (translate, screen, rank).
* ``spec``    translate only; prints the ``ScreenSpec`` JSON.
* ``catalog`` prints the feature catalog (``--category`` filters by category or source).
* ``demo``    offline self-test on the built-in synthetic market (no internet, no keys).

Data
----
``--provider`` defaults to ``free`` (real US data from Yahoo Finance + SEC EDGAR; override the
default with ``AITRADING_PROVIDER``). ``synthetic`` is the built-in simulated market; ``bloomberg`` /
``lseg`` / ``capiq`` need the corresponding entitlements. ``--as-of`` defaults to the latest
business day (the synthetic market's last day for ``--provider synthetic``).

Engines
-------
By default Claude (``AnthropicLLM``; ``--model``, ``--effort``) translates and explains when Anthropic
credentials are configured; otherwise the CLI says so and uses the deterministic offline translator
and explainer. ``--offline`` forces the offline engine; ``--claude`` requires Claude and fails with
guidance when it cannot be used.

Providers are created by :func:`make_provider`, which imports the vendor adapter lazily and
raises :class:`ProviderUnavailableError` with an installation hint when it is missing.

Exit codes: 0 ok, 1 runtime error, 2 usage error (130 on Ctrl-C).
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import os
import sys
import traceback
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Any, Sequence

from aitrading import __version__

__all__ = [
    "main",
    "build_parser",
    "make_provider",
    "make_llm",
    "credentials_configured",
    "credentials_help",
    "is_auth_error",
    "PROVIDERS",
    "DEFAULT_AS_OF",
    "DEFAULT_PROVIDER",
    "DEMO_OBSERVATION",
    "resolve_engine",
    "default_as_of",
    "EFFORTS",
    "CLIError",
    "UsageError",
    "ProviderUnavailableError",
    "CredentialsError",
]

EXIT_OK, EXIT_ERROR, EXIT_USAGE, EXIT_INTERRUPTED = 0, 1, 2, 130
DEFAULT_AS_OF = date(2026, 9, 30)  # last day of the synthetic market
DEFAULT_OUT = "aitrading_output"
DEFAULT_PROVIDER = "free"
DEMO_OBSERVATION = (
    "Find US mid-caps ($2-20B) that were in established uptrends (50-day above 200-day, positive 12-1 momentum) "
    "but have pulled back 15-40% from their 52-week highs over the past few months on heavy volume, now oversold "
    "(RSI under 40), still generating strong free cash flow (FCF yield above 4%) with revenue growth above 8%, and "
    "where short interest is elevated (above 6% of float). Rank by FCF yield, growth and the size of the drawdown, "
    "then read the latest earnings calls and explain the dislocation."
)
EFFORTS = ("low", "medium", "high", "xhigh", "max")
CREDENTIAL_ENV = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")

# name -> (module, class, what is needed to use it)
PROVIDERS: dict[str, tuple[str, str, str]] = {
    "synthetic": ("aitrading.data.synthetic", "SyntheticProvider", "built in; no data licence needed"),
    "free": (
        "aitrading.data.free",
        "FreeDataProvider",
        'install the free extra (pip install -e ".[free]") and set SEC_USER_AGENT',
    ),
    "bloomberg": (
        "aitrading.data.bloomberg",
        "BloombergProvider",
        "run inside Bloomberg BQuant (bql) or install blpapi against the Desktop API on an entitled Terminal",
    ),
    "lseg": (
        "aitrading.data.lseg",
        "LSEGProvider",
        'install the LSEG extra (pip install -e ".[lseg]") and run with an LSEG Workspace or platform session',
    ),
    "capiq": (
        "aitrading.data.capiq",
        "CapIQProvider",
        "an S&P Capital IQ GDS API entitlement (CAPIQ_USERNAME / CAPIQ_PASSWORD) and a universe (--tickers)",
    ),
}


class CLIError(Exception):
    """An error the CLI reports as one message (no traceback)."""

    exit_code = EXIT_ERROR


class UsageError(CLIError):
    exit_code = EXIT_USAGE


class ProviderUnavailableError(CLIError):
    """The requested data provider cannot be imported or constructed."""


class CredentialsError(CLIError):
    """Claude could not be reached for lack of (valid) credentials."""


# --------------------------------------------------------------------------------------------
# Factories
# --------------------------------------------------------------------------------------------


def make_provider(name: str, **kwargs: Any) -> Any:
    """Construct the named ``MarketDataProvider``, importing its adapter module lazily."""
    key = name.strip().lower()
    if key not in PROVIDERS:
        raise UsageError(f"unknown provider '{name}'; choose from: {', '.join(PROVIDERS)}")
    module_name, class_name, needs = PROVIDERS[key]
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        missing = exc.name or "?"
        if module_name == missing or module_name.startswith(missing + "."):
            reason = f"module {module_name} is not part of this installation"
        else:
            reason = f"{module_name} needs the '{missing}' package, which is not installed"
        raise ProviderUnavailableError(f"provider '{key}' is not available: {reason}. To use it: {needs}.") from exc
    except ImportError as exc:
        raise ProviderUnavailableError(f"provider '{key}' is not available: importing {module_name} failed ({exc}). To use it: {needs}.") from exc
    cls = getattr(module, class_name, None)
    if cls is None:
        raise ProviderUnavailableError(
            f"provider '{key}' is not available: {module_name} has no class {class_name}. To use it: {needs}."
        )
    unsupported = _unsupported_kwargs(cls, kwargs)
    if unsupported:
        raise UsageError(f"provider '{key}' does not accept: {', '.join(unsupported)}")
    try:
        return cls(**kwargs)
    except Exception as exc:  # noqa: BLE001 - vendor SDK / session failures surface here
        raise ProviderUnavailableError(f"provider '{key}' could not be started ({type(exc).__name__}: {exc}). To use it: {needs}.") from exc


def _unsupported_kwargs(cls: Any, kwargs: dict[str, Any]) -> list[str]:
    try:
        params = inspect.signature(cls).parameters
    except (TypeError, ValueError):  # no introspectable signature: let the constructor decide
        return []
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return []
    return sorted(k for k in kwargs if k not in params)


def credentials_configured() -> bool:
    """True when ANTHROPIC_API_KEY or ANTHROPIC_AUTH_TOKEN is set (other SDK auth sources aside)."""
    return any(os.environ.get(k, "").strip() for k in CREDENTIAL_ENV)


def credentials_help(detail: str) -> str:
    """The message printed when Claude cannot be used; always suggests ``--offline``."""
    lines = [f"Claude is not available: {detail}"]
    if not credentials_configured():
        lines += [
            "No Anthropic credentials are configured: ANTHROPIC_API_KEY and ANTHROPIC_AUTH_TOKEN are not set.",
            "Set one (for example: export ANTHROPIC_API_KEY=sk-ant-...) to use Claude, or rerun with --offline to use",
            "the deterministic heuristic translator and explainer (no API key needed).",
        ]
    else:
        lines += [
            "Check that the API key / token is valid and has access to the model (--model or AITRADING_MODEL),",
            "or rerun with --offline to use the deterministic heuristic translator and explainer.",
        ]
    return "\n".join(lines)


def is_auth_error(exc: BaseException) -> bool:
    """Whether ``exc`` (or anything in its cause / context chain) is a missing or rejected credential."""
    try:
        import anthropic

        auth_types: tuple[type[BaseException], ...] = (anthropic.AuthenticationError, anthropic.PermissionDeniedError)
    except Exception:  # noqa: BLE001 - SDK missing or broken
        auth_types = ()
    seen: set[int] = set()
    e: BaseException | None = exc
    while e is not None and id(e) not in seen:
        seen.add(id(e))
        text = str(e).lower()
        if auth_types and isinstance(e, auth_types):
            return True
        if isinstance(e, TypeError) and "authentication method" in text:  # SDK: no credentials resolved
            return True
        if isinstance(e, CredentialsError) or "authentication/permission error" in text:
            return True
        e = e.__cause__ or e.__context__
    return False


def make_llm(model: str | None = None, effort: str = "high") -> Any:
    """``AnthropicLLM`` for the live path; construction failures become :class:`CredentialsError`."""
    try:
        from aitrading.llm.anthropic_client import AnthropicLLM

        return AnthropicLLM(model=model, effort=effort)
    except Exception as exc:  # noqa: BLE001 - SDK import / client construction
        raise CredentialsError(credentials_help(f"could not create the Anthropic client ({type(exc).__name__}: {exc})")) from exc


def resolve_engine(args: argparse.Namespace) -> str:
    """'offline' or 'claude'. Auto mode picks Claude only when credentials are configured."""
    if getattr(args, "offline", False):
        return "offline"
    if getattr(args, "claude", False) or credentials_configured():
        return "claude"
    _note(
        "No Anthropic credentials found (ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN) - running offline with the "
        "rule-based translator and explainer. Set ANTHROPIC_API_KEY to let Claude do the reasoning."
    )
    return "offline"


def _components(args: argparse.Namespace, *, explain: bool) -> tuple[Any, Any]:
    """(translator, explainer) for the chosen engine; the explainer is None when not explaining."""
    if resolve_engine(args) == "offline":
        from aitrading.agent.offline import HeuristicExplainer
        from aitrading.screen.nl import HeuristicScreenTranslator

        return HeuristicScreenTranslator(), (HeuristicExplainer() if explain else None)
    from aitrading.agent.explain import Explainer
    from aitrading.screen.nl import NLScreenTranslator

    llm = make_llm(args.model, args.effort)
    return NLScreenTranslator(llm, effort=args.effort), (Explainer(llm, effort=args.effort) if explain else None)


# --------------------------------------------------------------------------------------------
# Argument parsing
# --------------------------------------------------------------------------------------------


def _iso_date(text: str) -> date:
    try:
        return date.fromisoformat(text.strip())
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid date '{text}' (expected YYYY-MM-DD)") from None


def _top_n(text: str) -> int:
    try:
        n = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid integer '{text}'") from None
    if not 1 <= n <= 100:
        raise argparse.ArgumentTypeError(f"must be between 1 and 100, got {n}")
    return n


def _non_negative(text: str) -> int:
    try:
        n = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid integer '{text}'") from None
    if n < 0:
        raise argparse.ArgumentTypeError(f"must be >= 0, got {n}")
    return n


def build_parser() -> argparse.ArgumentParser:
    base = argparse.ArgumentParser(add_help=False)
    base.add_argument("--debug", action="store_true", help="print a traceback when a command fails")

    obs = argparse.ArgumentParser(add_help=False)
    obs.add_argument("observation", nargs="*", help="the investment observation (quote it in the shell)")
    obs.add_argument("-f", "--observation-file", metavar="PATH", help="read the observation from a UTF-8 text file ('-' = stdin)")

    engine = argparse.ArgumentParser(add_help=False)
    mode = engine.add_mutually_exclusive_group()
    mode.add_argument("--offline", action="store_true", help="force the rule-based translator + explainer (no API key needed)")
    mode.add_argument("--claude", action="store_true", help="require Claude (fail instead of falling back to offline)")
    engine.add_argument("--model", default=None, help="Claude model id (default: $AITRADING_MODEL or claude-opus-5-5)")
    engine.add_argument("--effort", choices=EFFORTS, default="high", help="Claude effort level (default: high)")

    default_provider = (os.environ.get("AITRADING_PROVIDER") or DEFAULT_PROVIDER).strip().lower()
    if default_provider not in PROVIDERS:
        default_provider = DEFAULT_PROVIDER
    data = argparse.ArgumentParser(add_help=False)
    data.add_argument("--provider", choices=list(PROVIDERS), default=default_provider,
                      help=f"data provider (default: {default_provider}; 'free' = Yahoo + SEC EDGAR, 'synthetic' = simulated market)")
    data.add_argument("--as-of", type=_iso_date, default=None, metavar="YYYY-MM-DD",
                      help="point-in-time date (default: latest business day; the synthetic market's last day for --provider synthetic)")
    tick = data.add_mutually_exclusive_group()
    tick.add_argument("--tickers", default=None, metavar="LIST",
                      help="explicit universe for providers that take one: comma-separated, or @PATH (one per line)")
    tick.add_argument("--universe-file", default=None, metavar="PATH",
                      help="explicit universe from a file (one ticker per line, '#' comments)")

    output = argparse.ArgumentParser(add_help=False)
    output.add_argument("--top", type=_top_n, default=None, metavar="N", help="number of ranked candidates (1-100; default: the spec's top_n)")
    output.add_argument("--out", default=DEFAULT_OUT, metavar="DIR", help=f"directory for run artifacts (default: {DEFAULT_OUT})")
    output.add_argument("--no-save", action="store_true", help="do not write run artifacts")
    output.add_argument("--format", choices=("md", "json"), default="md", help="stdout format (default: md)")
    browser = output.add_mutually_exclusive_group()
    browser.add_argument("--open", dest="open_browser", action="store_true", default=None,
                         help="open the HTML report in the browser (default when run from a terminal)")
    browser.add_argument("--no-open", dest="open_browser", action="store_false", help="do not open the HTML report")

    parser = argparse.ArgumentParser(
        prog="aitrading",
        description="Equity research pipeline: observation -> screen -> ranked candidates -> grounded explanations.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    run = sub.add_parser("run", parents=[base, obs, engine, data, output], help="full pipeline with explanations",
                         description="Translate, screen, rank and explain; print the report.")
    run.add_argument("--explain", type=_non_negative, default=5, metavar="K", help="explain the top K candidates (default: 5)")
    run.set_defaults(func=_cmd_run)

    screen = sub.add_parser("screen", parents=[base, obs, engine, data, output], help="translate + screen + rank (no explanations)",
                            description="Translate, screen and rank; no explanations.")
    screen.set_defaults(func=_cmd_screen)

    spec = sub.add_parser("spec", parents=[base, obs, engine], help="translate the observation and print the ScreenSpec JSON",
                          description="Translate the observation into a ScreenSpec and print it as JSON.")
    spec.add_argument("--top", type=_top_n, default=None, metavar="N", help="override top_n (1-100)")
    spec.set_defaults(func=_cmd_spec)

    catalog = sub.add_parser("catalog", parents=[base], help="print the feature catalog",
                             description="Print the feature catalog (names, sources, units, definitions).")
    catalog.add_argument("--category", default=None, help="only this category (e.g. trend, valuation) or source (technical, ...)")
    catalog.add_argument("--format", choices=("md", "json"), default="md", help="output format (default: md)")
    catalog.set_defaults(func=_cmd_catalog)

    demo = sub.add_parser("demo", parents=[base, output], help="offline self-test on the built-in synthetic market",
                          description="Run the representative task on the built-in synthetic market with the offline engine "
                                      "(no internet, no keys) and score the explanations against the market's planted ground truth.")
    demo.set_defaults(func=_cmd_demo, out=str(Path(DEFAULT_OUT) / "demo"), explain=10, top=None)
    return parser


# --------------------------------------------------------------------------------------------
# I/O helpers
# --------------------------------------------------------------------------------------------


def _emit(text: str) -> None:
    """Write to stdout; characters the console cannot encode are replaced, a closed pipe is ignored."""
    try:
        sys.stdout.write(text)
        sys.stdout.flush()
    except UnicodeEncodeError:
        buffer = getattr(sys.stdout, "buffer", None)
        if buffer is None:
            raise
        buffer.write(text.encode(sys.stdout.encoding or "utf-8", errors="replace"))
        buffer.flush()
    except BrokenPipeError:  # e.g. `aitrading run ... | head`
        try:
            sys.stdout = open(os.devnull, "w")  # noqa: SIM115 - silences the flush at interpreter exit
        except OSError:
            pass


def _note(text: str) -> None:
    print(text, file=sys.stderr)


def _read_observation(args: argparse.Namespace) -> str:
    words = [w for w in (args.observation or []) if w is not None]
    path = args.observation_file
    if words and path:
        raise UsageError("give the observation either as text or with --observation-file, not both")
    if path:
        if path == "-":
            text = sys.stdin.read()
        else:
            try:
                text = Path(path).read_text(encoding="utf-8-sig")
            except (OSError, UnicodeDecodeError) as exc:
                raise UsageError(f"cannot read observation file '{path}': {exc}") from exc
    else:
        text = " ".join(words)
    text = text.strip()
    if not text:
        raise UsageError("an observation is required (as text or with --observation-file)")
    return text


def _read_tickers(spec: str | None) -> list[str] | None:
    """``--tickers``: 'A,B C' or '@path' (one ticker per line, '#' comments); None when not given."""
    if spec is None:
        return None
    text = spec
    if spec.startswith("@"):
        try:
            text = Path(spec[1:]).read_text(encoding="utf-8-sig")
        except (OSError, UnicodeDecodeError) as exc:
            raise UsageError(f"cannot read tickers file '{spec[1:]}': {exc}") from exc
        text = "\n".join(line.split("#", 1)[0] for line in text.splitlines())
    tickers = list(dict.fromkeys(t.strip().upper() for t in text.replace(",", " ").split() if t.strip()))
    if not tickers:
        raise UsageError("--tickers is empty")
    return tickers


def _describe(exc: BaseException) -> str:
    errors = getattr(exc, "errors", None)
    if isinstance(errors, list) and errors and all(isinstance(e, str) for e in errors):
        head = str(exc).split(":", 1)[0] if ":" in str(exc) else type(exc).__name__
        return head + ":\n" + "\n".join(f"  - {e}" for e in errors)
    from aitrading.data.base import ProviderError

    if isinstance(exc, ProviderError):
        return f"data provider error: {exc}\n(--provider synthetic runs on the built-in synthetic market without any data licence)"
    return f"{type(exc).__name__}: {exc}"


# --------------------------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------------------------


def default_as_of(provider_name: str, provider: Any = None, today: date | None = None) -> date:
    """Latest business day on or before today; the synthetic market's last day for 'synthetic'."""
    if provider_name == "synthetic":
        end = getattr(provider, "end", None)
        return end if isinstance(end, date) else DEFAULT_AS_OF
    d = today or date.today()
    while d.weekday() >= 5:
        d = date.fromordinal(d.toordinal() - 1)
    return d


def _provider_from_args(args: argparse.Namespace) -> Any:
    spec = getattr(args, "tickers", None)
    if getattr(args, "universe_file", None):
        spec = "@" + args.universe_file
    tickers = _read_tickers(spec)
    provider = make_provider(args.provider, **({"tickers": tickers} if tickers else {}))
    diagnostics = getattr(provider, "diagnostics", None)
    if callable(diagnostics):
        for line in diagnostics() or []:
            _note(f"setup: {line}")
    return provider


def _should_open(args: argparse.Namespace) -> bool:
    if getattr(args, "open_browser", None) is not None:
        return bool(args.open_browser)
    if os.environ.get("AITRADING_NO_BROWSER"):
        return False
    try:
        return sys.stdout.isatty()
    except (AttributeError, ValueError):
        return False


def _write_html(run_dir: Path, html: str, args: argparse.Namespace) -> Path:
    path = run_dir / "report.html"
    path.write_text(html, encoding="utf-8")
    if _should_open(args):
        from aitrading.report.dashboard import open_in_browser

        open_in_browser(path)
    return path


def _pipeline(args: argparse.Namespace, *, explain: bool) -> int:
    from aitrading.pipeline import ResearchPipeline
    from aitrading.report.html import render_pipeline_html
    from aitrading.report.markdown import render_markdown

    observation = _read_observation(args)
    provider = _provider_from_args(args)
    translator, explainer = _components(args, explain=explain)
    as_of = args.as_of or default_as_of(args.provider, provider)
    out_dir = None if args.no_save else Path(args.out)
    pipe = ResearchPipeline(
        provider,
        translator,
        explainer,
        out_dir=out_dir,
        explain_top_k=args.explain if explain else 0,
    )
    result = pipe.run(observation, as_of, top_n=args.top)
    report = render_markdown(result)
    run_dir = out_dir / result.run_id if out_dir is not None else None
    if run_dir is not None:
        (run_dir / "report.md").write_text(report, encoding="utf-8")
        _write_html(run_dir, render_pipeline_html(result), args)
    _emit(report if args.format == "md" else result.model_dump_json(indent=2) + "\n")
    explained = sum(i.thesis is not None for i in result.ideas)
    failed = sum(bool(i.error) for i in result.ideas)
    _note(
        f"{result.survivors} of {result.universe_size} names passed the screen; {len(result.ideas)} ranked, "
        f"{explained} explained" + (f", {failed} failed" if failed else "") + f"; {len(result.llm_calls)} LLM call(s)"
    )
    _note(f"run directory: {run_dir}" if run_dir is not None else "run directory: none (--no-save)")
    return EXIT_OK


def _cmd_run(args: argparse.Namespace) -> int:
    return _pipeline(args, explain=True)


def _cmd_screen(args: argparse.Namespace) -> int:
    return _pipeline(args, explain=False)


def _cmd_spec(args: argparse.Namespace) -> int:
    observation = _read_observation(args)
    translator, _ = _components(args, explain=False)
    out = translator.translate(observation)
    spec = getattr(out, "spec", out)
    if args.top is not None:
        spec = spec.model_copy(update={"top_n": args.top})
    _emit(spec.model_dump_json(indent=2) + "\n")
    attempts = getattr(out, "attempts", 1)
    _note(f"translator: {getattr(out, 'translator', type(translator).__name__)}; attempts: {attempts}")
    return EXIT_OK


def _cmd_demo(args: argparse.Namespace) -> int:
    """Representative task on the synthetic market, offline, scored against the planted ground truth."""
    from aitrading.agent.offline import HeuristicExplainer
    from aitrading.pipeline import ResearchPipeline
    from aitrading.report.html import render_pipeline_html
    from aitrading.report.markdown import render_markdown
    from aitrading.screen.nl import HeuristicScreenTranslator

    provider = make_provider("synthetic")
    out_dir = None if args.no_save else Path(args.out)
    pipe = ResearchPipeline(provider, HeuristicScreenTranslator(), HeuristicExplainer(), out_dir=out_dir, explain_top_k=args.explain)
    result = pipe.run(DEMO_OBSERVATION, default_as_of("synthetic", provider), top_n=args.top or 10)
    report = render_markdown(result)
    run_dir = out_dir / result.run_id if out_dir is not None else None
    if run_dir is not None:
        (run_dir / "report.md").write_text(report, encoding="utf-8")
        _write_html(run_dir, render_pipeline_html(result), args)
    if args.format == "json":
        _emit(result.model_dump_json(indent=2) + "\n")

    truth = provider.archetypes() if hasattr(provider, "archetypes") else {}
    expected = {
        "transitory_shock": "transitory_fundamental_shock",
        "value_trap": "structural_decline_value_trap",
        "guidance_reset": "guidance_reset_overreaction",
        "sector_contagion": "sector_or_macro_contagion",
    }
    scored = [(i.candidate.ticker, truth.get(i.candidate.ticker), i.thesis) for i in result.ideas if i.thesis is not None]
    correct = sum(expected.get(t) == th.dislocation_type for _, t, th in scored)
    lines = [
        "AItrading demo (synthetic market, offline engine)",
        f"  {result.survivors} of {result.universe_size} names passed the screen; {len(scored)} explained.",
        f"  Dislocation type matched the planted ground truth for {correct}/{len(scored)} explained names.",
        "  (The synthetic market is a self-test of the pipeline, not evidence of real-world skill.)",
    ]
    for ticker, t, th in scored:
        mark = "ok " if expected.get(t) == th.dislocation_type else "MISS"
        lines.append(f"  [{mark}] {ticker:6s} planted={t or '?':17s} explained={th.dislocation_type}")
    if run_dir is not None:
        lines.append(f"  Report: {run_dir / 'report.html'}")
    print("\n".join(lines), file=sys.stderr if args.format == "json" else sys.stdout)
    return EXIT_OK


def _catalog_markdown(features: list[Any], title: str) -> str:
    from aitrading.report.markdown import escape_cell

    lines = [f"# {title} ({len(features)} features)"]
    categories = list(dict.fromkeys(f.category for f in features))
    for cat in categories:
        group = [f for f in features if f.category == cat]
        lines += ["", f"## {cat} ({len(group)})", ""]
        lines.append("| Feature | Source | Type | Unit | Ranking hint | Definition |")
        lines.append("|---|---|---|---|---|---|")
        for f in group:
            hint = {True: "higher is better", False: "lower is better"}.get(f.higher_is_better, "")
            cells = [f"`{f.name}`", f.source, f.dtype, f.unit, hint, f.description]
            lines.append("| " + " | ".join(escape_cell(c) for c in cells) + " |")
    return "\n".join(lines) + "\n"


def _cmd_catalog(args: argparse.Namespace) -> int:
    from aitrading.screen.catalog import default_catalog

    catalog = default_catalog()
    features = list(catalog)
    title = "Feature catalog"
    if args.category:
        key = args.category.strip().lower()
        features = [f for f in features if f.category.lower() == key or f.source.lower() == key]
        if not features:
            categories = sorted({f.category for f in catalog})
            sources = sorted({f.source for f in catalog})
            raise UsageError(
                f"unknown category '{args.category}'; categories: {', '.join(categories)}; sources: {', '.join(sources)}"
            )
        title = f"Feature catalog: {key}"
    if args.format == "json":
        _emit(json.dumps([asdict(f) for f in features], indent=2) + "\n")
    else:
        _emit(_catalog_markdown(features, title))
    return EXIT_OK


# --------------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------------


def _exit_code(code: Any) -> int:
    if code is None:
        return EXIT_OK
    if isinstance(code, int):
        return code
    print(code, file=sys.stderr)
    return EXIT_ERROR


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI; returns the exit code (0 ok, 1 runtime error, 2 usage error)."""
    parser = build_parser()
    try:
        args = parser.parse_args(list(argv) if argv is not None else None)
    except SystemExit as exc:  # argparse: --help / --version -> 0, bad arguments -> 2
        return _exit_code(exc.code)
    if not getattr(args, "command", None):
        parser.print_usage(sys.stderr)
        _note("aitrading: error: a command is required (run, screen, spec, catalog or demo)")
        return EXIT_USAGE
    prog = f"aitrading {args.command}"
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        _note(f"{prog}: interrupted")
        return EXIT_INTERRUPTED
    except Exception as exc:  # noqa: BLE001 - every failure becomes a message and an exit code
        if getattr(args, "debug", False):
            traceback.print_exc()
        if isinstance(exc, CLIError):
            _note(f"{prog}: error: {exc}")
            return exc.exit_code
        if is_auth_error(exc):
            _note(f"{prog}: error: " + credentials_help(f"the Claude API call failed with an authentication error ({type(exc).__name__}: {exc})"))
            return EXIT_ERROR
        _note(f"{prog}: error: {_describe(exc)}")
        return EXIT_ERROR


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
