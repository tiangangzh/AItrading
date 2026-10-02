"""Idea-lab CLI commands end to end, offline: backtest, library, strategy (simulated trading),
discover (interactive 'Try this strategy?' loop with a fake paper), ideas, dashboard."""

from __future__ import annotations

import json
from datetime import date, datetime, timezone

import pytest

from aitrading import cli, cli_lab
from aitrading.data.synthetic import SyntheticProvider
from aitrading.discovery.models import SourceDocument

MOMENTUM_ABSTRACT = (
    "We document that stocks with high returns over the past twelve months, skipping the most recent month, "
    "continue to outperform stocks with low past returns over the following months. A long-short portfolio that buys "
    "the top decile of 12-1 momentum and sells the bottom decile earns 1.2% per month from 1965 to 2019, with a "
    "t-statistic of 4.1. The effect is robust across size groups and survives transaction costs."
)


@pytest.fixture(scope="module")
def lab_provider():
    return SyntheticProvider(n_tickers=120, start=date(2015, 1, 2))


@pytest.fixture
def env(tmp_path, monkeypatch, lab_provider):
    monkeypatch.setenv("AITRADING_HOME", str(tmp_path / "home"))
    for k in cli.CREDENTIAL_ENV:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(cli_lab, "_lab_provider", lambda args: lab_provider)
    monkeypatch.setattr(cli_lab, "_runner", lambda provider: __import__("aitrading.backtest.runner", fromlist=["x"]).StrategyRunner(provider, factor_loader=None))
    return tmp_path


def run(capsys, *argv):
    code = cli.main(list(argv))
    out = capsys.readouterr()
    return code, out.out, out.err


def test_library_lists_templates(capsys, env):
    code, out, _ = run(capsys, "library")
    assert code == 0 and "ff3" in out and "momentum_12_1" in out and "Jegadeesh" in out


def test_backtest_save_and_simulated_trading(capsys, env):
    out_dir = env / "out"
    code, out, err = run(capsys, "backtest", "--offline", "--no-open", "--out", str(out_dir), "12-1 momentum deciles", "--save", "mom")
    assert code == 0, err
    assert "Strategy: CAGR" in out and "Verdict:" in out and "saved strategy 'mom'" in err
    reports = list((out_dir / "backtests").glob("*/report.html"))
    assert len(reports) == 1

    code, out, _ = run(capsys, "strategy", "list", "--out", str(out_dir))
    assert code == 0 and "| mom |" in out

    code, out, err = run(capsys, "strategy", "run", "mom", "--out", str(out_dir), "--no-open")
    assert code == 0 and "simulated order(s)" in out and "strategy page" in err
    code, out, _ = run(capsys, "strategy", "run", "mom", "--out", str(out_dir), "--no-open")
    assert code == 0 and "no trades" in out  # idempotent on the same day

    code, out, _ = run(capsys, "strategy", "status", "mom", "--out", str(out_dir), "--no-open")
    assert code == 0 and "NAV" in out
    assert (out_dir / "strategies" / "mom.html").exists()

    code, out, _ = run(capsys, "dashboard", "--out", str(out_dir), "--no-open")
    assert code == 0 and (out_dir / "index.html").exists()


def test_backtest_requires_an_idea(capsys, env):
    code, _, err = run(capsys, "backtest", "--offline")
    assert code == 2 and "give an idea" in err


def _fake_sources(monkeypatch):
    from aitrading.discovery import sources

    doc = SourceDocument(
        source_type="arxiv", url="https://arxiv.org/abs/2601.00001", title="Twelve-month momentum revisited",
        authors=["A. Researcher"], published=date(2020, 1, 15), text=MOMENTUM_ABSTRACT,
        fetched_at=datetime(2026, 10, 1, tzinfo=timezone.utc), source_name="arXiv q-fin.PM",
    )

    class FakeArxiv:
        warnings: list[str] = []

        def search_many(self, *a, **k):
            return [doc]

    class FakeFeeds:
        warnings: list[str] = []

        def fetch(self, *a, **k):
            return []

    monkeypatch.setattr(sources, "ArxivSource", FakeArxiv)
    monkeypatch.setattr(sources, "FeedSource", FakeFeeds)
    return doc


def test_discover_interactive_try_and_save(capsys, env, monkeypatch):
    _fake_sources(monkeypatch)
    answers = iter(["y", "y"])  # try it; save it
    args = cli.build_parser().parse_args(["discover", "--offline", "--sources", "arxiv,feeds", "--out", str(env / "out"), "--no-open"])
    code = cli_lab._cmd_discover(args, ask=lambda prompt: next(answers))
    out = capsys.readouterr()
    assert code == 0
    assert "Twelve-month momentum revisited" in out.out and "Replication:" in out.out and "Saved as" in out.out
    from aitrading.discovery.inbox import IdeaInbox

    (idea,) = IdeaInbox().list()
    assert idea.status == "saved" and idea.backtest_run_id and idea.replication is not None

    code, o, _ = run(capsys, "ideas", "list")
    assert code == 0 and idea.idea_id in o and "saved" in o
    code, o, _ = run(capsys, "ideas", "show", idea.idea_id)
    assert code == 0 and "Evidence quotes verified" in o


def test_discover_reject_and_later(capsys, env, monkeypatch):
    _fake_sources(monkeypatch)
    args = cli.build_parser().parse_args(["discover", "--offline", "--sources", "arxiv", "--out", str(env / "out"), "--no-open"])
    assert cli_lab._cmd_discover(args, ask=lambda p: "l") == 0
    from aitrading.discovery.inbox import IdeaInbox

    (idea,) = IdeaInbox().list()
    assert idea.status == "deferred"
    code, out, _ = run(capsys, "ideas", "reject", idea.idea_id)
    assert code == 0 and IdeaInbox().get(idea.idea_id).status == "rejected"


def test_discover_non_interactive_lists_only(capsys, env, monkeypatch):
    _fake_sources(monkeypatch)
    code, out, _ = run(capsys, "discover", "--offline", "--sources", "arxiv", "--no-interactive", "--out", str(env / "out"))
    assert code == 0 and "aitrading ideas try" in out


def test_ideas_unknown_id(capsys, env):
    code, _, err = run(capsys, "ideas", "show", "nope")
    assert code == 1 and "no idea with id" in err


def test_ask_defaults_on_eof():
    def boom(prompt):
        raise EOFError

    assert cli_lab._ask("? ", "yn", "n", boom) == "n"
    assert cli_lab._ask("? ", "ynlq", "l", lambda p: "") == "l"
    assert cli_lab._ask("? ", "yn", "n", lambda p: "Yes") == "y"


# --------------------------------------------------------------------------------------------
# discover: failing web source, already-read documents, sources.json, terminal safety
# --------------------------------------------------------------------------------------------


NOT_AN_IDEA = "A survey of the history of stock exchanges in Europe and their trading hours, with no strategy or signal."


def _doc(url: str, title: str, text: str = MOMENTUM_ABSTRACT) -> SourceDocument:
    return SourceDocument(source_type="arxiv", url=url, title=title, authors=["A. Researcher"], published=date(2020, 1, 15),
                          text=text, fetched_at=datetime(2026, 10, 1, tzinfo=timezone.utc), source_name="arXiv q-fin.PM")


def _arxiv_returning(monkeypatch, docs, record=None):
    from aitrading.discovery import sources

    class FakeArxiv:
        def __init__(self, categories=None, **kw):
            self.warnings: list[str] = []
            if record is not None:
                record["categories"] = categories

        def search_many(self, queries=None, **k):
            if record is not None:
                record["queries"] = queries
            return list(docs)

    class FakeFeeds:
        warnings: list[str] = []

        def fetch(self, *a, **k):
            return []

    monkeypatch.setattr(sources, "ArxivSource", FakeArxiv)
    monkeypatch.setattr(sources, "FeedSource", FakeFeeds)


@pytest.fixture
def claude_engine(monkeypatch, tmp_path):
    """Claude 'configured': a fake LLM, and an IdeaExtractor that counts calls (heuristic underneath)."""
    from types import SimpleNamespace

    from aitrading.discovery import extract

    (tmp_path / "empty-sources.json").write_text("{}", encoding="utf-8")
    monkeypatch.setenv("AITRADING_SOURCES", str(tmp_path / "empty-sources.json"))  # never the developer's own file
    calls: list[str] = []

    class CountingExtractor:
        def __init__(self, llm, **kw):
            self.inner = extract.HeuristicIdeaExtractor()

        def extract(self, doc):
            calls.append(doc.url)
            return self.inner.extract(doc)

    monkeypatch.setattr(extract, "IdeaExtractor", CountingExtractor)
    monkeypatch.setattr(cli_lab, "_llm_or_none", lambda args: SimpleNamespace(model="claude-opus-5-5"))
    return calls


def test_discover_web_search_failure_keeps_arxiv_documents(capsys, env, monkeypatch, claude_engine):
    from aitrading.discovery import websearch
    from aitrading.discovery.inbox import IdeaInbox
    from aitrading.llm.base import LLMError

    _arxiv_returning(monkeypatch, [_doc("https://arxiv.org/abs/2601.00001", "Twelve-month momentum revisited")])

    def boom(self, brief, *, max_ideas=10):
        raise LLMError("[discover:search] request rejected: web search is not enabled for this organization")

    monkeypatch.setattr(websearch.ClaudeWebSearchSource, "discover", boom)
    code, out, err = run(capsys, "discover", "--no-interactive", "--out", str(env / "out"))  # default sources: arxiv,feeds,web
    assert code == 0, err
    assert "web: skipped (LLMError" in err and "web search is not enabled" in err
    assert len(IdeaInbox().all()) == 1 and claude_engine == ["https://arxiv.org/abs/2601.00001"]


def test_discover_does_not_extract_documents_read_before(capsys, env, monkeypatch, claude_engine):
    from aitrading.discovery.inbox import IdeaInbox

    idea = _doc("https://arxiv.org/abs/2601.00001", "Twelve-month momentum revisited")
    other = _doc("https://arxiv.org/abs/2601.00002", "Stock exchanges of Europe", NOT_AN_IDEA)
    _arxiv_returning(monkeypatch, [idea, other])
    argv = ["discover", "--sources", "arxiv", "--no-interactive", "--out", str(env / "out")]
    code, _, err = run(capsys, *argv)
    assert code == 0 and sorted(claude_engine) == sorted([idea.url, other.url])
    assert len(IdeaInbox().all()) == 1

    claude_engine.clear()  # day 2: the same documents again (one with a new title) -> no Claude calls
    retitled = _doc("https://arxiv.org/abs/2601.00001v2", "Twelve-month momentum revisited (v2)")
    _arxiv_returning(monkeypatch, [idea, retitled, other])
    code, _, err = run(capsys, *argv)
    assert code == 0 and claude_engine == []
    assert "3 document(s) were read in an earlier run" in err and "--reextract" in err

    code, _, err = run(capsys, *argv, "--reextract")  # an intentional refresh reads them again
    assert code == 0 and len(claude_engine) == 3 and len(IdeaInbox().all()) == 1

    # a pruned (rejected long ago) idea is not paid for again either
    inbox = IdeaInbox()
    (c,) = inbox.all()
    inbox.set_status(c.idea_id, "rejected")
    assert inbox.prune(older_than_days=-1) == 1
    claude_engine.clear()
    _arxiv_returning(monkeypatch, [idea])
    code, _, _ = run(capsys, *argv)
    assert code == 0 and claude_engine == [] and IdeaInbox().all() == []


def test_discover_explicit_url_already_in_the_inbox_is_not_reread(capsys, env, monkeypatch, claude_engine):
    from aitrading.discovery import sources

    doc = _doc("https://example.com/paper", "Twelve-month momentum revisited")
    monkeypatch.setattr(sources, "fetch_url", lambda url: doc)
    code, _, _ = run(capsys, "discover", "--url", doc.url, "--no-interactive", "--out", str(env / "out"))
    assert code == 0 and claude_engine == [doc.url]
    code, _, err = run(capsys, "discover", "--url", doc.url, "--no-interactive", "--out", str(env / "out"))
    assert code == 0 and claude_engine == [doc.url] and "already read: https://example.com/paper" in err


def test_discover_applies_sources_json(capsys, env, monkeypatch, claude_engine, tmp_path):
    from aitrading.discovery import websearch

    cfg = tmp_path / "sources.json"
    cfg.write_text(json.dumps({"web_allowed_domains": ["ssrn.com", "arxiv.org"], "arxiv_categories": ["q-fin.PM"],
                               "arxiv_queries": ["earnings drift"]}), encoding="utf-8")
    monkeypatch.setenv("AITRADING_SOURCES", str(cfg))
    seen: dict = {}
    _arxiv_returning(monkeypatch, [], record=seen)

    def fake_discover(self, brief, *, max_ideas=10):
        seen["tools"] = self.tools()
        return []

    monkeypatch.setattr(websearch.ClaudeWebSearchSource, "discover", fake_discover)
    code, _, err = run(capsys, "discover", "--no-interactive", "--out", str(env / "out"))
    assert code == 0, err
    assert seen["categories"] == ["q-fin.PM"] and seen["queries"] == ["earnings drift"]
    assert seen["tools"] and all(t.get("allowed_domains") == ["ssrn.com", "arxiv.org"] for t in seen["tools"])
    assert "web: searching only ssrn.com, arxiv.org" in err

    # contradictory lists: a usage error before anything is fetched
    cfg.write_text(json.dumps({"web_allowed_domains": ["ssrn.com"], "web_blocked_domains": ["example.com"]}), encoding="utf-8")
    seen.clear()
    code, _, err = run(capsys, "discover", "--no-interactive", "--out", str(env / "out"))
    assert code == 2 and "not both" in err and "queries" not in seen


EVIL_TITLE = "Momentum in US stocks\x1b[8m"
EVIL_TEXT = MOMENTUM_ABSTRACT + " \x1b]52;c;ZWNobyBoaQ==\x07 \x1b[31mred\x1b[0m\r\x9b2K"


def test_untrusted_text_cannot_send_terminal_escapes(capsys, env, monkeypatch):
    from aitrading.discovery.extract import HeuristicIdeaExtractor
    from aitrading.discovery.inbox import IdeaInbox

    doc = _doc("https://example.com/p", EVIL_TITLE, EVIL_TEXT)
    (c,) = IdeaInbox().add([HeuristicIdeaExtractor().extract(doc)])
    for argv in (["ideas", "show", c.idea_id], ["ideas", "list"]):
        code, out, err = run(capsys, *argv)
        assert code == 0
        for ch in ("\x1b", "\x07", "\r", "\x9b"):
            assert ch not in out and ch not in err, (argv, repr(ch))
    code, out, _ = run(capsys, "ideas", "show", c.idea_id)
    assert "Momentum in US stocks�[8m" in out
    cli._note("warning: \x1b[2Kforged")
    assert "\x1b" not in capsys.readouterr().err


def test_review_output_survives_a_cp1252_console(env, monkeypatch):
    import io
    import sys

    from aitrading.discovery.extract import HeuristicIdeaExtractor
    from aitrading.discovery.inbox import IdeaInbox

    doc = _doc("https://example.com/a", "Momentum with α and an em dash — revisited")
    (c,) = IdeaInbox().add([HeuristicIdeaExtractor().extract(doc)])
    raw = io.BytesIO()
    monkeypatch.setattr(sys, "stdout", io.TextIOWrapper(raw, encoding="cp1252", errors="strict"))
    cli_lab._review_loop([c], None, lab=None, inbox=IdeaInbox(), ask=lambda p: "l")
    sys.stdout.flush()
    assert "Momentum with ? and an em dash".encode("cp1252") in raw.getvalue()
    assert IdeaInbox().get(c.idea_id).status == "deferred"


# --------------------------------------------------------------------------------------------
# saving accepted ideas, strategy run data
# --------------------------------------------------------------------------------------------


def test_accepted_idea_never_replaces_an_existing_strategy(capsys, env, monkeypatch):
    from aitrading.trading.store import StrategyStore

    out_dir = env / "out"
    code, _, err = run(capsys, "backtest", "--offline", "--no-open", "--out", str(out_dir), "12-1 momentum deciles", "--save", "momentum_12_1")
    assert code == 0, err
    store = StrategyStore()
    before = (store.path("momentum_12_1") / "spec.json").read_text(encoding="utf-8")
    _fake_sources(monkeypatch)
    answers = iter(["y", "y"])
    args = cli.build_parser().parse_args(["discover", "--offline", "--sources", "arxiv", "--out", str(out_dir), "--no-open"])
    assert cli_lab._cmd_discover(args, ask=lambda prompt: next(answers)) == 0
    out = capsys.readouterr().out
    from aitrading.discovery.inbox import IdeaInbox

    (idea,) = IdeaInbox().list()
    assert f"already exists; this idea is saved as 'momentum_12_1-{idea.idea_id}'" in out
    assert (store.path("momentum_12_1") / "spec.json").read_text(encoding="utf-8") == before
    assert store.load_meta("momentum_12_1")["notes"] == ""
    assert idea.idea_id in store.load_meta(f"momentum_12_1-{idea.idea_id}")["notes"]


def test_strategy_run_uses_the_data_it_was_backtested_and_paper_traded_on(capsys, env, monkeypatch, lab_provider):
    used: list[tuple] = []

    def lab_provider_spy(args):
        used.append((args.provider, args.tickers))
        return lab_provider

    monkeypatch.setattr(cli_lab, "_lab_provider", lab_provider_spy)
    out_dir = env / "out"
    code, _, err = run(capsys, "backtest", "--offline", "--provider", "synthetic", "--no-open", "--out", str(out_dir),
                       "12-1 momentum deciles", "--save", "mom")
    assert code == 0, err
    monkeypatch.setenv("AITRADING_PROVIDER", "free")  # the default a scheduled run would otherwise pick up
    code, out, err = run(capsys, "strategy", "run", "mom", "--out", str(out_dir), "--no-open")
    assert code == 0, err
    assert used[-1] == ("synthetic", None) and "simulated order(s)" in out
    assert "provider 'synthetic'" in err

    code, _, err = run(capsys, "strategy", "run", "mom", "--provider", "free", "--out", str(out_dir), "--no-open")
    assert code == 1 and "would mix markets in one ledger" in err and "aitrading strategy reset mom" in err
    code, _, err = run(capsys, "strategy", "run", "mom", "--tickers", "AAA,BBB", "--out", str(out_dir), "--no-open")
    assert code == 1 and "would mix markets" in err
    assert len(used) == 2  # neither refused run touched a provider or the ledger

    code, _, _ = run(capsys, "strategy", "reset", "mom", "--out", str(out_dir), "--no-open")
    assert code == 0
    code, _, err = run(capsys, "strategy", "run", "mom", "--provider", "free", "--out", str(out_dir), "--no-open")
    assert code == 0 and used[-1] == ("free", None)
    assert "was backtested on provider 'synthetic'" in err


def test_strategy_page_falls_back_when_the_working_folder_is_read_only(capsys, env, monkeypatch, tmp_path):
    out_dir = env / "out"
    code, _, err = run(capsys, "backtest", "--offline", "--no-open", "--out", str(out_dir), "12-1 momentum deciles", "--save", "mom")
    assert code == 0, err
    real = cli_lab._strategy_page

    def page(store, name, out_root):
        if not out_root.is_absolute():
            raise PermissionError(13, "Access is denied", str(out_root))
        return real(store, name, out_root)

    monkeypatch.setattr(cli_lab, "_strategy_page", page)
    monkeypatch.setenv("HOME", str(tmp_path / "userhome"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "userhome"))
    code, out, err = run(capsys, "strategy", "run", "mom", "--no-open")  # default relative --out, as under Task Scheduler
    assert code == 0, err
    assert "simulated order(s)" in out and "wrote the page under" in err
    assert (tmp_path / "userhome" / "aitrading_output" / "strategies" / "mom.html").exists()
    code, _, err = run(capsys, "strategy", "status", "mom", "--out", "elsewhere", "--no-open")
    assert code == 0 and "could not write the strategy page" in err


def test_backtest_summary_flags_numbers_not_in_the_results():
    from test_interpret import make_result

    from aitrading.backtest.models import BacktestInterpretation
    from aitrading.strategy.interpret import HeuristicInterpreter

    res = make_result(sharpe=0.42, alpha_t=1.6, mono=0.5)
    fabricated = BacktestInterpretation(summary="Sharpe of 1.4 with a t-stat of 3.2 - robust.", verdict="promising",
                                        key_findings=["Alpha t-stat 3.2, Sharpe 1.4"], cited_metrics=[],
                                        biases_and_caveats=[], next_experiments=[])
    lines = cli_lab._summary_lines(res.model_copy(update={"interpretation": fabricated}))
    assert any(line.startswith("UNVERIFIED: the verdict text states 1.4, 3.2") for line in lines)
    honest = HeuristicInterpreter().interpret(res)[0]
    assert not any("UNVERIFIED" in line for line in cli_lab._summary_lines(res.model_copy(update={"interpretation": honest})))
