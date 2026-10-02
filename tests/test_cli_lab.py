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
