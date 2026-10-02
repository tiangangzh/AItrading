"""Tests for aitrading.discovery.rank (scoring, novelty, ranking) and aitrading.discovery.inbox (persistence)."""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import threading
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from aitrading.core.models import EvidenceCheck
from aitrading.discovery.inbox import (
    SCHEMA_VERSION,
    IdeaInbox,
    InboxError,
    InboxLockTimeout,
    default_inbox_path,
)
from aitrading.discovery.models import IdeaCandidate, IdeaExtraction, ReplicationReport, RobustnessCheck, SourceDocument
from aitrading.discovery.rank import (
    WEIGHTS,
    assess_novelty,
    is_peer_reviewed,
    rank_candidates,
    sample_years,
    score_breakdown,
    score_candidate,
    title_similarity,
)

NOW = datetime(2026, 10, 2, 12, 0, tzinfo=timezone.utc)
TODAY = NOW.date()


class Clock:
    def __init__(self, now: datetime = NOW):
        self.now = now

    def __call__(self) -> datetime:
        return self.now


TEMPLATES = {
    "momentum_12_1": SimpleNamespace(key="momentum_12_1", title="12-1 price momentum", aliases=["momentum"], description="", references=[], build=None),
    "betting_against_beta": SimpleNamespace(key="betting_against_beta", title="Betting against beta", aliases=["BAB", "low beta"], description="", references=[], build=None),
}


def make_candidate(
    url: str = "https://arxiv.org/abs/2401.00001",
    title: str = "A new anomaly in stock returns",
    *,
    idea_title: str = "Earnings-call vocal stress",
    testability: str = "testable_now",
    is_trading_idea: bool = True,
    quotes: tuple[str, ...] = ("verified",),
    t_stat: float | None = 3.5,
    sample: str | None = "1990-2020",
    notes: tuple[str, ...] = ("Published in the Journal of Finance (peer-reviewed).",),
    published: date | None = date(2025, 6, 1),
    template: str | None = None,
    status: str = "new",
    score: float = 0.0,
    discovered_at: datetime = NOW,
) -> IdeaCandidate:
    doc = SourceDocument(source_type="arxiv", url=url, title=title, published=published, text="Some text.", fetched_at=NOW)
    ext = IdeaExtraction(
        is_trading_idea=is_trading_idea,
        title=idea_title,
        summary="s",
        claimed_effect="c",
        signal_description="sig",
        asset_class="us_equities",
        holding_period="1 month",
        reported_t_stat=t_stat,
        sample_period=sample,
        evidence_quotes=["q"] * len(quotes),
        data_requirements=["daily prices"],
        testability=testability,
        missing_data=[],
        proposed_strategy_idea="Long-short quintiles on beta_1y.",
        closest_library_template=template,
        credibility_notes=list(notes),
    )
    checks = [EvidenceCheck(kind="quote", ref=url, claim="q", status=s) for s in quotes]
    return IdeaCandidate(idea_id=doc.doc_key, source=doc, extraction=ext, quote_checks=checks, discovered_at=discovered_at,
                         status=status, score=score)


def replication(idea_id: str, run_id: str = "run-1") -> ReplicationReport:
    return ReplicationReport(
        idea_id=idea_id,
        backtest_run_id=run_id,
        base_sharpe=0.6,
        base_alpha_t_stat=2.8,
        claimed_sharpe=0.8,
        claimed_t_stat=3.5,
        replication_ratio=0.75,
        checks=[RobustnessCheck(name="first_half", description="1990-2005", sharpe=0.7, cagr_pct=6.0, alpha_t_stat=2.0, n_periods=180, passed=True)],
        verdict="partially_replicates",
        summary="Weaker than claimed.",
        caveats=["costs"],
    )


# =============================================================================================
# rank.py
# =============================================================================================


class TestScoring:
    def test_weights_sum_to_one(self):
        assert sum(WEIGHTS.values()) == pytest.approx(1.0)

    def test_best_case_scores_one_and_breakdown(self):
        c = make_candidate(sample="1963-2019")
        parts = score_breakdown(c, today=TODAY)
        assert parts == {"testability": 1.0, "evidence": 1.0, "credibility": 1.0, "novelty": 1.0, "recency": 1.0, "total": 1.0}
        assert score_candidate(c, today=TODAY) == 1.0

    def test_testability_ladder(self):
        scores = [score_candidate(make_candidate(testability=t), today=TODAY)
                  for t in ("testable_now", "partially_testable", "needs_institutional_data", "not_testable")]
        assert scores == sorted(scores, reverse=True) and len(set(scores)) == 4
        assert scores[0] - scores[-1] == pytest.approx(WEIGHTS["testability"])

    def test_evidence_credibility_novelty_recency(self):
        base = score_candidate(make_candidate(), today=TODAY)
        assert score_candidate(make_candidate(quotes=("verified", "not_found")), today=TODAY) == pytest.approx(base - WEIGHTS["evidence"] / 2)
        assert score_candidate(make_candidate(quotes=()), today=TODAY) == pytest.approx(base - WEIGHTS["evidence"])
        assert score_candidate(make_candidate(t_stat=2.2), today=TODAY) == pytest.approx(base - WEIGHTS["credibility"] / 6, abs=1e-4)
        assert score_candidate(make_candidate(t_stat=None, sample="2015-2020", notes=("arXiv preprint: not peer-reviewed.",)),
                               today=TODAY) == pytest.approx(base - WEIGHTS["credibility"])
        variant = make_candidate().model_copy(update={"novelty": "variant_of_library"})
        assert score_candidate(variant, today=TODAY) == pytest.approx(base - 0.4 * WEIGHTS["novelty"])
        assert score_candidate(make_candidate(published=date(2019, 1, 1)), today=TODAY) == pytest.approx(base - 0.5 * WEIGHTS["recency"])
        assert score_candidate(make_candidate(published=date(1993, 3, 1)), today=TODAY) == pytest.approx(base - WEIGHTS["recency"])
        assert score_candidate(make_candidate(published=None), today=TODAY) == pytest.approx(base - 0.75 * WEIGHTS["recency"])

    def test_not_a_trading_idea_scores_zero(self):
        assert score_candidate(make_candidate(is_trading_idea=False), today=TODAY) == 0.0

    def test_peer_review_and_sample_helpers(self):
        assert is_peer_reviewed(make_candidate(notes=("Published in the Review of Financial Studies.",)))
        assert not is_peer_reviewed(make_candidate(notes=("arXiv preprint: not peer-reviewed.",)))
        assert not is_peer_reviewed(make_candidate(notes=("Blog post.",)))
        assert sample_years("1963-2019") == 56
        assert sample_years("July 1963 to December 2000") == 37
        assert sample_years("1990-present", date(2026, 1, 1)) == 36
        assert sample_years(None) is None and sample_years("post-war") is None


class TestNovelty:
    def test_duplicate_by_url_including_arxiv_variants(self):
        existing = [make_candidate(url="https://arxiv.org/abs/2401.00001v1", title="Paper A")]
        c = make_candidate(url="http://www.arxiv.org/pdf/2401.00001v3.pdf", title="Completely different title")
        assert assess_novelty(c, existing, TEMPLATES) == "duplicate"

    def test_duplicate_by_title_similarity(self):
        existing = [make_candidate(url="https://ssrn.example/1", title="Betting Against Beta: New Evidence from 40 Countries")]
        c = make_candidate(url="https://blog.example/post", title="Betting against beta - new evidence from 40 countries!")
        assert assess_novelty(c, existing, TEMPLATES) == "duplicate"
        other = make_candidate(url="https://blog.example/other", title="Betting against correlation")
        assert assess_novelty(other, existing, TEMPLATES) != "duplicate"

    def test_variant_via_template_key_or_title_similarity(self):
        assert assess_novelty(make_candidate(template="momentum_12_1"), [], TEMPLATES) == "variant_of_library"
        assert assess_novelty(make_candidate(idea_title="Betting against beta (global)"), [], TEMPLATES) == "variant_of_library"
        assert assess_novelty(make_candidate(idea_title="Price momentum"), [], TEMPLATES) == "variant_of_library"  # alias "momentum"
        assert assess_novelty(make_candidate(idea_title="Earnings-call vocal stress"), [], TEMPLATES) == "new"
        assert assess_novelty(make_candidate(idea_title="Betting against beta"), [], None) == "new"  # no templates known

    def test_candidate_is_not_its_own_duplicate(self):
        c = make_candidate()
        assert assess_novelty(c, [c], TEMPLATES) == "new"

    def test_title_similarity(self):
        assert title_similarity("Momentum: Everywhere!", "momentum everywhere") == 1.0
        assert title_similarity("", "x") == 0.0
        assert 0.0 < title_similarity("Value and momentum everywhere", "Momentum crashes") < 0.6


def test_rank_candidates_by_score_then_published():
    a = make_candidate(url="u/a", title="A", score=0.5, published=date(2020, 1, 1))
    b = make_candidate(url="u/b", title="B", score=0.9, published=date(2010, 1, 1))
    c = make_candidate(url="u/c", title="C", score=0.5, published=date(2024, 1, 1))
    d = make_candidate(url="u/d", title="D", score=0.5, published=None)
    assert [x.source.title for x in rank_candidates([a, b, c, d])] == ["B", "C", "A", "D"]
    rescored = rank_candidates([make_candidate(url="u/x", title="X", testability="not_testable"), make_candidate(url="u/y", title="Y")],
                               rescore=True, today=TODAY)
    assert [x.source.title for x in rescored] == ["Y", "X"] and rescored[0].score > rescored[1].score > 0


# =============================================================================================
# inbox.py
# =============================================================================================


@pytest.fixture()
def inbox(tmp_path) -> IdeaInbox:
    return IdeaInbox(tmp_path / "ideas.json", templates=TEMPLATES, clock=Clock())


def test_default_path_uses_aitrading_home(monkeypatch, tmp_path):
    monkeypatch.setenv("AITRADING_HOME", str(tmp_path / "home"))
    assert default_inbox_path() == tmp_path / "home" / "ideas.json"
    assert IdeaInbox().path == tmp_path / "home" / "ideas.json"
    monkeypatch.delenv("AITRADING_HOME")
    assert default_inbox_path() == Path.home() / ".aitrading" / "ideas.json"


def test_round_trip_persistence(inbox, tmp_path):
    c1 = make_candidate(url="https://a.example/1", title="Alpha paper")
    c2 = make_candidate(url="https://b.example/2", title="Beta paper", testability="not_testable", template="momentum_12_1")
    added = inbox.add([c1, c2])
    assert [a.idea_id for a in added] == [c1.idea_id, c2.idea_id]
    assert added[0].novelty == "new" and added[1].novelty == "variant_of_library"
    assert added[0].score == score_candidate(added[0], today=TODAY) > added[1].score

    raw = json.loads((tmp_path / "ideas.json").read_text(encoding="utf-8"))
    assert raw["schema_version"] == SCHEMA_VERSION and len(raw["ideas"]) == 2

    reopened = IdeaInbox(tmp_path / "ideas.json", templates=TEMPLATES, clock=Clock())
    assert reopened.all() == added
    assert len(reopened) == 2 and c1.idea_id in reopened and "nope" not in reopened
    assert reopened.get(c1.idea_id) == added[0]
    assert reopened.get(c1.idea_id[:6]) == added[0]  # unique prefix
    assert reopened.get("does-not-exist") is None


def test_add_skips_duplicates_against_inbox_batch_and_arxiv_variants(inbox):
    first = inbox.add([make_candidate(url="https://arxiv.org/abs/2401.00001", title="Paper one")])
    assert len(first) == 1
    again = inbox.add([
        make_candidate(url="https://arxiv.org/pdf/2401.00001v2", title="Paper one (v2)"),  # same arXiv id
        make_candidate(url="https://c.example/x", title="Paper One"),  # same title
        make_candidate(url="https://d.example/y", title="Fresh idea"),
        make_candidate(url="https://d.example/y", title="Fresh idea"),  # duplicate inside the batch
    ])
    assert [c.source.title for c in again] == ["Fresh idea"]
    assert len(inbox) == 2
    assert inbox.add([]) == []


def test_list_sorted_filtered_and_limited(inbox):
    inbox.add([
        make_candidate(url="u/1", title="Low", testability="not_testable"),
        make_candidate(url="u/2", title="High"),
        make_candidate(url="u/3", title="Mid", testability="partially_testable"),
    ])
    assert [c.source.title for c in inbox.list()] == ["High", "Mid", "Low"]
    assert [c.source.title for c in inbox.list(limit=2)] == ["High", "Mid"]
    low = inbox.list()[-1]
    inbox.set_status(low.idea_id, "rejected")
    assert [c.source.title for c in inbox.list(status="new")] == ["High", "Mid"]
    assert [c.source.title for c in inbox.list(status="rejected")] == ["Low"]
    assert [c.source.title for c in inbox.list(status=["rejected", "new"], limit=1)] == ["High"]


def test_status_transitions(tmp_path):
    clock = Clock()
    inbox = IdeaInbox(tmp_path / "ideas.json", templates=TEMPLATES, clock=clock)
    [c] = inbox.add([make_candidate()])
    assert c.status == "new" and c.decided_at is None

    p = inbox.set_status(c.idea_id, "proposed")
    assert p.status == "proposed" and p.decided_at is None
    clock.now = NOW + timedelta(hours=2)
    a = inbox.set_status(c.idea_id, "accepted", note="Looks interesting, try it")
    assert a.status == "accepted" and a.decided_at == NOW + timedelta(hours=2)
    assert a.notes[-1].endswith("accepted] Looks interesting, try it")
    assert inbox.get(c.idea_id).decided_at == NOW + timedelta(hours=2)  # persisted
    for s in ("rejected", "deferred"):
        clock.now += timedelta(hours=1)
        assert inbox.set_status(c.idea_id, s).decided_at == clock.now
    t = inbox.set_status(c.idea_id, "tested")
    assert t.status == "tested" and t.decided_at == clock.now  # 'tested' keeps the decision time

    with pytest.raises(ValueError):
        inbox.set_status(c.idea_id, "maybe")
    with pytest.raises(KeyError):
        inbox.set_status("nope-nope", "accepted")


def test_attach_spec_and_result(inbox):
    from aitrading.strategy.spec import SignalComponent, StrategySpec

    [c] = inbox.add([make_candidate()])
    spec = StrategySpec(name="bab", idea="Betting against beta", kind="cross_sectional",
                        signal=[SignalComponent(feature="beta_1y", direction="lower_is_better")])
    out = inbox.attach_spec(c.idea_id, spec)
    assert out.strategy_spec["signal"][0]["feature"] == "beta_1y"
    assert StrategySpec.model_validate(inbox.get(c.idea_id).strategy_spec) == spec
    assert inbox.attach_spec(c.idea_id, {"name": "x", "start": date(2010, 1, 1)}).strategy_spec == {"name": "x", "start": "2010-01-01"}

    rep = replication(c.idea_id)
    res = inbox.attach_result(c.idea_id, "run-1", rep)
    assert res.status == "tested" and res.backtest_run_id == "run-1" and res.replication == rep
    stored = inbox.get(c.idea_id)
    assert stored.replication == rep and stored.status == "tested"
    failed = inbox.attach_result(c.idea_id, "run-2", replication(c.idea_id, "run-2").model_dump(mode="json"), status="failed")
    assert failed.status == "failed" and failed.replication.backtest_run_id == "run-2"
    assert inbox.attach_result(c.idea_id, "run-3").replication.backtest_run_id == "run-2"  # None keeps the report
    with pytest.raises(KeyError):
        inbox.attach_result("missing!", "run-4")


def test_prune_removes_old_rejected_and_remembers_them(tmp_path):
    clock = Clock()
    inbox = IdeaInbox(tmp_path / "ideas.json", templates=TEMPLATES, clock=clock)
    old_rej, new_rej, old_new = inbox.add([
        make_candidate(url="u/1", title="Old rejected"),
        make_candidate(url="u/2", title="New rejected"),
        make_candidate(url="u/3", title="Old but undecided"),
    ])
    inbox.set_status(old_rej.idea_id, "rejected")
    clock.now = NOW + timedelta(days=40)
    inbox.set_status(new_rej.idea_id, "rejected")
    clock.now = NOW + timedelta(days=45)

    assert inbox.prune(30) == 1
    assert {c.source.title for c in inbox.all()} == {"New rejected", "Old but undecided"}
    assert inbox.prune(30, statuses=("new",)) == 1
    assert {c.source.title for c in inbox.all()} == {"New rejected"}
    with pytest.raises(ValueError):
        inbox.prune(1, statuses=("bogus",))

    # a pruned (rejected) paper that is discovered again is not proposed again
    assert inbox.add([make_candidate(url="u/1", title="Old rejected")]) == []
    assert inbox.add([make_candidate(url="https://elsewhere.example/p", title="Old rejected")]) == []
    raw = json.loads((tmp_path / "ideas.json").read_text(encoding="utf-8"))
    assert {t["title"] for t in raw["pruned"]} == {"Old rejected", "Old but undecided"}


def test_prune_handles_naive_datetimes(tmp_path):
    inbox = IdeaInbox(tmp_path / "ideas.json", templates=TEMPLATES, clock=Clock())
    inbox.add([make_candidate(discovered_at=datetime(2020, 1, 1), status="rejected")])  # naive
    assert inbox.prune(10) == 1


def test_remove(inbox):
    [c] = inbox.add([make_candidate()])
    assert inbox.remove(c.idea_id).idea_id == c.idea_id
    assert len(inbox) == 0
    assert len(inbox.add([make_candidate()])) == 1  # removal leaves no tombstone


def test_ambiguous_prefix(inbox):
    a, b = make_candidate(url="u/a", title="A"), make_candidate(url="u/b", title="B")
    a = a.model_copy(update={"idea_id": "abcd1111"})
    b = b.model_copy(update={"idea_id": "abcd2222"})
    inbox.add([a, b])
    assert inbox.get("abcd") is None
    with pytest.raises(KeyError, match="ambiguous"):
        inbox.set_status("abcd", "accepted")
    assert inbox.set_status("abcd2", "accepted").idea_id == "abcd2222"


def test_atomic_write_keeps_old_file_on_failure(inbox, tmp_path, monkeypatch):
    inbox.add([make_candidate(url="u/1", title="Kept")])
    before = (tmp_path / "ideas.json").read_bytes()

    def boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError, match="disk full"):
        inbox.add([make_candidate(url="u/2", title="Lost")])
    monkeypatch.undo()
    assert (tmp_path / "ideas.json").read_bytes() == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["ideas.json"]  # no temp file, lock released
    assert [c.source.title for c in inbox.all()] == ["Kept"]


def test_failed_mutation_does_not_write(inbox, tmp_path):
    inbox.add([make_candidate()])
    before = (tmp_path / "ideas.json").read_bytes()
    with pytest.raises(KeyError):
        inbox.set_status("missing!", "accepted")
    assert (tmp_path / "ideas.json").read_bytes() == before
    assert not (tmp_path / "ideas.json.lock").exists()


def test_lock_blocks_other_writers_and_times_out(tmp_path):
    path = tmp_path / "ideas.json"
    holder = IdeaInbox(path, templates=TEMPLATES, clock=Clock())
    other = IdeaInbox(path, templates=TEMPLATES, clock=Clock(), lock_timeout_s=0.2)
    with holder.locked():
        assert (tmp_path / "ideas.json.lock").exists()
        holder.add([make_candidate(url="u/1", title="Inside the lock")])  # re-entrant, no deadlock
        t0 = time.monotonic()
        with pytest.raises(InboxLockTimeout):
            other.add([make_candidate(url="u/2", title="Blocked")])
        assert time.monotonic() - t0 >= 0.2
        assert len(other.list()) == 1  # reads do not need the lock
    assert not (tmp_path / "ideas.json.lock").exists()
    assert len(other.add([make_candidate(url="u/2", title="Now fine")])) == 1


def test_stale_lock_is_broken(tmp_path):
    path = tmp_path / "ideas.json"
    lock = tmp_path / "ideas.json.lock"
    lock.write_text("pid=99999\n")
    old = time.time() - 3600
    os.utime(lock, (old, old))
    inbox = IdeaInbox(path, templates=TEMPLATES, clock=Clock(), lock_timeout_s=1.0, stale_lock_s=60)
    assert len(inbox.add([make_candidate()])) == 1
    assert not lock.exists()


def test_release_never_removes_a_lock_owned_by_someone_else(tmp_path):
    inbox = IdeaInbox(tmp_path / "ideas.json", templates=TEMPLATES, clock=Clock())
    lock = tmp_path / "ideas.json.lock"
    with inbox.locked():
        lock.write_text("pid=1 token=someone-else\n")  # our lock was broken as stale and re-taken
    assert lock.read_text() == "pid=1 token=someone-else\n"


def test_threads_with_separate_instances_do_not_lose_updates(tmp_path):
    path = tmp_path / "ideas.json"
    errors: list[BaseException] = []

    def worker(k: int) -> None:
        box = IdeaInbox(path, templates=TEMPLATES, clock=Clock())  # own instance -> contends via the lock file
        try:
            for i in range(5):
                box.add([make_candidate(url=f"https://t{k}.example/{i}", title=unique_title(f"thread{k}-{i}"))])
        except BaseException as e:  # pragma: no cover - surfaced by the assert below
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(k,)) for k in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert len(IdeaInbox(path, templates=TEMPLATES).all()) == 20


def unique_title(seed: str) -> str:
    """Titles that are not near-duplicates of each other (similar titles are de-duplicated)."""
    return "Study " + hashlib.sha1(seed.encode()).hexdigest()[:16]


def _process_worker(path: str, k: int) -> None:
    box = IdeaInbox(Path(path), templates=TEMPLATES, clock=Clock())
    for i in range(5):
        box.add([make_candidate(url=f"https://p{k}.example/{i}", title=unique_title(f"process{k}-{i}"))])


def test_processes_do_not_corrupt_the_inbox(tmp_path):
    path = tmp_path / "ideas.json"
    ctx = multiprocessing.get_context("spawn")  # same start method as Windows / macOS
    procs = [ctx.Process(target=_process_worker, args=(str(path), k)) for k in range(3)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(60)
    assert all(p.exitcode == 0 for p in procs)
    ideas = IdeaInbox(path, templates=TEMPLATES).all()
    assert len(ideas) == 15 and len({c.idea_id for c in ideas}) == 15


def test_corrupt_file_is_backed_up_and_reads_are_safe(tmp_path, caplog):
    path = tmp_path / "ideas.json"
    path.write_text("{not json", encoding="utf-8")
    inbox = IdeaInbox(path, templates=TEMPLATES, clock=Clock())
    assert inbox.list() == []  # read: warn, treat as empty, leave the file alone
    assert path.read_text(encoding="utf-8") == "{not json"
    assert len(inbox.add([make_candidate()])) == 1  # write: back up the corrupt file, start fresh
    backups = list(tmp_path.glob("ideas.json.corrupt-*"))
    assert len(backups) == 1 and backups[0].read_text(encoding="utf-8") == "{not json"
    assert len(inbox) == 1


def test_newer_schema_is_refused(tmp_path):
    path = tmp_path / "ideas.json"
    path.write_text(json.dumps({"schema_version": SCHEMA_VERSION + 1, "ideas": []}), encoding="utf-8")
    with pytest.raises(InboxError, match="schema_version"):
        IdeaInbox(path, templates=TEMPLATES).list()


def test_invalid_entries_are_quarantined_not_lost(tmp_path):
    path = tmp_path / "ideas.json"
    good = make_candidate().model_dump(mode="json")
    path.write_text(json.dumps({"schema_version": 1, "ideas": [good, {"idea_id": "broken"}]}), encoding="utf-8")
    inbox = IdeaInbox(path, templates=TEMPLATES, clock=Clock())
    assert [c.idea_id for c in inbox.all()] == [good["idea_id"]]
    inbox.add([make_candidate(url="u/new", title="New one")])
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["quarantined"] == [{"idea_id": "broken"}] and len(raw["ideas"]) == 2


def test_bare_list_file_is_accepted(tmp_path):
    path = tmp_path / "ideas.json"
    path.write_text(json.dumps([make_candidate().model_dump(mode="json")]), encoding="utf-8")
    assert len(IdeaInbox(path, templates=TEMPLATES).all()) == 1


def test_inbox_uses_library_templates_lazily(tmp_path, monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "aitrading.strategy.library", None)  # library unavailable -> no templates
    inbox = IdeaInbox(tmp_path / "ideas.json", clock=Clock())
    [c] = inbox.add([make_candidate(idea_title="Betting against beta")])
    assert c.novelty == "new"
