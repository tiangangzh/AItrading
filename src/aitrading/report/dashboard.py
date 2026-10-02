"""Local dashboard: one ``index.html`` linking every saved report, strategy and idea.

:func:`write_dashboard` scans the artefacts the platform saves on the trader's PC and writes a
small static site (relative links only, so the folder can be moved, zipped or opened from any OS)::

    <out_dir>/index.html                 summary tables + links
    <out_dir>/backtests/<run>.html       one page per backtest result
    <out_dir>/screens/<run>.html         one page per screening (pipeline) result
    <out_dir>/strategies/<slug>.html     one page per saved strategy (idea, paper account, backtest)
    <out_dir>/ideas.html                 the idea inbox

Sources (each optional; defaults under ``$AITRADING_HOME`` or ``~/.aitrading``):

* ``runs_dir`` (default ``<home>/runs``): every ``result.json`` / ``backtest.json`` below it is read
  as a :class:`~aitrading.backtest.models.BacktestResult` (or, failing that, a screening
  :class:`~aitrading.core.models.PipelineResult`);
* ``strategies_dir`` (default ``<home>/strategies``): the strategy store layout
  (``<slug>/spec.json``, ``backtest.json``, ``meta.json``, ``ledger.json``); the paper account
  summary comes from :class:`aitrading.trading.paper.PaperAccount` (read-only);
* ``inbox_path`` (default ``<home>/ideas.json``): the idea inbox.

Missing directories are fine; unreadable or corrupt files are skipped and listed under "Skipped
files" on the index page - one bad file never stops the dashboard. Nothing is ever written
outside ``out_dir`` and the source files are only read.
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import webbrowser
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import ValidationError

from aitrading.backtest.models import BacktestResult
from aitrading.core.models import PipelineResult
from aitrading.discovery.models import IdeaCandidate
from aitrading.report import svg
from aitrading.report.html import (
    _VERDICTS,
    _anchor,
    _badge,
    _e,
    _finite,
    _int,
    _num,
    _pct,
    _primary_key,
    _section,
    _table,
    _TESTABILITY,
    _STATUS_ORDER,
    _STATUS_TONE,
    html_page,
    render_backtest_html,
    render_ideas_inbox,
    render_pipeline_html,
    render_strategy_page,
    template_info,
)
from aitrading.strategy.spec import StrategySpec

__all__ = [
    "ENV_HOME",
    "aitrading_home",
    "default_runs_dir",
    "default_strategies_dir",
    "default_inbox_path",
    "write_dashboard",
    "open_in_browser",
]

log = logging.getLogger(__name__)

ENV_HOME = "AITRADING_HOME"
RESULT_FILE_NAMES = ("result.json", "backtest.json")
MAX_BACKTEST_ROWS = 200


def aitrading_home() -> Path:
    """``$AITRADING_HOME`` when set, else ``~/.aitrading``."""
    env = os.environ.get(ENV_HOME, "").strip()
    return Path(env).expanduser() if env else Path.home() / ".aitrading"


def default_runs_dir() -> Path:
    return aitrading_home() / "runs"


def default_strategies_dir() -> Path:
    return aitrading_home() / "strategies"


def default_inbox_path() -> Path:
    return aitrading_home() / "ideas.json"


# ------------------------------------------------------------------------------------------------
# Small I/O helpers
# ------------------------------------------------------------------------------------------------


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


_WINDOWS_RESERVED = frozenset({"con", "prn", "aux", "nul", *(f"com{i}" for i in range(10)), *(f"lpt{i}" for i in range(10))})


def _slug(text: str, used: set[str], max_len: int = 80) -> str:
    """Filesystem- and URL-safe file stem (also on Windows), unique within ``used`` (case-insensitive)."""
    s = re.sub(r"[^A-Za-z0-9._-]+", "-", str(text)).strip("-._")[:max_len].rstrip("-._") or "item"
    if s.split(".")[0].lower() in _WINDOWS_RESERVED:
        s = f"_{s}"
    base, i = s, 2
    while s.lower() in used:
        s = f"{base}-{i}"
        i += 1
    used.add(s.lower())
    return s


def _short_err(e: BaseException) -> str:
    msg = str(e).strip().splitlines()[0] if str(e).strip() else ""
    return f"{type(e).__name__}: {msg}"[:300]


def _as_dt(v: Any) -> datetime | None:
    if isinstance(v, datetime):
        return v if v.tzinfo else v.replace(tzinfo=timezone.utc)
    if isinstance(v, date):
        return datetime(v.year, v.month, v.day, tzinfo=timezone.utc)
    if isinstance(v, str) and v.strip():
        try:
            d = datetime.fromisoformat(v.strip().replace("Z", "+00:00"))
        except ValueError:
            return None
        return d if d.tzinfo else d.replace(tzinfo=timezone.utc)
    return None


# ------------------------------------------------------------------------------------------------
# Scanning
# ------------------------------------------------------------------------------------------------


@dataclass
class _Scan:
    backtests: list[tuple[BacktestResult, Path]] = field(default_factory=list)
    screens: list[tuple[PipelineResult, Path]] = field(default_factory=list)
    strategies: list[dict[str, Any]] = field(default_factory=list)
    ideas: list[IdeaCandidate] | None = None
    notes: list[str] = field(default_factory=list)


def _scan_runs(runs_dir: Path, scan: _Scan) -> None:
    if not runs_dir.is_dir():
        return
    files: list[Path] = []
    try:
        for name in RESULT_FILE_NAMES:
            files.extend(runs_dir.rglob(name))
    except OSError as e:
        scan.notes.append(f"Could not scan {runs_dir}: {_short_err(e)}")
        return
    for f in sorted(set(files)):
        try:
            data = _read_json(f)
        except (OSError, ValueError, UnicodeDecodeError) as e:
            scan.notes.append(f"Skipped {f}: not readable JSON ({_short_err(e)})")
            continue
        if not isinstance(data, dict):
            scan.notes.append(f"Skipped {f}: expected a JSON object")
            continue
        try:
            scan.backtests.append((BacktestResult.model_validate(data), f))
            continue
        except ValidationError as e_bt:
            bt_err = e_bt
        try:
            scan.screens.append((PipelineResult.model_validate(data), f))
        except ValidationError:
            scan.notes.append(f"Skipped {f}: not a backtest or screening result ({bt_err.error_count()} validation errors)")


def _paper_summary(strategies_dir: Path, slug: str, backtest: BacktestResult | None) -> dict[str, Any]:
    """Read-only paper account summary (``PaperAccount.summary``) of a stored strategy."""
    from aitrading.trading.paper import PaperAccount  # lazy: optional dependency of the dashboard
    from aitrading.trading.store import StrategyStore

    store = StrategyStore(strategies_dir)
    if store.path(slug).name != slug:  # a hand-made directory the store cannot address by name
        raise ValueError(f"directory name {slug!r} is not a strategy-store slug ({store.path(slug).name!r})")
    account = PaperAccount(store, slug)
    return account.summary(backtest=backtest)


def _scan_strategies(strategies_dir: Path, scan: _Scan) -> None:
    if not strategies_dir.is_dir():
        return
    try:
        dirs = sorted(p for p in strategies_dir.iterdir() if p.is_dir() and not p.name.startswith("."))
    except OSError as e:
        scan.notes.append(f"Could not scan {strategies_dir}: {_short_err(e)}")
        return
    for d in dirs:
        spec_file = d / "spec.json"
        if not spec_file.is_file():
            continue
        entry: dict[str, Any] = {"slug": d.name, "dir": d, "name": d.name, "meta": {}, "spec": None, "backtest": None, "paper": None}
        meta_file = d / "meta.json"
        if meta_file.is_file():
            try:
                meta = _read_json(meta_file)
                if isinstance(meta, dict):
                    entry["meta"] = meta
                    entry["name"] = str(meta.get("name") or d.name)
                else:
                    scan.notes.append(f"Ignored {meta_file}: expected a JSON object")
            except (OSError, ValueError, UnicodeDecodeError) as e:
                scan.notes.append(f"Ignored {meta_file}: {_short_err(e)}")
        try:
            raw_spec = _read_json(spec_file)
        except (OSError, ValueError, UnicodeDecodeError) as e:
            scan.notes.append(f"Skipped strategy {d.name}: {spec_file} is not readable JSON ({_short_err(e)})")
            continue
        try:
            entry["spec"] = StrategySpec.model_validate(raw_spec)
        except ValidationError as e:
            scan.notes.append(f"Strategy {d.name}: {spec_file} is not a valid StrategySpec ({e.error_count()} errors); shown as saved")
            entry["spec"] = raw_spec if isinstance(raw_spec, dict) else None
        bt_file = d / "backtest.json"
        if bt_file.is_file():
            try:
                entry["backtest"] = BacktestResult.model_validate(_read_json(bt_file))
            except (OSError, ValueError, UnicodeDecodeError) as e:
                scan.notes.append(f"Strategy {d.name}: ignored unreadable {bt_file} ({_short_err(e)})")
        if (d / "ledger.json").is_file():
            try:
                entry["paper"] = _paper_summary(strategies_dir, d.name, entry["backtest"])
            except Exception as e:  # noqa: BLE001 - a broken ledger must not break the dashboard
                scan.notes.append(f"Strategy {d.name}: paper ledger could not be read ({_short_err(e)})")
        scan.strategies.append(entry)


def _scan_inbox(inbox_path: Path, scan: _Scan) -> None:
    if not inbox_path.is_file():
        return
    try:
        data = _read_json(inbox_path)
    except (OSError, ValueError, UnicodeDecodeError) as e:
        scan.notes.append(f"Skipped idea inbox {inbox_path}: not readable JSON ({_short_err(e)})")
        return
    raw = data.get("ideas") if isinstance(data, dict) else data if isinstance(data, list) else None
    if not isinstance(raw, list):
        scan.notes.append(f"Skipped idea inbox {inbox_path}: no 'ideas' list")
        return
    ideas: list[IdeaCandidate] = []
    bad = 0
    for item in raw:
        try:
            ideas.append(IdeaCandidate.model_validate(item))
        except (ValidationError, TypeError, ValueError):
            bad += 1
    if bad:
        scan.notes.append(f"Idea inbox {inbox_path}: skipped {bad} invalid idea entr{'y' if bad == 1 else 'ies'}")
    if isinstance(data, dict) and isinstance(data.get("quarantined"), list) and data["quarantined"]:
        scan.notes.append(f"Idea inbox {inbox_path}: {len(data['quarantined'])} quarantined entr"
                          f"{'y' if len(data['quarantined']) == 1 else 'ies'} not shown")
    scan.ideas = ideas


# ------------------------------------------------------------------------------------------------
# Page writing
# ------------------------------------------------------------------------------------------------


def _bt_date(bt: BacktestResult) -> datetime:
    return _as_dt(bt.finished_at) or _as_dt(bt.started_at) or datetime.min.replace(tzinfo=timezone.utc)


def _template_args(spec_name: str | None) -> dict[str, Any]:
    info = template_info(spec_name)
    if info is None:
        return {}
    title, desc, refs = info
    return {"template_title": title, "template_description": desc, "references": refs}


def _verdict_badge(verdict: str | None) -> str:
    if not verdict:
        return '<span class="muted">n/a</span>'
    label, tone, icon, meaning = _VERDICTS.get(verdict, (verdict, "muted", "?", ""))
    return _badge(label, tone, icon=icon, title=meaning)


def _nav_sparkline(paper: dict[str, Any] | None) -> str:
    if not paper:
        return ""
    pts = {}
    for item in paper.get("nav_history") or []:
        if isinstance(item, (list, tuple)) and len(item) >= 2 and _finite(item[1]) is not None:
            try:
                pts[pd.Timestamp(item[0])] = float(item[1])
            except (ValueError, TypeError):
                continue
    if len(pts) < 2:
        return ""
    return svg.sparkline(pd.Series(pts).sort_index(), title="Paper NAV", width=100, height=24)


def write_dashboard(
    out_dir: Path,
    *,
    runs_dir: Path | None = None,
    strategies_dir: Path | None = None,
    inbox_path: Path | None = None,
) -> Path:
    """Scan saved artefacts and write ``out_dir/index.html`` (plus one page per item); returns its path."""
    out_dir = Path(out_dir).expanduser()
    runs_dir = Path(runs_dir).expanduser() if runs_dir is not None else default_runs_dir()
    strategies_dir = Path(strategies_dir).expanduser() if strategies_dir is not None else default_strategies_dir()
    inbox_path = Path(inbox_path).expanduser() if inbox_path is not None else default_inbox_path()
    out_dir.mkdir(parents=True, exist_ok=True)

    scan = _Scan()
    _scan_runs(runs_dir, scan)
    _scan_strategies(strategies_dir, scan)
    _scan_inbox(inbox_path, scan)

    used: dict[str, set[str]] = {"backtests": set(), "screens": set(), "strategies": set()}

    # ---- backtests (runs + strategy backtests, de-duplicated by run id) ----------------------
    bt_rows: list[tuple[datetime, list[str]]] = []
    bt_links: dict[str, str] = {}
    all_bts: list[tuple[BacktestResult, str]] = [(bt, str(p)) for bt, p in scan.backtests]
    all_bts += [(s["backtest"], str(s["dir"] / "backtest.json")) for s in scan.strategies if s["backtest"] is not None]
    for bt, src in all_bts:
        if bt.run_id in bt_links:
            continue
        stem = _slug(bt.run_id, used["backtests"])
        rel = f"backtests/{stem}.html"
        spec_name = bt.spec.get("name") if isinstance(bt.spec, dict) else None
        try:
            page = render_backtest_html(bt, home_href="../index.html", **_template_args(spec_name))
            _write_text_atomic(out_dir / "backtests" / f"{stem}.html", page)
        except Exception as e:  # noqa: BLE001 - one bad result must not stop the dashboard
            scan.notes.append(f"Could not render backtest {bt.run_id} from {src}: {_short_err(e)}")
            continue
        bt_links[bt.run_id] = rel
        key = _primary_key(bt)
        st = bt.stats.get(key) if key else None
        verdict = bt.interpretation.verdict if bt.interpretation else None
        when = _bt_date(bt)
        bt_rows.append((when, [
            f'<a href="{_e(rel)}">{_e(bt.idea[:140])}</a>',
            _e(spec_name or ""),
            _verdict_badge(verdict),
            _num(st.sharpe) if st else "n/a",
            _pct(st.cagr_pct, 1, signed=True) if st else "n/a",
            _pct(st.max_drawdown_pct) if st else "n/a",
            _e(f"{bt.start.isoformat()} to {bt.end.isoformat()}"),
            _e(when.strftime("%Y-%m-%d %H:%M") if when.year > 1 else "n/a"),
        ]))
    bt_rows.sort(key=lambda r: r[0], reverse=True)

    # ---- screening runs -----------------------------------------------------------------------
    screen_rows: list[tuple[datetime, list[str]]] = []
    for res, src in scan.screens:
        stem = _slug(res.run_id, used["screens"])
        rel = f"screens/{stem}.html"
        try:
            _write_text_atomic(out_dir / "screens" / f"{stem}.html", render_pipeline_html(res, home_href="../index.html"))
        except Exception as e:  # noqa: BLE001
            scan.notes.append(f"Could not render screening run {res.run_id} from {src}: {_short_err(e)}")
            continue
        when = _as_dt(res.finished_at) or _as_dt(res.started_at) or datetime.min.replace(tzinfo=timezone.utc)
        screen_rows.append((when, [f'<a href="{_e(rel)}">{_e(res.observation[:140])}</a>', _e(res.as_of.isoformat()),
                                   _int(res.universe_size), _int(res.survivors), _int(len(res.ideas)), _e(res.provider)]))
    screen_rows.sort(key=lambda r: r[0], reverse=True)

    # ---- strategies ---------------------------------------------------------------------------
    strat_rows: list[list[str]] = []
    for s in scan.strategies:
        stem = _slug(s["slug"], used["strategies"])
        rel = f"strategies/{stem}.html"
        try:
            page = render_strategy_page(s["name"], s["spec"], s["backtest"], s["paper"], home_href="../index.html")
            _write_text_atomic(out_dir / "strategies" / f"{stem}.html", page)
        except Exception as e:  # noqa: BLE001
            scan.notes.append(f"Could not render strategy {s['name']}: {_short_err(e)}")
            continue
        paper = s["paper"] or {}
        bt: BacktestResult | None = s["backtest"]
        headline = (s["meta"] or {}).get("backtest") or {}
        st = bt.stats.get(_primary_key(bt) or "") if bt is not None else None
        sharpe = st.sharpe if st is not None else headline.get("sharpe") if isinstance(headline, dict) else None
        cagr = st.cagr_pct if st is not None else headline.get("cagr_pct") if isinstance(headline, dict) else None
        verdict = (bt.interpretation.verdict if bt is not None and bt.interpretation else
                   headline.get("verdict") if isinstance(headline, dict) else None)
        saved = str((s["meta"] or {}).get("updated") or "")[:10]
        last_run = paper.get("last_run") or (f"saved {saved}" if saved else "")
        spec = s["spec"]
        kind = spec.kind if isinstance(spec, StrategySpec) else (spec or {}).get("kind", "") if isinstance(spec, dict) else ""
        strat_rows.append([
            f'<a href="{_e(rel)}">{_e(s["name"])}</a>',
            _e(str(kind).replace("_", " ")),
            _e(str(last_run)[:16]),
            (_pct(paper.get("since_start_return_pct"), 2, signed=True) + " " + _nav_sparkline(paper)) if paper else '<span class="muted">not paper-trading</span>',
            _pct(paper.get("backtest_expected_return_pct"), 2, signed=True) if paper else "",
            _num(sharpe),
            _pct(cagr, 1, signed=True),
            _verdict_badge(verdict),
        ])

    # ---- ideas --------------------------------------------------------------------------------
    ideas_html = ""
    if scan.ideas is not None:
        try:
            _write_text_atomic(out_dir / "ideas.html", render_ideas_inbox(scan.ideas, home_href="index.html"))
            ideas_ok = True
        except Exception as e:  # noqa: BLE001
            scan.notes.append(f"Could not render the idea inbox: {_short_err(e)}")
            ideas_ok = False
        counts: dict[str, int] = {}
        for c in scan.ideas:
            counts[c.status] = counts.get(c.status, 0) + 1
        chips = "".join(_badge(f"{st}: {counts[st]}", _STATUS_TONE.get(st, "muted"), icon="")
                        for st in sorted(counts, key=lambda x: _STATUS_ORDER.index(x) if x in _STATUS_ORDER else 99))
        new = sorted((c for c in scan.ideas if c.status == "new"), key=lambda c: -(_finite(c.score) or 0.0))[:5]
        rows = []
        for c in new:
            t_label, t_tone, t_icon = _TESTABILITY.get(c.extraction.testability, (c.extraction.testability, "muted", "?"))
            n_ok = sum(ch.status == "verified" for ch in c.quote_checks)
            title = _e(c.extraction.title)
            link = f'<a href="ideas.html#idea-{_e(_anchor(c.idea_id))}">{title}</a>' if ideas_ok else title
            rows.append([link, _badge(t_label, t_tone, icon=t_icon), _num(c.score),
                         _e(f"{n_ok}/{len(c.extraction.evidence_quotes)}"), f'<code class="cmd">aitrading ideas try {_e(c.idea_id)}</code>'])
        body = f'<div class="badges">{chips or "<span class=muted>empty</span>"}</div>'
        if rows:
            body += _table(["Top new ideas", "Testability", "Score", "Quotes verified", "Try it"], rows, num_cols=[2, 3])
        else:
            body += '<p class="muted">No new ideas waiting for a decision.</p>'
        if ideas_ok:
            body += f'<p><a href="ideas.html">Open the full idea inbox ({len(scan.ideas):,} ideas) →</a></p>'
        ideas_html = _section("ideas", "Idea inbox", body, intro="Ideas found by the auto strategy creator, waiting for your decision.")
    else:
        ideas_html = _section("ideas", "Idea inbox", f'<p class="muted">No idea inbox found at {_e(inbox_path)}. '
                                                     "Run <code>aitrading ideas discover</code> to find ideas.</p>")

    # ---- index --------------------------------------------------------------------------------
    now = datetime.now().astimezone()
    parts = [
        '<header class="top"><p class="eyebrow">Equity research and idea lab - simulated trading only</p>'
        "<h1>AItrading dashboard</h1>"
        f'<div class="meta"><span>Generated: <b>{_e(now.strftime("%Y-%m-%d %H:%M %Z"))}</b></span>'
        f"<span>Strategies: <b>{len(strat_rows)}</b></span><span>Backtests: <b>{len(bt_rows)}</b></span>"
        f"<span>Screens: <b>{len(screen_rows)}</b></span>"
        f"<span>Ideas: <b>{len(scan.ideas) if scan.ideas is not None else 0}</b></span></div>"
        '<nav class="toc"><a href="#strategies">Strategies</a><a href="#backtests">Backtests</a>'
        + ('<a href="#screens">Screens</a>' if screen_rows else "")
        + '<a href="#ideas">Ideas</a>' + ('<a href="#skipped">Skipped files</a>' if scan.notes else "") + "</nav></header>"
    ]
    if strat_rows:
        parts.append(_section("strategies", "Strategies", _table(
            ["Strategy", "Kind", "Last run", "Paper return since start", "Backtest expected", "Backtest Sharpe", "Backtest CAGR", "Verdict"],
            strat_rows, num_cols=[3, 4, 5, 6]), intro="Saved strategies and their simulated (paper) accounts."))
    else:
        parts.append(_section("strategies", "Strategies", f'<p class="muted">No saved strategies in {_e(strategies_dir)}.</p>'))
    if bt_rows:
        shown = bt_rows[:MAX_BACKTEST_ROWS]
        more = f'<p class="small muted">Showing the latest {MAX_BACKTEST_ROWS} of {len(bt_rows):,}.</p>' if len(bt_rows) > MAX_BACKTEST_ROWS else ""
        parts.append(_section("backtests", "Recent backtests", _table(
            ["Idea", "Strategy", "Verdict", "Sharpe", "CAGR", "Max drawdown", "Period", "Run date"],
            [r for _, r in shown], num_cols=[3, 4, 5]) + more, intro="Every saved backtest, newest first."))
    else:
        parts.append(_section("backtests", "Recent backtests", f'<p class="muted">No backtest results found in {_e(runs_dir)}.</p>'))
    if screen_rows:
        parts.append(_section("screens", "Screening runs", _table(
            ["Observation", "As of", "Universe", "Survivors", "Ideas", "Data"], [r for _, r in screen_rows], num_cols=[2, 3, 4])))
    parts.append(ideas_html)
    if scan.notes:
        parts.append(_section("skipped", "Skipped files", "<ul>" + "".join(f"<li>{_e(n)}</li>" for n in scan.notes) + "</ul>",
                              intro="These files could not be read or rendered and were left out."))
    parts.append(f'<p class="small muted">Sources: runs {_e(runs_dir)}; strategies {_e(strategies_dir)}; inbox {_e(inbox_path)}.</p>')
    index = out_dir / "index.html"
    _write_text_atomic(index, html_page("AItrading dashboard", "".join(parts), subtitle="Strategies, backtests and ideas"))
    return index


def open_in_browser(path: Path | str) -> bool:
    """Open a local file in the default browser; returns False (never raises) if that fails."""
    try:
        uri = Path(path).expanduser().resolve().as_uri()
        return bool(webbrowser.open(uri))
    except Exception:  # noqa: BLE001 - headless machines, missing browsers, odd paths
        return False
