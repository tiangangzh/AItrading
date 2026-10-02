"""Self-contained HTML reports that SHOW THE IDEA: what a strategy does, why, how it performed.

Pages (each a single HTML document - inline CSS and inline SVG charts, no external assets, no
JavaScript required, light / dark theme via ``prefers-color-scheme``):

* :func:`render_backtest_html` - one backtest: the idea in plain English, verdict + verified
  interpretation, key numbers, growth of $1, drawdown, quantile spread, factor exposures, factor
  construction checks, full statistics, latest holdings, data provenance, warnings, LLM audit.
* :func:`render_pipeline_html` - a screening run (spec, funnel, candidates, per-idea thesis with
  grounding marks, quotes, audit).
* :func:`render_strategy_page` - a saved strategy: the idea, the simulated (paper) account's
  holdings, trade blotter and NAV against the backtest expectation, plus the backtest itself.
* :func:`render_ideas_inbox` - discovered ideas as cards with verified quotes, testability and the
  exact CLI command to try each one.

Every piece of text that comes from a model, a paper, a web page or the user is escaped with
:func:`html.escape`; links are only emitted for ``http(s)`` URLs. All results are simulated: the
platform never connects to a broker.
"""

from __future__ import annotations

import json
import math
import re
from datetime import date, datetime
from html import escape
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from pydantic import ValidationError

from aitrading.backtest.metrics import compound, drawdown_series
from aitrading.backtest.models import (
    BacktestResult,
    FactorConstructionCheck,
    FactorRegression,
    PerformanceStats,
    QuantileAnalysis,
)
from aitrading.core.models import EvidenceCheck, InvestmentIdea, LLMCallRecord, PipelineResult
from aitrading.discovery.models import IdeaCandidate, ReplicationReport
from aitrading.report import svg
from aitrading.screen.spec import Condition, ScreenSpec
from aitrading.strategy.spec import StrategySpec

__all__ = [
    "DISCLAIMER",
    "html_page",
    "describe_strategy",
    "template_info",
    "returns_frame",
    "render_backtest_html",
    "render_pipeline_html",
    "render_strategy_page",
    "render_ideas_inbox",
]

DISCLAIMER = "Research tool; simulated results; not investment advice."

# ------------------------------------------------------------------------------------------------
# Formatting helpers
# ------------------------------------------------------------------------------------------------


def _e(text: object) -> str:
    """Escape any value for HTML text or attribute context."""
    if text is None:
        return ""
    return escape(str(text), quote=True)


def _finite(v: object) -> float | None:
    if v is None or isinstance(v, bool):
        return None
    try:
        f = float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _pct(v: object, d: int = 1, *, signed: bool = False) -> str:
    """Format a value that is already in percent."""
    f = _finite(v)
    if f is None:
        return "n/a"
    return f"{f:+,.{d}f}%" if signed else f"{f:,.{d}f}%"


def _frac_pct(v: object, d: int = 1, *, signed: bool = False) -> str:
    """Format a fraction (0.05) as percent (5.0%)."""
    f = _finite(v)
    return "n/a" if f is None else _pct(f * 100.0, d, signed=signed)


def _num(v: object, d: int = 2) -> str:
    f = _finite(v)
    return "n/a" if f is None else f"{f:,.{d}f}"


def _int(v: object) -> str:
    f = _finite(v)
    return "n/a" if f is None else f"{int(round(f)):,}"


def _money(v: object, d: int = 2) -> str:
    f = _finite(v)
    if f is None:
        return "n/a"
    return f"-${abs(f):,.{d}f}" if f < 0 else f"${f:,.{d}f}"


def _dt(v: object) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, datetime):
        return v.strftime("%Y-%m-%d %H:%M")
    if isinstance(v, date):
        return v.isoformat()
    return str(v)


def _anchor(text: object) -> str:
    s = re.sub(r"[^A-Za-z0-9_-]+", "-", str(text)).strip("-")
    return s or "x"


def _safe_url(url: object) -> str | None:
    """Only http(s) URLs become links (never javascript:, data:, file: ...)."""
    if not isinstance(url, str):
        return None
    u = url.strip()
    try:
        parsed = urlparse(u)
    except ValueError:
        return None
    if parsed.scheme.lower() in ("http", "https") and parsed.netloc:
        return u
    return None


def _link(url: object, text: object) -> str:
    safe = _safe_url(url)
    if safe is None:
        return _e(text)
    return f'<a href="{_e(safe)}" rel="noopener noreferrer" target="_blank">{_e(text)}</a>'


_STATUS_ICON = {"good": "✓", "info": "↗", "warn": "!", "bad": "✗", "muted": "?"}


def _badge(label: str, tone: str = "muted", *, icon: str | None = None, title: str | None = None, big: bool = False) -> str:
    ic = _STATUS_ICON.get(tone, "") if icon is None else icon
    t = f' title="{_e(title)}"' if title else ""
    icon_html = f'<span aria-hidden="true">{_e(ic)}</span>' if ic else ""
    size = " big" if big else ""
    return f'<span class="badge {_e(tone)}{size}"{t}>{icon_html}{_e(label)}</span>'


def _mark(status: str | None, detail: str = "") -> str:
    """✓ / ✗ / ? mark for a verification status."""
    t = f' title="{_e(detail)}"' if detail else ""
    if status == "verified":
        return f'<span class="mark ok"{t} aria-label="verified">✓</span>'
    if status in ("mismatch", "not_found", "failed", "unverified"):
        return f'<span class="mark bad"{t} aria-label="{_e(status)}">✗</span>'
    return f'<span class="mark unk"{t} aria-label="not checked">?</span>'


def _table(headers: Sequence[str], rows: Iterable[Sequence[str]], *, num_cols: Iterable[int] = (), cls: str = "",
           caption: str | None = None) -> str:
    """Table from pre-escaped cell HTML. ``num_cols`` are right-aligned with tabular figures."""
    nums = set(num_cols)
    num_attr = ' class="num"'
    head = "".join(f'<th scope="col"{num_attr if i in nums else ""}>{_e(h)}</th>' for i, h in enumerate(headers))
    body = []
    for r in rows:
        cells = "".join(f'<td{num_attr if i in nums else ""}>{c}</td>' for i, c in enumerate(r))
        body.append(f"<tr>{cells}</tr>")
    cap = f"<caption>{_e(caption)}</caption>" if caption else ""
    c = f' class="{cls}"' if cls else ""
    return f'<div class="table-wrap"><table{c}>{cap}<thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


def _ul(items: Iterable[object], cls: str = "") -> str:
    lis = "".join(f"<li>{_e(i)}</li>" for i in items if i is not None and str(i).strip())
    if not lis:
        return ""
    c = f' class="{cls}"' if cls else ""
    return f"<ul{c}>{lis}</ul>"


def _section(anchor: str, title: str, body: str, *, intro: str | None = None) -> str:
    intro_html = f'<p class="intro">{_e(intro)}</p>' if intro else ""
    return f'<section class="card" id="{_e(anchor)}"><h2>{_e(title)}</h2>{intro_html}{body}</section>'


def _stat_card(label: str, value: str, sub: str = "", *, extra: str = "", tone: str = "") -> str:
    sub_html = f'<div class="sub">{sub}</div>' if sub else ""
    t = f" {tone}" if tone else ""
    return f'<div class="stat{t}"><div class="label">{_e(label)}</div><div class="value">{value}</div>{sub_html}{extra}</div>'


def _chart(svg_text: str, caption: str | None = None) -> str:
    cap = f"<figcaption>{_e(caption)}</figcaption>" if caption else ""
    return f'<figure class="chart">{svg_text}{cap}</figure>'


# ------------------------------------------------------------------------------------------------
# Page shell (theme tokens on :root, dark mode via prefers-color-scheme and data-theme)
# ------------------------------------------------------------------------------------------------

_LIGHT = """
  color-scheme: light;
  --bg: #f9f9f7; --surface: #fcfcfb; --surface-2: #f1f0ec; --border: rgba(11,11,11,0.10); --border-strong: #c3c2b7;
  --text: #0b0b0b; --text-2: #52514e; --muted: #6b6a65; --link: #1c5cab; --accent: #2a78d6;
  --good-text: #006300; --good-bg: rgba(12,163,12,0.10); --info-text: #1c5cab; --info-bg: rgba(42,120,214,0.10);
  --warn-text: #8a5300; --warn-bg: rgba(250,178,25,0.16); --bad-text: #b42323; --bad-bg: rgba(208,59,59,0.10);
  --muted-bg: rgba(11,11,11,0.05);
  --chart-surface: #fcfcfb; --chart-text: #0b0b0b; --chart-text-2: #52514e; --chart-muted: #6b6a65;
  --chart-grid: #e1e0d9; --chart-axis: #c3c2b7; --neg: #e34948;
  --series-1: #2a78d6; --series-2: #eb6834; --series-3: #1baf7a; --series-4: #eda100;
  --series-5: #e87ba4; --series-6: #008300; --series-7: #4a3aa7; --series-8: #e34948;
"""

_DARK = """
  color-scheme: dark;
  --bg: #0d0d0d; --surface: #1a1a19; --surface-2: #242422; --border: rgba(255,255,255,0.10); --border-strong: #4a4a46;
  --text: #ffffff; --text-2: #c3c2b7; --muted: #a3a29b; --link: #86b6ef; --accent: #3987e5;
  --good-text: #3fc43f; --good-bg: rgba(12,163,12,0.16); --info-text: #86b6ef; --info-bg: rgba(57,135,229,0.16);
  --warn-text: #fab219; --warn-bg: rgba(250,178,25,0.14); --bad-text: #ff8a8a; --bad-bg: rgba(230,103,103,0.16);
  --muted-bg: rgba(255,255,255,0.06);
  --chart-surface: #1a1a19; --chart-text: #ffffff; --chart-text-2: #c3c2b7; --chart-muted: #898781;
  --chart-grid: #2c2c2a; --chart-axis: #4a4a46; --neg: #e66767;
  --series-1: #3987e5; --series-2: #d95926; --series-3: #199e70; --series-4: #c98500;
  --series-5: #d55181; --series-6: #008300; --series-7: #9085e9; --series-8: #e66767;
"""

_CSS = (
    ":root {" + _LIGHT + "}\n"
    "@media (prefers-color-scheme: dark) { :root:not([data-theme=\"light\"]) {" + _DARK + "} }\n"
    ":root[data-theme=\"dark\"] {" + _DARK + "}\n"
    """
* { box-sizing: border-box; }
html { -webkit-text-size-adjust: 100%; }
body { margin: 0; background: var(--bg); color: var(--text);
  font: 15px/1.55 system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; }
.wrap { max-width: 1100px; margin: 0 auto; padding: 20px 16px 40px; }
a { color: var(--link); }
h1 { font-size: 26px; line-height: 1.25; margin: 4px 0 10px; overflow-wrap: anywhere; }
h2 { font-size: 19px; line-height: 1.3; margin: 0 0 10px; }
h3 { font-size: 16px; margin: 18px 0 8px; }
h4 { font-size: 14px; margin: 14px 0 6px; color: var(--text-2); }
p { margin: 8px 0; }
.eyebrow { font-size: 12px; text-transform: uppercase; letter-spacing: .06em; color: var(--muted); }
.lead { font-size: 17px; }
.intro { color: var(--text-2); font-size: 14px; margin-top: -4px; }
.meta { display: flex; flex-wrap: wrap; gap: 4px 18px; color: var(--text-2); font-size: 13px; margin: 6px 0; }
.meta b { color: var(--text); font-weight: 600; }
.back { font-size: 13px; }
header.top { padding: 4px 0 6px; }
.card { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 20px; margin: 16px 0;
  overflow-wrap: anywhere; }
nav.toc { display: flex; flex-wrap: wrap; gap: 4px 14px; font-size: 13px; margin: 10px 0 0; }
.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin: 4px 0; }
.stat { border: 1px solid var(--border); border-radius: 10px; padding: 12px 14px; background: var(--surface); min-width: 0; }
.stat .label { font-size: 12px; color: var(--text-2); }
.stat .value { font-size: 24px; font-weight: 600; margin-top: 2px; line-height: 1.2; }
.stat .sub { font-size: 12px; color: var(--muted); margin-top: 2px; }
.stat.good .value { color: var(--good-text); }
.stat.bad .value { color: var(--bad-text); }
.badge { display: inline-flex; align-items: center; gap: 5px; padding: 2px 10px; border-radius: 999px; font-size: 12px;
  font-weight: 600; line-height: 1.6; white-space: nowrap; background: var(--muted-bg); color: var(--text-2); }
.badge.good { color: var(--good-text); background: var(--good-bg); }
.badge.info { color: var(--info-text); background: var(--info-bg); }
.badge.warn { color: var(--warn-text); background: var(--warn-bg); }
.badge.bad { color: var(--bad-text); background: var(--bad-bg); }
.badge.big { font-size: 15px; padding: 4px 14px; }
.badges { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; margin: 6px 0; }
.mark { font-weight: 700; display: inline-block; min-width: 1.1em; text-align: center; }
.mark.ok { color: var(--good-text); }
.mark.bad { color: var(--bad-text); }
.mark.unk { color: var(--muted); }
.callout { border-radius: 10px; padding: 12px 16px; margin: 16px 0; background: var(--warn-bg); border: 1px solid var(--border); }
.callout.bad { background: var(--bad-bg); }
.callout h2 { font-size: 16px; margin: 0 0 6px; }
.callout ul { margin: 6px 0 0; padding-left: 20px; }
.callout li.bad { color: var(--bad-text); font-weight: 600; }
.table-wrap { overflow-x: auto; -webkit-overflow-scrolling: touch; margin: 8px 0; }
table { border-collapse: collapse; width: 100%; font-size: 13px; }
caption { text-align: left; font-weight: 600; padding: 4px 0 6px; color: var(--text-2); }
th, td { padding: 6px 10px; border-bottom: 1px solid var(--border); text-align: left; vertical-align: top; }
th { font-weight: 600; color: var(--text-2); background: var(--surface-2); }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
tr.total td { font-weight: 600; }
figure.chart { margin: 12px 0; overflow-x: auto; }
figure.chart svg { min-width: 480px; }
figure.chart figcaption { font-size: 12px; color: var(--muted); margin-top: 4px; }
.chart .hz:hover, .chart .hit:hover { fill: var(--text); fill-opacity: .06; }
.two-col { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(300px, 100%), 1fr)); gap: 16px; }
.two-col > * { min-width: 0; }
blockquote { margin: 8px 0; padding: 8px 12px; border-left: 3px solid var(--border-strong); background: var(--surface-2);
  border-radius: 6px; }
blockquote .src { display: block; font-size: 12px; color: var(--muted); margin-top: 4px; }
code, pre, .cmd { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 13px; }
pre { white-space: pre-wrap; overflow-wrap: anywhere; background: var(--surface-2); padding: 10px; border-radius: 8px; }
.cmd { display: inline-block; background: var(--surface-2); border: 1px solid var(--border); padding: 4px 10px; border-radius: 6px;
  user-select: all; overflow-wrap: anywhere; }
.wbar { display: inline-block; height: 8px; border-radius: 2px; vertical-align: middle; margin-left: 6px; background: var(--series-1); }
.wbar.neg { background: var(--neg); }
.meter { display: inline-block; width: 80px; height: 8px; border-radius: 4px; background: var(--muted-bg); vertical-align: middle; overflow: hidden; }
.meter > span { display: block; height: 100%; background: var(--series-1); border-radius: 4px; }
.idea-card { border: 1px solid var(--border); border-radius: 12px; padding: 16px 18px; margin: 14px 0; background: var(--surface); }
.idea-card h3 { margin: 0 0 4px; font-size: 18px; }
.kv { display: grid; grid-template-columns: max-content 1fr; gap: 4px 14px; font-size: 14px; margin: 8px 0; }
.kv dt { color: var(--text-2); }
.kv dd { margin: 0; }
details { margin: 10px 0; }
summary { cursor: pointer; color: var(--text-2); font-weight: 600; }
.muted { color: var(--muted); }
.small { font-size: 12px; }
footer.disclaimer { margin-top: 28px; padding-top: 12px; border-top: 1px solid var(--border); color: var(--muted); font-size: 13px; }
@media (max-width: 640px) {
  .card { padding: 14px; }
  .idea-card { padding: 12px; }
  h1 { font-size: 21px; }
  .stat .value { font-size: 20px; }
  .kv { grid-template-columns: 1fr; }
}
@media print { .card, .idea-card { break-inside: avoid; } body { background: #fff; } }
"""
)


def html_page(title: str, body: str, *, home_href: str | None = None, subtitle: str | None = None) -> str:
    """Wrap ``body`` (pre-escaped HTML) into a complete, self-contained page with the shared theme."""
    back = f'<p class="back"><a href="{_e(home_href)}">← All reports</a></p>' if home_href else ""
    sub = f'<meta name="description" content="{_e(subtitle)}">' if subtitle else ""
    return (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{_e(title)}</title>\n{sub}\n<style>{_CSS}</style>\n</head>\n"
        f'<body>\n<main class="wrap">\n{back}{body}\n'
        f'<footer class="disclaimer"><strong>{_e(DISCLAIMER)}</strong> Simulated (paper) trading only - '
        "no broker connection; past and backtested performance does not predict future returns.</footer>\n"
        "</main>\n</body>\n</html>\n"
    )


# ------------------------------------------------------------------------------------------------
# The idea in plain English
# ------------------------------------------------------------------------------------------------

_REBAL_PHRASE = {
    "daily": "Every trading day",
    "weekly": "Each week",
    "monthly": "Each month",
    "quarterly": "Each quarter",
    "annual": "Once a year",
}
_REBAL_ADJ = {"daily": "daily", "weekly": "weekly", "monthly": "monthly", "quarterly": "quarterly", "annual": "annually"}

_FEATURE_LABELS = {
    "return_12m_ex_1m_pct": "12-1 momentum (return from 12 months to 1 month ago)",
    "return_1m_pct": "last month's return",
    "return_3m_pct": "3-month return",
    "return_6m_pct": "6-month return",
    "return_12m_pct": "12-month return",
    "volatility_20d_pct": "20-day volatility",
    "volatility_60d_pct": "60-day volatility",
    "beta_1y": "1-year beta",
    "fcf_yield_pct": "free-cash-flow yield",
    "earnings_yield_ntm_pct": "forward (NTM) earnings yield",
    "ev_to_ebitda": "EV/EBITDA",
    "ev_to_sales": "EV/sales",
    "pe_ntm": "forward P/E",
    "roe_pct": "return on equity",
    "gross_margin_pct": "gross margin",
    "operating_margin_pct": "operating margin",
    "fcf_conversion_pct": "FCF conversion",
    "net_debt_to_ebitda": "net debt / EBITDA",
    "market_cap_usd_bn": "market cap",
    "eps_revision_3m_pct": "3-month EPS estimate revisions",
    "short_interest_pct_float": "short interest (% of float)",
    "rsi_14": "14-day RSI",
    "price_vs_sma_200_pct": "price vs its 200-day moving average",
    "price_vs_sma_50_pct": "price vs its 50-day moving average",
    "sma_50_vs_sma_200_pct": "50-day vs 200-day moving average",
    "drawdown_from_52w_high_pct": "distance from the 52-week high",
    "max_volume_ratio_20d": "peak volume vs normal (20 days)",
    "revenue_growth_yoy_pct": "revenue growth (YoY)",
    "avg_dollar_volume_20d_usd_mn": "20-day average dollar volume",
}
_ACRONYMS = {"sma": "SMA", "ntm": "NTM", "yoy": "YoY", "eps": "EPS", "ev": "EV", "fcf": "FCF", "roe": "ROE", "rsi": "RSI",
             "iv": "IV", "macd": "MACD", "atr": "ATR", "pe": "P/E", "usd": "USD", "ebitda": "EBITDA", "oi": "OI", "gics": "GICS"}
_SPANS = {"d": "day", "w": "week", "m": "month", "q": "quarter", "y": "year"}

_FACTOR_MODELS = {
    "capm": ("CAPM (single-factor market model)", ["Mkt-RF"]),
    "ff3": ("Fama-French 3-factor model", ["Mkt-RF", "SMB", "HML"]),
    "carhart4": ("Carhart 4-factor model", ["Mkt-RF", "SMB", "HML", "Mom"]),
    "ff5": ("Fama-French 5-factor model", ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]),
}
_FACTOR_MEANING = {
    "Mkt-RF": "the market minus T-bills",
    "SMB": "small minus big - size",
    "HML": "cheap minus expensive on book-to-market - value",
    "Mom": "past winners minus past losers - momentum",
    "RMW": "robust minus weak profitability",
    "CMA": "conservative minus aggressive investment",
}
_BUCKET_NAMES = {2: "half", 3: "tercile", 4: "quartile", 5: "quintile", 10: "decile"}
_WEIGHTING = {
    "equal": "equal weighted",
    "value": "weighted by market cap",
    "signal": "weighted by signal strength",
    "inverse_vol": "weighted by inverse volatility",
}
_OPS = {">": "above", ">=": "at least", "<": "below", "<=": "at most", "==": "equal to", "!=": "not equal to"}


def _feature_label(name: str) -> str:
    if name in _FEATURE_LABELS:
        return _FEATURE_LABELS[name]
    base = re.sub(r"_(pct|pp|usd_bn|usd_mn|bn|mn)$", "", name)
    words = []
    for tok in base.split("_"):
        m = re.fullmatch(r"(\d+)([dwmqy])", tok)
        if m:
            words.append(f"{m.group(1)}-{_SPANS[m.group(2)]}")
        else:
            words.append(_ACRONYMS.get(tok.lower(), tok))
    return " ".join(words) or name


def _fmt_threshold(feature: str, v: float | None) -> str:
    if v is None:
        return "?"
    if feature.endswith("_pct") or feature.endswith("_pp"):
        return f"{v:g}%" if feature.endswith("_pct") else f"{v:g} pp"
    if feature.endswith("_usd_bn"):
        return f"${v:g}bn"
    if feature.endswith("_usd_mn"):
        return f"${v:g}mn"
    if feature in ("price",):
        return f"${v:g}"
    if feature.endswith("_ratio") or feature.endswith("ratio_20d") or feature in ("beta_1y", "ev_to_ebitda", "ev_to_sales", "pe_ntm"):
        return f"{v:g}x"
    return f"{v:g}"


def _describe_condition(c: Condition) -> str:
    label = _feature_label(c.feature)
    if c.op == "between":
        text = f"{label} between {_fmt_threshold(c.feature, c.value)} and {_fmt_threshold(c.feature, c.value_high)}"
    elif c.op in ("in", "not_in"):
        vals = ", ".join(c.values or [])
        text = f"{label} {'in' if c.op == 'in' else 'not in'} [{vals}]"
    elif c.other_feature:
        rhs = _feature_label(c.other_feature)
        if c.multiplier != 1.0:
            rhs = f"{c.multiplier:g} x {rhs}"
        text = f"{label} {_OPS.get(c.op, c.op)} {rhs}"
    else:
        text = f"{label} {_OPS.get(c.op, c.op)} {_fmt_threshold(c.feature, c.value)}"
    if c.rationale:
        text += f" ({c.rationale})"
    return text


def _universe_phrase(spec: StrategySpec) -> str:
    u = spec.universe
    country = (u.country or "").upper()
    where = "US" if country == "US" else country or "global"
    kinds = [t.replace("_", " ") for t in (u.security_types or [])]
    noun = "stocks" if kinds in ([], ["common stock"]) else " / ".join(kinds) + "s"
    return f"{where} {noun}"


def _universe_sentence(spec: StrategySpec) -> str:
    u = spec.universe
    bits = []
    if u.min_price is not None:
        bits.append(f"priced above ${u.min_price:g}")
    if u.min_avg_dollar_volume_usd_mn is not None:
        bits.append(f"trading at least ${u.min_avg_dollar_volume_usd_mn:g}mn a day")
    text = f"Universe: {_universe_phrase(spec)}"
    if bits:
        text += " " + " and ".join(bits)
    if u.exclude_sectors:
        text += f", excluding {', '.join(u.exclude_sectors)}"
    return text + "."


def _pct_of(n: int) -> str:
    v = 100.0 / n
    return f"{v:.0f}%" if abs(v - round(v)) < 1e-9 else f"{v:.1f}%"


def _selection_phrase(spec: StrategySpec) -> str:
    p = spec.portfolio
    ls = p.style == "long_short"
    if p.selection == "top_n" and p.top_n:
        text = f"buy the top {p.top_n} names" + (f" and short the bottom {p.top_n}" if ls else "")
    else:
        q = p.n_quantiles
        share = _pct_of(q)
        bucket = _BUCKET_NAMES.get(q, f"1/{q}")
        text = f"buy the top {share} ({bucket})" + (f" and short the bottom {share}" if ls else "")
    text += ", " + _WEIGHTING.get(p.weighting, p.weighting.replace("_", " "))
    if p.max_weight:
        text += f", capped at {p.max_weight * 100:g}% per name"
    return text


def _signal_phrase(spec: StrategySpec) -> str:
    sig = spec.signal
    if not sig:
        return "rank them (no ranking signal defined)"
    if len(sig) == 1:
        s = sig[0]
        direction = "higher is better" if s.direction == "higher_is_better" else "lower is better"
        text = f"rank them by {_feature_label(s.feature)} ({direction})"
    else:
        total = sum(s.weight for s in sig) or 1.0
        parts = []
        for s in sig:
            arrow = "higher" if s.direction == "higher_is_better" else "lower"
            parts.append(f"{_feature_label(s.feature)} ({arrow} is better, {s.weight / total * 100:.0f}%)")
        text = f"rank them on a composite of {len(sig)} signals: " + ", ".join(parts)
    notes = []
    if any(s.sector_neutral for s in sig):
        notes.append("within each sector")
    if any(s.transform == "zscore" for s in sig):
        notes.append("using z-scores")
    if notes:
        text += " " + " and ".join(notes)
    return text


def _coerce_spec(spec: object) -> StrategySpec | None:
    if isinstance(spec, StrategySpec):
        return spec
    if isinstance(spec, Mapping):
        try:
            return StrategySpec.model_validate(dict(spec))
        except ValidationError:
            return None
    return None


def describe_strategy(spec: StrategySpec | Mapping[str, Any] | None) -> list[str]:
    """Plain-English sentences describing what the strategy does (first sentence = the core rule).

    Example (12-1 momentum): "Each month, take US stocks and rank them by 12-1 momentum (return from
    12 months to 1 month ago) (higher is better); buy the top 10% (decile) and short the bottom 10%,
    equal weighted." followed by costs, universe, benchmark and attribution sentences.
    """
    s = _coerce_spec(spec)
    if s is None:
        if isinstance(spec, Mapping) and spec:
            kind = spec.get("kind", "unknown")
            reb = spec.get("rebalance", "unknown")
            costs = spec.get("costs_bps")
            out = [f"A {str(kind).replace('_', '-')} strategy rebalanced {reb} (the saved specification could not be fully read)."]
            if _finite(costs) is not None:
                out.append(f"Trading costs: {float(costs):g} bps one-way, charged on turnover.")
            return out
        return ["No strategy specification was saved with this result."]

    when = _REBAL_PHRASE.get(s.rebalance, f"Every {s.rebalance} period")
    universe = _universe_phrase(s)
    out: list[str] = []
    if s.kind == "cross_sectional":
        out.append(f"{when}, take {universe} and {_signal_phrase(s)}; {_selection_phrase(s)}.")
        if s.filters:
            out.append("Only stocks that pass these filters are ranked: " + "; ".join(_describe_condition(c) for c in s.filters) + ".")
    elif s.kind == "screen":
        n = len(s.filters)
        weighting = _WEIGHTING.get(s.portfolio.weighting, s.portfolio.weighting)
        if s.portfolio.max_weight:
            weighting += f", capped at {s.portfolio.max_weight * 100:g}% per name"
        cond_word = "condition" if n == 1 else f"{n} conditions"
        out.append(f"{when}, hold every one of the {universe} that passes {'the' if n == 1 else 'all'} {cond_word} below, "
                   f"{weighting}; when nothing passes, the strategy sits in cash.")
        for i, c in enumerate(s.filters, 1):
            out.append(f"Condition {i}: {_describe_condition(c)}.")
    elif s.kind == "factor_model":
        title, factors = _FACTOR_MODELS.get(s.factor_model or "", (str(s.factor_model or "factor model"), []))
        fac = ", ".join(f"{f} ({_FACTOR_MEANING.get(f, f)})" for f in factors)
        out.append(
            f"Build the {title} from {universe}" + (f": {fac}" if fac else "")
            + " - using the standard Fama-French portfolio sorts - and compare each constructed factor with the official "
            f"Kenneth French series. Portfolios are re-formed {_REBAL_ADJ.get(s.rebalance, s.rebalance)}."
        )
    elif s.kind == "time_series":
        ts = s.time_series
        if ts is None or not ts.assets:
            out.append(f"{when}, apply a timing rule (no assets or entry conditions were defined).")
        else:
            assets = ", ".join(ts.assets)
            entry = " and ".join(_describe_condition(c) for c in ts.entry) or "the entry rule holds"
            flat = "go short" if ts.when_flat == "short" else "move to cash"
            check = {"daily": "At every close", "weekly": "At each week's close", "monthly": "At each month-end",
                     "quarterly": "At each quarter-end", "annual": "At each year-end"}.get(s.rebalance, when)
            out.append(f"{check}, hold {assets} (long) while {entry}; otherwise {flat}.")
            if ts.exit:
                out.append("Exit early when " + " and ".join(_describe_condition(c) for c in ts.exit) + ".")
            if len(ts.assets) > 1:
                out.append("Each asset is timed independently with its own signal.")
    else:  # pragma: no cover - Literal guards this
        out.append(f"A {s.kind} strategy rebalanced {s.rebalance}.")

    if s.kind != "factor_model":
        out.append(f"Trading costs: {s.costs_bps:g} bps one-way, charged on every trade (turnover).")
        if s.delisting_return:
            out.append(f"A holding that is delisted is booked at {s.delisting_return * 100:+g}% (delisting penalty).")
        else:
            out.append("Delisted holdings are closed at their last price (no delisting penalty - optimistic).")
    if s.kind in ("cross_sectional", "screen", "factor_model"):
        out.append(_universe_sentence(s))
    out.append(f"Benchmark: {s.benchmark or 'the data provider default broad US index'}.")
    if s.attribution_model and s.kind != "factor_model":
        title = _FACTOR_MODELS.get(s.attribution_model, (s.attribution_model, []))[0]
        out.append(f"Returns are attributed with the {title}: alpha is what the known factors cannot explain.")
    if s.start or s.end:
        out.append(f"Test period: {s.start.isoformat() if s.start else 'earliest data'} to {s.end.isoformat() if s.end else 'latest data'}.")
    return out


def template_info(key: str | None) -> tuple[str, str, list[str]] | None:
    """(title, description, references) of the built-in idea template ``key``, if there is one."""
    if not key:
        return None
    try:
        from aitrading.strategy.library import TEMPLATES
    except Exception:  # pragma: no cover - library is part of the package
        return None
    t = TEMPLATES.get(str(key))
    if t is None:
        return None
    return t.title, t.description, list(t.references)


def _idea_section(spec_obj: object, *, template_title: str | None, template_description: str | None,
                  references: Sequence[str] | None, idea_text: str | None = None) -> str:
    sentences = describe_strategy(spec_obj)  # type: ignore[arg-type]
    s = _coerce_spec(spec_obj)
    parts: list[str] = []
    if template_title:
        parts.append(f'<p class="eyebrow">Based on the template: {_e(template_title)}</p>')
    if template_description:
        parts.append(f'<p class="lead">{_e(template_description)}</p>')
        parts.append("<h3>How the backtest trades it</h3>")
        parts.append(f"<p>{_e(sentences[0])}</p>")
    else:
        parts.append(f'<p class="lead">{_e(sentences[0])}</p>')
    if len(sentences) > 1:
        parts.append(_ul(sentences[1:]))
    if s is not None and s.unsupported_requests:
        parts.append('<div class="callout bad"><h2>Not testable as asked</h2>'
                     "<p>These parts of the idea could not be expressed with the available data and were left out:</p>"
                     + _ul(s.unsupported_requests) + "</div>")
    if s is not None and s.assumptions:
        parts.append("<h3>Methodology notes</h3>" + _ul(s.assumptions))
    if references:
        parts.append("<h3>References</h3>" + _ul(references))
    if idea_text and s is not None and s.idea and s.idea != idea_text:
        parts.append(f'<p class="muted small">Original idea text: {_e(s.idea)}</p>')
    return _section("idea", "The idea in plain English", "".join(parts))


# ------------------------------------------------------------------------------------------------
# Backtest result helpers
# ------------------------------------------------------------------------------------------------

_SERIES_LABELS = {"strategy": "Strategy", "benchmark": "Benchmark", "long": "Long leg", "short": "Short leg"}
_SERIES_SLOTS = {"strategy": 1, "benchmark": 2, "long": 3, "short": 5}


def _series_label(key: str) -> str:
    return _SERIES_LABELS.get(key, key)


def returns_frame(result: BacktestResult) -> pd.DataFrame:
    """Periodic returns of every series as a DataFrame on a DatetimeIndex (None -> NaN).

    Series longer or shorter than ``result.dates`` are truncated / padded with NaN so a malformed
    result still renders.
    """
    idx = pd.DatetimeIndex(pd.to_datetime([pd.Timestamp(d) for d in result.dates])) if result.dates else pd.DatetimeIndex([])
    n = len(idx)
    data: dict[str, np.ndarray] = {}
    for key, vals in result.returns.items():
        arr = np.full(n, np.nan)
        for i, v in enumerate(list(vals)[:n]):
            f = _finite(v)
            if f is not None:
                arr[i] = f
        data[str(key)] = arr
    df = pd.DataFrame(data, index=idx)
    if n:
        df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


def _wealth(r: pd.Series) -> pd.Series:
    """Growth of $1; NaN returns stay NaN (gaps), leading NaNs mean 'not started yet'."""
    w = (1.0 + r.fillna(0.0)).cumprod()
    w[r.isna()] = np.nan
    return w


def _primary_key(result: BacktestResult) -> str | None:
    if "strategy" in result.stats or "strategy" in result.returns:
        return "strategy"
    if result.stats:
        return next(iter(result.stats))
    return next(iter(result.returns), None)


def _growth_section(result: BacktestResult, frame: pd.DataFrame) -> str:
    keys = [k for k in ("strategy", "benchmark", "long", "short") if k in frame.columns and frame[k].notna().any()]
    if not keys:
        keys = [k for k in frame.columns if frame[k].notna().any()][:8]
    if not keys:
        return _section("growth", "Growth of $1", '<p class="muted">No return series were saved with this result.</p>')
    series: dict[str, pd.Series] = {}
    colors: dict[str, int] = {}
    next_slot = iter([s for s in range(1, 9) if s not in _SERIES_SLOTS.values()])
    for k in keys:
        w = _wealth(frame[k])
        first = w.first_valid_index()
        if first is not None and first == frame.index[0] and result.start < first.date():
            start_ts = pd.Timestamp(result.start)
            w = pd.concat([pd.Series([1.0], index=[start_ts]), w])
        label = _series_label(k)
        series[label] = w
        colors[label] = _SERIES_SLOTS.get(k) or next(next_slot, 8)
    vals = pd.concat(series.values()).to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    # svg.line_chart falls back to a linear axis when any value is <= 0 (e.g. a -100% period), so
    # only ask for (and title the chart with) a log scale when it will really be drawn.
    use_log = bool(vals.size) and bool((vals > 0).all()) and float(vals.max()) / float(vals.min()) > 5.0
    chart = svg.line_chart(
        series, title="Growth of $1 (after costs)" + (" - log scale" if use_log else ""), y_label="Value of $1",
        log_scale=use_log, y_format="${:,.2f}", colors=colors, reference=1.0,
    )
    legs = " The long and short legs are shown separately: a long-short strategy earns long minus short." \
        if "long" in keys and "short" in keys else ""
    intro = "What $1 invested at the start would have grown to, after trading costs." + legs
    return _section("growth", "Growth of $1", _chart(chart) + _calendar_table(frame, keys), intro=intro)


def _calendar_table(frame: pd.DataFrame, keys: Sequence[str]) -> str:
    if frame.empty:
        return ""
    try:
        annual = compound(frame[list(keys)], "A")
    except (ValueError, TypeError):
        return ""
    if annual.empty:
        return ""
    first, last = frame.index.min(), frame.index.max()
    rows = []
    for ts, row in annual.sort_index(ascending=False).iterrows():
        year = ts.year
        partial = (year == first.year and (first.month, first.day) > (1, 31)) or (year == last.year and last.month < 12)
        label = f"{year}" + (" (partial)" if partial else "")
        rows.append([_e(label)] + [_frac_pct(row[k], 1) for k in keys])
    table = _table(["Year"] + [_series_label(k) for k in keys], rows, num_cols=range(1, len(keys) + 1))
    return f"<details><summary>Calendar-year returns (table view)</summary>{table}</details>"


def _drawdown_section(frame: pd.DataFrame, key: str | None, stats: PerformanceStats | None) -> str:
    if key is None or key not in frame.columns or not frame[key].notna().any():
        return ""
    r = frame[key]
    first = r.first_valid_index()
    dd = drawdown_series(r.loc[first:])
    chart = svg.area_drawdown_chart(dd, title=f"{_series_label(key)} drawdown (month-end values)")
    extra = ""
    if stats is not None:
        unit = {252: "trading days", 52: "weeks", 12: "months", 4: "quarters", 1: "years"}.get(round(stats.periods_per_year), "periods")
        extra = (f"<p>Worst peak-to-trough loss on daily data: <b>{_pct(stats.max_drawdown_pct)}</b> (the chart uses month-end values, "
                 f"so intra-month troughs look shallower); longest time under water: "
                 f"<b>{_int(stats.max_drawdown_duration_periods)}</b> {unit}.</p>")
    return _section("drawdown", "Drawdown", _chart(chart) + extra,
                    intro="How far the strategy fell below its previous peak - the losses you would have had to sit through.")


def _quantile_section(q: QuantileAnalysis | None) -> str:
    if q is None:
        return ""
    n = len(q.annual_return_by_quantile_pct)
    labels = [f"Q{i}" + (" worst" if i == 1 else " best" if i == n else "") for i in range(1, n + 1)]
    chart = svg.bar_chart(labels, q.annual_return_by_quantile_pct, title="Annual return by signal bucket", value_format="{:.1f}%",
                          y_label="% per year")
    mono_tone = "good" if q.monotonicity >= 0.8 else "warn" if q.monotonicity >= 0.3 else "bad"
    ic_tone = "good" if abs(q.ic_t_stat) >= 3 else "warn" if abs(q.ic_t_stat) >= 2 else "bad"
    cards = "".join([
        _stat_card("Spread (best - worst)", _pct(q.spread_annual_pct, signed=True), "per year"),
        _stat_card("Monotonicity", _num(q.monotonicity), "1 = returns rise bucket by bucket", tone=mono_tone),
        _stat_card("Mean rank IC", _num(q.ic_mean, 3), "signal vs next-period return"),
        _stat_card("IC t-stat", _num(q.ic_t_stat), "|t| > 2 weak, > 3 strong", tone=ic_tone),
        _stat_card("IC hit rate", _pct(q.ic_hit_rate_pct), "periods with a positive IC"),
    ])
    table = _table(["Bucket", "Annual return"], [[_e(lab), _pct(v)] for lab, v in zip(labels, q.annual_return_by_quantile_pct)], num_cols=[1])
    body = _chart(chart) + f'<div class="cards">{cards}</div>' + f"<details><summary>Table view</summary>{table}</details>"
    intro = (f"Each period, stocks are sorted into {q.n_quantiles} buckets by the signal. If the idea works, returns rise "
             f"steadily from Q1 (worst signal) to Q{n} (best signal).")
    return _section("quantiles", "Does the signal sort returns?", body, intro=intro)


def _regression_section(reg: FactorRegression | None) -> str:
    if reg is None:
        return ""
    names = list(reg.betas)
    betas = [reg.betas[k] for k in names]
    tstats = [reg.beta_t_stats.get(k) for k in names]
    errs = []
    for b, t in zip(betas, tstats):
        ft = _finite(t)
        errs.append(1.96 * abs(b / ft) if ft not in (None, 0.0) else None)
    notes = [f"t={_num(t, 1)}" if _finite(t) is not None else None for t in tstats]
    model_title = _FACTOR_MODELS.get(reg.model, (reg.model, []))[0]
    chart = svg.horizontal_bar_chart(names, betas, errs, annotations=notes, title=f"Factor betas ({model_title})",
                                     value_format="{:.2f}") if names else ""
    t = abs(reg.alpha_t_stat)
    tone = "good" if t >= 3 else "warn" if t >= 2 else "bad"
    verdict = ("strong evidence (|t| >= 3)" if t >= 3 else "weak evidence (2 <= |t| < 3)" if t >= 2
               else "not statistically distinguishable from zero (|t| < 2)")
    cards = "".join([
        _stat_card("Alpha", _pct(reg.alpha_annual_pct, 2, signed=True), "per year, after factor exposures"),
        _stat_card("Alpha t-stat", _num(reg.alpha_t_stat), "Newey-West", tone=tone),
        _stat_card("R-squared", _num(reg.r_squared), "share of variance explained by the factors"),
        _stat_card("Observations", _int(reg.n), _e(reg.factor_source)),
    ])
    rows = [["<b>Alpha (annual)</b>", _pct(reg.alpha_annual_pct, 2, signed=True), _num(reg.alpha_t_stat)]]
    for k, b, tt in zip(names, betas, tstats):
        rows.append([_e(k) + (f' <span class="muted small">{_e(_FACTOR_MEANING[k])}</span>' if k in _FACTOR_MEANING else ""),
                     _num(b), _num(tt)])
    table = _table(["Term", "Estimate", "t-stat"], rows, num_cols=[1, 2])
    body = (f'<div class="cards">{cards}</div>'
            f"<p>Alpha is <b>{_e(verdict)}</b>. A t-stat above 3 is the bar for a new factor (Harvey, Liu &amp; Zhu 2016); "
            "betas show how much of the return simply comes from known factors.</p>"
            + (_chart(chart, "Bars: factor betas; whiskers: 95% confidence interval.") if chart else "") + table)
    return _section("factors", "Factor exposure and alpha", body,
                    intro=f"Regression of the strategy's returns on the {model_title} factors ({reg.factor_source}).")


def _factor_checks_section(checks: Sequence[FactorConstructionCheck]) -> str:
    if not checks:
        return ""
    rows = []
    for c in checks:
        corr = _finite(c.correlation_with_official)
        if corr is None:
            mark = _mark(None, "no official series to compare")
        elif corr >= 0.7:
            mark = _mark("verified", "tracks the official factor closely (correlation >= 0.7)")
        elif corr >= 0.5:
            mark = '<span class="mark unk" title="moderate correlation (0.5-0.7)">~</span>'
        else:
            mark = _mark("mismatch", "weak correlation with the official factor (< 0.5)")
        rows.append([f"{mark} {_e(c.factor)}", _num(c.correlation_with_official), _pct(c.annual_premium_constructed_pct, 2),
                     _pct(c.annual_premium_official_pct, 2), _int(c.n_overlap_periods)])
    table = _table(["Factor", "Correlation with official", "Premium (constructed)", "Premium (official)", "Overlap periods"],
                   rows, num_cols=[1, 2, 3, 4])
    return _section("factor-checks", "Factor construction checks", table,
                    intro="How closely the factors built from this universe track the official Kenneth French series "
                          "(✓ correlation >= 0.7, ~ 0.5-0.7, ✗ below 0.5).")


_STAT_ROWS: list[tuple[str, Any]] = [
    ("Period", lambda s: f"{s.start.isoformat()} to {s.end.isoformat()}"),
    ("Periods (per year)", lambda s: f"{s.n_periods:,} ({s.periods_per_year:g}/yr)"),
    ("Total return", lambda s: _pct(s.total_return_pct)),
    ("CAGR", lambda s: _pct(s.cagr_pct, 2)),
    ("Volatility (annual)", lambda s: _pct(s.volatility_pct)),
    ("Sharpe", lambda s: _num(s.sharpe)),
    ("Sortino", lambda s: _num(s.sortino)),
    ("Max drawdown", lambda s: _pct(s.max_drawdown_pct)),
    ("Longest drawdown (periods)", lambda s: _int(s.max_drawdown_duration_periods)),
    ("Calmar", lambda s: _num(s.calmar)),
    ("Hit rate", lambda s: _pct(s.hit_rate_pct)),
    ("Best period", lambda s: _pct(s.best_period_pct, 2)),
    ("Worst period", lambda s: _pct(s.worst_period_pct, 2)),
    ("Skew", lambda s: _num(s.skew)),
    ("Excess kurtosis", lambda s: _num(s.excess_kurtosis)),
    ("Mean-return t-stat (Newey-West)", lambda s: _num(s.mean_return_t_stat)),
    ("Avg one-way turnover per rebalance", lambda s: _pct(s.avg_turnover_pct)),
    ("Beta to benchmark", lambda s: _num(s.beta_to_benchmark)),
    ("Tracking error", lambda s: _pct(s.tracking_error_pct)),
    ("Information ratio", lambda s: _num(s.information_ratio)),
]


def _stats_table(stats: Mapping[str, PerformanceStats]) -> str:
    keys = list(stats)
    rows = [[_e(label)] + [_e(fn(stats[k])) for k in keys] for label, fn in _STAT_ROWS]
    return _table(["Statistic"] + [_series_label(k) for k in keys], rows, num_cols=range(1, len(keys) + 1))


def _stats_section(stats: Mapping[str, PerformanceStats]) -> str:
    if not stats:
        return ""
    return _section("stats", "Full statistics", _stats_table(stats),
                    intro="Annualised statistics for every series (percent fields in %, ratios unitless).")


def _key_numbers(result: BacktestResult, frame: pd.DataFrame) -> str:
    key = _primary_key(result)
    st = result.stats.get(key) if key else None
    bench = result.stats.get("benchmark") if key != "benchmark" else None
    if st is None:
        return ""

    def vs(v: str) -> str:
        return f"Benchmark {v}" if bench is not None else ""

    spark = ""
    if key in frame.columns and frame[key].notna().any():
        spark = '<div class="spark">' + svg.sparkline(_wealth(frame[key]), title="Growth of $1", width=160, height=30) + "</div>"
    sharpe_tone = "good" if (st.sharpe or 0) >= 0.8 else "bad" if (st.sharpe is not None and st.sharpe < 0) else ""
    cards = [
        _stat_card("Total return", _pct(st.total_return_pct), _e(f"{st.start.isoformat()} to {st.end.isoformat()}"), extra=spark),
        _stat_card("CAGR", _pct(st.cagr_pct, 1, signed=True), _e(vs(_pct(bench.cagr_pct, 1, signed=True)) if bench else ""),
                   tone="bad" if st.cagr_pct < 0 else ""),
        _stat_card("Sharpe", _num(st.sharpe), _e(vs(_num(bench.sharpe)) if bench else ""), tone=sharpe_tone),
        _stat_card("Max drawdown", _pct(st.max_drawdown_pct), _e(vs(_pct(bench.max_drawdown_pct)) if bench else "")),
    ]
    reg = result.regression
    if reg is not None:
        t = abs(reg.alpha_t_stat)
        cards.append(_stat_card("Alpha t-stat", _num(reg.alpha_t_stat),
                                _e(f"alpha {_pct(reg.alpha_annual_pct, 1, signed=True)}/yr ({reg.model})"),
                                tone="good" if t >= 3 else "" if t >= 2 else "bad"))
    else:
        cards.append(_stat_card("Alpha t-stat", "n/a", "no factor regression"))
    cards.append(_stat_card("Volatility", _pct(st.volatility_pct), _e(f"turnover {_pct(st.avg_turnover_pct)} / rebalance")))
    return _section("key-numbers", "Key numbers", f'<div class="cards">{"".join(cards)}</div>')


_VERDICTS = {
    "robust": ("Robust", "good", "✓", "Strong, statistically significant evidence that survives the checks."),
    "promising": ("Promising", "info", "↗", "Encouraging but not conclusive - worth more testing."),
    "weak": ("Weak", "warn", "!", "Some signs of an effect, not enough evidence."),
    "likely_spurious": ("Likely spurious", "bad", "✗", "No reliable alpha and no orderly signal pattern."),
    "inconclusive": ("Inconclusive", "muted", "?", "Too little data (or missing statistics) to judge."),
}

_FAIL_WORDS = re.compile(r"unverified|mismatch|not[ _]found|failed|does not match|incorrect", re.I)


def _metric_checks(result: BacktestResult, metric_checks: Sequence[EvidenceCheck] | None) -> dict[str, EvidenceCheck]:
    interp = result.interpretation
    if interp is None:
        return {}
    checks: list[EvidenceCheck] = []
    if metric_checks is not None:
        checks = list(metric_checks)
    elif interp.cited_metrics:
        try:
            from aitrading.strategy.interpret import verify_interpretation

            checks = verify_interpretation(interp, result)
        except Exception:  # verification is best effort in a renderer
            checks = []
    out = {c.ref: c for c in checks}
    for cm in interp.cited_metrics:  # warnings that flag a citation override a missing / passing check
        for w in result.warnings:
            if cm.path in w and _FAIL_WORDS.search(w):
                out[cm.path] = EvidenceCheck(kind="quant", ref=cm.path, claim=f"{cm.path}={cm.value:g}", status="mismatch", detail=w)
    return out


def _verdict_section(result: BacktestResult, metric_checks: Sequence[EvidenceCheck] | None) -> str:
    interp = result.interpretation
    if interp is None:
        return _section("verdict", "Verdict", '<p class="muted">No interpretation was saved with this result.</p>')
    label, tone, icon, meaning = _VERDICTS.get(interp.verdict, (interp.verdict, "muted", "?", ""))
    checks = _metric_checks(result, metric_checks)
    big_badge = _badge(label, tone, icon=icon, title=meaning, big=True)
    parts = [f'<div class="badges">{big_badge}<span class="muted small">{_e(meaning)}</span></div>',
             f'<p class="lead">{_e(interp.summary)}</p>']
    if interp.key_findings:
        parts.append("<h3>Key findings</h3>" + _ul(interp.key_findings))
    if interp.biases_and_caveats:
        parts.append("<h3>Biases and caveats</h3>" + _ul(interp.biases_and_caveats))
    if interp.next_experiments:
        parts.append("<h3>Next experiments</h3>" + _ul(interp.next_experiments))
    if interp.cited_metrics:
        rows = []
        n_ok = 0
        for cm in interp.cited_metrics:
            c = checks.get(cm.path)
            status = c.status if c else None
            n_ok += status == "verified"
            rows.append([_mark(status, c.detail if c else "not checked"), f"<code>{_e(cm.path)}</code>", _num(cm.value, 4).rstrip("0").rstrip("."),
                         _e(cm.meaning)])
        checked = sum(1 for cm in interp.cited_metrics if cm.path in checks)
        summary = (f"{n_ok} of {len(interp.cited_metrics)} cited numbers verified against the results"
                   if checked else "Cited numbers (not verified)")
        parts.append(f"<h3>Numbers cited by the interpretation</h3><p class=\"small muted\">{_e(summary)}.</p>"
                     + _table(["", "Metric", "Value", "Meaning"], rows, num_cols=[2]))
    return _section("verdict", "Verdict", "".join(parts))


def _holdings_section(holdings: Mapping[str, float]) -> str:
    if not holdings:
        return ""
    clean = {str(k): float(v) for k, v in holdings.items() if _finite(v) is not None}
    longs = sorted(((k, v) for k, v in clean.items() if v > 0), key=lambda kv: -abs(kv[1]))
    shorts = sorted(((k, v) for k, v in clean.items() if v < 0), key=lambda kv: -abs(kv[1]))
    max_abs = max((abs(v) for v in clean.values()), default=0.0) or 1.0

    def table(items: list[tuple[str, float]], title: str, neg: bool) -> str:
        shown = items[:25]
        rows = []
        for i, (t, w) in enumerate(shown, 1):
            bar = f'<span class="wbar{" neg" if neg else ""}" style="width:{abs(w) / max_abs * 60:.0f}px"></span>'
            rows.append([str(i), _e(t), _frac_pct(w, 2) + bar])
        more = f'<p class="small muted">+ {len(items) - 25} more</p>' if len(items) > 25 else ""
        return "<div>" + _table(["#", "Ticker", "Weight"], rows, num_cols=[0, 2], caption=f"{title} ({len(items)})") + more + "</div>"

    gross = sum(abs(v) for v in clean.values())
    net = sum(clean.values())
    summary = (f'<div class="cards">{_stat_card("Long positions", _int(len(longs)))}{_stat_card("Short positions", _int(len(shorts)))}'
               f'{_stat_card("Gross exposure", _frac_pct(gross))}{_stat_card("Net exposure", _frac_pct(net, signed=True))}</div>')
    cols = table(longs, "Top long positions", False)
    if shorts:
        cols += table(shorts, "Top short positions", True)
    return _section("holdings", "Latest holdings", summary + f'<div class="two-col">{cols}</div>',
                    intro="What the strategy holds after its most recent rebalance (top 25 per side by absolute weight).")


def _data_usage_section(result: BacktestResult) -> str:
    if not result.data_usage:
        return ""
    rows = []
    for d in result.data_usage:
        pit = _mark("verified", "point-in-time") + " yes" if d.point_in_time else _mark("mismatch", "not point-in-time: look-ahead risk") + " no"
        rows.append([_e(d.dataset), _e(d.source), _e(d.coverage), pit, _e(d.notes)])
    return _section("data", "Data used (provenance)",
                    _table(["Dataset", "Source", "Coverage", "Point-in-time", "Notes"], rows),
                    intro="Where every input came from. Point-in-time data only uses what was known on each date (no look-ahead).")


def _llm_section(calls: Sequence[LLMCallRecord]) -> str:
    if not calls:
        return _section("llm-audit", "LLM call audit", '<p class="muted">No LLM calls were made (offline / heuristic mode).</p>')
    rows = []
    tot_in = tot_out = 0
    tot_lat = 0.0
    for c in calls:
        tot_in += c.input_tokens
        tot_out += c.output_tokens
        tot_lat += c.latency_s
        status = _mark("mismatch", c.error) + " " + _e(c.error) if c.error else (_badge("fallback", "warn") if c.served_by_fallback else "ok")
        rows.append([_e(c.purpose), _e(c.model), _int(c.input_tokens), _int(c.output_tokens), _int(c.cache_read_input_tokens),
                     _int(c.cache_creation_input_tokens), _num(c.latency_s, 1) + "s", _e(c.stop_reason or ""), _e(c.request_id or ""), status])
    rows.append(['<b>Total</b>', "", _int(tot_in), _int(tot_out), "", "", _num(tot_lat, 1) + "s", "", "", ""])
    return _section("llm-audit", "LLM call audit",
                    _table(["Purpose", "Model", "Input tok", "Output tok", "Cache read", "Cache write", "Latency", "Stop", "Request id", "Status"],
                           rows, num_cols=[2, 3, 4, 5, 6]))


_SEVERE = re.compile(r"survivorship|look-?ahead|delist|not point-in-time|stale|synthetic", re.I)


def _warnings_callout(warnings: Sequence[str], title: str = "Read this before trusting the numbers") -> str:
    if not warnings:
        return ""
    severe = [w for w in warnings if _SEVERE.search(w)]
    rest = [w for w in warnings if not _SEVERE.search(w)]
    items = "".join(f'<li class="bad">{_e(w)}</li>' for w in severe) + "".join(f"<li>{_e(w)}</li>" for w in rest)
    cls = "callout bad" if severe else "callout"
    return f'<section class="{cls}" id="warnings" role="note"><h2>⚠ {_e(title)}</h2><ul>{items}</ul></section>'


def _backtest_body(result: BacktestResult, *, template_title: str | None, template_description: str | None,
                   references: Sequence[str] | None, metric_checks: Sequence[EvidenceCheck] | None,
                   include_idea: bool = True, include_warnings: bool = True) -> tuple[str, list[tuple[str, str]]]:
    """Sections of a backtest report; returns (html, toc entries)."""
    frame = returns_frame(result)
    key = _primary_key(result)
    parts: list[tuple[str, str, str]] = []  # (anchor, toc label, html)
    if include_warnings:
        parts.append(("warnings", "Warnings", _warnings_callout(result.warnings)))
    if include_idea:
        parts.append(("idea", "The idea", _idea_section(result.spec, template_title=template_title,
                                                         template_description=template_description, references=references,
                                                         idea_text=result.idea)))
    parts.append(("key-numbers", "Key numbers", _key_numbers(result, frame)))
    parts.append(("verdict", "Verdict", _verdict_section(result, metric_checks)))
    parts.append(("growth", "Growth of $1", _growth_section(result, frame)))
    parts.append(("drawdown", "Drawdown", _drawdown_section(frame, key, result.stats.get(key) if key else None)))
    parts.append(("quantiles", "Quantiles", _quantile_section(result.quantiles)))
    parts.append(("factors", "Factor exposure", _regression_section(result.regression)))
    parts.append(("factor-checks", "Factor checks", _factor_checks_section(result.factor_checks)))
    parts.append(("stats", "Statistics", _stats_section(result.stats)))
    parts.append(("holdings", "Holdings", _holdings_section(result.latest_holdings)))
    parts.append(("data", "Data", _data_usage_section(result)))
    parts.append(("llm-audit", "LLM audit", _llm_section(result.llm_calls)))
    html_parts = [h for _, _, h in parts if h]
    toc = [(a, label) for a, label, h in parts if h]
    return "\n".join(html_parts), toc


def _toc(entries: Sequence[tuple[str, str]]) -> str:
    links = "".join(f'<a href="#{_e(a)}">{_e(label)}</a>' for a, label in entries)
    return f'<nav class="toc" aria-label="Sections">{links}</nav>'


def _meta(items: Sequence[tuple[str, object]]) -> str:
    spans = "".join(f"<span>{_e(k)}: <b>{_e(v)}</b></span>" for k, v in items if v not in (None, ""))
    return f'<div class="meta">{spans}</div>'


# ------------------------------------------------------------------------------------------------
# Public renderers
# ------------------------------------------------------------------------------------------------


def render_backtest_html(
    result: BacktestResult,
    *,
    template_title: str | None = None,
    template_description: str | None = None,
    references: list[str] | None = None,
    metric_checks: Sequence[EvidenceCheck] | None = None,
    home_href: str | None = None,
) -> str:
    """One self-contained HTML page for a backtest that shows the idea, the evidence and the caveats.

    ``metric_checks`` are the verification results of the interpretation's cited metrics (from
    ``verify_interpretation``); when omitted they are recomputed, and warnings that name a cited
    path as unverified / mismatched mark it ✗. ``home_href`` adds a back link (dashboard).
    """
    spec_name = result.spec.get("name") if isinstance(result.spec, Mapping) else None
    body, toc = _backtest_body(result, template_title=template_title, template_description=template_description,
                               references=references, metric_checks=metric_checks)
    verdict = result.interpretation.verdict if result.interpretation else None
    badge = ""
    if verdict:
        label, tone, icon, meaning = _VERDICTS.get(verdict, (verdict, "muted", "?", ""))
        badge = f'<div class="badges">{_badge(label, tone, icon=icon, title=meaning)}</div>'
    header = (
        '<header class="top">'
        '<p class="eyebrow">Backtest report - simulated</p>'
        f"<h1>{_e(result.idea)}</h1>{badge}"
        + _meta([
            ("Strategy", spec_name or "n/a"),
            ("Template", template_title),
            ("Period", f"{result.start.isoformat()} to {result.end.isoformat()}"),
            ("Rebalance", result.rebalance),
            ("Data", result.provider),
            ("LLM", result.llm),
            ("Run", result.run_id),
            ("Started", _dt(result.started_at)),
            ("Finished", _dt(result.finished_at) if result.finished_at else None),
        ])
        + _toc(toc)
        + "</header>"
    )
    title = f"Backtest - {spec_name or result.idea}"[:120]
    return html_page(title, header + body, home_href=home_href, subtitle=result.idea[:300])


# ------------------------------------------------------------------------------------------------
# Screening pipeline
# ------------------------------------------------------------------------------------------------


def _match_grounding(idea: InvestmentIdea) -> tuple[dict[int, EvidenceCheck], dict[int, EvidenceCheck]]:
    """Map quant / quote evidence (by index) to their grounding checks."""
    quant: dict[int, EvidenceCheck] = {}
    quotes: dict[int, EvidenceCheck] = {}
    if idea.thesis is None or idea.grounding is None:
        return quant, quotes
    q_checks = [c for c in idea.grounding.checks if c.kind == "quant"]
    n_checks = [c for c in idea.grounding.checks if c.kind == "quote"]
    used: set[int] = set()
    for i, ev in enumerate(idea.thesis.quant_evidence):
        for j, c in enumerate(q_checks):
            if j not in used and c.ref == ev.feature:
                quant[i] = c
                used.add(j)
                break
    used = set()
    for i, ev in enumerate(idea.thesis.narrative_evidence):
        found = None
        for j, c in enumerate(n_checks):
            if j not in used and c.claim.strip() == ev.quote.strip() and (c.ref == ev.doc_id or not c.ref):
                found = j
                break
        if found is None:
            for j, c in enumerate(n_checks):
                if j not in used and c.claim.strip() == ev.quote.strip():
                    found = j
                    break
        if found is None and i < len(n_checks) and i not in used and len(n_checks) == len(idea.thesis.narrative_evidence):
            found = i
        if found is not None:
            used.add(found)
            quotes[i] = n_checks[found]
    return quant, quotes


def _spec_section_screen(spec: Mapping[str, Any], pushdown: str | None) -> str:
    try:
        s = ScreenSpec.model_validate(dict(spec))
    except (ValidationError, TypeError, ValueError):
        s = None
    if s is None:
        body = "<p class=\"muted\">The screen specification could not be parsed; raw JSON:</p>" \
               f"<pre>{_e(json.dumps(spec, indent=2, default=str, ensure_ascii=False))}</pre>"
        return _section("spec", "Screen specification", body)
    parts = []
    rows = [[str(i), _e(_describe_condition(c)), f"<code>{_e(c.describe())}</code>"] for i, c in enumerate(s.conditions, 1)]
    parts.append(_table(["#", "Condition", "Exact rule"], rows, num_cols=[0], caption="All of these must hold"))
    for gi, group in enumerate(s.any_of, 1):
        rows = [[_e(_describe_condition(c)), f"<code>{_e(c.describe())}</code>"] for c in group]
        parts.append(_table(["Condition", "Exact rule"], rows, caption=f"At least one of group {gi}"))
    total = sum(f.weight for f in s.ranking) or 1.0
    rows = [[_e(_feature_label(f.feature)), f"<code>{_e(f.feature)}</code>",
             "higher is better" if f.direction == "higher_is_better" else "lower is better",
             _pct(f.weight / total * 100, 0), _e(f.rationale)] for f in s.ranking]
    parts.append(_table(["Ranking factor", "Feature", "Direction", "Weight", "Why"], rows, num_cols=[3], caption=f"Ranking (top {s.top_n})"))
    u = s.universe
    parts.append(_meta([("Universe", u.country), ("Security types", ", ".join(u.security_types)),
                        ("Min price", f"${u.min_price:g}" if u.min_price is not None else "none"),
                        ("Min avg $ volume", f"${u.min_avg_dollar_volume_usd_mn:g}mn" if u.min_avg_dollar_volume_usd_mn is not None else "none"),
                        ("Excluded sectors", ", ".join(u.exclude_sectors))]))
    if s.assumptions:
        parts.append("<h3>Assumptions</h3>" + _ul(s.assumptions))
    if s.unsupported_requests:
        parts.append('<div class="callout bad"><h2>Not expressible with the available data</h2>' + _ul(s.unsupported_requests) + "</div>")
    if pushdown:
        parts.append(f"<h3>Vendor-side query</h3><pre>{_e(pushdown)}</pre>")
    return _section("spec", "Screen specification", "".join(parts))


def _idea_card(idea: InvestmentIdea) -> str:
    c = idea.candidate
    anchor = f"idea-{_anchor(c.ticker)}"
    head = (f"<h3>#{c.rank} {_e(c.ticker)} <span class=\"muted\">{_e(c.name)}</span></h3>"
            f'<div class="badges">{_badge(f"score {c.score:.2f}", "muted", icon="")}')
    t = idea.thesis
    g = idea.grounding
    if t is not None:
        head += _badge(t.dislocation_type.replace("_", " "), "info", icon="")
        head += _badge(f"{t.conviction} conviction", "good" if t.conviction == "high" else "warn" if t.conviction == "medium" else "muted", icon="")
        head += _badge("actionable", "good") if t.is_actionable else _badge("not actionable", "bad")
    if g is not None and g.checks:
        tone = "good" if g.is_fully_grounded else "warn" if g.verified_ratio >= 0.5 else "bad"
        head += _badge(f"{g.n_verified}/{len(g.checks)} evidence verified", tone)
    head += "</div>"
    parts = [head]
    if idea.error:
        parts.append(f'<div class="callout bad"><b>Explanation failed:</b> {_e(idea.error)}</div>')
    if t is not None:
        quant_map, quote_map = _match_grounding(idea)
        parts.append(f'<p class="lead">{_e(t.headline)}</p>')
        parts.append('<dl class="kv">'
                     f"<dt>Market narrative</dt><dd>{_e(t.market_narrative)}</dd>"
                     f"<dt>Variant view</dt><dd>{_e(t.variant_view)}</dd>"
                     f"<dt>Why the dislocation exists</dt><dd>{_e(t.why_dislocation_exists)}</dd></dl>")
        if t.quant_evidence:
            rows = []
            for i, ev in enumerate(t.quant_evidence):
                chk = quant_map.get(i)
                rows.append([_mark(chk.status if chk else None, chk.detail if chk else "not checked"), f"<code>{_e(ev.feature)}</code>",
                             _num(ev.value), _e(ev.interpretation)])
            parts.append("<h4>Quantitative evidence</h4>" + _table(["", "Feature", "Value", "Interpretation"], rows, num_cols=[2]))
        if t.narrative_evidence:
            parts.append("<h4>What management / the documents said</h4>")
            for i, ev in enumerate(t.narrative_evidence):
                chk = quote_map.get(i)
                src = f"{ev.speaker} - " if ev.speaker else ""
                parts.append(f"<blockquote>{_mark(chk.status if chk else None, chk.detail if chk else 'not checked')} "
                             f"“{_e(ev.quote)}”<span class=\"src\">{_e(src)}{_e(ev.doc_id)}</span>"
                             f"<span class=\"src\">{_e(ev.interpretation)}</span></blockquote>")
        lists = [("Catalysts", t.catalysts), ("Risks", t.risks), ("What would prove it wrong", t.invalidation_triggers),
                 ("Data gaps", t.data_gaps)]
        parts.append('<div class="two-col">' + "".join(f"<div><h4>{_e(h)}</h4>{_ul(items) or '<p class=muted>none</p>'}</div>"
                                                       for h, items in lists) + "</div>")
    if c.features:
        rows = [[f"<code>{_e(k)}</code>", _e(_num(v) if _finite(v) is not None else (v if v is not None else "n/a"))]
                for k, v in c.features.items()]
        parts.append(f"<details><summary>Feature values ({len(rows)})</summary>{_table(['Feature', 'Value'], rows, num_cols=[1])}</details>")
    if idea.documents_used:
        parts.append(f'<p class="small muted">Documents shown to the explainer: {_e(", ".join(idea.documents_used))}</p>')
    return f'<article class="idea-card" id="{_e(anchor)}">{"".join(parts)}</article>'


def render_pipeline_html(result: PipelineResult, *, home_href: str | None = None) -> str:
    """The screening report (observation -> screen -> ranked candidates -> grounded theses) as HTML."""
    parts: list[tuple[str, str, str]] = []
    parts.append(("warnings", "Warnings", _warnings_callout(result.warnings)))
    parts.append(("spec", "Screen", _spec_section_screen(result.spec, result.pushdown_query)))

    if result.feature_coverage:
        rows = []
        for f, cov in sorted(result.feature_coverage.items(), key=lambda kv: kv[1]):
            v = _finite(cov) or 0.0
            mark = _mark("mismatch", "less than half of the universe has this feature") if v < 0.5 else ""
            rows.append([f"{mark} <code>{_e(f)}</code>", _frac_pct(v, 0) + f'<span class="meter"><span style="width:{max(0.0, min(v, 1.0)) * 100:.0f}%"></span></span>'])
        parts.append(("coverage", "Coverage", _section("coverage", "Data coverage", _table(["Feature", "Universe coverage"], rows, num_cols=[1]),
                                                        intro="Share of the universe with a value for each feature; missing data excludes a name.")))
    if result.funnel:
        labels = ["Universe"] + [s.label for s in result.funnel]
        values = [result.universe_size] + [s.remaining for s in result.funnel]
        chart = svg.horizontal_bar_chart(labels, values, title="Names remaining after each condition", value_format="{:,.0f}")
        rows = [[_e(s.label), _int(s.passed_alone), _int(s.remaining), _int(s.missing_data)] for s in result.funnel]
        body = _chart(chart) + _table(["Condition", "Pass alone", "Remaining", "Missing data"], rows, num_cols=[1, 2, 3])
        parts.append(("funnel", "Funnel", _section("funnel", "Screening funnel", body,
                                                    intro=f"{result.universe_size:,} names screened; {result.survivors:,} survived every condition.")))
    if result.ideas:
        rows = []
        for idea in result.ideas:
            c = idea.candidate
            top = sorted(c.factor_scores.items(), key=lambda kv: -kv[1])[:4]
            fs = ", ".join(f"{k} {v * 100:.0f}%" for k, v in top)
            g = idea.grounding
            grounded = f"{g.n_verified}/{len(g.checks)}" if g and g.checks else "n/a"
            headline = idea.thesis.headline if idea.thesis else (idea.error or "")
            rows.append([str(c.rank), f'<a href="#idea-{_e(_anchor(c.ticker))}">{_e(c.ticker)}</a>', _e(c.name), _num(c.score),
                         _e(fs), _e(grounded), _e(headline)])
        parts.append(("candidates", "Candidates", _section("candidates", "Ranked candidates",
                                                            _table(["Rank", "Ticker", "Name", "Score", "Top factor percentiles", "Grounded", "Thesis"],
                                                                   rows, num_cols=[0, 3]))))
        cards = "".join(_idea_card(i) for i in result.ideas)
        parts.append(("theses", "Theses", _section("theses", "Ideas and theses", cards,
                                                    intro="✓ = the number or quote was found verbatim in the data / documents; ✗ = it was not.")))
    parts.append(("llm-audit", "LLM audit", _llm_section(result.llm_calls)))
    body = "\n".join(h for _, _, h in parts if h)
    toc = [(a, label) for a, label, h in parts if h]
    header = (
        '<header class="top"><p class="eyebrow">Screening report</p>'
        f"<h1>{_e(result.observation)}</h1>"
        + _meta([("As of", result.as_of.isoformat()), ("Data", result.provider), ("LLM", result.llm),
                 ("Universe", f"{result.universe_size:,}"), ("Survivors", f"{result.survivors:,}"), ("Run", result.run_id),
                 ("Started", _dt(result.started_at)), ("Finished", _dt(result.finished_at) if result.finished_at else None)])
        + _toc(toc) + "</header>"
    )
    return html_page(f"Screen - {result.observation}"[:120], header + body, home_href=home_href, subtitle=result.observation[:300])


# ------------------------------------------------------------------------------------------------
# Strategy page (paper account + backtest)
# ------------------------------------------------------------------------------------------------


def _nav_history(paper: Mapping[str, Any]) -> pd.Series:
    """NAV history of a paper summary as a Series (accepts (date, nav) pairs or {date, nav} dicts).

    When the account's start date and initial capital are known and precede the first NAV point,
    the starting capital is prepended so the chart starts where the account started.
    """
    pts: dict[pd.Timestamp, float] = {}
    for item in paper.get("nav_history") or []:
        d = v = None
        if isinstance(item, Mapping):
            d, v = item.get("date"), item.get("nav")
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            d, v = item[0], item[1]
        f = _finite(v)
        if d is None or f is None:
            continue
        try:
            ts = pd.Timestamp(d)
        except (ValueError, TypeError):
            continue
        if ts.tzinfo is not None:
            ts = ts.tz_localize(None)
        pts[ts.normalize()] = f
    if not pts:
        return pd.Series(dtype=float)
    nav = pd.Series(pts, dtype=float).sort_index()
    cap, started = _finite(paper.get("initial_capital")), paper.get("started")
    if cap is not None and cap > 0 and started:
        try:
            t0 = pd.Timestamp(started).normalize()
        except (ValueError, TypeError):
            t0 = None
        if t0 is not None and t0.tzinfo is None and t0 < nav.index[0]:
            nav = pd.concat([pd.Series([cap], index=[t0]), nav])
    return nav


def _expected_path(nav: pd.Series, backtest: BacktestResult | None, expected_pct: float | None,
                   base: float | None = None) -> tuple[pd.Series | None, str]:
    """Backtest-expected NAV path from the paper account's start.

    Preference: the backtest's realised returns over the same dates (when the backtest covers the
    paper window), else compounding at the backtest CAGR, else a smooth path to the paper
    summary's ``backtest_expected_return_pct``.
    """
    if nav.empty:
        return None, ""
    nav0 = float(base) if base is not None and base > 0 else float(nav.iloc[0])
    t0, t1 = nav.index[0], nav.index[-1]
    if backtest is not None:
        frame = returns_frame(backtest)
        key = _primary_key(backtest)
        if key in frame.columns and frame[key].notna().any():
            r = frame[key]
            # same convention as PaperAccount.summary: backtest periods ending in (start, last run]
            window = r[(r.index > t0) & (r.index <= t1)]
            if r.index[0] <= t0 and r.index[-1] >= t1 and len(window):
                w = nav0 * (1.0 + window.fillna(0.0)).cumprod()
                return pd.concat([pd.Series([nav0], index=[t0]), w]), "Backtest (same dates)"
        st = backtest.stats.get(key) if key else None
        if st is not None and _finite(st.cagr_pct) is not None and st.cagr_pct > -100:
            g = 1.0 + st.cagr_pct / 100.0
            years = np.asarray((nav.index - t0) / pd.Timedelta(days=365.25), dtype=float)
            return pd.Series(nav0 * g ** years, index=nav.index), f"Backtest-expected (CAGR {_pct(st.cagr_pct, 1, signed=True)})"
    f = _finite(expected_pct)
    if f is not None and len(nav) >= 2 and f > -100:
        span = (nav.index[-1] - t0) / pd.Timedelta(days=1) or 1.0
        frac = np.asarray((nav.index - t0) / pd.Timedelta(days=1), dtype=float) / span
        return pd.Series(nav0 * (1.0 + f / 100.0) ** frac, index=nav.index), "Backtest-expected"
    return None, ""


def _trades_latest_first(trades: Sequence[Any]) -> list[Mapping[str, Any]]:
    """Trades newest first. Input already newest first (``PaperAccount.summary``) keeps its order;
    oldest-first input is reversed; anything else is stably sorted by date."""
    items = [t for t in trades if isinstance(t, Mapping)]
    dates = [str(t.get("date", "")) for t in items]
    if all(a >= b for a, b in zip(dates, dates[1:])):
        return items
    if all(a <= b for a, b in zip(dates, dates[1:])):
        return items[::-1]
    return sorted(items, key=lambda t: str(t.get("date", "")), reverse=True)


def _paper_section(paper: Mapping[str, Any] | None, backtest: BacktestResult | None) -> str:
    title = "Paper trading (simulated)"
    intro = ("A virtual account that follows the strategy's target portfolio at each rebalance - the same code path as "
             "the backtest. No broker, no real money.")
    if not paper or (paper.get("status") == "not_started" and not paper.get("nav_history")):
        return _section("paper", title,
                        '<p class="muted">This strategy is not being paper-traded yet. Paper trading is simulated: orders are '
                        "filled at the latest close in a virtual account; no broker is involved.</p>")
    nav = _nav_history(paper)
    since = _finite(paper.get("since_start_return_pct"))
    expected = _finite(paper.get("backtest_expected_return_pct"))
    holdings = [h for h in (paper.get("holdings") or []) if isinstance(h, Mapping)]
    trades = _trades_latest_first(paper.get("trades") or [])
    gap = since - expected if since is not None and expected is not None else None
    n_trades = paper.get("n_trades", len(trades))
    cards = [
        _stat_card("Net asset value", _money(paper.get("nav")), _e(f"cash {_money(paper.get('cash'))}")),
        _stat_card("Return since start", _pct(since, 2, signed=True), _e(f"started {paper.get('started') or 'n/a'}"),
                   tone="bad" if since is not None and since < 0 else ""),
        _stat_card("Backtest expected", _pct(expected, 2, signed=True), "backtest return over the same window"),
        _stat_card("Paper minus backtest", (f"{gap:+,.2f} pp" if gap is not None else "n/a"), "tracking gap"),
        _stat_card("Positions", _int(len(holdings)), _e(f"{_int(n_trades)} trades, costs {_money(paper.get('total_costs'))}"
                                                        if paper.get("total_costs") is not None else f"{_int(n_trades)} trades")),
        _stat_card("Last run", _e(paper.get("last_run") or "n/a"),
                   _e(f"next rebalance {paper['next_rebalance']}") if paper.get("next_rebalance") else "run again to update"),
    ]
    if paper.get("max_drawdown_pct") is not None:
        cards.append(_stat_card("Max drawdown", _pct(paper.get("max_drawdown_pct"), 2), "of the paper NAV"))
    if paper.get("gross_exposure_pct") is not None:
        cards.append(_stat_card("Gross / net exposure", _pct(paper.get("gross_exposure_pct"), 0),
                                _e(f"net {_pct(paper.get('net_exposure_pct'), 0, signed=True)}")))
    parts = [f'<div class="cards">{"".join(cards)}</div>']
    if paper.get("warnings"):
        parts.append('<div class="callout"><b>Last run warnings</b>' + _ul(paper.get("warnings") or []) + "</div>")
    if len(nav):
        base = _finite(paper.get("initial_capital"))
        base = base if base is not None and base > 0 else float(nav.iloc[0])
        path, label = _expected_path(nav, backtest, expected, base)
        series = {"Paper NAV": nav}
        colors: dict[str, int] = {"Paper NAV": 1}
        if path is not None:
            series[label] = path
            colors[label] = 2
        chart = svg.line_chart(series, title="Paper NAV vs backtest expectation", y_label="Account value", y_format="${:,.0f}",
                               colors=colors, reference=base)
        parts.append(_chart(chart, "The simulated account's value and the path the backtest implies from the same start."))
        if gap is not None:
            parts.append(f"<p>Since the start the paper account returned <b>{_e(_pct(since, 2, signed=True))}</b> versus "
                         f"<b>{_e(_pct(expected, 2, signed=True))}</b> in the backtest over the same window "
                         f"(difference {gap:+,.2f} pp). Paper fills use the close of the run day, whole-share rounding and "
                         "skipped small orders, so small differences are expected.</p>")
    # holdings
    if holdings:
        def value(h: Mapping[str, Any]) -> float | None:
            return _finite(h.get("value", h.get("market_value")))

        def frac(h: Mapping[str, Any], key: str) -> float | None:
            # ``weight`` / ``target_weight`` are always fractions of NAV (PaperAccount.summary), even
            # when |w| > 1 (a short that ran against us, a levered book); percent only via explicit *_pct keys
            f = _finite(h.get(key))
            if f is not None:
                return f
            p = _finite(h.get(f"{key}_pct"))
            return p / 100.0 if p is not None else None

        def size(h: Mapping[str, Any]) -> float:
            return abs(frac(h, "weight") or value(h) or 0.0)

        has_target = any(frac(h, "target_weight") is not None for h in holdings)
        rows = []
        for h in sorted(holdings, key=size, reverse=True):
            row = [_e(h.get("ticker", "?")), _num(h.get("shares"), 2), _money(h.get("price")), _money(value(h)),
                   _frac_pct(frac(h, "weight"), 2)]
            if has_target:
                row.append(_frac_pct(frac(h, "target_weight"), 2))
            rows.append(row)
        cash_row = ['<span class="muted">Cash</span>', "", "", _money(paper.get("cash")), ""] + ([""] if has_target else [])
        rows.append(cash_row)
        headers = ["Ticker", "Shares", "Price", "Value", "Weight"] + (["Target weight"] if has_target else [])
        parts.append("<h3>Current holdings</h3>" + _table(headers, rows, num_cols=range(1, len(headers))))
    else:
        parts.append('<h3>Current holdings</h3><p class="muted">No open positions.</p>')
    # trade blotter, latest first
    if trades:
        shown = trades[:500]
        tot_cost = sum(_finite(t.get("cost")) or 0.0 for t in trades)
        has_reason = any(t.get("reason") for t in shown)
        rows = []
        for t in shown:
            side = str(t.get("side", "")).lower()
            tone = "info" if side in ("buy", "cover") else "muted"
            row = [_e(t.get("date", "")), _e(t.get("ticker", "")), _badge(side or "?", tone, icon=""), _num(t.get("shares"), 2),
                   _money(t.get("price")), _money(t.get("notional")), _money(t.get("cost"))]
            if has_reason:
                row.append(_e(t.get("reason") or ""))
            rows.append(row)
        headers = ["Date", "Ticker", "Side", "Shares", "Price", "Notional", "Cost"] + (["Reason"] if has_reason else [])
        more = f'<p class="small muted">Showing the latest 500 of {len(trades):,} trades.</p>' if len(trades) > 500 else ""
        parts.append(f'<h3 id="trades">Trade blotter (latest first)</h3><p class="small muted">Simulated fills; total costs '
                     f"{_e(_money(tot_cost))}.</p>" + _table(headers, rows, num_cols=[3, 4, 5, 6]) + more)
    else:
        parts.append('<h3 id="trades">Trade blotter</h3><p class="muted">No trades yet.</p>')
    return _section("paper", title, "".join(parts), intro=intro)


def render_strategy_page(
    name: str,
    spec: StrategySpec | Mapping[str, Any] | None,
    backtest: BacktestResult | None,
    paper: dict | None,
    *,
    home_href: str | None = None,
) -> str:
    """A saved strategy: the idea, the simulated account (holdings, trades, NAV vs backtest) and the backtest.

    ``paper`` is ``PaperAccount.summary()``: nav_history [(date_iso, nav)], holdings [{ticker, shares,
    price, value, weight}], trades [{date, ticker, side, shares, price, notional, cost}], cash, nav,
    started, last_run, since_start_return_pct, backtest_expected_return_pct.
    """
    s = _coerce_spec(spec)
    spec_obj: object = s if s is not None else spec
    tmpl = template_info(s.name if s else None)
    t_title, t_desc, t_refs = tmpl if tmpl else (None, None, None)
    idea_text = s.idea if s else (backtest.idea if backtest else "")
    parts: list[tuple[str, str, str]] = []
    if backtest is not None:
        parts.append(("warnings", "Warnings", _warnings_callout(backtest.warnings, "Backtest caveats")))
    parts.append(("idea", "The idea", _idea_section(spec_obj, template_title=t_title, template_description=t_desc,
                                                     references=t_refs)))
    parts.append(("paper", "Paper trading", _paper_section(paper, backtest)))
    if backtest is not None:
        body, _ = _backtest_body(backtest, template_title=t_title, template_description=t_desc, references=t_refs,
                                 metric_checks=None, include_idea=False, include_warnings=False)
        bt_head = _section("backtest", "Backtest", _meta([
            ("Run", backtest.run_id), ("Period", f"{backtest.start.isoformat()} to {backtest.end.isoformat()}"),
            ("Rebalance", backtest.rebalance), ("Data", backtest.provider), ("LLM", backtest.llm)]),
            intro="The historical simulation this strategy was saved from.")
        parts.append(("backtest", "Backtest", bt_head + body))
    else:
        parts.append(("backtest", "Backtest", _section("backtest", "Backtest", '<p class="muted">No backtest result is saved for this strategy yet.</p>')))
    toc = [(a, label) for a, label, h in parts if h]
    header = (
        '<header class="top"><p class="eyebrow">Strategy - simulated (paper) trading only</p>'
        f"<h1>{_e(name)}</h1>"
        + (f'<p class="lead">{_e(idea_text)}</p>' if idea_text else "")
        + _meta([("Kind", s.kind.replace("_", " ") if s else None), ("Rebalance", s.rebalance if s else None),
                 ("Costs", f"{s.costs_bps:g} bps" if s else None),
                 ("Paper since", (paper or {}).get("started")), ("Last run", (paper or {}).get("last_run"))])
        + _toc(toc) + "</header>"
    )
    body = "\n".join(h for _, _, h in parts if h)
    return html_page(f"Strategy - {name}"[:120], header + body, home_href=home_href, subtitle=idea_text[:300] if idea_text else None)


# ------------------------------------------------------------------------------------------------
# Ideas inbox
# ------------------------------------------------------------------------------------------------

_STATUS_ORDER = ["new", "proposed", "accepted", "tested", "saved", "deferred", "failed", "rejected"]
_STATUS_TONE = {"new": "info", "proposed": "info", "accepted": "good", "tested": "good", "saved": "good", "deferred": "muted",
                "failed": "bad", "rejected": "muted"}
_TESTABILITY = {
    "testable_now": ("Testable now", "good", "✓"),
    "partially_testable": ("Partially testable", "warn", "!"),
    "needs_institutional_data": ("Needs institutional data", "warn", "!"),
    "not_testable": ("Not testable", "bad", "✗"),
}
_REPLICATION = {
    "replicates": ("Replicates", "good", "✓"),
    "partially_replicates": ("Partially replicates", "warn", "!"),
    "fails_to_replicate": ("Fails to replicate", "bad", "✗"),
    "inconclusive": ("Inconclusive", "muted", "?"),
}


def _quote_checks(c: IdeaCandidate) -> list[EvidenceCheck | None]:
    quotes = c.extraction.evidence_quotes
    checks = list(c.quote_checks)
    out: list[EvidenceCheck | None] = []
    used: set[int] = set()
    for i, q in enumerate(quotes):
        found = next((j for j, ch in enumerate(checks) if j not in used and ch.claim.strip() == q.strip()), None)
        if found is None and len(checks) == len(quotes) and i not in used:
            found = i
        if found is not None:
            used.add(found)
            out.append(checks[found])
        else:
            out.append(None)
    return out


def _replication_html(rep: ReplicationReport) -> str:
    label, tone, icon = _REPLICATION.get(rep.verdict, (rep.verdict, "muted", "?"))
    parts = [f'<h4>Replication</h4><div class="badges">{_badge(label, tone, icon=icon)}'
             f'<span class="small muted">run {_e(rep.backtest_run_id)}</span></div>',
             f"<p>{_e(rep.summary)}</p>"]
    parts.append(_table(["", "Replicated", "Claimed by the source"],
                        [["Sharpe", _num(rep.base_sharpe), _num(rep.claimed_sharpe)],
                         ["Alpha / effect t-stat", _num(rep.base_alpha_t_stat), _num(rep.claimed_t_stat)],
                         ["Replication ratio (Sharpe)", _num(rep.replication_ratio), ""]], num_cols=[1, 2]))
    if rep.checks:
        rows = []
        for ch in rep.checks:
            status = "verified" if ch.passed else ("mismatch" if ch.passed is False else None)
            rows.append([_mark(status, "passed" if ch.passed else "failed" if ch.passed is False else "could not be run"),
                         _e(ch.name), _e(ch.description), _num(ch.sharpe), _pct(ch.cagr_pct), _num(ch.alpha_t_stat), _int(ch.n_periods),
                         _e(ch.note)])
        parts.append(_table(["", "Check", "What it tests", "Sharpe", "CAGR", "Alpha t", "Periods", "Note"], rows, num_cols=[3, 4, 5, 6],
                            caption="Robustness checks"))
    if rep.caveats:
        parts.append("<h4>Caveats</h4>" + _ul(rep.caveats))
    return "".join(parts)


def _idea_inbox_card(c: IdeaCandidate) -> str:
    x = c.extraction
    src = c.source
    anchor = f"idea-{_anchor(c.idea_id)}"
    t_label, t_tone, t_icon = _TESTABILITY.get(x.testability, (x.testability, "muted", "?"))
    badges = (_badge(c.status, _STATUS_TONE.get(c.status, "muted"), icon="")
              + _badge(t_label, t_tone, icon=t_icon)
              + _badge({"new": "novel", "variant_of_library": "variant of a library idea", "duplicate": "duplicate"}.get(c.novelty, c.novelty),
                       "info" if c.novelty == "new" else "muted", icon="")
              + f'<span class="small muted">score {_num(c.score)} <span class="meter"><span style="width:{max(0.0, min(_finite(c.score) or 0.0, 1.0)) * 100:.0f}%"></span></span></span>')
    if not x.is_trading_idea:
        badges += _badge("not a trading idea", "bad")
    authors = ", ".join(src.authors[:3]) + (" et al." if len(src.authors) > 3 else "")
    source_line = " - ".join(p for p in [
        _link(src.url, src.title or src.url),
        _e(src.source_name or src.source_type),
        _e(authors) if authors else "",
        _e(f"published {src.published.isoformat()}") if src.published else "",
    ] if p)
    parts = [f"<h3>{_e(x.title)}</h3>", f'<div class="badges">{badges}</div>', f'<p class="small">{source_line}</p>',
             f'<p class="lead">{_e(x.summary)}</p>']
    parts.append('<dl class="kv">'
                 f"<dt>Claimed effect</dt><dd>{_e(x.claimed_effect)}</dd>"
                 f"<dt>Signal</dt><dd>{_e(x.signal_description)}</dd>"
                 + (f"<dt>Strategy to test</dt><dd>{_e(x.proposed_strategy_idea)}</dd>" if x.proposed_strategy_idea else "")
                 + f"<dt>Asset class</dt><dd>{_e(x.asset_class)}</dd>"
                 + (f"<dt>Holding period</dt><dd>{_e(x.holding_period)}</dd>" if x.holding_period else "")
                 + (f"<dt>Closest library idea</dt><dd>{_e(x.closest_library_template)}</dd>" if x.closest_library_template else "")
                 + "</dl>")
    parts.append(_table(["Reported Sharpe", "Reported annual return", "Reported t-stat", "Sample period"],
                        [[_num(x.reported_sharpe), _pct(x.reported_annual_return_pct), _num(x.reported_t_stat), _e(x.sample_period or "n/a")]],
                        num_cols=[0, 1, 2], caption="What the source reports"))
    if x.missing_data or x.data_requirements:
        parts.append('<div class="two-col">'
                     f"<div><h4>Data needed</h4>{_ul(x.data_requirements) or '<p class=muted>not stated</p>'}</div>"
                     f"<div><h4>Missing from this platform</h4>{_ul(x.missing_data) or '<p class=muted>nothing - all data available</p>'}</div>"
                     "</div>")
    if x.evidence_quotes:
        checks = _quote_checks(c)
        n_ok = sum(1 for ch in checks if ch is not None and ch.status == "verified")
        parts.append(f"<h4>Evidence quotes ({n_ok}/{len(x.evidence_quotes)} verified verbatim in the source)</h4>")
        for q, ch in zip(x.evidence_quotes, checks):
            detail = (ch.detail or ch.status) if ch else "not checked"
            parts.append(f"<blockquote>{_mark(ch.status if ch else None, detail)} “{_e(q)}”"
                         f'<span class="src">{_e(detail)}</span></blockquote>')
    if x.credibility_notes:
        parts.append("<h4>Credibility</h4>" + _ul(x.credibility_notes))
    parts.append(f'<h4>Try it</h4><p><code class="cmd">aitrading ideas try {_e(c.idea_id)}</code></p>')
    if c.replication is not None:
        parts.append(_replication_html(c.replication))
    meta = [("Idea id", c.idea_id), ("Discovered", _dt(c.discovered_at)), ("Decided", _dt(c.decided_at) if c.decided_at else None),
            ("Backtest run", c.backtest_run_id)]
    parts.append(_meta(meta))
    if c.notes:
        parts.append("<details><summary>Notes</summary>" + _ul(c.notes) + "</details>")
    return f'<article class="idea-card" id="{_e(anchor)}">{"".join(parts)}</article>'


def render_ideas_inbox(candidates: list[IdeaCandidate], *, home_href: str | None = None) -> str:
    """The idea inbox as cards: source, summary, claims, testability, verified quotes, how to try it."""
    def order(c: IdeaCandidate) -> tuple[int, float]:
        rank = _STATUS_ORDER.index(c.status) if c.status in _STATUS_ORDER else len(_STATUS_ORDER)
        return rank, -(_finite(c.score) or 0.0)

    ordered = sorted(candidates, key=order)
    counts: dict[str, int] = {}
    for c in candidates:
        counts[c.status] = counts.get(c.status, 0) + 1
    chips = "".join(_badge(f"{st}: {counts[st]}", _STATUS_TONE.get(st, "muted"), icon="")
                    for st in sorted(counts, key=lambda s: _STATUS_ORDER.index(s) if s in _STATUS_ORDER else 99))
    header = ('<header class="top"><p class="eyebrow">Auto strategy discovery</p><h1>Idea inbox</h1>'
              f'<p class="lead">{len(candidates):,} idea{"s" if len(candidates) != 1 else ""} found in papers and research sites. '
              "Each one shows what the source claims, whether its quotes were found verbatim in the source, whether the "
              "platform can test it, and the command to try it.</p>"
              f'<div class="badges">{chips}</div></header>')
    if not ordered:
        body = _section("ideas", "Ideas", '<p class="muted">The inbox is empty. Run <code>aitrading ideas discover</code> to look for new ideas.</p>')
    else:
        body = "".join(_idea_inbox_card(c) for c in ordered)
    return html_page("Idea inbox", header + body, home_href=home_href, subtitle="Discovered research ideas")
