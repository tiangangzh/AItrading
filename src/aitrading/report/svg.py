"""Small, dependency-free SVG chart helpers for the HTML reports.

Every helper returns a standalone ``<svg>`` string that:

* is ``viewBox``-based and responsive (``width="100%"``; it scales with its container);
* carries an accessible ``<title>`` / ``<desc>`` (``role="img"``, ``aria-labelledby``);
* takes its colours from CSS custom properties with hard-coded fallbacks
  (``var(--series-1, #2a78d6)``), so the page's light / dark theme tokens restyle it and the SVG
  still renders on its own;
* escapes every piece of text (series names, labels, titles) - labels are untrusted data;
* never raises on empty / all-NaN input: it renders a "No data" placeholder instead.

Line and area charts break their lines at NaN gaps, downsample long daily series to at most
:data:`MAX_POINTS` points per series while keeping each bucket's extremes (so peaks and troughs -
including the maximum drawdown - survive) and carry a hover layer of vertical bands whose native
tooltips list every series at that date. Bars carry per-bar tooltips.

Colour slots follow the reference categorical order (blue, orange, aqua, yellow, magenta, green,
violet, red); a slot is assigned to an entity by the caller and never by rank.
"""

from __future__ import annotations

import itertools
import math
from datetime import date, datetime
from html import escape
from typing import Callable, Mapping, Sequence, Union

import numpy as np
import pandas as pd

__all__ = [
    "MAX_POINTS",
    "SERIES_FALLBACK_COLORS",
    "series_color",
    "downsample",
    "nice_ticks",
    "log_ticks",
    "line_chart",
    "area_drawdown_chart",
    "bar_chart",
    "horizontal_bar_chart",
    "sparkline",
]

#: Maximum number of points drawn per line (long daily series are downsampled, extremes kept).
MAX_POINTS = 800

#: Light-mode fallbacks for ``--series-1`` ... ``--series-8`` (the page redefines them per theme).
SERIES_FALLBACK_COLORS = (
    "#2a78d6",  # 1 blue
    "#eb6834",  # 2 orange
    "#1baf7a",  # 3 aqua
    "#eda100",  # 4 yellow
    "#e87ba4",  # 5 magenta
    "#008300",  # 6 green
    "#4a3aa7",  # 7 violet
    "#e34948",  # 8 red
)

_TEXT = "var(--chart-text, #0b0b0b)"
_TEXT2 = "var(--chart-text-2, #52514e)"
_MUTED = "var(--chart-muted, #6b6a65)"  # >= 4.5:1 on the light chart surface
_GRID = "var(--chart-grid, #e1e0d9)"
_AXIS = "var(--chart-axis, #c3c2b7)"
_SURFACE = "var(--chart-surface, #fcfcfb)"
_NEG = "var(--neg, #e34948)"
_FONT = "system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif"

_MAX_HOVER_BANDS = 120
_EPOCH = pd.Timestamp("1970-01-01")
_ids = itertools.count(1)

ValueFormat = Union[str, Callable[[float], str]]


def series_color(slot: int) -> str:
    """CSS colour for categorical slot ``slot`` (1-based, 1..8)."""
    i = (int(slot) - 1) % len(SERIES_FALLBACK_COLORS)
    return f"var(--series-{i + 1}, {SERIES_FALLBACK_COLORS[i]})"


# ------------------------------------------------------------------------------------------------
# Small helpers
# ------------------------------------------------------------------------------------------------


def _e(text: object) -> str:
    return escape(str(text), quote=True)


def _n(x: float) -> str:
    """Compact coordinate (1 decimal, no trailing zeros)."""
    if not math.isfinite(x):
        return "0"
    s = f"{x:.1f}"
    return s[:-2] if s.endswith(".0") else s


def _new_id(kind: str) -> str:
    return f"{kind}{next(_ids)}"


def _fmt_value(v: float | None, fmt: ValueFormat | None, default_decimals: int = 2) -> str:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "n/a"
    if fmt is None:
        return f"{v:,.{default_decimals}f}"
    if callable(fmt):
        return fmt(v)
    try:
        return fmt.format(v)
    except (ValueError, IndexError, KeyError):
        return f"{v:,.{default_decimals}f}"


def _text_width(text: str, size: float = 12.0) -> float:
    """Rough rendered width of ``text`` in viewBox units for the system sans."""
    return 0.58 * size * len(text)


def _truncate(text: str, max_width: float, size: float = 12.0) -> str:
    if _text_width(text, size) <= max_width:
        return text
    n = max(1, int(max_width / (0.58 * size)) - 1)
    return text[:n] + "…"


def _svg_open(width: float, height: float, title: str, desc: str, *, cls: str, max_width: str | None = None) -> tuple[str, str]:
    cid = _new_id("c")
    style = f"font-family:{_FONT};max-width:{max_width or '100%'};height:auto;display:block"
    head = (
        f'<svg xmlns="http://www.w3.org/2000/svg" class="{cls}" viewBox="0 0 {_n(width)} {_n(height)}" '
        f'width="100%" preserveAspectRatio="xMidYMid meet" role="img" '
        f'aria-labelledby="{cid}-t {cid}-d" style="{style}">'
        f'<title id="{cid}-t">{_e(title)}</title><desc id="{cid}-d">{_e(desc)}</desc>'
    )
    return head, cid


def _empty_svg(title: str, *, width: float = 760, height: float = 160, message: str = "No data to chart") -> str:
    head, _ = _svg_open(width, height, title, message, cls="chart chart-empty")
    return (
        head
        + _visible_title(title)
        + f'<text x="{_n(width / 2)}" y="{_n(height / 2 + 10)}" text-anchor="middle" font-size="13" '
        f'style="fill:{_MUTED}">{_e(message)}</text></svg>'
    )


def _visible_title(title: str, x: float = 0, y: float = 18) -> str:
    if not title:
        return ""
    return f'<text x="{_n(x)}" y="{_n(y)}" font-size="14" font-weight="600" style="fill:{_TEXT}">{_e(title)}</text>'


def _clean_series(s: object) -> pd.Series:
    """Float series, sorted index, unique labels (last wins), +-inf -> NaN."""
    ser = s.copy() if isinstance(s, pd.Series) else pd.Series(s)
    ser = pd.to_numeric(ser, errors="coerce").astype(float)
    ser = ser.replace([np.inf, -np.inf], np.nan)
    if len(ser) == 0:
        return ser
    try:
        ser = ser.sort_index(kind="mergesort")
    except TypeError:
        pass
    if ser.index.has_duplicates:
        ser = ser[~ser.index.duplicated(keep="last")]
    return ser


def _as_datetime_index(index: pd.Index) -> pd.DatetimeIndex | None:
    if isinstance(index, pd.DatetimeIndex):
        dt = index
    elif len(index) and all(isinstance(v, (date, datetime, pd.Timestamp, np.datetime64)) for v in index[:50]):
        try:
            dt = pd.DatetimeIndex(pd.to_datetime(list(index)))
        except (ValueError, TypeError):
            return None
    elif len(index) and index.dtype == object and all(isinstance(v, str) for v in index[:50]):
        try:
            dt = pd.DatetimeIndex(pd.to_datetime(list(index)))
        except (ValueError, TypeError):
            return None
    else:
        return None
    if dt.tz is not None:
        dt = dt.tz_localize(None)
    return dt


def _x_numbers(index: pd.Index) -> tuple[np.ndarray, bool]:
    """Index -> (float x positions, is_date). Dates become days since the epoch."""
    dt = _as_datetime_index(index)
    if dt is not None:
        return np.asarray((dt - _EPOCH) / pd.Timedelta(days=1), dtype=float), True
    try:
        return np.asarray(index, dtype=float), False
    except (TypeError, ValueError):
        return np.arange(len(index), dtype=float), False


def _day_to_ts(x: float) -> pd.Timestamp:
    return _EPOCH + pd.Timedelta(days=float(x))


def _fmt_x(x: float, is_date: bool, span_days: float) -> str:
    if not is_date:
        return f"{x:g}"
    ts = _day_to_ts(x)
    return ts.strftime("%Y-%m-%d")


# ------------------------------------------------------------------------------------------------
# Downsampling & ticks
# ------------------------------------------------------------------------------------------------


def downsample(series: pd.Series, max_points: int = MAX_POINTS) -> pd.Series:
    """Reduce ``series`` to at most ``max_points`` points, keeping the shape and the extremes.

    The interior is split into equal buckets; each bucket keeps the positions of its minimum and
    maximum (so the global extremes and every local spike survive). The first and last points are
    always kept. Whenever the original series has a NaN between two consecutive kept finite points,
    one of those NaN positions is kept too, so every gap that would separate two drawn points still
    breaks the line (no segment is ever drawn across missing data). Series that already fit are
    returned unchanged.
    """
    s = series if isinstance(series, pd.Series) else pd.Series(series)
    n = len(s)
    max_points = max(int(max_points), 4)
    if n <= max_points:
        return s
    vals = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
    nan = ~np.isfinite(vals)
    has_nan = bool(nan.any())
    # K kept extremes need at most K - 1 gap markers: 4 per bucket + 3 stays within max_points
    n_buckets = (max_points - 3) // 4 if has_nan else (max_points - 2) // 2
    edges = np.linspace(1, n - 1, n_buckets + 1).astype(int)
    keep: set[int] = {0, n - 1}
    for b in range(n_buckets):
        lo, hi = int(edges[b]), int(edges[b + 1])
        if hi <= lo:
            continue
        seg = vals[lo:hi]
        ok = np.flatnonzero(~nan[lo:hi])
        if ok.size:
            keep.add(lo + int(ok[np.argmin(seg[ok])]))
            keep.add(lo + int(ok[np.argmax(seg[ok])]))
    if has_nan:
        kept = np.array(sorted(keep))
        nan_pos = np.flatnonzero(nan)
        a, b = kept[:-1], kept[1:]
        # first original NaN after each kept point; it lies inside (a, b) iff it is < b
        j = np.searchsorted(nan_pos, a, side="right")
        first = nan_pos[np.minimum(j, nan_pos.size - 1)]
        bridged = (j < nan_pos.size) & (first < b) & ~nan[a] & ~nan[b]
        keep.update(int(x) for x in first[bridged])
    return s.iloc[sorted(keep)]


def _nice_num(x: float, round_: bool) -> float:
    if x <= 0 or not math.isfinite(x):
        return 1.0
    exp = math.floor(math.log10(x))
    f = x / 10**exp
    if round_:
        nf = 1 if f < 1.5 else 2 if f < 3 else 5 if f < 7 else 10
    else:
        nf = 1 if f <= 1 else 2 if f <= 2 else 5 if f <= 5 else 10
    return nf * 10**exp


def nice_ticks(lo: float, hi: float, n: int = 5) -> list[float]:
    """Round tick values covering [lo, hi] (about ``n`` ticks; steps of 1, 2 or 5 x 10^k)."""
    if not (math.isfinite(lo) and math.isfinite(hi)):
        return [0.0, 1.0]
    if hi < lo:
        lo, hi = hi, lo
    if hi - lo < 1e-12 * max(1.0, abs(hi)):
        pad = abs(hi) * 0.1 if hi != 0 else 1.0
        lo, hi = lo - pad, hi + pad
    step = _nice_num((hi - lo) / max(n - 1, 1), True)
    start = math.floor(lo / step + 1e-9) * step
    stop = math.ceil(hi / step - 1e-9) * step
    decimals = max(0, -int(math.floor(math.log10(step))) + 1)
    ticks: list[float] = []
    v = start
    while v <= stop + step * 1e-6 and len(ticks) < 50:
        ticks.append(round(v, decimals) + 0.0)
        v += step
    return ticks


def log_ticks(lo: float, hi: float) -> list[float]:
    """Tick values for a log axis on [lo, hi] (lo > 0): 1-2-5 per decade, thinned to <= 8."""
    if lo <= 0 or hi <= 0 or not (math.isfinite(lo) and math.isfinite(hi)):
        return []
    if hi < lo:
        lo, hi = hi, lo
    e0, e1 = math.floor(math.log10(lo)), math.ceil(math.log10(hi))
    for mults in ((1, 2, 5), (1, 3), (1,)):
        ticks = [m * 10.0**e for e in range(e0, e1 + 1) for m in mults]
        ticks = [round(t, 12) for t in ticks if lo * (1 - 1e-9) <= t <= hi * (1 + 1e-9)]
        if len(ticks) <= 8:
            break
    if len(ticks) > 8:  # many decades: every k-th decade
        k = math.ceil(len(ticks) / 8)
        ticks = ticks[::k]
    return ticks


def _date_ticks(x0: float, x1: float, max_ticks: int = 8) -> list[tuple[float, str]]:
    """Calendar-aligned ticks (years, months or days) between two day numbers."""
    t0, t1 = _day_to_ts(x0), _day_to_ts(x1)
    span = x1 - x0
    out: list[tuple[float, str]] = []
    if span > 3 * 366:
        years = list(range(t0.year + (0 if (t0.month, t0.day) == (1, 1) else 1), t1.year + 1))
        step = 1
        for step in (1, 2, 5, 10, 20, 25, 50, 100):
            if len(years[::step]) <= max_ticks:
                break
        for y in years[::step]:
            ts = pd.Timestamp(year=y, month=1, day=1)
            out.append(((ts - _EPOCH) / pd.Timedelta(days=1), str(y)))
        return out
    if span > 62:
        first = pd.Timestamp(year=t0.year, month=t0.month, day=1)
        if first < t0.normalize():
            first = first + pd.offsets.MonthBegin(1)
        months = pd.date_range(first, t1, freq="MS")
        step = 1
        for step in (1, 2, 3, 6, 12):
            if len(months[::step]) <= max_ticks:
                break
        if step in (3, 6, 12):  # align to calendar quarters / halves / years
            months = months[(months.month - 1) % step == 0]
        else:
            months = months[::step]
        for ts in months:
            label = ts.strftime("%b %Y") if ts.month == 1 or len(out) == 0 else ts.strftime("%b")
            out.append(((ts - _EPOCH) / pd.Timedelta(days=1), label))
        return out
    step_days = 1
    for step_days in (1, 2, 7, 14, 28):
        if span / step_days <= max_ticks:
            break
    days = pd.date_range(t0.normalize(), t1, freq=f"{step_days}D")
    for ts in days:
        if ts < t0.normalize():
            continue
        out.append(((ts - _EPOCH) / pd.Timedelta(days=1), ts.strftime("%b %d")))
    return out


def _tick_formatter(ticks: Sequence[float], percent: bool, y_format: ValueFormat | None) -> Callable[[float], str]:
    if y_format is not None:
        return lambda v: _fmt_value(v, y_format)
    steps = [abs(b - a) for a, b in zip(ticks, ticks[1:]) if abs(b - a) > 0]
    step = min(steps) if steps else (abs(ticks[0]) if ticks and ticks[0] else 1.0)
    if percent:
        step *= 100.0
    decimals = 0 if step >= 1 else min(6, max(0, -int(math.floor(math.log10(step)))))
    if percent:
        return lambda v: f"{v * 100:,.{decimals}f}%"
    return lambda v: f"{v:,.{decimals}f}"


# ------------------------------------------------------------------------------------------------
# Building blocks
# ------------------------------------------------------------------------------------------------


def _path_with_gaps(xs: np.ndarray, ys: np.ndarray) -> tuple[str, list[tuple[float, float]]]:
    """SVG path data that breaks at NaN; returns (d, isolated points to draw as dots)."""
    parts: list[str] = []
    isolated: list[tuple[float, float]] = []
    run: list[tuple[float, float]] = []

    def flush() -> None:
        if len(run) == 1:
            isolated.append(run[0])
        elif len(run) > 1:
            parts.append("M" + " L".join(f"{_n(x)} {_n(y)}" for x, y in run))
        run.clear()

    for x, y in zip(xs, ys):
        if math.isfinite(x) and math.isfinite(y):
            run.append((float(x), float(y)))
        else:
            flush()
    flush()
    return " ".join(parts), isolated


def _bar_path(x: float, y_base: float, y_end: float, w: float, r: float = 4.0, horizontal: bool = False) -> str:
    """Bar with a rounded data end (radius ``r``) and a square baseline end.

    Vertical: x = left edge, the bar spans y_base -> y_end. Horizontal: x is the y of the top
    edge, the bar spans y_base -> y_end along the x axis (pass x positions as y_base / y_end).
    """
    length = abs(y_end - y_base)
    r = max(0.0, min(r, w / 2, length))
    if not horizontal:
        x0, x1 = x, x + w
        if y_end <= y_base:  # grows up (positive)
            return (f"M{_n(x0)} {_n(y_base)} L{_n(x0)} {_n(y_end + r)} Q{_n(x0)} {_n(y_end)} {_n(x0 + r)} {_n(y_end)} "
                    f"L{_n(x1 - r)} {_n(y_end)} Q{_n(x1)} {_n(y_end)} {_n(x1)} {_n(y_end + r)} L{_n(x1)} {_n(y_base)} Z")
        return (f"M{_n(x0)} {_n(y_base)} L{_n(x0)} {_n(y_end - r)} Q{_n(x0)} {_n(y_end)} {_n(x0 + r)} {_n(y_end)} "
                f"L{_n(x1 - r)} {_n(y_end)} Q{_n(x1)} {_n(y_end)} {_n(x1)} {_n(y_end - r)} L{_n(x1)} {_n(y_base)} Z")
    y0, y1 = x, x + w
    xb, xe = y_base, y_end
    if xe >= xb:  # grows right
        return (f"M{_n(xb)} {_n(y0)} L{_n(xe - r)} {_n(y0)} Q{_n(xe)} {_n(y0)} {_n(xe)} {_n(y0 + r)} "
                f"L{_n(xe)} {_n(y1 - r)} Q{_n(xe)} {_n(y1)} {_n(xe - r)} {_n(y1)} L{_n(xb)} {_n(y1)} Z")
    return (f"M{_n(xb)} {_n(y0)} L{_n(xe + r)} {_n(y0)} Q{_n(xe)} {_n(y0)} {_n(xe)} {_n(y0 + r)} "
            f"L{_n(xe)} {_n(y1 - r)} Q{_n(xe)} {_n(y1)} {_n(xe + r)} {_n(y1)} L{_n(xb)} {_n(y1)} Z")


def _legend(items: Sequence[tuple[str, str]], x0: float, y0: float, max_x: float, *, kind: str = "line") -> tuple[str, float]:
    """Legend row(s) of (label, colour) with line (or square) keys; returns (svg, height used)."""
    out: list[str] = []
    x, y = x0, y0
    row_h = 18.0
    for label, color in items:
        w = 22 + _text_width(label, 12) + 16
        if x > x0 and x + w > max_x:
            x, y = x0, y + row_h
        if kind == "line":
            out.append(f'<line x1="{_n(x)}" y1="{_n(y - 4)}" x2="{_n(x + 16)}" y2="{_n(y - 4)}" '
                       f'style="stroke:{color};stroke-width:2.5;stroke-linecap:round"/>')
        else:
            out.append(f'<rect x="{_n(x + 3)}" y="{_n(y - 10)}" width="10" height="10" rx="2" style="fill:{color}"/>')
        out.append(f'<text x="{_n(x + 22)}" y="{_n(y)}" font-size="12" style="fill:{_TEXT2}">{_e(label)}</text>')
        x += w
    return '<g class="legend">' + "".join(out) + "</g>", (y - y0) + row_h


def _hover_bands(
    names: Sequence[str],
    cleaned: Sequence[pd.Series],
    xs_all: np.ndarray,
    is_date: bool,
    sx: Callable[[float], float],
    top: float,
    height: float,
    fmt: Callable[[float], str],
) -> str:
    """Invisible vertical bands with native tooltips listing every series at that x."""
    ux = np.unique(xs_all[np.isfinite(xs_all)])
    if ux.size == 0:
        return ""
    k = min(_MAX_HOVER_BANDS, ux.size)
    anchors = ux[np.unique(np.linspace(0, ux.size - 1, k).round().astype(int))]
    lookup: list[tuple[np.ndarray, np.ndarray, float, float]] = []
    for s in cleaned:
        xs, _ = _x_numbers(s.index)
        fin = xs[np.isfinite(xs)]
        # a series only has values inside its own date span (+- half its typical spacing);
        # outside it (it starts later / ends earlier) the tooltip says n/a instead of reusing an end value
        tol = 0.5 * float(np.median(np.diff(fin))) if fin.size > 1 else 0.0
        span = (float(fin.min()) - tol, float(fin.max()) + tol) if fin.size else (math.inf, -math.inf)
        lookup.append((xs, s.to_numpy(dtype=float), *span))
    out: list[str] = ['<g class="hover-layer">']
    px = [sx(a) for a in anchors]
    span = (anchors[-1] - anchors[0]) if anchors.size > 1 else 0.0
    for i, a in enumerate(anchors):
        left = (px[i - 1] + px[i]) / 2 if i > 0 else px[i] - (px[1] - px[0]) / 2 if len(px) > 1 else px[i] - 4
        right = (px[i] + px[i + 1]) / 2 if i + 1 < len(px) else px[i] + (px[i] - px[i - 1]) / 2 if len(px) > 1 else px[i] + 4
        rows = [_fmt_x(a, is_date, span)]
        for name, (xs, vs, lo, hi) in zip(names, lookup):
            if xs.size == 0:
                continue
            if not (lo - 1e-9 <= a <= hi + 1e-9):
                rows.append(f"{name}: n/a")
                continue
            j = int(np.nanargmin(np.abs(xs - a)))
            v = vs[j]
            rows.append(f"{name}: {fmt(v) if math.isfinite(v) else 'n/a'}")
        out.append(f'<rect class="hz" x="{_n(left)}" y="{_n(top)}" width="{_n(max(right - left, 0.5))}" height="{_n(height)}" '
                   f'fill="#000" fill-opacity="0"><title>{_e(" | ".join(rows))}</title></rect>')
    out.append("</g>")
    return "".join(out)


# ------------------------------------------------------------------------------------------------
# Line chart
# ------------------------------------------------------------------------------------------------


def line_chart(
    series: Mapping[str, pd.Series],
    *,
    title: str,
    y_label: str = "",
    log_scale: bool = False,
    percent: bool = False,
    y_format: ValueFormat | None = None,
    colors: Mapping[str, int | str] | None = None,
    reference: float | None = None,
    width: float = 760,
    height: float = 340,
    max_points: int = MAX_POINTS,
    legend_values: bool = True,
) -> str:
    """Multi-line chart over a date (or numeric) x axis.

    * ``series``: name -> pd.Series indexed by date (DatetimeIndex, ``date`` objects or ISO strings)
      or numbers. At most 8 series are drawn (one per colour slot); NaN values break the line.
    * ``log_scale``: log10 y axis (falls back to linear when any value is <= 0).
    * ``percent``: values are fractions, tick labels are shown in % (0.05 -> 5%).
    * ``y_format``: format string (``"${:.2f}"``) or callable for tick / legend / tooltip values.
    * ``colors``: name -> slot number (1..8) or CSS colour; default = slots in insertion order.
    * ``reference``: draw a hairline at this y value (e.g. 1.0 for growth of $1, 0 for returns).
    """
    names_all = [str(k) for k in series.keys()]
    items = list(series.items())[:8]
    dropped = len(names_all) - len(items)
    cleaned: list[pd.Series] = []
    names: list[str] = []
    for name, s in items:
        cs = _clean_series(s)
        names.append(str(name))
        cleaned.append(cs)
    finite_vals = np.concatenate([c.to_numpy(dtype=float) for c in cleaned]) if cleaned else np.array([])
    finite_vals = finite_vals[np.isfinite(finite_vals)]
    if finite_vals.size == 0:
        return _empty_svg(title, width=width)

    if log_scale and (finite_vals <= 0).any():
        log_scale = False

    # x domain
    xs_list: list[np.ndarray] = []
    is_date = True
    for c in cleaned:
        xs, d = _x_numbers(c.index)
        xs_list.append(xs)
        is_date = is_date and d if len(c) else is_date
    xs_all = np.concatenate(xs_list) if xs_list else np.array([])
    xs_valid = xs_all[np.isfinite(xs_all)]
    x0, x1 = float(xs_valid.min()), float(xs_valid.max())
    if x1 <= x0:
        x0, x1 = x0 - 1.0, x1 + 1.0

    # y domain + ticks
    lo, hi = float(finite_vals.min()), float(finite_vals.max())
    if reference is not None and math.isfinite(reference) and (not log_scale or reference > 0):
        lo, hi = min(lo, reference), max(hi, reference)
    if log_scale:
        llo, lhi = math.log10(lo), math.log10(hi)
        pad = max((lhi - llo) * 0.04, 0.01)
        dlo, dhi = 10 ** (llo - pad), 10 ** (lhi + pad)
        ticks = log_ticks(dlo, dhi)
        if len(ticks) < 3:  # narrow range: round linear values placed on the log axis
            ticks = [t for t in nice_ticks(dlo, dhi, 6) if dlo <= t <= dhi and t > 0] or ticks
        tlo, thi = math.log10(dlo), math.log10(dhi)
        ty = lambda v: math.log10(v) if v > 0 else float("nan")  # noqa: E731
    else:
        pad = (hi - lo) * 0.04 if hi > lo else (abs(hi) * 0.1 or 1.0)
        ticks = nice_ticks(lo - pad if lo != 0 else lo, hi + pad if hi != 0 else hi)
        tlo, thi = ticks[0], ticks[-1]
        ty = lambda v: v  # noqa: E731
    if thi <= tlo:
        thi = tlo + 1.0
    fmt = _tick_formatter(ticks, percent, y_format)
    tick_labels = [fmt(t) for t in ticks]

    # layout
    left = max(40.0, max(_text_width(t, 11) for t in tick_labels) + 12)
    right = 16.0
    top = 28.0 if title else 8.0
    legend_svg = ""
    if len(names) >= 2 or dropped:
        legend_items = []
        for name, c in zip(names, cleaned):
            last = c.dropna()
            label = f"{name} {fmt(float(last.iloc[-1]))}" if legend_values and len(last) else name
            legend_items.append((label, _color_for(name, names, colors)))
        if dropped:
            legend_items.append((f"+{dropped} more not shown", _MUTED))
        legend_svg, lh = _legend(legend_items, left, top + 10, width - right)
        top += lh + 4
    if y_label:
        top += 18
    bottom = 26.0
    plot_h = height - top - bottom
    if plot_h < 80:
        height = top + bottom + 80
        plot_h = 80.0
    plot_w = width - left - right

    def sx(x: float) -> float:
        return left + (x - x0) / (x1 - x0) * plot_w

    def sy(v: float) -> float:
        t = ty(v)
        return top + (thi - t) / (thi - tlo) * plot_h if math.isfinite(t) else float("nan")

    desc_bits = []
    for name, c in zip(names, cleaned):
        valid = c.dropna()
        if len(valid):
            desc_bits.append(f"{name}: {len(valid)} points, last {fmt(float(valid.iloc[-1]))}, "
                             f"min {fmt(float(valid.min()))}, max {fmt(float(valid.max()))}")
    desc = f"{'Log-scale' if log_scale else 'Linear'} line chart. " + "; ".join(desc_bits)
    head, _ = _svg_open(width, height, title, desc, cls="chart chart-line")
    out = [head, _visible_title(title)]
    if y_label:
        out.append(f'<text x="0" y="{_n(top - 10)}" font-size="11" style="fill:{_MUTED}">{_e(y_label)}</text>')
    out.append(legend_svg)

    # grid + y ticks
    out.append('<g class="grid">')
    for t, lab in zip(ticks, tick_labels):
        y = sy(t)
        if not math.isfinite(y):
            continue
        out.append(f'<line x1="{_n(left)}" y1="{_n(y)}" x2="{_n(width - right)}" y2="{_n(y)}" style="stroke:{_GRID};stroke-width:1"/>')
        out.append(f'<text x="{_n(left - 6)}" y="{_n(y + 4)}" font-size="11" text-anchor="end" '
                   f'style="fill:{_MUTED};font-variant-numeric:tabular-nums">{_e(lab)}</text>')
    out.append("</g>")
    if reference is not None and math.isfinite(reference) and (not log_scale or reference > 0):
        yr = sy(reference)
        out.append(f'<line class="reference" x1="{_n(left)}" y1="{_n(yr)}" x2="{_n(width - right)}" y2="{_n(yr)}" '
                   f'style="stroke:{_AXIS};stroke-width:1.5"/>')

    # x axis
    base_y = top + plot_h
    out.append(f'<line x1="{_n(left)}" y1="{_n(base_y)}" x2="{_n(width - right)}" y2="{_n(base_y)}" style="stroke:{_AXIS};stroke-width:1"/>')
    out.append(_x_axis(x0, x1, is_date, sx, base_y, left, width - right))

    # lines
    for name, c in zip(names, cleaned):
        if not len(c):
            continue
        ds = downsample(c, max_points)
        dxs, _ = _x_numbers(ds.index)
        vals = ds.to_numpy(dtype=float)
        pxs = np.array([sx(x) for x in dxs])
        pys = np.array([sy(v) if math.isfinite(v) else float("nan") for v in vals])
        d, isolated = _path_with_gaps(pxs, pys)
        color = _color_for(name, names, colors)
        out.append(f'<g class="series" data-series="{_e(name)}">')
        if d:
            out.append(f'<path d="{d}" fill="none" style="stroke:{color};stroke-width:2;stroke-linejoin:round;stroke-linecap:round"/>')
        for px, py in isolated:
            out.append(f'<circle cx="{_n(px)}" cy="{_n(py)}" r="2.5" style="fill:{color}"/>')
        valid = np.flatnonzero(np.isfinite(pys))
        if valid.size:
            j = int(valid[-1])
            out.append(f'<circle cx="{_n(pxs[j])}" cy="{_n(pys[j])}" r="4" '
                       f'style="fill:{color};stroke:{_SURFACE};stroke-width:2"/>')
        out.append("</g>")

    out.append(_hover_bands(names, cleaned, xs_all, is_date, sx, top, plot_h, fmt))
    out.append("</svg>")
    return "".join(out)


def _color_for(name: str, names: Sequence[str], colors: Mapping[str, int | str] | None) -> str:
    if colors and name in colors:
        c = colors[name]
        if isinstance(c, int):
            return series_color(c)
        return str(c)
    return series_color(names.index(name) + 1)


def _x_axis(x0: float, x1: float, is_date: bool, sx: Callable[[float], float], base_y: float, xmin: float, xmax: float) -> str:
    out = ['<g class="x-axis">']
    if is_date:
        ticks = _date_ticks(x0, x1)
    else:
        ticks = [(t, f"{t:g}") for t in nice_ticks(x0, x1, 6) if x0 - 1e-9 <= t <= x1 + 1e-9]
    last_right = -1e9
    for t, lab in ticks:
        px = sx(t)
        if px < xmin - 0.5 or px > xmax + 0.5:
            continue
        w = _text_width(lab, 11)
        anchor = "middle"
        lx = px
        if px - w / 2 < xmin - 4:
            anchor, lx = "start", px
        elif px + w / 2 > xmax + 4:
            anchor, lx = "end", px
        left_edge = lx - (w / 2 if anchor == "middle" else 0 if anchor == "start" else w)
        if left_edge < last_right + 6:
            continue  # avoid overlapping labels
        last_right = left_edge + w
        out.append(f'<line x1="{_n(px)}" y1="{_n(base_y)}" x2="{_n(px)}" y2="{_n(base_y + 4)}" style="stroke:{_AXIS};stroke-width:1"/>')
        out.append(f'<text x="{_n(lx)}" y="{_n(base_y + 17)}" font-size="11" text-anchor="{anchor}" style="fill:{_MUTED}">{_e(lab)}</text>')
    out.append("</g>")
    return "".join(out)


# ------------------------------------------------------------------------------------------------
# Drawdown area chart
# ------------------------------------------------------------------------------------------------


def area_drawdown_chart(
    drawdown: pd.Series,
    *,
    title: str = "Drawdown",
    width: float = 760,
    height: float = 240,
    max_points: int = MAX_POINTS,
) -> str:
    """Underwater chart of a drawdown series (fractions, <= 0), max drawdown annotated."""
    dd = _clean_series(drawdown)
    vals = dd.to_numpy(dtype=float)
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        return _empty_svg(title, width=width)
    lo = min(float(finite.min()), 0.0)
    ticks = nice_ticks(lo if lo < 0 else -0.01, 0.0, 5)
    ticks = [t for t in ticks if t <= 1e-12] or [-0.01, 0.0]
    if ticks[-1] < 0:
        ticks.append(0.0)
    tlo, thi = ticks[0], 0.0
    if thi - tlo <= 0:
        tlo = -0.01
    fmt = _tick_formatter(ticks, True, None)
    tick_labels = [fmt(t) for t in ticks]
    xs_full, is_date = _x_numbers(dd.index)
    x0, x1 = float(np.nanmin(xs_full)), float(np.nanmax(xs_full))
    if x1 <= x0:
        x0, x1 = x0 - 1.0, x1 + 1.0
    left = max(40.0, max(_text_width(t, 11) for t in tick_labels) + 12)
    right, top, bottom = 16.0, 30.0 if title else 10.0, 26.0
    plot_w, plot_h = width - left - right, height - top - bottom

    def sx(x: float) -> float:
        return left + (x - x0) / (x1 - x0) * plot_w

    def sy(v: float) -> float:
        return top + (thi - v) / (thi - tlo) * plot_h

    i_min = int(np.nanargmin(np.where(np.isfinite(vals), vals, np.inf)))
    mdd, mdd_x = float(vals[i_min]), float(xs_full[i_min])
    when = _day_to_ts(mdd_x).strftime("%b %Y") if is_date else f"{mdd_x:g}"
    desc = f"Drawdown from the running peak; maximum drawdown {mdd * 100:.1f}% ({when})."
    head, _ = _svg_open(width, height, title, desc, cls="chart chart-drawdown")
    out = [head, _visible_title(title), '<g class="grid">']
    for t, lab in zip(ticks, tick_labels):
        y = sy(t)
        out.append(f'<line x1="{_n(left)}" y1="{_n(y)}" x2="{_n(width - right)}" y2="{_n(y)}" style="stroke:{_GRID};stroke-width:1"/>')
        out.append(f'<text x="{_n(left - 6)}" y="{_n(y + 4)}" font-size="11" text-anchor="end" '
                   f'style="fill:{_MUTED};font-variant-numeric:tabular-nums">{_e(lab)}</text>')
    out.append("</g>")
    base = sy(0.0)
    out.append(f'<line x1="{_n(left)}" y1="{_n(base)}" x2="{_n(width - right)}" y2="{_n(base)}" style="stroke:{_AXIS};stroke-width:1"/>')
    out.append(_x_axis(x0, x1, is_date, sx, top + plot_h, left, width - right))

    ds = downsample(dd, max_points)
    dxs, _ = _x_numbers(ds.index)
    dvals = ds.to_numpy(dtype=float)
    pxs = np.array([sx(x) for x in dxs])
    pys = np.array([sy(v) if math.isfinite(v) else float("nan") for v in dvals])
    # area: one closed polygon per contiguous finite run
    areas: list[str] = []
    run: list[tuple[float, float]] = []

    def flush() -> None:
        if len(run) >= 2:
            pts = " L".join(f"{_n(x)} {_n(y)}" for x, y in run)
            areas.append(f"M{_n(run[0][0])} {_n(base)} L{pts} L{_n(run[-1][0])} {_n(base)} Z")
        run.clear()

    for px, py in zip(pxs, pys):
        if math.isfinite(py):
            run.append((px, py))
        else:
            flush()
    flush()
    if areas:
        out.append(f'<path class="area" d="{" ".join(areas)}" style="fill:{_NEG};fill-opacity:0.14;stroke:none"/>')
    d, isolated = _path_with_gaps(pxs, pys)
    if d:
        out.append(f'<path d="{d}" fill="none" style="stroke:{_NEG};stroke-width:2;stroke-linejoin:round;stroke-linecap:round"/>')
    for px, py in isolated:
        out.append(f'<circle cx="{_n(px)}" cy="{_n(py)}" r="2.5" style="fill:{_NEG}"/>')
    if mdd < 0:
        # the extreme is marked on the line; its label sits in the title row (never over the data)
        mx, my = sx(mdd_x), sy(mdd)
        label = f"Max drawdown {mdd * 100:.1f}% ({when})"
        out.append(f'<circle cx="{_n(mx)}" cy="{_n(my)}" r="4" style="fill:{_NEG};stroke:{_SURFACE};stroke-width:2">'
                   f'<title>{_e(label)}</title></circle>')
        tx = width - right
        out.append(f'<circle cx="{_n(tx - _text_width(label, 12) - 10)}" cy="{_n(14)}" r="4" style="fill:{_NEG}"/>')
        out.append(f'<text class="annotation" x="{_n(tx)}" y="{_n(18)}" font-size="12" text-anchor="end" '
                   f'style="fill:{_TEXT2}">{_e(label)}</text>')
    out.append(_hover_bands(["Drawdown"], [dd], xs_full, is_date, sx, top, plot_h, fmt))
    out.append("</svg>")
    return "".join(out)


# ------------------------------------------------------------------------------------------------
# Bar charts
# ------------------------------------------------------------------------------------------------


def _finite_or_none(v: object) -> float | None:
    try:
        f = float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def bar_chart(
    labels: Sequence[str],
    values: Sequence[float | None],
    *,
    title: str,
    highlight_negative: bool = True,
    value_format: ValueFormat = "{:.1f}",
    y_label: str = "",
    color: int | str = 1,
    width: float = 760,
    height: float = 300,
) -> str:
    """Vertical bars from a zero baseline; values on the caps; negatives in the 'negative' colour.

    ``value_format`` is a format string (``"{:.1f}%"``) or a callable. ``None`` / NaN values draw
    no bar and are labelled "n/a".
    """
    labels = [str(x) for x in labels]
    vals = [_finite_or_none(v) for v in values]
    if len(labels) != len(vals):
        raise ValueError("labels and values must have the same length")
    finite = [v for v in vals if v is not None]
    if not finite:
        return _empty_svg(title, width=width)
    lo, hi = min(0.0, min(finite)), max(0.0, max(finite))
    span = hi - lo or 1.0
    lo_p = lo - span * 0.12 if lo < 0 else lo
    hi_p = hi + span * 0.12 if hi > 0 else hi
    ticks = nice_ticks(lo_p, hi_p, 5)
    tlo, thi = ticks[0], ticks[-1]
    tick_fmt = _tick_formatter(ticks, False, None)
    tick_labels = [tick_fmt(t) for t in ticks]
    left = max(40.0, max(_text_width(t, 11) for t in tick_labels) + 12)
    right, top, bottom = 12.0, (30.0 if title else 10.0) + (14.0 if y_label else 0.0), 30.0
    plot_w, plot_h = width - left - right, height - top - bottom
    n = len(labels)
    band = plot_w / max(n, 1)
    bar_w = min(32.0, band * 0.6)
    base_color = series_color(color) if isinstance(color, int) else color

    def sy(v: float) -> float:
        return top + (thi - v) / (thi - tlo) * plot_h

    desc = "Bar chart. " + "; ".join(f"{lab}: {_fmt_value(v, value_format)}" for lab, v in zip(labels, vals))
    head, _ = _svg_open(width, height, title, desc, cls="chart chart-bar")
    out = [head, _visible_title(title)]
    if y_label:
        out.append(f'<text x="0" y="{_n(top - 8)}" font-size="11" style="fill:{_MUTED}">{_e(y_label)}</text>')
    out.append('<g class="grid">')
    for t, lab in zip(ticks, tick_labels):
        y = sy(t)
        out.append(f'<line x1="{_n(left)}" y1="{_n(y)}" x2="{_n(width - right)}" y2="{_n(y)}" style="stroke:{_GRID};stroke-width:1"/>')
        out.append(f'<text x="{_n(left - 6)}" y="{_n(y + 4)}" font-size="11" text-anchor="end" '
                   f'style="fill:{_MUTED};font-variant-numeric:tabular-nums">{_e(lab)}</text>')
    out.append("</g>")
    y0 = sy(0.0)
    show_values = n <= 16
    for i, (lab, v) in enumerate(zip(labels, vals)):
        cx = left + band * (i + 0.5)
        x = cx - bar_w / 2
        tip = f"{lab}: {_fmt_value(v, value_format)}"
        out.append('<g class="bar">')
        if v is not None:
            neg = v < 0
            fill = _NEG if (neg and highlight_negative) else base_color
            ye = sy(v)
            out.append(f'<path d="{_bar_path(x, y0, ye, bar_w)}" style="fill:{fill}"><title>{_e(tip)}</title></path>')
            # transparent, larger hit target
            out.append(f'<rect class="hit" x="{_n(cx - band / 2)}" y="{_n(top)}" width="{_n(band)}" height="{_n(plot_h)}" '
                       f'fill="#000" fill-opacity="0"><title>{_e(tip)}</title></rect>')
            if show_values:
                ly = ye - 6 if not neg else ye + 14
                out.append(f'<text x="{_n(cx)}" y="{_n(ly)}" font-size="11" text-anchor="middle" '
                           f'style="fill:{_TEXT2}">{_e(_fmt_value(v, value_format))}</text>')
        else:
            out.append(f'<text x="{_n(cx)}" y="{_n(y0 - 6)}" font-size="11" text-anchor="middle" style="fill:{_MUTED}">n/a'
                       f'<title>{_e(tip)}</title></text>')
        out.append(f'<text x="{_n(cx)}" y="{_n(top + plot_h + 18)}" font-size="11" text-anchor="middle" '
                   f'style="fill:{_TEXT2}">{_e(_truncate(lab, band - 4, 11))}</text>')
        out.append("</g>")
    out.append(f'<line x1="{_n(left)}" y1="{_n(y0)}" x2="{_n(width - right)}" y2="{_n(y0)}" style="stroke:{_AXIS};stroke-width:1"/>')
    out.append("</svg>")
    return "".join(out)


def horizontal_bar_chart(
    labels: Sequence[str],
    values: Sequence[float | None],
    errors: Sequence[float | None] | None = None,
    *,
    annotations: Sequence[str | None] | None = None,
    title: str = "",
    value_format: ValueFormat = "{:.2f}",
    highlight_negative: bool = False,
    color: int | str = 1,
    width: float = 760,
    row_height: float = 30,
) -> str:
    """Horizontal bars from a zero line (e.g. factor betas).

    ``errors`` are symmetric whisker half-widths in value units (e.g. 1.96 x standard error);
    ``annotations`` are short texts appended to each value label (e.g. ``"t=3.1"``).
    """
    labels = [str(x) for x in labels]
    vals = [_finite_or_none(v) for v in values]
    n = len(labels)
    if len(vals) != n:
        raise ValueError("labels and values must have the same length")
    errs = [(_finite_or_none(e) if errors is not None and i < len(errors) else None) for i, e in
            enumerate(errors if errors is not None else [None] * n)]
    errs = [abs(e) if e is not None else None for e in errs] + [None] * (n - len(errs))
    notes = list(annotations) if annotations is not None else [None] * n
    notes += [None] * (n - len(notes))
    finite = [v for v in vals if v is not None]
    if not finite:
        return _empty_svg(title, width=width)
    ext_lo = min([0.0] + [v - (e or 0.0) for v, e in zip(vals, errs) if v is not None])
    ext_hi = max([0.0] + [v + (e or 0.0) for v, e in zip(vals, errs) if v is not None])
    span = ext_hi - ext_lo or 1.0
    # room for the value labels outside the bar ends
    lo_p = ext_lo - span * 0.28 if ext_lo < 0 else ext_lo
    hi_p = ext_hi + span * 0.28 if ext_hi > 0 else ext_hi + span * 0.05
    ticks = nice_ticks(lo_p, hi_p, 5)
    tlo, thi = ticks[0], ticks[-1]
    tick_fmt = _tick_formatter(ticks, False, None)
    label_w = min(200.0, max(_text_width(lab, 12) for lab in labels) + 14)
    left, right = max(60.0, label_w), 16.0
    top = 30.0 if title else 10.0
    plot_h = n * row_height
    height = top + plot_h + 30
    plot_w = width - left - right
    base_color = series_color(color) if isinstance(color, int) else color

    def sx(v: float) -> float:
        return left + (v - tlo) / (thi - tlo) * plot_w

    desc = "Horizontal bar chart. " + "; ".join(
        f"{lab}: {_fmt_value(v, value_format)}" + (f" ({a})" if a else "") for lab, v, a in zip(labels, vals, notes)
    )
    head, _ = _svg_open(width, height, title, desc, cls="chart chart-hbar")
    out = [head, _visible_title(title), '<g class="grid">']
    for t in ticks:
        x = sx(t)
        out.append(f'<line x1="{_n(x)}" y1="{_n(top)}" x2="{_n(x)}" y2="{_n(top + plot_h)}" style="stroke:{_GRID};stroke-width:1"/>')
        out.append(f'<text x="{_n(x)}" y="{_n(top + plot_h + 16)}" font-size="11" text-anchor="middle" '
                   f'style="fill:{_MUTED};font-variant-numeric:tabular-nums">{_e(tick_fmt(t))}</text>')
    out.append("</g>")
    x_zero = sx(0.0)
    bar_h = min(18.0, row_height * 0.6)
    for i, (lab, v, e, note) in enumerate(zip(labels, vals, errs, notes)):
        cy = top + row_height * (i + 0.5)
        out.append('<g class="bar">')
        out.append(f'<text x="{_n(left - 8)}" y="{_n(cy + 4)}" font-size="12" text-anchor="end" style="fill:{_TEXT2}">'
                   f'{_e(_truncate(lab, left - 12, 12))}</text>')
        text = _fmt_value(v, value_format) + (f"  ({note})" if note else "")
        tip = f"{lab}: {text}" + (f" ± {_fmt_value(e, value_format)}" if e else "")
        if v is None:
            out.append(f'<text x="{_n(x_zero + 6)}" y="{_n(cy + 4)}" font-size="11" style="fill:{_MUTED}">n/a<title>{_e(tip)}</title></text>')
            out.append("</g>")
            continue
        fill = _NEG if (v < 0 and highlight_negative) else base_color
        out.append(f'<path d="{_bar_path(cy - bar_h / 2, x_zero, sx(v), bar_h, horizontal=True)}" style="fill:{fill}">'
                   f'<title>{_e(tip)}</title></path>')
        end = v
        if e:
            a, b = sx(v - e), sx(v + e)
            out.append(f'<line class="whisker" x1="{_n(a)}" y1="{_n(cy)}" x2="{_n(b)}" y2="{_n(cy)}" style="stroke:{_TEXT2};stroke-width:1.5"/>')
            out.append(f'<line x1="{_n(a)}" y1="{_n(cy - 5)}" x2="{_n(a)}" y2="{_n(cy + 5)}" style="stroke:{_TEXT2};stroke-width:1.5"/>')
            out.append(f'<line x1="{_n(b)}" y1="{_n(cy - 5)}" x2="{_n(b)}" y2="{_n(cy + 5)}" style="stroke:{_TEXT2};stroke-width:1.5"/>')
            end = v + e if v >= 0 else v - e
        if v >= 0:
            out.append(f'<text x="{_n(sx(end) + 6)}" y="{_n(cy + 4)}" font-size="11" style="fill:{_TEXT}">{_e(text)}</text>')
        else:
            out.append(f'<text x="{_n(sx(end) - 6)}" y="{_n(cy + 4)}" font-size="11" text-anchor="end" style="fill:{_TEXT}">{_e(text)}</text>')
        out.append(f'<rect class="hit" x="{_n(left)}" y="{_n(cy - row_height / 2)}" width="{_n(plot_w)}" height="{_n(row_height)}" '
                   f'fill="#000" fill-opacity="0"><title>{_e(tip)}</title></rect>')
        out.append("</g>")
    out.append(f'<line x1="{_n(x_zero)}" y1="{_n(top)}" x2="{_n(x_zero)}" y2="{_n(top + plot_h)}" style="stroke:{_AXIS};stroke-width:1.5"/>')
    out.append("</svg>")
    return "".join(out)


# ------------------------------------------------------------------------------------------------
# Sparkline
# ------------------------------------------------------------------------------------------------


def sparkline(
    series: pd.Series,
    *,
    title: str = "Trend",
    width: float = 120,
    height: float = 28,
    color: int | str = 1,
    max_points: int = 200,
) -> str:
    """Tiny axis-free trend line (NaN gaps break it), end point marked."""
    s = _clean_series(series)
    vals = s.to_numpy(dtype=float)
    finite = vals[np.isfinite(vals)]
    cid = _new_id("s")
    style = f"width:100%;max-width:{_n(width)}px;height:auto;display:inline-block;vertical-align:middle"
    head = (
        f'<svg xmlns="http://www.w3.org/2000/svg" class="sparkline" viewBox="0 0 {_n(width)} {_n(height)}" '
        f'width="100%" preserveAspectRatio="xMidYMid meet" role="img" aria-labelledby="{cid}-t" style="{style}">'
    )
    if finite.size == 0:
        return head + f'<title id="{cid}-t">{_e(title)}: no data</title></svg>'
    first, last = float(finite[0]), float(finite[-1])
    head += f'<title id="{cid}-t">{_e(title)}: from {first:,.4g} to {last:,.4g}</title>'
    ds = downsample(s, max_points)
    xs, _ = _x_numbers(ds.index)
    if xs.size and np.isfinite(xs).all():
        x0, x1 = float(xs.min()), float(xs.max())
    else:
        xs = np.arange(len(ds), dtype=float)
        x0, x1 = 0.0, float(max(len(ds) - 1, 1))
    if x1 <= x0:
        x1 = x0 + 1.0
    lo, hi = float(finite.min()), float(finite.max())
    if hi <= lo:
        lo, hi = lo - 1.0, hi + 1.0
    pad = 3.0
    pxs = np.array([pad + (x - x0) / (x1 - x0) * (width - 2 * pad) for x in xs])
    pys = np.array([pad + (hi - v) / (hi - lo) * (height - 2 * pad) if math.isfinite(v) else float("nan")
                    for v in ds.to_numpy(dtype=float)])
    c = series_color(color) if isinstance(color, int) else color
    d, isolated = _path_with_gaps(pxs, pys)
    out = [head]
    if d:
        out.append(f'<path d="{d}" fill="none" vector-effect="non-scaling-stroke" '
                   f'style="stroke:{c};stroke-width:1.5;stroke-linejoin:round;stroke-linecap:round"/>')
    for px, py in isolated:
        out.append(f'<circle cx="{_n(px)}" cy="{_n(py)}" r="1.5" style="fill:{c}"/>')
    valid = np.flatnonzero(np.isfinite(pys))
    if valid.size:
        j = int(valid[-1])
        out.append(f'<circle cx="{_n(pxs[j])}" cy="{_n(pys[j])}" r="2.5" style="fill:{c}"/>')
    out.append("</svg>")
    return "".join(out)
