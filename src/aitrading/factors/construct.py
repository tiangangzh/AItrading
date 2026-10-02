"""Build Fama-French-style factors from the user's own universe.

The idea lab uses this to *build* an academic factor model from point-in-time data on the stocks
the user can actually see, and then to compare the result with the official Kenneth French series
(:mod:`aitrading.factors.french`) via :func:`compare_with_official`.

Methodology
-----------
References: Fama & French (1993) "Common risk factors in the returns on stocks and bonds",
*JFE* 33; Fama & French (2015) "A five-factor asset pricing model", *JFE* 116; Carhart (1997)
"On persistence in mutual fund performance", *JF* 52; and the construction notes on the Kenneth
French Data Library web site.

Timing. ``returns.loc[t]`` is the return over the period *ending* at ``t``; ``market_cap.loc[t]``
and every characteristic ``.loc[t]`` are values known at the end of ``t``. A portfolio formed at
``t`` uses only data at or before ``t`` and earns the returns of periods ``t+1`` onwards (no
look-ahead). Characteristic panels are joined *as of* each return date (the latest row at or
before it), so annual book equity can be passed on its own (already lagged, point-in-time) dates;
a row that exists with NaN stays NaN.

Market. ``Mkt-RF`` = value-weighted return of every name with a return in period ``s`` and a
positive market cap at the end of period ``s-1``, minus ``RF`` of period ``s`` (when ``rf`` is
not supplied, the raw market return is reported and a warning is issued). ``RF`` must be a rate per
return period: a finer ``rf`` (daily RF, monthly returns) is compounded over each return period
``(previous date, date]`` (calendar months for monthly returns) with a warning, and periods it only
partly covers are NaN; a coarser ``rf`` (monthly RF, daily returns) raises ``ValueError``.

Value weights. Every portfolio return is value-weighted with weights = market cap at the end of
the previous period (``market_cap.shift(1)``), renormalised over the members that have a return in
that period. Within an annual holding year this is the buy-and-hold return of the June portfolio
(caps drift with returns); names without a return in a period (e.g. delisted) drop out of that
period - delisting returns are not imputed.

Formation.
* ``annual_june`` (Fama-French): portfolios are formed at the end of June of each year ``y`` (the
  last index date in June) and held from July ``y`` to June ``y+1``. Size = market cap at the end
  of June ``y``; B/M = book equity known in June ``y`` (lagged by the caller, e.g. the fiscal year
  ending in calendar ``y-1``) divided by the market cap at the end of December ``y-1`` (the last
  index date in December; if December ``y-1`` is not in the sample the June cap is used and a
  warning is issued). OP and INV are the values known at the June formation date.
* ``monthly``: portfolios are re-formed at the end of every period ``t`` with the characteristics
  at ``t`` (B/M = book equity / current market cap) and held for period ``t+1``.
* Momentum (Carhart's UMD / French's ``Mom``) is always re-formed every period (as in the French
  library) using ``momentum_12_1`` at ``t`` (the caller's t-12..t-1 formation return known at
  ``t``), whichever ``formation`` is chosen.

Breakpoints. Each 2x3 sort uses the names that can enter it (positive market cap at formation and
a finite characteristic; for B/M, book equity <= 0 and a non-positive December cap are excluded).
Size: the median market cap of the NYSE names among them when at least ``MIN_NYSE_NAMES`` (20)
NYSE names qualify, otherwise the median of all of them (a warning is issued). B/M, OP, INV and
momentum: 30th/70th percentiles (``numpy.percentile``, linear interpolation) of the same names,
under the same 20-NYSE-name rule. This is the sample the standard WRDS replication of the French
factors uses (NYSE names with positive BE and ME). Assignment follows the same convention:
Small if cap <= median else Big; Low if x <= p30, Middle if p30 < x <= p70, High if x > p70.
OP and INV are used as given - set OP to NaN when book equity is not positive.

Sorts are independent 2x3 sorts: size (S, B) x tercile of the characteristic:
B/M -> L, M, H; OP -> W (weak), N, R (robust); INV -> C (conservative = low asset growth), N,
A (aggressive); momentum -> Down, Mid, Up.

* ``SMB`` (ff3, carhart4) = 1/3 (S/L + S/M + S/H) - 1/3 (B/L + B/M + B/H)
* ``HML`` = 1/2 (S/H + B/H) - 1/2 (S/L + B/L)
* ``SMB`` (ff5) = 1/3 (SMB_BM + SMB_OP + SMB_INV), each SMB_x the size spread of its 2x3 sort
* ``RMW`` = 1/2 (S/R + B/R) - 1/2 (S/W + B/W)
* ``CMA`` = 1/2 (S/C + B/C) - 1/2 (S/A + B/A)
* ``Mom`` = 1/2 (S/Up + B/Up) - 1/2 (S/Down + B/Down)

A portfolio with fewer than ``min_names_per_portfolio`` names (with a return and a prior cap) in a
period makes every factor that uses it NaN for that period, and a warning is issued.

Differences from the official series to expect: the universe is the user's (often large caps
only, so "Small" is mid-cap), breakpoints usually fall back to all names, book equity comes from
filings rather than Compustat's definition, and no delisting returns are applied.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from aitrading.backtest.models import FactorConstructionCheck
from aitrading.factors.french import (
    _SPACING_RANK,
    _canon,
    _compound_onto,
    _find_column,
    _spacing,
    factor_columns,
)

__all__ = [
    "MIN_NYSE_NAMES",
    "NYSE_CODES",
    "CharacteristicsPanel",
    "SortBreakpoints",
    "compare_with_official",
    "construct_factors",
    "two_by_three_sort",
]

FactorModel = Literal["capm", "ff3", "carhart4", "ff5"]
Formation = Literal["annual_june", "monthly"]

MIN_NYSE_NAMES = 20
#: Exchange labels treated as the NYSE main board (``exchange`` values are upper-cased and stripped).
#: NYSE American (AMEX) and NYSE Arca are *not* NYSE for breakpoint purposes.
NYSE_CODES = frozenset({"NYSE", "N", "XNYS", "NYQ", "NEW YORK STOCK EXCHANGE"})

_SOURCE = "constructed from universe"


@dataclass
class CharacteristicsPanel:
    """Point-in-time inputs for factor construction (all frames: DatetimeIndex x ticker columns).

    * ``returns`` - period returns (fractions), the period ending at each date; monthly by default.
    * ``market_cap`` - market cap at each period end (any consistent currency unit).
    * ``book_equity`` - book equity as known at each date (already lagged by the caller);
      required for ff3 / carhart4 / ff5, may be None for capm.
    * ``operating_profitability`` - OP known at each date (ff5).
    * ``investment`` - asset growth known at each date (ff5).
    * ``momentum_12_1`` - formation-period return (t-12..t-1) known at each date (carhart4).
    * ``exchange`` - ticker -> listing exchange ('NYSE', 'NASDAQ', ...), for NYSE breakpoints.
    * ``rf`` - risk-free return per period (fractions). At the returns' frequency it is matched by
      date (by calendar month for monthly data); a finer series (e.g. the French daily RF with
      monthly returns) is compounded over each return period; a coarser one raises ``ValueError``.
    """

    returns: pd.DataFrame
    market_cap: pd.DataFrame
    book_equity: pd.DataFrame | None
    operating_profitability: pd.DataFrame | None = None
    investment: pd.DataFrame | None = None
    momentum_12_1: pd.DataFrame | None = None
    exchange: pd.Series | None = None
    rf: pd.Series | None = None


@dataclass(frozen=True)
class SortBreakpoints:
    """Breakpoints of one 2x3 sort: size median, 30th and 70th characteristic percentiles."""

    size: float
    low: float
    high: float
    from_nyse: bool = False  # True when computed from NYSE names only (>= MIN_NYSE_NAMES of them)
    n_names: int = 0         # names that entered the sort
    n_nyse: int = 0          # NYSE names among them


# ------------------------------------------------------------------------------------ sorting


def _sort_codes(
    size: np.ndarray, char: np.ndarray, nyse: np.ndarray | None, min_nyse: int
) -> tuple[np.ndarray, SortBreakpoints | None]:
    """2x3 codes (size_code * 3 + tercile, -1 = not sorted) for one formation date."""
    codes = np.full(size.shape, -1, dtype=np.int64)
    with np.errstate(invalid="ignore"):
        ok = np.isfinite(size) & (size > 0) & np.isfinite(char)
    if not ok.any():
        return codes, None
    ny = np.zeros_like(ok) if nyse is None else (nyse & ok)
    n_nyse = int(ny.sum())
    base = ny if n_nyse >= min_nyse else ok
    size_bp = float(np.percentile(size[base], 50.0))
    lo, hi = (float(x) for x in np.percentile(char[base], [30.0, 70.0]))
    size_code = np.where(size <= size_bp, 0, 1)
    char_code = np.where(char <= lo, 0, np.where(char <= hi, 1, 2))
    codes[ok] = (size_code * 3 + char_code)[ok]
    return codes, SortBreakpoints(size_bp, lo, hi, n_nyse >= min_nyse, int(ok.sum()), n_nyse)


def _nyse_mask(exchange: pd.Series | None, tickers: pd.Index) -> np.ndarray | None:
    if exchange is None:
        return None
    ex = exchange.reindex(tickers)
    return np.array([isinstance(v, str) and v.strip().upper() in NYSE_CODES for v in ex], dtype=bool)


def two_by_three_sort(
    size: pd.Series,
    characteristic: pd.Series,
    *,
    exchange: pd.Series | None = None,
    labels: tuple[str, str, str] = ("L", "M", "H"),
    min_nyse: int = MIN_NYSE_NAMES,
) -> tuple[pd.Series, SortBreakpoints | None]:
    """Independent 2x3 sort of one cross-section (see the module docstring for the rules).

    Returns ``(labels, breakpoints)``: labels such as ``"S/L"`` / ``"B/H"`` indexed by ticker (None
    for names that cannot be sorted) and the breakpoints used (None when nothing is sortable).
    Names enter the sort with a positive size and a finite characteristic; the size median and the
    30/70 breakpoints are computed from the NYSE names among them when at least ``min_nyse``
    qualify, otherwise from all of them. ``exchange`` maps ticker -> exchange label
    (see :data:`NYSE_CODES`).
    """
    tickers = size.index.union(characteristic.index)
    s = size.reindex(tickers).to_numpy(dtype=float)
    c = characteristic.reindex(tickers).to_numpy(dtype=float)
    codes, bps = _sort_codes(s, c, _nyse_mask(exchange, tickers), min_nyse)
    names = [f"{sz}/{lab}" for sz in ("S", "B") for lab in labels]
    out = pd.Series([names[k] if k >= 0 else None for k in codes], index=tickers, dtype=object)
    return out, bps


# ------------------------------------------------------------------------------------ helpers


def _asof(frame: pd.DataFrame | None, index: pd.DatetimeIndex, columns: pd.Index, name: str) -> np.ndarray | None:
    """Values of ``frame`` as of each date of ``index`` (latest row at or before it)."""
    if frame is None:
        return None
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError(f"{name} must have a DatetimeIndex")
    f = frame.reindex(columns=columns)
    f = f[~f.index.duplicated(keep="last")].sort_index()
    return f.reindex(index, method="ffill").to_numpy(dtype=float)


def _infer_monthly(index: pd.Index) -> bool:
    if len(index) < 2:
        return True
    gaps = np.diff(index.values).astype("timedelta64[D]").astype(float)
    return float(np.median(gaps)) >= 20.0


def _align_series(s: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """Exact date alignment, falling back to calendar-month alignment for monthly series."""
    s = s[~s.index.duplicated(keep="last")].sort_index()
    out = s.reindex(index)
    if out.isna().any() and _infer_monthly(index) and isinstance(s.index, pd.DatetimeIndex):
        by_month = s.groupby(s.index.to_period("M")).last()
        fill = pd.Series(by_month.reindex(index.to_period("M")).to_numpy(), index=index)
        out = out.fillna(fill)
    return out.astype(float)


def _align_rf(rf: pd.Series, index: pd.DatetimeIndex) -> tuple[np.ndarray, list[str]]:
    """Per-period risk-free returns on the returns ``index`` (+ warnings).

    The spacing of ``rf`` and of the returns is inferred from the median gap between dates. Same
    spacing: :func:`_align_series`. RF finer than the returns (e.g. the French daily RF with monthly
    returns): RF is compounded over each return period (:func:`aitrading.factors.french._compound_onto`;
    periods RF only partly covers are NaN) and a warning says so. RF coarser than the returns (a
    monthly RF with daily returns): ``ValueError`` - a monthly rate is not a daily one.
    """
    if isinstance(rf.index, pd.PeriodIndex):
        rf = pd.Series(rf.to_numpy(), index=rf.index.to_timestamp(how="end").normalize(), name=rf.name)
    if not isinstance(rf.index, pd.DatetimeIndex):
        raise TypeError("panel.rf must have a DatetimeIndex")
    rf = rf[~rf.index.duplicated(keep="last")].sort_index().astype(float)
    r_sp, s_sp = _spacing(index), _spacing(rf.index)
    if r_sp is None or s_sp is None or r_sp == s_sp:
        return _align_series(rf, index).to_numpy(), []
    if _SPACING_RANK[s_sp] > _SPACING_RANK[r_sp]:
        raise ValueError(
            f"panel.rf is {s_sp} but panel.returns is {r_sp}: a {s_sp} risk-free rate cannot be used as the "
            f"per-period rate of {r_sp} returns. Supply RF at the returns' frequency (e.g. the RF column of "
            f"load_french_factors(model, 'daily') for daily returns)."
        )
    comp, n_partial = _compound_onto(rf.to_frame("RF"), index, r_sp)
    msg = f"panel.rf is {s_sp} but panel.returns is {r_sp}: RF was compounded over each return period"
    if n_partial:
        msg += f" ({n_partial} period(s) only partly covered by RF were set to NaN)"
    return comp["RF"].to_numpy(dtype=float), [msg]


def _fmt(ts: pd.Timestamp) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%d")


def _vw_portfolios(
    R: np.ndarray, W: np.ndarray, held: np.ndarray, n_ports: int
) -> tuple[np.ndarray, np.ndarray]:
    """Value-weighted returns and name counts of portfolios ``0..n_ports-1`` (rows x ports)."""
    valid = np.isfinite(R) & np.isfinite(W) & (W > 0)
    Rz = np.where(valid, R, 0.0)
    rets = np.full((R.shape[0], n_ports), np.nan)
    counts = np.zeros((R.shape[0], n_ports), dtype=np.int64)
    for k in range(n_ports):
        m = valid & (held == k)
        w = np.where(m, W, 0.0)
        den = w.sum(axis=1)
        num = (w * Rz).sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            rets[:, k] = np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)
        counts[:, k] = m.sum(axis=1)
    return rets, counts


@dataclass
class _SortSpec:
    key: str          # "bm", "op", "inv", "mom"
    title: str        # human-readable, for warnings
    labels: tuple[str, str, str]


_SORTS = {
    "bm": _SortSpec("bm", "size-B/M", ("L", "M", "H")),
    "op": _SortSpec("op", "size-OP", ("W", "N", "R")),
    "inv": _SortSpec("inv", "size-INV", ("C", "N", "A")),
    "mom": _SortSpec("mom", "size-momentum", ("Down", "Mid", "Up")),
}


class _Builder:
    def __init__(self, panel: CharacteristicsPanel, min_names: int) -> None:
        R = panel.returns
        if not isinstance(R.index, pd.DatetimeIndex):
            raise TypeError("panel.returns must have a DatetimeIndex")
        R = R[~R.index.duplicated(keep="last")].sort_index()
        self.index: pd.DatetimeIndex = R.index
        self.tickers: pd.Index = R.columns
        self.R = R.to_numpy(dtype=float)
        me = _asof(panel.market_cap, self.index, self.tickers, "market_cap")
        assert me is not None
        self.ME = me
        self.W = np.vstack([np.full((1, me.shape[1]), np.nan), me[:-1]]) if len(me) else me  # prior-period cap
        self.panel = panel
        self.min_names = min_names
        self.nyse = _nyse_mask(panel.exchange, self.tickers)
        self.warnings: list[str] = []
        self._fallbacks: dict[str, list[pd.Timestamp]] = {}
        self._n_formations: dict[str, int] = {}

    # -- formation dates ------------------------------------------------------------------
    def june_rows(self) -> list[int]:
        idx = self.index
        if len(idx) == 0:
            return []
        ym = idx.year * 100 + idx.month
        last_in_month = np.r_[ym[1:] != ym[:-1], True]
        return [int(i) for i in np.flatnonzero((idx.month == 6) & last_in_month)]

    def december_row(self, year: int) -> int | None:
        idx = self.index
        rows = np.flatnonzero((idx.year == year) & (idx.month == 12))
        return int(rows[-1]) if rows.size else None

    # -- sorting ----------------------------------------------------------------------------
    def _record(self, label: str, bps: SortBreakpoints | None, row: int) -> None:
        if bps is None or self.nyse is None:
            return
        self._n_formations[label] = self._n_formations.get(label, 0) + 1
        if not bps.from_nyse:
            self._fallbacks.setdefault(label, []).append(self.index[row])

    def held_codes(
        self, sort: _SortSpec, rows: list[int], size_rows: list[np.ndarray], chars: list[np.ndarray]
    ) -> tuple[np.ndarray, int | None]:
        """Codes held in each period (the portfolio formed at the latest formation row before it)
        and the first formation row at which any name could be sorted (None if never)."""
        n, m = self.R.shape
        held = np.full((n, m), np.nan)
        first: int | None = None
        for k, row in enumerate(rows):
            end = rows[k + 1] if k + 1 < len(rows) else n - 1
            if end <= row:
                continue  # formed on the last date: never held
            codes, bps = _sort_codes(size_rows[k], chars[k], self.nyse, MIN_NYSE_NAMES)
            self._record(sort.title, bps, row)
            if bps is not None and first is None:
                first = row
            held[row + 1 : end + 1, :] = np.where(codes >= 0, codes, np.nan)
        return held, first

    def portfolios(
        self, sort: _SortSpec, held_first: tuple[np.ndarray, int | None], factors_using: dict[int, list[str]]
    ) -> np.ndarray:
        """Portfolio returns (rows x 6); thin portfolios (used by a factor) -> NaN + one warning."""
        held, first_row = held_first
        rets, counts = _vw_portfolios(self.R, self.W, held, 6)
        if first_row is None:
            return rets
        thin = counts < self.min_names
        thin[: first_row + 1, :] = False  # nothing was held before the first effective formation
        unused = [k for k in range(6) if not factors_using.get(k)]
        thin[:, unused] = False
        if thin.any():
            names = [f"{sz}/{lab}" for sz in ("S", "B") for lab in sort.labels]
            parts = [f"{names[k]}: {int(thin[:, k].sum())}" for k in range(6) if thin[:, k].any()]
            affected = sorted({f for k in range(6) if thin[:, k].any() for f in factors_using.get(k, [])})
            rows_thin = np.flatnonzero(thin.any(axis=1))
            self.warnings.append(
                f"{sort.title} portfolios with fewer than {self.min_names} names in {rows_thin.size} period(s) "
                f"(first {_fmt(self.index[rows_thin[0]])}; periods per portfolio - {', '.join(parts)}); "
                f"{', '.join(affected)} set to NaN there"
            )
            rets = np.where(thin, np.nan, rets)
        return rets

    def finish_warnings(self) -> None:
        if self.panel.exchange is None:
            self.warnings.append(
                "no exchange data: breakpoints use all names instead of NYSE names as in Fama-French, "
                "so the size median and 30/70 breakpoints are tilted toward whatever the universe holds"
            )
            return
        for label, dates in self._fallbacks.items():
            self.warnings.append(
                f"{label} breakpoints: fewer than {MIN_NYSE_NAMES} NYSE names with data at {len(dates)} of "
                f"{self._n_formations.get(label, len(dates))} formation date(s) (first {_fmt(dates[0])}); "
                f"the size median and 30/70 percentiles of all names were used instead"
            )


def _spread(rets: np.ndarray, high: int, low: int) -> np.ndarray:
    """1/2 (S/high + B/high) - 1/2 (S/low + B/low) for tercile codes ``high`` / ``low``."""
    return 0.5 * (rets[:, high] + rets[:, 3 + high]) - 0.5 * (rets[:, low] + rets[:, 3 + low])


def _smb(rets: np.ndarray) -> np.ndarray:
    return (rets[:, 0] + rets[:, 1] + rets[:, 2]) / 3.0 - (rets[:, 3] + rets[:, 4] + rets[:, 5]) / 3.0


# ------------------------------------------------------------------------------------ main API


def construct_factors(
    panel: CharacteristicsPanel,
    model: FactorModel,
    *,
    formation: Formation = "annual_june",
    min_names_per_portfolio: int = 5,
) -> tuple[pd.DataFrame, list[str]]:
    """Construct ``model``'s factors from ``panel`` (methodology in the module docstring).

    Returns ``(factors, warnings)``: a float frame indexed like ``panel.returns`` (sorted) with
    columns ``factor_columns(model)`` (+ ``"RF"`` when ``panel.rf`` is given), in fractions per
    period. Periods before the first formation (and the first period, which has no prior market
    cap) are NaN. Raises ``ValueError`` for an unknown model / formation, missing characteristics,
    or a ``panel.rf`` coarser than the returns (see :class:`CharacteristicsPanel`).
    """
    if model not in ("capm", "ff3", "carhart4", "ff5"):
        raise ValueError(f"unknown factor model {model!r}; expected capm, ff3, carhart4 or ff5")
    if formation not in ("annual_june", "monthly"):
        raise ValueError(f"formation must be 'annual_june' or 'monthly', got {formation!r}")
    if min_names_per_portfolio < 1:
        raise ValueError("min_names_per_portfolio must be >= 1")
    if model in ("ff3", "carhart4", "ff5") and panel.book_equity is None:
        raise ValueError(f"{model} needs panel.book_equity")
    if model == "ff5" and (panel.operating_profitability is None or panel.investment is None):
        raise ValueError("ff5 needs panel.operating_profitability and panel.investment")
    if model == "carhart4" and panel.momentum_12_1 is None:
        raise ValueError("carhart4 needs panel.momentum_12_1")

    b = _Builder(panel, min_names_per_portfolio)
    cols = factor_columns(model)
    n = len(b.index)
    out: dict[str, np.ndarray] = {}  # keyed by canonical name: MKTRF, SMB, HML, RMW, CMA, MOM
    mkt_name = cols[0]

    # --- market ---------------------------------------------------------------------------
    mkt, mcount = _vw_portfolios(b.R, b.W, np.zeros_like(b.R), 1)
    mkt = mkt[:, 0]
    thin_mkt = (mcount[:, 0] < min_names_per_portfolio)
    thin_mkt[:1] = False
    if thin_mkt.any():
        thin_rows = np.flatnonzero(thin_mkt)
        b.warnings.append(
            f"market portfolio had fewer than {min_names_per_portfolio} names in {thin_rows.size} period(s) "
            f"(first {_fmt(b.index[thin_rows[0]])}); {mkt_name} set to NaN there"
        )
        mkt = np.where(thin_mkt, np.nan, mkt)
    rf: np.ndarray | None = None
    if panel.rf is not None:
        rf, rf_warnings = _align_rf(panel.rf, b.index)
        b.warnings.extend(rf_warnings)
    if rf is not None:
        out["MKTRF"] = mkt - rf
        missing_rf = np.isnan(rf) & np.isfinite(mkt)
        if missing_rf.any():
            b.warnings.append(f"risk-free rate missing in {int(missing_rf.sum())} period(s); {mkt_name} is NaN there")
    else:
        out["MKTRF"] = mkt
        b.warnings.append(f"no risk-free rate supplied: {mkt_name} is the raw value-weighted market return (RF = 0)")

    # --- characteristic sorts ----------------------------------------------------------------
    if model != "capm":
        BE = _asof(panel.book_equity, b.index, b.tickers, "book_equity")
        if formation == "annual_june":
            rows = b.june_rows()
            if not rows:
                b.warnings.append(
                    "annual_june formation: no end-of-June date in the sample, so no size/value"
                    + ("/profitability/investment" if model == "ff5" else "")
                    + " portfolios could be formed (those factors are all NaN)"
                )
            denom_rows: list[int] = []
            missing_dec: list[pd.Timestamp] = []
            for row in rows:
                dec = b.december_row(int(b.index[row].year) - 1)
                if dec is None:
                    missing_dec.append(b.index[row])
                    dec = row
                denom_rows.append(dec)
            if missing_dec:
                b.warnings.append(
                    f"B/M: no December market cap in the sample before {len(missing_dec)} June formation(s) "
                    f"(first {_fmt(missing_dec[0])}); the June market cap was used as the denominator"
                )
        else:
            rows = list(range(n))
            denom_rows = rows
        sizes = [b.ME[r] for r in rows]

        # B/M = book equity / (December or current) market cap; book equity <= 0 is excluded
        assert BE is not None
        bm_chars: list[np.ndarray] = []
        n_neg = 0
        for r, d, size in zip(rows, denom_rows, sizes):
            be, me_d = BE[r], b.ME[d]
            if r < n - 1:  # count only formations that are held for at least one period
                n_neg += int((np.isfinite(be) & (be <= 0) & np.isfinite(size) & (size > 0)).sum())
            with np.errstate(invalid="ignore", divide="ignore"):
                bm_chars.append(np.where((be > 0) & (me_d > 0), be / np.where(me_d > 0, me_d, 1.0), np.nan))
        if n_neg:
            b.warnings.append(
                f"{n_neg} name-formation(s) with non-positive book equity were excluded from the B/M sorts"
            )
        bm_held = b.held_codes(_SORTS["bm"], rows, sizes, bm_chars)
        bm_rets = b.portfolios(_SORTS["bm"], bm_held,
                               {k: ["SMB"] + (["HML"] if k in (0, 2, 3, 5) else []) for k in range(6)})
        smb_bm = _smb(bm_rets)
        hml = _spread(bm_rets, high=2, low=0)

        if model == "ff5":
            OP = _asof(panel.operating_profitability, b.index, b.tickers, "operating_profitability")
            INV = _asof(panel.investment, b.index, b.tickers, "investment")
            assert OP is not None and INV is not None
            op_held = b.held_codes(_SORTS["op"], rows, sizes, [OP[r] for r in rows])
            op_rets = b.portfolios(_SORTS["op"], op_held,
                                   {k: ["SMB"] + (["RMW"] if k in (0, 2, 3, 5) else []) for k in range(6)})
            inv_held = b.held_codes(_SORTS["inv"], rows, sizes, [INV[r] for r in rows])
            inv_rets = b.portfolios(_SORTS["inv"], inv_held,
                                    {k: ["SMB"] + (["CMA"] if k in (0, 2, 3, 5) else []) for k in range(6)})
            out["SMB"] = (smb_bm + _smb(op_rets) + _smb(inv_rets)) / 3.0
            out["RMW"] = _spread(op_rets, high=2, low=0)   # robust minus weak
            out["CMA"] = _spread(inv_rets, high=0, low=2)  # conservative minus aggressive
        else:
            out["SMB"] = smb_bm
        out["HML"] = hml

        if model == "carhart4":
            MOM = _asof(panel.momentum_12_1, b.index, b.tickers, "momentum_12_1")
            assert MOM is not None
            mrows = list(range(n))
            mom_held = b.held_codes(_SORTS["mom"], mrows, [b.ME[r] for r in mrows], [MOM[r] for r in mrows])
            mom_rets = b.portfolios(_SORTS["mom"], mom_held, {k: ["Mom"] for k in (0, 2, 3, 5)})
            out["MOM"] = _spread(mom_rets, high=2, low=0)

    b.finish_warnings()
    missing = [c for c in cols if _canon(c) not in out]
    if missing:
        raise ValueError(f"factor_columns({model!r}) has columns {missing} this module cannot construct")
    frame = pd.DataFrame({c: out[_canon(c)] for c in cols}, index=b.index)
    if rf is not None:
        frame["RF"] = rf
    frame = frame.astype(float)
    frame.attrs.update({"source": _SOURCE, "model": model, "formation": formation, "units": "fraction"})
    return frame, b.warnings


# ------------------------------------------------------------------------------------ comparison


def _periods_per_year(index: pd.Index) -> float:
    if len(index) < 2:
        return 12.0
    if isinstance(index, pd.PeriodIndex):
        index = index.to_timestamp()
    gaps = np.diff(index.values).astype("timedelta64[D]").astype(float)
    med = float(np.median(gaps))
    if med <= 4:
        return 252.0
    if med <= 10:
        return 52.0
    if med <= 40:
        return 12.0
    if med <= 120:
        return 4.0
    return 1.0


def _to_monthly(s: pd.Series) -> pd.Series:
    """Compound a daily/weekly series to calendar months (NaN if any observation in the month is NaN)."""
    per = s.index.to_period("M")
    growth = (1.0 + s.fillna(0.0)).groupby(per).prod() - 1.0
    has_nan = s.isna().groupby(per).any()
    return growth.where(~has_nan)


def _monthly_key(s: pd.Series) -> pd.Series:
    out = s.copy()
    out.index = s.index.to_period("M")
    return out[~out.index.duplicated(keep="last")]


def _align_pair(a: pd.Series, b: pd.Series) -> tuple[pd.Series, pd.Series, float]:
    """Inner-join two factor series on comparable periods; also return periods per year."""
    a = a[~a.index.duplicated(keep="last")].sort_index()
    b = b[~b.index.duplicated(keep="last")].sort_index()
    a_low, b_low = _periods_per_year(a.index) <= 12, _periods_per_year(b.index) <= 12
    if a_low or b_low:  # at least one side is monthly (or coarser): align on calendar months
        a = _monthly_key(a) if a_low else _to_monthly(a)
        b = _monthly_key(b) if b_low else _to_monthly(b)
    else:  # both daily / weekly: align on dates
        a.index = pd.DatetimeIndex(a.index).normalize()
        b.index = pd.DatetimeIndex(b.index).normalize()
    ppy = _periods_per_year(a.index) if len(a) >= 2 else _periods_per_year(b.index)
    joined = pd.concat([a.rename("a"), b.rename("b")], axis=1, join="inner").dropna()
    return joined["a"], joined["b"], ppy


def compare_with_official(constructed: pd.DataFrame, official: pd.DataFrame) -> list[FactorConstructionCheck]:
    """Compare constructed factors with the official French series, factor by factor.

    For every column of ``constructed`` except ``RF``: Pearson correlation with the matching
    official column (names matched case/punctuation-insensitively, ``UMD``/``WML`` = ``Mom``) over
    the periods where both are available, and the annualised arithmetic mean of each (mean per
    period x periods per year, in percent) over those same overlapping periods. Monthly series are
    aligned by calendar month (so a last-trading-day index matches the official month-end index);
    when one side is daily and the other monthly, the daily side is compounded to months first.
    With no overlap (or no official column) the constructed premium is computed on its own sample,
    the official premium on its own sample (None when the column is missing) and the correlation is
    None. Correlations need at least 3 overlapping periods and non-constant series.
    """
    checks: list[FactorConstructionCheck] = []
    for col in constructed.columns:
        if str(col).upper() == "RF":
            continue
        mine = constructed[col].astype(float)
        src = _find_column(official, str(col))
        corr: float | None = None
        prem_c: float | None = None
        prem_o: float | None = None
        n_overlap = 0
        if src is not None:
            a, o, ppy = _align_pair(mine, official[src].astype(float))
            n_overlap = int(len(a))
            if n_overlap > 0:
                prem_c = float(a.mean() * ppy * 100.0)
                prem_o = float(o.mean() * ppy * 100.0)
            if n_overlap >= 3 and a.std() > 0 and o.std() > 0:
                corr = float(np.corrcoef(a.to_numpy(), o.to_numpy())[0, 1])
        if n_overlap == 0:
            own = mine.dropna()
            prem_c = float(own.mean() * _periods_per_year(own.index) * 100.0) if len(own) else None
            if src is not None:
                off = official[src].astype(float).dropna()
                prem_o = float(off.mean() * _periods_per_year(off.index) * 100.0) if len(off) else None
        checks.append(
            FactorConstructionCheck(
                factor=str(col),
                correlation_with_official=corr,
                annual_premium_constructed_pct=prem_c,
                annual_premium_official_pct=prem_o,
                n_overlap_periods=n_overlap,
            )
        )
    return checks
