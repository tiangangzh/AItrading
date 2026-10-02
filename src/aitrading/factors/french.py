"""Loader for the official Kenneth R. French Data Library factor returns (free and public).

Source: https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html

The library publishes the canonical academic factor series:

* Fama-French 3 factors (Mkt-RF, SMB, HML) and the 1-month T-bill rate (RF) - Fama & French (1993),
  "Common risk factors in the returns on stocks and bonds", *Journal of Financial Economics* 33.
* Fama-French 5 factors, 2x3 sorts (adds RMW, CMA) - Fama & French (2015), "A five-factor asset
  pricing model", *Journal of Financial Economics* 116.
* The momentum factor (Mom, a.k.a. UMD) used in the Carhart (1997) 4-factor model - Carhart (1997),
  "On persistence in mutual fund performance", *Journal of Finance* 52.

The library is rebuilt roughly monthly from CRSP/Compustat and lags the calendar by one to two
months, so the most recent months are typically absent; past values are occasionally revised when
French re-runs the construction on a new CRSP vintage. Each zip is therefore cached on disk and
refreshed when older than ``max_age_days``.

File format (the parser is written for the real files, quirks included)
----------------------------------------------------------------------
Each URL serves a zip containing a single CSV:

    This file was created by CMPT_ME_BEME_RETS using the 202408 CRSP database.
    The 1-month TBill return is from Ibbotson and Associates, Inc.

    ,Mkt-RF,SMB,HML,RF
    192607,    2.96,   -2.56,   -2.43,    0.22
    ...

     Annual Factors: January-December
    ,Mkt-RF,SMB,HML,RF
    1927,   29.47,   -2.46,   -3.75,    3.12
    ...

    Copyright 2024 Kenneth R. French

* free-text preamble lines (which may themselves contain commas);
* a header row whose first field is empty; column names may carry padding (``"Mom   "``);
* monthly rows keyed ``YYYYMM`` (daily files: ``YYYYMMDD``), values in **percent**;
* then a blank line and an ``Annual Factors: January-December`` section keyed ``YYYY`` (ignored),
  and a trailing copyright line;
* missing values coded ``-99.99`` or ``-999``.

:func:`parse_french_csv` returns **fractions** (0.01 = 1%) on a ``DatetimeIndex`` at period end:
month-end timestamps for monthly files, the trading date for daily files.

This module needs network access to ``mba.tuck.dartmouth.edu`` only on a cache miss; inject
``fetch`` to supply bytes from elsewhere (tests, a proxy, a manual download).
"""

from __future__ import annotations

import importlib
import io
import math
import os
import tempfile
import time
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Literal

import numpy as np
import pandas as pd

from aitrading.data.base import ProviderError

__all__ = [
    "FRENCH_BASE_URL",
    "FRENCH_DATASETS",
    "FRENCH_SOURCE",
    "DEFAULT_FACTOR_COLUMNS",
    "align_factor_dates",
    "FrenchDataset",
    "FrenchDataUnavailable",
    "default_french_cache_dir",
    "factor_columns",
    "load_french_factors",
    "parse_french_csv",
    "read_french_zip",
]

FRENCH_BASE_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
FRENCH_SOURCE = "Kenneth R. French Data Library"
USER_AGENT = (
    "aitrading/0.1 (equity research platform; Fama-French factor loader; "
    "+https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html)"
)
ENV_CACHE_DIR = "AITRADING_CACHE_DIR"
_HTTP_TIMEOUT_S = 60.0
_MISSING_SENTINELS = (-99.99, -999.0)

FactorModel = Literal["capm", "ff3", "carhart4", "ff5"]
Frequency = Literal["daily", "monthly"]


class FrenchDataUnavailable(ProviderError):
    """The Kenneth French Data Library could not be reached and no cached copy exists."""


@dataclass(frozen=True)
class FrenchDataset:
    key: str
    filename: str
    frequency: Frequency
    columns: tuple[str, ...]
    description: str

    @property
    def url(self) -> str:
        return FRENCH_BASE_URL + self.filename


FRENCH_DATASETS: dict[str, FrenchDataset] = {
    d.key: d
    for d in (
        FrenchDataset("ff3_monthly", "F-F_Research_Data_Factors_CSV.zip", "monthly",
                      ("Mkt-RF", "SMB", "HML", "RF"), "Fama/French 3 factors + 1-month T-bill, monthly"),
        FrenchDataset("ff3_daily", "F-F_Research_Data_Factors_daily_CSV.zip", "daily",
                      ("Mkt-RF", "SMB", "HML", "RF"), "Fama/French 3 factors + T-bill, daily"),
        FrenchDataset("ff5_monthly", "F-F_Research_Data_5_Factors_2x3_CSV.zip", "monthly",
                      ("Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"), "Fama/French 5 factors (2x3), monthly"),
        FrenchDataset("ff5_daily", "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip", "daily",
                      ("Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"), "Fama/French 5 factors (2x3), daily"),
        FrenchDataset("mom_monthly", "F-F_Momentum_Factor_CSV.zip", "monthly",
                      ("Mom",), "Momentum factor (Mom / UMD), monthly"),
        FrenchDataset("mom_daily", "F-F_Momentum_Factor_daily_CSV.zip", "daily",
                      ("Mom",), "Momentum factor (Mom / UMD), daily"),
    )
}

# Which files make up each model (the first one supplies RF).
_MODEL_FILES: dict[str, tuple[str, ...]] = {
    "capm": ("ff3",),
    "ff3": ("ff3",),
    "carhart4": ("ff3", "mom"),
    "ff5": ("ff5",),
}

#: Column names used when ``aitrading.*.regression.FACTOR_COLUMNS`` is not importable; they match
#: the official file headers.
DEFAULT_FACTOR_COLUMNS: dict[str, list[str]] = {
    "capm": ["Mkt-RF"],
    "ff3": ["Mkt-RF", "SMB", "HML"],
    "carhart4": ["Mkt-RF", "SMB", "HML", "Mom"],
    "ff5": ["Mkt-RF", "SMB", "HML", "RMW", "CMA"],
}

_REGRESSION_MODULES = ("aitrading.backtest.regression",)

# Canonical (upper-case, alphanumeric-only) name -> canonical name of the official column.
_ALIASES = {"UMD": "MOM", "WML": "MOM", "PR1YR": "MOM", "MKT": "MKTRF", "MKTEXCESS": "MKTRF", "RISKFREE": "RF"}


def factor_columns(model: str) -> list[str]:
    """Factor column names for ``model`` (excluding RF).

    Uses ``aitrading.backtest.regression.FACTOR_COLUMNS`` (imported lazily) so the whole platform
    agrees on one naming; :data:`DEFAULT_FACTOR_COLUMNS` (the official headers, identical today) is
    the fallback when that module is unavailable.
    """
    if model not in DEFAULT_FACTOR_COLUMNS:
        raise ValueError(f"unknown factor model {model!r}; expected one of {sorted(DEFAULT_FACTOR_COLUMNS)}")
    for mod_name in _REGRESSION_MODULES:
        try:
            mod = importlib.import_module(mod_name)
        except ImportError:
            continue
        cols = getattr(mod, "FACTOR_COLUMNS", None)
        if isinstance(cols, Mapping) and model in cols:
            return [str(c) for c in cols[model]]
    return list(DEFAULT_FACTOR_COLUMNS[model])


def _canon(name: str) -> str:
    c = "".join(ch for ch in str(name).upper() if ch.isalnum())
    return _ALIASES.get(c, c)


def _find_column(frame: pd.DataFrame, wanted: str) -> str | None:
    if wanted in frame.columns:
        return wanted
    target = _canon(wanted)
    for col in frame.columns:
        if _canon(col) == target:
            return col
    return None


# ------------------------------------------------------------------------------------ parsing


def _is_number(s: str) -> bool:
    try:
        float(s)
    except ValueError:
        return False
    return True


def _header_names(fields: list[str]) -> list[str] | None:
    """Column names when ``fields`` is a header row (empty first field, non-numeric names), else None."""
    names = fields[1:]
    while names and names[-1] == "":
        names = names[:-1]  # tolerate trailing commas
    if fields[0] != "" or not names or any(f == "" or _is_number(f) for f in names):
        return None
    return names


def _to_value(s: str) -> float:
    if s == "":
        return math.nan
    x = float(s)
    if any(abs(x - m) < 1e-9 for m in _MISSING_SENTINELS):
        return math.nan
    return x


def parse_french_csv(text: str) -> pd.DataFrame:
    """Parse the first data section of a French data-library CSV.

    Returns a float frame of **fractions** (the files are in percent) whose columns are the header
    names with surrounding whitespace removed, indexed by a ``DatetimeIndex`` named ``"date"``:
    month-end timestamps for ``YYYYMM`` keys, the date itself for ``YYYYMMDD`` keys. Only the first
    section is read; the ``Annual Factors`` section (``YYYY`` keys) and the copyright line are
    ignored. ``-99.99`` and ``-999`` become NaN. ``.attrs["frequency"]`` is ``"monthly"`` or ``"daily"``.

    Raises ``ValueError`` when no header + data section is found or a data row is malformed.
    """
    text = text.lstrip("\ufeff")
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    header: list[str] | None = None
    keys: list[str] = []
    rows: list[list[float]] = []
    key_len: int | None = None
    for lineno, line in enumerate(lines, start=1):
        fields = [f.strip() for f in line.split(",")]
        key = fields[0]
        is_data = key.isdigit() and len(key) in (6, 8) and (key_len is None or len(key) == key_len)
        if header is None or not rows:
            names = _header_names(fields)
            if names is not None:
                header = names  # (re)start: the last header before the data wins
                continue
            if header is None or not is_data:
                continue  # preamble / blank lines between header and data
        if not is_data:
            break  # end of the first section: blank line, "Annual Factors", copyright, ...
        values = fields[1:]
        while len(values) > len(header) and values[-1] == "":
            values.pop()  # tolerate trailing commas
        if len(values) != len(header):
            raise ValueError(
                f"line {lineno}: expected {len(header)} values for columns {header}, got {len(values)}: {line!r}"
            )
        try:
            rows.append([_to_value(v) for v in values])
        except ValueError as exc:
            raise ValueError(f"line {lineno}: non-numeric value in {line!r}") from exc
        keys.append(key)
        key_len = len(key)
    if header is None or not rows:
        raise ValueError("not a Kenneth French data-library CSV: no header row followed by YYYYMM/YYYYMMDD data")

    if key_len == 6:
        index = pd.to_datetime(keys, format="%Y%m") + pd.offsets.MonthEnd(0)
        frequency = "monthly"
    else:
        index = pd.to_datetime(keys, format="%Y%m%d")
        frequency = "daily"
    index = pd.DatetimeIndex(index, name="date")
    frame = pd.DataFrame(np.asarray(rows, dtype=float) / 100.0, index=index, columns=header)
    if frame.index.has_duplicates:
        raise ValueError("duplicate dates in the data section")
    frame = frame.sort_index()
    frame.attrs["frequency"] = frequency
    return frame


def read_french_zip(data: bytes) -> str:
    """Text of the (single) CSV inside a French data-library zip. Raises ``zipfile.BadZipFile``."""
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        names = [n for n in zf.namelist() if not n.endswith("/")]
        csvs = [n for n in names if n.lower().endswith((".csv", ".txt"))] or names
        if not csvs:
            raise zipfile.BadZipFile("empty zip archive")
        raw = zf.read(sorted(csvs)[0])
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return raw.decode("latin-1")


# ------------------------------------------------------------------------------------ fetching


def default_french_cache_dir() -> Path:
    """``$AITRADING_CACHE_DIR/french`` when the variable is set, else ``~/.aitrading/cache/french``."""
    env = os.environ.get(ENV_CACHE_DIR, "").strip()
    root = Path(env).expanduser() if env else Path.home() / ".aitrading" / "cache"
    return root / "french"


def _default_fetch(url: str) -> bytes:
    import httpx  # local import keeps module import cheap

    resp = httpx.get(
        url,
        headers={"User-Agent": USER_AGENT, "Accept": "application/zip, */*"},
        follow_redirects=True,
        timeout=_HTTP_TIMEOUT_S,
    )
    resp.raise_for_status()
    return resp.content


def _write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp-", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _parse_zip(data: bytes) -> pd.DataFrame:
    return parse_french_csv(read_french_zip(data))


def _fmt_age(path: Path) -> tuple[str, float]:
    mtime = path.stat().st_mtime
    stamp = datetime.fromtimestamp(mtime, tz=timezone.utc).strftime("%Y-%m-%d")
    return stamp, (time.time() - mtime) / 86400.0


def _load_dataset(
    ds: FrenchDataset,
    cache_dir: Path,
    fetch: Callable[[str], bytes],
    max_age_days: float,
    warnings: list[str],
) -> pd.DataFrame:
    path = cache_dir / ds.filename
    cached: pd.DataFrame | None = None
    if path.is_file():
        try:
            cached = _parse_zip(path.read_bytes())
        except (OSError, ValueError, zipfile.BadZipFile):
            cached = None  # corrupt cache entry = miss
            warnings.append(f"{ds.filename}: cached copy at {path} was unreadable; downloading a fresh one")
        if cached is not None:
            _, age_days = _fmt_age(path)
            if age_days <= max_age_days:
                return cached

    try:
        data = fetch(ds.url)
        if not isinstance(data, (bytes, bytearray)):
            raise TypeError(f"fetch returned {type(data).__name__}, expected bytes")
        try:
            frame = _parse_zip(bytes(data))
        except (ValueError, zipfile.BadZipFile) as exc:
            head = bytes(data[:40])
            raise ValueError(f"response is not a French data-library zip ({len(data)} bytes, starts {head!r}): {exc}") from exc
    except Exception as exc:  # noqa: BLE001 - any network / parsing failure falls back to the cache
        if cached is not None:
            stamp, age_days = _fmt_age(path)
            warnings.append(
                f"{FRENCH_SOURCE} unreachable for {ds.filename} ({type(exc).__name__}: {exc}); using the cached copy "
                f"downloaded {stamp} ({age_days:.0f} days old) - the most recent months may be missing"
            )
            return cached
        raise FrenchDataUnavailable(
            f"Could not download {ds.url} ({type(exc).__name__}: {exc}) and there is no cached copy in {cache_dir}. "
            f"Check your internet access (the {FRENCH_SOURCE} must be reachable at {FRENCH_BASE_URL}), or download "
            f"{ds.filename} manually and place it in {cache_dir}."
        ) from exc

    try:
        _write_atomic(path, bytes(data))
    except OSError as exc:
        warnings.append(f"could not write the {ds.filename} cache to {path} ({exc}); it will be downloaded again next time")
    return frame


def load_french_factors(
    model: FactorModel,
    frequency: Frequency = "monthly",
    *,
    cache_dir: Path | None = None,
    fetch: Callable[[str], bytes] | None = None,
    max_age_days: int = 7,
) -> pd.DataFrame:
    """Official factor returns for ``model`` from the Kenneth R. French Data Library.

    Returns fractions (0.01 = 1%) with columns ``factor_columns(model) + ["RF"]`` on a
    ``DatetimeIndex`` (month-end for monthly, trading dates for daily):

    * ``capm``     - Mkt-RF (+ RF) from the 3-factor file;
    * ``ff3``      - Mkt-RF, SMB, HML - Fama & French (1993);
    * ``carhart4`` - the 3-factor file inner-joined on dates with the momentum file (Mom) -
      Carhart (1997);
    * ``ff5``      - Mkt-RF, SMB, HML, RMW, CMA from the 2x3 five-factor file - Fama & French (2015).

    The library is updated monthly with a lag of one to two months (and history is occasionally
    revised), so the latest months may be missing. Raw zip bytes are cached in ``cache_dir``
    (default :func:`default_french_cache_dir`) and re-downloaded when older than ``max_age_days``.
    If the download fails, a stale cached copy is used and a warning is recorded in
    ``frame.attrs["warnings"]``; with no cached copy :class:`FrenchDataUnavailable` is raised.

    ``fetch(url) -> bytes`` replaces the default HTTP client (httpx, descriptive User-Agent,
    redirects followed, 60 s timeout).

    ``frame.attrs`` also holds ``source``, ``model``, ``frequency``, ``datasets`` (file names),
    ``urls`` and ``units`` (``"fraction"``).
    """
    if model not in _MODEL_FILES:
        raise ValueError(f"unknown factor model {model!r}; expected one of {sorted(_MODEL_FILES)}")
    if frequency not in ("daily", "monthly"):
        raise ValueError(f"frequency must be 'daily' or 'monthly', got {frequency!r}")
    cache_dir = Path(cache_dir).expanduser() if cache_dir is not None else default_french_cache_dir()
    fetch = fetch or _default_fetch
    warnings: list[str] = []

    datasets = [FRENCH_DATASETS[f"{name}_{frequency}"] for name in _MODEL_FILES[model]]
    frames = [_load_dataset(ds, cache_dir, fetch, float(max_age_days), warnings) for ds in datasets]

    wanted = factor_columns(model) + ["RF"]
    index = frames[0].index
    for frame in frames[1:]:
        index = index.intersection(frame.index)  # carhart4: inner join of the 3-factor and momentum files
    out = pd.DataFrame(index=index.sort_values())
    for col in wanted:
        for frame in frames:
            src = _find_column(frame, col)
            if src is not None:
                out[col] = frame[src].reindex(out.index)
                break
        else:
            raise ValueError(
                f"column {col!r} for model {model!r} not found in {[ds.filename for ds in datasets]} "
                f"(columns: {[list(f.columns) for f in frames]}) - the file format may have changed"
            )
    out = out[wanted].astype(float)
    out.index.name = "date"
    out.attrs.update(
        {
            "source": FRENCH_SOURCE,
            "model": model,
            "frequency": frequency,
            "units": "fraction",
            "datasets": [ds.filename for ds in datasets],
            "urls": [ds.url for ds in datasets],
            "warnings": warnings,
        }
    )
    return out


# ------------------------------------------------------------------------------------ frequency alignment

#: Spacing classes from the median gap between dates, finest first.
_SPACING_RANK: dict[str, int] = {"daily": 0, "weekly": 1, "monthly": 2, "quarterly": 3, "annual": 4}
#: A source series may start this long after a period starts and still cover it (a weekend plus a holiday).
_START_SLACK = pd.Timedelta(days=3)


def _median_gap_days(index: pd.Index) -> float | None:
    """Median gap in days between the distinct dates of ``index`` (None with fewer than 2)."""
    idx = pd.DatetimeIndex(index)
    idx = idx[~idx.isna()].unique().sort_values()
    if len(idx) < 2:
        return None
    gaps = np.diff(idx.values).astype("timedelta64[s]").astype(float) / 86400.0
    return float(np.median(gaps))


def _spacing(index: pd.Index) -> str | None:
    """'daily' / 'weekly' / 'monthly' / 'quarterly' / 'annual' from the median gap (None with < 2 dates).

    Thresholds (days): <= 4 daily (weekends and holidays included), <= 10 weekly, <= 40 monthly,
    <= 120 quarterly, else annual.
    """
    gap = _median_gap_days(index)
    if gap is None:
        return None
    if gap <= 4:
        return "daily"
    if gap <= 10:
        return "weekly"
    if gap <= 40:
        return "monthly"
    if gap <= 120:
        return "quarterly"
    return "annual"


def _naive(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return index.tz_localize(None) if index.tz is not None else index


def _period_ends(dates: pd.DatetimeIndex, spacing: str) -> pd.DatetimeIndex:
    """End of the period each date closes: the calendar month-end for monthly-or-coarser dates (so a
    last-trading-day index covers whole calendar months), the date itself (normalised) otherwise."""
    d = pd.DatetimeIndex(dates).normalize()
    if _SPACING_RANK[spacing] >= _SPACING_RANK["monthly"]:
        d = d + pd.offsets.MonthEnd(0)
    return d


def _compound_onto(frame: pd.DataFrame, dates: pd.DatetimeIndex, spacing: str) -> tuple[pd.DataFrame, int]:
    """Compound the finer-grained per-period returns of ``frame`` over the period ending at each of ``dates``.

    ``spacing`` is the spacing class of ``dates``. Period ``i`` is ``(end[i-1], end[i]]`` with the
    ends from :func:`_period_ends`; the first period has the median length (the median number of
    calendar months between ends, or the median gap in days). Each source observation goes to the
    period containing its date; a column's value is ``prod(1 + x) - 1`` over the period. When the
    frame has an ``RF`` column, excess-market columns (``Mkt-RF``) are compounded as
    ``prod(1 + Mkt-RF + RF) - prod(1 + RF)`` (the excess return of the compounded market). Long-short
    columns are compounded as they are (a daily-rebalanced spread).

    A period is NaN when it holds no observation, any of its observations is NaN (per column), or
    the source does not span it: the first observation's own period (date minus the source's median
    gap) starts more than 3 days after the period starts, or the last observation is before the
    target date rolled back to a weekday. Returns the frame indexed like ``dates`` and the number of
    periods that held observations but were set to NaN because the source only partly covers them.
    """
    out_index = pd.DatetimeIndex(dates)
    dates = _naive(out_index)  # periods are wall-clock dates; time zones are ignored
    cols = list(frame.columns)
    empty = pd.DataFrame(np.nan, index=out_index, columns=cols)
    src = frame[~frame.index.duplicated(keep="last")].sort_index()
    sidx = _naive(pd.DatetimeIndex(src.index)).normalize()
    keys = _period_ends(dates, spacing)
    ends = keys[~keys.isna()].unique().sort_values()
    if len(ends) == 0 or len(sidx) == 0:
        return empty, 0

    if _SPACING_RANK[spacing] >= _SPACING_RANK["monthly"]:
        months = np.asarray(ends.year * 12 + ends.month, dtype=float)
        step = max(1, int(round(float(np.median(np.diff(months)))))) if len(ends) > 1 else 1
        first_start = ends[0] - pd.offsets.MonthEnd(step)
    else:
        gap = _median_gap_days(ends) or 1.0
        first_start = ends[0] - pd.Timedelta(days=max(1, int(round(gap))))
    starts = pd.DatetimeIndex([first_start]).append(ends[:-1])

    pos = ends.searchsorted(sidx, side="left")  # first period end at or after each observation
    inside = pos < len(ends)
    pos_c = np.minimum(pos, len(ends) - 1)
    inside &= np.asarray(sidx > starts[pos_c])
    bucket = np.where(inside, pos, -1)
    keep = bucket >= 0

    vals = src.to_numpy(dtype=float)
    growth = 1.0 + vals
    rf_j = next((j for j, c in enumerate(cols) if _canon(c) == "RF"), None)
    mkt_js = [j for j, c in enumerate(cols) if _canon(c) == "MKTRF"] if rf_j is not None else []
    for j in mkt_js:
        growth[:, j] = 1.0 + vals[:, j] + vals[:, rf_j]  # gross market return
    n = len(ends)
    if keep.any():
        g = pd.DataFrame(growth[keep]).groupby(bucket[keep])
        prod = g.prod()
        has_nan = pd.DataFrame(np.isnan(growth[keep])).groupby(bucket[keep]).any()
        res = np.array((prod - 1.0).where(~has_nan).reindex(range(n)), dtype=float)  # writable copy
    else:
        res = np.full((n, len(cols)), np.nan)
    for j in mkt_js:
        res[:, j] = res[:, j] - res[:, rf_j]  # prod(1 + Mkt) - prod(1 + RF)
    counts = np.bincount(bucket[keep], minlength=n)

    # coverage: the source must span every period it is compounded over
    src_gap = _median_gap_days(sidx) or 0.0
    cov_start = sidx[0] - pd.Timedelta(days=src_gap)
    start_ok = np.asarray(cov_start <= starts + _START_SLACK)
    last_target = pd.Series(dates.normalize(), index=keys).groupby(level=0).max().reindex(ends)
    due = np.busday_offset(last_target.to_numpy().astype("datetime64[D]"), 0, roll="backward")
    end_ok = np.asarray(sidx[-1] >= pd.DatetimeIndex(due))
    covered = start_ok & end_ok
    n_partial = int(((counts > 0) & ~covered).sum())
    res[~covered, :] = np.nan

    out = pd.DataFrame(res, index=ends, columns=cols).reindex(keys)
    out.index = out_index
    return out.astype(float), n_partial


def align_factor_dates(factors: pd.DataFrame, dates: pd.DatetimeIndex | pd.Index) -> pd.DataFrame:
    """Re-index factor returns onto ``dates`` (e.g. a strategy's period-end dates) without look-ahead.

    The spacing of both the factors and ``dates`` is inferred from the median gap between dates
    (daily / weekly / monthly / quarterly / annual), and the alignment depends on how they compare:

    * **same frequency** - monthly-or-coarser factors are matched by calendar month (the official
      monthly series sit on calendar month-ends while backtests usually sit on the last trading day,
      2024-05-31 vs 2024-05-30, so an exact date join would silently drop months); daily / weekly
      factors are matched by date. Dates without a match are NaN.
    * **factors finer than dates** (e.g. daily factors, monthly strategy) - the factors are
      compounded over each target period, ``(previous date, date]`` (monthly-or-coarser dates are
      moved to their calendar month-end first); ``Mkt-RF`` is compounded with ``RF`` as
      ``prod(1 + Mkt-RF + RF) - prod(1 + RF)`` when ``RF`` is present, and periods the factors only
      partly cover are NaN (see :func:`_compound_onto`). A warning is recorded: compounded daily
      long-short returns differ slightly from the official monthly series, so load the frequency of
      the strategy when the library publishes it.
    * **factors coarser than dates** (e.g. monthly factors, daily strategy) - ``ValueError``: a
      monthly factor return cannot be split into daily ones, and copying it onto every day would give
      nonsense betas and alpha. Load the daily factors instead.

    With fewer than two distinct ``dates`` (or factor dates) the spacing cannot be inferred and the
    same-frequency rule is used (with the factors' ``attrs["frequency"]`` when they have one row).
    ``out.attrs`` copies ``factors.attrs`` (the ``warnings`` list is copied, never mutated) and adds
    ``alignment`` (``"calendar month"``, ``"date"`` or ``"compounded <from> -> <to>"``); after
    compounding ``frequency`` is the spacing of ``dates``.
    """
    dates = pd.DatetimeIndex(dates)
    f = factors[~factors.index.duplicated(keep="last")].sort_index()
    idx = pd.DatetimeIndex(f.index)
    f_sp = _spacing(idx)
    if f_sp is None and factors.attrs.get("frequency") in _SPACING_RANK:
        f_sp = str(factors.attrs["frequency"])  # a single row: trust the loader's label
    d_sp = _spacing(dates)
    attrs = dict(factors.attrs)
    warnings = [str(w) for w in (factors.attrs.get("warnings") or [])]

    if f_sp is not None and d_sp is not None and f_sp != d_sp:
        if _SPACING_RANK[f_sp] > _SPACING_RANK[d_sp]:
            raise ValueError(
                f"cannot align {f_sp} factor returns onto {d_sp} dates: a {f_sp} factor return cannot be "
                f"split into {d_sp} returns, and copying it onto every {d_sp} date would make the regression "
                f"meaningless. Load the factors at the strategy's frequency (e.g. "
                f"load_french_factors(model, 'daily') for a daily strategy) or compound the strategy returns "
                f"to {f_sp} periods."
            )
        out, n_partial = _compound_onto(f, dates, d_sp)
        msg = (
            f"{f_sp} factor returns were compounded over each {d_sp} strategy period "
            f"(prod(1 + f) - 1; Mkt-RF as prod(1 + Mkt-RF + RF) - prod(1 + RF) when RF is present)"
        )
        if n_partial:
            msg += f"; {n_partial} period(s) only partly covered by the factor data were set to NaN"
        if d_sp == "monthly":
            msg += (f"; compounded {f_sp} long-short returns differ slightly from the official monthly series "
                    f"(load_french_factors(model, 'monthly') for a monthly strategy)")
        warnings.append(msg)
        attrs["alignment"] = f"compounded {f_sp} -> {d_sp}"
        attrs["frequency"] = d_sp
    else:
        sp = f_sp or d_sp
        if sp is None or _SPACING_RANK[sp] >= _SPACING_RANK["monthly"]:
            by_month = f.copy()
            by_month.index = idx.to_period("M")
            out = by_month.reindex(dates.to_period("M"))
            attrs["alignment"] = "calendar month"
        else:
            by_day = f.copy()
            by_day.index = idx.normalize()
            out = by_day.reindex(dates.normalize())
            attrs["alignment"] = "date"
        out.index = dates
    attrs["warnings"] = warnings
    out.attrs = attrs
    return out
