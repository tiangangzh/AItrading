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

_REGRESSION_MODULES = ("aitrading.factors.regression", "aitrading.backtest.regression")

# Canonical (upper-case, alphanumeric-only) name -> canonical name of the official column.
_ALIASES = {"UMD": "MOM", "WML": "MOM", "PR1YR": "MOM", "MKT": "MKTRF", "MKTEXCESS": "MKTRF", "RISKFREE": "RF"}


def factor_columns(model: str) -> list[str]:
    """Factor column names for ``model`` (excluding RF).

    Uses ``FACTOR_COLUMNS`` from the regression module (``aitrading.factors.regression`` or
    ``aitrading.backtest.regression``) when it is importable, so the whole platform agrees on one
    naming; otherwise :data:`DEFAULT_FACTOR_COLUMNS` (the official headers).
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


def _is_header(fields: list[str]) -> bool:
    return (
        len(fields) >= 2
        and fields[0] == ""
        and all(f != "" for f in fields[1:])
        and not any(_is_number(f) for f in fields[1:])
    )


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
    text = text.lstrip("﻿")
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
            if _is_header(fields):
                header = fields[1:]  # (re)start: the last header before the data wins
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
