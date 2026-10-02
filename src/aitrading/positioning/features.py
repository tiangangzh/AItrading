"""Positioning features: short interest and options summary -> catalog positioning features.

Inputs are the provider snapshots (indexed by ticker, ``fields.SHORT_INTEREST_COLUMNS`` /
``fields.OPTIONS_COLUMNS``, ratios as fractions) plus two technical inputs: the 20-session average
volume in shares (``technical.features.average_volume_shares``) and ``volatility_20d_pct``.

Conventions
-----------
* Output units follow the catalog: implied volatility is converted from a fraction to %.
* Any ratio whose denominator is missing, zero or negative is NaN; negative counts (shares,
  contracts, volatilities) are treated as missing.
* iv_rank_1y is NaN when the 1y high <= 1y low (no range) and is clipped to [0, 100] (an IV above
  the vendor's 1y high is at a new high: 100).
* Tickers absent from an input get NaN for the features that depend on it; missing input columns
  behave like all-NaN columns. Duplicate input rows: the last one wins.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from aitrading.core import fields
from aitrading.screen.catalog import POSITIONING_FEATURES


def _frame(data: pd.DataFrame | None, columns: list[str], index: pd.Index) -> pd.DataFrame:
    """Numeric ``columns`` of ``data`` re-indexed to ``index`` (NaN where absent)."""
    if data is None or len(data) == 0:
        return pd.DataFrame(np.nan, index=index, columns=columns, dtype="float64")
    if fields.TICKER in data.columns and data.index.name != fields.TICKER:
        data = data.set_index(fields.TICKER)
    data = data[~data.index.duplicated(keep="last")]
    out = data.reindex(index=index, columns=columns)
    return out.apply(pd.to_numeric, errors="coerce").astype("float64")


def _series(data: pd.Series | None, index: pd.Index) -> pd.Series:
    if data is None or len(data) == 0:
        return pd.Series(np.nan, index=index, dtype="float64")
    data = data[~data.index.duplicated(keep="last")]
    return pd.to_numeric(data.reindex(index), errors="coerce").astype("float64")


def _nonneg(s: pd.Series) -> pd.Series:
    return s.where(np.isfinite(s) & (s >= 0))


def _ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    """num / den where den > 0 (and both finite), else NaN."""
    ok = np.isfinite(num) & np.isfinite(den) & (den > 0)
    return (num / den.where(ok)).where(ok)


def compute_positioning_features(
    short_interest: pd.DataFrame | None,
    options: pd.DataFrame | None,
    avg_volume_20d_shares: pd.Series | None,
    realized_vol_20d_pct: pd.Series | None,
    tickers: Sequence[str],
) -> pd.DataFrame:
    """Catalog positioning features for ``tickers`` (order kept, duplicates dropped).

    Returns a float frame indexed by ``ticker`` with exactly the columns
    ``aitrading.screen.catalog.POSITIONING_FEATURES``, in that order.
    """
    index = pd.Index(list(dict.fromkeys(tickers)), name="ticker")
    si = _frame(short_interest, [fields.SHORT_INTEREST_SHARES, fields.SHORT_INTEREST_SHARES_1M_AGO, fields.FLOAT_SHARES], index)
    opt = _frame(options, list(fields.OPTIONS_COLUMNS), index)
    avg_vol = _nonneg(_series(avg_volume_20d_shares, index))
    rv_pct = _nonneg(_series(realized_vol_20d_pct, index))

    iv = _nonneg(opt[fields.IV_30D_ATM])
    iv_hi = _nonneg(opt[fields.IV_30D_ATM_1Y_HIGH])
    iv_lo = _nonneg(opt[fields.IV_30D_ATM_1Y_LOW])
    iv_pct = iv * 100.0
    iv_rank = (_ratio(iv - iv_lo, iv_hi - iv_lo) * 100.0).clip(lower=0.0, upper=100.0)

    shorts = _nonneg(si[fields.SHORT_INTEREST_SHARES])
    shorts_1m = _nonneg(si[fields.SHORT_INTEREST_SHARES_1M_AGO])

    out = pd.DataFrame(
        {
            "iv_30d_pct": iv_pct,
            "iv_rank_1y": iv_rank,
            "iv_to_realized_vol_ratio": _ratio(iv_pct, rv_pct),
            "put_call_volume_ratio": _ratio(_nonneg(opt[fields.PUT_VOLUME]), _nonneg(opt[fields.CALL_VOLUME])),
            "put_call_oi_ratio": _ratio(_nonneg(opt[fields.PUT_OPEN_INTEREST]), _nonneg(opt[fields.CALL_OPEN_INTEREST])),
            "short_interest_pct_float": _ratio(shorts, _nonneg(si[fields.FLOAT_SHARES])) * 100.0,
            "days_to_cover": _ratio(shorts, avg_vol),
            "short_interest_change_1m_pct": (_ratio(shorts, shorts_1m) - 1.0) * 100.0,
        },
        index=index,
    )
    return out[POSITIONING_FEATURES].astype("float64")
