"""OLS with Newey-West (HAC) standard errors and factor-model attribution (numpy only)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from aitrading.backtest.metrics import default_nw_lags, infer_periods_per_year
from aitrading.backtest.models import FactorRegression

__all__ = ["FACTOR_COLUMNS", "MIN_REGRESSION_OBS", "ols_newey_west", "factor_regression"]

FACTOR_COLUMNS: dict[str, list[str]] = {
    "capm": ["Mkt-RF"],
    "ff3": ["Mkt-RF", "SMB", "HML"],
    "carhart4": ["Mkt-RF", "SMB", "HML", "Mom"],
    "ff5": ["Mkt-RF", "SMB", "HML", "RMW", "CMA"],
}

MIN_REGRESSION_OBS = 24


def ols_newey_west(
    y: pd.Series, X: pd.DataFrame, lags: int | None = None
) -> tuple[pd.Series, pd.Series, float, int]:
    """OLS of ``y`` on ``X`` plus an intercept, with Newey-West (Bartlett) HAC t-statistics.

    Rows are aligned on the index (inner join) and rows with any NaN are dropped. Returns
    ``(coef, tstat, r2, n)``; ``coef`` / ``tstat`` are indexed ``['const', *X.columns]``.

    Covariance: V = (X'X)^-1 [sum_t u_t^2 x_t x_t' + sum_{l=1..L} w_l sum_t u_t u_{t-l}
    (x_t x_{t-l}' + x_{t-l} x_t')] (X'X)^-1 * n / (n - k), with w_l = 1 - l / (L + 1) and
    L = floor(4 (n/100)^(2/9)) by default. ``lags=0`` gives White/HC1; with only an intercept
    and ``lags=0`` the t-stat equals the classic mean / (s / sqrt(n)).
    ``r2`` is the centred R^2 (NaN if y is constant). t-stats are NaN when a standard error is 0.
    """
    if isinstance(X, pd.Series):
        X = X.to_frame()
    X = pd.DataFrame(X)
    if "const" in X.columns:
        raise ValueError("X must not contain a 'const' column; the intercept is added automatically")
    data = pd.concat([pd.Series(y, dtype=float).rename("__y__"), X.astype(float)], axis=1, join="inner")
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    names = ["const", *[str(c) for c in X.columns]]
    n, k = len(data), len(names)
    if n <= k:
        raise ValueError(f"need more observations ({n}) than regressors ({k}) for OLS")

    yv = data["__y__"].to_numpy()
    Xv = np.column_stack([np.ones(n), data.drop(columns="__y__").to_numpy()])
    beta, *_ = np.linalg.lstsq(Xv, yv, rcond=None)
    resid = yv - Xv @ beta

    L = default_nw_lags(n) if lags is None else int(lags)
    if L < 0:
        raise ValueError("lags must be >= 0")
    L = min(L, n - 1)
    xu = Xv * resid[:, None]
    meat = xu.T @ xu
    for lag in range(1, L + 1):
        gamma = xu[lag:].T @ xu[:-lag]
        meat += (1.0 - lag / (L + 1.0)) * (gamma + gamma.T)
    bread = np.linalg.pinv(Xv.T @ Xv)
    cov = bread @ meat @ bread * (n / (n - k))
    se = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    scale = np.maximum(np.abs(beta), 1.0) * 1e-12
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(se > scale, beta / se, np.nan)

    sst = float(((yv - yv.mean()) ** 2).sum())
    r2 = 1.0 - float(resid @ resid) / sst if sst > 0 else float("nan")
    return pd.Series(beta, index=names), pd.Series(t, index=names), r2, int(n)


def _finite(x: float) -> float:
    """Required float fields must serialise: undefined (NaN/inf) values are reported as 0.0."""
    x = float(x)
    return x if math.isfinite(x) else 0.0


def factor_regression(
    returns: pd.Series,
    factors: pd.DataFrame,
    *,
    model: str,
    factor_source: str,
    excess: bool = True,
    lags: int | None = None,
) -> FactorRegression:
    """Regress a strategy's periodic returns on a factor model with Newey-West t-stats.

    ``factors`` holds the factor returns as fractions (columns per ``FACTOR_COLUMNS[model]`` and,
    when ``excess`` is True, ``RF``) at the same periodicity as ``returns``. Dates are inner
    joined. ``y = returns - RF`` when ``excess`` (a long-only portfolio), otherwise ``returns``
    (a self-financing long-short portfolio). ``alpha_annual_pct = const * periods_per_year *
    100`` (arithmetic annualisation; periodicity inferred from the aligned dates). Requires at
    least ``MIN_REGRESSION_OBS`` (24) aligned observations. Undefined t-stats / R^2 (perfect fit,
    constant y) are reported as 0.0.
    """
    if model not in FACTOR_COLUMNS:
        raise ValueError(f"unknown factor model {model!r}; expected one of {sorted(FACTOR_COLUMNS)}")
    cols = FACTOR_COLUMNS[model]
    needed = cols + (["RF"] if excess else [])
    missing = [c for c in needed if c not in factors.columns]
    if missing:
        raise ValueError(f"factor data for {model} is missing column(s): {', '.join(missing)}")

    r = pd.Series(returns, dtype=float)
    f = factors[needed].astype(float)
    if not isinstance(r.index, pd.DatetimeIndex):
        r.index = pd.DatetimeIndex(r.index)
    if not isinstance(f.index, pd.DatetimeIndex):
        f = f.copy()
        f.index = pd.DatetimeIndex(f.index)
    data = pd.concat([r.rename("__r__"), f], axis=1, join="inner")
    data = data.replace([np.inf, -np.inf], np.nan).dropna().sort_index()
    if len(data) < MIN_REGRESSION_OBS:
        raise ValueError(
            f"{model} regression needs at least {MIN_REGRESSION_OBS} overlapping observations, "
            f"got {len(data)} (check that returns and factors share the same dates/periodicity)"
        )
    y = data["__r__"] - data["RF"] if excess else data["__r__"]
    coef, tstat, r2, n = ols_newey_west(y, data[cols], lags=lags)
    ppy = infer_periods_per_year(data.index)
    return FactorRegression(
        model=model,
        factor_source=factor_source,
        n=n,
        alpha_annual_pct=float(coef["const"]) * ppy * 100.0,
        alpha_t_stat=_finite(tstat["const"]),
        betas={c: float(coef[c]) for c in cols},
        beta_t_stats={c: _finite(tstat[c]) for c in cols},
        r_squared=_finite(r2),
    )
