"""Composite ranking of screen survivors from the spec's RankFactors.

Method
------
* Each factor is scored by its percentile rank among the survivors that have a value:
  ``(rank - 1) / (n - 1)`` with average ranks for ties, so the best value scores 1.0 and the worst
  0.0. ``lower_is_better`` factors are mirrored (``1 - percentile``).
* A survivor without a value for a factor (NaN / +-inf / non-numeric) gets the neutral 0.5 for it.
* A factor observed for a single survivor scores 1.0 for that survivor (it tops its own
  cross-section), so a lone survivor with complete data scores 1.0 overall.
* Weights are normalised to sum to 1 and the composite is the weighted mean, in [0, 1]. Factors
  repeated with the same direction have their weights added; conflicting directions are an error.
* Scores are rounded to 12 decimals (so mathematically equal composites tie exactly), sorted by
  score descending then ticker ascending, and the first ``top_n`` get ranks 1..top_n.
* Percentile ranks are already robust to outliers (only the order matters), so no winsorisation is
  applied; ``winsor`` is validated and kept for API compatibility with a z-score variant.
"""

from __future__ import annotations

import math
from datetime import date, datetime

import numpy as np
import pandas as pd

from aitrading.core.models import RankedCandidate
from aitrading.screen.spec import RankFactor

NEUTRAL_SCORE = 0.5
_DECIMALS = 12


def factor_percentile(values: pd.Series, higher_is_better: bool = True) -> pd.Series:
    """Direction-aware percentile score in [0, 1] per row; missing values get NEUTRAL_SCORE."""
    x = pd.to_numeric(values, errors="coerce").astype("float64")
    x = x.where(np.isfinite(x))
    valid = x.notna()
    n = int(valid.sum())
    out = pd.Series(NEUTRAL_SCORE, index=values.index, dtype="float64")
    if n == 1:
        out[valid] = 1.0
    elif n > 1:
        pct = (x[valid].rank(method="average") - 1.0) / (n - 1)
        out[valid] = pct if higher_is_better else 1.0 - pct
    return out


def _merge_factors(ranking: list[RankFactor]) -> dict[str, tuple[bool, float]]:
    """feature -> (higher_is_better, normalised weight), in first-seen order."""
    merged: dict[str, tuple[bool, float]] = {}
    for f in ranking:
        higher = f.direction == "higher_is_better"
        if f.feature in merged:
            prev_higher, prev_w = merged[f.feature]
            if prev_higher != higher:
                raise ValueError(f"ranking factor '{f.feature}' appears with conflicting directions")
            merged[f.feature] = (higher, prev_w + f.weight)
        else:
            merged[f.feature] = (higher, f.weight)
    total = sum(w for _, w in merged.values())
    return {k: (h, w / total) for k, (h, w) in merged.items()}


def _py(value: object) -> float | str | None:
    """JSON-friendly python value: numbers -> float (non-finite -> None), dates -> ISO, else str."""
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_, int, float, np.integer, np.floating)):
        f = float(value)
        return f if math.isfinite(f) else None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return str(value)


def rank_candidates(
    frame: pd.DataFrame,
    ranking: list[RankFactor],
    survivors: list[str],
    top_n: int,
    *,
    winsor: tuple[float, float] = (0.02, 0.98),
    extra_features: list[str] | None = None,
) -> list[RankedCandidate]:
    """Rank ``survivors`` (rows of ``frame``) by the weighted percentile composite of ``ranking``.

    Returns at most ``top_n`` candidates, best first. Raises ValueError when ``ranking`` is empty,
    a ranking feature is not a frame column, a survivor is not in the frame, or top_n < 1.
    """
    lo, hi = winsor
    if not 0.0 <= lo < hi <= 1.0:
        raise ValueError(f"winsor bounds must satisfy 0 <= low < high <= 1, got {winsor}")
    if top_n < 1:
        raise ValueError(f"top_n must be >= 1, got {top_n}")
    if not ranking:
        raise ValueError("ranking needs at least one factor")
    factors = _merge_factors(ranking)
    absent = [f for f in factors if f not in frame.columns]
    if absent:
        raise ValueError(f"ranking features not in the feature frame: {', '.join(absent)}")
    tickers = list(dict.fromkeys(survivors))
    unknown = [t for t in tickers if t not in frame.index]
    if unknown:
        raise ValueError(f"survivors not in the feature frame: {', '.join(map(str, unknown[:10]))}")
    if not tickers:
        return []

    sub = frame.loc[tickers]
    if sub.index.has_duplicates:
        raise ValueError("feature frame has duplicate rows for some survivors")
    scores = pd.DataFrame({f: factor_percentile(sub[f], higher) for f, (higher, _) in factors.items()}, index=sub.index)
    weights = pd.Series({f: w for f, (_, w) in factors.items()})
    composite = (scores[weights.index] * weights).sum(axis=1).clip(0.0, 1.0).round(_DECIMALS)

    score_of = composite.to_dict()
    order = sorted(tickers, key=lambda t: (-score_of[t], str(t)))[:top_n]
    shown = list(dict.fromkeys([*factors, *(extra_features or [])]))
    names = sub["name"] if "name" in sub.columns else pd.Series(index=sub.index, dtype=object)

    out: list[RankedCandidate] = []
    for rank, t in enumerate(order, start=1):
        name = _py(names[t])
        row = sub.loc[t]
        out.append(
            RankedCandidate(
                ticker=str(t),
                name=name if isinstance(name, str) and name else str(t),
                rank=rank,
                score=float(score_of[t]),
                factor_scores={f: float(scores.at[t, f]) for f in factors},
                features={f: (_py(row[f]) if f in sub.columns else None) for f in shown},
            )
        )
    return out
