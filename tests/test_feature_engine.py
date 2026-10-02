"""Tests for aitrading.screen.features (FeatureEngine) against the synthetic provider."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from aitrading.core import fields
from aitrading.data.base import Capability, MarketDataProvider, ProviderError
from aitrading.data.synthetic import SyntheticProvider
from aitrading.screen.catalog import FUNDAMENTAL_FEATURES, POSITIONING_FEATURES, TECHNICAL_FEATURES, default_catalog
from aitrading.screen.engine import run_screen
from aitrading.screen.features import (
    CROSS_SECTIONAL_FEATURES,
    FEATURE_DATASETS,
    LOOKBACK_CALENDAR_DAYS,
    RAW_REFERENCE_COLUMNS,
    FeatureEngine,
    FeatureFrame,
)
from aitrading.screen.spec import Condition, RankFactor, ScreenSpec, UniverseSpec

AS_OF = date(2026, 9, 30)
CATALOG = default_catalog()
ALL = CATALOG.names()


class Recording:
    """Delegating provider that records dataset calls and can drop capabilities or fail on demand."""

    def __init__(self, inner: SyntheticProvider, *, drop: set[Capability] = frozenset(), fail: dict[str, Exception] | None = None):
        self.inner = inner
        self.name = inner.name
        self.boundary = inner.boundary
        self.capabilities = set(inner.capabilities) - set(drop)
        self.fail = fail or {}
        self.calls: list[tuple[str, tuple]] = []

    def _do(self, method: str, *args):
        self.calls.append((method, args))
        if method in self.fail:
            raise self.fail[method]
        return getattr(self.inner, method)(*args)

    def methods(self) -> set[str]:
        return {m for m, _ in self.calls}

    def get_universe(self, spec, as_of):
        return self._do("get_universe", spec, as_of)

    def get_price_history(self, tickers, start, end):
        return self._do("get_price_history", tickers, start, end)

    def get_benchmark_history(self, start, end, symbol=None):
        return self._do("get_benchmark_history", start, end)

    def get_fundamentals(self, tickers, as_of):
        return self._do("get_fundamentals", tickers, as_of)

    def get_estimates(self, tickers, as_of):
        return self._do("get_estimates", tickers, as_of)

    def get_short_interest(self, tickers, as_of):
        return self._do("get_short_interest", tickers, as_of)

    def get_options_summary(self, tickers, as_of):
        return self._do("get_options_summary", tickers, as_of)

    def get_documents(self, ticker, kinds, start, end, limit=10):
        return self._do("get_documents", ticker, kinds, start, end, limit)


@pytest.fixture(scope="module")
def provider() -> SyntheticProvider:
    return SyntheticProvider()


@pytest.fixture(scope="module")
def universe(provider) -> pd.DataFrame:
    return provider.get_universe(UniverseSpec(), AS_OF)


@pytest.fixture(scope="module")
def full(provider, universe) -> FeatureFrame:
    return FeatureEngine(provider).build(universe, AS_OF)


def canonical_spec() -> ScreenSpec:
    c = Condition
    return ScreenSpec(
        name="canonical",
        observation="canonical demo",
        conditions=[
            c(feature="market_cap_usd_bn", op="between", value=2, value_high=20),
            c(feature="sma_50_vs_sma_200_pct", op=">", value=0),
            c(feature="return_12m_ex_1m_pct", op=">", value=0),
            c(feature="drawdown_from_52w_high_pct", op="between", value=-40, value_high=-15),
            c(feature="max_volume_ratio_20d", op=">=", value=2),
            c(feature="rsi_14", op="<", value=40),
            c(feature="fcf_yield_pct", op=">", value=4),
            c(feature="revenue_growth_yoy_pct", op=">", value=8),
            c(feature="short_interest_pct_float", op=">", value=6),
        ],
        ranking=[
            RankFactor(feature="fcf_yield_pct", direction="higher_is_better"),
            RankFactor(feature="revenue_growth_yoy_pct", direction="higher_is_better"),
            RankFactor(feature="drawdown_from_52w_high_pct", direction="lower_is_better"),
        ],
    )


# --------------------------------------------------------------------------------------------
# Full build
# --------------------------------------------------------------------------------------------


def test_recording_wrapper_is_a_provider(provider):
    assert isinstance(Recording(provider), MarketDataProvider)


def test_full_build_has_every_catalog_column(full, universe):
    f = full.frame
    assert list(f.columns) == RAW_REFERENCE_COLUMNS + ALL
    assert f.index.name == "ticker"
    assert list(f.index) == list(universe.index)
    assert full.warnings == []
    assert full.universe.index.equals(f.index)
    for name in ALL:
        if CATALOG[name].dtype == "category":
            assert f[name].dtype == object
        else:
            assert f[name].dtype == np.float64, name


def test_full_build_coverage(full):
    assert set(full.coverage) == set(ALL)
    assert all(0.0 <= v <= 1.0 for v in full.coverage.values())
    expected = {n: float(full.frame[n].notna().mean()) for n in ALL}
    assert full.coverage == pytest.approx(expected)
    # The synthetic provider fills every dataset; only short-history or option-less names are missing.
    for name in TECHNICAL_FEATURES + FUNDAMENTAL_FEATURES + ["market_cap_usd_bn", "gics_sector", "short_interest_pct_float"]:
        assert full.coverage[name] > 0.85, name
    assert full.coverage["iv_30d_pct"] > 0.3


def test_reference_features(full, universe):
    f = full.frame
    np.testing.assert_allclose(f["market_cap_usd_bn"], universe[fields.MARKET_CAP].astype(float) / 1e9)
    assert (f["gics_sector"] == universe[fields.GICS_SECTOR].astype(object)).all()
    assert (f["gics_industry"] == universe[fields.GICS_INDUSTRY].astype(object)).all()
    assert (f["exchange"] == universe[fields.EXCHANGE].astype(object)).all()
    assert (f["name"] == universe[fields.NAME].astype(object)).all()
    assert set(f["country"]) == {"US"}
    assert set(f["security_type"]) == {"common_stock"}


def test_values_match_direct_engine_calls(provider, universe, full):
    from aitrading.fundamental.features import compute_fundamental_features
    from aitrading.technical.features import compute_technical_features

    tickers = list(universe.index)
    start = AS_OF - timedelta(days=LOOKBACK_CALENDAR_DAYS)
    panel = provider.get_price_history(tickers, start, AS_OF)
    tech = compute_technical_features(panel, provider.get_benchmark_history(start, AS_OF), AS_OF)
    pd.testing.assert_frame_equal(full.frame[TECHNICAL_FEATURES], tech, check_names=False)
    fund = compute_fundamental_features(
        universe, provider.get_fundamentals(tickers, AS_OF), provider.get_estimates(tickers, AS_OF), tech["price"], AS_OF
    )
    pd.testing.assert_frame_equal(full.frame[FUNDAMENTAL_FEATURES], fund, check_names=False)


def test_frame_runs_through_screen(full, provider):
    out = run_screen(canonical_spec(), full.frame)
    assert len(out.survivors) >= 5
    arch = {provider.archetype(t) for t in out.survivors}
    assert arch <= {"transitory_shock", "value_trap", "guidance_reset", "sector_contagion"}


# --------------------------------------------------------------------------------------------
# Selective fetching
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "features, expected",
    [
        ({"rsi_14"}, {"get_price_history"}),
        ({"beta_1y"}, {"get_price_history", "get_benchmark_history"}),
        ({"fcf_yield_pct"}, {"get_fundamentals"}),
        ({"eps_revision_3m_pct"}, {"get_estimates"}),
        ({"pe_ntm"}, {"get_estimates", "get_price_history"}),
        ({"short_interest_pct_float"}, {"get_short_interest"}),
        ({"days_to_cover"}, {"get_short_interest", "get_price_history"}),
        ({"iv_rank_1y"}, {"get_options_summary"}),
        ({"iv_to_realized_vol_ratio"}, {"get_options_summary", "get_price_history"}),
        ({"market_cap_usd_bn", "gics_sector"}, set()),
    ],
)
def test_fetches_only_needed_datasets(provider, universe, features, expected):
    rec = Recording(provider)
    ff = FeatureEngine(rec).build(universe, AS_OF, features)
    assert rec.methods() == expected
    assert set(ff.coverage) == features
    for f in features:
        assert ff.coverage[f] > 0.3, f
    assert list(ff.frame.columns) == RAW_REFERENCE_COLUMNS + ALL


def test_unrequested_datasets_are_nan(provider, universe):
    ff = FeatureEngine(provider).build(universe, AS_OF, {"rsi_14"})
    f = ff.frame
    assert f[FUNDAMENTAL_FEATURES + POSITIONING_FEATURES].isna().all().all()
    assert f[["rel_strength_3m_pp", "beta_1y"]].isna().all().all()  # benchmark not fetched
    assert f["rsi_14"].notna().all()
    assert f["return_6m_percentile"].notna().any()  # computed from the fetched prices anyway


def test_price_request_window(provider, universe):
    rec = Recording(provider)
    FeatureEngine(rec).build(universe, AS_OF, {"price"})
    (method, (tickers, start, end)), = rec.calls
    assert method == "get_price_history"
    assert tickers == list(universe.index)
    assert start == AS_OF - timedelta(days=LOOKBACK_CALENDAR_DAYS) and end == AS_OF


def test_as_of_accepts_datetime_and_string(provider, universe, full):
    a = FeatureEngine(provider).build(universe, pd.Timestamp("2026-09-30 16:00"), {"rsi_14", "fcf_yield_pct"})
    b = FeatureEngine(provider).build(universe, "2026-09-30", {"rsi_14", "fcf_yield_pct"})
    pd.testing.assert_series_equal(a.frame["rsi_14"], full.frame["rsi_14"])
    pd.testing.assert_series_equal(b.frame["fcf_yield_pct"], full.frame["fcf_yield_pct"])


def test_feature_dataset_map_covers_catalog():
    assert set(FEATURE_DATASETS) == set(ALL)
    for f in CATALOG:
        if f.source == "reference":
            assert FEATURE_DATASETS[f.name] == frozenset()
        else:
            assert FEATURE_DATASETS[f.name], f.name
    assert CROSS_SECTIONAL_FEATURES <= set(TECHNICAL_FEATURES)


# --------------------------------------------------------------------------------------------
# Degradation: missing capabilities and provider errors
# --------------------------------------------------------------------------------------------


def test_missing_capability_gives_nan_and_warning(provider, universe):
    rec = Recording(provider, drop={Capability.OPTIONS, Capability.SHORT_INTEREST})
    ff = FeatureEngine(rec).build(universe, AS_OF, None)
    assert "get_options_summary" not in rec.methods()
    assert "get_short_interest" not in rec.methods()
    assert ff.frame[POSITIONING_FEATURES].isna().all().all()
    assert all(ff.coverage[f] == 0.0 for f in POSITIONING_FEATURES)
    text = " | ".join(ff.warnings)
    assert "lacks capability 'options'" in text and "lacks capability 'short_interest'" in text
    assert "iv_30d_pct" in text and "days_to_cover" in text
    assert ff.frame["fcf_yield_pct"].notna().mean() > 0.85  # everything else unaffected
    assert ff.frame["rsi_14"].notna().mean() > 0.85


def test_missing_capability_not_needed_is_silent(provider, universe):
    rec = Recording(provider, drop={Capability.OPTIONS})
    ff = FeatureEngine(rec).build(universe, AS_OF, {"rsi_14", "fcf_yield_pct"})
    assert ff.warnings == []


def test_missing_price_capability_degrades(provider, universe):
    rec = Recording(provider, drop={Capability.PRICES})
    ff = FeatureEngine(rec).build(universe, AS_OF, {"rsi_14", "pe_ntm", "fcf_yield_pct"})
    assert "get_price_history" not in rec.methods()
    assert ff.frame[TECHNICAL_FEATURES].isna().all().all()
    assert ff.frame["pe_ntm"].isna().all()
    assert ff.frame["fcf_yield_pct"].notna().any()  # needs no price: market cap is in the universe
    assert any("lacks capability 'prices'" in w and "rsi_14" in w and "pe_ntm" in w for w in ff.warnings)


@pytest.mark.parametrize(
    "method, features, nan_features",
    [
        ("get_estimates", {"eps_revision_3m_pct", "fcf_yield_pct"}, ["eps_revision_3m_pct"]),
        ("get_short_interest", {"short_interest_pct_float", "rsi_14"}, ["short_interest_pct_float"]),
        ("get_options_summary", {"iv_30d_pct", "rsi_14"}, ["iv_30d_pct"]),
        ("get_benchmark_history", {"beta_1y", "rel_strength_6m_pp", "rsi_14"}, ["beta_1y", "rel_strength_6m_pp"]),
    ],
)
def test_optional_dataset_errors_degrade(provider, universe, method, features, nan_features):
    rec = Recording(provider, fail={method: ProviderError("entitlement denied")})
    ff = FeatureEngine(rec).build(universe, AS_OF, features)
    for f in nan_features:
        assert ff.frame[f].isna().all(), f
        assert ff.coverage[f] == 0.0
    for f in features - set(nan_features):
        assert ff.coverage[f] > 0.85, f
    assert any("request failed" in w and "entitlement denied" in w for w in ff.warnings)


def test_optional_dataset_unexpected_exception_degrades(provider, universe):
    rec = Recording(provider, fail={"get_options_summary": KeyError("boom")})
    ff = FeatureEngine(rec).build(universe, AS_OF, {"iv_30d_pct"})
    assert ff.frame["iv_30d_pct"].isna().all()
    assert any("KeyError" in w for w in ff.warnings)


@pytest.mark.parametrize("method, feature", [("get_price_history", "rsi_14"), ("get_fundamentals", "fcf_yield_pct")])
def test_required_dataset_errors_raise(provider, universe, method, feature):
    rec = Recording(provider, fail={method: ProviderError("vendor down")})
    with pytest.raises(ProviderError, match="vendor down"):
        FeatureEngine(rec).build(universe, AS_OF, {feature})


def test_required_dataset_non_provider_error_is_wrapped(provider, universe):
    rec = Recording(provider, fail={"get_price_history": RuntimeError("socket closed")})
    with pytest.raises(ProviderError, match="prices request failed.*socket closed"):
        FeatureEngine(rec).build(universe, AS_OF, {"rsi_14"})


# --------------------------------------------------------------------------------------------
# Edge cases
# --------------------------------------------------------------------------------------------


def test_empty_universe(provider, universe):
    rec = Recording(provider)
    ff = FeatureEngine(rec).build(universe.iloc[:0], AS_OF, None)
    assert rec.calls == []
    assert ff.frame.empty and list(ff.frame.columns) == RAW_REFERENCE_COLUMNS + ALL
    assert set(ff.coverage) == set(ALL) and all(v == 0.0 for v in ff.coverage.values())
    assert ff.warnings == []


def test_unknown_feature_names_warn(provider, universe):
    ff = FeatureEngine(provider).build(universe.iloc[:5], AS_OF, {"rsi_14", "not_a_feature"})
    assert set(ff.coverage) == {"rsi_14"}
    assert any("not_a_feature" in w for w in ff.warnings)


def test_duplicate_universe_rows_are_dropped(provider, universe):
    dup = pd.concat([universe.iloc[:3], universe.iloc[:2]])
    ff = FeatureEngine(provider).build(dup, AS_OF, {"rsi_14"})
    assert list(ff.frame.index) == list(universe.index[:3])
    assert ff.frame.index.is_unique


def test_unknown_ticker_gets_nan_row(provider, universe):
    extra = universe.iloc[:2].copy()
    extra.index = pd.Index(["ZZZZ1", "ZZZZ2"], name="ticker")
    uni = pd.concat([universe.iloc[:3], extra])
    ff = FeatureEngine(provider).build(uni, AS_OF, None)
    assert ff.frame.loc[["ZZZZ1", "ZZZZ2"], TECHNICAL_FEATURES + FUNDAMENTAL_FEATURES].isna().all().all()
    assert ff.frame.loc[universe.index[:3], "rsi_14"].notna().all()


def test_bad_market_cap_and_blank_labels(provider, universe):
    uni = universe.iloc[:4].copy()
    uni[fields.MARKET_CAP] = [np.nan, 0.0, -5e9, 3e9]
    uni[fields.GICS_SECTOR] = uni[fields.GICS_SECTOR].astype(object)
    uni.iloc[0, uni.columns.get_loc(fields.GICS_SECTOR)] = "  "
    uni.iloc[1, uni.columns.get_loc(fields.GICS_SECTOR)] = None
    ff = FeatureEngine(provider).build(uni, AS_OF, {"market_cap_usd_bn", "gics_sector"})
    assert ff.frame["market_cap_usd_bn"].iloc[:3].isna().all()
    assert ff.frame["market_cap_usd_bn"].iloc[3] == pytest.approx(3.0)
    assert ff.frame["gics_sector"].iloc[0] is None and ff.frame["gics_sector"].iloc[1] is None
    assert ff.coverage["gics_sector"] == pytest.approx(0.5)
    assert ff.coverage["market_cap_usd_bn"] == pytest.approx(0.25)


def test_universe_without_reference_columns(provider, universe):
    uni = universe.iloc[:3][[fields.MARKET_CAP]]
    ff = FeatureEngine(provider).build(uni, AS_OF, {"rsi_14", "gics_sector"})
    assert ff.frame["gics_sector"].isna().all() and ff.frame["name"].isna().all()
    assert ff.frame["rsi_14"].notna().all()


def test_short_history_before_provider_start(universe):
    p = SyntheticProvider(n_tickers=40, seed=3, start=date(2026, 6, 1), end=date(2026, 9, 30))
    uni = p.get_universe(UniverseSpec(), AS_OF)
    ff = FeatureEngine(p).build(uni, AS_OF, {"sma_200", "rsi_14", "return_12m_pct"})
    assert ff.coverage["sma_200"] == 0.0 and ff.coverage["return_12m_pct"] == 0.0
    assert ff.coverage["rsi_14"] > 0.9


def test_provider_without_capabilities_attribute_is_asked_for_everything(provider, universe):
    class Bare:
        name = "bare"

        def __getattr__(self, item):
            if item == "capabilities":
                raise AttributeError(item)
            return getattr(provider, item)

    bare = Bare()
    ff = FeatureEngine(bare).build(universe.iloc[:10], AS_OF, {"iv_30d_pct", "fcf_yield_pct"})
    assert ff.warnings == []
    assert ff.coverage["fcf_yield_pct"] > 0.5
