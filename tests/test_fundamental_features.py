"""Tests for aitrading.fundamental.features.compute_fundamental_features."""

from __future__ import annotations

import math
from datetime import date

import numpy as np
import pandas as pd
import pytest

from aitrading.core import fields
from aitrading.fundamental.features import compute_fundamental_features
from aitrading.screen.catalog import FUNDAMENTAL_FEATURES

AS_OF = date(2026, 9, 30)
NAN = float("nan")


def frame(rows: dict[str, dict], columns: list[str]) -> pd.DataFrame:
    df = pd.DataFrame.from_dict(rows, orient="index").reindex(columns=columns)
    df.index.name = fields.TICKER
    return df


def universe(mcaps: dict[str, float]) -> pd.DataFrame:
    df = pd.DataFrame({fields.MARKET_CAP: pd.Series(mcaps, dtype="float64")})
    df[fields.NAME] = [f"{t} Corp" for t in df.index]
    df.index.name = fields.TICKER
    return df


@pytest.fixture()
def inputs():
    uni = universe({"AAA": 10e9, "BBB": 4e9, "CCC": 2e9, "DDD": 3e9})
    fa = frame(
        {
            "AAA": {
                fields.PERIOD_END: date(2026, 6, 30), fields.REPORT_DATE: date(2026, 8, 5),
                fields.REVENUE_TTM: 5e9, fields.REVENUE_TTM_PRIOR_YEAR: 4e9,
                fields.REVENUE_LAST_Q: 1.4e9, fields.REVENUE_LAST_Q_PRIOR_YEAR: 1.12e9,
                fields.GROSS_PROFIT_TTM: 2e9, fields.GROSS_PROFIT_TTM_PRIOR_YEAR: 1.4e9,
                fields.OPERATING_INCOME_TTM: 1e9, fields.OPERATING_INCOME_TTM_PRIOR_YEAR: 0.6e9,
                fields.EBITDA_TTM: 1.25e9, fields.NET_INCOME_TTM: 0.8e9,
                fields.CFO_TTM: 1.0e9, fields.CAPEX_TTM: 0.2e9, fields.FCF_TTM: 0.8e9,
                fields.TOTAL_DEBT: 3e9, fields.CASH: 1e9, fields.INTEREST_EXPENSE_TTM: 0.05e9,
                fields.TOTAL_EQUITY: 4e9, fields.SHARES_OUTSTANDING: 2e8,
                fields.TOTAL_ASSETS: 12e9, fields.TOTAL_ASSETS_PRIOR_YEAR: 10e9,
            },
            "BBB": {  # shrinking loss-maker, FCF from CFO - capex
                fields.PERIOD_END: date(2026, 6, 30), fields.REPORT_DATE: "2026-09-10",
                fields.REVENUE_TTM: 2e9, fields.REVENUE_TTM_PRIOR_YEAR: 2.5e9,
                fields.REVENUE_LAST_Q: 0.45e9, fields.REVENUE_LAST_Q_PRIOR_YEAR: 0.6e9,
                fields.GROSS_PROFIT_TTM: 0.6e9, fields.GROSS_PROFIT_TTM_PRIOR_YEAR: 0.875e9,
                fields.OPERATING_INCOME_TTM: -0.1e9, fields.OPERATING_INCOME_TTM_PRIOR_YEAR: 0.125e9,
                fields.EBITDA_TTM: -0.05e9, fields.NET_INCOME_TTM: -0.2e9,
                fields.CFO_TTM: 0.1e9, fields.CAPEX_TTM: 0.3e9, fields.FCF_TTM: NAN,
                fields.TOTAL_DEBT: 1e9, fields.CASH: 0.5e9, fields.INTEREST_EXPENSE_TTM: 0.08e9,
                fields.TOTAL_EQUITY: 1e9, fields.SHARES_OUTSTANDING: 2e8,
                fields.TOTAL_ASSETS: 5e9, fields.TOTAL_ASSETS_PRIOR_YEAR: 5.5e9,
            },
            "CCC": {  # net cash, debt-free, zero / negative denominators
                fields.PERIOD_END: date(2026, 6, 30), fields.REPORT_DATE: pd.NaT,
                fields.REVENUE_TTM: 1e9, fields.REVENUE_TTM_PRIOR_YEAR: 0.0,
                fields.REVENUE_LAST_Q: 0.3e9, fields.REVENUE_LAST_Q_PRIOR_YEAR: NAN,
                fields.GROSS_PROFIT_TTM: 0.5e9, fields.GROSS_PROFIT_TTM_PRIOR_YEAR: NAN,
                fields.OPERATING_INCOME_TTM: 0.2e9, fields.OPERATING_INCOME_TTM_PRIOR_YEAR: 0.1e9,
                fields.EBITDA_TTM: 0.0, fields.NET_INCOME_TTM: 0.1e9,
                fields.CFO_TTM: 0.2e9, fields.CAPEX_TTM: 0.05e9, fields.FCF_TTM: 0.15e9,
                fields.TOTAL_DEBT: 0.0, fields.CASH: 2.5e9, fields.INTEREST_EXPENSE_TTM: NAN,
                fields.TOTAL_EQUITY: -0.5e9, fields.SHARES_OUTSTANDING: 2e8,
                fields.TOTAL_ASSETS: 4e9, fields.TOTAL_ASSETS_PRIOR_YEAR: NAN,
            },
        },
        fields.FUNDAMENTAL_COLUMNS + fields.FUNDAMENTAL_OPTIONAL_COLUMNS,
    )
    est = frame(
        {
            "AAA": {
                fields.REVENUE_NTM_EST: 5.5e9, fields.REVENUE_NTM_EST_3M_AGO: 5.0e9,
                fields.EPS_NTM_EST: 4.0, fields.EPS_NTM_EST_3M_AGO: 3.2, fields.EPS_TTM: 3.2,
                fields.NUM_ANALYSTS: 12, fields.TARGET_PRICE_MEAN: 60.0, fields.LAST_EPS_SURPRISE: 0.05,
                fields.LAST_EARNINGS_DATE: date(2026, 8, 4), fields.NEXT_EARNINGS_DATE: pd.Timestamp("2026-11-03"),
            },
            "BBB": {
                fields.REVENUE_NTM_EST: 1.8e9, fields.REVENUE_NTM_EST_3M_AGO: 2.0e9,
                fields.EPS_NTM_EST: -0.5, fields.EPS_NTM_EST_3M_AGO: -1.0, fields.EPS_TTM: -1.2,
                fields.NUM_ANALYSTS: 5, fields.TARGET_PRICE_MEAN: 18.0, fields.LAST_EPS_SURPRISE: -0.2,
                fields.LAST_EARNINGS_DATE: None, fields.NEXT_EARNINGS_DATE: None,
            },
            "CCC": {
                fields.REVENUE_NTM_EST: 1.1e9, fields.REVENUE_NTM_EST_3M_AGO: 0.0,
                fields.EPS_NTM_EST: 0.5, fields.EPS_NTM_EST_3M_AGO: 0.0, fields.EPS_TTM: 0.0,
                fields.NUM_ANALYSTS: 0, fields.TARGET_PRICE_MEAN: NAN, fields.LAST_EPS_SURPRISE: NAN,
                fields.LAST_EARNINGS_DATE: "2026-10-15", fields.NEXT_EARNINGS_DATE: "2026-09-01",
            },
        },
        fields.ESTIMATE_COLUMNS,
    )
    price = pd.Series({"AAA": 50.0, "BBB": 20.0, "CCC": 10.0, "DDD": 30.0})
    return uni, fa, est, price


def compute(inputs, as_of=AS_OF):
    return compute_fundamental_features(*inputs, as_of)


def check(row: pd.Series, expected: dict[str, float]) -> None:
    for name, value in expected.items():
        if isinstance(value, float) and math.isnan(value):
            assert math.isnan(row[name]), f"{name}: expected NaN, got {row[name]}"
        else:
            assert row[name] == pytest.approx(value, rel=1e-9, abs=1e-12), name


def test_columns_index_dtype(inputs):
    out = compute(inputs)
    assert list(out.columns) == FUNDAMENTAL_FEATURES
    assert out.index.name == "ticker"
    assert list(out.index) == ["AAA", "BBB", "CCC", "DDD"]
    assert (out.dtypes == "float64").all()


def test_every_feature_hand_computed_profitable_grower(inputs):
    a = compute(inputs).loc["AAA"]
    expected = {
        "enterprise_value_usd_bn": 12.0,  # 10 + 3 - 1
        "fcf_yield_pct": 8.0,  # 0.8 / 10
        "ev_to_ebitda": 9.6,  # 12 / 1.25
        "ev_to_sales": 2.4,  # 12 / 5
        "pe_ntm": 12.5,  # 50 / 4
        "earnings_yield_ntm_pct": 8.0,  # 4 / 50
        "target_price_upside_pct": 20.0,  # 60 / 50 - 1
        "revenue_growth_yoy_pct": 25.0,  # 5 / 4 - 1
        "revenue_growth_last_q_yoy_pct": 25.0,  # 1.4 / 1.12 - 1
        "revenue_growth_ntm_est_pct": 10.0,  # 5.5 / 5 - 1
        "eps_growth_ntm_est_pct": 25.0,  # 4 / 3.2 - 1
        "gross_margin_pct": 40.0,
        "operating_margin_pct": 20.0,
        "ebitda_margin_pct": 25.0,
        "fcf_margin_pct": 16.0,
        "net_margin_pct": 16.0,
        "gross_margin_change_yoy_pp": 5.0,  # 40 - 35
        "operating_margin_change_yoy_pp": 5.0,  # 20 - 15
        "fcf_conversion_pct": 100.0,  # 0.8 / 0.8
        "roe_pct": 20.0,  # 0.8 / 4
        "net_debt_usd_bn": 2.0,
        "net_debt_to_ebitda": 1.6,  # 2 / 1.25
        "interest_coverage": 20.0,  # 1 / 0.05
        "cash_pct_market_cap": 10.0,
        "eps_revision_3m_pct": 25.0,  # (4 - 3.2) / 3.2
        "revenue_revision_3m_pct": 10.0,  # 5.5 / 5 - 1
        "num_analysts": 12.0,
        "last_eps_surprise_pct": 5.0,
        "days_since_last_earnings": 57.0,  # 2026-08-04 -> 2026-09-30
        "days_to_next_earnings": 34.0,  # 2026-09-30 -> 2026-11-03
        "book_to_market": 0.4,  # equity 4 / market cap 10
        "operating_profitability_pct": 23.75,  # (1 - 0.05) / 4
        "asset_growth_yoy_pct": 20.0,  # 12 / 10 - 1
        "gross_profitability_pct": 2 / 12 * 100,  # gross profit / total assets
        "earnings_yield_ttm_pct": 8.0,  # 0.8 / 10
    }
    assert set(expected) == set(FUNDAMENTAL_FEATURES)
    check(a, expected)


def test_every_feature_hand_computed_loss_maker(inputs):
    b = compute(inputs).loc["BBB"]
    expected = {
        "enterprise_value_usd_bn": 4.5,
        "fcf_yield_pct": -5.0,  # (0.1 - 0.3) / 4: fallback to CFO - capex
        "ev_to_ebitda": NAN,  # EBITDA <= 0
        "ev_to_sales": 2.25,
        "pe_ntm": NAN,  # EPS <= 0
        "earnings_yield_ntm_pct": -2.5,  # -0.5 / 20
        "target_price_upside_pct": -10.0,
        "revenue_growth_yoy_pct": -20.0,
        "revenue_growth_last_q_yoy_pct": -25.0,
        "revenue_growth_ntm_est_pct": -10.0,
        "eps_growth_ntm_est_pct": NAN,  # TTM EPS <= 0
        "gross_margin_pct": 30.0,
        "operating_margin_pct": -5.0,
        "ebitda_margin_pct": -2.5,
        "fcf_margin_pct": -10.0,
        "net_margin_pct": -10.0,
        "gross_margin_change_yoy_pp": -5.0,  # 30 - 35
        "operating_margin_change_yoy_pp": -10.0,  # -5 - 5
        "fcf_conversion_pct": NAN,  # net income <= 0
        "roe_pct": -20.0,
        "net_debt_usd_bn": 0.5,
        "net_debt_to_ebitda": NAN,
        "interest_coverage": -1.25,  # -0.1 / 0.08
        "cash_pct_market_cap": 12.5,
        "eps_revision_3m_pct": 50.0,  # (-0.5 - -1.0) / |-1.0|: upward revision is positive
        "revenue_revision_3m_pct": -10.0,
        "num_analysts": 5.0,
        "last_eps_surprise_pct": -20.0,
        "days_since_last_earnings": 20.0,  # falls back to report_date 2026-09-10
        "days_to_next_earnings": NAN,
        "book_to_market": 0.25,  # 1 / 4
        "operating_profitability_pct": -18.0,  # (-0.1 - 0.08) / 1
        "asset_growth_yoy_pct": 5 / 5.5 * 100 - 100,  # shrinking balance sheet
        "gross_profitability_pct": 12.0,  # 0.6 / 5
        "earnings_yield_ttm_pct": -5.0,  # loss: -0.2 / 4
    }
    assert set(expected) == set(FUNDAMENTAL_FEATURES)
    check(b, expected)


def test_every_feature_hand_computed_denominator_guards(inputs):
    c = compute(inputs).loc["CCC"]
    expected = {
        "enterprise_value_usd_bn": -0.5,  # net cash above market cap
        "fcf_yield_pct": 7.5,
        "ev_to_ebitda": NAN,  # EBITDA == 0
        "ev_to_sales": -0.5,
        "pe_ntm": 20.0,
        "earnings_yield_ntm_pct": 5.0,
        "target_price_upside_pct": NAN,
        "revenue_growth_yoy_pct": NAN,  # prior revenue 0
        "revenue_growth_last_q_yoy_pct": NAN,
        "revenue_growth_ntm_est_pct": 10.0,
        "eps_growth_ntm_est_pct": NAN,  # TTM EPS == 0
        "gross_margin_pct": 50.0,
        "operating_margin_pct": 20.0,
        "ebitda_margin_pct": 0.0,
        "fcf_margin_pct": 15.0,
        "net_margin_pct": 10.0,
        "gross_margin_change_yoy_pp": NAN,
        "operating_margin_change_yoy_pp": NAN,  # prior revenue 0
        "fcf_conversion_pct": 150.0,
        "roe_pct": NAN,  # negative equity
        "net_debt_usd_bn": -2.5,
        "net_debt_to_ebitda": NAN,
        "interest_coverage": 100.0,  # debt-free, no interest expense, operating income > 0
        "cash_pct_market_cap": 125.0,
        "eps_revision_3m_pct": NAN,  # EPS 3m ago == 0
        "revenue_revision_3m_pct": NAN,
        "num_analysts": 0.0,
        "last_eps_surprise_pct": NAN,
        "days_since_last_earnings": NAN,  # date after as_of: not yet public
        "days_to_next_earnings": NAN,  # stale date before as_of
        "book_to_market": NAN,  # negative equity
        "operating_profitability_pct": NAN,  # negative equity
        "asset_growth_yoy_pct": NAN,  # prior-year total assets missing
        "gross_profitability_pct": 12.5,  # 0.5 / 4
        "earnings_yield_ttm_pct": 5.0,  # 0.1 / 2
    }
    assert set(expected) == set(FUNDAMENTAL_FEATURES)
    check(c, expected)


def test_ticker_missing_from_inputs_is_all_nan(inputs):
    d = compute(inputs).loc["DDD"]
    assert d.isna().all()


def test_ticker_missing_everywhere_except_universe_has_no_keyerror(inputs):
    uni, fa, est, price = inputs
    uni2 = pd.concat([uni, universe({"ZZZ": 1e9})])
    out = compute_fundamental_features(uni2, fa, est, price, AS_OF)
    assert out.loc["ZZZ"].isna().all()
    assert list(out.index)[-1] == "ZZZ"


def test_empty_and_none_inputs(inputs):
    uni, _, _, _ = inputs
    out = compute_fundamental_features(uni, None, None, None, AS_OF)
    assert list(out.columns) == FUNDAMENTAL_FEATURES
    assert out.isna().all().all()
    out = compute_fundamental_features(uni, pd.DataFrame(), pd.DataFrame(), pd.Series(dtype=float), AS_OF)
    assert out.shape == (4, len(FUNDAMENTAL_FEATURES)) and out.isna().all().all()


def test_empty_universe():
    out = compute_fundamental_features(universe({}), pd.DataFrame(), pd.DataFrame(), pd.Series(dtype=float), AS_OF)
    assert out.empty and list(out.columns) == FUNDAMENTAL_FEATURES and out.index.name == "ticker"


def test_market_cap_guards(inputs):
    uni, fa, est, price = inputs
    uni = uni.copy()
    uni.loc["AAA", fields.MARKET_CAP] = 0.0
    uni.loc["BBB", fields.MARKET_CAP] = -1e9
    uni.loc["CCC", fields.MARKET_CAP] = NAN
    out = compute_fundamental_features(uni, fa, est, price, AS_OF)
    for col in ("fcf_yield_pct", "cash_pct_market_cap", "enterprise_value_usd_bn", "ev_to_sales"):
        assert out.loc[["AAA", "BBB", "CCC"], col].isna().all(), col
    # features that do not use market cap are unaffected
    assert out.loc["AAA", "gross_margin_pct"] == pytest.approx(40.0)
    assert out.loc["AAA", "net_debt_usd_bn"] == pytest.approx(2.0)


def test_price_guards(inputs):
    uni, fa, est, _ = inputs
    price = pd.Series({"AAA": 0.0, "BBB": -5.0, "CCC": NAN})
    out = compute_fundamental_features(uni, fa, est, price, AS_OF)
    for col in ("pe_ntm", "earnings_yield_ntm_pct", "target_price_upside_pct"):
        assert out[col].isna().all(), col


def test_revenue_zero_or_negative_blanks_margins(inputs):
    uni, fa, est, price = inputs
    fa = fa.copy()
    fa.loc["AAA", fields.REVENUE_TTM] = 0.0
    fa.loc["BBB", fields.REVENUE_TTM] = -1e9
    out = compute_fundamental_features(uni, fa, est, price, AS_OF)
    cols = ["ev_to_sales", "gross_margin_pct", "operating_margin_pct", "ebitda_margin_pct", "fcf_margin_pct",
            "net_margin_pct", "gross_margin_change_yoy_pp", "operating_margin_change_yoy_pp", "revenue_growth_ntm_est_pct"]
    assert out.loc[["AAA", "BBB"], cols].isna().all().all()
    # current revenue of 0 vs a positive prior year is a -100% decline, not missing
    assert out.loc["AAA", "revenue_growth_yoy_pct"] == pytest.approx(-100.0)


def test_interest_coverage_cap_and_zero_interest(inputs):
    uni, fa, est, price = inputs
    fa = fa.copy()
    fa.loc["AAA", fields.INTEREST_EXPENSE_TTM] = 1e6  # 1e9 / 1e6 = 1000 -> capped
    fa.loc["BBB", fields.INTEREST_EXPENSE_TTM] = 0.0  # no interest but operating loss -> NaN
    fa.loc["CCC", fields.INTEREST_EXPENSE_TTM] = 0.0  # no interest, OI > 0 -> 100
    out = compute_fundamental_features(uni, fa, est, price, AS_OF)
    assert out.loc["AAA", "interest_coverage"] == 100.0
    assert math.isnan(out.loc["BBB", "interest_coverage"])
    assert out.loc["CCC", "interest_coverage"] == 100.0


def test_interest_missing_with_debt_is_nan(inputs):
    uni, fa, est, price = inputs
    fa = fa.copy()
    fa.loc["AAA", fields.INTEREST_EXPENSE_TTM] = NAN  # has debt: cannot assume no interest
    out = compute_fundamental_features(uni, fa, est, price, AS_OF)
    assert math.isnan(out.loc["AAA", "interest_coverage"])


def test_negative_amounts_treated_as_missing(inputs):
    uni, fa, est, price = inputs
    fa, est = fa.copy(), est.copy()
    fa.loc["AAA", fields.CASH] = -1.0
    est.loc["AAA", fields.TARGET_PRICE_MEAN] = -10.0
    est.loc["AAA", fields.NUM_ANALYSTS] = -3
    out = compute_fundamental_features(uni, fa, est, price, AS_OF)
    a = out.loc["AAA"]
    for col in ("cash_pct_market_cap", "net_debt_usd_bn", "enterprise_value_usd_bn", "ev_to_ebitda",
                "net_debt_to_ebitda", "target_price_upside_pct", "num_analysts"):
        assert math.isnan(a[col]), col


def test_eps_revision_uses_absolute_denominator(inputs):
    uni, fa, est, price = inputs
    est = est.copy()
    est.loc["AAA", [fields.EPS_NTM_EST, fields.EPS_NTM_EST_3M_AGO]] = [-3.0, -2.0]
    out = compute_fundamental_features(uni, fa, est, price, AS_OF)
    assert out.loc["AAA", "eps_revision_3m_pct"] == pytest.approx(-50.0)  # cut from -2 to -3


def test_fcf_ttm_preferred_over_cfo_minus_capex(inputs):
    uni, fa, est, price = inputs
    fa = fa.copy()
    fa.loc["AAA", fields.FCF_TTM] = 0.5e9  # vendor FCF differs from CFO - capex (0.8e9)
    out = compute_fundamental_features(uni, fa, est, price, AS_OF)
    assert out.loc["AAA", "fcf_yield_pct"] == pytest.approx(5.0)


def test_days_relative_to_as_of_and_same_day(inputs):
    uni, fa, est, price = inputs
    out = compute_fundamental_features(uni, fa, est, price, pd.Timestamp("2026-11-03"))
    assert out.loc["AAA", "days_to_next_earnings"] == 0.0
    assert out.loc["AAA", "days_since_last_earnings"] == pytest.approx(91.0)  # Aug 4 -> Nov 3
    out = compute_fundamental_features(uni, fa, est, price, "2026-08-04")
    assert out.loc["AAA", "days_since_last_earnings"] == 0.0


def test_unparseable_dates_are_nan(inputs):
    uni, fa, est, price = inputs
    est = est.copy().astype({fields.LAST_EARNINGS_DATE: object, fields.NEXT_EARNINGS_DATE: object})
    est.loc["AAA", fields.LAST_EARNINGS_DATE] = "not a date"
    est.loc["AAA", fields.NEXT_EARNINGS_DATE] = "n/a"
    fa = fa.copy().astype({fields.REPORT_DATE: object})
    fa.loc["AAA", fields.REPORT_DATE] = None
    out = compute_fundamental_features(uni, fa, est, price, AS_OF)
    assert math.isnan(out.loc["AAA", "days_since_last_earnings"])
    assert math.isnan(out.loc["AAA", "days_to_next_earnings"])


def test_ticker_column_input_and_duplicate_rows(inputs):
    uni, fa, est, price = inputs
    dup = pd.concat([fa.loc[["AAA"]].assign(**{fields.REVENUE_TTM_PRIOR_YEAR: 1e9}), fa])  # last row wins
    out = compute_fundamental_features(uni, dup.reset_index(), est.reset_index(), price, AS_OF)
    assert out.loc["AAA", "revenue_growth_yoy_pct"] == pytest.approx(25.0)
    assert out.loc["AAA", "pe_ntm"] == pytest.approx(12.5)


def test_missing_columns_behave_like_nan(inputs):
    uni, fa, est, price = inputs
    fa = fa.drop(columns=[fields.EBITDA_TTM, fields.FCF_TTM])
    est = est.drop(columns=[fields.NEXT_EARNINGS_DATE])
    out = compute_fundamental_features(uni, fa, est, price, AS_OF)
    assert out["ev_to_ebitda"].isna().all()
    assert out.loc["AAA", "fcf_yield_pct"] == pytest.approx(8.0)  # CFO - capex fallback
    assert out["days_to_next_earnings"].isna().all()


def test_string_numbers_and_inf_are_coerced(inputs):
    uni, fa, est, price = inputs
    fa = fa.copy().astype({fields.REVENUE_TTM: object, fields.TOTAL_EQUITY: object})
    fa.loc["AAA", fields.REVENUE_TTM] = "5e9"
    fa.loc["AAA", fields.TOTAL_EQUITY] = np.inf
    out = compute_fundamental_features(uni, fa, est, price, AS_OF)
    assert out.loc["AAA", "gross_margin_pct"] == pytest.approx(40.0)
    assert math.isnan(out.loc["AAA", "roe_pct"])
    assert np.isfinite(out.to_numpy()[~np.isnan(out.to_numpy())]).all()


def test_canonical_demo_thresholds_reachable(inputs):
    """AAA would pass the demo's fundamental conditions (FCF yield > 4%, revenue growth > 8%)."""
    a = compute(inputs).loc["AAA"]
    assert a["fcf_yield_pct"] > 4 and a["revenue_growth_yoy_pct"] > 8


# ---------------------------------------------------------------------------------------------
# Factor characteristics
# ---------------------------------------------------------------------------------------------

FACTOR_CHARACTERISTICS = ["book_to_market", "operating_profitability_pct", "asset_growth_yoy_pct",
                          "gross_profitability_pct", "earnings_yield_ttm_pct"]


def test_factor_characteristics_are_catalogued_last_in_their_own_category():
    from aitrading.screen.catalog import default_catalog

    cat = default_catalog()
    assert FUNDAMENTAL_FEATURES[-5:] == FACTOR_CHARACTERISTICS
    assert {cat[f].category for f in FACTOR_CHARACTERISTICS} == {"factor_characteristics"}
    assert {f: cat[f].higher_is_better for f in FACTOR_CHARACTERISTICS} == {
        "book_to_market": True, "operating_profitability_pct": True, "asset_growth_yoy_pct": False,
        "gross_profitability_pct": True, "earnings_yield_ttm_pct": True,
    }
    assert [cat[f].unit for f in FACTOR_CHARACTERISTICS] == ["x", "%", "%", "%", "%"]


def test_optional_total_assets_columns_absent_give_nan(inputs):
    """Vendor adapters need not supply the optional total-assets columns: the asset features are NaN."""
    uni, fa, est, price = inputs
    out = compute_fundamental_features(uni, fa[fields.FUNDAMENTAL_COLUMNS], est, price, AS_OF)
    assert out[["asset_growth_yoy_pct", "gross_profitability_pct"]].isna().all().all()
    # the characteristics that only need the required columns are unaffected
    assert out.loc["AAA", "book_to_market"] == pytest.approx(0.4)
    assert out.loc["AAA", "operating_profitability_pct"] == pytest.approx(23.75)
    assert out.loc["AAA", "earnings_yield_ttm_pct"] == pytest.approx(8.0)


def test_total_assets_guards(inputs):
    uni, fa, est, price = inputs
    fa = fa.copy()
    fa.loc["AAA", fields.TOTAL_ASSETS] = 0.0  # data error: no asset ratio, no growth
    fa.loc["BBB", fields.TOTAL_ASSETS_PRIOR_YEAR] = -1e9  # non-positive base
    fa.loc["CCC", fields.TOTAL_ASSETS] = np.inf
    out = compute_fundamental_features(uni, fa, est, price, AS_OF)
    assert math.isnan(out.loc["AAA", "gross_profitability_pct"]) and math.isnan(out.loc["AAA", "asset_growth_yoy_pct"])
    assert math.isnan(out.loc["BBB", "asset_growth_yoy_pct"])
    assert out.loc["BBB", "gross_profitability_pct"] == pytest.approx(12.0)  # current assets still fine
    assert math.isnan(out.loc["CCC", "gross_profitability_pct"])


def test_operating_profitability_treats_missing_interest_as_zero(inputs):
    uni, fa, est, price = inputs
    fa = fa.copy()
    fa.loc["AAA", fields.INTEREST_EXPENSE_TTM] = NAN  # has debt, interest not tagged
    out = compute_fundamental_features(uni, fa, est, price, AS_OF)
    assert out.loc["AAA", "operating_profitability_pct"] == pytest.approx(25.0)  # 1 / 4
    assert math.isnan(out.loc["AAA", "interest_coverage"])  # coverage stays conservative


def test_market_cap_guards_factor_characteristics(inputs):
    uni, fa, est, price = inputs
    uni = uni.copy()
    uni.loc["AAA", fields.MARKET_CAP] = 0.0
    uni.loc["BBB", fields.MARKET_CAP] = -1e9
    out = compute_fundamental_features(uni, fa, est, price, AS_OF)
    assert out.loc[["AAA", "BBB"], ["book_to_market", "earnings_yield_ttm_pct"]].isna().all().all()
    assert out.loc["AAA", "gross_profitability_pct"] == pytest.approx(2 / 12 * 100)  # no market cap needed
