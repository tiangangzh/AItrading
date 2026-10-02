"""Tests for the factor library: the Kenneth French loader and factor construction from a universe.

The French fixtures in ``tests/fixtures/french`` reproduce the real file format (see
``build_fixtures.py``); their numbers are illustrative. The construction tests use tiny panels whose
portfolio returns are worked out by hand in the comments.
"""

from __future__ import annotations

import io
import os
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aitrading.backtest.models import FactorConstructionCheck
from aitrading.factors import french
from aitrading.factors.construct import (
    CharacteristicsPanel,
    compare_with_official,
    construct_factors,
    two_by_three_sort,
)
from aitrading.factors.french import (
    DEFAULT_FACTOR_COLUMNS,
    FRENCH_BASE_URL,
    FRENCH_DATASETS,
    FrenchDataUnavailable,
    default_french_cache_dir,
    factor_columns,
    load_french_factors,
    parse_french_csv,
    read_french_zip,
)

FIXTURES = Path(__file__).parent / "fixtures" / "french"
FF3_M = "F-F_Research_Data_Factors_CSV.zip"
FF3_D = "F-F_Research_Data_Factors_daily_CSV.zip"
FF5_M = "F-F_Research_Data_5_Factors_2x3_CSV.zip"
MOM_M = "F-F_Momentum_Factor_CSV.zip"


def fixture_text(name: str) -> str:
    return read_french_zip((FIXTURES / name).read_bytes())


class FixtureFetch:
    """Injected ``fetch``: serves fixture zips by URL basename and records every call."""

    def __init__(self, fail: bool = False, payload: bytes | None = None) -> None:
        self.calls: list[str] = []
        self.fail = fail
        self.payload = payload

    def __call__(self, url: str) -> bytes:
        self.calls.append(url)
        if self.fail:
            raise ConnectionError("network unreachable (test)")
        if self.payload is not None:
            return self.payload
        return (FIXTURES / url.rsplit("/", 1)[-1]).read_bytes()


def ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s)


# =============================================================================================
# parse_french_csv on every fixture
# =============================================================================================


def test_parse_ff3_monthly_fixture():
    df = parse_french_csv(fixture_text(FF3_M))
    assert list(df.columns) == ["Mkt-RF", "SMB", "HML", "RF"]
    assert len(df) == 20  # 2023-01 .. 2024-08; the annual section (2023) is not appended
    assert df.index.name == "date"
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index[0] == ts("2023-01-31") and df.index[-1] == ts("2024-08-31")
    assert df.index.is_month_end.all()
    assert ts("2024-02-29") in df.index  # leap-year month end
    assert df.attrs["frequency"] == "monthly"
    # percent -> fraction
    assert df.loc["2023-01-31", "Mkt-RF"] == pytest.approx(0.0664)
    assert df.loc["2023-01-31", "RF"] == pytest.approx(0.0035)
    assert df.loc["2024-08-31", "SMB"] == pytest.approx(-0.0365)
    # -99.99 -> NaN, the other values in that row survive
    assert np.isnan(df.loc["2023-03-31", "HML"])
    assert df.loc["2023-03-31", "SMB"] == pytest.approx(-0.0694)
    assert df.isna().sum().sum() == 1
    # annual row "2023, 21.69, -3.94, -13.81, 5.00" must be ignored
    assert not np.isclose(df["Mkt-RF"], 0.2169).any()
    assert df.dtypes.eq(float).all()


def test_parse_ff3_daily_fixture():
    df = parse_french_csv(fixture_text(FF3_D))
    assert list(df.columns) == ["Mkt-RF", "SMB", "HML", "RF"]
    assert df.attrs["frequency"] == "daily"
    assert len(df) == 11
    assert list(df.index[:3]) == [ts("2024-08-16"), ts("2024-08-19"), ts("2024-08-20")]
    assert df.index[-1] == ts("2024-08-30")
    assert df.loc["2024-08-19", "Mkt-RF"] == pytest.approx(0.0101)
    assert df.loc["2024-08-16", "RF"] == pytest.approx(0.00022)  # "0.022" percent per day
    assert np.isnan(df.loc["2024-08-29", "Mkt-RF"])  # coded -999
    assert df.loc["2024-08-29", "HML"] == pytest.approx(0.0030)


def test_parse_ff5_monthly_fixture():
    df = parse_french_csv(fixture_text(FF5_M))
    assert list(df.columns) == ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]
    assert len(df) == 20
    assert df.loc["2024-08-31", "RMW"] == pytest.approx(0.0085)
    assert df.loc["2023-01-31", "CMA"] == pytest.approx(-0.0457)
    assert np.isnan(df.loc["2023-04-30", "CMA"])  # -99.99
    assert not np.isclose(df["SMB"], -0.0501).any()  # annual-section value


def test_parse_momentum_monthly_fixture():
    text = fixture_text(MOM_M)
    assert ",Mom   " in text  # the real header is padded
    df = parse_french_csv(text)
    assert list(df.columns) == ["Mom"]  # whitespace stripped
    assert len(df) == 19
    assert df.index[0] == ts("2023-02-28")
    assert df.loc["2023-02-28", "Mom"] == pytest.approx(-0.0006)
    assert np.isnan(df.loc["2023-05-31", "Mom"])  # -99.99
    assert np.isnan(df.loc["2024-08-31", "Mom"])  # -999
    assert not np.isclose(df["Mom"], -0.1433).any()  # annual-section value


def test_parse_handles_lf_bom_trailing_commas_and_minus_999_decimal():
    text = (
        "\ufeffSome preamble, with a comma\n"
        "\n"
        "  ,Mkt-RF , SMB ,\n"
        "199912,  1.00, -999.00,\n"
        "200001,  -2.50,  0.50,\n"
        "\n"
        "Annual Factors: January-December\n"
        ",Mkt-RF,SMB\n"
        "2000, 10.0, 20.0\n"
    )
    df = parse_french_csv(text)
    assert list(df.columns) == ["Mkt-RF", "SMB"]
    assert list(df.index) == [ts("1999-12-31"), ts("2000-01-31")]
    assert df.loc["1999-12-31", "Mkt-RF"] == pytest.approx(0.01)
    assert np.isnan(df.loc["1999-12-31", "SMB"])
    assert df.loc["2000-01-31", "Mkt-RF"] == pytest.approx(-0.025)


def test_parse_rejects_non_french_text_and_malformed_rows():
    with pytest.raises(ValueError, match="not a Kenneth French"):
        parse_french_csv("<html><body>Service unavailable</body></html>")
    with pytest.raises(ValueError, match="expected 2 values"):
        parse_french_csv(",A,B\n202001, 1.0\n")
    with pytest.raises(ValueError, match="non-numeric"):
        parse_french_csv(",A\n202001, abc\n")


def test_read_french_zip_rejects_non_zip():
    with pytest.raises(zipfile.BadZipFile):
        read_french_zip(b"<html>not a zip</html>")


def test_dataset_registry_urls():
    assert set(FRENCH_DATASETS) == {"ff3_monthly", "ff3_daily", "ff5_monthly", "ff5_daily", "mom_monthly", "mom_daily"}
    assert FRENCH_DATASETS["ff3_monthly"].url == FRENCH_BASE_URL + "F-F_Research_Data_Factors_CSV.zip"
    assert FRENCH_DATASETS["mom_daily"].url.endswith("/ftp/F-F_Momentum_Factor_daily_CSV.zip")
    assert FRENCH_DATASETS["ff5_daily"].filename == "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
    assert FRENCH_BASE_URL == "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"


def test_factor_columns_defaults():
    assert DEFAULT_FACTOR_COLUMNS["capm"] == ["Mkt-RF"]
    assert DEFAULT_FACTOR_COLUMNS["carhart4"] == ["Mkt-RF", "SMB", "HML", "Mom"]
    assert DEFAULT_FACTOR_COLUMNS["ff5"] == ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
    for model in ("capm", "ff3", "carhart4", "ff5"):
        cols = factor_columns(model)
        assert len(cols) == len(DEFAULT_FACTOR_COLUMNS[model])
    with pytest.raises(ValueError):
        factor_columns("ff7")


def test_factor_columns_follow_the_regression_contract():
    regression = pytest.importorskip("aitrading.backtest.regression")
    for model, cols in regression.FACTOR_COLUMNS.items():
        assert factor_columns(model) == list(cols)
        assert DEFAULT_FACTOR_COLUMNS[model] == list(cols)


def test_align_factor_dates_matches_months_and_days():
    official = parse_french_csv(fixture_text(FF3_M))
    last_trading_days = pd.DatetimeIndex(["2024-05-31", "2024-06-28", "2024-07-31", "2024-09-30"])
    aligned = french.align_factor_dates(official, last_trading_days)
    assert list(aligned.index) == list(last_trading_days)
    assert aligned.loc["2024-06-28", "Mkt-RF"] == pytest.approx(0.0277)  # June 2024 row
    assert aligned.loc["2024-09-30"].isna().all()  # not published yet
    assert aligned.attrs["alignment"] == "calendar month"
    daily = parse_french_csv(fixture_text(FF3_D))
    d = french.align_factor_dates(daily, pd.DatetimeIndex(["2024-08-19 16:00", "2024-08-20", "2024-08-24"]))
    assert d.iloc[0]["Mkt-RF"] == pytest.approx(0.0101)
    assert d.iloc[1]["SMB"] == pytest.approx(-0.0067)
    assert d.iloc[2].isna().all()  # a Saturday: no daily observation
    assert d.attrs["alignment"] == "date" and d.attrs["warnings"] == []


# Regression (review): align_factor_dates chose month/day matching from the factors' spacing only, so
# monthly factors on daily dates copied the whole month's return onto every trading day (and daily
# factors on monthly dates gave a one-day return per month), silently.


def test_align_factor_dates_refuses_coarser_factors_than_dates():
    monthly = pd.DataFrame({"Mkt-RF": [0.05, 0.02]}, index=pd.to_datetime(["2020-01-31", "2020-02-29"]))
    with pytest.raises(ValueError, match="cannot align monthly factor returns onto daily dates"):
        french.align_factor_dates(monthly, pd.bdate_range("2020-01-01", periods=25))
    with pytest.raises(ValueError, match="onto weekly dates"):
        french.align_factor_dates(monthly, pd.date_range("2020-01-03", periods=8, freq="W-FRI"))
    official = parse_french_csv(fixture_text(FF3_M))
    with pytest.raises(ValueError, match="load_french_factors"):
        french.align_factor_dates(official, pd.bdate_range("2024-05-01", "2024-06-30"))


def test_align_factor_dates_compounds_daily_factors_onto_month_ends():
    days = pd.bdate_range("2024-01-01", "2024-02-29")  # Jan: 23 weekdays, Feb: 21
    assert (days.month == 1).sum() == 23 and (days.month == 2).sum() == 21
    daily = pd.DataFrame({"Mkt-RF": 0.0, "SMB": 0.0, "RF": 0.0001}, index=days)
    daily.loc["2024-01-02", "Mkt-RF"] = 0.01
    daily.loc["2024-01-31", "Mkt-RF"] = 0.02   # last day of the January period
    daily.loc["2024-02-01", "Mkt-RF"] = -0.01  # first day of the February period
    daily.loc[["2024-01-15", "2024-01-16"], "SMB"] = 0.01
    daily.attrs = {"source": "test", "frequency": "daily", "warnings": ["pre-existing"]}
    out = french.align_factor_dates(daily, pd.DatetimeIndex(["2024-01-31", "2024-02-29"]))
    # RF = prod(1 + rf) - 1; Mkt-RF = prod(1 + Mkt-RF + RF) - prod(1 + RF)
    assert out.loc["2024-01-31", "RF"] == pytest.approx(1.0001**23 - 1)
    assert out.loc["2024-01-31", "Mkt-RF"] == pytest.approx(1.0101 * 1.0201 * 1.0001**21 - 1.0001**23)
    assert out.loc["2024-01-31", "Mkt-RF"] == pytest.approx(0.0302665, abs=1e-7)
    # ... not the naive prod(1 + Mkt-RF) - 1 = 1.01 * 1.02 - 1 = 0.0302, nor one day's 0.02
    assert abs(out.loc["2024-01-31", "Mkt-RF"] - 0.0302) > 6e-5
    assert out.loc["2024-01-31", "SMB"] == pytest.approx(1.01**2 - 1)  # 0.0201
    assert out.loc["2024-02-29", "Mkt-RF"] == pytest.approx(0.9901 * 1.0001**20 - 1.0001**21)
    assert out.loc["2024-02-29", "SMB"] == 0.0
    assert out.loc["2024-02-29", "RF"] == pytest.approx(1.0001**21 - 1)
    # the result is in monthly units, not a single day's return or a sum of copies
    assert out.attrs["alignment"] == "compounded daily -> monthly" and out.attrs["frequency"] == "monthly"
    assert out.attrs["warnings"][0] == "pre-existing" and len(out.attrs["warnings"]) == 2
    assert "compounded over each monthly strategy period" in out.attrs["warnings"][1]
    assert daily.attrs["warnings"] == ["pre-existing"]  # the input's attrs are not mutated
    # calendar month-end target dates (2024-03-31 is a Sunday) give the same periods
    days3 = pd.bdate_range("2024-01-01", "2024-03-31")
    d3 = pd.DataFrame({"SMB": 0.001}, index=days3)
    cal = french.align_factor_dates(d3, pd.DatetimeIndex(["2024-01-31", "2024-02-29", "2024-03-31"]))
    assert cal["SMB"].tolist() == pytest.approx([1.001**23 - 1, 1.001**21 - 1, 1.001**21 - 1])
    # unsorted / repeated / time-zone-aware target dates keep their order and get their own period
    odd = pd.DatetimeIndex(["2024-02-29", "2024-01-31", "2024-02-29"]).tz_localize("UTC")
    o = french.align_factor_dates(d3, odd)
    assert list(o.index) == list(odd)
    assert o["SMB"].tolist() == pytest.approx([1.001**21 - 1, 1.001**23 - 1, 1.001**21 - 1])


def test_align_factor_dates_compounds_daily_fixture_onto_weeks_and_flags_partial_periods():
    daily = parse_french_csv(fixture_text(FF3_D))  # 2024-08-16 .. 2024-08-30, Mkt-RF NaN on 08-29
    fridays = pd.DatetimeIndex(["2024-08-23", "2024-08-30", "2024-09-06"])
    w = french.align_factor_dates(daily, fridays)
    # week 1 = (08-16, 08-23]: 08-19 .. 08-23
    smb1 = (1 - 0.0010) * (1 - 0.0067) * (1 + 0.0066) * (1 - 0.0012) * (1 + 0.0197) - 1
    mkt1 = (1.0101 + 0.00022) * (0.9976 + 0.00022) * (1.0048 + 0.00022) * (0.9907 + 0.00022) * (1.0124 + 0.00022)
    assert w.loc["2024-08-23", "SMB"] == pytest.approx(smb1)
    assert w.loc["2024-08-23", "Mkt-RF"] == pytest.approx(mkt1 - 1.00022**5)
    assert w.loc["2024-08-23", "RF"] == pytest.approx(1.00022**5 - 1)
    # week 2 = (08-23, 08-30]: Mkt-RF is NaN on 08-29 -> NaN for the week; HML is complete
    assert np.isnan(w.loc["2024-08-30", "Mkt-RF"])
    assert w.loc["2024-08-30", "HML"] == pytest.approx(1.0077 * 0.9958 * 1.0042 * 1.0030 * 1.0068 - 1)
    assert w.loc["2024-09-06"].isna().all()  # not published
    assert w.attrs["alignment"] == "compounded daily -> weekly"
    assert not any("partly covered" in m for m in w.attrs["warnings"])
    # a month the daily data only covers from 08-16 is NaN, never a half-month return
    m = french.align_factor_dates(daily, pd.DatetimeIndex(["2024-07-31", "2024-08-30"]))
    assert m.isna().all().all()
    assert any("1 period(s) only partly covered" in x for x in m.attrs["warnings"])


# =============================================================================================
# load_french_factors (injected fetch + temporary cache)
# =============================================================================================


def test_load_ff3_monthly_downloads_and_caches(tmp_path):
    fetch = FixtureFetch()
    df = load_french_factors("ff3", cache_dir=tmp_path, fetch=fetch)
    assert fetch.calls == [FRENCH_BASE_URL + FF3_M]
    assert list(df.columns) == factor_columns("ff3") + ["RF"]
    assert df.loc["2023-01-31", "Mkt-RF"] == pytest.approx(0.0664)
    assert (tmp_path / FF3_M).read_bytes() == (FIXTURES / FF3_M).read_bytes()
    assert df.attrs["warnings"] == []
    assert df.attrs["source"] == "Kenneth R. French Data Library"
    assert df.attrs["urls"] == [FRENCH_BASE_URL + FF3_M]
    assert df.attrs["model"] == "ff3" and df.attrs["frequency"] == "monthly"


def test_load_capm_ff5_and_daily(tmp_path):
    fetch = FixtureFetch()
    capm = load_french_factors("capm", cache_dir=tmp_path, fetch=fetch)
    assert list(capm.columns) == ["Mkt-RF", "RF"]
    ff5 = load_french_factors("ff5", cache_dir=tmp_path, fetch=fetch)
    assert list(ff5.columns) == ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]
    assert ff5.loc["2023-01-31", "CMA"] == pytest.approx(-0.0457)
    daily = load_french_factors("ff3", "daily", cache_dir=tmp_path, fetch=fetch)
    assert len(daily) == 11 and daily.index[-1] == ts("2024-08-30")
    # capm reused the cached FF3 file: one download per distinct file
    assert fetch.calls == [FRENCH_BASE_URL + FF3_M, FRENCH_BASE_URL + FF5_M, FRENCH_BASE_URL + FF3_D]


def test_load_carhart4_joins_momentum_on_dates(tmp_path):
    fetch = FixtureFetch()
    df = load_french_factors("carhart4", cache_dir=tmp_path, fetch=fetch)
    assert sorted(c.rsplit("/", 1)[-1] for c in fetch.calls) == sorted([FF3_M, MOM_M])
    assert list(df.columns) == ["Mkt-RF", "SMB", "HML", "Mom", "RF"]
    # momentum starts in 2023-02, so the inner join drops 2023-01
    assert df.index[0] == ts("2023-02-28") and len(df) == 19
    assert df.loc["2023-02-28", "Mom"] == pytest.approx(-0.0006)
    assert df.loc["2023-02-28", "Mkt-RF"] == pytest.approx(-0.0259)
    assert np.isnan(df.loc["2023-05-31", "Mom"])
    assert df.loc["2023-05-31", "HML"] == pytest.approx(-0.0772)


def test_load_maps_regression_column_aliases(tmp_path, monkeypatch):
    monkeypatch.setattr(french, "factor_columns", lambda model: ["MKT_RF", "SMB", "HML", "UMD"])
    df = load_french_factors("carhart4", cache_dir=tmp_path, fetch=FixtureFetch())
    assert list(df.columns) == ["MKT_RF", "SMB", "HML", "UMD", "RF"]
    assert df.loc["2023-02-28", "UMD"] == pytest.approx(-0.0006)


def test_cache_hit_does_not_refetch(tmp_path):
    load_french_factors("ff3", cache_dir=tmp_path, fetch=FixtureFetch())
    offline = FixtureFetch(fail=True)
    df = load_french_factors("ff3", cache_dir=tmp_path, fetch=offline)
    assert offline.calls == []  # fresh cache: no network at all
    assert df.attrs["warnings"] == []
    assert len(df) == 20


def test_stale_cache_is_refreshed_when_online(tmp_path):
    load_french_factors("ff3", cache_dir=tmp_path, fetch=FixtureFetch())
    old = time.time() - 30 * 86400
    os.utime(tmp_path / FF3_M, (old, old))
    fetch = FixtureFetch()
    df = load_french_factors("ff3", cache_dir=tmp_path, fetch=fetch, max_age_days=7)
    assert fetch.calls == [FRENCH_BASE_URL + FF3_M]
    assert df.attrs["warnings"] == []
    assert time.time() - (tmp_path / FF3_M).stat().st_mtime < 3600  # rewritten


def test_stale_cache_fallback_when_offline(tmp_path):
    load_french_factors("ff3", cache_dir=tmp_path, fetch=FixtureFetch())
    old = time.time() - 30 * 86400
    os.utime(tmp_path / FF3_M, (old, old))
    offline = FixtureFetch(fail=True)
    df = load_french_factors("ff3", cache_dir=tmp_path, fetch=offline)
    assert offline.calls == [FRENCH_BASE_URL + FF3_M]
    assert len(df) == 20
    (warning,) = df.attrs["warnings"]
    assert "unreachable" in warning and "cached copy" in warning and "30 days old" in warning


def test_unavailable_without_cache_raises(tmp_path):
    with pytest.raises(FrenchDataUnavailable, match="internet") as exc_info:
        load_french_factors("ff3", cache_dir=tmp_path, fetch=FixtureFetch(fail=True))
    assert FF3_M in str(exc_info.value)
    assert not (tmp_path / FF3_M).exists()


def test_html_error_page_is_a_failed_download(tmp_path):
    html = FixtureFetch(payload=b"<!DOCTYPE html><html>Access denied</html>")
    with pytest.raises(FrenchDataUnavailable, match="not a French data-library zip"):
        load_french_factors("ff3", cache_dir=tmp_path, fetch=html)
    assert not (tmp_path / FF3_M).exists()  # garbage is never cached
    # with a stale cache the garbage response falls back to it
    load_french_factors("ff3", cache_dir=tmp_path, fetch=FixtureFetch())
    old = time.time() - 30 * 86400
    os.utime(tmp_path / FF3_M, (old, old))
    df = load_french_factors("ff3", cache_dir=tmp_path, fetch=html)
    assert len(df) == 20 and len(df.attrs["warnings"]) == 1


def test_corrupt_cache_is_redownloaded(tmp_path):
    (tmp_path / FF3_M).write_bytes(b"truncated")
    fetch = FixtureFetch()
    df = load_french_factors("ff3", cache_dir=tmp_path, fetch=fetch)
    assert fetch.calls == [FRENCH_BASE_URL + FF3_M]
    assert len(df) == 20
    assert any("unreadable" in w for w in df.attrs["warnings"])
    assert (tmp_path / FF3_M).read_bytes() == (FIXTURES / FF3_M).read_bytes()


def test_default_cache_dir_uses_env(tmp_path, monkeypatch):
    monkeypatch.setenv("AITRADING_CACHE_DIR", str(tmp_path))
    assert default_french_cache_dir() == tmp_path / "french"
    load_french_factors("ff3", fetch=FixtureFetch())
    assert (tmp_path / "french" / FF3_M).is_file()
    monkeypatch.delenv("AITRADING_CACHE_DIR")
    assert default_french_cache_dir() == Path.home() / ".aitrading" / "cache" / "french"


def test_default_fetch_uses_httpx_with_user_agent(tmp_path, monkeypatch):
    import httpx

    seen: dict = {}

    def fake_get(url, **kwargs):
        seen["url"] = url
        seen.update(kwargs)
        return httpx.Response(200, content=(FIXTURES / FF3_M).read_bytes(), request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx, "get", fake_get)
    df = load_french_factors("capm", cache_dir=tmp_path)
    assert seen["url"] == FRENCH_BASE_URL + FF3_M
    assert seen["follow_redirects"] is True
    assert seen["timeout"] == 60.0
    assert "aitrading" in seen["headers"]["User-Agent"]
    assert list(df.columns) == ["Mkt-RF", "RF"]


def test_default_fetch_http_error_becomes_unavailable(tmp_path, monkeypatch):
    import httpx

    def fake_get(url, **kwargs):
        return httpx.Response(503, content=b"down", request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx, "get", fake_get)
    with pytest.raises(FrenchDataUnavailable, match="HTTPStatusError"):
        load_french_factors("ff3", cache_dir=tmp_path)


def test_load_rejects_bad_arguments(tmp_path):
    with pytest.raises(ValueError):
        load_french_factors("ff4", cache_dir=tmp_path, fetch=FixtureFetch())
    with pytest.raises(ValueError):
        load_french_factors("ff3", "weekly", cache_dir=tmp_path, fetch=FixtureFetch())


# =============================================================================================
# construct_factors: hand-built panels
# =============================================================================================
#
# Twelve names: s1..s6 are small, b1..b6 big. Caps alternate 10/30 (small) and 100/300 (big), so
# every 2-name portfolio has a value-weighted return that differs from its equal-weighted one.
# B/M: {s1, s2, b1, b2} low, {s3, s4, b3, b4} middle, {s5, s6, b5, b6} high, with 12 distinct values
# so the 30th/70th percentiles (linear interpolation over all 12 names) fall cleanly between groups:
#   sorted B/M = .15 .20 .22 .25 | .45 .50 .55 .60 | .90 1.0 1.1 1.2 -> p30 = .31, p70 = .81.
# Size median of {10,30,10,30,10,30,100,300,100,300,100,300} = (30 + 100) / 2 = 65.

T = ["s1", "s2", "s3", "s4", "s5", "s6", "b1", "b2", "b3", "b4", "b5", "b6"]
CAP0 = dict(zip(T, [10, 30, 10, 30, 10, 30, 100, 300, 100, 300, 100, 300]))
CAP1 = dict(zip(T, [30, 10, 30, 10, 30, 10, 300, 100, 300, 100, 300, 100]))  # pairs swapped
BM = dict(zip(T, [0.20, 0.25, 0.50, 0.55, 1.00, 1.20, 0.15, 0.22, 0.45, 0.60, 0.90, 1.10]))
RET1 = dict(zip(T, [0.04, 0.08, 0.02, -0.02, 0.10, 0.06, 0.01, 0.03, 0.00, 0.04, 0.05, 0.01]))
RET2 = dict(zip(T, [0.10, 0.00, 0.00, 0.04, -0.02, 0.02, 0.02, 0.06, 0.01, -0.03, 0.03, 0.07]))
DATES3 = pd.to_datetime(["2020-01-31", "2020-02-29", "2020-03-31"])


def frame(rows: list[dict], index) -> pd.DataFrame:
    return pd.DataFrame(rows, index=pd.DatetimeIndex(index))[list(rows[0])]


def three_period_panel(**overrides) -> CharacteristicsPanel:
    caps = [CAP0, CAP1, CAP0]  # the t2 cap differs from the prior (t1) cap used as the t2 weight
    kw = dict(
        returns=frame([{t: 0.0 for t in T}, RET1, RET2], DATES3),
        market_cap=frame(caps, DATES3),
        book_equity=frame([{t: BM[t] * c[t] for t in T} for c in caps], DATES3),
        rf=pd.Series([0.001, 0.002, 0.003], index=DATES3),
    )
    kw.update(overrides)
    return CharacteristicsPanel(**kw)


def vw(pairs: list[tuple[float, float]]) -> float:
    """Value-weighted return of [(weight, return), ...]."""
    return sum(w * r for w, r in pairs) / sum(w for w, _ in pairs)


def test_ff3_monthly_formation_matches_hand_calculation():
    f, warnings = construct_factors(three_period_panel(), "ff3", formation="monthly", min_names_per_portfolio=2)
    assert list(f.columns) == ["Mkt-RF", "SMB", "HML", "RF"]
    assert f.iloc[0][["Mkt-RF", "SMB", "HML"]].isna().all()  # no prior cap / no formation yet

    # t1 (formed at t0, weights = t0 caps):
    #   S/L = (10*.04 + 30*.08)/40 = .07   S/M = (10*.02 - 30*.02)/40 = -.01  S/H = (10*.10 + 30*.06)/40 = .07
    #   B/L = (100*.01 + 300*.03)/400 = .025  B/M = (0 + 300*.04)/400 = .03  B/H = (100*.05 + 300*.01)/400 = .02
    sl, sm, sh = 0.07, -0.01, 0.07
    bl, bmid, bh = 0.025, 0.03, 0.02
    assert vw([(10, 0.04), (30, 0.08)]) == pytest.approx(sl)
    assert f.loc["2020-02-29", "SMB"] == pytest.approx((sl + sm + sh) / 3 - (bl + bmid + bh) / 3)  # 0.0183333
    assert f.loc["2020-02-29", "HML"] == pytest.approx(0.5 * (sh + bh) - 0.5 * (sl + bl))  # -0.0025
    assert f.loc["2020-02-29", "SMB"] == pytest.approx(0.055 / 3)
    assert f.loc["2020-02-29", "HML"] == pytest.approx(-0.0025)

    # t2 (formed at t1 with t1 caps 30/10 - pairs swapped - which are also the t2 weights):
    #   S/L = (30*.10 + 10*0)/40 = .075  S/M = (30*0 + 10*.04)/40 = .01  S/H = (-30*.02 + 10*.02)/40 = -.01
    #   B/L = (300*.02 + 100*.06)/400 = .03  B/M = (300*.01 - 100*.03)/400 = 0  B/H = (300*.03 + 100*.07)/400 = .04
    assert f.loc["2020-03-31", "SMB"] == pytest.approx((0.075 + 0.01 - 0.01) / 3 - (0.03 + 0.0 + 0.04) / 3)
    assert f.loc["2020-03-31", "HML"] == pytest.approx(0.5 * (-0.01 + 0.04) - 0.5 * (0.075 + 0.03))  # -0.0375
    assert f["RF"].tolist() == pytest.approx([0.001, 0.002, 0.003])
    assert any("no exchange data" in w for w in warnings)
    assert f.attrs["source"] == "constructed from universe"


def test_mkt_rf_is_value_weighted_with_prior_period_caps():
    f, _ = construct_factors(three_period_panel(), "capm", min_names_per_portfolio=2)
    assert list(f.columns) == ["Mkt-RF", "RF"]
    # t1: weights = t0 caps (total 1320); sum(cap * ret) = 5.2 (small) + 30 (big) = 35.2
    assert f.loc["2020-02-29", "Mkt-RF"] == pytest.approx(35.2 / 1320 - 0.002)
    # t2: weights = t1 caps (total 1320); sum(cap * ret) = 3.0 + 28 = 31
    assert f.loc["2020-03-31", "Mkt-RF"] == pytest.approx(31 / 1320 - 0.003)
    # neither equal weighting nor current-period caps give this
    assert f.loc["2020-03-31", "Mkt-RF"] != pytest.approx(np.mean(list(RET2.values())) - 0.003)
    assert f.loc["2020-03-31", "Mkt-RF"] != pytest.approx(
        sum(CAP0[t] * RET2[t] for t in T) / sum(CAP0.values()) - 0.003
    )


def test_without_rf_market_is_raw_and_warned():
    f, warnings = construct_factors(three_period_panel(rf=None), "capm", min_names_per_portfolio=2)
    assert list(f.columns) == ["Mkt-RF"]
    assert f.loc["2020-02-29", "Mkt-RF"] == pytest.approx(35.2 / 1320)
    assert any("no risk-free rate" in w for w in warnings)


def test_rf_aligned_by_month_when_dates_differ():
    idx = pd.to_datetime(["2020-04-30", "2020-05-29", "2020-06-30"])  # 05-29 = last trading day
    rf = pd.Series([0.001, 0.002, 0.003], index=pd.to_datetime(["2020-04-30", "2020-05-31", "2020-06-30"]))
    p = CharacteristicsPanel(
        returns=frame([{t: 0.0 for t in T}, RET1, RET2], idx),
        market_cap=frame([CAP0, CAP0, CAP0], idx),
        book_equity=None,
        rf=rf,
    )
    f, _ = construct_factors(p, "capm", min_names_per_portfolio=1)
    assert f["RF"].tolist() == pytest.approx([0.001, 0.002, 0.003])
    assert f.loc["2020-05-29", "Mkt-RF"] == pytest.approx(35.2 / 1320 - 0.002)


# Regression (review): a daily RF passed with monthly returns was matched by exact date, so each
# month's "RF" was one day's rate (~0.0002) and Mkt-RF was overstated by ~20x RF, silently.


def _capm_panel(index, rf) -> CharacteristicsPanel:
    n = len(index)
    rets = [{t: 0.0 for t in T}, RET1, RET2] + [RET1] * max(0, n - 3)
    return CharacteristicsPanel(
        returns=frame(rets[:n], index), market_cap=frame([CAP0] * n, index), book_equity=None, rf=rf
    )


def test_daily_rf_is_compounded_onto_monthly_returns():
    days = pd.bdate_range("2020-01-01", "2020-03-31")
    assert [(days.month == m).sum() for m in (1, 2, 3)] == [23, 20, 22]
    rf = pd.Series(np.select([days.month == 1, days.month == 2], [0.0001, 0.0002], 0.0003), index=days)
    idx = pd.to_datetime(["2020-01-31", "2020-02-28", "2020-03-31"])  # last trading days
    f, warnings = construct_factors(_capm_panel(idx, rf), "capm", min_names_per_portfolio=1)
    expected_rf = [1.0001**23 - 1, 1.0002**20 - 1, 1.0003**22 - 1]  # ~0.0023, 0.0040, 0.0066
    assert f["RF"].tolist() == pytest.approx(expected_rf)
    assert f.loc["2020-02-28", "Mkt-RF"] == pytest.approx(35.2 / 1320 - expected_rf[1])
    # t2 weights = t1 caps = CAP0: sum(cap * ret) = 2.6 (small) + 36 (big) = 38.6
    assert f.loc["2020-03-31", "Mkt-RF"] == pytest.approx(38.6 / 1320 - expected_rf[2])
    assert any("panel.rf is daily but panel.returns is monthly: RF was compounded" in w for w in warnings)
    # the reviewer's probe: a constant daily rate on calendar-month data is ~20x larger once compounded
    flat = pd.Series(0.0002, index=pd.bdate_range("2020-01-01", "2020-12-31"))
    month_ends = pd.date_range("2020-01-31", periods=3, freq="ME")  # 2020-02-29 is a Saturday
    g, _ = construct_factors(_capm_panel(month_ends, flat), "capm", min_names_per_portfolio=1)
    assert g["RF"].tolist() == pytest.approx([1.0002**23 - 1, 1.0002**20 - 1, 1.0002**22 - 1])


def test_partly_covered_rf_periods_are_nan_and_warned():
    days = pd.bdate_range("2020-01-15", "2020-03-13")  # starts and ends mid-month
    idx = pd.to_datetime(["2020-01-31", "2020-02-28", "2020-03-31"])
    f, warnings = construct_factors(_capm_panel(idx, pd.Series(0.0001, index=days)), "capm", min_names_per_portfolio=1)
    assert np.isnan(f.loc["2020-01-31", "RF"]) and np.isnan(f.loc["2020-03-31", "RF"])
    assert f.loc["2020-02-28", "RF"] == pytest.approx(1.0001**20 - 1)
    assert np.isnan(f.loc["2020-03-31", "Mkt-RF"])
    assert any("2 period(s) only partly covered by RF were set to NaN" in w for w in warnings)
    assert any("risk-free rate missing in 1 period(s)" in w for w in warnings)


def test_daily_rf_is_compounded_onto_weekly_returns():
    days = pd.bdate_range("2020-01-01", "2020-01-17")
    rf = pd.Series(0.0001 * np.arange(1, len(days) + 1), index=days)  # 1bp, 2bp, ... per day
    fridays = pd.to_datetime(["2020-01-03", "2020-01-10", "2020-01-17"])
    f, warnings = construct_factors(_capm_panel(fridays, rf), "capm", min_names_per_portfolio=1)
    # week (01-03, 01-10] = Jan 6..10 = days 4..8; week (01-10, 01-17] = days 9..13
    assert f.loc["2020-01-10", "RF"] == pytest.approx(np.prod(1 + 0.0001 * np.arange(4, 9)) - 1)
    assert f.loc["2020-01-17", "RF"] == pytest.approx(np.prod(1 + 0.0001 * np.arange(9, 14)) - 1)
    assert f.loc["2020-01-10", "Mkt-RF"] == pytest.approx(35.2 / 1320 - (np.prod(1 + 0.0001 * np.arange(4, 9)) - 1))
    # the first week (12-27, 01-03] is missing Dec 30-31 in the RF data -> NaN, not a 3-day rate
    assert np.isnan(f.loc["2020-01-03", "RF"])
    assert any("panel.rf is daily but panel.returns is weekly" in w for w in warnings)


def test_coarser_rf_than_returns_is_refused():
    days = pd.bdate_range("2020-01-01", periods=5)
    monthly_rf = pd.Series([0.004, 0.004], index=pd.to_datetime(["2019-12-31", "2020-01-31"]))
    with pytest.raises(ValueError, match="panel.rf is monthly but panel.returns is daily"):
        construct_factors(_capm_panel(days, monthly_rf), "capm", min_names_per_portfolio=1)
    # same frequency: unchanged exact-date behaviour, no conversion warning
    f, warnings = construct_factors(_capm_panel(days, pd.Series(0.0002, index=days)), "capm", min_names_per_portfolio=1)
    assert f["RF"].tolist() == pytest.approx([0.0002] * 5)
    assert not any("panel.rf is" in w for w in warnings)


def test_thin_portfolios_give_nan_factors_and_a_warning():
    f, warnings = construct_factors(three_period_panel(), "ff3", formation="monthly")  # default min 5 names
    assert f[["SMB", "HML"]].isna().all().all()  # every portfolio holds only 2 names
    assert f["Mkt-RF"].notna().sum() == 2  # the market (12 names) is unaffected
    (thin,) = [w for w in warnings if "fewer than 5 names" in w]
    assert "size-B/M" in thin and "HML, SMB" in thin and "2 period(s)" in thin


def test_one_thin_portfolio_only_kills_the_factors_that_use_it():
    # s3 has no return at t1, so S/M holds only s4 there (1 name < 2). S/M enters SMB but not HML.
    rets = frame([{t: 0.0 for t in T}, {**RET1, "s3": np.nan}, RET2], DATES3)
    f, warnings = construct_factors(three_period_panel(returns=rets), "ff3", formation="monthly", min_names_per_portfolio=2)
    assert np.isnan(f.loc["2020-02-29", "SMB"])
    assert f.loc["2020-02-29", "HML"] == pytest.approx(-0.0025)
    assert f.loc["2020-03-31", "SMB"] == pytest.approx(0.005 / 3)
    (thin,) = [w for w in warnings if "fewer than 2 names" in w]
    assert "S/M: 1" in thin and "SMB set to NaN" in thin


def test_non_positive_book_equity_is_excluded_from_sorts():
    # x: a huge name with negative book equity and a +50% month; y: zero book equity, -40% month.
    # Included, x would land in B/L (lowest B/M) and dominate it; excluded, SMB and HML equal the
    # 12-name values, while the market still contains both names.
    extra_caps = {"x": 1000.0, "y": 50.0}
    caps = [{**c, **extra_caps} for c in (CAP0, CAP1, CAP0)]
    be = [{**{t: BM[t] * c[t] for t in T}, "x": -500.0, "y": 0.0} for c in (CAP0, CAP1, CAP0)]
    rets = [{t: 0.0 for t in T + ["x", "y"]}, {**RET1, "x": 0.5, "y": -0.4}, {**RET2, "x": 0.0, "y": 0.0}]
    p = CharacteristicsPanel(frame(rets, DATES3), frame(caps, DATES3), frame(be, DATES3), rf=None)
    f, warnings = construct_factors(p, "ff3", formation="monthly", min_names_per_portfolio=2)
    assert f.loc["2020-02-29", "SMB"] == pytest.approx(0.055 / 3)
    assert f.loc["2020-02-29", "HML"] == pytest.approx(-0.0025)
    assert f.loc["2020-03-31", "HML"] == pytest.approx(-0.0375)
    assert f.loc["2020-02-29", "Mkt-RF"] == pytest.approx((35.2 + 1000 * 0.5 + 50 * -0.4) / (1320 + 1050))
    # x and y at the two formations that are held (t0, t1)
    assert any("non-positive book equity" in w and w.startswith("4 ") for w in warnings)


def test_annual_june_formation_uses_december_cap_and_holds_july_to_june():
    # Monthly 2019-12 .. 2021-08. Caps are constant except December 2019, when s1 was worth 50 and s5
    # only 2. Book equity is constant: s1 = 10, s5 = 2 (others BM * cap).
    #  * June 2020 formation: B/M uses the Dec-2019 cap -> s1 = 10/50 = .2 (Low), s5 = 2/2 = 1 (High).
    #    (With the June cap, s1 = 1.0 would be High and s5 = .2 Low.)
    #  * June 2021 formation: Dec-2020 cap = 10 for both -> s1 High, s5 Low.
    # Breakpoints are unchanged (same multiset {.2 x4, .5 x4, 1 x4} -> p30 = .29, p70 = .85).
    idx = pd.date_range("2019-12-31", "2021-08-31", freq="ME")
    bm = {"s1": 0.2, "s2": 0.2, "s3": 0.5, "s4": 0.5, "s5": 1.0, "s6": 1.0,
          "b1": 0.2, "b2": 0.2, "b3": 0.5, "b4": 0.5, "b5": 1.0, "b6": 1.0}
    dec19 = {**CAP0, "s1": 50, "s5": 2}
    caps = [dec19] + [CAP0] * (len(idx) - 1)
    be_row = {t: bm[t] * dec19[t] for t in T}  # s1 = 10, s5 = 2
    p = CharacteristicsPanel(
        returns=frame([RET1] * len(idx), idx),  # every name earns the same return each month
        market_cap=frame(caps, idx),
        book_equity=frame([be_row] * len(idx), idx),
    )
    f, warnings = construct_factors(p, "ff3", formation="annual_june", min_names_per_portfolio=2)

    assert f.loc[:"2020-06-30", ["SMB", "HML"]].isna().all().all()  # nothing formed before June 2020
    # July 2020 - June 2021: S/L = {s1, s2} = .07, S/H = {s5, s6} = .07; big side as before
    year1 = f.loc["2020-07-31":"2021-06-30", "HML"]
    assert len(year1) == 12
    assert year1.to_numpy() == pytest.approx([0.5 * (0.07 + 0.02) - 0.5 * (0.07 + 0.025)] * 12)  # -0.0025
    # July 2021 on: S/L = {s5 (10), s2 (30)} = (1.0 + 2.4)/40 = .085, S/H = {s1, s6} = (.4 + 1.8)/40 = .055
    year2 = f.loc["2021-07-31":, "HML"]
    assert year2.to_numpy() == pytest.approx([0.5 * (0.055 + 0.02) - 0.5 * (0.085 + 0.025)] * 2)  # -0.0175
    assert f.loc["2020-07-31":, "SMB"].to_numpy() == pytest.approx([0.055 / 3] * 14)
    assert not any("December" in w for w in warnings)

    # Monthly formation uses the current cap, so s1/s5 flip immediately.
    fm, _ = construct_factors(p, "ff3", formation="monthly", min_names_per_portfolio=2)
    assert fm.loc["2020-07-31", "HML"] == pytest.approx(-0.0175)


def test_annual_june_without_december_falls_back_to_june_cap_with_warning():
    idx = pd.date_range("2020-01-31", "2020-08-31", freq="ME")
    p = CharacteristicsPanel(
        returns=frame([RET1] * len(idx), idx),
        market_cap=frame([CAP0] * len(idx), idx),
        book_equity=frame([{t: BM[t] * CAP0[t] for t in T}] * len(idx), idx),
    )
    f, warnings = construct_factors(p, "ff3", min_names_per_portfolio=2)
    assert f.loc["2020-07-31", "HML"] == pytest.approx(-0.0025)
    assert any("no December market cap" in w for w in warnings)


def test_annual_june_without_any_june_warns_and_leaves_value_nan():
    f, warnings = construct_factors(three_period_panel(), "ff3", min_names_per_portfolio=2)
    assert f[["SMB", "HML"]].isna().all().all()
    assert f["Mkt-RF"].notna().sum() == 2
    assert any("no end-of-June date" in w for w in warnings)


def test_sparse_book_equity_is_joined_as_of_each_date():
    caps = [CAP0, CAP0, CAP0]
    dense = CharacteristicsPanel(
        frame([{t: 0.0 for t in T}, RET1, RET2], DATES3), frame(caps, DATES3),
        frame([{t: BM[t] * CAP0[t] for t in T}] * 3, DATES3),
    )
    sparse = CharacteristicsPanel(
        dense.returns, dense.market_cap,
        frame([{t: BM[t] * CAP0[t] for t in T}], [pd.Timestamp("2019-12-31")]),  # one stale-dated row
    )
    a, _ = construct_factors(dense, "ff3", formation="monthly", min_names_per_portfolio=2)
    b, _ = construct_factors(sparse, "ff3", formation="monthly", min_names_per_portfolio=2)
    pd.testing.assert_frame_equal(a, b)
    # a row dated *after* a formation date is not visible at that date
    late = CharacteristicsPanel(
        dense.returns, dense.market_cap,
        frame([{t: BM[t] * CAP0[t] for t in T}], [pd.Timestamp("2020-02-15")]),
    )
    c, _ = construct_factors(late, "ff3", formation="monthly", min_names_per_portfolio=2)
    assert np.isnan(c.loc["2020-02-29", "HML"])  # formation at 01-31 had no book equity yet
    assert c.loc["2020-03-31", "HML"] == pytest.approx(b.loc["2020-03-31", "HML"])


# --- five factors -------------------------------------------------------------------------------
# OP:  W (low) = {s5, b3, b4, b6}  N = {s1, s2, s6, b5}  R (high) = {s3, s4, b1, b2}
# INV: C (low) = {s1, s3, s4, b6}  N = {s5, s6, b1, b5}  A (high) = {s2, b2, b3, b4}
# Uneven small/big splits make the three SMB versions differ.
OP = dict(s5=0.01, b3=0.02, b4=0.03, b6=0.04, s1=0.10, s2=0.11, s6=0.12, b5=0.13, s3=0.30, s4=0.31, b1=0.32, b2=0.33)
INV = dict(s1=-0.10, s3=-0.08, s4=-0.06, b6=-0.04, s5=0.02, s6=0.03, b1=0.04, b5=0.05, s2=0.20, b2=0.22, b3=0.24, b4=0.26)
MOM = dict(s2=-0.30, s3=-0.25, b1=-0.20, b4=-0.15, s1=0.00, s5=0.02, b5=0.04, b6=0.06, s4=0.30, s6=0.35, b2=0.40, b3=0.45)
DATES2 = pd.to_datetime(["2020-01-31", "2020-02-29"])


def two_period_panel() -> CharacteristicsPanel:
    def const(d):
        return frame([{t: d[t] for t in T}] * 2, DATES2)

    return CharacteristicsPanel(
        returns=frame([{t: 0.0 for t in T}, RET1], DATES2),
        market_cap=const(CAP0),
        book_equity=const({t: BM[t] * CAP0[t] for t in T}),
        operating_profitability=const(OP),
        investment=const(INV),
        momentum_12_1=const(MOM),
    )


def test_ff5_matches_hand_calculation():
    f, _ = construct_factors(two_period_panel(), "ff5", formation="monthly", min_names_per_portfolio=1)
    assert list(f.columns) == ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
    r = RET1
    c = CAP0
    # size-B/M sort (as in the ff3 test)
    smb_bm = (0.07 - 0.01 + 0.07) / 3 - (0.025 + 0.03 + 0.02) / 3
    # size-OP sort
    s_w = r["s5"]                                                        # .10
    s_n = vw([(c["s1"], r["s1"]), (c["s2"], r["s2"]), (c["s6"], r["s6"])])  # 4.6/70
    s_r = vw([(c["s3"], r["s3"]), (c["s4"], r["s4"])])                   # -.01
    b_w = vw([(c["b3"], r["b3"]), (c["b4"], r["b4"]), (c["b6"], r["b6"])])  # 15/700
    b_n = r["b5"]                                                        # .05
    b_r = vw([(c["b1"], r["b1"]), (c["b2"], r["b2"])])                   # .025
    assert s_n == pytest.approx(4.6 / 70) and b_w == pytest.approx(15 / 700)
    smb_op = (s_w + s_n + s_r) / 3 - (b_w + b_n + b_r) / 3
    rmw = 0.5 * (s_r + b_r) - 0.5 * (s_w + b_w)
    # size-INV sort
    s_c = vw([(c["s1"], r["s1"]), (c["s3"], r["s3"]), (c["s4"], r["s4"])])  # 0
    s_ni = vw([(c["s5"], r["s5"]), (c["s6"], r["s6"])])                  # .07
    s_a = r["s2"]                                                        # .08
    b_c = r["b6"]                                                        # .01
    b_ni = vw([(c["b1"], r["b1"]), (c["b5"], r["b5"])])                  # .03
    b_a = vw([(c["b2"], r["b2"]), (c["b3"], r["b3"]), (c["b4"], r["b4"])])  # .03
    smb_inv = (s_c + s_ni + s_a) / 3 - (b_c + b_ni + b_a) / 3            # .08/3
    cma = 0.5 * (s_c + b_c) - 0.5 * (s_a + b_a)                          # -.05

    row = f.loc["2020-02-29"]
    assert row["SMB"] == pytest.approx((smb_bm + smb_op + smb_inv) / 3)
    assert row["SMB"] == pytest.approx(0.0215873016)
    assert row["HML"] == pytest.approx(-0.0025)
    assert row["RMW"] == pytest.approx(rmw) and rmw == pytest.approx(-0.0532142857)
    assert row["CMA"] == pytest.approx(cma) and cma == pytest.approx(-0.05)
    assert len({round(smb_bm, 9), round(smb_op, 9), round(smb_inv, 9)}) == 3  # the averaging matters


def test_carhart4_momentum_matches_hand_calculation_and_is_monthly():
    f, _ = construct_factors(two_period_panel(), "carhart4", formation="monthly", min_names_per_portfolio=1)
    assert list(f.columns) == ["Mkt-RF", "SMB", "HML", "Mom"]
    # Up = {s4, s6 | b2, b3}, Down = {s2, s3 | b1, b4}
    s_up = vw([(30, -0.02), (30, 0.06)])     # .02
    b_up = vw([(300, 0.03), (100, 0.00)])    # .0225
    s_dn = vw([(30, 0.08), (10, 0.02)])      # .065
    b_dn = vw([(100, 0.01), (300, 0.04)])    # .0325
    assert f.loc["2020-02-29", "Mom"] == pytest.approx(0.5 * (s_up + b_up) - 0.5 * (s_dn + b_dn))
    assert f.loc["2020-02-29", "Mom"] == pytest.approx(-0.0275)
    assert f.loc["2020-02-29", "SMB"] == pytest.approx(0.055 / 3)
    # with annual_june formation (and no June in the sample) momentum is still re-formed monthly
    fa, warnings = construct_factors(two_period_panel(), "carhart4", min_names_per_portfolio=1)
    assert fa.loc["2020-02-29", "Mom"] == pytest.approx(-0.0275)
    assert np.isnan(fa.loc["2020-02-29", "HML"])


def test_construct_maps_factor_names_from_the_regression_contract(monkeypatch):
    import aitrading.factors.construct as construct_mod

    monkeypatch.setattr(construct_mod, "factor_columns", lambda model: ["MKT_RF", "UMD", "HML", "SMB"])
    f, _ = construct_factors(two_period_panel(), "carhart4", formation="monthly", min_names_per_portfolio=1)
    assert list(f.columns) == ["MKT_RF", "UMD", "HML", "SMB"]  # mapped by name, not by position
    assert f.loc["2020-02-29", "UMD"] == pytest.approx(-0.0275)
    assert f.loc["2020-02-29", "SMB"] == pytest.approx(0.055 / 3)


def test_construct_validates_inputs():
    p = two_period_panel()
    with pytest.raises(ValueError, match="book_equity"):
        construct_factors(CharacteristicsPanel(p.returns, p.market_cap, None), "ff3")
    with pytest.raises(ValueError, match="operating_profitability"):
        construct_factors(CharacteristicsPanel(p.returns, p.market_cap, p.book_equity), "ff5")
    with pytest.raises(ValueError, match="momentum_12_1"):
        construct_factors(CharacteristicsPanel(p.returns, p.market_cap, p.book_equity), "carhart4")
    with pytest.raises(ValueError):
        construct_factors(p, "ff4")
    with pytest.raises(ValueError):
        construct_factors(p, "ff3", formation="weekly")
    with pytest.raises(ValueError):
        construct_factors(p, "ff3", min_names_per_portfolio=0)
    # capm needs no characteristics at all
    f, _ = construct_factors(CharacteristicsPanel(p.returns, p.market_cap, None), "capm", min_names_per_portfolio=1)
    assert list(f.columns) == ["Mkt-RF"]


# --- NYSE breakpoints ---------------------------------------------------------------------------


def nyse_universe(n_nyse: int = 20):
    """20 NYSE names (cap i, characteristic i for i = 1..20) + 10 tiny NASDAQ names (cap .5, char 100)."""
    names = [f"N{i}" for i in range(1, 21)] + [f"Q{i}" for i in range(1, 11)]
    size = pd.Series([float(i) for i in range(1, 21)] + [0.5] * 10, index=names)
    char = pd.Series([float(i) for i in range(1, 21)] + [100.0] * 10, index=names)
    exch = pd.Series(["NYSE"] * 20 + ["NASDAQ"] * 10, index=names)
    exch.iloc[n_nyse:20] = "NASDAQ"
    return size, char, exch


def test_nyse_breakpoints_when_enough_nyse_names():
    size, char, exch = nyse_universe(20)
    labels, bps = two_by_three_sort(size, char, exchange=exch)
    # NYSE: size median = 10.5, p30 = 1 + .3*19 = 6.7, p70 = 1 + .7*19 = 14.3
    assert bps.from_nyse and bps.n_nyse == 20 and bps.n_names == 30
    assert bps.size == pytest.approx(10.5)
    assert (bps.low, bps.high) == pytest.approx((6.7, 14.3))
    assert labels["N8"] == "S/M"     # all-names breakpoints would make it B/L
    assert labels["N11"] == "B/M"
    assert labels["N15"] == "B/H"
    assert labels["Q1"] == "S/H"


def test_breakpoints_fall_back_to_all_names_below_20_nyse():
    size, char, exch = nyse_universe(19)
    labels, bps = two_by_three_sort(size, char, exchange=exch)
    # all 30 names: size median = 5.5; char p30 = 9 + .7*1 = 9.7, p70 = 100
    assert not bps.from_nyse and bps.n_nyse == 19
    assert bps.size == pytest.approx(5.5)
    assert (bps.low, bps.high) == pytest.approx((9.7, 100.0))
    assert labels["N8"] == "B/L"


def test_exchange_labels_and_unsortable_names():
    size, char, exch = nyse_universe(20)
    exch = exch.copy()
    exch.iloc[:5] = ["nyse ", "NYQ", "XNYS", "N", "NYSE"]  # all count as NYSE
    _, bps = two_by_three_sort(size, char, exchange=exch)
    assert bps.n_nyse == 20
    exch.iloc[0] = "NYSE American"  # formerly AMEX: not NYSE
    _, bps = two_by_three_sort(size, char, exchange=exch)
    assert bps.n_nyse == 19 and not bps.from_nyse
    size2 = size.copy()
    size2["N1"] = np.nan
    char2 = char.copy()
    char2["N2"] = np.nan
    labels, _ = two_by_three_sort(size2, char2, exchange=exch, labels=("W", "N", "R"))
    assert labels["N1"] is None and labels["N2"] is None
    assert set(labels.dropna()) <= {"S/W", "S/N", "S/R", "B/W", "B/N", "B/R"}
    empty, none_bps = two_by_three_sort(pd.Series([np.nan], index=["a"]), pd.Series([1.0], index=["a"]))
    assert none_bps is None and empty["a"] is None


def _wide_panel(n_nyse: int, months: int = 4) -> CharacteristicsPanel:
    size, char, exch = nyse_universe(n_nyse)
    idx = pd.date_range("2020-01-31", periods=months, freq="ME")
    names = list(size.index)
    rng = np.random.default_rng(0)
    return CharacteristicsPanel(
        returns=pd.DataFrame(rng.normal(0.01, 0.05, (months, len(names))), index=idx, columns=names),
        market_cap=pd.DataFrame([size.to_numpy()] * months, index=idx, columns=names),
        book_equity=pd.DataFrame([(char * size).to_numpy()] * months, index=idx, columns=names),
        exchange=exch,
    )


def test_construct_warns_only_when_nyse_breakpoints_unavailable():
    _, warnings = construct_factors(_wide_panel(20), "ff3", formation="monthly", min_names_per_portfolio=1)
    assert not any("NYSE" in w for w in warnings)
    _, warnings = construct_factors(_wide_panel(19), "ff3", formation="monthly", min_names_per_portfolio=1)
    (w,) = [w for w in warnings if "NYSE" in w]
    # 4 monthly dates -> 3 formations that are held (the one on the last date never is)
    assert "size-B/M breakpoints" in w and "fewer than 20 NYSE names" in w and "3 of 3" in w


# --- no look-ahead ------------------------------------------------------------------------------


def _random_panel(seed: int, n_names: int = 40, start: str = "2018-01-31", months: int = 40) -> CharacteristicsPanel:
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=months, freq="ME")
    names = [f"T{i:02d}" for i in range(n_names)]

    def panel(values):
        return pd.DataFrame(values, index=idx, columns=names)

    cap = np.exp(rng.normal(3, 1.5, n_names)) * np.cumprod(1 + rng.normal(0.01, 0.06, (months, n_names)), axis=0)
    return CharacteristicsPanel(
        returns=panel(rng.normal(0.01, 0.06, (months, n_names))),
        market_cap=panel(cap),
        book_equity=panel(cap * np.exp(rng.normal(-0.5, 0.6, (months, n_names)))),
        operating_profitability=panel(rng.normal(0.2, 0.1, (months, n_names))),
        investment=panel(rng.normal(0.08, 0.1, (months, n_names))),
        momentum_12_1=panel(rng.normal(0.1, 0.3, (months, n_names))),
        exchange=pd.Series(["NYSE" if i % 2 else "NASDAQ" for i in range(n_names)], index=names),
        rf=pd.Series(0.001, index=idx),
    )


def _scramble_after(p: CharacteristicsPanel, cut: pd.Timestamp, seed: int) -> CharacteristicsPanel:
    rng = np.random.default_rng(seed)

    def scr(df):
        df = df.copy()
        later = df.index > cut
        df.loc[later] = rng.permutation(df.loc[later].to_numpy().ravel()).reshape(df.loc[later].shape) * 3.0
        return df

    rf = p.rf.copy()
    rf[rf.index > cut] = 0.05
    return CharacteristicsPanel(
        scr(p.returns), scr(p.market_cap), scr(p.book_equity), scr(p.operating_profitability),
        scr(p.investment), scr(p.momentum_12_1), p.exchange, rf,
    )


@pytest.mark.parametrize("model", ["ff3", "ff5", "carhart4"])
@pytest.mark.parametrize("formation", ["annual_june", "monthly"])
def test_no_look_ahead(model, formation):
    p = _random_panel(1)
    base, _ = construct_factors(p, model, formation=formation, min_names_per_portfolio=1)
    assert base.iloc[20:].notna().all().all()
    for cut in (pd.Timestamp("2019-06-30"), pd.Timestamp("2019-11-30"), pd.Timestamp("2020-06-30")):
        changed, _ = construct_factors(_scramble_after(p, cut, 7), model, formation=formation, min_names_per_portfolio=1)
        pd.testing.assert_frame_equal(base.loc[:cut], changed.loc[:cut])
        assert not np.allclose(base.loc[cut:].iloc[1:].to_numpy(), changed.loc[cut:].iloc[1:].to_numpy())


def test_momentum_warmup_is_nan_without_thin_warnings():
    # momentum_12_1 needs 12 months of history: NaN before that is "no portfolio yet", not "thin"
    p = _random_panel(2)
    mom = p.momentum_12_1.copy()
    mom.iloc[:12] = np.nan
    p.momentum_12_1 = mom
    f, warnings = construct_factors(p, "carhart4", formation="monthly", min_names_per_portfolio=1)
    assert f["Mom"].iloc[:13].isna().all()  # rows 0-11 have no signal; first formation at row 12, held from 13
    assert f["Mom"].iloc[13:].notna().all()
    assert not any("size-momentum" in w for w in warnings)


def test_constructed_hml_tracks_the_generating_factor():
    # 200 names whose returns load +1 / -1 on a latent HML according to their (fixed) B/M tercile:
    # the constructed HML must be close to 2 x latent with very high correlation.
    rng = np.random.default_rng(42)
    n, months = 200, 48
    idx = pd.date_range("2015-01-31", periods=months, freq="ME")
    names = [f"S{i:03d}" for i in range(n)]
    bm = rng.uniform(0.1, 2.0, n)
    loading = np.where(bm > np.percentile(bm, 70), 1.0, np.where(bm <= np.percentile(bm, 30), -1.0, 0.0))
    mkt = rng.normal(0.008, 0.04, months)
    hml_true = rng.normal(0.003, 0.03, months)
    rets = mkt[:, None] + hml_true[:, None] * loading[None, :] + rng.normal(0, 0.01, (months, n))
    cap = np.exp(rng.normal(8, 1, n))
    p = CharacteristicsPanel(
        returns=pd.DataFrame(rets, index=idx, columns=names),
        market_cap=pd.DataFrame([cap] * months, index=idx, columns=names),
        book_equity=pd.DataFrame([bm * cap] * months, index=idx, columns=names),
        rf=pd.Series(0.002, index=idx),
    )
    f, warnings = construct_factors(p, "ff3")  # annual June formation, default thresholds
    assert f.loc["2015-07-31":, "HML"].notna().all()
    official = pd.DataFrame({"Mkt-RF": mkt - 0.002, "HML": hml_true}, index=idx)
    checks = {c.factor: c for c in compare_with_official(f, official)}
    assert checks["HML"].correlation_with_official > 0.95
    assert checks["Mkt-RF"].correlation_with_official > 0.98
    assert checks["HML"].n_overlap_periods == months - 6  # July 2015 onward
    assert checks["SMB"].correlation_with_official is None  # no official SMB supplied


# =============================================================================================
# compare_with_official
# =============================================================================================


def test_compare_with_official_on_known_pairs():
    rng = np.random.default_rng(3)
    months = pd.date_range("2018-01-31", periods=36, freq="ME")
    official = pd.DataFrame(
        rng.normal(0.004, 0.03, (36, 4)), index=months, columns=["Mkt-RF", "SMB", "HML", "RF"]
    )
    # constructed series sit on the last *business* day of each month and must align by month
    bdays = pd.date_range("2018-01-31", periods=36, freq="BME")
    noise = rng.normal(0, 0.03, 36)
    constructed = pd.DataFrame(
        {
            "Mkt-RF": official["Mkt-RF"].to_numpy() + noise,
            "SMB": 2.0 * official["SMB"].to_numpy() + 0.001,   # perfectly correlated
            "HML": -official["HML"].to_numpy(),                # perfectly anti-correlated
            "Mom": rng.normal(0.01, 0.04, 36),                 # no official counterpart
            "RF": official["RF"].to_numpy(),
        },
        index=bdays,
    )
    constructed.iloc[:3, 0] = np.nan  # first three Mkt-RF values missing
    checks = compare_with_official(constructed, official)
    assert all(isinstance(c, FactorConstructionCheck) for c in checks)
    by = {c.factor: c for c in checks}
    assert list(by) == ["Mkt-RF", "SMB", "HML", "Mom"]  # RF is not a factor

    expected = np.corrcoef(constructed["Mkt-RF"].iloc[3:], official["Mkt-RF"].iloc[3:])[0, 1]
    assert by["Mkt-RF"].correlation_with_official == pytest.approx(expected)
    assert by["Mkt-RF"].n_overlap_periods == 33
    assert by["Mkt-RF"].annual_premium_official_pct == pytest.approx(official["Mkt-RF"].iloc[3:].mean() * 1200)

    assert by["SMB"].correlation_with_official == pytest.approx(1.0)
    assert by["SMB"].n_overlap_periods == 36
    assert by["SMB"].annual_premium_constructed_pct == pytest.approx((2 * official["SMB"].mean() + 0.001) * 1200)
    assert by["SMB"].annual_premium_official_pct == pytest.approx(official["SMB"].mean() * 1200)
    assert by["HML"].correlation_with_official == pytest.approx(-1.0)

    assert by["Mom"].correlation_with_official is None
    assert by["Mom"].n_overlap_periods == 0
    assert by["Mom"].annual_premium_official_pct is None
    assert by["Mom"].annual_premium_constructed_pct == pytest.approx(constructed["Mom"].mean() * 1200)


def test_compare_with_official_matches_umd_alias_and_compounds_daily():
    days = pd.bdate_range("2021-01-01", "2021-06-30")
    rng = np.random.default_rng(5)
    daily = pd.Series(rng.normal(0.0005, 0.01, len(days)), index=days)
    monthly = (1 + daily).groupby(days.to_period("M")).prod() - 1
    official = pd.DataFrame({"UMD": monthly.to_numpy()}, index=monthly.index.to_timestamp(how="end").normalize())
    checks = compare_with_official(pd.DataFrame({"Mom": daily}), official)
    (c,) = checks
    assert c.factor == "Mom"
    assert c.n_overlap_periods == 6
    assert c.correlation_with_official == pytest.approx(1.0)
    assert c.annual_premium_constructed_pct == pytest.approx(c.annual_premium_official_pct)
    assert c.annual_premium_official_pct == pytest.approx(monthly.mean() * 1200)


def test_compare_with_official_handles_tiny_or_constant_overlap():
    idx = pd.date_range("2020-01-31", periods=2, freq="ME")
    checks = compare_with_official(
        pd.DataFrame({"SMB": [0.01, 0.02]}, index=idx), pd.DataFrame({"SMB": [0.03, 0.01]}, index=idx)
    )
    assert checks[0].n_overlap_periods == 2 and checks[0].correlation_with_official is None
    idx5 = pd.date_range("2020-01-31", periods=5, freq="ME")
    checks = compare_with_official(
        pd.DataFrame({"HML": [0.01] * 5}, index=idx5), pd.DataFrame({"HML": [0.0, 0.01, 0.02, 0.0, 0.01]}, index=idx5)
    )
    assert checks[0].correlation_with_official is None  # constant constructed series
    assert checks[0].annual_premium_constructed_pct == pytest.approx(12.0)


def test_french_fixture_roundtrip_through_compare(tmp_path):
    official = load_french_factors("ff3", cache_dir=tmp_path, fetch=FixtureFetch())
    constructed = official[["Mkt-RF", "SMB", "HML"]] * 0.8
    checks = {c.factor: c for c in compare_with_official(constructed, official)}
    assert checks["SMB"].correlation_with_official == pytest.approx(1.0)
    assert checks["HML"].n_overlap_periods == 19  # 2023-03 HML is missing (-99.99)
    assert checks["SMB"].annual_premium_constructed_pct == pytest.approx(0.8 * checks["SMB"].annual_premium_official_pct)


def test_fixture_builder_is_reproducible(tmp_path):
    import importlib.util

    spec = importlib.util.spec_from_file_location("build_fixtures", FIXTURES / "build_fixtures.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    for path in mod.build(tmp_path):
        # compare decompressed contents (raw DEFLATE bytes can differ between zlib builds)
        with zipfile.ZipFile(io.BytesIO(path.read_bytes())) as zf, zipfile.ZipFile(FIXTURES / path.name) as ref:
            (member,) = zf.namelist()
            assert ref.namelist() == [member]
            assert zf.read(member) == ref.read(member)
            assert b"Copyright 2024 Kenneth R. French" in zf.read(member)
            assert b"\r\n" in zf.read(member)
