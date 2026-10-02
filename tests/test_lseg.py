"""Offline tests for the LSEG adapter (``aitrading.data.lseg``) and SCREEN compiler (``compile_lseg``).

No LSEG SDK and no network: a fake ``lseg.data`` module is injected through ``sys.modules`` (or the
``ld_module=`` constructor argument). It records every call and mimics the documented behaviour of
lseg-data 2.1.1 that the adapter relies on: ``get_data`` returns an ``Instrument`` column plus one
column per *known* field (unknown / unentitled fields are dropped silently), ``get_history`` returns
MultiIndex (RIC, field) columns for several RICs and flat field columns for one RIC. The tests pin
the exact SCREEN strings and request arguments against docs/VENDOR_REFERENCE.md section 2.
"""

from __future__ import annotations

import json
import math
import re
import sys
import types
from datetime import date, datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from aitrading.core import fields as F
from aitrading.core.models import Document, DocumentKind
from aitrading.core.policy import DataBoundary
from aitrading.data.base import MarketDataProvider, ProviderError, ProviderUnavailable, PushdownResult, ScreenPushdown
from aitrading.data.lseg import (
    BOUNDARY_NOTE,
    DEFAULT_FIELDMAP_PATH,
    FIELDMAP_ENV,
    FieldCheck,
    LSEGProvider,
    load_fieldmap,
    ric_to_ticker,
    validate_fieldmap,
)
from aitrading.screen import compile_lseg
from aitrading.screen.compile_lseg import compile_screen, compile_universe, format_number, preflight_codes
from aitrading.screen.spec import Condition, RankFactor, ScreenSpec, UniverseSpec

TODAY = date(2026, 10, 2)
AS_OF = date(2026, 10, 1)
UNIVERSE = 'U(IN(Equity(active,public,primary))/*UNV:Public*/)'
LISTING = 'IN(TR.ExchangeCountryCode,"US"), IN(TR.InstrumentTypeCode,"ORD"), NOT_IN(TR.ExchangeMarketIdCode,"OTCM")'
LISTING_EXPR = f"SCREEN({UNIVERSE}, {LISTING}, CURN=USD)"
ANY_TYPE_EXPR = f'SCREEN({UNIVERSE}, IN(TR.ExchangeCountryCode,"US"), NOT_IN(TR.ExchangeMarketIdCode,"OTCM"), CURN=USD)'
SRC = Path(__file__).resolve().parents[1] / "src" / "aitrading"


# =============================================================================================
# Fake lseg.data
# =============================================================================================


class FakeConfig:
    def __init__(self) -> None:
        self.params: dict[str, object] = {}

    def set_param(self, key: str, value: object) -> None:
        self.params[key] = value


class FakeLD(types.ModuleType):
    """Minimal stand-in for ``lseg.data`` (2.1.1 shapes)."""

    def __init__(self) -> None:
        super().__init__("lseg.data")
        self.calls: list[tuple[str, dict]] = []
        self.values: dict[tuple[str, str], object] = {}  # (RIC, code) -> value
        self.screens: dict[str, list[str]] = {}  # SCREEN expression -> RICs
        self.history: dict[tuple[str, str], pd.Series] = {}  # (RIC, code) -> daily series
        self.history_errors: set[str] = set()
        self.headlines: dict[str, pd.DataFrame] = {}
        self.stories: dict[str, str] = {}
        self.config = FakeConfig()
        self.HeaderType = SimpleNamespace(NAME="name", TITLE="title")
        self.news = SimpleNamespace(get_headlines=self._get_headlines, get_story=self._get_story)
        self.closed = False

    # --- session
    def open_session(self, **kwargs):
        self.calls.append(("open_session", kwargs))
        return SimpleNamespace(**kwargs)

    def close_session(self):
        self.closed = True

    def get_config(self):
        return self.config

    # --- data
    def _known(self, code: str) -> bool:
        return any(c == code for (_, c) in self.values)

    def get_data(self, universe, fields, parameters=None, header_type=None):
        self.calls.append(("get_data", {"universe": universe, "fields": list(fields), "parameters": parameters,
                                        "header_type": header_type}))
        if isinstance(universe, str) and universe.startswith("SCREEN("):
            if universe not in self.screens:
                raise RuntimeError(f"unexpected SCREEN {universe}")
            rics = self.screens[universe]
        else:
            rics = [universe] if isinstance(universe, str) else list(universe)
        cols = {"Instrument": rics}
        for code in fields:
            if self._known(code):  # LSEG drops bad / unentitled fields silently
                cols[code.upper()] = [self.values.get((r, code)) for r in rics]
        return pd.DataFrame(cols)

    def get_history(self, universe, fields, interval="daily", start=None, end=None, adjustments=None, count=None):
        self.calls.append(("get_history", {"universe": list(universe), "fields": list(fields), "interval": interval,
                                           "start": start, "end": end, "adjustments": adjustments}))
        rics = list(universe)
        if any(r in self.history_errors for r in rics):
            raise RuntimeError("history request failed")
        lo, hi = pd.Timestamp(start), pd.Timestamp(end)
        frames = {}
        for r in rics:
            cols = {f: self.history[(r, f)] for f in fields if (r, f) in self.history}
            if cols:
                df = pd.DataFrame(cols)
                frames[r] = df[(df.index >= lo) & (df.index <= hi)]
        if not frames:
            return pd.DataFrame()
        if len(rics) == 1:
            df = frames[rics[0]]
            df.index.name = "Date"
            df.columns.name = rics[0]
            return df
        out = pd.concat(frames, axis=1)
        out.index.name = "Date"
        return out

    # --- news
    def _get_headlines(self, query, start=None, end=None, count=10):
        self.calls.append(("get_headlines", {"query": query, "start": start, "end": end, "count": count}))
        return self.headlines.get(query, pd.DataFrame())

    def _get_story(self, story_id, *args, **kwargs):
        self.calls.append(("get_story", {"story_id": story_id}))
        if story_id not in self.stories:
            raise RuntimeError("no story")
        return self.stories[story_id]

    def call_list(self, name: str) -> list[dict]:
        return [kw for n, kw in self.calls if n == name]


def _days(n: int = 10, end: str = "2026-10-01") -> pd.DatetimeIndex:
    return pd.bdate_range(end=end, periods=n)


def _base_fake() -> FakeLD:
    ld = FakeLD()
    v = ld.values
    # test RIC answers every verified field
    for code, val in {
        "TR.CommonName": "International Business Machines Corp", "TR.GICSSector": "Information Technology",
        "TR.ExchangeMarketIdCode": "XNYS", "TR.ExchangeCountryCode": "US", "TR.InstrumentTypeCode": "ORD",
        "TR.CompanyMarketCap(Scale=6)": 230000.0, "TR.PriceClose": 250.0,
    }.items():
        v[("IBM.N", code)] = val
    return ld


@pytest.fixture()
def fake(monkeypatch):
    ld = _base_fake()
    pkg = types.ModuleType("lseg")
    pkg.data = ld
    pkg.__path__ = []
    monkeypatch.setitem(sys.modules, "lseg", pkg)
    monkeypatch.setitem(sys.modules, "lseg.data", ld)
    monkeypatch.delenv(FIELDMAP_ENV, raising=False)
    monkeypatch.setenv("LSEG_APP_KEY", "test-app-key")
    return ld


def _provider(**kw) -> LSEGProvider:
    p = LSEGProvider(today=lambda: TODAY, **kw)
    p.request_pause_s = 0.0
    return p


def _spec(conditions, universe: UniverseSpec | None = None, any_of=None) -> ScreenSpec:
    return ScreenSpec(name="t", observation="t", universe=universe or UniverseSpec(min_price=None,
                      min_avg_dollar_volume_usd_mn=None), conditions=conditions, any_of=any_of or [],
                      ranking=[RankFactor(feature="rsi_14", direction="lower_is_better")])


def _representative_spec() -> ScreenSpec:
    """The ADR's representative observation, in catalog units."""
    return ScreenSpec(
        name="dislocation", observation="US $2-20bn names ...", universe=UniverseSpec(),
        conditions=[
            Condition(feature="market_cap_usd_bn", op="between", value=2, value_high=20),
            Condition(feature="sma_50_vs_sma_200_pct", op=">", value=0),
            Condition(feature="drawdown_from_52w_high_pct", op="between", value=-45, value_high=-20),
            Condition(feature="rsi_14", op="<", value=45),
            Condition(feature="rel_volume_5d", op=">", value=1.5),
            Condition(feature="fcf_yield_pct", op=">=", value=5),
            Condition(feature="revenue_growth_yoy_pct", op=">=", value=10),
            Condition(feature="short_interest_pct_float", op=">=", value=5),
            Condition(feature="return_3m_pct", op="<", value=0),
        ],
        ranking=[RankFactor(feature="fcf_yield_pct", direction="higher_is_better")],
    )


# =============================================================================================
# Field map
# =============================================================================================


def test_default_fieldmap_valid_and_statuses_match_reference(monkeypatch):
    monkeypatch.delenv(FIELDMAP_ENV, raising=False)
    fm = load_fieldmap()
    assert validate_fieldmap(fm) == []
    assert fm["vendor"] == "lseg" and fm["test_ric"] == "IBM.N"
    feat, raw = fm["features"], fm["raw"]
    # table 2.3 statuses, verbatim
    assert (feat["market_cap_usd_bn"]["expression"], feat["market_cap_usd_bn"]["status"]) == ("TR.CompanyMarketCap(Scale=6)", "confirmed")
    assert (feat["return_3m_pct"]["expression"], feat["return_3m_pct"]["status"]) == ("TR.TotalReturn3Mo", "corrected")
    assert feat["return_6m_pct"]["status"] == "unverifiable"
    assert feat["drawdown_from_52w_high_pct"]["status"] == "confirmed"
    assert feat["drawdown_from_52w_high_pct"]["units_verified"] is False  # sign/scale UNVERIFIED
    assert feat["ev_to_ebitda"]["status"] == "corrected" and feat["net_debt_to_ebitda"]["status"] == "confirmed"
    assert feat["rsi_14"]["expression"] is None  # computed locally (Wilder)
    assert raw["fundamentals"]["fcf_ttm"]["status"] == "unverifiable"
    assert raw["fundamentals"]["revenue_last_q"]["code"] == "TR.RevenueActValue(Period=FQ0)"
    assert raw["fundamentals"]["revenue_last_q_prior_year"]["status"] == "unverifiable"  # FQ-4 combination
    assert raw["short_interest"]["short_interest_shares"]["status"] == "unverifiable"
    assert raw["short_interest"]["float_shares"]["status"] == "unverifiable"
    assert raw["universe"]["gics_industry"]["status"] == "unverifiable"
    assert raw["universe"]["market_cap"]["to_canonical"] == 1_000_000
    close = fm["history"]["fields"]["close"]
    assert (close["code"], close["status"]) == ("TRDPRC_1", "confirmed")
    assert fm["history"]["adjustments"]["status"] == "unverifiable"
    statuses = re.findall(r'"status":\s*"([^"]+)"', DEFAULT_FIELDMAP_PATH.read_text())
    assert statuses and set(statuses) <= {"confirmed", "corrected", "unverifiable"}


def test_unverifiable_codes_are_not_hard_coded_in_logic(monkeypatch):
    """Unverifiable codes live only in the field map data file, never in the Python modules."""
    monkeypatch.delenv(FIELDMAP_ENV, raising=False)
    fm = load_fieldmap()
    codes: set[str] = set()

    def walk(node):
        if isinstance(node, dict):
            if node.get("status") == "unverifiable":
                for k in ("code", "expression"):
                    if isinstance(node.get(k), str):
                        codes.add(re.split(r"[({]", node[k])[0])
            for v in node.values():
                walk(v)

    walk({k: v for k, v in fm.items() if k != "status_meaning"})
    assert {"TR.FreeCashFlow", "TR.ShortInterest", "TR.SharesFreeFloat", "OPEN_PRC"} <= codes
    sources = (SRC / "data" / "lseg.py").read_text() + (SRC / "screen" / "compile_lseg.py").read_text()
    for code in codes:
        assert code not in sources, code


def test_env_override_is_deep_merged(tmp_path, monkeypatch):
    over = tmp_path / "lseg_override.json"
    over.write_text(json.dumps({
        "features": {"drawdown_from_52w_high_pct": {"units_verified": True}},
        "raw": {"fundamentals": {"fcf_ttm": None}},  # null deletes an entry
        "test_ric": "MSFT.O",
    }))
    monkeypatch.setenv(FIELDMAP_ENV, str(over))
    fm = load_fieldmap()
    dd = fm["features"]["drawdown_from_52w_high_pct"]
    assert dd["units_verified"] is True and dd["expression"] == "TR.PricePctChg52WkHigh"  # other keys kept
    assert "fcf_ttm" not in fm["raw"]["fundamentals"]
    assert fm["test_ric"] == "MSFT.O"
    assert fm["_sources"][-1] == str(over)
    # constructor mapping is merged on top of the env file
    fm2 = load_fieldmap({"test_ric": "AAPL.O"})
    assert fm2["test_ric"] == "AAPL.O" and fm2["features"]["drawdown_from_52w_high_pct"]["units_verified"] is True


@pytest.mark.parametrize("bad, msg", [
    ({"features": {"market_cap_usd_bn": {"status": "verified"}}}, "status"),
    ({"features": {"not_a_feature": {"expression": "TR.X", "status": "confirmed"}}}, "not a catalog feature"),
    ({"raw": {"fundamentals": {"bogus_col": {"code": "TR.X", "status": "confirmed"}}}}, "canonical column"),
    ({"features": {"price": {"screen": "always"}}}, "screen"),
    ({"features": {"price": {"threshold_scale": 0}}}, "threshold_scale"),
])
def test_invalid_overrides_are_rejected(bad, msg, monkeypatch):
    monkeypatch.delenv(FIELDMAP_ENV, raising=False)
    with pytest.raises(ValueError, match=msg):
        load_fieldmap(bad)


def test_missing_override_file_is_reported(monkeypatch, tmp_path):
    monkeypatch.setenv(FIELDMAP_ENV, str(tmp_path / "nope.json"))
    with pytest.raises(ValueError, match="does not exist"):
        load_fieldmap()


# =============================================================================================
# Compiler
# =============================================================================================


@pytest.fixture()
def fm(monkeypatch):
    monkeypatch.delenv(FIELDMAP_ENV, raising=False)
    return load_fieldmap()


def test_compile_representative_screen_exact_string(fm):
    spec = _representative_spec()
    assert preflight_codes(spec, fm, AS_OF, today=TODAY) == ["TR.PriceClose"]
    cq = compile_screen(spec, fm, AS_OF, preflight={"TR.PriceClose": True}, today=TODAY)
    assert cq.expression == (
        f"SCREEN({UNIVERSE}, {LISTING}, TR.PriceClose>=5, TR.CompanyMarketCap(Scale=6)>=2000, "
        "TR.CompanyMarketCap(Scale=6)<=20000, TR.TotalReturn3Mo<0, CURN=USD)"
    )
    assert cq.point_in_time
    assert cq.pushed_conditions == [
        "country == US", "security_type in [common_stock]",
        "universe: exchange market not OTCM (OTC Markets excluded from the LSEG universe)",
        "price >= 5", "market_cap_usd_bn between 2 and 20", "return_3m_pct < 0",
    ]
    assert cq.residual_conditions == [
        "avg_dollar_volume_20d_usd_mn >= 5", "sma_50_vs_sma_200_pct > 0",
        "drawdown_from_52w_high_pct between -45 and -20", "rsi_14 < 45", "rel_volume_5d > 1.5",
        "fcf_yield_pct >= 5", "revenue_growth_yoy_pct >= 10", "short_interest_pct_float >= 5",
    ]
    reasons = {r.condition: r.reason for r in cq.residual}
    assert "UNVERIFIED" in reasons["drawdown_from_52w_high_pct between -45 and -20"]
    assert "computed locally" in reasons["rsi_14 < 45"]
    # every pushed predicate maps to a confirmed / corrected expression
    assert {p.status for p in cq.pushed} <= {"confirmed", "corrected"}
    mcap = next(p for p in cq.pushed if p.feature == "market_cap_usd_bn")
    assert mcap.expressions == ("TR.CompanyMarketCap(Scale=6)>=2000", "TR.CompanyMarketCap(Scale=6)<=20000")


def test_preflight_gates_unofficial_predicates(fm):
    spec = _spec([Condition(feature="market_cap_usd_bn", op=">=", value=2)], universe=UniverseSpec())
    no_pf = compile_screen(spec, fm, AS_OF, today=TODAY)
    assert "TR.PriceClose" not in no_pf.expression
    assert "price >= 5" in no_pf.residual_conditions
    failed = compile_screen(spec, fm, AS_OF, preflight={"TR.PriceClose": False}, today=TODAY)
    assert "TR.PriceClose" not in failed.expression
    ok = compile_screen(spec, fm, AS_OF, preflight={"TR.PriceClose": True}, today=TODAY)
    assert "TR.PriceClose>=5" in ok.predicates and dict(ok.preflight) == {"TR.PriceClose": True}


def test_unverifiable_or_unadmitted_features_are_never_pushed(fm):
    spec = _spec([Condition(feature="market_cap_usd_bn", op=">=", value=2),
                  Condition(feature="return_6m_pct", op=">", value=10)])
    over = load_fieldmap({"features": {"market_cap_usd_bn": {"status": "unverifiable"},
                                       "return_6m_pct": {"screen": "official"}}})
    cq = compile_screen(spec, over, AS_OF, preflight={"TR.TotalReturn6Mo": True}, today=TODAY)
    assert cq.expression == LISTING_EXPR
    assert cq.residual_conditions == ["market_cap_usd_bn >= 2", "return_6m_pct > 10"]
    assert all("unverifiable" in r.reason for r in cq.residual)


def test_admitted_units_override_pushes_and_negative_scale_flips(fm):
    spec = _spec([Condition(feature="drawdown_from_52w_high_pct", op="between", value=-45, value_high=-20)])
    pf = {"TR.PricePctChg52WkHigh": True}
    assert compile_screen(spec, fm, AS_OF, preflight=pf, today=TODAY).expression == LISTING_EXPR
    admitted = load_fieldmap({"features": {"drawdown_from_52w_high_pct": {"units_verified": True}}})
    assert preflight_codes(spec, admitted, AS_OF, today=TODAY) == ["TR.PricePctChg52WkHigh"]
    cq = compile_screen(spec, admitted, AS_OF, preflight=pf, today=TODAY)
    assert cq.predicates[-2:] == ("TR.PricePctChg52WkHigh>=-45", "TR.PricePctChg52WkHigh<=-20")
    positive = load_fieldmap({"features": {"drawdown_from_52w_high_pct": {"units_verified": True, "threshold_scale": -1}}})
    cq2 = compile_screen(spec, positive, AS_OF, preflight=pf, today=TODAY)
    assert cq2.predicates[-2:] == ("TR.PricePctChg52WkHigh<=45", "TR.PricePctChg52WkHigh>=20")


def test_category_predicates_single_value_forms_only(fm):
    pf = {"TR.GICSSector": True}
    uni = UniverseSpec(min_price=None, min_avg_dollar_volume_usd_mn=None, exclude_sectors=["Energy", "Utilities"])
    one = Condition(feature="gics_sector", op="in", values=["Information Technology"])
    two = Condition(feature="gics_sector", op="in", values=["Energy", "Materials"])
    cq = compile_screen(_spec([one, two], universe=uni), fm, AS_OF, preflight=pf, today=TODAY)
    assert cq.predicates[3:] == ('NOT_IN(TR.GICSSector,"Energy")', 'NOT_IN(TR.GICSSector,"Utilities")',
                                 'IN(TR.GICSSector,"Information Technology")')
    assert "gics_sector not in [Energy, Utilities]" in cq.pushed_conditions
    assert cq.residual_conditions == ["gics_sector in [Energy, Materials]"]
    quoted = Condition(feature="gics_sector", op="in", values=['Bad "label"'])
    assert compile_screen(_spec([quoted]), fm, AS_OF, preflight=pf, today=TODAY).residual_conditions == [quoted.describe()]
    # SCREEN matching is exact: labels are pushed in the vendor's spelling, unknown labels stay local
    lower = Condition(feature="gics_sector", op="==", values=["health care"])
    assert compile_screen(_spec([lower]), fm, AS_OF, preflight=pf, today=TODAY).predicates[-1] == 'IN(TR.GICSSector,"Health Care")'
    typo = Condition(feature="gics_sector", op="not_in", values=["Tech"])
    cq = compile_screen(_spec([typo]), fm, AS_OF, preflight=pf, today=TODAY)
    assert cq.expression == LISTING_EXPR and "not among" in cq.residual[0].reason
    lc = compile_universe(UniverseSpec(country="us"), fm)
    assert 'IN(TR.ExchangeCountryCode,"US")' in lc.expression and lc.pushed_conditions[0] == "country == us"


def test_other_feature_equality_and_any_of_stay_local(fm):
    conds = [Condition(feature="market_cap_usd_bn", op=">", other_feature="enterprise_value_usd_bn"),
             Condition(feature="market_cap_usd_bn", op="==", value=5),
             Condition(feature="market_cap_usd_bn", op="!=", value=5)]
    any_of = [[Condition(feature="market_cap_usd_bn", op="<", value=1), Condition(feature="return_3m_pct", op=">", value=5)]]
    cq = compile_screen(_spec(conds, any_of=any_of), fm, AS_OF, today=TODAY)
    assert cq.expression == LISTING_EXPR
    assert len(cq.residual) == 4 and cq.residual[-1].condition.startswith("any of: ")


def test_historical_as_of_pushes_static_listing_only(fm):
    spec = _representative_spec()
    old = date(2025, 6, 30)
    assert preflight_codes(spec, fm, old, today=TODAY) == []
    cq = compile_screen(spec, fm, old, preflight={"TR.PriceClose": True}, today=TODAY)
    assert not cq.point_in_time
    assert cq.expression == LISTING_EXPR
    assert "market_cap_usd_bn between 2 and 20" in cq.residual_conditions
    assert all("historical" in r.reason for r in cq.residual if r.condition.startswith(("market_cap", "return_3m", "price")))


def test_threshold_scaling_is_exact_decimal(fm):
    cond = [Condition(feature="market_cap_usd_bn", op=">=", value=1.1), Condition(feature="market_cap_usd_bn", op="<", value=0.3),
            Condition(feature="market_cap_usd_bn", op="<=", value=12345.678)]
    cq = compile_screen(_spec(cond), fm, AS_OF, today=TODAY)
    assert cq.predicates[3:] == ("TR.CompanyMarketCap(Scale=6)>=1100", "TR.CompanyMarketCap(Scale=6)<300",
                                 "TR.CompanyMarketCap(Scale=6)<=12345678")
    assert [format_number(x) for x in (2000.0, -20, 0.5, 1e-7, 1e10, -0.0)] == ["2000", "-20", "0.5", "0.0000001",
                                                                               "10000000000", "0"]
    with pytest.raises(ValueError):
        format_number(float("nan"))


def test_compiled_query_audit_record(fm):
    cq = compile_screen(_representative_spec(), fm, AS_OF, today=TODAY)
    rec = cq.to_audit()
    json.dumps(rec)  # serialisable
    assert rec["query"] == cq.expression and len(rec["query_sha256"]) == 64 and rec["as_of"] == "2026-10-01"
    assert rec["fieldmap_version"] == "2026-10-02" and len(rec["fieldmap_sha256"]) == 64
    assert compile_screen(_representative_spec(), fm, AS_OF, today=TODAY).fieldmap_sha256 == cq.fieldmap_sha256


def test_compile_universe_and_unmapped_security_type(fm):
    assert compile_universe(UniverseSpec(), fm).expression == LISTING_EXPR
    cq = compile_universe(UniverseSpec(security_types=["common_stock", "adr"]), fm)
    assert 'TR.InstrumentTypeCode' not in cq.expression
    assert cq.residual_conditions == ["security_type in [common_stock, adr]"]
    assert "ORD" not in compile_universe(UniverseSpec(security_types=["reit"]), fm).expression


# =============================================================================================
# Provider: setup, boundary, SDK
# =============================================================================================


def test_sdk_missing_raises_provider_unavailable(monkeypatch):
    monkeypatch.setitem(sys.modules, "lseg", None)
    monkeypatch.setitem(sys.modules, "lseg.data", None)
    monkeypatch.delenv(FIELDMAP_ENV, raising=False)
    p = _provider()  # constructing never imports the SDK
    with pytest.raises(ProviderUnavailable, match=r"pip install lseg-data==2\.1\.1"):
        p.get_universe(UniverseSpec(), AS_OF)
    with pytest.raises(ProviderUnavailable, match="platform.ldp"):
        p.pushdown_screen(_representative_spec(), AS_OF)
    assert any("lseg-data" in d for d in p.diagnostics())


def test_default_boundary_denies_text_and_values(fake):
    p = _provider()
    b = p.boundary
    assert b.provider == "lseg" and b.allowed_document_kinds == set() and b.allow_numeric_features is False
    assert "G3" in b.note and b.note == BOUNDARY_NOTE
    doc = Document(doc_id="x", ticker="IBM", kind=DocumentKind.NEWS, title="t", published_at=datetime(2026, 9, 1),
                   source="LSEG", text="t")
    assert not b.permits(doc)
    with pytest.raises(ValueError, match="G3"):
        _provider(boundary=DataBoundary(provider="lseg", note="legal said ok"))
    with pytest.raises(ValueError, match="provider"):
        _provider(boundary=DataBoundary(provider="bloomberg", allowed_document_kinds=set(), allow_numeric_features=False))
    widened = DataBoundary(provider="lseg", allowed_document_kinds={DocumentKind.NEWS},
                           note="G3: LSEG written AI-use confirmation for Reuters news, 2026-11-01")
    assert _provider(boundary=widened).boundary is widened


def test_protocols_and_capabilities(fake):
    p = _provider()
    assert isinstance(p, MarketDataProvider) and isinstance(p, ScreenPushdown)
    assert p.name == "lseg"
    from aitrading.data.base import Capability as C
    assert C.SCREEN_PUSHDOWN in p.capabilities and C.NEWS in p.capabilities
    assert C.TRANSCRIPTS not in p.capabilities and C.FILINGS not in p.capabilities
    assert fake.calls == []  # lazy: nothing touched yet


def test_session_opened_lazily_once(fake, monkeypatch):
    p = _provider(session_name="platform.ldp")
    fake.screens[LISTING_EXPR] = ["IBM.N"]
    p.get_universe(UniverseSpec(), AS_OF)
    p.get_universe(UniverseSpec(), AS_OF)
    assert fake.call_list("open_session") == [{"name": "platform.ldp", "app_key": "test-app-key"}]
    assert fake.config.params == {"http.request-timeout": 300}
    monkeypatch.delenv("LSEG_APP_KEY")
    p2 = _provider(ld_module=fake)
    p2.get_universe(UniverseSpec(), AS_OF)
    assert fake.call_list("open_session")[-1] == {}  # library default (desktop.workspace)
    p2.close()
    assert fake.closed


def test_session_failure_is_provider_unavailable(fake):
    def boom(**kw):
        raise RuntimeError("Workspace is not running")

    fake.open_session = boom
    with pytest.raises(ProviderUnavailable, match="Workspace is not running"):
        _provider().get_universe(UniverseSpec(), AS_OF)


def test_diagnostics_platform_needs_config(fake, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("LD_LIB_CONFIG_PATH", raising=False)
    assert any("lseg-data.config.json" in d for d in _provider(session_name="platform.ldp").diagnostics())
    cfgdir = tmp_path / "cfg"
    cfgdir.mkdir()
    (cfgdir / "lseg-data.config.json").write_text("{}")
    monkeypatch.setenv("LD_LIB_CONFIG_PATH", str(cfgdir))
    assert _provider(session_name="platform.ldp").diagnostics() == []
    assert _provider().diagnostics() == []  # desktop: app key set by the fixture


# =============================================================================================
# Provider: universe and push-down
# =============================================================================================


def _universe_values(ld: FakeLD) -> None:
    rows = {
        "AAPL.O": ("Apple Inc", "Information Technology", "XNAS", "US", "ORD", 3_500_000.0),
        "BRKb.N": ("Berkshire Hathaway Inc", "Financials", "XNYS", "US", "ORD", 1_000_000.0),
        "XOM.N": ("Exxon Mobil Corp", "Energy", "XNYS", "US", "ORD", None),
        "ODD.N": ("Odd Corp", "", "ZZZZ", "US", "PRF", "n/a"),
    }
    codes = ["TR.CommonName", "TR.GICSSector", "TR.ExchangeMarketIdCode", "TR.ExchangeCountryCode",
             "TR.InstrumentTypeCode", "TR.CompanyMarketCap(Scale=6)"]
    for ric, vals in rows.items():
        for code, val in zip(codes, vals):
            ld.values[(ric, code)] = val
    ld.screens[LISTING_EXPR] = list(rows)
    ld.screens[ANY_TYPE_EXPR] = list(rows)


def test_get_universe_request_and_canonical_units(fake):
    _universe_values(fake)
    p = _provider()
    df = p.get_universe(UniverseSpec(exclude_sectors=["Energy"]), AS_OF)
    # one preflight for the unverifiable TR.GICSIndustry on the test RIC (reference form, no parameters)
    pf = [c for c in fake.call_list("get_data") if c["universe"] == "IBM.N"]
    assert pf == [{"universe": "IBM.N", "fields": ["TR.GICSIndustry"], "parameters": None, "header_type": "name"}]
    main = [c for c in fake.call_list("get_data") if c["universe"] == LISTING_EXPR]
    assert main == [{"universe": LISTING_EXPR,
                     "fields": ["TR.CommonName", "TR.GICSSector", "TR.ExchangeMarketIdCode", "TR.ExchangeCountryCode",
                                "TR.InstrumentTypeCode", "TR.CompanyMarketCap(Scale=6)"],
                     "parameters": {"Curn": "USD", "SDate": "2026-10-01"}, "header_type": "name"}]
    assert p.last_universe_query == LISTING_EXPR
    assert list(df.columns) == F.UNIVERSE_COLUMNS and df.index.name == "ticker"
    # XOM excluded by sector; ODD is not common stock
    assert list(df.index) == ["AAPL", "BRK-B"]
    assert df.loc["AAPL", F.MARKET_CAP] == pytest.approx(3.5e12)
    assert df.loc["AAPL", F.EXCHANGE] == "NASDAQ" and df.loc["BRK-B", F.EXCHANGE] == "NYSE"
    assert df.loc["AAPL", F.SECURITY_TYPE] == "common_stock" and df.loc["AAPL", F.CURRENCY] == "USD"
    assert df.loc["BRK-B", F.VENDOR_ID] == "BRKb.N"
    assert pd.isna(df.loc["AAPL", F.GICS_INDUSTRY])
    assert any("LEG NOT EVALUATED" in w and "gics_industry" in w for w in p.warnings)
    full = p.get_universe(UniverseSpec(security_types=[]), AS_OF)
    assert math.isnan(full.loc["XOM", F.MARKET_CAP])  # None -> NaN, never 0
    assert math.isnan(full.loc["ODD", F.MARKET_CAP]) and pd.isna(full.loc["ODD", F.GICS_SECTOR])
    assert full.loc["ODD", F.EXCHANGE] == "ZZZZ" and full.loc["ODD", F.SECURITY_TYPE] == "PRF"
    assert p.ric_for("AAPL") == "AAPL.O"


def test_pushdown_screen_exact_call_and_ticker_mapping(fake):
    spec = _representative_spec()
    expr = (f"SCREEN({UNIVERSE}, {LISTING}, TR.PriceClose>=5, TR.CompanyMarketCap(Scale=6)>=2000, "
            "TR.CompanyMarketCap(Scale=6)<=20000, TR.TotalReturn3Mo<0, CURN=USD)")
    fake.screens[expr] = ["MSFT.O", "BRKb.N", "XYZ.N^K20"]
    p = _provider()
    res = p.pushdown_screen(spec, AS_OF)
    assert isinstance(res, PushdownResult)
    assert res.query == expr  # recorded verbatim
    assert fake.call_list("get_data")[0] == {"universe": "IBM.N", "fields": ["TR.PriceClose"], "parameters": None,
                                             "header_type": "name"}  # preflight
    assert fake.call_list("get_data")[-1] == {"universe": expr, "fields": ["TR.CommonName"], "parameters": None,
                                              "header_type": "name"}
    assert res.tickers == ["BRK-B", "MSFT", "XYZ"]
    assert p.ric_for("BRK-B") == "BRKb.N" and p.ric_for("XYZ") == "XYZ.N^K20"
    assert res.pushed_conditions == p.last_pushdown.pushed_conditions
    assert "rsi_14 < 45" in res.residual_conditions and "market_cap_usd_bn between 2 and 20" in res.pushed_conditions
    # preflight is memoised for the day
    p.pushdown_screen(spec, AS_OF)
    assert len([c for c in fake.call_list("get_data") if c["universe"] == "IBM.N"]) == 1


def test_pushdown_preflight_failure_keeps_predicate_local(fake):
    del fake.values[("IBM.N", "TR.PriceClose")]
    fake.values[("ZZZ.N", "TR.PriceClose")] = 1.0  # field known, but empty for the test RIC
    spec = _spec([Condition(feature="market_cap_usd_bn", op=">=", value=2)], universe=UniverseSpec())
    expr = f"SCREEN({UNIVERSE}, {LISTING}, TR.CompanyMarketCap(Scale=6)>=2000, CURN=USD)"
    fake.screens[expr] = []
    p = _provider()
    res = p.pushdown_screen(spec, AS_OF)
    assert res.query == expr and res.tickers == []
    assert "price >= 5" in res.residual_conditions
    assert any("0 instruments" in w for w in p.warnings)


def test_pushdown_historical_as_of_warns(fake):
    fake.screens[LISTING_EXPR] = ["IBM.N"]
    p = _provider()
    res = p.pushdown_screen(_representative_spec(), date(2025, 3, 31))
    assert res.query == LISTING_EXPR and res.tickers == ["IBM"]
    assert any("survivorship" in w for w in p.warnings)
    assert not [c for c in fake.call_list("get_data") if c["universe"] == "IBM.N"]  # no preflight needed


def test_ticker_collision_and_unresolved_tickers(fake):
    fake.screens[ANY_TYPE_EXPR] = ["ABC.N", "ABC.O"]
    for r in ("ABC.N", "ABC.O"):
        fake.values[(r, "TR.CommonName")] = "Abc"
    p = _provider(rics={"IBM": "IBM.N"})
    df = p.get_universe(UniverseSpec(security_types=[]), AS_OF)
    assert list(df.index) == ["ABC", "ABC.O"]
    assert any("already ABC.N" in w for w in p.warnings)
    fake.values[("IBM.N", "TR.SharesOutstanding")] = 9.2e8
    out = p.get_fundamentals(["IBM", "NOPE"], AS_OF)
    assert out.loc["IBM", F.SHARES_OUTSTANDING] == pytest.approx(9.2e8)
    assert out.loc["NOPE"].isna().all()
    assert any("no RIC for 1 ticker(s) (NOPE)" in w for w in p.warnings)


@pytest.mark.parametrize("ric, ticker", [("AAPL.O", "AAPL"), ("IBM.N", "IBM"), ("BRKb.N", "BRK-B"), ("BFb.N", "BF-B"),
                                         (".SPX", ".SPX"), ("XYZ.N^K20", "XYZ"), ("AAPL", "AAPL"), ("SPY.P", "SPY")])
def test_ric_to_ticker(ric, ticker):
    assert ric_to_ticker(ric) == ticker


# =============================================================================================
# Provider: snapshots
# =============================================================================================


def test_fundamentals_units_preflight_and_field_drop(fake):
    p = _provider(rics={"AAA": "AAA.N", "BBB": "BBB.O", "LATE": "LATE.N"})
    v = fake.values
    v[("IBM.N", "TR.RevenueActValue(Period=FQ-4)")] = 1.5e10  # preflight passes
    v[("IBM.N", "TR.FreeCashFlow(Period=LTM,Scale=6)")] = None  # preflight fails -> leg not evaluated
    for ric, rev, rev4, sh, rd in [("AAA.N", 1.2e9, 1.0e9, 5e8, "2026-07-30"), ("BBB.O", None, 2.0e9, "", "2026-08-05"),
                                   ("LATE.N", 3e9, 2e9, 1e9, "2026-10-15")]:
        v[(ric, "TR.RevenueActValue(Period=FQ0)")] = rev
        v[(ric, "TR.RevenueActValue(Period=FQ-4)")] = rev4
        v[(ric, "TR.SharesOutstanding")] = sh
        # TR.RevenueActReportDate deliberately unknown at first: dropped silently by the fake
    df = p.get_fundamentals(["AAA", "BBB", "LATE"], AS_OF)
    assert list(df.columns) == F.FUNDAMENTAL_COLUMNS
    assert df.loc["AAA", F.REVENUE_LAST_Q] == pytest.approx(1.2e9)
    assert df.loc["AAA", F.REVENUE_LAST_Q_PRIOR_YEAR] == pytest.approx(1.0e9)
    assert math.isnan(df.loc["BBB", F.REVENUE_LAST_Q]) and math.isnan(df.loc["BBB", F.SHARES_OUTSTANDING])
    assert df[F.FCF_TTM].isna().all() and df[F.REVENUE_TTM].isna().all()
    assert df[F.REPORT_DATE].isna().all() and str(df[F.REPORT_DATE].dtype) == "datetime64[ns]"
    assert any("LEG NOT EVALUATED" in w and "fcf_ttm" in w for w in p.warnings)
    assert any("re-requested one field at a time" in w for w in p.warnings)
    assert any("TR.RevenueActReportDate" in w and "no column" in w for w in p.warnings)
    batched = [c for c in fake.call_list("get_data") if len(c["fields"]) > 1]
    assert batched[0]["universe"] == ["AAA.N", "BBB.O", "LATE.N"]
    assert batched[0]["fields"] == ["TR.RevenueActValue(Period=FQ0)", "TR.RevenueActValue(Period=FQ-4)",
                                    "TR.SharesOutstanding", "TR.RevenueActReportDate"]
    assert batched[0]["parameters"] == {"Curn": "USD", "SDate": "2026-10-01"}
    assert p.field_log["fundamentals.fcf_ttm"]["used"] is False
    assert p.field_log["fundamentals.revenue_last_q_prior_year"]["preflight"] is True
    # with the report date known, a report after as_of blanks the row (look-ahead guard)
    for ric, rd in [("AAA.N", "2026-07-30"), ("BBB.O", "2026-08-05"), ("LATE.N", "2026-10-15")]:
        v[(ric, "TR.RevenueActReportDate")] = rd
    v[("AAA.N", "TR.FreeCashFlow(Period=LTM,Scale=6)")] = 1.0
    df2 = p.get_fundamentals(["AAA", "BBB", "LATE"], AS_OF)
    assert df2.loc["AAA", F.REPORT_DATE] == pd.Timestamp("2026-07-30")
    assert df2.loc["LATE"].isna().all()
    assert any("look-ahead" in w for w in p.warnings)


def test_fcf_scale_six_converts_to_usd(fake):
    fake.values[("IBM.N", "TR.FreeCashFlow(Period=LTM,Scale=6)")] = 12000.0
    fake.values[("AAA.N", "TR.FreeCashFlow(Period=LTM,Scale=6)")] = 1234.5
    fake.values[("AAA.N", "TR.RevenueActReportDate")] = "2026-07-30T00:00:00Z"
    p = _provider(rics={"AAA": "AAA.N"})
    df = p.get_fundamentals(["AAA"], AS_OF)
    assert df.loc["AAA", F.FCF_TTM] == pytest.approx(1.2345e9)
    assert df.loc["AAA", F.REPORT_DATE] == pd.Timestamp("2026-07-30")


def test_short_interest_codes_anchor_absolute_dates(fake):
    v = fake.values
    for code, val in {"TR.ShortInterest(SDate=2026-10-01)": 1e6, "TR.ShortInterest(SDate=2026-09-01)": 8e5,
                      "TR.SharesFreeFloat(SDate=2026-10-01)": 2e7, "TR.ShortInterest(SDate=2026-10-01).date": "2026-09-15"}.items():
        v[("IBM.N", code)] = val
        v[("AAA.N", code)] = val
    p = _provider(rics={"AAA": "AAA.N"})
    df = p.get_short_interest(["AAA"], AS_OF)
    assert list(df.columns) == F.SHORT_INTEREST_COLUMNS
    assert df.loc["AAA", F.SHORT_INTEREST_SHARES] == 1e6 and df.loc["AAA", F.SHORT_INTEREST_SHARES_1M_AGO] == 8e5
    assert df.loc["AAA", F.FLOAT_SHARES] == 2e7 and df.loc["AAA", F.SI_SETTLEMENT_DATE] == pd.Timestamp("2026-09-15")
    req = [c for c in fake.call_list("get_data") if c["universe"] == ["AAA.N"]][0]
    assert req["fields"] == ["TR.ShortInterest(SDate=2026-10-01)", "TR.ShortInterest(SDate=2026-09-01)",
                             "TR.SharesFreeFloat(SDate=2026-10-01)", "TR.ShortInterest(SDate=2026-10-01).date"]
    assert all("0D" not in f for c in fake.call_list("get_data") for f in c["fields"])


def test_estimates_unverified_units_not_served(fake):
    fake.values[("AAA.N", "TR.EPSActSurprise")] = 12.0
    fake.values[("IBM.N", "TR.RevenueMeanEstimate(Period=NTM,Scale=6,Curn=USD)")] = 64000.0
    fake.values[("AAA.N", "TR.RevenueMeanEstimate(Period=NTM,Scale=6,Curn=USD)")] = 5000.0
    p = _provider(rics={"AAA": "AAA.N"})
    df = p.get_estimates(["AAA"], AS_OF)
    assert df.loc["AAA", F.REVENUE_NTM_EST] == pytest.approx(5e9)
    assert math.isnan(df.loc["AAA", F.LAST_EPS_SURPRISE])
    assert not any("TR.EPSActSurprise" in c["fields"] for c in fake.call_list("get_data"))
    assert any("units UNVERIFIED" in w for w in p.warnings)
    admitted = _provider(rics={"AAA": "AAA.N"}, ld_module=fake,
                         fieldmap={"raw": {"estimates": {"last_eps_surprise": {"units_verified": True}}}})
    assert admitted.get_estimates(["AAA"], AS_OF).loc["AAA", F.LAST_EPS_SURPRISE] == pytest.approx(0.12)


def test_options_iv_from_atmiv_rics(fake):
    call, put = ("TR.30DAYATTHEMONEYIMPLIEDVOLATILITYINDEXFORCALLOPTIONS",
                 "TR.30DAYATTHEMONEYIMPLIEDVOLATILITYINDEXFORPUTOPTIONS")
    fake.values[("WMTATMIV.U", call)] = 24.0
    fake.values[("WMTATMIV.U", put)] = 26.0
    fake.values[("AAPLATMIV.U", call)] = None
    fake.values[("AAPLATMIV.U", put)] = 30.0
    p = _provider(rics={"WMT": "WMT.N", "AAPL": "AAPL.O", "ZZZ": "ZZZ.N"})
    df = p.get_options_summary(["WMT", "AAPL", "ZZZ"], AS_OF)
    assert df.loc["WMT", F.IV_30D_ATM] == pytest.approx(0.25)
    assert df.loc["AAPL", F.IV_30D_ATM] == pytest.approx(0.30)
    assert math.isnan(df.loc["ZZZ", F.IV_30D_ATM]) and df[F.PUT_VOLUME].isna().all()
    req = fake.call_list("get_data")[-1]
    assert req["universe"] == ["AAPLATMIV.U", "WMTATMIV.U", "ZZZATMIV.U"] and req["fields"] == [call, put]


# =============================================================================================
# Provider: prices
# =============================================================================================


def _history(ld: FakeLD, ric: str, base: float, with_ohlc: bool = True, n: int = 10) -> None:
    idx = _days(n)
    close = pd.Series(base + np.arange(n, dtype=float), index=idx)
    ld.history[(ric, "TRDPRC_1")] = close
    ld.history[(ric, "ACVOL_UNS")] = pd.Series(1e6, index=idx)
    if with_ohlc:
        ld.history[(ric, "OPEN_PRC")] = close - 0.5
        ld.history[(ric, "HIGH_1")] = close + 1
        ld.history[(ric, "LOW_1")] = close - 1


def test_price_history_multi_ric_request_and_layout(fake):
    for ric, base in [("IBM.N", 250.0), ("AAA.N", 10.0), ("BBB.O", 20.0)]:
        _history(fake, ric, base)
    fake.history[("BBB.O", "TRDPRC_1")].iloc[3] = np.nan
    p = _provider(rics={"AAA": "AAA.N", "BBB": "BBB.O"})
    panel = p.get_price_history(["AAA", "BBB", "NOPE"], date(2026, 9, 1), AS_OF)
    main = [c for c in fake.call_list("get_history") if c["universe"] == ["AAA.N", "BBB.O"]]
    assert main == [{"universe": ["AAA.N", "BBB.O"], "fields": ["TRDPRC_1", "ACVOL_UNS", "OPEN_PRC", "HIGH_1", "LOW_1"],
                     "interval": "daily", "start": "2026-09-01", "end": "2026-10-01",
                     "adjustments": ["exchangeCorrection", "manualCorrection", "CCH", "CRE", "RTS", "RPO"]}]
    assert panel.tickers == ["AAA", "BBB", "NOPE"]
    assert panel.close.index.name == "date" and len(panel.close) == 10
    assert panel.close["AAA"].iloc[-1] == 19.0 and panel.high["BBB"].iloc[0] == 21.0
    assert math.isnan(panel.close["BBB"].iloc[3]) and panel.close["NOPE"].isna().all()
    assert panel.volume["AAA"].iloc[0] == 1e6
    assert any("UNVERIFIED" in w and "adjustment" in w for w in p.warnings)
    assert any("NOPE" in w for w in p.warnings)


def test_price_history_single_ric_flat_columns_and_failed_ohlc_preflight(fake):
    _history(fake, "IBM.N", 250.0, with_ohlc=False)
    _history(fake, "AAA.N", 10.0, with_ohlc=False)
    p = _provider(rics={"AAA": "AAA.N"})
    panel = p.get_price_history(["AAA"], date(2026, 9, 1), AS_OF)
    assert panel.close["AAA"].tolist() == [10.0 + i for i in range(10)]
    assert panel.high["AAA"].isna().all() and panel.open["AAA"].isna().all()
    assert fake.call_list("get_history")[-1]["fields"] == ["TRDPRC_1", "ACVOL_UNS"]
    assert any("LEG NOT EVALUATED" in w and "HIGH_1" in w for w in p.warnings)


def test_price_history_no_data_raises(fake):
    _history(fake, "IBM.N", 250.0)
    with pytest.raises(ProviderError, match="No LSEG price history"):
        _provider(rics={"AAA": "AAA.N"}).get_price_history(["AAA"], date(2026, 9, 1), AS_OF)
    with pytest.raises(ProviderError, match="No RIC"):
        _provider().get_price_history(["AAA"], date(2026, 9, 1), AS_OF)


def test_benchmark_history_and_fallback(fake):
    _history(fake, "IBM.N", 250.0)
    fake.history[(".SPX", "TRDPRC_1")] = pd.Series(5000.0, index=_days())
    p = _provider()
    s = p.get_benchmark_history(date(2026, 9, 1), AS_OF)
    assert s.name == ".SPX" and len(s) == 10 and s.iloc[0] == 5000.0
    del fake.history[(".SPX", "TRDPRC_1")]
    fake.history[("SPY.P", "TRDPRC_1")] = pd.Series(500.0, index=_days())
    s2 = _provider().get_benchmark_history(date(2026, 9, 1), AS_OF)
    assert s2.name == "SPY.P" and s2.iloc[-1] == 500.0
    with pytest.raises(ProviderError):
        _provider().get_benchmark_history(date(2026, 9, 1), AS_OF, symbol=".NDX")


def test_crosscheck_adjustment_flags_unadjusted_history(fake):
    _history(fake, "IBM.N", 250.0)
    _history(fake, "AAA.N", 10.0)
    _history(fake, "BBB.O", 20.0)
    fake.values[("AAA.N", "TR.Price52WeekHigh")] = 20.0  # local high = 19 + 1 = 20
    fake.values[("BBB.O", "TR.Price52WeekHigh")] = 60.0  # local high 30 vs vendor 60 -> flagged
    p = _provider(rics={"AAA": "AAA.N", "BBB": "BBB.O"})
    out = p.crosscheck_adjustment(["AAA", "BBB"], AS_OF)
    assert out.loc["AAA", "ratio"] == pytest.approx(1.0) and not out.loc["AAA", "flagged"]
    assert out.loc["BBB", "ratio"] == pytest.approx(0.5) and out.loc["BBB", "flagged"]


# =============================================================================================
# Provider: documents
# =============================================================================================


def test_news_headlines_and_story_bodies(fake):
    hl = pd.DataFrame({"headline": ["IBM beats", "IBM old", "IBM no body"],
                       "storyId": ["urn:1", "urn:0", "urn:2"], "sourceCode": ["NS:RTRS", "NS:RTRS", "NS:RTRS"]},
                      index=pd.DatetimeIndex(["2026-09-20T13:00:00Z", "2026-07-01T09:00:00Z", "2026-09-25T10:30:00Z"],
                                             name="versionCreated"))
    fake.headlines["R:IBM.N"] = hl
    fake.stories["urn:1"] = "<p>IBM <b>beat</b> estimates &amp; raised guidance.</p><script>x()</script>"
    p = _provider(rics={"IBM": "IBM.N"})
    docs = p.get_documents("IBM", {DocumentKind.NEWS, DocumentKind.TRANSCRIPT, DocumentKind.FILING},
                           date(2026, 9, 1), AS_OF, limit=5)
    req = fake.call_list("get_headlines")[0]
    assert req == {"query": "R:IBM.N", "start": datetime(2026, 9, 1), "end": datetime(2026, 10, 1, 23, 59, 59, 999999),
                   "count": 5}
    assert [d.doc_id for d in docs] == ["lseg:news:urn:2", "lseg:news:urn:1"]  # newest first, in window
    d1 = docs[1]
    assert d1.text == "IBM beat estimates & raised guidance." and d1.kind == DocumentKind.NEWS
    assert d1.published_at == datetime(2026, 9, 20, 13, 0) and d1.source == "LSEG News (NS:RTRS)"
    assert d1.metadata["licence_class"] == "L3" and d1.metadata["ric"] == "IBM.N"
    assert docs[0].text == "IBM no body" and docs[0].metadata["text_unavailable"] == "true"
    assert any("StreetEvents" in w for w in p.warnings)
    assert any("SEC EDGAR" in w for w in p.warnings)
    assert not p.boundary.filter_documents(docs)[0]  # default boundary withholds all LSEG text
    assert p.get_documents("IBM", {DocumentKind.TRANSCRIPT}, date(2026, 9, 1), AS_OF) == []


def test_filings_enabled_by_override(fake):
    class Definition:
        seen: list[dict] = []

        def __init__(self, **kw):
            Definition.seen.append(kw)

        def get_data(self):
            df = pd.DataFrame({"DocumentTitle": ["10-Q Q2 2026", "10-K 2024"], "FilingDate": ["2026-08-01", "2025-02-01"],
                               "Filename": ["ecp_1", "ecp_0"]})
            return SimpleNamespace(data=SimpleNamespace(df=df))

    filings = SimpleNamespace(Feed=SimpleNamespace(EDGAR="EDGAR-ENUM"), search=SimpleNamespace(Definition=Definition))
    fake.content = SimpleNamespace(filings=filings)
    over = {"documents": {"filings": {"enabled": True, "search_parameters": {"query": "{ric}", "limit": 3}}}}
    p = _provider(rics={"IBM": "IBM.N"}, fieldmap=over)
    from aitrading.data.base import Capability as C
    assert C.FILINGS in p.capabilities
    docs = p.get_documents("IBM", {DocumentKind.FILING}, date(2026, 1, 1), AS_OF)
    assert Definition.seen == [{"feed": "EDGAR-ENUM", "query": "IBM.N", "limit": 3}]
    assert [d.title for d in docs] == ["10-Q Q2 2026"] and docs[0].metadata["filename"] == "ecp_1"


# =============================================================================================
# Field self-check
# =============================================================================================


def test_verify_fields_reports_each_mapped_field(fake):
    _history(fake, "IBM.N", 250.0, with_ohlc=False)
    fake.values[("IBM.N", "TR.FreeCashFlow(Period=LTM,Scale=6)")] = 11800.0
    fake.values[("IBM.N", "TR.PricePctChg52WkHigh")] = -12.5
    p = _provider()
    checks = p.verify_fields(as_of=AS_OF)
    assert all(isinstance(c, FieldCheck) for c in checks)
    by = {(c.field, c.code): c for c in checks}
    mc = by[("universe.market_cap", "TR.CompanyMarketCap(Scale=6)")]
    assert mc.ok and mc.status_in_map == "confirmed" and mc.returned_value == 230000.0
    assert "canonical 230000000000.0" in mc.note
    mcf = by[("feature.market_cap_usd_bn", "TR.CompanyMarketCap(Scale=6)")]
    assert mcf.ok and "canonical 230.0" in mcf.note  # USD mn / 1000 = catalog USD bn
    fcf = by[("fundamentals.fcf_ttm", "TR.FreeCashFlow(Period=LTM,Scale=6)")]
    assert fcf.ok and fcf.status_in_map == "unverifiable"
    dd = by[("feature.drawdown_from_52w_high_pct", "TR.PricePctChg52WkHigh")]
    assert dd.ok and "units UNVERIFIED" in dd.note
    si = by[("short_interest.short_interest_shares", "TR.ShortInterest(SDate=2026-10-01)")]
    assert not si.ok and si.returned_value is None and "no column" in si.note
    assert by[("history.close", "TRDPRC_1")].ok and by[("history.close", "TRDPRC_1")].returned_value == 259.0
    assert not by[("history.high", "HIGH_1")].ok
    iv = [c for c in checks if c.field == "options.iv_30d_atm"]
    assert len(iv) == 2 and not any(c.ok for c in iv)
    singles = [c for c in fake.call_list("get_data")]
    assert all(len(c["fields"]) == 1 for c in singles)  # every field requested alone
    assert {c["universe"] for c in singles} == {"IBM.N", "IBMATMIV.U"}
    assert singles[0]["parameters"] == {"Curn": "USD", "SDate": "2026-10-01"}
    with pytest.raises(ProviderError, match="RIC"):
        p.verify_fields("IBM")  # not a RIC and not seen this session


def test_compile_screen_loads_default_fieldmap_when_none(monkeypatch):
    monkeypatch.delenv(FIELDMAP_ENV, raising=False)
    cq = compile_lseg.compile_screen(_spec([]), None, AS_OF, today=TODAY)
    assert cq.expression == LISTING_EXPR
