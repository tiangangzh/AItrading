"""Bloomberg adapter, BQL push-down compiler and field-map loader - fully offline.

Fake ``bql`` and ``blpapi`` modules stand in for the vendor SDKs (injected through ``sys.modules`` or
the constructor). The tests pin the exact BQL strings and blpapi request payloads the adapter
generates against the grammar in docs/VENDOR_REFERENCE.md section 1, plus unit conversions, NaN
handling, look-ahead guards, the licensing boundary and ProviderUnavailable when an SDK is absent.
"""

from __future__ import annotations

import ast
import inspect
import json
import re
import sys
import types
from datetime import date
from typing import Any

import numpy as np
import pandas as pd
import pytest

from aitrading.core import fields as F
from aitrading.core.models import DocumentKind
from aitrading.core.policy import DataBoundary
from aitrading.data import bloomberg as bbg
from aitrading.data.base import (
    Capability,
    MarketDataProvider,
    ProviderError,
    ProviderUnavailable,
    PushdownResult,
    ScreenPushdown,
)
from aitrading.data.bloomberg import BloombergProvider, FieldCheck, to_canonical_ticker, to_vendor_id
from aitrading.data.fieldmaps import (
    PUSHABLE_STATUSES,
    STATUSES,
    FieldMapError,
    env_var,
    iter_statuses,
    load_fieldmap,
    merge,
)
from aitrading.screen import compile_bql
from aitrading.screen.compile_bql import (
    CompileError,
    bql_number,
    compile_screen,
    feature_pushability,
    render_template,
)
from aitrading.screen.spec import Condition, RankFactor, ScreenSpec, UniverseSpec

AS_OF = date(2026, 10, 1)
TODAY = date(2026, 10, 2)
IBM = "IBM US Equity"


# =============================================================================================
# Fake SDKs
# =============================================================================================


class FakeBQL:
    """Fake BQuant: ``bql.Service().execute(query)`` + ``bql.combined_df(resp)``.

    ``db`` maps a rendered BQL item (or a ``prefix*``) to ``{security: scalar | pd.Series | Exception}``
    or to an Exception (the whole request fails, like an invalid item). ``universe`` maps the exact
    ``for(...)`` clause of a screening request to the securities it returns.
    """

    def __init__(self, db: dict[str, Any] | None = None, universe: dict[str, list[str]] | None = None,
                 hash_headers: bool = True):
        self.db = dict(db or {})
        self.universe = dict(universe or {})
        self.queries: list[str] = []
        self.hash_headers = hash_headers
        mod = types.ModuleType("bql")
        mod.Service = lambda: self  # type: ignore[attr-defined]
        mod.combined_df = lambda resp: resp  # type: ignore[attr-defined]
        self.module = mod

    def _lookup(self, expr: str) -> Any:
        if expr in self.db:
            return self.db[expr]
        for k, v in self.db.items():
            if k.endswith("*") and expr.startswith(k[:-1]):
                return v
        raise RuntimeError(f"BQL error: unknown item {expr}")

    def execute(self, query: str) -> pd.DataFrame:
        self.queries.append(query)
        m = re.fullmatch(r"let\((.*)\) get\((.*)\) for\((.*)\)", query)
        assert m, f"malformed BQL: {query}"
        lets: dict[str, str] = {}
        for part in m.group(1).split(";"):
            part = part.strip()
            if part:
                name, _, expr = part.partition("=")
                assert name.startswith("#"), part
                lets[name[1:]] = expr
        gets = [g.strip().lstrip("#") for g in m.group(2).split(",")]
        assert gets == list(lets), (gets, list(lets))
        for_clause = m.group(3)
        if for_clause.startswith("["):
            ids = re.findall(r"'([^']*)'", for_clause)
        else:
            assert for_clause in self.universe, f"unexpected universe clause: {for_clause}"
            ids = self.universe[for_clause]
        rows: dict[tuple[str, Any], dict[str, Any]] = {}
        for var, expr in lets.items():
            data = self._lookup(expr)
            if isinstance(data, Exception):
                raise data
            for sec in ids:
                v = data.get(sec, np.nan)
                if isinstance(v, Exception):
                    raise v
                col = f"#{var}" if self.hash_headers else var.upper()
                if isinstance(v, pd.Series):
                    for d, x in v.items():
                        rows.setdefault((sec, pd.Timestamp(d)), {})[col] = x
                else:
                    rows.setdefault((sec, None), {})[col] = v
        recs = []
        for (sec, d), vals in rows.items():
            rec = {"ID": sec, **vals}
            if d is not None:
                rec["DATE"] = d
            recs.append(rec)
        df = pd.DataFrame(recs)
        if df.empty:
            df = pd.DataFrame(columns=["ID", *[f"#{v}" for v in lets]])
        return df.set_index("ID")


class _Msg:
    def __init__(self, d: dict[str, Any]):
        self._d = d

    def toPy(self) -> dict[str, Any]:  # noqa: N802 - blpapi naming
        return self._d


class _Event:
    def __init__(self, etype: int, msgs: list[dict[str, Any]]):
        self._t = etype
        self._msgs = [_Msg(m) for m in msgs]

    def eventType(self) -> int:  # noqa: N802
        return self._t

    def __iter__(self):
        return iter(self._msgs)


class FakeBlp:
    """Fake ``blpapi`` module + Desktop API session.

    ``bdp``: code -> {security: value}; ``bdp_errors``: code -> errorInfo for every security, or
    code -> {security: errorInfo}; ``bdh``: code -> {security: {date: value}}; ``beqs``: screen ->
    securities; ``reject``: request element names that make ``fromPy`` raise.
    """

    TIMEOUT, PARTIAL, RESPONSE = 10, 6, 5

    def __init__(self, bdp=None, bdp_errors=None, bdh=None, beqs=None, reject=(), timeout=False, start_ok=True):
        self.bdp = dict(bdp or {})
        self.bdp_errors = dict(bdp_errors or {})
        self.bdh = dict(bdh or {})
        self.beqs = dict(beqs or {})
        self.reject = set(reject)
        self.timeout = timeout
        self.start_ok = start_ok
        self.requests: list[tuple[str, dict[str, Any]]] = []
        self.services: list[str] = []
        self.options: dict[str, Any] = {}
        self._queue: list[_Event] = []
        fake = self

        class SessionOptions:
            def setServerHost(self, h):  # noqa: N802
                fake.options["host"] = h

            def setServerPort(self, p):  # noqa: N802
                fake.options["port"] = p

        mod = types.ModuleType("blpapi")
        mod.SessionOptions = SessionOptions  # type: ignore[attr-defined]
        mod.Session = lambda opts: fake  # type: ignore[attr-defined]
        mod.Event = types.SimpleNamespace(TIMEOUT=self.TIMEOUT, PARTIAL_RESPONSE=self.PARTIAL, RESPONSE=self.RESPONSE)  # type: ignore[attr-defined]
        self.module = mod

    # session API
    def start(self) -> bool:
        return self.start_ok

    def openService(self, name: str) -> bool:  # noqa: N802
        self.services.append(name)
        return True

    def getService(self, name: str):  # noqa: N802
        fake = self

        class Request:
            def __init__(self, rtype: str):
                self.rtype = rtype
                self.payload: dict[str, Any] | None = None

            def fromPy(self, d):  # noqa: N802
                bad = fake.reject & set(d)
                if bad:
                    raise ValueError(f"Invalid element: {sorted(bad)}")
                self.payload = d

        class Service:
            def createRequest(self, rtype):  # noqa: N802
                return Request(rtype)

        assert name == "//blp/refdata"
        return Service()

    def sendRequest(self, req) -> None:  # noqa: N802
        self.requests.append((req.rtype, req.payload))
        if self.timeout:
            self._queue.append(_Event(self.TIMEOUT, []))
            return
        msgs = getattr(self, f"_{req.rtype}")(req.payload)
        first, rest = msgs[:1], msgs[1:]
        self._queue.append(_Event(self.PARTIAL, [{"sessionStatus": "noise"}, *first]))
        self._queue.append(_Event(self.RESPONSE, rest))

    def nextEvent(self, timeout_ms: int) -> _Event:  # noqa: N802
        return self._queue.pop(0)

    def stop(self) -> None:
        pass

    # responses
    def _ReferenceDataRequest(self, p):  # noqa: N802
        out = []
        for sec in p["securities"]:
            fd, fe = {}, []
            for code in p["fields"]:
                err = self.bdp_errors.get(code)
                if isinstance(err, dict) and sec in err:
                    err = err[sec]
                elif isinstance(err, dict) and "category" not in err:
                    err = None
                if err:
                    fe.append({"fieldId": code, "errorInfo": err})
                elif sec in self.bdp.get(code, {}):
                    fd[code] = self.bdp[code][sec]
            out.append({"securityData": [{"security": sec, "fieldData": fd, "fieldExceptions": fe}]})
        return out

    def _HistoricalDataRequest(self, p):  # noqa: N802
        out = []
        start, end = pd.Timestamp(p["startDate"]), pd.Timestamp(p["endDate"])
        for sec in p["securities"]:
            dates = sorted({pd.Timestamp(d) for c in p["fields"] for d in self.bdh.get(c, {}).get(sec, {})})
            rows = []
            for d in dates:
                if start <= d <= end:
                    row = {"date": d.date()}
                    for c in p["fields"]:
                        v = self.bdh.get(c, {}).get(sec, {}).get(d.strftime("%Y-%m-%d"))
                        if v is not None:
                            row[c] = v
                    rows.append(row)
            out.append({"securityData": {"security": sec, "fieldData": rows, "fieldExceptions": []}})
        return out

    def _BeqsRequest(self, p):  # noqa: N802
        secs = self.beqs.get(p["screenName"], [])
        return [{"data": {"securityData": [{"security": s, "fieldData": {}} for s in secs]}}]


def _spec(conditions: list[Condition], universe: UniverseSpec | None = None, any_of=None) -> ScreenSpec:
    return ScreenSpec(
        name="t", observation="o", universe=universe or UniverseSpec(), conditions=conditions,
        any_of=any_of or [], ranking=[RankFactor(feature="rsi_14", direction="lower_is_better")], top_n=15,
    )


def representative_spec() -> ScreenSpec:
    """The VENDOR_REFERENCE representative screen in catalog units."""
    return _spec([
        Condition(feature="market_cap_usd_bn", op="between", value=2, value_high=20),
        Condition(feature="sma_50", op=">", other_feature="sma_200"),
        Condition(feature="drawdown_from_52w_high_pct", op="between", value=-45, value_high=-20),
        Condition(feature="rel_volume_5d", op=">", value=1.5),
        Condition(feature="rsi_14", op="<", value=45),
        Condition(feature="fcf_yield_pct", op=">=", value=5),
        Condition(feature="revenue_growth_yoy_pct", op=">=", value=10),
        Condition(feature="short_interest_pct_float", op=">=", value=5),
    ])


PX_2Y = "px_last(dates=range('2024-10-01','2026-10-01'), fill='prev', ca_adj='full')"
REPRESENTATIVE_QUERY = (
    "let("
    "#price=px_last(dates='2026-10-01', fill='prev', ca_adj='full'); "
    "#market_cap_usd_bn=cur_mkt_cap(currency='USD', dates='2026-10-01', fill='prev'); "
    f"#sma_50=smavg({PX_2Y}, period=50).last('1'); "
    f"#sma_200=smavg({PX_2Y}, period=200).last('1'); "
    "#drawdown_from_52w_high_pct=px_last(dates='2026-10-01', fill='prev')/max(px_high(dates=range('2025-10-01','2026-10-01'), fill='prev')) - 1; "
    "#rel_volume_5d=px_volume(dates=range('2026-09-17','2026-10-01')).dropna().last(5).avg()"
    "/px_volume(dates=range('2026-06-28','2026-10-01')).dropna().last(60).avg(); "
    f"#rsi_14=rsi({PX_2Y}, period=14).last('1');"
    ") get(#price, #market_cap_usd_bn, #sma_50, #sma_200, #drawdown_from_52w_high_pct, #rel_volume_5d, #rsi_14) "
    "for(filter(filter(equitiesuniv(['ACTIVE','PRIMARY']), "
    "cntry_of_risk()=='US' and #market_cap_usd_bn >= 2000000000 and #market_cap_usd_bn <= 20000000000), "
    "#price >= 5 and #sma_50 > #sma_200 and #drawdown_from_52w_high_pct >= -0.45 and "
    "#drawdown_from_52w_high_pct <= -0.2 and #rel_volume_5d > 1.5 and #rsi_14 < 45))"
)


@pytest.fixture(autouse=True)
def _no_env_override(monkeypatch):
    monkeypatch.delenv(env_var("bloomberg"), raising=False)


@pytest.fixture
def fm() -> dict[str, Any]:
    return load_fieldmap("bloomberg")


def bql_provider(fake: FakeBQL, blp: FakeBlp | None = None, **kw) -> BloombergProvider:
    kw.setdefault("today", lambda: TODAY)
    if blp is not None:
        kw.setdefault("blpapi_module", blp.module)
    return BloombergProvider(backend="bql", bql_module=fake.module, **kw)


def blp_provider(blp: FakeBlp, **kw) -> BloombergProvider:
    kw.setdefault("today", lambda: TODAY)
    return BloombergProvider(backend="blpapi", blpapi_module=blp.module, **kw)


# =============================================================================================
# Field-map loader (generic)
# =============================================================================================


def test_default_bloomberg_map_loads_and_every_status_is_valid(fm):
    assert fm["vendor"] == "bloomberg"
    assert fm["_sources"][0].endswith("bloomberg.json")
    statuses = [v for _, _, v in iter_statuses(fm)]
    assert statuses and set(statuses) <= set(STATUSES)


def test_map_transcribes_reference_statuses(fm):
    feats, raw = fm["features"], fm["raw"]
    assert feats["market_cap_usd_bn"]["status"] == "confirmed"
    assert feats["market_cap_usd_bn"]["threshold_scale"] == 1e9
    assert feats["drawdown_from_52w_high_pct"]["threshold_scale"] == 0.01
    assert feats["sma_50"]["status"] == "corrected"
    assert feats["rsi_14"]["status"] == "corrected"
    assert feats["high_52w"]["status"] == "confirmed"
    assert feats["ev_to_ebitda"]["status"] == "corrected"
    assert feats["fcf_yield_pct"]["status"] == "unverifiable"
    assert feats["short_interest_pct_float"]["status"] == "unverifiable"
    assert feats["short_interest_pct_float"]["expression"] is None
    assert raw["short_interest"]["short_interest_shares"]["bdp"]["code"] == "SHORT_INT"
    assert raw["short_interest"]["short_interest_shares"]["bdp"]["status"] == "unverifiable"
    assert raw["estimates"]["eps_ntm_est"]["bdp"] == {**raw["estimates"]["eps_ntm_est"]["bdp"],
                                                      "code": "BEST_EPS", "overrides": {"BEST_FPERIOD_OVERRIDE": "1BF"},
                                                      "status": "confirmed"}
    assert raw["prices"]["close"]["bql"]["expression"] == "px_last(dates=range('{start}','{end}'), fill='prev', ca_adj='full')"
    assert raw["universe"]["market_cap"]["bdp"]["code"] == "CUR_MKT_CAP"
    assert fm["universe"]["base"]["expression"] == "equitiesuniv(['ACTIVE','PRIMARY'])"
    # the semantic substitution FY1-forward growth for trailing growth is refused
    assert "sales_growth" not in feats["revenue_growth_yoy_pct"]["expression"]


def test_env_override_is_deep_merged(tmp_path, monkeypatch):
    p = tmp_path / "bbg.json"
    p.write_text(json.dumps({
        "features": {"fcf_yield_pct": {"status": "confirmed", "units_verified": True, "threshold_scale": 0.01},
                     "rel_strength_3m_pp": None},
        "test_security": "MSFT US Equity",
    }))
    monkeypatch.setenv("AITRADING_FIELDMAP_BLOOMBERG", str(p))
    fm = load_fieldmap("bloomberg")
    assert fm["features"]["fcf_yield_pct"]["status"] == "confirmed"
    assert fm["features"]["fcf_yield_pct"]["expression"].startswith("free_cash_flow_yield(")  # kept
    assert "rel_strength_3m_pp" not in fm["features"]  # null deletes
    assert fm["test_security"] == "MSFT US Equity"
    assert fm["_sources"][1] == str(p)
    # explicit override wins over the env file
    fm2 = load_fieldmap("bloomberg", {"test_security": "AAPL US Equity"})
    assert fm2["test_security"] == "AAPL US Equity" and fm2["_sources"][-1] == "override:<mapping>"


def test_fieldmap_errors(tmp_path, monkeypatch):
    with pytest.raises(FieldMapError, match="status 'admitted'"):
        load_fieldmap("bloomberg", {"features": {"rsi_14": {"status": "admitted"}}})
    with pytest.raises(FieldMapError, match="expected 'bloomberg'"):
        load_fieldmap("bloomberg", {"vendor": "lseg"})
    monkeypatch.setenv("AITRADING_FIELDMAP_BLOOMBERG", str(tmp_path / "missing.json"))
    with pytest.raises(FieldMapError, match="not found"):
        load_fieldmap("bloomberg")
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    monkeypatch.setenv("AITRADING_FIELDMAP_BLOOMBERG", str(bad))
    with pytest.raises(FieldMapError, match="not valid JSON"):
        load_fieldmap("bloomberg")
    with pytest.raises(FieldMapError):
        load_fieldmap("no-such-vendor-xyz")


def test_merge_semantics():
    base = {"a": {"x": 1, "y": [1, 2]}, "b": 2}
    out = merge(base, {"a": {"y": [3], "z": None}, "b": None, "c": {"k": 1}})
    assert out == {"a": {"x": 1, "y": [3]}, "c": {"k": 1}}
    assert base == {"a": {"x": 1, "y": [1, 2]}, "b": 2}  # not mutated


def test_no_unverifiable_vendor_code_is_hard_coded_in_logic(fm):
    """ADR rule: an unverifiable field is configuration only - never a string literal in adapter logic."""
    unverifiable: set[str] = set()

    def walk(node):
        if isinstance(node, dict):
            if node.get("status") == "unverifiable":
                for k in ("expression", "code"):
                    if isinstance(node.get(k), str):
                        unverifiable.add(node[k].split("(")[0])
            for v in node.values():
                walk(v)

    walk(fm)
    assert {"SHORT_INT", "EQY_FLOAT", "SHORT_INT_DT", "free_cash_flow_yield", "SECURITY_TYP"} <= unverifiable
    for module in (bbg, compile_bql):
        tree = ast.parse(inspect.getsource(module))
        docstrings = {id(n.body[0].value) for n in ast.walk(tree)
                      if isinstance(n, (ast.Module, ast.ClassDef, ast.FunctionDef)) and n.body
                      and isinstance(n.body[0], ast.Expr) and isinstance(n.body[0].value, ast.Constant)}
        literals = [n.value for n in ast.walk(tree)
                    if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in docstrings]
        for code in unverifiable:
            hits = [s for s in literals if re.search(rf"(?<![A-Za-z_]){re.escape(code)}(?![A-Za-z_])", s)]
            assert not hits, f"{module.__name__} hard-codes unverifiable item {code!r}: {hits}"


# =============================================================================================
# BQL push-down compiler
# =============================================================================================


def test_compile_representative_screen_exact_query(fm):
    cq = compile_screen(representative_spec(), fm, AS_OF)
    assert cq.query == REPRESENTATIVE_QUERY
    assert [c.feature for c in cq.pushed] == ["market_cap_usd_bn", "sma_50", "drawdown_from_52w_high_pct",
                                             "rel_volume_5d", "rsi_14"]
    assert [c.feature for c in cq.residual] == ["fcf_yield_pct", "revenue_growth_yoy_pct", "short_interest_pct_float"]
    assert "unverifiable" in cq.reasons["fcf_yield_pct >= 5"]
    assert "pushdown disabled" in cq.reasons["revenue_growth_yoy_pct >= 10"]
    assert "no BQL expression" in cq.reasons["short_interest_pct_float >= 5"]
    assert cq.universe_conditions[0].startswith("country == US")
    assert cq.universe_conditions[1] == "price >= 5 (UniverseSpec.min_price)"
    assert set(cq.universe_residual) == {"security_type in [common_stock]", "avg_dollar_volume_20d_usd_mn >= 5"}
    assert cq.as_of == AS_OF and len(cq.sha256) == 64 and len(cq.fieldmap_digest) == 64
    assert any("G5" in w for w in cq.warnings)
    assert any(w.startswith("#rsi_14") and "VERIFY shape" in w for w in cq.warnings)
    assert any("today's universe" in w for w in cq.warnings)


def test_only_confirmed_or_corrected_features_are_ever_pushable(fm):
    seen = 0
    for name, entry in fm["features"].items():
        ok, why = feature_pushability(name, fm)
        if entry.get("status") not in PUSHABLE_STATUSES or not entry.get("expression"):
            assert not ok, name
            seen += 1
        if entry.get("units_verified") is False or entry.get("pushdown") is False:
            assert not ok, name
    assert seen >= 8
    pushable = {n for n in fm["features"] if feature_pushability(n, fm)[0]}
    assert {"market_cap_usd_bn", "sma_50", "sma_200", "drawdown_from_52w_high_pct", "rsi_14", "rel_volume_5d",
            "price", "gics_sector"} <= pushable
    assert not pushable & {"fcf_yield_pct", "short_interest_pct_float", "revenue_growth_yoy_pct", "iv_30d_pct",
                           "rel_strength_3m_pp", "gics_industry", "exchange", "days_to_cover"}


def test_compiled_query_obeys_grammar_rules(fm):
    q = compile_screen(representative_spec(), fm, AS_OF).query
    assert "'0D'" not in q.upper() and "-1Y" not in q.upper() and "0d" not in q  # absolute dates only
    assert re.findall(r"'(\d{4}-\d{2}-\d{2})'", q)
    assert "'2B'" not in q and "2e9" not in q  # plain-number literals
    for banned in ("grouprank", "groupzscore", "groupsort", "top(", "bqlsvc", "with("):
        assert banned not in q
    # static predicates in the inner filter, time-series studies in the outer one
    inner = q[q.index("filter(filter(") : q.index("), #price >= 5")]
    assert "#market_cap_usd_bn" in inner and "#rsi_14" not in inner and "#sma_50" not in inner
    assert q.count("filter(") == 2 and q.startswith("let(") and " get(" in q and " for(" in q


@pytest.mark.parametrize("cond,fragment", [
    (Condition(feature="market_cap_usd_bn", op=">", value=2.5), "#market_cap_usd_bn > 2500000000"),
    (Condition(feature="market_cap_usd_bn", op="<=", value=0.3), "#market_cap_usd_bn <= 300000000"),
    (Condition(feature="drawdown_from_52w_high_pct", op="<=", value=-20), "#drawdown_from_52w_high_pct <= -0.2"),
    (Condition(feature="return_3m_pct", op=">", value=12.5), "#return_3m_pct > 0.125"),
    (Condition(feature="rsi_14", op="<", value=30), "#rsi_14 < 30"),
    (Condition(feature="target_price_upside_pct", op=">=", value=15), "#target_price_upside_pct >= 0.15"),
    (Condition(feature="sma_50_vs_sma_200_pct", op=">", value=0), "#sma_50_vs_sma_200_pct > 0"),
    (Condition(feature="operating_margin_pct", op="between", value=10, value_high=35),
     "#operating_margin_pct >= 0.1 and #operating_margin_pct <= 0.35"),
    (Condition(feature="sma_50", op=">", other_feature="sma_200", multiplier=1.02), "#sma_50 > 1.02*#sma_200"),
    (Condition(feature="price_vs_sma_50_pct", op="<", other_feature="price_vs_sma_200_pct", multiplier=0.5),
     "#price_vs_sma_50_pct < 0.5*#price_vs_sma_200_pct"),
    (Condition(feature="gics_sector", op="in", values=["health care"]), "#gics_sector=='Health Care'"),
    (Condition(feature="gics_sector", op="==", values=["Utilities"]), "#gics_sector=='Utilities'"),
])
def test_threshold_scaling_into_vendor_units(fm, cond, fragment):
    cq = compile_screen(_spec([cond], UniverseSpec(min_price=None, min_avg_dollar_volume_usd_mn=None,
                                                   security_types=[])), fm, AS_OF)
    assert cq.pushed == [cond], cq.reasons
    assert fragment in cq.query


def test_market_cap_literal_is_exact_against_cur_mkt_cap(fm):
    cq = compile_screen(_spec([Condition(feature="market_cap_usd_bn", op=">=", value=2)]), fm, AS_OF)
    assert cq.lets["market_cap_usd_bn"] == "cur_mkt_cap(currency='USD', dates='2026-10-01', fill='prev')"
    assert "#market_cap_usd_bn >= 2000000000" in cq.query
    assert bql_number(0.1 + 0.2) == "0.30000000000000004"  # never rounds a threshold silently
    assert bql_number(2e10) == "20000000000" and bql_number(-0.45) == "-0.45"


@pytest.mark.parametrize("cond,reason", [
    (Condition(feature="rsi_14", op="==", value=30), "tolerance"),
    (Condition(feature="rsi_14", op="!=", value=30), "tolerance"),
    (Condition(feature="gics_sector", op="not_in", values=["Energy"]), "evaluated locally"),
    (Condition(feature="gics_sector", op="in", values=["Energy", "Utilities"]), "multi-label"),
    (Condition(feature="gics_sector", op="in", values=["Health Care') or x==('"]), "allowed_values"),
    (Condition(feature="gics_industry", op="in", values=["Banks"]), "unverifiable"),
    (Condition(feature="golden_cross_20d", op="==", value=1), "no Bloomberg mapping"),
    (Condition(feature="volatility_20d_pct", op=">", value=30), "units unverified"),
    (Condition(feature="iv_30d_pct", op=">", value=30), "units unverified"),
    (Condition(feature="rel_strength_3m_pp", op=">", value=0), "pushdown disabled"),
    (Condition(feature="revenue_growth_ntm_est_pct", op=">", value=10), "pushdown disabled"),
    (Condition(feature="sma_50", op=">", other_feature="avg_dollar_volume_20d_usd_mn"), "other feature"),
    (Condition(feature="not_a_feature", op=">", value=1), "unknown feature"),
])
def test_residual_conditions_and_reasons(fm, cond, reason):
    cq = compile_screen(_spec([cond, Condition(feature="rsi_14", op="<", value=50)]), fm, AS_OF)
    assert cond in cq.residual
    assert reason in cq.reasons[cond.describe()]
    assert "Health Care')" not in cq.query


def test_any_of_groups_and_universe_exclusions_stay_local(fm):
    group = [Condition(feature="rsi_14", op="<", value=30), Condition(feature="return_1m_pct", op="<", value=-10)]
    spec = _spec([Condition(feature="market_cap_usd_bn", op=">", value=1)],
                 UniverseSpec(exclude_sectors=["Utilities"]), any_of=[group])
    cq = compile_screen(spec, fm, AS_OF)
    assert cq.residual_groups == [group]
    assert "#rsi_14" not in cq.query and "#return_1m_pct" not in cq.query
    assert any(r.startswith("any of: rsi_14 < 30 | return_1m_pct < -10") for r in cq.residual_descriptions)
    assert "gics_sector not in [Utilities]" in cq.universe_residual
    assert "OR groups" in cq.reasons["any of: rsi_14 < 30 | return_1m_pct < -10"]


def test_short_interest_is_filtered_before_any_ranking_when_mapped_confirmed(fm):
    """An admitted short-interest item is pushed INSIDE filter(); the query never ranks or truncates."""
    override = {"features": {"short_interest_pct_float": {
        "expression": "admitted_si_item(dates='{as_of}')", "status": "confirmed", "threshold_scale": 0.01,
        "catalog_unit": "%", "stage": "series"}}}
    fm2 = merge(fm, override)
    cq = compile_screen(representative_spec(), fm2, AS_OF)
    assert "short_interest_pct_float" in [c.feature for c in cq.pushed]
    assert "#short_interest_pct_float=admitted_si_item(dates='2026-10-01');" in cq.query
    outer = cq.query[cq.query.index("), #price >= 5") :]
    assert "#short_interest_pct_float >= 0.05" in outer and outer.endswith("))")
    assert "group" not in cq.query.lower() and "top" not in cq.query.lower()


def test_admitted_set_enforces_gate_g5(fm):
    cq = compile_screen(representative_spec(), fm, AS_OF, admitted={"market_cap_usd_bn"})
    assert [c.feature for c in cq.pushed] == ["market_cap_usd_bn"]
    assert "field-admission log" in cq.reasons["rsi_14 < 45"]
    assert "UniverseSpec.min_price" not in " ".join(cq.universe_conditions)


def test_dependency_on_unverifiable_helper_blocks_push(fm):
    fm2 = merge(fm, {"helpers": {"px_2y": {"status": "unverifiable"}}})
    ok, why = feature_pushability("sma_50", fm2)
    assert not ok and "#px_2y" in why
    ok, why = feature_pushability("market_cap_usd_bn", fm2)
    assert ok and why == ""


def test_catalog_unit_drift_is_refused(fm):
    fm2 = merge(fm, {"features": {"market_cap_usd_bn": {"catalog_unit": "USD mn"}}})
    ok, why = feature_pushability("market_cap_usd_bn", fm2)
    assert not ok and "catalog_unit" in why


def test_universe_expr_members_for_backtests(fm):
    cq = compile_screen(_spec([Condition(feature="rsi_14", op="<", value=30)], UniverseSpec(country="", min_price=None)),
                        fm, AS_OF, universe_expr="members('RAY Index', dates='{as_of}')")
    assert cq.query.endswith("for(filter(members('RAY Index', dates='2026-10-01'), #rsi_14 < 30))")
    assert not any("today's universe" in w for w in cq.warnings)


def test_probe_item_when_only_the_universe_is_pushed(fm):
    cq = compile_screen(_spec([Condition(feature="fcf_yield_pct", op=">", value=5)],
                              UniverseSpec(min_price=None, min_avg_dollar_volume_usd_mn=None)), fm, AS_OF)
    assert cq.query == ("let(#probe=cur_mkt_cap(currency='USD', dates='2026-10-01', fill='prev');) get(#probe) "
                        "for(filter(equitiesuniv(['ACTIVE','PRIMARY']), cntry_of_risk()=='US'))")
    assert cq.pushed == []


def test_nothing_to_wrap_equitiesuniv_raises(fm):
    spec = _spec([Condition(feature="fcf_yield_pct", op=">", value=5)],
                 UniverseSpec(country="", min_price=None, min_avg_dollar_volume_usd_mn=None))
    with pytest.raises(CompileError, match="wrapped in filter"):
        compile_screen(spec, fm, AS_OF)


def test_bad_country_is_never_interpolated(fm):
    cq = compile_screen(_spec([Condition(feature="rsi_14", op="<", value=30)], UniverseSpec(country="U'S")), fm, AS_OF)
    assert "U'S" not in cq.query and "country == U'S" in cq.universe_residual


def test_render_template_placeholders():
    assert render_template("a='{as_of}' b='{d365}' c='{as_of_1y}' d='{d91}'", date(2024, 2, 29)) == \
        "a='2024-02-29' b='2023-03-01' c='2023-02-28' d='2023-11-30'"
    assert render_template("range('{start}','{end}') {benchmark}", AS_OF, start=date(2026, 1, 2),
                           end=AS_OF, benchmark="SPX Index") == "range('2026-01-02','2026-10-01') SPX Index"
    with pytest.raises(FieldMapError):
        render_template("x({mystery})", AS_OF)
    with pytest.raises(FieldMapError):
        render_template("x('{start}')", AS_OF)


# =============================================================================================
# Provider: construction, boundary, SDK availability
# =============================================================================================


def test_ticker_helpers():
    assert to_canonical_ticker("AAPL US Equity") == "AAPL"
    assert to_canonical_ticker("BRK/B UN Equity", ("US", "UN")) == "BRK-B"
    assert to_canonical_ticker("VOD LN Equity") == "VOD.LN"
    assert to_canonical_ticker("SPX Index") == "SPX Index"
    assert to_vendor_id("AAPL") == "AAPL US Equity"
    assert to_vendor_id("brk.b") == "BRK/B US Equity" == to_vendor_id("BRK-B")
    assert to_vendor_id("VOD.LN") == "VOD LN Equity"
    assert to_vendor_id("SPX Index") == "SPX Index"
    assert to_vendor_id("IBM US Equity") == "IBM US Equity"


def test_default_boundary_denies_text_and_values_until_g1():
    p = BloombergProvider(bql_module=FakeBQL().module, today=lambda: TODAY)
    b = p.boundary
    assert b.provider == "bloomberg" and b.allowed_document_kinds == set() and b.allow_numeric_features is False
    assert "G1" in b.note and "L4" in b.note
    assert p.get_documents("AAPL", {DocumentKind.TRANSCRIPT, DocumentKind.NEWS}, date(2026, 1, 1), AS_OF) == []
    assert any("gate G1" in w for w in p.warnings)
    assert Capability.TRANSCRIPTS not in p.capabilities and Capability.NEWS not in p.capabilities


def test_wider_boundary_still_has_no_verified_text_api():
    wide = DataBoundary(provider="bloomberg", allowed_document_kinds={DocumentKind.TRANSCRIPT}, note="G1 cleared (test)")
    p = BloombergProvider(bql_module=FakeBQL().module, boundary=wide, today=lambda: TODAY)
    assert any("wider than the ADR-001 default" in w for w in p.warnings)
    with pytest.raises(NotImplementedError, match="transcript: No BQL/BDP item"):
        p.get_documents("AAPL", {DocumentKind.TRANSCRIPT}, date(2026, 1, 1), AS_OF)
    assert p.get_documents("AAPL", {DocumentKind.NEWS}, date(2026, 1, 1), AS_OF) == []  # news still denied
    with pytest.raises(ValueError):
        BloombergProvider(boundary=DataBoundary(provider="lseg"))


def test_protocols_and_capabilities():
    p = BloombergProvider(bql_module=FakeBQL().module, today=lambda: TODAY)
    assert isinstance(p, MarketDataProvider) and isinstance(p, ScreenPushdown)
    assert p.name == "bloomberg"
    assert {Capability.PRICES, Capability.FUNDAMENTALS, Capability.ESTIMATES, Capability.SHORT_INTEREST,
            Capability.SCREEN_PUSHDOWN} <= p.capabilities
    q = BloombergProvider(backend="blpapi", blpapi_module=FakeBlp().module, today=lambda: TODAY)
    assert Capability.SCREEN_PUSHDOWN not in q.capabilities and Capability.PRICES in q.capabilities
    with pytest.raises(ValueError):
        BloombergProvider(backend="bqlsvc")


def test_bql_absent_raises_provider_unavailable(monkeypatch):
    monkeypatch.setitem(sys.modules, "bql", None)
    p = BloombergProvider(today=lambda: TODAY)
    with pytest.raises(ProviderUnavailable, match="BQuant"):
        p.get_universe(UniverseSpec(), AS_OF)
    with pytest.raises(ProviderUnavailable):
        p.pushdown_screen(representative_spec(), AS_OF)
    with pytest.raises(ProviderUnavailable):
        p.verify_fields()
    assert any("BQuant" in d for d in p.diagnostics())


def test_blpapi_absent_raises_provider_unavailable(monkeypatch):
    monkeypatch.setitem(sys.modules, "blpapi", None)
    p = BloombergProvider(backend="blpapi", tickers=["AAPL"], today=lambda: TODAY)
    with pytest.raises(ProviderUnavailable, match=re.escape("blpapi.bloomberg.com/repository/releases/python/simple/")):
        p.get_short_interest(["AAPL"], AS_OF)
    with pytest.raises(ProviderUnavailable):
        p.get_price_history(["AAPL"], date(2026, 9, 1), AS_OF)
    assert p.diagnostics()


def test_sdks_are_imported_lazily_through_sys_modules(monkeypatch):
    fake = FakeBQL({"cur_mkt_cap(currency='USD', dates='2026-10-01', fill='prev')": {"AAPL US Equity": 3.5e12}})
    monkeypatch.setitem(sys.modules, "bql", fake.module)
    p = BloombergProvider(today=lambda: TODAY, preflight=False,
                          fieldmap={"raw": {"universe": {"name": None, "gics_sector": None, "gics_industry": None, "exchange": None}}})
    fake.universe["filter(equitiesuniv(['ACTIVE','PRIMARY']), cntry_of_risk()=='US')"] = ["AAPL US Equity"]
    df = p.get_universe(UniverseSpec(), AS_OF)
    assert list(df.index) == ["AAPL"] and df.loc["AAPL", F.MARKET_CAP] == 3.5e12


def test_desktop_session_start_failure():
    blp = FakeBlp(start_ok=False)
    p = blp_provider(blp, tickers=["AAPL"])
    with pytest.raises(ProviderUnavailable, match="localhost:8194"):
        p.get_short_interest(["AAPL"], AS_OF)


# =============================================================================================
# Provider: BQL backend
# =============================================================================================

UNIVERSE_CLAUSE = "filter(equitiesuniv(['ACTIVE','PRIMARY']), cntry_of_risk()=='US')"
MCAP = "cur_mkt_cap(currency='USD', dates='2026-10-01', fill='prev')"


def universe_fake(**overrides) -> FakeBQL:
    ids = ["AAPL UW Equity", "BRK/B UN Equity", "XYZ US Equity", "VOD LN Equity"]
    db: dict[str, Any] = {
        "name()": {IBM: "IBM", "AAPL UW Equity": "Apple Inc", "BRK/B UN Equity": "Berkshire Hathaway",
                   "VOD LN Equity": "Vodafone"},
        "gics_sector_name()": {"AAPL UW Equity": "Information Technology", "BRK/B UN Equity": "Financials",
                               "VOD LN Equity": "Communication Services"},
        "gics_industry_name()": {IBM: "IT Services", "AAPL UW Equity": "Technology Hardware, Storage & Peripherals"},
        "exch_code()": {IBM: "UN", "AAPL UW Equity": "UW", "BRK/B UN Equity": "UN", "XYZ US Equity": "ZZ"},
        "cur_mkt_cap(*": {"AAPL UW Equity": 3.5e12, "BRK/B UN Equity": "1.0e12", "XYZ US Equity": np.inf,
                          "VOD LN Equity": 4e10},
    }
    db.update(overrides)
    return FakeBQL(db, {UNIVERSE_CLAUSE: ids})


def test_get_universe_bql_exact_query_and_canonical_tickers():
    fake = universe_fake()
    p = bql_provider(fake)
    df = p.get_universe(UniverseSpec(), AS_OF)
    bulk = fake.queries[-1]
    assert bulk == ("let(#name=name(); #gics_sector=gics_sector_name(); #gics_industry=gics_industry_name(); "
                    f"#exchange=exch_code(); #market_cap={MCAP};) "
                    "get(#name, #gics_sector, #gics_industry, #exchange, #market_cap) "
                    f"for({UNIVERSE_CLAUSE})")
    # unverifiable items were preflighted alone on the test security first
    assert fake.queries[:3] == [f"let(#v={item};) get(#v) for(['{IBM}'])"
                                for item in ("name()", "gics_industry_name()", "exch_code()")]
    assert list(df.columns) == F.UNIVERSE_COLUMNS
    assert list(df.index) == ["AAPL", "BRK-B", "XYZ"]  # VOD: listing country LN is not US -> dropped
    assert df.index.name == "ticker"
    assert df.loc["AAPL", F.VENDOR_ID] == "AAPL UW Equity" and df.loc["BRK-B", F.VENDOR_ID] == "BRK/B UN Equity"
    assert df.loc["AAPL", F.EXCHANGE] == "NASDAQ" and df.loc["BRK-B", F.EXCHANGE] == "NYSE"
    assert df.loc["XYZ", F.EXCHANGE] == "ZZ"  # unmapped codes pass through
    assert (df[F.COUNTRY] == "US").all() and (df[F.CURRENCY] == "USD").all()
    assert (df[F.SECURITY_TYPE] == "common_stock").all()
    assert any("labelled 'common_stock'" in w for w in p.warnings)
    assert df.loc["AAPL", F.MARKET_CAP] == 3.5e12 and df.loc["BRK-B", F.MARKET_CAP] == 1.0e12
    assert np.isnan(df.loc["XYZ", F.MARKET_CAP])  # non-finite -> NaN, never 0
    assert pd.isna(df.loc["XYZ", F.NAME]) and pd.isna(df.loc["XYZ", F.GICS_INDUSTRY])
    assert df[F.MARKET_CAP].dtype == "float64"
    # later requests reuse the vendor id
    assert p._vendor_id("BRK-B") == "BRK/B UN Equity"


def test_universe_preflight_failure_suspends_leg():
    fake = universe_fake(**{"gics_industry_name()": RuntimeError("Field not valid")})
    p = bql_provider(fake)
    df = p.get_universe(UniverseSpec(), AS_OF)
    assert "gics_industry_name()" not in fake.queries[-1]
    assert df[F.GICS_INDUSTRY].isna().all()
    assert any(w.startswith("LEG NOT EVALUATED: bloomberg gics_industry") for w in p.warnings)


def test_universe_security_type_filter_and_historical_warning():
    fake = universe_fake()
    p = bql_provider(fake)
    df = p.get_universe(UniverseSpec(security_types=["adr"]), date(2025, 6, 30))
    assert df.empty
    assert any("survivorship" in w for w in p.warnings)


def test_universe_expr_placeholder(monkeypatch):
    clause = "filter(members('RAY Index', dates='2026-10-01'), cntry_of_risk()=='US')"
    fake = FakeBQL({MCAP: {"AAPL US Equity": 3e12}}, {clause: ["AAPL US Equity"]})
    p = bql_provider(fake, universe_expr="members('RAY Index', dates='{as_of}')", preflight=False,
                     fieldmap={"raw": {"universe": {"name": None, "gics_sector": None, "gics_industry": None,
                                                    "exchange": None}}})
    df = p.get_universe(UniverseSpec(), AS_OF)
    assert fake.queries[-1].endswith(f"for({clause})")
    assert list(df.index) == ["AAPL"]


def _series(values: dict[str, float]) -> pd.Series:
    return pd.Series(values)


PX_DATES = ["2026-09-25", "2026-09-28", "2026-09-29", "2026-09-30", "2026-10-01"]


def price_fake() -> FakeBQL:
    def ser(vals):
        return pd.Series(dict(zip(PX_DATES, vals)))

    nan = np.nan
    one = {IBM: pd.Series({"2026-10-01": 1.0})}
    return FakeBQL({
        "px_open(*": {**one, "AAPL US Equity": ser([10, 11, 11, 12, 13]), "MSFT US Equity": ser([20, 21, 21, 22, 23])},
        "px_high(*": {**one, "AAPL US Equity": ser([11, 12, 12, 13, 14]), "MSFT US Equity": ser([21, 22, 22, 23, 24])},
        "px_low(*": {**one, "AAPL US Equity": ser([9, 10, 10, 11, 12]), "MSFT US Equity": ser([19, 20, 20, 21, 22])},
        "px_last(*": {**one, "AAPL US Equity": ser([10.5, 11.5, 11.5, 12.5, 13.5]),
                      "MSFT US Equity": ser([20.5, 21.5, 21.5, nan, 23.5])},
        # 2026-09-29 is a "holiday": prices filled ('prev'), no volume for anyone
        "px_volume(*": {**one, "AAPL US Equity": ser([100, 110, nan, 130, 140]),
                        "MSFT US Equity": ser([200, nan, nan, 230, 240])},
    })


def test_price_history_bql_query_adjustment_and_sessions():
    fake = price_fake()
    p = bql_provider(fake)
    panel = p.get_price_history(["AAPL", "MSFT"], date(2026, 9, 25), AS_OF)
    bulk = fake.queries[-1]
    rng = "dates=range('2026-09-25','2026-10-01')"
    assert bulk == (f"let(#open=px_open({rng}, fill='prev', ca_adj='full'); #high=px_high({rng}, fill='prev', ca_adj='full'); "
                    f"#low=px_low({rng}, fill='prev', ca_adj='full'); #close=px_last({rng}, fill='prev', ca_adj='full'); "
                    f"#volume=px_volume({rng});) get(#open, #high, #low, #close, #volume) "
                    "for(['AAPL US Equity','MSFT US Equity'])")
    assert len(fake.queries) == 4  # three preflights (px_open/px_high/px_low are unverifiable) + one bulk
    assert list(panel.close.columns) == ["AAPL", "MSFT"]
    assert [d.strftime("%Y-%m-%d") for d in panel.close.index] == ["2026-09-25", "2026-09-28", "2026-09-30", "2026-10-01"]
    assert panel.close.loc["2026-10-01", "AAPL"] == 13.5
    assert np.isnan(panel.close.loc["2026-09-30", "MSFT"])  # missing stays NaN
    assert np.isnan(panel.volume.loc["2026-09-28", "MSFT"])
    assert panel.volume.loc["2026-10-01", "MSFT"] == 240
    assert panel.high.loc["2026-09-25", "MSFT"] == 21
    for f in (panel.open, panel.high, panel.low, panel.close, panel.volume):
        assert (f.dtypes == "float64").all()


def test_price_history_preflight_failure_keeps_close():
    fake = price_fake()
    fake.db["px_open(*"] = RuntimeError("unknown item px_open")
    p = bql_provider(fake)
    panel = p.get_price_history(["AAPL"], date(2026, 9, 25), AS_OF)
    assert panel.open["AAPL"].isna().all() and panel.close["AAPL"].notna().any()
    assert "px_open" not in fake.queries[-1]
    assert any("LEG NOT EVALUATED: bloomberg open" in w for w in p.warnings)


def test_price_history_no_data_raises():
    fake = price_fake()
    p = bql_provider(fake, preflight=False)
    with pytest.raises(ProviderError, match="no prices"):
        p.get_price_history(["NOPE"], date(2026, 9, 25), AS_OF)


def test_benchmark_history_bql():
    fake = FakeBQL({
        "px_last(*": {"SPX Index": pd.Series({"2026-09-30": 6000.0, "2026-10-01": 6010.0})},
        "px_volume(*": {"SPX Index": pd.Series({"2026-09-30": 1e9, "2026-10-01": 1.1e9})},
    })
    p = bql_provider(fake)
    s = p.get_benchmark_history(date(2026, 9, 30), AS_OF)
    assert s.name == "SPX Index" and list(s.values) == [6000.0, 6010.0]
    assert fake.queries[-1].endswith("for(['SPX Index'])")


def test_bql_response_without_requested_column_raises():
    class Broken(FakeBQL):
        def execute(self, query):
            self.queries.append(query)
            return pd.DataFrame({"OTHER": [1.0]}, index=pd.Index(["AAPL US Equity"], name="ID"))

    p = bql_provider(Broken(), preflight=False)
    with pytest.raises(ProviderError, match="lacks column"):
        p.get_fundamentals(["AAPL"], AS_OF)


def test_fundamentals_bql_point_in_time_and_secondary_bdp_unavailable(monkeypatch):
    monkeypatch.setitem(sys.modules, "blpapi", None)
    a = "AAPL US Equity"
    fake = FakeBQL({
        "sales_rev_turn(fpt='LTM', dates='2026-10-01', fill='prev')": {a: 4.0e11},
        "sales_rev_turn(fpt='LTM', dates='2025-10-01', fill='prev')": {a: 3.6e11},
        "is_oper_inc(fpt='LTM', dates='2026-10-01', fill='prev')": {a: 1.2e11},
        "is_oper_inc(fpt='LTM', dates='2025-10-01', fill='prev')": {a: np.nan},
        "ebitda(fpt='LTM', dates='2026-10-01', fill='prev')": {a: "N/A"},
    })
    p = bql_provider(fake)
    df = p.get_fundamentals(["AAPL", "MSFT"], AS_OF)
    assert fake.queries == [(
        "let(#revenue_ttm=sales_rev_turn(fpt='LTM', dates='2026-10-01', fill='prev'); "
        "#revenue_ttm_prior_year=sales_rev_turn(fpt='LTM', dates='2025-10-01', fill='prev'); "
        "#operating_income_ttm=is_oper_inc(fpt='LTM', dates='2026-10-01', fill='prev'); "
        "#operating_income_ttm_prior_year=is_oper_inc(fpt='LTM', dates='2025-10-01', fill='prev'); "
        "#ebitda_ttm=ebitda(fpt='LTM', dates='2026-10-01', fill='prev');) "
        "get(#revenue_ttm, #revenue_ttm_prior_year, #operating_income_ttm, #operating_income_ttm_prior_year, #ebitda_ttm) "
        "for(['AAPL US Equity','MSFT US Equity'])")]
    assert list(df.columns) == F.FUNDAMENTAL_COLUMNS and list(df.index) == ["AAPL", "MSFT"]
    assert df.loc["AAPL", F.REVENUE_TTM] == 4.0e11 and df.loc["AAPL", F.REVENUE_TTM_PRIOR_YEAR] == 3.6e11
    assert np.isnan(df.loc["AAPL", F.EBITDA_TTM])  # 'N/A' -> NaN
    assert df.loc["MSFT"].drop([F.PERIOD_END, F.REPORT_DATE]).isna().all()
    assert df[F.FCF_TTM].isna().all()  # units unverified: not served
    assert any("fcf_ttm" in w and "unverified units" in w for w in p.warnings)
    assert df[F.REPORT_DATE].isna().all() and df[F.REPORT_DATE].dtype == "datetime64[ns]"
    assert any(w.startswith("LEG NOT EVALUATED: bloomberg report_date") and "Desktop API" in w for w in p.warnings)
    assert "LEG NOT EVALUATED: bloomberg operating_income_ttm_prior_year came back missing for every requested security" \
        in p.warnings
    assert "LEG NOT EVALUATED: bloomberg ebitda_ttm came back missing for every requested security" in p.warnings


def test_estimates_bql_plus_bdp_secondary_channel():
    a, m = "AAPL US Equity", "MSFT US Equity"
    fake = FakeBQL({
        "sales_rev_turn(fpt='BT', fpo='1', dates='2026-10-01', fill='prev')": {a: 4.2e11, m: 3.0e11},
        "sales_rev_turn(fpt='BT', fpo='1', dates='2026-07-02', fill='prev')": {a: 4.1e11, m: 2.9e11},
        "best_target_price(dates='2026-10-01', fill='prev')": {a: 250.0, m: 520.0},
    })
    blp = FakeBlp(
        bdp={"BEST_EPS": {a: 7.5, m: 14.0, IBM: 11.0},
             "LATEST_ANNOUNCEMENT_DT": {a: "2026-07-30", m: date(2026, 10, 2), IBM: "2026-07-20"},
             "EXPECTED_REPORT_DT": {a: "2026-10-29", m: "2026-10-27", IBM: "2026-10-20"}},
    )
    p = bql_provider(fake, blp)
    df = p.get_estimates(["AAPL", "MSFT"], AS_OF)
    assert fake.queries[-1].endswith("for(['AAPL US Equity','MSFT US Equity'])")
    assert "#revenue_ntm_est_3m_ago=sales_rev_turn(fpt='BT', fpo='1', dates='2026-07-02', fill='prev');" in fake.queries[-1]
    ref = [pl for t, pl in blp.requests if t == "ReferenceDataRequest" and pl["securities"] != [IBM]]
    assert {"securities": [a, m], "fields": ["BEST_EPS"],
            "overrides": [{"fieldId": "BEST_FPERIOD_OVERRIDE", "value": "1BF"}]} in ref
    assert {"securities": [a, m], "fields": ["LATEST_ANNOUNCEMENT_DT", "EXPECTED_REPORT_DT"], "overrides": []} in ref
    # unverifiable BDP fields were preflighted alone on the test security
    assert ("ReferenceDataRequest", {"securities": [IBM], "fields": ["LATEST_ANNOUNCEMENT_DT"], "overrides": []}) in blp.requests
    assert blp.services == ["//blp/refdata"] and blp.options == {"host": "localhost", "port": 8194}
    assert df.loc["AAPL", F.EPS_NTM_EST] == 7.5 and df.loc["MSFT", F.REVENUE_NTM_EST] == 3.0e11
    assert df.loc["AAPL", F.TARGET_PRICE_MEAN] == 250.0
    assert df.loc["AAPL", F.LAST_EARNINGS_DATE] == pd.Timestamp("2026-07-30")
    assert pd.isna(df.loc["MSFT", F.LAST_EARNINGS_DATE])  # after as_of: look-ahead guard
    assert df.loc["MSFT", F.NEXT_EARNINGS_DATE] == pd.Timestamp("2026-10-27")  # forward dates are kept
    assert any("look-ahead guard" in w for w in p.warnings)
    assert np.isnan(df.loc["AAPL", F.NUM_ANALYSTS])
    assert any(q.startswith("ReferenceDataRequest ") for q in p.query_log)


# =============================================================================================
# Provider: blpapi backend
# =============================================================================================


def test_short_interest_blpapi_requests_guards_and_units():
    a, m, x = "AAPL US Equity", "MSFT US Equity", "XYZ US Equity"
    blp = FakeBlp(bdp={
        "SHORT_INT": {IBM: 1e7, a: 1.2e8, m: 5.0e7, x: 9e6},
        "EQY_FLOAT": {IBM: 900.0, a: 15000.0, m: 7400.0, x: 50.0},
        "SHORT_INT_DT": {IBM: "2026-09-15", a: "2026-09-15", m: "2026-09-15", x: "2026-10-15"},
    })
    p = blp_provider(blp)
    df = p.get_short_interest(["AAPL", "MSFT", "XYZ"], AS_OF)
    bulk = [pl for t, pl in blp.requests if t == "ReferenceDataRequest" and pl["securities"] != [IBM]]
    # EQY_FLOAT has unverified units: not requested, not served
    assert bulk == [{"securities": [a, m, x], "fields": ["SHORT_INT", "SHORT_INT_DT"], "overrides": []}]
    assert list(df.columns) == F.SHORT_INTEREST_COLUMNS
    assert df.loc["AAPL", F.SHORT_INTEREST_SHARES] == 1.2e8
    assert df.loc["AAPL", F.SI_SETTLEMENT_DATE] == pd.Timestamp("2026-09-15")
    assert df[F.FLOAT_SHARES].isna().all() and df[F.SHORT_INTEREST_SHARES_1M_AGO].isna().all()
    assert df.loc["XYZ"].isna().all()  # settles after as_of: whole row blanked
    assert any("settle after as_of" in w for w in p.warnings)
    assert any("float_shares" in w and "unverified units" in w for w in p.warnings)


def test_short_interest_float_scale_guards_after_admission_override():
    secs = [f"S{i} US Equity" for i in range(3)]
    bdp = {
        "SHORT_INT": {IBM: 1e7, **{s: 2e6 for s in secs}},
        "EQY_FLOAT": {IBM: 900.0, **{s: 40.0 for s in secs}},  # quoted in millions
        "SHORT_INT_DT": {IBM: "2026-09-15", **{s: "2026-09-15" for s in secs}},
    }
    admitted = {"raw": {"short_interest": {"float_shares": {"bdp": {"units_verified": True}}}}}
    # 1) units flagged verified but the scale is wrong: the preflight on IBM catches the implausible value
    p = blp_provider(FakeBlp(bdp=bdp), fieldmap=admitted)
    df = p.get_short_interest(["S0", "S1", "S2"], AS_OF)
    assert df[F.FLOAT_SHARES].isna().all()
    assert any("failed its preflight" in w and "implausible scale" in w for w in p.warnings)
    # 2) without the preflight the bulk values are served, and both scale guards fire
    p1 = blp_provider(FakeBlp(bdp=bdp), fieldmap=admitted, preflight=False)
    df1 = p1.get_short_interest(["S0", "S1", "S2"], AS_OF)
    assert (df1[F.FLOAT_SHARES] == 40.0).all()
    assert any("float_shares looks mis-scaled" in w for w in p1.warnings)
    assert any("probably in millions" in w and "EQY_FLOAT" in w for w in p1.warnings)
    # 3) the admitted override with the right scale
    fixed = {"raw": {"short_interest": {"float_shares": {"bdp": {"units_verified": True, "to_canonical": 1e6}}}}}
    p2 = blp_provider(FakeBlp(bdp=bdp), fieldmap=fixed)
    df2 = p2.get_short_interest(["S0", "S1", "S2"], AS_OF)
    assert (df2[F.FLOAT_SHARES] == 4.0e7).all()
    assert not any("millions" in w or "mis-scaled" in w for w in p2.warnings)


def test_fatal_field_exception_aborts_leg_and_per_security_exception_blanks_cell():
    a, m = "AAPL US Equity", "MSFT US Equity"
    blp = FakeBlp(
        bdp={"SHORT_INT": {IBM: 1e7, a: 1e8, m: 2e8}, "SHORT_INT_DT": {IBM: "2026-09-15", a: "2026-09-15", m: "2026-09-15"}},
        bdp_errors={"SHORT_INT": {m: {"category": "BAD_SEC", "message": "N/A for this security"}}},
    )
    p = blp_provider(blp)
    df = p.get_short_interest(["AAPL", "MSFT"], AS_OF)
    assert df.loc["AAPL", F.SHORT_INTEREST_SHARES] == 1e8 and np.isnan(df.loc["MSFT", F.SHORT_INTEREST_SHARES])
    assert any("unavailable for 1 security" in w for w in p.warnings)

    blp2 = FakeBlp(
        bdp={"SHORT_INT_DT": {IBM: "2026-09-15", a: "2026-09-15"}, "SHORT_INT": {IBM: 1e7, a: 1e8}},
        bdp_errors={"SHORT_INT": {a: {"category": "BAD_FLD", "subcategory": "INVALID_FIELD", "message": "Field not valid"}}},
    )
    p2 = blp_provider(blp2)
    df2 = p2.get_short_interest(["AAPL"], AS_OF)
    assert df2[F.SHORT_INTEREST_SHARES].isna().all()
    assert any(w.startswith("LEG NOT EVALUATED: bloomberg short_interest_shares") for w in p2.warnings)


def test_preflight_failure_on_test_security_skips_bdp_field():
    blp = FakeBlp(
        bdp={"SHORT_INT_DT": {IBM: "2026-09-15", "AAPL US Equity": "2026-09-15"}},
        bdp_errors={"SHORT_INT": {"category": "NO_AUTH", "message": "Not authorized for field"}},
    )
    p = blp_provider(blp)
    df = p.get_short_interest(["AAPL"], AS_OF)
    bulk = [pl for t, pl in blp.requests if pl.get("securities") == ["AAPL US Equity"]]
    assert bulk == [{"securities": ["AAPL US Equity"], "fields": ["SHORT_INT_DT"], "overrides": []}]
    assert df[F.SHORT_INTEREST_SHARES].isna().all()
    assert any("failed its preflight" in w and "Not authorized" in w for w in p.warnings)


def test_bdp_snapshot_is_blank_for_a_historical_as_of():
    blp = FakeBlp(bdp={"SHORT_INT": {"AAPL US Equity": 1e8}})
    p = blp_provider(blp)
    df = p.get_short_interest(["AAPL"], date(2026, 6, 30))
    assert df.isna().all().all() and blp.requests == []
    assert any("current values only" in w for w in p.warnings)


def test_options_iv_needs_admitted_units():
    blp = FakeBlp(bdp={"30DAY_IMPVOL_100.0%MNY_DF": {IBM: 22.0, "AAPL US Equity": 25.0}})
    p = blp_provider(blp)
    df = p.get_options_summary(["AAPL"], AS_OF)
    assert df[F.IV_30D_ATM].isna().all() and not blp.requests
    ok = {"raw": {"options": {"iv_30d_atm": {"bdp": {"units_verified": True}}}}}
    p2 = blp_provider(blp, fieldmap=ok)
    df2 = p2.get_options_summary(["AAPL"], AS_OF)
    assert df2.loc["AAPL", F.IV_30D_ATM] == pytest.approx(0.25)  # vol points -> fraction
    assert df2[F.PUT_VOLUME].isna().all()


def bdh_fake(**kw) -> FakeBlp:
    a = "AAPL US Equity"
    one = {IBM: {"2026-09-30": 1.0}}
    return FakeBlp(bdh={
        "PX_OPEN": {**one, a: {"2026-09-30": 10.0, "2026-10-01": 11.0}},
        "PX_HIGH": {**one, a: {"2026-09-30": 12.0, "2026-10-01": 13.0}},
        "PX_LOW": {**one, a: {"2026-09-30": 9.0, "2026-10-01": 10.0}},
        "PX_LAST": {**one, a: {"2026-09-30": 11.0, "2026-10-01": 12.5}},
        "PX_VOLUME": {**one, a: {"2026-09-30": 1000.0, "2026-10-01": 1100.0}},
    }, **kw)


def test_price_history_blpapi_historical_request():
    blp = bdh_fake()
    p = blp_provider(blp)
    panel = p.get_price_history(["AAPL"], date(2026, 9, 1), AS_OF)
    adj = {"adjustmentNormal": True, "adjustmentAbnormal": True, "adjustmentSplit": True}
    assert blp.requests[-1] == ("HistoricalDataRequest", {
        "securities": ["AAPL US Equity"], "fields": ["PX_OPEN", "PX_HIGH", "PX_LOW", "PX_LAST", "PX_VOLUME"],
        "periodicitySelection": "DAILY", **adj, "startDate": "20260901", "endDate": "20261001"})
    preflights = [pl["fields"] for t, pl in blp.requests[:-1]]
    assert preflights == [["PX_OPEN"], ["PX_HIGH"], ["PX_LOW"]]
    assert panel.close.loc["2026-10-01", "AAPL"] == 12.5 and panel.volume.loc["2026-09-30", "AAPL"] == 1000.0
    assert list(panel.close.index) == [pd.Timestamp("2026-09-30"), pd.Timestamp("2026-10-01")]


def test_price_history_blpapi_retries_without_rejected_adjustment_elements():
    blp = bdh_fake(reject={"adjustmentNormal"})
    p = blp_provider(blp, preflight=False)
    panel = p.get_price_history(["AAPL"], date(2026, 9, 1), AS_OF)
    assert "adjustmentNormal" not in blp.requests[-1][1]
    assert panel.close.loc["2026-10-01", "AAPL"] == 12.5
    assert any("NOT dividend-adjusted" in w for w in p.warnings)


def test_blpapi_timeout_raises_provider_error():
    p = blp_provider(FakeBlp(timeout=True), preflight=False)
    with pytest.raises(ProviderError, match="timed out"):
        p.get_price_history(["AAPL"], date(2026, 9, 1), AS_OF)


def test_blpapi_universe_from_saved_eqs_screen():
    blp = FakeBlp(beqs={"Mid caps": ["AAPL UW Equity", "BRK/B UN Equity"]},
                  bdp={"CUR_MKT_CAP": {"AAPL UW Equity": 3.5e12, "BRK/B UN Equity": 1.0e12},
                       "SECURITY_TYP": {IBM: "Common Stock", "AAPL UW Equity": "Common Stock", "BRK/B UN Equity": "REIT"}})
    p = blp_provider(blp, universe_expr="GLOBAL:Mid caps")
    df = p.get_universe(UniverseSpec(security_types=[]), date(2026, 9, 30))
    assert blp.requests[0] == ("BeqsRequest", {"screenName": "Mid caps", "screenType": "GLOBAL", "asOfDate": "20260930"})
    assert list(df.index) == ["AAPL", "BRK-B"]
    assert df.loc["BRK-B", F.SECURITY_TYPE] == "reit" and df.loc["AAPL", F.SECURITY_TYPE] == "common_stock"
    assert df.loc["AAPL", F.MARKET_CAP] == 3.5e12 and df[F.NAME].isna().all()
    assert any("no blpapi-backend field-map entry for universe column(s) name" in w for w in p.warnings)
    df2 = p.get_universe(UniverseSpec(), date(2026, 9, 30))
    assert list(df2.index) == ["AAPL"]  # default UniverseSpec keeps common stock only


def test_blpapi_universe_needs_tickers_or_screen_and_cannot_push_down():
    p = blp_provider(FakeBlp())
    with pytest.raises(ProviderError, match="cannot screen"):
        p.get_universe(UniverseSpec(), AS_OF)
    with pytest.raises(ProviderError, match="backend='bql'"):
        p.pushdown_screen(representative_spec(), AS_OF)
    assert any("cannot screen" in d for d in p.diagnostics())


# =============================================================================================
# Push-down execution and field self-check
# =============================================================================================


def test_pushdown_screen_runs_one_request_and_returns_canonical_tickers():
    fake = FakeBQL({"*": {"AAPL UW Equity": 1.0, "BRK/B UN Equity": 2.0}},
                   {REPRESENTATIVE_QUERY[REPRESENTATIVE_QUERY.index(" for(") + 5 : -1]: ["AAPL UW Equity", "BRK/B UN Equity"]})
    p = bql_provider(fake)
    res = p.pushdown_screen(representative_spec(), AS_OF)
    assert isinstance(res, PushdownResult)
    assert fake.queries == [REPRESENTATIVE_QUERY] and res.query == REPRESENTATIVE_QUERY
    assert p.query_log == [REPRESENTATIVE_QUERY]
    assert res.tickers == ["AAPL", "BRK-B"]
    assert p._vendor_id("AAPL") == "AAPL UW Equity"
    assert "rsi_14 < 45" in res.pushed_conditions and "market_cap_usd_bn between 2 and 20" in res.pushed_conditions
    assert "short_interest_pct_float >= 5" in res.residual_conditions
    assert "universe: security_type in [common_stock]" in res.residual_conditions
    assert any(c.startswith("universe: country == US") for c in res.pushed_conditions)
    assert p.last_pushdown is not None and p.last_pushdown.query == res.query
    assert any(w.startswith("bloomberg push-down: #rsi_14") for w in p.warnings)


def test_pushdown_empty_result_warns_about_silent_drops():
    clause = REPRESENTATIVE_QUERY[REPRESENTATIVE_QUERY.index(" for(") + 5 : -1]
    p = bql_provider(FakeBQL({"*": {}}, {clause: []}))
    res = p.pushdown_screen(representative_spec(), AS_OF)
    assert res.tickers == [] and any("silently drops" in w for w in p.warnings)


def test_pushdown_compile_failure_is_a_provider_error():
    p = bql_provider(FakeBQL())
    spec = _spec([Condition(feature="fcf_yield_pct", op=">", value=5)],
                 UniverseSpec(country="", min_price=None, min_avg_dollar_volume_usd_mn=None))
    with pytest.raises(ProviderError, match="compile failed"):
        p.pushdown_screen(spec, AS_OF)


def test_verify_fields_reports_each_mapped_item():
    fake = FakeBQL({
        "cntry_of_risk()": {IBM: "US"},
        MCAP: {IBM: 2.2e11},
        "free_cash_flow_yield(dates='2026-10-01', fill='prev')": RuntimeError("Unknown item free_cash_flow_yield"),
        "px_last(dates='2026-10-01', fill='prev', ca_adj='full')": {IBM: 250.0},
        "sales_rev_turn(fpt='LTM', dates='2026-10-01', fill='prev')": {IBM: 6.3e4},  # wrong scale on purpose
        "px_last(dates=range('2026-09-21','2026-10-01'), fill='prev', ca_adj='full')": {
            IBM: pd.Series({"2026-09-30": 249.0, "2026-10-01": 250.0})},
    })
    blp = FakeBlp(bdp={"SHORT_INT": {IBM: 2.1e7}},
                  bdp_errors={"EQY_FLOAT": {"category": "BAD_FLD", "message": "Field not valid"}})
    p = bql_provider(fake, blp)
    checks = p.verify_fields(as_of=AS_OF)
    by = {(c.field, c.channel): c for c in checks}
    assert all(isinstance(c, FieldCheck) for c in checks)
    assert by[("universe.country", "bql")].ok and by[("universe.country", "bql")].returned_value == "US"
    mc = by[("features.market_cap_usd_bn", "bql")]
    assert mc.ok and mc.returned_value == 2.2e11 and mc.status_in_map == "confirmed"
    assert "= 220 USD bn" in mc.note and mc.test_security == IBM and mc.as_of == AS_OF
    assert mc.expression == MCAP
    fcf = by[("features.fcf_yield_pct", "bql")]
    assert not fcf.ok and fcf.status_in_map == "unverifiable" and "Unknown item" in fcf.note
    assert "never a threshold until admitted" in fcf.note
    rev = by[("raw.fundamentals.revenue_ttm", "bql")]
    assert rev.ok and rev.returned_value == 6.3e4 and "canonical 63000" in rev.note  # recorded for the reviewer
    close = by[("raw.prices.close", "bql")]
    assert close.ok and close.returned_value == 250.0
    assert by[("raw.short_interest.short_interest_shares", "bdp")].ok
    flt = by[("raw.short_interest.float_shares", "bdp")]
    assert not flt.ok and "Field not valid" in flt.note and "units UNVERIFIED" not in flt.note
    assert not by[("raw.estimates.eps_ntm_est", "bdp")].ok  # all-NaN result
    assert "all-NaN" in by[("raw.estimates.eps_ntm_est", "bdp")].note
    assert by[("raw.universe.country", "derived")].returned_value == "US"
    row = mc.to_log_row(reviewer="analyst")
    assert row["vendor"] == "bloomberg" and row["test_ticker"] == IBM and row["value"] == 2.2e11 and row["reviewer"] == "analyst"
    # every features entry with an expression and every raw entry on this backend is checked
    n_feat = sum(1 for e in p.fieldmap["features"].values() if e.get("expression"))
    assert sum(1 for c in checks if c.field.startswith("features.")) == n_feat


def test_verify_fields_flags_implausible_scale():
    fake = FakeBQL({MCAP: {IBM: 220.0}})  # e.g. a map pointing at a 'billions' item
    p = bql_provider(fake, fieldmap={"universe": {"country": None}, "helpers": None, "features": None,
                                     "raw": {k: None for k in ("prices", "fundamentals", "estimates",
                                                               "short_interest", "options")}})
    p._blp_mod = FakeBlp().module
    checks = {c.field: c for c in p.verify_fields(as_of=AS_OF) if c.channel == "bql"}
    mc = checks["raw.universe.market_cap"]
    assert not mc.ok and "implausible scale" in mc.note


def test_pushdown_and_universe_have_no_relative_dates():
    fake = universe_fake()
    p = bql_provider(fake)
    p.get_universe(UniverseSpec(), AS_OF)
    for q in fake.queries:
        assert "0D" not in q and "-1y" not in q.lower() and "-52W" not in q
