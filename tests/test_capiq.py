"""Offline tests for the S&P Capital IQ adapter (``aitrading.data.capiq``).

No network and no vendor SDK: the GDS REST API is an ``httpx.MockTransport`` (``FakeGDS``) that implements
the token endpoint, ``clientservice.json`` (GDSP / GDSPV / GDSHE / GDSHV with the ``GDSSDKResponse`` shape the
community clients cited in docs/VENDOR_REFERENCE.md read) and ``usageservice.json``. Kensho clients are
duck-typed fakes; ``kfinance`` is injected through ``sys.modules``. The tests pin the exact request bodies
against VENDOR_REFERENCE section 3, the unit conversions into canonical units, NaN handling, the boundary
defaults and ``ProviderUnavailable`` when credentials or the SDK are missing.
"""

from __future__ import annotations

import ast
import json
import math
import re
import sys
import types
import urllib.parse
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pandas as pd
import pytest

from aitrading.core import fields as F
from aitrading.core.models import DocumentKind
from aitrading.core.policy import DataBoundary
from aitrading.data.base import (
    Capability,
    MarketDataProvider,
    PricePanel,
    ProviderError,
    ProviderUnavailable,
    ScreenPushdown,
)
from aitrading.data.capiq import (
    BOUNDARY_NOTE,
    DEFAULT_BASE_URL,
    FIELDMAP_ENV,
    CapIQProvider,
    FieldCheck,
    citation_problem,
    identifier_to_ticker,
    is_identifier,
    kensho_client_from_env,
    load_fieldmap,
    parse_transcript,
    ticker_to_identifier,
    validate_fieldmap,
)
from aitrading.data.fieldmaps import FieldMapError
from aitrading.screen.spec import UniverseSpec

TODAY = date(2026, 10, 2)
AS_OF = date(2026, 10, 1)
AS_OF_STR = "10/01/2026"
TEST_ID = "IBM:NYSE"
BASE = "https://api-ciq.marketintelligence.spglobal.com/gdsapi/rest"
SRC = Path(__file__).resolve().parents[1] / "src" / "aitrading" / "data"
G2_NOTE = "G2: written S&P confirmation 2026-09-30 (ref LEGAL-123), reviewed by A. Lawyer"
G3_NOTE = "G3: S&P AI-use letter AI-2026-7 for CIQ GDS, 2026-09-28, reviewed by B. Counsel"


# =============================================================================================
# Fake GDS REST API
# =============================================================================================


class FakeGDS:
    """Minimal Capital IQ GDS REST API: token, clientservice (GDSP/GDSPV/GDSHE/GDSHV) and usage."""

    def __init__(self) -> None:
        self.point: dict[tuple[str, str], Any] = {}  # (identifier, mnemonic) -> value
        self.point_by_props: dict[tuple[str, str, str], Any] = {}  # (identifier, mnemonic, json props) -> value
        self.errors: dict[tuple[str, str], str] = {}  # (identifier, mnemonic) -> ErrMsg
        self.history: dict[tuple[str, str], list[tuple[str, Any]]] = {}  # (identifier, mnemonic) -> [(date, value)]
        self.members: dict[str, list[str]] = {}  # index identifier -> constituent identifiers
        self.global_error: str | None = None
        self.mangle: str | None = None  # 'short' | 'swap'
        self.expired: set[str] = set()  # tokens answered with HTTP 401
        self.token_counter = 0
        self.requests: list[httpx.Request] = []
        self.token_calls: list[dict[str, Any]] = []
        self.data_bodies: list[dict[str, Any]] = []
        self.data_auth: list[str] = []
        self.usage_bodies: list[dict[str, Any]] = []

    # --- helpers for tests
    @property
    def input_requests(self) -> list[dict[str, Any]]:
        return [r for b in self.data_bodies for r in b["inputRequests"]]

    def sent(self, mnemonic: str | None = None, identifier: str | None = None) -> list[dict[str, Any]]:
        return [r for r in self.input_requests
                if (mnemonic is None or r["mnemonic"] == mnemonic) and (identifier is None or r["identifier"] == identifier)]

    # --- transport
    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path
        if path.endswith("/authenticate/api/v1/token"):
            form = dict(urllib.parse.parse_qsl(request.content.decode()))
            self.token_calls.append({"form": form, "content_type": request.headers.get("content-type"),
                                     "url": str(request.url)})
            if form != {"username": "alice", "password": "s3cret"}:
                return httpx.Response(401, json={"error": "invalid_grant"})
            self.token_counter += 1
            return httpx.Response(200, json={"access_token": f"tok{self.token_counter}", "token_type": "bearer"})
        token = request.headers.get("authorization", "").removeprefix("Bearer ")
        if not token.startswith("tok") or token in self.expired:
            return httpx.Response(401, json={"error": "token expired"})
        body = json.loads(request.content)
        if path.endswith("/v3/usageservice.json"):
            self.usage_bodies.append(body)
            return httpx.Response(200, json={"GDSSDKResponse": [{"Headers": ["USAGE_METRICS"], "Rows": [{"Row": ["123", "10000"]}], "ErrMsg": ""}]})
        if path.endswith("/v3/clientservice.json"):
            self.data_bodies.append(body)
            self.data_auth.append(token)
            if self.global_error:
                return httpx.Response(200, json={"GDSSDKResponse": [{"ErrMsg": self.global_error}]})
            out = [self._answer(r) for r in body["inputRequests"]]
            if self.mangle == "short":
                out = out[:-1]
            elif self.mangle == "swap" and len(out) > 1:
                out = [out[1], out[0], *out[2:]]
            return httpx.Response(200, json={"GDSSDKResponse": out})
        return httpx.Response(404, text="not found")

    def _answer(self, r: dict[str, Any]) -> dict[str, Any]:
        fn, ident, mn, props = r["function"], r["identifier"], r["mnemonic"], r.get("properties") or {}
        el = {"Function": fn, "Identifier": ident, "Mnemonic": mn, "Properties": props, "ErrMsg": "",
              "Headers": [mn], "Rows": [], "NumCols": 1, "NumRows": 0, "CacheExpiryTime": "0"}
        err = self.errors.get((ident, mn))
        if err:
            el["ErrMsg"] = err
            return el
        if fn == "GDSHE":  # the fake ignores the dates on purpose: the adapter must drop rows after endDate
            rows = [{"Row": [v, d]} for d, v in self.history.get((ident, mn), [])]
        elif fn == "GDSHV":
            members = self.members.get(ident, [])
            rows = [{"Row": [m]} for m in members[int(props["StartRank"]) - 1:int(props["EndRank"])]]
        else:
            key = (ident, mn, json.dumps(props, sort_keys=True))
            v = self.point_by_props.get(key, self.point.get((ident, mn), "Data Unavailable"))
            rows = [{"Row": [v]}]
        el["Rows"], el["NumRows"] = rows, len(rows)
        return el


def _leg_entries(fm: dict[str, Any]) -> list[tuple[str, dict[str, Any], str]]:
    """(label, entry, scale key) for every point mnemonic in the map (raw + features, incl. derived inputs)."""
    out = []
    for section, scale_key in (("raw", "to_canonical"), ("features", "to_catalog")):
        groups = fm[section].items() if section == "raw" else [("", fm["features"])]
        for ds, entries in groups:
            for col, e in entries.items():
                path = f"raw.{ds}.{col}" if section == "raw" else f"features.{col}"
                if e.get("mnemonic"):
                    out.append((path, e, scale_key))
                for i, inp in enumerate(e.get("inputs") or []):
                    if inp.get("mnemonic"):
                        out.append((f"{path}.inputs[{i}]", inp, "to_canonical"))
    return out


def _good_value(e: dict[str, Any], scale_key: str) -> str:
    kind = e.get("kind", "number")
    if kind == "label":
        return "Label"
    if kind == "date":
        return "10/28/2026"
    scale = float(e.get(scale_key, 1))
    lo, hi = e.get("plausible", [1, 3])
    return f"{(lo + hi) / 2 / scale:.6g}"


def _bdays(start: date, end: date) -> list[str]:
    return [d.strftime("%m/%d/%Y") for d in pd.bdate_range(start, end)]


def seed_test_identifier(fake: FakeGDS, fm: dict[str, Any]) -> None:
    """Plausible values for every mapped item on the test identifier, so preflights pass."""
    for _, e, scale_key in _leg_entries(fm):
        fake.point.setdefault((TEST_ID, e["mnemonic"]), _good_value(e, scale_key))
    days = _bdays(AS_OF - timedelta(days=14), AS_OF)
    for f, e in fm["history"]["fields"].items():
        fake.history.setdefault((TEST_ID, e["mnemonic"]), [(d, "1.25" if f == "volume" else "140.5") for d in days])
    bm = fm["benchmark"]
    fake.history.setdefault((bm["identifier"], bm["mnemonic"]), [(d, "5700.5") for d in days])
    fake.history.setdefault((bm["fallback_identifier"], bm["fallback_mnemonic"]), [(d, "570.1") for d in days])


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setenv("CAPIQ_USERNAME", "alice")
    monkeypatch.setenv("CAPIQ_PASSWORD", "s3cret")
    monkeypatch.delenv(FIELDMAP_ENV, raising=False)


@pytest.fixture
def fm(env) -> dict[str, Any]:
    return load_fieldmap()


@pytest.fixture
def fake(fm) -> FakeGDS:
    g = FakeGDS()
    seed_test_identifier(g, fm)
    return g


def make(fake: FakeGDS, **kw: Any) -> CapIQProvider:
    kw.setdefault("today", lambda: TODAY)
    return CapIQProvider(transport=httpx.MockTransport(fake.handler), **kw)


# =============================================================================================
# Field map
# =============================================================================================


def test_default_fieldmap_valid_and_statuses_match_reference(fm):
    assert validate_fieldmap(fm) == []
    raw = fm["raw"]
    mc = raw["universe"]["market_cap"]
    assert (mc["mnemonic"], mc["status"], mc["to_canonical"], mc["units_verified"]) == ("IQ_MARKETCAP", "confirmed", 1_000_000, False)
    assert mc["properties"]["currencyId"] == "USD"
    assert (raw["universe"]["exchange"]["mnemonic"], raw["universe"]["exchange"]["status"]) == ("IQ_EXCHANGE", "confirmed")
    assert raw["universe"]["gics_sector"]["status"] == "unverifiable"
    fu = raw["fundamentals"]
    assert (fu["fcf_ttm"]["mnemonic"], fu["fcf_ttm"]["status"]) == ("IQ_LEVERED_FCF", "corrected")
    assert (fu["capex_ttm"]["mnemonic"], fu["capex_ttm"]["status"], fu["capex_ttm"]["to_canonical"]) == ("IQ_CAPEX", "corrected", -1_000_000)
    assert (fu["cfo_ttm"]["mnemonic"], fu["cfo_ttm"]["status"]) == ("IQ_CASH_OPER", "corrected")
    assert (fu["total_debt"]["mnemonic"], fu["total_debt"]["status"]) == ("IQ_TOTAL_DEBT", "confirmed")
    assert (fu["cash_and_equivalents"]["mnemonic"], fu["cash_and_equivalents"]["status"]) == ("IQ_CASH_EQUIV", "confirmed")
    growth = fu["revenue_ttm_prior_year"]["inputs"][1]
    assert (growth["mnemonic"], growth["status"]) == ("IQ_TOTAL_REV_1YR_ANN_GROWTH", "confirmed")
    assert growth["properties"]["periodType"] == "IQ_LTM"  # exactly as in the 3.1 example
    assert fu["revenue_ttm"]["status"] == "unverifiable"  # IQ_TOTAL_REV is not a row of 3.3
    assert fu["ebitda_ttm"]["status"] == "unverifiable"  # derived combination
    assert [i["status"] for i in fu["ebitda_ttm"]["inputs"]] == ["confirmed", "confirmed"]
    es = raw["estimates"]
    assert (es["revenue_ntm_est"]["mnemonic"], es["revenue_ntm_est"]["status"]) == ("IQ_REVENUE_EST", "unverifiable")
    assert es["revenue_ntm_est"]["properties"]["periodType"] == "IQ_NTM"
    assert es["revenue_ntm_est_3m_ago"]["properties"]["asOfDate"] == "{as_of_3m}"
    assert (es["target_price_mean"]["mnemonic"], es["target_price_mean"]["status"]) == ("IQ_PRICE_TARGET", "confirmed")
    assert (es["next_earnings_date"]["mnemonic"], es["next_earnings_date"]["kind"]) == ("IQ_NEXT_EARNINGS_DATE", "date")
    assert all(e["available"] is False and e["status"] == "confirmed" for e in raw["options"].values())
    assert all(e["available"] is False and e["status"] == "unverifiable" for e in raw["short_interest"].values())
    hist = fm["history"]
    assert hist["function"] == "GDSHE" and hist["properties"] == {"startDate": "{start}", "endDate": "{end}"}
    assert (hist["fields"]["close"]["mnemonic"], hist["fields"]["close"]["status"]) == ("IQ_CLOSEPRICE_ADJ", "confirmed")
    assert (hist["fields"]["volume"]["mnemonic"], hist["fields"]["volume"]["status"]) == ("IQ_VOLUME", "confirmed")
    assert all(hist["fields"][f]["status"] == "unverifiable" for f in ("open", "high", "low"))
    cons = fm["universe"]["constituents"]
    assert (cons["function"], cons["mnemonic"], cons["status"]) == ("GDSHV", "IQ_CONSTITUENTS", "corrected")
    assert set(cons["properties"]) == {"StartRank", "EndRank"}
    assert (fm["benchmark"]["identifier"], fm["benchmark"]["mnemonic"], fm["benchmark"]["status"]) == ("^SPX", "IQ_CLOSEPRICE", "unverifiable")
    feats = fm["features"]
    assert (feats["ev_to_ebitda"]["mnemonic"], feats["ev_to_ebitda"]["status"]) == ("IQ_TEV_EBITDA", "confirmed")
    assert feats["fcf_yield_pct"]["status"] == "unverifiable" and feats["fcf_yield_pct"]["inputs"][0]["mnemonic"] == "IQ_MARKET_CAP_LFCF"
    assert (feats["high_52w"]["mnemonic"], feats["high_52w"]["status"]) == ("IQ_YEARHIGH", "unverifiable")
    assert feats["iv_30d_pct"]["available"] is False and feats["put_call_volume_ratio"]["status"] == "confirmed"
    assert feats["short_interest_pct_float"]["available"] is False
    docs = fm["documents"]
    assert docs["transcripts"]["status"] == "corrected"
    assert (docs["transcripts"]["latest_earnings_method"], docs["transcripts"]["transcript_method"]) == (
        "get_latest_earnings_from_identifiers", "get_transcript_from_key_dev_id")
    assert docs["research"]["status"] == "confirmed" and docs["research"]["available"] is False
    assert docs["news"]["status"] == "unverifiable"
    api = fm["api"]
    assert api["token_path"]["status"] == "corrected" and api["token_path"]["value"] == "/authenticate/api/v1/token"
    assert api["data_path"]["value"] == "/v3/clientservice.json"
    assert set(api["functions"]["value"]) == {"GDSP", "GDSPV", "GDSHE", "GDSHV", "GDST", "GDSG"}
    assert api["daily_request_limit"]["value"] == 10000
    assert fm["screen"]["pushdown"] is False
    assert fm["test_identifier"] == TEST_ID


def test_base_url_matches_reference(fm):
    assert DEFAULT_BASE_URL == BASE == fm["api"]["base_url"]["value"]


def _code_constants(src: str) -> list[str]:
    tree = ast.parse(src)
    docstrings: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                docstrings.add(id(first.value))
    return [n.value for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in docstrings]


def test_logic_hard_codes_no_vendor_mnemonic():
    offenders = [c for c in _code_constants((SRC / "capiq.py").read_text()) if re.search(r"\bIQ_[A-Z]", c)]
    assert offenders == []


def test_unverifiable_mnemonics_appear_nowhere_in_the_module(fm):
    src = (SRC / "capiq.py").read_text()
    statuses: dict[str, set[str]] = {}

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            if isinstance(node.get("mnemonic"), str):
                statuses.setdefault(node["mnemonic"], set()).add(node.get("status", ""))
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk({k: v for k, v in fm.items() if not k.startswith("_")})
    unverifiable = {m for m, s in statuses.items() if s == {"unverifiable"}}
    assert {"IQ_TOTAL_REV", "IQ_REVENUE_EST", "IQ_YEARHIGH", "IQ_MARKET_CAP_LFCF", "IQ_COMPANY_NAME"} <= unverifiable
    present = sorted(m for m in unverifiable if re.search(rf"(?<![A-Z0-9_]){re.escape(m)}(?![A-Z0-9_])", src))
    assert present == []


def test_env_override_is_deep_merged_then_constructor_override(env, tmp_path, monkeypatch):
    p = tmp_path / "capiq_override.json"
    p.write_text(json.dumps({"raw": {"fundamentals": {"revenue_ttm": {"status": "confirmed", "units_verified": True}}},
                             "api": {"max_requests_per_call": {"value": 7}}}))
    monkeypatch.setenv(FIELDMAP_ENV, str(p))
    fm = load_fieldmap()
    e = fm["raw"]["fundamentals"]["revenue_ttm"]
    assert (e["status"], e["units_verified"], e["mnemonic"]) == ("confirmed", True, "IQ_TOTAL_REV")  # other keys kept
    assert fm["_sources"][-1] == str(p)
    prov = CapIQProvider(fieldmap={"api": {"max_requests_per_call": {"value": 3}}, "raw": {"universe": {"name": None}}},
                         today=lambda: TODAY)
    assert prov.max_requests_per_call == 3  # constructor override wins over the env file
    assert "name" not in prov.fieldmap["raw"]["universe"]  # null deletes
    assert prov.fieldmap["raw"]["fundamentals"]["revenue_ttm"]["status"] == "confirmed"


@pytest.mark.parametrize("override, msg", [
    ({"raw": {"fundamentals": {"revenue_ttm": {"status": "verified"}}}}, "status"),
    ({"raw": {"fundamentals": {"not_a_column": {"mnemonic": "X", "status": "confirmed"}}}}, "not a canonical column"),
    ({"raw": {"fundamentals": {"ebitda_ttm": {"derive": "power"}}}}, "derive"),
    ({"history": {"fields": {"close": None}}}, "history.fields.close"),
    ({"features": {"not_a_feature": {"compute": "local", "status": "confirmed"}}}, "not a catalog feature"),
    ({"raw": {"universe": {"market_cap": {"plausible": [5, 1]}}}}, "plausible"),
])
def test_invalid_overrides_are_rejected(env, override, msg):
    with pytest.raises(FieldMapError, match=msg):
        CapIQProvider(fieldmap=override)


# =============================================================================================
# Protocol, capabilities, boundary
# =============================================================================================


def test_protocol_no_pushdown_and_capabilities(fake):
    p = make(fake, tickers=["IBM"])
    assert isinstance(p, MarketDataProvider)
    assert not isinstance(p, ScreenPushdown)
    assert not hasattr(p, "pushdown_screen")
    assert p.name == "capiq"
    assert p.capabilities == {Capability.PRICES, Capability.FUNDAMENTALS, Capability.ESTIMATES}
    assert Capability.SCREEN_PUSHDOWN not in p.capabilities
    assert fake.requests == []  # construction makes no network call


def test_default_boundary_denies_text_and_values(fake):
    p = make(fake)
    b = p.boundary
    assert b.provider == "capiq"
    assert b.allowed_document_kinds == set()
    assert b.allow_numeric_features is False
    assert b.note == BOUNDARY_NOTE
    assert "G2" in b.note and "G3" in b.note and "L1" in b.note and "L3" in b.note


@pytest.mark.parametrize("kwargs, msg", [
    ({"allowed_document_kinds": {DocumentKind.TRANSCRIPT}, "allow_numeric_features": False, "note": "approved by legal"}, "G2"),
    ({"allowed_document_kinds": set(), "allow_numeric_features": True, "note": "G2: letter"}, "G3"),
    ({"allowed_document_kinds": {DocumentKind.FILING}, "allow_numeric_features": False, "note": "G2: letter"}, "G3"),
    ({"allowed_document_kinds": {DocumentKind.RESEARCH}, "allow_numeric_features": False, "note": "G2 G3"}, "L6"),
    ({"allowed_document_kinds": {DocumentKind.TRANSCRIPT}, "allow_numeric_features": False, "note": BOUNDARY_NOTE}, "not a gate citation"),
])
def test_widened_boundary_must_cite_its_gate(fake, kwargs, msg):
    with pytest.raises(ValueError, match=msg):
        make(fake, boundary=DataBoundary(provider="capiq", **kwargs))


@pytest.mark.parametrize("note, why", [
    ("pending G2", "clause starting with 'G2:'"),  # a bare gate token is not a citation
    ("G2 not yet received", "clause starting with 'G2:'"),
    ("legal is fine; see the G2 file", "clause starting with 'G2:'"),
    ("G2: pending", "not been received"),
    ("G2: S&P letter KEN-1, 2026-09-15, reviewed by J. Smith - countersignature not yet received", "not been received"),
    ("G2: <confirmation reference>, <YYYY-MM-DD>, reviewed by <lawyer>", "placeholder"),
    ("G2: S&P letter KEN-1, reviewed by J. Smith", "YYYY-MM-DD"),
    ("G2: S&P letter KEN-1, 2026-11-01, reviewed by J. Smith", "YYYY-MM-DD"),  # dated after today
    ("G2: S&P letter KEN-1, 2026-09-15", "reviewing lawyer"),
    ("G3: " + G2_NOTE[4:], "clause starting with 'G2:'"),  # the wrong gate
])
def test_widened_boundary_needs_a_structured_g2_citation(fake, note, why):
    """Regression: a bare 'G2' token anywhere in the note used to permit Kensho transcript text."""
    b = DataBoundary(provider="capiq", allowed_document_kinds={DocumentKind.TRANSCRIPT}, allow_numeric_features=False,
                     note=note)
    with pytest.raises(ValueError, match=re.escape(why)):
        make(fake, boundary=b)


def test_numeric_values_need_their_own_g3_citation(fake):
    # DataBoundary defaults allow_numeric_features=True: a G2-only note must not slip numeric values through
    with pytest.raises(ValueError, match="allow_numeric_features=False"):
        make(fake, boundary=DataBoundary(provider="capiq", allowed_document_kinds={DocumentKind.TRANSCRIPT}, note=G2_NOTE))
    with pytest.raises(ValueError, match="G3: the citation says the confirmation has not been received"):
        make(fake, boundary=DataBoundary(provider="capiq", allowed_document_kinds=set(), allow_numeric_features=True,
                                         note=f"{G2_NOTE}; G3: pending"))
    with pytest.raises(ValueError, match="other than transcripts"):
        make(fake, boundary=DataBoundary(provider="capiq", allowed_document_kinds={DocumentKind.NEWS},
                                         allow_numeric_features=False, note=G2_NOTE))


def test_citation_problem_helper():
    assert citation_problem(G2_NOTE, "G2", TODAY) is None
    assert citation_problem(G2_NOTE, "g2", TODAY) is None
    assert "YYYY-MM-DD" in citation_problem(G2_NOTE, "G2", date(2026, 9, 1))  # confirmation dated in the future
    assert "clause starting with 'G3:'" in citation_problem(G2_NOTE, "G3", TODAY)
    assert citation_problem(f"{G2_NOTE}; {G3_NOTE}", "G3", TODAY) is None
    # a second, valid clause for the same gate is enough
    assert citation_problem(f"G2: pending; {G2_NOTE}", "G2", TODAY) is None
    assert "clause starting" in citation_problem(BOUNDARY_NOTE, "G2", TODAY)  # the default note cites nothing
    assert "clause starting" in citation_problem(BOUNDARY_NOTE, "G3", TODAY)


def test_widened_boundary_with_citation_is_accepted(fake):
    b = DataBoundary(provider="capiq", allowed_document_kinds={DocumentKind.TRANSCRIPT}, allow_numeric_features=False, note=G2_NOTE)
    assert make(fake, boundary=b).boundary is b
    b3 = DataBoundary(provider="capiq", allowed_document_kinds=set(), allow_numeric_features=True, note=G3_NOTE)
    assert make(fake, boundary=b3).boundary.allow_numeric_features is True
    both = DataBoundary(provider="capiq", allowed_document_kinds={DocumentKind.TRANSCRIPT, DocumentKind.FILING},
                        allow_numeric_features=True, note=f"{G2_NOTE}; {G3_NOTE}")
    assert make(fake, boundary=both).boundary is both
    multi_line = DataBoundary(provider="capiq", allowed_document_kinds={DocumentKind.TRANSCRIPT},
                              allow_numeric_features=True, note=f"{G2_NOTE}\n{G3_NOTE}")
    assert make(fake, boundary=multi_line).boundary is multi_line
    with pytest.raises(ValueError, match="provider"):
        make(fake, boundary=DataBoundary(provider="lseg", allowed_document_kinds=set(), allow_numeric_features=False))


# =============================================================================================
# Availability: credentials, SDK, diagnostics
# =============================================================================================


def test_missing_credentials_raise_provider_unavailable(fake, monkeypatch):
    monkeypatch.delenv("CAPIQ_USERNAME")
    p = make(fake, tickers=["IBM"])
    with pytest.raises(ProviderUnavailable, match="CAPIQ_USERNAME"):
        p.get_fundamentals(["IBM"], AS_OF)
    assert fake.requests == []  # nothing sent without credentials


def test_custom_credential_env_names(fake, monkeypatch):
    monkeypatch.setenv("MY_CIQ_USER", "alice")
    monkeypatch.setenv("MY_CIQ_PW", "s3cret")
    monkeypatch.delenv("CAPIQ_USERNAME")
    monkeypatch.delenv("CAPIQ_PASSWORD")
    fake.point[("ACME:", "IQ_PRICE_TARGET")] = "50"
    p = make(fake, username_env="MY_CIQ_USER", password_env="MY_CIQ_PW")
    assert p.get_estimates(["ACME"], AS_OF).loc["ACME", F.TARGET_PRICE_MEAN] == 50.0


def test_rejected_login_is_provider_unavailable(fake, monkeypatch):
    monkeypatch.setenv("CAPIQ_PASSWORD", "wrong")
    p = make(fake)
    with pytest.raises(ProviderUnavailable, match="authentication failed"):
        p.get_estimates(["ACME"], AS_OF)
    assert fake.data_bodies == []


def test_unreachable_host_is_provider_unavailable(env):
    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    p = CapIQProvider(transport=httpx.MockTransport(boom), today=lambda: TODAY)
    with pytest.raises(ProviderUnavailable, match="cannot reach"):
        p.get_estimates(["ACME"], AS_OF)


def test_plain_http_base_url_is_refused_before_credentials_are_sent(fake):
    p = make(fake, base_url="http://api-ciq.example.com/gdsapi/rest")
    with pytest.raises(ProviderUnavailable, match="must be HTTPS"):
        p.get_estimates(["ACME"], AS_OF)
    assert fake.requests == []  # no credential ever left the process
    assert any("not HTTPS" in d for d in p.diagnostics())
    fake.point[("ACME:", "IQ_PRICE_TARGET")] = "10"
    local = make(fake, base_url="http://localhost:8080/gdsapi/rest")  # a local test double / forwarding proxy
    assert local.get_estimates(["ACME"], AS_OF).loc["ACME", F.TARGET_PRICE_MEAN] == 10.0
    assert str(fake.token_calls[0]["url"]) == "http://localhost:8080/gdsapi/rest/authenticate/api/v1/token"


def test_cli_factory_builds_the_provider_with_tickers(env, monkeypatch):
    from aitrading.cli import make_provider

    p = make_provider("capiq", tickers=["IBM", "MSFT"])
    assert isinstance(p, CapIQProvider) and p.tickers == ["IBM", "MSFT"]
    assert p.boundary.allowed_document_kinds == set() and p.boundary.allow_numeric_features is False


def test_kensho_client_from_env_without_sdk(monkeypatch):
    monkeypatch.setitem(sys.modules, "kfinance", None)
    with pytest.raises(ProviderUnavailable, match="pip install kensho-kfinance"):
        kensho_client_from_env()


def test_kensho_client_from_env_with_fake_sdk(monkeypatch):
    created: list[dict[str, Any]] = []

    class Client:
        def __init__(self, **kw: Any) -> None:
            created.append(kw)

    mod = types.ModuleType("kfinance.client.kfinance")
    mod.Client = Client
    monkeypatch.setitem(sys.modules, "kfinance", types.ModuleType("kfinance"))
    monkeypatch.setitem(sys.modules, "kfinance.client", types.ModuleType("kfinance.client"))
    monkeypatch.setitem(sys.modules, "kfinance.client.kfinance", mod)
    with pytest.raises(ProviderUnavailable, match="KENSHO_CLIENT_ID"):
        kensho_client_from_env(env={})
    client = kensho_client_from_env(env={"KENSHO_CLIENT_ID": "cid", "KENSHO_PRIVATE_KEY": "-----BEGIN KEY-----"})
    assert isinstance(client, Client)
    assert created == [{"client_id": "cid", "private_key": "-----BEGIN KEY-----"}]


def test_diagnostics_is_offline_and_explains_setup(fake, monkeypatch):
    p = make(fake, tickers=["IBM"])
    assert p.diagnostics() == []
    monkeypatch.delenv("CAPIQ_PASSWORD")
    q = make(fake, kensho=SimpleNamespace())
    probs = " | ".join(q.diagnostics())
    assert "CAPIQ_PASSWORD" in probs
    assert "no screen function" in probs and "Xpressfeed" in probs
    assert "G2" in probs  # kensho injected but the default boundary denies transcripts
    assert fake.requests == []


# =============================================================================================
# Auth, transport, batching, budget
# =============================================================================================


def test_token_auth_and_exact_gdsp_requests(fake):
    fake.point[("ACME:", "IQ_PRICE_TARGET")] = "120.5"
    p = make(fake)
    p.get_estimates(["ACME"], AS_OF)
    tok = fake.token_calls[0]
    assert tok["url"] == f"{BASE}/authenticate/api/v1/token"
    assert tok["form"] == {"username": "alice", "password": "s3cret"}
    assert tok["content_type"] == "application/x-www-form-urlencoded"
    data_req = [r for r in fake.requests if r.url.path.endswith("clientservice.json")]
    assert all(str(r.url) == f"{BASE}/v3/clientservice.json" and r.method == "POST" for r in data_req)
    assert all(r.headers["authorization"] == "Bearer tok1" for r in data_req)
    assert all(r.headers["content-type"] == "application/json" for r in data_req)
    assert fake.sent("IQ_PRICE_TARGET", "ACME:") == [
        {"function": "GDSP", "identifier": "ACME:", "mnemonic": "IQ_PRICE_TARGET",
         "properties": {"currencyId": "USD", "asOfDate": AS_OF_STR}}]
    # the audit log holds the bodies verbatim and never the credentials
    assert [r.content.decode() for r in data_req] == p.query_log
    assert all("s3cret" not in q and "alice" not in q for q in p.query_log)
    assert len(fake.token_calls) == 1  # one login reused for every call


def test_reauthenticates_once_on_401(fake):
    fake.expired = {"tok1"}
    fake.point[("ACME:", "IQ_PRICE_TARGET")] = "99"
    p = make(fake)
    df = p.get_estimates(["ACME"], AS_OF)
    assert df.loc["ACME", F.TARGET_PRICE_MEAN] == 99.0
    assert len(fake.token_calls) == 2
    assert fake.data_auth and set(fake.data_auth) == {"tok2"}
    fake.expired = {"tok2", "tok3"}
    with pytest.raises(ProviderUnavailable, match="refused"):
        p.get_estimates(["OTHER"], AS_OF)


def test_batching_and_request_counter(fake):
    p = make(fake, tickers=["AAA", "BBB", "CCC"], fieldmap={"api": {"max_requests_per_call": {"value": 4}}})
    p.get_universe(None, AS_OF)
    sizes = [len(b["inputRequests"]) for b in fake.data_bodies]
    # 3 preflights (name, sector, industry on IBM:NYSE) in one call, then 5 items x 3 tickers in batches of 4
    assert sizes == [3, 4, 4, 4, 3]
    assert {r["identifier"] for r in fake.data_bodies[0]["inputRequests"]} == {TEST_ID}
    assert p.requests_today == 18 and p.http_calls == 5
    assert p.requests_remaining == 10000 - 18


def test_request_budget_warns_then_refuses_before_sending(fake):
    p = make(fake, tickers=["ACME"],
             fieldmap={"api": {"daily_request_limit": {"value": 10, "warn_fraction": 0.5}, "max_requests_per_call": {"value": 4}}})
    p.get_universe(None, AS_OF)  # 3 preflights + 5 items
    assert p.requests_today == 8
    assert any("observed daily limit of 10" in w for w in p.warnings)
    calls = len(fake.data_bodies)
    with pytest.raises(ProviderError, match="request budget"):
        p.get_fundamentals(["ACME"], AS_OF)  # 1 preflight fits, the 12 fundamentals requests do not
    assert len(fake.data_bodies) == calls + 1 and p.requests_today == 9
    later = {"value": 1}
    p2 = make(fake, fieldmap={"api": {"daily_request_limit": later}})
    p2._today = lambda: TODAY  # noqa: SLF001
    p2.requests_today = 1
    with pytest.raises(ProviderError, match="request budget"):
        p2.get_estimates(["ACME"], AS_OF)
    p2._today = lambda: TODAY + timedelta(days=1)  # noqa: SLF001 - a new day resets the counter
    assert p2.requests_remaining == 1


def test_daily_limit_errmsg_stops_further_requests(fake):
    fake.global_error = "Daily Request Limit of 10000 Exceeded"
    p = make(fake)
    with pytest.raises(ProviderError, match="daily request limit"):
        p.get_estimates(["ACME"], AS_OF)
    n = len(fake.data_bodies)
    fake.global_error = None
    with pytest.raises(ProviderError, match="earlier today"):
        p.get_estimates(["ACME"], AS_OF)
    assert len(fake.data_bodies) == n


def test_other_whole_call_error_is_provider_error(fake):
    fake.global_error = "Invalid request format"
    with pytest.raises(ProviderError, match="Invalid request format"):
        make(fake).get_estimates(["ACME"], AS_OF)


@pytest.mark.parametrize("mangle, msg", [("short", "result"), ("swap", "out of order")])
def test_response_shape_is_asserted(fake, mangle, msg):
    fake.mangle = mangle
    with pytest.raises(ProviderError, match=msg):
        make(fake, preflight=False).get_estimates(["ACME", "BETA"], AS_OF)


def test_responses_and_preflights_are_cached(fake):
    fake.errors[(TEST_ID, "IQ_TOTAL_REV")] = "Data item not available"
    p = make(fake)
    first = p.get_fundamentals(["ACME"], AS_OF)
    n = len(fake.input_requests)
    second = p.get_fundamentals(["ACME"], AS_OF)
    assert len(fake.input_requests) == n  # nothing re-sent: GDS budget is precious
    pd.testing.assert_frame_equal(first, second)
    # the failed preflight was sent once and IQ_TOTAL_REV was never requested in bulk
    assert fake.sent("IQ_TOTAL_REV") == fake.sent("IQ_TOTAL_REV", TEST_ID) and len(fake.sent("IQ_TOTAL_REV")) == 1
    p.clear_cache()
    p.get_fundamentals(["ACME"], AS_OF)
    assert len(fake.input_requests) > n


def test_gdspv_function_from_the_map_and_close(fake):
    fake.point[("ACME:", "IQ_PRICE_TARGET")] = "77"
    with make(fake, fieldmap={"raw": {"estimates": {"target_price_mean": {"function": "GDSPV"}}}}) as p:
        assert p.get_estimates(["ACME"], AS_OF).loc["ACME", F.TARGET_PRICE_MEAN] == 77.0
        assert fake.sent("IQ_PRICE_TARGET", "ACME:")[0]["function"] == "GDSPV"
        assert p._http is not None  # noqa: SLF001
    assert p._http is None and p._token is None  # noqa: SLF001 - closed on exit
    with pytest.raises(FieldMapError, match="function"):
        CapIQProvider(fieldmap={"raw": {"estimates": {"target_price_mean": {"function": "SCREEN"}}}})


def test_usage_endpoint(fake):
    p = make(fake)
    out = p.usage()
    assert fake.usage_bodies == [{"inputRequests": [{"mnemonic": "USAGE_METRICS"}]}]
    req = [r for r in fake.requests if r.url.path.endswith("usageservice.json")][0]
    assert str(req.url) == f"{BASE}/v3/usageservice.json" and req.headers["authorization"] == "Bearer tok1"
    assert out["GDSSDKResponse"][0]["Rows"][0]["Row"] == ["123", "10000"]
    assert p.requests_today == 0


# =============================================================================================
# Universe
# =============================================================================================


def _seed_universe(fake: FakeGDS) -> None:
    fake.point.update({
        (TEST_ID, "IQ_MARKETCAP"): "215000.5", (TEST_ID, "IQ_EXCHANGE"): "NYSE",
        (TEST_ID, "IQ_COMPANY_NAME"): "International Business Machines Corporation",
        (TEST_ID, "IQ_INDUSTRY_SECTOR"): "Information Technology", (TEST_ID, "IQ_INDUSTRY"): "IT Services",
        ("BRK.B:", "IQ_MARKETCAP"): "1000000", ("BRK.B:", "IQ_EXCHANGE"): "NYSE",
        ("BRK.B:", "IQ_COMPANY_NAME"): "Berkshire Hathaway Inc.", ("BRK.B:", "IQ_INDUSTRY_SECTOR"): "Financials",
        ("MSFT:NasdaqGS", "IQ_MARKETCAP"): "3100000", ("MSFT:NasdaqGS", "IQ_COMPANY_NAME"): "Microsoft Corporation",
        ("MSFT:NasdaqGS", "IQ_INDUSTRY_SECTOR"): "Information Technology",
    })
    fake.errors[("ZZZZ:", "IQ_MARKETCAP")] = "Invalid Identifier"
    fake.errors[("ZZZZ:", "IQ_EXCHANGE")] = "Invalid Identifier"


def test_universe_explicit_tickers_units_labels_and_preflight(fake):
    _seed_universe(fake)
    p = make(fake, tickers=["IBM", "BRK-B", "MSFT:NasdaqGS", "ZZZZ"], identifiers={"IBM": TEST_ID})
    df = p.get_universe(UniverseSpec(), AS_OF)
    assert list(df.columns) == F.UNIVERSE_COLUMNS
    assert list(df.index) == ["IBM", "BRK-B", "MSFT", "ZZZZ"] and df.index.name == "ticker"
    assert df.loc["IBM", F.MARKET_CAP] == pytest.approx(215_000.5e6)
    assert df.loc["BRK-B", F.MARKET_CAP] == pytest.approx(1e12)
    assert math.isnan(df.loc["ZZZZ", F.MARKET_CAP])  # ErrMsg -> NaN, never 0
    assert df[F.MARKET_CAP].dtype == "float64"
    assert list(df[F.VENDOR_ID]) == [TEST_ID, "BRK.B:", "MSFT:NasdaqGS", "ZZZZ:"]
    assert df.loc["IBM", F.EXCHANGE] == "NYSE"
    assert df.loc["MSFT", F.EXCHANGE] == "NASDAQ"  # missing item -> exchange part of the identifier, mapped
    assert list(df[F.COUNTRY].fillna("<NA>")) == ["US", "US", "US", "<NA>"]
    assert list(df[F.CURRENCY].fillna("<NA>")) == ["USD", "USD", "USD", "<NA>"]
    assert df[F.SECURITY_TYPE].isna().all()  # no verified GDS item: no label is assumed
    assert df.loc["BRK-B", F.GICS_SECTOR] == "Financials"
    assert df.loc["IBM", F.NAME] == "International Business Machines Corporation"
    assert fake.sent("IQ_MARKETCAP", TEST_ID) == [
        {"function": "GDSP", "identifier": TEST_ID, "mnemonic": "IQ_MARKETCAP", "properties": {"currencyId": "USD", "asOfDate": AS_OF_STR}}]
    assert fake.sent("IQ_EXCHANGE", "BRK.B:") == [
        {"function": "GDSP", "identifier": "BRK.B:", "mnemonic": "IQ_EXCHANGE", "properties": {}}]
    # unverifiable label items were preflighted on the test identifier before bulk use (first call)
    assert {r["mnemonic"] for r in fake.data_bodies[0]["inputRequests"]} == {"IQ_COMPANY_NAME", "IQ_INDUSTRY_SECTOR", "IQ_INDUSTRY"}
    warn = " | ".join(p.warnings)
    assert "units UNVERIFIED" in warn and "IQ_MARKETCAP x1e+06" in warn
    assert "labelled" not in warn
    assert "LEG NOT EVALUATED: capiq security_type" in warn and "security_types=[]" in warn
    assert "Invalid Identifier" in warn
    filtered = p.get_universe(UniverseSpec(exclude_sectors=["financials"]), AS_OF)
    # CapIQ's own (unadmitted) classification is not pre-applied as a filter; the local engine evaluates it
    assert list(filtered.index) == ["IBM", "BRK-B", "MSFT", "ZZZZ"]  # unknown country kept for the local engine
    assert any("exclude_sectors was not pre-applied" in w and "G5" in w for w in p.warnings)


def test_security_type_only_from_an_explicit_assertion_or_a_mapped_item(fake):
    """Regression: every name used to be labelled common_stock, so REITs / ADRs passed the default filter."""
    _seed_universe(fake)
    p = make(fake, tickers=["BRK-B"],
             fieldmap={"raw": {"universe": {"security_type": {"available": None, "assume": "common_stock"}}}})
    df = p.get_universe(UniverseSpec(), AS_OF)
    assert df.loc["BRK-B", F.SECURITY_TYPE] == "common_stock"
    assert any("field-map assertion" in w for w in p.warnings)
    assert not any("LEG NOT EVALUATED: capiq security_type" in w for w in p.warnings)
    # a mapped (admitted) item classifies: known mismatches are dropped, unknowns are kept for the local engine
    fake.point.update({("AAA:", "IQ_SECTYPE_TEST"): "reit", ("BBB:", "IQ_SECTYPE_TEST"): "Common_Stock"})
    q = make(fake, tickers=["AAA", "BBB", "CCC"], fieldmap={"raw": {"universe": {"security_type": {
        "available": None, "mnemonic": "IQ_SECTYPE_TEST", "kind": "label", "status": "confirmed"}}}})
    out = q.get_universe(UniverseSpec(), AS_OF)
    assert list(out.index) == ["BBB", "CCC"] and pd.isna(out.loc["CCC", F.SECURITY_TYPE])
    assert list(q.get_universe(UniverseSpec(security_types=[]), AS_OF).index) == ["AAA", "BBB", "CCC"]
    # without any source the leg is suspended (banner), never silently filled
    r = make(fake, tickers=["AAA", "BBB"])
    assert list(r.get_universe(UniverseSpec(), AS_OF).index) == ["AAA", "BBB"]
    assert any(w.startswith("LEG NOT EVALUATED: capiq security_type") for w in r.warnings)
    s = make(fake, tickers=["AAA"])
    s.get_universe(None, AS_OF)  # no spec: no security-type filter requested, so no banner
    assert not any("LEG NOT EVALUATED: capiq security_type" in w for w in s.warnings)


def test_universe_country_from_exchange_and_known_mismatch(fake):
    fake.point[("SHOP:TSX", "IQ_EXCHANGE")] = "TSX"
    fake.point[("IBM:", "IQ_EXCHANGE")] = "NYSE"
    p = make(fake, tickers=["SHOP:TSX", "IBM"])
    df = p.get_universe(UniverseSpec(country="US"), AS_OF)
    assert df.loc["SHOP", F.EXCHANGE] == "TSX"  # unmapped exchange passes through ...
    assert pd.isna(df.loc["SHOP", F.COUNTRY])  # ... its country is unknown (the local engine excludes it)
    q = make(fake, tickers=["SHOP:TSX", "IBM"],
             fieldmap={"raw": {"universe": {"country": {"value_map": {"TSX": "CA"}}}}})
    assert list(q.get_universe(UniverseSpec(country="US"), AS_OF).index) == ["IBM"]  # known mismatch dropped
    assert q.get_universe(None, AS_OF).loc["SHOP", F.COUNTRY] == "CA"


def test_universe_index_constituents_paging_and_symbology(fake):
    fake.members["^SPX"] = ["AAPL:NasdaqGS", "IQ999", "MSFT:"]
    p = make(fake, index="SPX", fieldmap={"universe": {"constituents": {"page_size": 2}}})
    df = p.get_universe(None, AS_OF)
    assert list(df.index) == ["AAPL", "MSFT"]
    assert list(df[F.VENDOR_ID]) == ["AAPL:NasdaqGS", "MSFT:"]
    assert fake.sent("IQ_CONSTITUENTS") == [
        {"function": "GDSHV", "identifier": "^SPX", "mnemonic": "IQ_CONSTITUENTS", "properties": {"StartRank": 1, "EndRank": 2}},
        {"function": "GDSHV", "identifier": "^SPX", "mnemonic": "IQ_CONSTITUENTS", "properties": {"StartRank": 3, "EndRank": 4}},
    ]
    assert any("1 constituent(s)" in w and "dropped" in w and "graft 6" in w for w in p.warnings)
    assert not any("IQ999" in w for w in p.warnings)  # vendor ids stay out of the persisted warnings ...
    assert any("IQ999" in d for d in p.details)  # ... and go to the console-only details
    q = make(fake, index="^SPX", identifiers={"XYZ": "IQ999"})
    assert list(q.get_universe(None, AS_OF).index) == ["AAPL", "XYZ", "MSFT"]
    assert fake.sent("IQ_CONSTITUENTS")[-1]["properties"] == {"StartRank": 1, "EndRank": 600}  # the 3.1 example form
    fake.errors[("^RUT", "IQ_CONSTITUENTS")] = "Invalid Identifier"
    with pytest.raises(ProviderError, match="untested"):
        make(fake, index="^RUT").get_universe(None, AS_OF)


def test_universe_without_tickers_or_index_raises(fake):
    p = make(fake)
    with pytest.raises(ProviderError) as ei:
        p.get_universe(UniverseSpec(), AS_OF)
    msg = str(ei.value)
    assert "no screen function" in msg and "tickers=[...]" in msg and "index='^SPX'" in msg and "Xpressfeed" in msg
    assert fake.requests == []


# =============================================================================================
# Fundamentals, estimates
# =============================================================================================


def _seed_fundamentals(fake: FakeGDS) -> None:
    fake.point.update({
        ("ACME:", "IQ_TOTAL_REV"): "1000", ("ACME:", "IQ_TOTAL_REV_1YR_ANN_GROWTH"): "25",
        ("ACME:", "IQ_GROSS_MARGIN"): "40", ("ACME:", "IQ_OPER_INC"): "150", ("ACME:", "IQ_TEV"): "5000",
        ("ACME:", "IQ_TEV_EBITDA"): "10", ("ACME:", "IQ_CASH_OPER"): "300", ("ACME:", "IQ_CAPEX"): "-120",
        ("ACME:", "IQ_LEVERED_FCF"): "170", ("ACME:", "IQ_TOTAL_DEBT"): "800", ("ACME:", "IQ_CASH_EQUIV"): "200",
        ("ACME:", "IQ_SHARESOUTSTANDING"): "50.5",
        ("BETA:", "IQ_TOTAL_REV"): "Data Unavailable", ("BETA:", "IQ_TOTAL_REV_1YR_ANN_GROWTH"): "12",
        ("BETA:", "IQ_GROSS_MARGIN"): "NM", ("BETA:", "IQ_TEV"): "4000", ("BETA:", "IQ_TEV_EBITDA"): "0",
        ("BETA:", "IQ_CASH_OPER"): "abc", ("BETA:", "IQ_CAPEX"): "75", ("BETA:", "IQ_LEVERED_FCF"): "CapabilityNeeded",
        ("BETA:", "IQ_TOTAL_DEBT"): "9999999999", ("BETA:", "IQ_CASH_EQUIV"): "0",
    })
    fake.errors[("BETA:", "IQ_OPER_INC")] = "Data item not available for this identifier"


def test_fundamentals_requests_units_and_derivations(fake):
    _seed_fundamentals(fake)
    p = make(fake)
    df = p.get_fundamentals(["ACME", "BETA"], AS_OF)
    assert list(df.columns) == F.FUNDAMENTAL_COLUMNS and list(df.index) == ["ACME", "BETA"]
    a = df.loc["ACME"]
    assert a[F.REVENUE_TTM] == pytest.approx(1e9)
    assert a[F.REVENUE_TTM_PRIOR_YEAR] == pytest.approx(8e8)  # 1e9 / (1 + 25%)
    assert a[F.GROSS_PROFIT_TTM] == pytest.approx(4e8)  # 1e9 x 40%
    assert a[F.OPERATING_INCOME_TTM] == pytest.approx(1.5e8)
    assert a[F.EBITDA_TTM] == pytest.approx(5e8)  # TEV 5000mn / 10x
    assert a[F.CFO_TTM] == pytest.approx(3e8)
    assert a[F.CAPEX_TTM] == pytest.approx(1.2e8)  # vendor -120mn -> +1.2e8 cash spent
    assert a[F.FCF_TTM] == pytest.approx(1.7e8)
    assert a[F.TOTAL_DEBT] == pytest.approx(8e8) and a[F.CASH] == pytest.approx(2e8)
    assert a[F.SHARES_OUTSTANDING] == pytest.approx(5.05e7)
    for col in (F.PERIOD_END, F.REPORT_DATE, F.NET_INCOME_TTM, F.TOTAL_EQUITY, F.REVENUE_LAST_Q):
        assert pd.isna(a[col])  # no mapping: NaN, never 0
    b = df.loc["BETA"]
    assert b[F.CASH] == 0.0  # a real zero stays zero
    for col in (F.REVENUE_TTM, F.REVENUE_TTM_PRIOR_YEAR, F.GROSS_PROFIT_TTM, F.OPERATING_INCOME_TTM, F.EBITDA_TTM,
                F.CFO_TTM, F.CAPEX_TTM, F.FCF_TTM, F.TOTAL_DEBT, F.SHARES_OUTSTANDING):
        assert math.isnan(b[col]), col
    req = fake.sent("IQ_TOTAL_REV", "ACME:")
    assert req == [{"function": "GDSP", "identifier": "ACME:", "mnemonic": "IQ_TOTAL_REV",
                    "properties": {"periodType": "IQ_LTM", "currencyId": "USD", "asOfDate": AS_OF_STR}}]
    assert fake.sent("IQ_TOTAL_REV_1YR_ANN_GROWTH", "ACME:")[0]["properties"] == {"periodType": "IQ_LTM", "asOfDate": AS_OF_STR}
    # 12 items x 2 names + 1 preflight (IQ_TOTAL_REV is the only unverifiable item)
    assert p.requests_today == 25
    assert fake.sent("IQ_TOTAL_REV", TEST_ID)[0]["properties"] == req[0]["properties"]
    warn = " | ".join(p.warnings)
    assert "implausible" in warn and "IQ_CAPEX" in warn  # positive capex fails the sign check
    assert "not entitled ('CapabilityNeeded')" in warn
    assert "non-numeric 'abc'" in warn
    assert "Data item not available" in warn
    assert "units UNVERIFIED" in warn
    assert "no S&P mapping (NaN)" in warn and "net_income_ttm" in warn


def test_unverifiable_preflight_failure_suspends_the_leg(fake):
    _seed_fundamentals(fake)
    fake.errors[(TEST_ID, "IQ_TOTAL_REV")] = "Invalid Data Item"
    p = make(fake)
    df = p.get_fundamentals(["ACME"], AS_OF)
    assert math.isnan(df.loc["ACME", F.REVENUE_TTM])
    assert math.isnan(df.loc["ACME", F.REVENUE_TTM_PRIOR_YEAR]) and math.isnan(df.loc["ACME", F.GROSS_PROFIT_TTM])
    assert df.loc["ACME", F.EBITDA_TTM] == pytest.approx(5e8)  # other legs unaffected
    assert fake.sent("IQ_TOTAL_REV", "ACME:") == []  # never requested in bulk
    assert any(w.startswith("LEG NOT EVALUATED") and "IQ_TOTAL_REV" in w and f"preflight on {TEST_ID} failed" in w
               for w in p.warnings)


def test_preflight_can_be_disabled(fake):
    _seed_fundamentals(fake)
    fake.errors[(TEST_ID, "IQ_TOTAL_REV")] = "Invalid Data Item"
    p = make(fake, preflight=False)
    assert p.get_fundamentals(["ACME"], AS_OF).loc["ACME", F.REVENUE_TTM] == pytest.approx(1e9)
    assert fake.sent("IQ_TOTAL_REV", TEST_ID) == []


def test_strict_units_withholds_unverified_units(fake):
    _seed_fundamentals(fake)
    _seed_universe(fake)
    p = make(fake, tickers=["BRK-B"], strict_units=True)
    assert Capability.FUNDAMENTALS not in p.capabilities and Capability.ESTIMATES in p.capabilities
    df = p.get_fundamentals(["ACME"], AS_OF)
    assert df.isna().all().all()
    assert fake.requests == []
    assert any("strict_units=True" in w for w in p.warnings)
    uni = p.get_universe(None, AS_OF)
    assert math.isnan(uni.loc["BRK-B", F.MARKET_CAP])
    assert fake.sent("IQ_MARKETCAP") == []
    assert any("LEG NOT EVALUATED" in w and "IQ_MARKETCAP" in w for w in p.warnings)


def test_estimates_asof_dates_and_units(fake):
    def props(period: bool, as_of: str) -> str:
        d = {"periodType": "IQ_NTM", "currencyId": "USD", "asOfDate": as_of} if period else {"currencyId": "USD", "asOfDate": as_of}
        return json.dumps(d, sort_keys=True)

    fake.point_by_props.update({
        ("ACME:", "IQ_REVENUE_EST", props(True, AS_OF_STR)): "1100",
        ("ACME:", "IQ_REVENUE_EST", props(True, "07/02/2026")): "1050",
        ("ACME:", "IQ_EPS_EST", props(True, AS_OF_STR)): "5.25",
        ("ACME:", "IQ_EPS_EST", props(True, "07/02/2026")): "5.00",
        # the lag check: on the test identifier the 3-month-ago value differs from the as_of value
        (TEST_ID, "IQ_REVENUE_EST", props(True, "07/02/2026")): "61000",
        (TEST_ID, "IQ_EPS_EST", props(True, "07/02/2026")): "10.5",
    })
    fake.point.update({("ACME:", "IQ_PRICE_TARGET"): "120.5", ("ACME:", "IQ_EST_EPS_SURPRISE_PERCENT"): "4.2",
                       ("ACME:", "IQ_NEXT_EARNINGS_DATE"): "10/28/2026"})
    p = make(fake)
    df = p.get_estimates(["ACME"], AS_OF)
    assert list(df.columns) == F.ESTIMATE_COLUMNS
    r = df.loc["ACME"]
    assert r[F.REVENUE_NTM_EST] == pytest.approx(1.1e9) and r[F.REVENUE_NTM_EST_3M_AGO] == pytest.approx(1.05e9)
    assert r[F.EPS_NTM_EST] == pytest.approx(5.25) and r[F.EPS_NTM_EST_3M_AGO] == pytest.approx(5.0)
    assert r[F.TARGET_PRICE_MEAN] == pytest.approx(120.5)
    assert r[F.LAST_EPS_SURPRISE] == pytest.approx(0.042)  # percent -> fraction
    assert r[F.NEXT_EARNINGS_DATE] == pd.Timestamp("2026-10-28")  # future date allowed for next earnings
    assert pd.isna(r[F.NUM_ANALYSTS]) and pd.isna(r[F.EPS_TTM])
    asofs = sorted(x["properties"]["asOfDate"] for x in fake.sent("IQ_REVENUE_EST", "ACME:"))
    assert asofs == ["07/02/2026", AS_OF_STR]  # as_of - 91 days, absolute dates only
    # each unverifiable estimate leg preflighted once; the lag check reuses those responses (no extra request)
    assert len(fake.sent(None, TEST_ID)) == 4
    assert p.unverified_columns[F.EPS_NTM_EST_3M_AGO].startswith("unverifiable item")
    assert "asOfDate property is not admitted" in p.unverified_columns[F.EPS_NTM_EST_3M_AGO]
    assert F.TARGET_PRICE_MEAN not in p.unverified_columns  # confirmed, units self-evident


def test_historical_as_of_blanks_current_only_items(fake):
    _seed_fundamentals(fake)
    hist = date(2026, 6, 30)
    fake.point[("ACME:", "IQ_TOTAL_EQUITY_TEST")] = "999"
    equity = {"total_equity": {"available": None, "mnemonic": "IQ_TOTAL_EQUITY_TEST", "properties": {"currencyId": "USD"},
                               "to_canonical": 1000000, "status": "confirmed"}}
    p = make(fake, fieldmap={"raw": {"fundamentals": equity}, "api": {"as_of_property": {"admitted": True}}})
    df = p.get_fundamentals(["ACME"], hist)
    assert math.isnan(df.loc["ACME", F.TOTAL_EQUITY])  # no as-of property: would leak today's value
    assert fake.sent("IQ_TOTAL_EQUITY_TEST") == []
    assert fake.sent("IQ_TOTAL_REV", "ACME:")[0]["properties"]["asOfDate"] == "06/30/2026"
    assert df.loc["ACME", F.REVENUE_TTM] == pytest.approx(1e9)  # served: asOfDate admitted in the field map
    warn = " | ".join(p.warnings)
    assert "historical" in warn and "admitted in the field map" in warn and "current-only" in warn


def test_historical_as_of_suspends_point_values_until_the_as_of_property_is_admitted(fake):
    """Regression: every default entry carries asOfDate, so nothing was ever blanked for a historical as_of and the
    UNVERIFIED property alone stood between today's values and a backtest (look-ahead)."""
    _seed_fundamentals(fake)
    fake.point.update({("ACME:", "IQ_PRICE_TARGET"): "50", ("ACME:", "IQ_NEXT_EARNINGS_DATE"): "05/01/2026"})
    hist = date(2026, 6, 30)
    p = make(fake)
    assert p.as_of_admitted is False and p.as_of_property == "asOfDate"
    fu = p.get_fundamentals(["ACME"], hist)
    es = p.get_estimates(["ACME"], hist)
    assert fu.drop(columns=[F.PERIOD_END, F.REPORT_DATE]).isna().all().all()
    assert es.isna().all().all()  # confirmed items (target price, next earnings date) included
    assert fake.input_requests == []  # nothing requested (no preflight either): no GDS budget spent
    warn = " | ".join(p.warnings)
    assert "historical and the asOfDate property is not admitted" in warn
    assert "LEG NOT EVALUATED: capiq raw.fundamentals.total_debt (IQ_TOTAL_DEBT): historical as_of 2026-06-30" in warn
    assert "LEG NOT EVALUATED: capiq raw.estimates.target_price_mean (IQ_PRICE_TARGET)" in warn
    # the latest completed session is not historical: served as before
    assert p.get_fundamentals(["ACME"], AS_OF).loc["ACME", F.TOTAL_DEBT] == pytest.approx(8e8)
    # labels (today's classification) are still served for a historical as_of
    fake.point[("ACME:", "IQ_EXCHANGE")] = "NYSE"
    q = make(fake, tickers=["ACME"])
    u = q.get_universe(None, hist)
    assert u.loc["ACME", F.EXCHANGE] == "NYSE" and math.isnan(u.loc["ACME", F.MARKET_CAP])


@pytest.mark.parametrize("today, as_of, staleness, historical", [
    (date(2026, 10, 2), date(2026, 10, 2), 0, False),  # today
    (date(2026, 10, 2), date(2026, 10, 1), 0, False),  # Friday run: Thursday is the latest completed session
    (date(2026, 10, 2), date(2026, 9, 30), 0, True),
    (date(2026, 10, 5), date(2026, 10, 2), 0, False),  # Monday run: Friday is the latest completed session
    (date(2026, 10, 5), date(2026, 10, 1), 0, True),
    (date(2026, 10, 5), date(2026, 10, 1), 1, False),  # explicit tolerance, in calendar days
])
def test_historical_rule_anchors_on_the_latest_completed_session(fake, today, as_of, staleness, historical):
    """Regression: any as_of within 5 calendar days of today used to receive current-only values (look-ahead)."""
    p = make(fake, today=lambda: today, snapshot_staleness_days=staleness)
    assert p._is_historical(as_of) is historical  # noqa: SLF001


def test_current_only_item_is_served_for_the_latest_session_only(fake):
    fake.point[("ACME:", "IQ_CURRENT_ONLY_TEST")] = "999"
    override = {"raw": {"fundamentals": {"total_equity": {
        "available": None, "mnemonic": "IQ_CURRENT_ONLY_TEST", "properties": {"currencyId": "USD"},
        "to_canonical": 1000000, "status": "confirmed"}}}}
    p = make(fake, today=lambda: date(2026, 10, 5), fieldmap=override)
    assert p.get_fundamentals(["ACME"], date(2026, 10, 2)).loc["ACME", F.TOTAL_EQUITY] == pytest.approx(9.99e8)
    assert math.isnan(p.get_fundamentals(["ACME"], date(2026, 10, 1)).loc["ACME", F.TOTAL_EQUITY])
    assert len(fake.sent("IQ_CURRENT_ONLY_TEST")) == 1  # never requested for the earlier as_of


def test_warnings_are_value_free_and_details_keep_the_values(fake):
    """Warnings reach the run result (and can reach a model); CIQ GDS values are L3, so none is quoted there."""
    _seed_fundamentals(fake)
    fake.point[("ACME:", "IQ_TOTAL_DEBT")] = "7654321987"  # x1e6: implausible scale
    fake.point[("ACME:", "IQ_CASH_OPER")] = "12.34.56"  # non-numeric, but holds digits
    fake.history[("ACME:", "IQ_CLOSEPRICE_ADJ")] = [("09/30/2026", "10"), ("10/01/2026", "10")]
    fake.history[("ACME:", "IQ_VOLUME")] = [("09/30/2026", "3456789"), ("10/01/2026", "3456789")]
    fake.members["^SPX"] = ["AAPL:NasdaqGS", "IQ424242"]
    p = make(fake)
    p.get_fundamentals(["ACME", "BETA"], AS_OF)
    p.get_price_history(["ACME", "NOPE"], date(2026, 9, 1), AS_OF)
    q = make(fake, index="^SPX")
    q.get_universe(None, AS_OF)
    warn = " | ".join(p.warnings + q.warnings)
    for leaked in ("7654321987", "7.654", "7.65e", "12.34.56", "3456789", "3.457", "3.46e", "9999999999", "9.99e",
                   "1e+16", "IQ424242"):
        assert leaked not in warn, leaked
    assert "implausible value outside" in warn and "non-numeric value (withheld" in warn
    assert "implausible median outside" in warn
    assert "1 of 2 ticker(s)" in warn and "NOPE" not in warn
    details = " | ".join(p.details + q.details)
    assert "7654321987" in details and "12.34.56" in details and "median 3.45679e+12" in details
    assert "NOPE" in details and "IQ424242" in details
    assert all(isinstance(d, str) for d in p.details)


def test_admitted_flag_counts_as_unit_verified(fake):
    _seed_fundamentals(fake)
    p = make(fake, strict_units=True, fieldmap={"raw": {"fundamentals": {"total_debt": {"admitted": True}}}})
    assert Capability.FUNDAMENTALS in p.capabilities
    df = p.get_fundamentals(["ACME"], AS_OF)
    assert df.loc["ACME", F.TOTAL_DEBT] == pytest.approx(8e8)
    assert df.drop(columns=[F.TOTAL_DEBT]).isna().all().all()
    # only the admitted item is requested: IQ_TEV_EBITDA alone cannot give EBITDA while IQ_TEV is withheld
    assert {r["mnemonic"] for r in fake.input_requests} == {"IQ_TOTAL_DEBT"}
    with pytest.raises(FieldMapError, match="admitted"):
        CapIQProvider(fieldmap={"raw": {"fundamentals": {"total_debt": {"admitted": "true"}}}})


def test_derived_column_with_a_suspended_input_requests_nothing_else(fake):
    _seed_fundamentals(fake)
    fake.errors[(TEST_ID, "IQ_TOTAL_REV")] = "Invalid Data Item"  # revenue_ttm preflight fails
    p = make(fake)
    df = p.get_fundamentals(["ACME"], AS_OF)
    assert math.isnan(df.loc["ACME", F.REVENUE_TTM_PRIOR_YEAR]) and math.isnan(df.loc["ACME", F.GROSS_PROFIT_TTM])
    # the growth and margin inputs would be wasted GDS requests: neither is sent
    assert fake.sent("IQ_TOTAL_REV_1YR_ANN_GROWTH") == [] and fake.sent("IQ_GROSS_MARGIN") == []
    warn = " | ".join(p.warnings)
    assert "LEG NOT EVALUATED: capiq raw.fundamentals.revenue_ttm_prior_year (derived)" in warn
    assert "LEG NOT EVALUATED: capiq raw.fundamentals.gross_profit_ttm (derived)" in warn
    assert df.loc["ACME", F.EBITDA_TTM] == pytest.approx(5e8)


# =============================================================================================
# Prices
# =============================================================================================


def test_price_history_request_panel_units_and_lookahead(fake):
    fake.history[("ACME:", "IQ_CLOSEPRICE_ADJ")] = [("09/28/2026", "10.0"), ("09/29/2026", "10.5"), ("09/30/2026", "11"),
                                                    ("10/01/2026", "12"), ("10/02/2026", "99")]
    fake.history[("ACME:", "IQ_VOLUME")] = [("09/28/2026", "1.5"), ("09/29/2026", "2.0"), ("09/30/2026", "1.0"),
                                           ("10/01/2026", "3.25")]
    fake.history[("ACME:", "IQ_HIGHPRICE_ADJ")] = [("09/30/2026", "11.5"), ("10/01/2026", "12.4")]
    fake.history[("BETA:", "IQ_CLOSEPRICE_ADJ")] = [("09/29/2026", "20"), ("09/30/2026", "Data Unavailable")]
    fake.errors[("BETA:", "IQ_VOLUME")] = "Invalid Identifier"
    del fake.history[(TEST_ID, "IQ_LOWPRICE_ADJ")]  # unverifiable low fails its preflight
    p = make(fake)
    panel = p.get_price_history(["ACME", "BETA"], date(2026, 9, 1), AS_OF)
    assert isinstance(panel, PricePanel)
    idx = pd.DatetimeIndex(["2026-09-28", "2026-09-29", "2026-09-30", "2026-10-01"])
    assert list(panel.close.index) == list(idx)  # the 10/02 row (after end) is dropped
    assert list(panel.close.columns) == ["ACME", "BETA"]
    assert list(panel.close["ACME"]) == [10.0, 10.5, 11.0, 12.0]
    assert panel.close["BETA"].isna().tolist() == [True, False, True, True] and panel.close.loc["2026-09-29", "BETA"] == 20.0
    assert list(panel.volume["ACME"]) == [1.5e6, 2.0e6, 1.0e6, 3.25e6]  # CapIQ millions -> shares
    assert panel.volume["BETA"].isna().all()
    assert panel.high.loc["2026-10-01", "ACME"] == 12.4 and math.isnan(panel.high.loc["2026-09-28", "ACME"])
    assert panel.low.isna().all().all()
    assert fake.sent("IQ_CLOSEPRICE_ADJ", "ACME:") == [
        {"function": "GDSHE", "identifier": "ACME:", "mnemonic": "IQ_CLOSEPRICE_ADJ",
         "properties": {"startDate": "09/01/2026", "endDate": AS_OF_STR}}]
    assert fake.sent("IQ_LOWPRICE_ADJ") == [
        {"function": "GDSHE", "identifier": TEST_ID, "mnemonic": "IQ_LOWPRICE_ADJ",
         "properties": {"startDate": "09/21/2026", "endDate": AS_OF_STR}}]  # preflight only, never bulk
    warn = " | ".join(p.warnings)
    assert "LEG NOT EVALUATED" in warn and "IQ_LOWPRICE_ADJ" in warn
    assert "Invalid Identifier" in warn


def test_price_history_implausible_scale_is_blanked(fake):
    fake.history[("ACME:", "IQ_CLOSEPRICE_ADJ")] = [("09/30/2026", "10"), ("10/01/2026", "10")]
    fake.history[("ACME:", "IQ_VOLUME")] = [("09/30/2026", "3500000"), ("10/01/2026", "4100000")]  # absolute shares
    p = make(fake)
    panel = p.get_price_history(["ACME"], date(2026, 9, 1), AS_OF)
    assert panel.volume["ACME"].isna().all()  # x1e6 would give 3.5e12 shares: implausible, blanked
    assert any("implausible median" in w for w in p.warnings)


def test_price_history_no_data_raises(fake):
    with pytest.raises(ProviderError, match="no IQ_CLOSEPRICE_ADJ history"):
        make(fake).get_price_history(["NOPE"], date(2026, 9, 1), AS_OF)


def test_benchmark_history_and_fallback(fake):
    p = make(fake)
    ser = p.get_benchmark_history(date(2026, 9, 20), AS_OF)
    assert ser.name == "^SPX" and (ser == 5700.5).all() and ser.index.max() <= pd.Timestamp(AS_OF)
    assert fake.sent("IQ_CLOSEPRICE", "^SPX")[0] == {"function": "GDSHE", "identifier": "^SPX", "mnemonic": "IQ_CLOSEPRICE",
                                                    "properties": {"startDate": "09/20/2026", "endDate": AS_OF_STR}}
    fake.errors[("^SPX", "IQ_CLOSEPRICE")] = "Invalid Identifier"
    q = make(fake)
    ser2 = q.get_benchmark_history(date(2026, 9, 20), AS_OF)
    assert ser2.name == "SPY:" and (ser2 == 570.1).all()
    assert any("fallback SPY:" in w for w in q.warnings)
    fake.history[("QQQ:", "IQ_CLOSEPRICE_ADJ")] = [("10/01/2026", "480")]
    assert q.get_benchmark_history(date(2026, 9, 20), AS_OF, symbol="QQQ").tolist() == [480.0]
    fake.errors[("SPY:", "IQ_CLOSEPRICE_ADJ")] = "Invalid Identifier"
    with pytest.raises(ProviderError, match="benchmark history unavailable"):
        make(fake).get_benchmark_history(date(2026, 9, 20), AS_OF)


# =============================================================================================
# Not available from S&P
# =============================================================================================


def test_options_and_short_interest_are_nan_frames_without_requests(fake):
    p = make(fake)
    opt = p.get_options_summary(["ACME", "BETA"], AS_OF)
    assert list(opt.columns) == F.OPTIONS_COLUMNS and list(opt.index) == ["ACME", "BETA"]
    assert opt.isna().all().all()
    si = p.get_short_interest(["ACME"], AS_OF)
    assert list(si.columns) == F.SHORT_INTEREST_COLUMNS and si.isna().all().all()
    assert fake.requests == []  # no request, no login
    warn = " | ".join(p.warnings)
    assert "options not available from S&P" in warn and "Implied vol" in warn
    assert "short_interest not available from S&P" in warn and "Do not ship" in warn
    assert Capability.OPTIONS not in p.capabilities and Capability.SHORT_INTEREST not in p.capabilities


# =============================================================================================
# Vendor feature values (reconciliation)
# =============================================================================================


def test_vendor_features_in_catalog_units(fake):
    fake.point.update({("ACME:", "IQ_MARKETCAP"): "250000", ("ACME:", "IQ_TEV_EBITDA"): "12.5",
                       ("ACME:", "IQ_MARKET_CAP_LFCF"): "20", ("ACME:", "IQ_INDUSTRY_SECTOR"): "Industrials",
                       ("ACME:", "IQ_TOTAL_REV_1YR_ANN_GROWTH"): "-3.5"})
    p = make(fake)
    df = p.get_vendor_features(["ACME"], AS_OF, ["market_cap_usd_bn", "ev_to_ebitda", "fcf_yield_pct", "gics_sector",
                                                 "revenue_growth_yoy_pct", "iv_30d_pct", "rsi_14"])
    r = df.loc["ACME"]
    assert r["market_cap_usd_bn"] == pytest.approx(250.0)  # USD mn x 0.001
    assert r["ev_to_ebitda"] == pytest.approx(12.5)
    assert r["fcf_yield_pct"] == pytest.approx(5.0)  # 100 / IQ_MARKET_CAP_LFCF
    assert r["gics_sector"] == "Industrials"
    assert r["revenue_growth_yoy_pct"] == pytest.approx(-3.5)
    assert math.isnan(r["iv_30d_pct"]) and math.isnan(r["rsi_14"])
    assert fake.sent("IQ_MARKET_CAP_LFCF", TEST_ID)  # unverifiable input preflighted
    assert any("iv_30d_pct" in w and "rsi_14" in w for w in p.warnings)
    with pytest.raises(ValueError, match="not catalog features"):
        p.get_vendor_features(["ACME"], AS_OF, ["bogus"])
    default = make(fake).get_vendor_features(["ACME"], AS_OF)
    assert "market_cap_usd_bn" in default.columns and "iv_30d_pct" not in default.columns and "sma_50" not in default.columns


# =============================================================================================
# Transcripts (Kensho)
# =============================================================================================

RAW_TRANSCRIPT = (
    "Operator: Good morning and welcome. After the speakers' remarks there will be a question-and-answer session.\n"
    "Martina Cheung: Thank you. Revenue grew 10%.\n"
    "We raised guidance for the full year.\n"
    "Operator: We will now begin the question-and-answer session. Our first question comes from Jane Doe.\n"
    "Jane Doe: Can you talk about margins?\n"
    "Martina Cheung: Margins expanded 150 basis points.\n"
)


class FakeKenshoTools:
    """Tool-style Kensho client (VENDOR_REFERENCE 3.3 transcripts row)."""

    def __init__(self, when: str = "2026-09-15T12:30:00Z") -> None:
        self.when = when
        self.calls: list[tuple[str, Any]] = []

    def get_latest_earnings_from_identifiers(self, identifiers: list[str]) -> dict[str, Any]:
        self.calls.append(("latest", list(identifiers)))
        return {"results": {"SPGI": {"name": "S&P Global Inc., Q3 2026 Earnings Call", "key_dev_id": 1234567,
                                     "datetime": self.when}}, "errors": {}}

    def get_transcript_from_key_dev_id(self, key_dev_id: int) -> dict[str, Any]:
        self.calls.append(("transcript", key_dev_id))
        return {"transcript": RAW_TRANSCRIPT}


def _g2() -> DataBoundary:
    return DataBoundary(provider="capiq", allowed_document_kinds={DocumentKind.TRANSCRIPT}, allow_numeric_features=False,
                        note=G2_NOTE)


def test_transcripts_denied_by_default_boundary_without_calling_kensho(fake):
    k = FakeKenshoTools()
    p = make(fake, kensho=k)
    assert p.get_documents("SPGI", {DocumentKind.TRANSCRIPT}, date(2026, 9, 1), AS_OF) == []
    assert k.calls == []
    assert Capability.TRANSCRIPTS not in p.capabilities
    assert any("G2" in w for w in p.warnings)


def test_transcripts_need_an_injected_client(fake):
    p = make(fake, boundary=_g2())
    assert p.get_documents("SPGI", {DocumentKind.TRANSCRIPT}, date(2026, 9, 1), AS_OF) == []
    assert any("no Kensho client" in w for w in p.warnings)
    assert any("no Kensho client" in d for d in p.diagnostics())


def test_transcripts_tool_route_with_g2_boundary(fake):
    k = FakeKenshoTools()
    p = make(fake, kensho=k, boundary=_g2())
    assert Capability.TRANSCRIPTS in p.capabilities
    docs = p.get_documents("SPGI", {DocumentKind.TRANSCRIPT, DocumentKind.NEWS}, date(2026, 9, 1), AS_OF)
    assert k.calls == [("latest", ["SPGI"]), ("transcript", 1234567)]
    assert len(docs) == 1
    d = docs[0]
    assert d.doc_id == "kensho-transcript-1234567" and d.kind == DocumentKind.TRANSCRIPT and d.ticker == "SPGI"
    assert d.published_at == datetime(2026, 9, 15, 12, 30)
    assert d.title == "S&P Global Inc., Q3 2026 Earnings Call"
    assert [s.section for s in d.segments] == ["prepared_remarks", "prepared_remarks", "qa", "qa", "qa"]
    assert [s.speaker for s in d.segments] == ["Operator", "Martina Cheung", "Operator", "Jane Doe", "Martina Cheung"]
    assert d.segments[1].text == "Thank you. Revenue grew 10%. We raised guidance for the full year."
    assert "Jane Doe: Can you talk about margins?" in d.text
    md = d.metadata
    assert (md["key_dev_id"], md["licence_class"], md["gate"], md["vendor"]) == ("1234567", "L1", "G2", "capiq")
    assert len(md["paragraph_sha256"].split(",")) == 5 and len(md["text_sha256"]) == 64
    assert any("news documents are not served" in w for w in p.warnings)
    assert fake.requests == []  # transcripts never touch GDS
    assert any(q.startswith("kensho:get_latest_earnings_from_identifiers") for q in p.query_log)


def test_transcripts_outside_the_window_are_not_returned(fake):
    k = FakeKenshoTools(when="2026-09-15T12:30:00Z")
    p = make(fake, kensho=k, boundary=_g2())
    assert p.get_documents("SPGI", {DocumentKind.TRANSCRIPT}, date(2026, 9, 20), AS_OF) == []
    assert k.calls == [("latest", ["SPGI"])]  # transcript text is not even fetched
    assert any("outside" in w for w in p.warnings)


def test_transcripts_object_route_kfinance_client(fake):
    earnings = SimpleNamespace(name="SPGI Q3 2026", key_dev_id=777,
                               datetime=datetime(2026, 9, 15, 12, 30, tzinfo=timezone.utc),
                               transcript=SimpleNamespace(raw=RAW_TRANSCRIPT))
    seen: list[str] = []

    def ticker(sym: str) -> Any:
        seen.append(sym)
        return SimpleNamespace(company=SimpleNamespace(latest_earnings=earnings))

    p = make(fake, kensho=SimpleNamespace(ticker=ticker), boundary=_g2())
    docs = p.get_documents("SPGI", {DocumentKind.TRANSCRIPT}, date(2026, 9, 1), AS_OF, limit=5)
    assert seen == ["SPGI"]
    assert [d.doc_id for d in docs] == ["kensho-transcript-777"]
    assert docs[0].published_at == datetime(2026, 9, 15, 12, 30)
    assert len(docs[0].segments) == 5


def test_kensho_failure_degrades_to_no_transcript(fake):
    class Denied(FakeKenshoTools):
        def get_transcript_from_key_dev_id(self, key_dev_id: int) -> Any:
            raise PermissionError("TranscriptsPermission required")

    p = make(fake, kensho=Denied(), boundary=_g2())
    assert p.get_documents("SPGI", {DocumentKind.TRANSCRIPT}, date(2026, 9, 1), AS_OF) == []
    assert any("TranscriptsPermission" in w for w in p.warnings)


def test_transcript_provenance_metadata(fake):
    p = make(fake, kensho=FakeKenshoTools(), boundary=_g2())
    md = p.get_documents("SPGI", {DocumentKind.TRANSCRIPT}, date(2026, 9, 1), AS_OF)[0].metadata
    # ADR section 7: every datum carries (vendor, channel, licence_class, field, as_of, mnpi_class)
    assert {k: md[k] for k in ("vendor", "channel", "licence_class", "field", "as_of", "mnpi_class", "gate")} == {
        "vendor": "capiq", "channel": "kensho", "licence_class": "L1", "field": "transcript",
        "as_of": AS_OF.isoformat(), "mnpi_class": "public", "gate": "G2"}
    assert all(isinstance(v, str) for v in md.values())


_REC = {"name": "S&P Global Q3 2026", "key_dev_id": 4242, "datetime": "2026-09-15T12:30:00Z"}


@pytest.mark.parametrize("payload, found", [
    (_REC, True),  # a bare record
    ({"SPGI": _REC}, True),
    ({"spgi": _REC}, True),
    ({"SPGI:NYSE": _REC}, True),
    ({"results": {"SPGI US": _REC}, "errors": {}}, True),
    ({"data": [_REC]}, True),
    ({"earnings_call": _REC}, True),  # a single snake_case wrapper field
    ({"MSFT": _REC}, False),  # filed under another company: never taken for SPGI's call
    ({"results": {"MSFT": _REC}, "errors": {}}, False),
    ({"results": {}, "errors": {"SPGI": "no permission"}}, False),
])
def test_kensho_record_must_belong_to_the_requested_identifier(fake, payload, found):
    class Tools(FakeKenshoTools):
        def get_latest_earnings_from_identifiers(self, identifiers: list[str]) -> Any:
            self.calls.append(("latest", list(identifiers)))
            return payload

    k = Tools()
    p = make(fake, kensho=k, boundary=_g2())
    docs = p.get_documents("SPGI", {DocumentKind.TRANSCRIPT}, date(2026, 9, 1), AS_OF)
    assert [d.doc_id for d in docs] == (["kensho-transcript-4242"] if found else [])
    if not found:
        assert k.calls == [("latest", ["SPGI"])]  # no transcript fetched for a record that is not SPGI's
        assert any("no latest earnings call for SPGI" in w for w in p.warnings)


def test_kensho_methods_are_called_once_whatever_their_signature(fake):
    calls: list[tuple[str, Any]] = []

    class Positional:
        def get_latest_earnings_from_identifiers(self, ids, /):  # positional-only parameter
            calls.append(("latest", ids))
            return [_REC]

        def get_transcript_from_key_dev_id(self, kd):  # differently named parameter
            calls.append(("transcript", kd))
            return RAW_TRANSCRIPT

    p = make(fake, kensho=Positional(), boundary=_g2())
    assert [d.doc_id for d in p.get_documents("SPGI", {DocumentKind.TRANSCRIPT}, date(2026, 9, 1), AS_OF)] == [
        "kensho-transcript-4242"]
    assert calls == [("latest", ["SPGI"]), ("transcript", 4242)]

    class Buggy(FakeKenshoTools):
        def get_transcript_from_key_dev_id(self, key_dev_id: int) -> Any:
            self.calls.append(("transcript", key_dev_id))
            raise TypeError("unsupported argument type inside the SDK")

    b = Buggy()
    q = make(fake, kensho=b, boundary=_g2())
    assert q.get_documents("SPGI", {DocumentKind.TRANSCRIPT}, date(2026, 9, 1), AS_OF) == []
    assert b.calls == [("latest", ["SPGI"]), ("transcript", 1234567)]  # a TypeError never triggers a second call
    assert any("TypeError" in w for w in q.warnings)


def test_documents_without_transcript_kind_and_limit(fake):
    k = FakeKenshoTools()
    p = make(fake, kensho=k, boundary=_g2())
    assert p.get_documents("SPGI", {DocumentKind.FILING}, date(2026, 9, 1), AS_OF) == []
    assert k.calls == []  # filings are not served: Kensho is not called
    assert any("filing documents are not served" in w and "EDGAR" in w for w in p.warnings)
    assert p.get_documents("SPGI", {DocumentKind.TRANSCRIPT}, date(2026, 9, 1), AS_OF, limit=0) == []


def test_parse_transcript_components_and_headers():
    segs = parse_transcript("Prepared remarks text without speaker\nCEO Name: Hello.\nQuestion-and-Answer Session\n"
                            "Analyst One: Why?\nCEO Name: Because.")
    assert [(s.speaker, s.section) for s in segs] == [
        ("", "prepared_remarks"), ("CEO Name", "prepared_remarks"), ("Analyst One", "qa"), ("CEO Name", "qa")]
    assert parse_transcript("") == []
    assert parse_transcript("Operator: Our first question comes from X.")[0].role == "Operator"


# =============================================================================================
# Field self-check (verify_fields)
# =============================================================================================


def test_verify_fields_reports_each_mapped_field(fake, fm):
    fake.point[(TEST_ID, "IQ_MARKETCAP")] = "215000"
    fake.errors[(TEST_ID, "IQ_YEARHIGH")] = "Invalid Data Item"
    fake.point[(TEST_ID, "IQ_TOTAL_DEBT")] = "999999999"  # x1e6 = 1e15: implausible scale
    p = make(fake)
    checks = p.verify_fields(as_of=AS_OF)
    assert all(isinstance(c, FieldCheck) for c in checks)
    expected = {label for label, _, _ in _leg_entries(fm)}
    expected |= {f"history.{f}" for f in fm["history"]["fields"]} | {"benchmark", "benchmark.fallback"}
    expected |= {"api.as_of_property"}
    assert {c.field for c in checks} == expected
    by = {c.field: c for c in checks}
    mc = by["raw.universe.market_cap"]
    assert mc.ok and mc.status_in_map == "confirmed" and mc.returned_value == "215000"
    assert mc.canonical_value == pytest.approx(2.15e11) and "x1e+06" in mc.note and "UNVERIFIED" in mc.note
    assert mc.identifier == TEST_ID and mc.properties == {"currencyId": "USD", "asOfDate": AS_OF_STR}
    yh = by["features.high_52w"]
    assert not yh.ok and "Invalid Data Item" in yh.note and yh.status_in_map == "unverifiable"
    assert "UNVERIFIABLE" in yh.note
    debt = by["raw.fundamentals.total_debt"]
    assert not debt.ok and "implausible" in debt.note
    close = by["history.close"]
    assert close.ok and close.function == "GDSHE" and close.properties == {"startDate": "09/21/2026", "endDate": AS_OF_STR}
    assert by["benchmark"].identifier == "^SPX" and by["benchmark.fallback"].identifier == "SPY:"
    assert by["raw.fundamentals.ebitda_ttm.inputs[1]"].mnemonic == "IQ_TEV_EBITDA"
    row = mc.to_log_row(reviewer="qa")
    assert row["vendor"] == "capiq" and row["item"] == "IQ_MARKETCAP" and row["test_ticker"] == TEST_ID
    assert row["as_of"] == AS_OF.isoformat() and row["reviewer"] == "qa"
    n = p.requests_today
    p.verify_fields(as_of=AS_OF)  # fresh requests every time (no cache)
    assert p.requests_today == 2 * n
    other = p.verify_fields("ACME", as_of=AS_OF)
    assert {c.identifier for c in other} == {"ACME:", "^SPX", "SPY:"}


def test_verify_fields_includes_constituents_when_index_is_set(fake):
    fake.members["^SPX"] = ["AAPL:NasdaqGS", "MSFT:"]
    checks = make(fake, index="^SPX").verify_fields()
    cons = [c for c in checks if c.field == "universe.constituents"]
    assert len(cons) == 1 and cons[0].ok and cons[0].properties == {"StartRank": 1, "EndRank": 5}
    assert cons[0].returned_value == "AAPL:NasdaqGS"


# =============================================================================================
# Integration with the local feature engine and screen
# =============================================================================================


def test_feature_engine_and_local_screen_run_on_capiq_output(fake):
    from aitrading.screen.engine import run_screen
    from aitrading.screen.features import FeatureEngine
    from aitrading.screen.spec import Condition, RankFactor, ScreenSpec

    _seed_fundamentals(fake)
    days = _bdays(date(2025, 6, 2), AS_OF)
    for i, ident in enumerate(["ACME:", "BETA:"]):
        fake.history[(ident, "IQ_CLOSEPRICE_ADJ")] = [(d, f"{50 + 0.05 * k * (1 + i):.4f}") for k, d in enumerate(days)]
        fake.history[(ident, "IQ_VOLUME")] = [(d, "1.5") for d in days]
        fake.point.update({(ident, "IQ_MARKETCAP"): "5000", (ident, "IQ_EXCHANGE"): "NYSE"})
    fake.history[("^SPX", "IQ_CLOSEPRICE")] = [(d, f"{5000 + k:.1f}") for k, d in enumerate(days)]
    p = make(fake, tickers=["ACME", "BETA"])
    universe = p.get_universe(None, AS_OF)
    frame = FeatureEngine(p).build(universe, AS_OF).frame
    acme = frame.loc["ACME"]
    assert acme["market_cap_usd_bn"] == pytest.approx(5.0)  # 5000 USD mn
    assert acme["fcf_yield_pct"] == pytest.approx(3.4)  # 170 / 5000, both CapIQ millions
    assert acme["revenue_growth_yoy_pct"] == pytest.approx(25.0)  # reproduces IQ_TOTAL_REV_1YR_ANN_GROWTH
    assert acme["gross_margin_pct"] == pytest.approx(40.0)
    assert acme["sma_50"] > acme["sma_200"] and math.isfinite(acme["rel_strength_3m_pp"])
    assert acme["avg_dollar_volume_20d_usd_mn"] > 5  # volume x1e6 shares
    assert math.isnan(acme["iv_30d_pct"]) and math.isnan(acme["short_interest_pct_float"])  # not from S&P
    assert math.isnan(frame.loc["BETA", "fcf_yield_pct"])  # 'CapabilityNeeded' -> NaN, never 0
    spec = ScreenSpec(
        name="capiq-smoke", observation="cash-generative mid caps",
        universe=UniverseSpec(security_types=[]),  # no verified S&P security-type item
        conditions=[Condition(feature="market_cap_usd_bn", op=">=", value=2),
                    Condition(feature="fcf_yield_pct", op=">=", value=3)],
        ranking=[RankFactor(feature="fcf_yield_pct", direction="higher_is_better")],
    )
    out = run_screen(spec, frame)
    assert out.survivors == ["ACME"]  # BETA's missing FCF yield excludes it
    assert [s.missing_data for s in out.funnel][-1] == 1


# =============================================================================================
# Identifiers
# =============================================================================================


@pytest.mark.parametrize("ident, ok", [("IBM:NYSE", True), ("IBM:", True), ("IQ24937", True), ("IQT2630413", True),
                                        ("I_US4592001014", True), ("CSP_459200101", True), ("GV006066", True),
                                        ("^SPX", True), ("IBM", False), ("IQV", False)])
def test_is_identifier(ident, ok):
    assert is_identifier(ident) is ok


def test_identifier_ticker_round_trip():
    assert ticker_to_identifier("IBM") == "IBM:"
    assert ticker_to_identifier("brk-b") == "BRK.B:"
    assert ticker_to_identifier("IBM:NYSE") == "IBM:NYSE"
    assert identifier_to_ticker("BRK.B:NYSE") == "BRK-B"
    assert identifier_to_ticker("IBM:") == "IBM"
    assert identifier_to_ticker("AAPL") == "AAPL"
    assert identifier_to_ticker("IQ24937") is None and identifier_to_ticker("^SPX") is None


# =============================================================================================
# Review regressions (point-in-time, admission gate G5, value-free warnings, citations)
# =============================================================================================


def _est_props(as_of: str) -> str:
    return json.dumps({"periodType": "IQ_NTM", "currencyId": "USD", "asOfDate": as_of}, sort_keys=True)


def test_lagged_estimates_suspended_when_gds_ignores_as_of_date(fake):
    """Regression: a GDS that ignores asOfDate returned the current estimate as the 3-month-ago one, so the EPS
    revision came out as exactly 0% (a fabricated value passing revision predicates) with no warning."""
    fake.point.update({("ACME:", "IQ_EPS_EST"): "5.0", ("ACME:", "IQ_REVENUE_EST"): "1000",
                       ("BETA:", "IQ_EPS_EST"): "3.0", ("BETA:", "IQ_REVENUE_EST"): "800"})
    p = make(fake)  # the fake (and the seeded test identifier) answer regardless of asOfDate
    df = p.get_estimates(["ACME", "BETA"], AS_OF)
    assert df[F.EPS_NTM_EST].tolist() == [5.0, 3.0] and df[F.REVENUE_NTM_EST].tolist() == [1e9, 8e8]
    assert df[F.EPS_NTM_EST_3M_AGO].isna().all() and df[F.REVENUE_NTM_EST_3M_AGO].isna().all()
    assert fake.sent("IQ_EPS_EST", "ACME:") == fake.sent("IQ_EPS_EST", "ACME:")[:1]  # the 3m-ago leg never requested
    warn = " | ".join(p.warnings)
    assert "LEG NOT EVALUATED: capiq raw.estimates.eps_ntm_est_3m_ago (IQ_EPS_EST): lag check on IBM:NYSE failed" in warn
    assert "appears to ignore the asOfDate property" in warn
    from aitrading.fundamental.features import compute_fundamental_features

    uni = pd.DataFrame({F.MARKET_CAP: [1e10, 1e10]}, index=pd.Index(["ACME", "BETA"], name=F.TICKER))
    feats = compute_fundamental_features(uni, None, df, pd.Series([50.0, 30.0], index=uni.index), AS_OF)
    assert feats["eps_revision_3m_pct"].isna().all()  # NaN, never a fabricated 0%


def test_lagged_estimates_serve_time_check_and_admission(fake):
    # the test identifier honours asOfDate, but every name's 3m-ago value equals its current value
    fake.point_by_props.update({(TEST_ID, "IQ_EPS_EST", _est_props("07/02/2026")): "10.5",
                                (TEST_ID, "IQ_REVENUE_EST", _est_props("07/02/2026")): "61000"})
    fake.point.update({("ACME:", "IQ_EPS_EST"): "5.0", ("BETA:", "IQ_EPS_EST"): "3.0"})
    p = make(fake)
    df = p.get_estimates(["ACME", "BETA"], AS_OF)
    assert df[F.EPS_NTM_EST].tolist() == [5.0, 3.0] and df[F.EPS_NTM_EST_3M_AGO].isna().all()
    assert any("every value at asOfDate 07/02/2026 equals the as_of value" in w for w in p.warnings)
    # one name cannot show that the property was ignored (an unchanged estimate is possible): served
    assert make(fake).get_estimates(["ACME"], AS_OF).loc["ACME", F.EPS_NTM_EST_3M_AGO] == 5.0
    # preflight=False cannot run the lag check: the lagged legs are suspended, not trusted
    q = make(fake, preflight=False)
    assert q.get_estimates(["ACME"], AS_OF)[F.EPS_NTM_EST_3M_AGO].isna().all()
    assert any("preflight=False, so the lag check cannot run" in w for w in q.warnings)
    # an admitted as-of property skips the lag preflight (the serve-time check still runs)
    fake.point_by_props.update({("ACME:", "IQ_EPS_EST", _est_props("07/02/2026")): "4.5"})
    r = make(fake, fieldmap={"api": {"as_of_property": {"admitted": True}}})
    assert r.get_estimates(["ACME", "BETA"], AS_OF)[F.EPS_NTM_EST_3M_AGO].tolist() == [4.5, 3.0]
    assert "lagged value" not in r.unverified_columns.get(F.EPS_NTM_EST_3M_AGO, "")


def test_verify_fields_checks_the_as_of_property(fake):
    mc = json.dumps({"currencyId": "USD", "asOfDate": "07/02/2026"}, sort_keys=True)
    p = make(fake)
    same = {c.field: c for c in p.verify_fields(as_of=AS_OF)}["api.as_of_property"]
    assert not same.ok and "appears to ignore" in same.note and same.mnemonic == "IQ_MARKETCAP"
    fake.point_by_props[(TEST_ID, "IQ_MARKETCAP", mc)] = "190000"
    fake.point[(TEST_ID, "IQ_MARKETCAP")] = "215000"
    diff = {c.field: c for c in p.verify_fields(as_of=AS_OF)}["api.as_of_property"]
    assert diff.ok and diff.returned_value == "215000" and diff.properties == {"currencyId": "USD", "asOfDate": AS_OF_STR}
    assert "second source" in diff.note and diff.to_log_row()["item"] == "IQ_MARKETCAP"
    # the as_of request coincides with raw.universe.market_cap (de-duplicated); the probe adds the as_of - 91 days one
    assert fake.sent("IQ_MARKETCAP", TEST_ID)[-1]["properties"] == {"currencyId": "USD", "asOfDate": "07/02/2026"}


def test_volume_scale_checks_blank_inflated_thin_names(fake):
    """Regression: raw IQ_VOLUME '8000' at a close of 50 became 8e9 shares (USD 400bn a day), passed the liquidity
    floor and produced no warning at all."""
    days = _bdays(date(2026, 8, 3), AS_OF)
    for ident, vol in (("ACME:", "8000"), ("BIG:", "2000"), ("THIN:", "500"), ("OK:", "1.5")):
        fake.history[(ident, "IQ_CLOSEPRICE_ADJ")] = [(d, "100" if ident == "BIG:" else "50") for d in days]
        fake.history[(ident, "IQ_VOLUME")] = [(d, vol) for d in days]
    fake.point.update({("THIN:", "IQ_MARKETCAP"): "500", ("OK:", "IQ_MARKETCAP"): "5000"})
    p = make(fake, tickers=["THIN", "OK"])
    p.get_universe(None, AS_OF)  # caches the market caps used by the turnover check (no extra request)
    n = len(fake.input_requests)
    panel = p.get_price_history(["ACME", "BIG", "THIN", "OK"], date(2026, 8, 3), AS_OF)
    assert len(fake.input_requests) - n == 4 * 5 + 3  # 5 fields x 4 tickers + 3 OHLC preflights; no market cap
    assert panel.volume["ACME"].isna().all()  # 8e9 shares/day: median outside [0, 5e9]
    assert panel.volume["BIG"].isna().all()  # 2e9 shares x USD 100: USD 200bn a day
    assert panel.volume["THIN"].isna().all()  # 5e8 shares x USD 50 vs a USD 500mn market cap: 50x turnover a day
    assert panel.volume["OK"].tolist() == [1.5e6] * len(days)
    warn = " | ".join(p.warnings)
    assert "implausible median dollar volume" in warn and "implausible median daily turnover" in warn
    assert "G5 OVERRIDE (strict_units=False): capiq history: units UNVERIFIED for history.volume (IQ_VOLUME" in warn
    assert "split adjustment UNVERIFIED" in warn
    assert p.unverified_columns[F.VOLUME].startswith("units UNVERIFIED")
    assert F.CLOSE not in p.unverified_columns
    for leaked in ("8000", "8e+09", "2.5e+10", "5e+08"):
        assert leaked not in warn, leaked
    assert any("THIN" in d and "market cap" in d for d in p.details)


def test_unverified_columns_and_g5_override_warning(fake):
    _seed_fundamentals(fake)
    _seed_universe(fake)
    p = make(fake, tickers=["BRK-B"], fieldmap={"raw": {"fundamentals": {"total_debt": {"admitted": True}}}})
    p.get_universe(None, AS_OF)
    p.get_fundamentals(["ACME"], AS_OF)
    uc = p.unverified_columns
    assert uc[F.MARKET_CAP] == "units UNVERIFIED (USD millions (ASSUMED))"
    assert uc[F.REVENUE_TTM].startswith("units UNVERIFIED") and "unverifiable item" in uc[F.REVENUE_TTM]
    assert "input IQ_TOTAL_REV_1YR_ANN_GROWTH: units UNVERIFIED" in uc[F.REVENUE_TTM_PRIOR_YEAR]
    assert "derived from revenue_ttm" in uc[F.REVENUE_TTM_PRIOR_YEAR]
    assert uc[F.COUNTRY] == "unverifiable item, not admitted" and uc[F.GICS_SECTOR] == "unverifiable item, not admitted"
    assert F.TOTAL_DEBT not in uc  # admitted (G5)
    assert F.EXCHANGE not in uc  # confirmed item, no units
    assert F.NET_INCOME_TTM not in uc  # never served
    assert all(not re.search(r"\d{3}", v) for v in uc.values())  # value-free
    warn = " | ".join(p.warnings)
    assert "G5 OVERRIDE (strict_units=False): capiq raw.universe: units UNVERIFIED" in warn
    assert "provider.unverified_columns" in warn and "z-scores only" in warn


def test_leg_missing_for_every_name_gets_a_banner(fake):
    """Regression: placeholders ('Data Unavailable', 'NM') or empty rows for every name left a column all-NaN with
    no LEG NOT EVALUATED banner (only an all-ErrMsg leg had one)."""
    p = make(fake)
    df = p.get_fundamentals(["ACME", "BETA"], AS_OF)  # the fake answers 'Data Unavailable' by default
    assert df[F.REVENUE_TTM].isna().all() and df[F.CFO_TTM].isna().all()
    warn = " | ".join(p.warnings)
    for col, mn in ((F.REVENUE_TTM, "IQ_TOTAL_REV"), (F.CFO_TTM, "IQ_CASH_OPER"), (F.TOTAL_DEBT, "IQ_TOTAL_DEBT"),
                    (F.SHARES_OUTSTANDING, "IQ_SHARESOUTSTANDING")):
        assert (f"LEG NOT EVALUATED: capiq raw.fundamentals.{col} ({mn}): no usable value for any of 2 name(s) "
                "(every value missing") in warn
    assert "LEG NOT EVALUATED: capiq raw.fundamentals.ebitda_ttm (derived): no usable value for any of 2" in warn
    assert not p.unverified_columns  # nothing was served
    fake.errors[("ACME:", "IQ_PRICE_TARGET")] = "Invalid Identifier"
    fake.point[("BETA:", "IQ_PRICE_TARGET")] = "abc"
    q = make(fake)
    q.get_estimates(["ACME", "BETA"], AS_OF)
    assert any(w.startswith("LEG NOT EVALUATED: capiq raw.estimates.target_price_mean (IQ_PRICE_TARGET): no usable value")
               and "Invalid Identifier" in w and "non-numeric 'abc'" in w for w in q.warnings)
    # a history field missing for every ticker
    fake.history[("ACME:", "IQ_CLOSEPRICE_ADJ")] = [("09/30/2026", "10"), ("10/01/2026", "10")]
    r = make(fake)
    r.get_price_history(["ACME"], date(2026, 9, 1), AS_OF)
    assert any(w.startswith("LEG NOT EVALUATED: capiq history.volume (IQ_VOLUME): no usable value for any of 1")
               for w in r.warnings)


def test_capiq_sector_labels_are_mapped_to_gics_names(fake):
    """Regression: CapIQ's own classification was served as a GICS label and used as an exclude_sectors filter, so a
    sector CapIQ spells differently from GICS was not matched and its names passed."""
    from aitrading.screen.engine import apply_universe

    fake.point.update({("AAA:", "IQ_INDUSTRY_SECTOR"): "Financial", ("BBB:", "IQ_INDUSTRY_SECTOR"): "healthcare",
                       ("CCC:", "IQ_INDUSTRY_SECTOR"): "Financials", ("DDD:", "IQ_INDUSTRY_SECTOR"): "Industrials",
                       ("AAA:", "IQ_INDUSTRY"): "Oil, Gas and Consumable Fuels", ("BBB:", "IQ_INDUSTRY"): "Diversified Banks"})
    p = make(fake, tickers=["AAA", "BBB", "CCC", "DDD"])
    spec = UniverseSpec(country="", security_types=[], min_price=None, min_avg_dollar_volume_usd_mn=None,
                        exclude_sectors=["Financials"])
    df = p.get_universe(spec, AS_OF)
    assert list(df.index) == ["AAA", "BBB", "CCC", "DDD"]  # not pre-applied by the adapter (unadmitted)
    assert pd.isna(df.loc["AAA", F.GICS_SECTOR])  # 'Financial' is not a GICS sector name: NaN, never a pass
    assert df.loc["BBB", F.GICS_SECTOR] == "Health Care" and df.loc["CCC", F.GICS_SECTOR] == "Financials"
    assert df.loc["AAA", F.GICS_INDUSTRY] == "Oil, Gas & Consumable Fuels"  # '&' / 'and' spelling only
    assert pd.isna(df.loc["BBB", F.GICS_INDUSTRY])  # a CapIQ sub-industry label is not a GICS industry
    mask, _ = apply_universe(spec, df)
    assert list(df.index[mask.to_numpy()]) == ["BBB", "DDD"]  # the local engine drops the excluded and the unknown
    q = make(fake, tickers=["AAA", "BBB", "CCC", "DDD"],
             fieldmap={"raw": {"universe": {"gics_sector": {"admitted": True}}}})
    assert list(q.get_universe(spec, AS_OF).index) == ["AAA", "BBB", "DDD"]  # admitted: pre-applied
    assert not any("exclude_sectors was not pre-applied" in w for w in q.warnings)
    fake.point.update({(k, "IQ_INDUSTRY_SECTOR"): "Tech" for k in ("AAA:", "BBB:", "CCC:", "DDD:")})
    r = make(fake, tickers=["AAA", "BBB"])
    r.get_universe(None, AS_OF)
    assert any("raw.universe.gics_sector (IQ_INDUSTRY_SECTOR): no usable value" in w and "value_map" in w
               for w in r.warnings)


def test_vendor_price_feature_uses_the_adjusted_close(fake):
    fake.point[("ACME:", "IQ_CLOSEPRICE_ADJ")] = "42.5"
    df = make(fake).get_vendor_features(["ACME"], AS_OF, ["price"])
    assert df.loc["ACME", "price"] == 42.5
    assert fake.sent("IQ_CLOSEPRICE_ADJ", "ACME:") == [
        {"function": "GDSP", "identifier": "ACME:", "mnemonic": "IQ_CLOSEPRICE_ADJ", "properties": {"asOfDate": AS_OF_STR}}]
    assert fake.sent("IQ_CLOSEPRICE") == []


def test_constituents_error_on_a_later_page_is_reported(fake):
    fake.members["^SPX"] = ["A1:", "A2:", "A3:", "A4:", "A5:"]
    answer = fake._answer  # noqa: SLF001

    def flaky(r: dict[str, Any]) -> dict[str, Any]:
        el = answer(r)
        if r["function"] == "GDSHV" and int(r["properties"]["StartRank"]) > 2:
            el["ErrMsg"], el["Rows"] = "Service temporarily unavailable", []
        return el

    fake._answer = flaky  # noqa: SLF001
    p = make(fake, index="^SPX", fieldmap={"universe": {"constituents": {"page_size": 2}}})
    assert p._members(AS_OF) == ["A1", "A2"]  # noqa: SLF001
    assert any("may be truncated after rank 2" in w and "Service temporarily unavailable" in w for w in p.warnings)
    q = make(fake, index="^SPX", fieldmap={"universe": {"constituents": {"page_size": 2, "end_messages": ["temporarily"]}}})
    q._members(AS_OF)  # noqa: SLF001
    assert not any("truncated" in w for w in q.warnings)
    with pytest.raises(FieldMapError, match="end_messages"):
        make(fake, fieldmap={"universe": {"constituents": {"end_messages": "x"}}})


def test_preflight_failure_note_is_value_free(fake):
    """Regression: a non-numeric preflight value ('61,860.0 (LTM)') was quoted verbatim in a LEG NOT EVALUATED warning."""
    _seed_fundamentals(fake)
    fake.point[(TEST_ID, "IQ_TOTAL_REV")] = "61,860.0 (LTM)"
    fake.point[(TEST_ID, "IQ_REVENUE_EST")] = "(1,234.5)"
    p = make(fake)
    p.get_fundamentals(["ACME"], AS_OF)
    p.get_estimates(["ACME"], AS_OF)
    warn = " | ".join(p.warnings)
    assert f"IQ_TOTAL_REV): preflight on {TEST_ID} failed (non-numeric value (withheld" in warn
    for leaked in ("61,860", "61860", "1,234", "1234"):
        assert leaked not in warn, leaked
    assert any("61,860.0 (LTM)" in d for d in p.details) and any("(1,234.5)" in d for d in p.details)
    fake.point[(TEST_ID, "IQ_TOTAL_REV")] = "CapabilityNeeded"
    q = make(fake)
    q.get_fundamentals(["ACME"], AS_OF)
    assert any("failed (not entitled ('CapabilityNeeded'))" in w for w in q.warnings)


@pytest.mark.parametrize("note, accepted", [
    ("G2: confirmation requested from S&P 2026-09-01, reviewed by Jane Doe", False),
    ("G2: S&P letter (unsigned draft) 2026-09-01, reviewed by Jane Doe", False),
    ("G2: S&P letter KEN-1 2026-09-01, countersignature outstanding, reviewed by Jane Doe", False),
    ("G2: S&P letter KEN-1 2026-09-01 under review, reviewed by Jane Doe", False),
    ("G2: no confirmation yet, 2026-09-01, reviewed by Jane Doe", False),
    ("G2: none", False),
    ("G2: No. 2026-117 letter, 2026-09-01, reviewed by Jane Doe", True),
    ("G2: S&P letter KEN-1 permitting Kensho data to be processed by Anthropic, 2026-09-01, reviewed by Jane Doe", True),
    (G2_NOTE, True),
])
def test_citation_not_yet_vocabulary_anywhere_in_the_clause(note, accepted):
    """Regression: 'requested' / 'draft' were caught only at the start of the clause, while 'No. 2026-117' was
    rejected by the start-anchored 'no'."""
    problem = citation_problem(note, "G2", TODAY)
    assert (problem is None) is accepted, problem


def test_diagnostics_boundary_template_is_accepted_once_filled_in(fake):
    """Regression: diagnostics() suggested a G2 boundary (note='G2: <reference>', no allow_numeric_features=False)
    that _resolve_boundary rejects."""
    from aitrading.data.capiq import G2_BOUNDARY_TEMPLATE

    p = make(fake, kensho=SimpleNamespace())
    text = " | ".join(p.diagnostics())
    assert G2_BOUNDARY_TEMPLATE in text and G2_BOUNDARY_TEMPLATE in BOUNDARY_NOTE
    expr = G2_BOUNDARY_TEMPLATE.removeprefix("boundary=")
    filled = (expr.replace("<confirmation reference>", "S&P letter KEN-1").replace("<YYYY-MM-DD>", "2026-09-15")
              .replace("<lawyer>", "J. Smith"))
    boundary = eval(filled, {"DataBoundary": DataBoundary, "DocumentKind": DocumentKind})  # noqa: S307 - our constant
    assert make(fake, boundary=boundary, kensho=SimpleNamespace()).boundary is boundary
    with pytest.raises(ValueError, match="placeholder"):
        make(fake, boundary=eval(expr, {"DataBoundary": DataBoundary, "DocumentKind": DocumentKind}))  # noqa: S307
