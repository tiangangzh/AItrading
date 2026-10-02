# Vendor reference for adapter implementers: Bloomberg, LSEG, S&P Capital IQ

Companion to [`ARCHITECTURE.md`](ARCHITECTURE.md) (ADR-001). This document is current as of 2026-10-02. Every entry point and field code below went through an adversarial fact-check against vendor-authored code (Bloomberg BQuant notebooks, LSEG-API-Samples, the lseg-data 2.1.1 wheel, kensho-kfinance 8.1.0), against Anthropic and vendor docs, or against community code where nothing better was reachable. Several vendor portals (bloomberg.com, developers.lseg.com, support.marketplace.spglobal.com, docs.kensho.com) were blocked during verification, so facts sourced from them carry at most medium confidence.

## How to read the status column

| Status | Meaning | What the adapter must do |
|---|---|---|
| `confirmed` | Seen in vendor-authored code or docs, in the form shown | May be a compiled constant. Must still pass the admission log (G5) before being used as a **filter predicate**. |
| `corrected` | Verified, but only in the form shown here. The originally reported form was wrong or untested. | Use exactly the form shown. Parts noted as VERIFY are still configurable. |
| `unverifiable` | Could not be confirmed (community-only, inferred, or behind a blocked portal) | **Configurable only** (FieldMap entry, never hard-coded). Preflight it on a known security before every run. A failure suspends the leg and labels it `NOT EVALUATED`. Never use it as a threshold until admitted. |

General rules for all adapters:
- NaN never passes a predicate.
- Record units and sign per field.
- Anchor every date to an absolute `as_of`.
- Assert the returned columns on every response.
- Never let the LLM write vendor query strings.

Representative ScreenSpec used in the examples below. Thresholds are decimals; `as_of` is absolute.

```python
spec = {"as_of": "2026-10-01", "mcap_min": 2e9, "mcap_max": 2e10,
        "ma_fast": 50, "ma_slow": 200, "dd_min": -0.45, "dd_max": -0.20, "rsi_max": 45,
        "vol_mult": 1.5, "fcf_yield_min": 0.05, "rev_growth_ltm_min": 0.10,
        "si_pct_float_min": 0.05, "top_n": 15}
```

---

## 1. Bloomberg

### 1.1 Entry points

| Entry point | Use in this pipeline | Status |
|---|---|---|
| `bql` package inside BQuant (BQNT<GO> Desktop or BQuant Enterprise) | Universe push-down, technicals, fundamentals, ranking | confirmed. I found no evidence that `bql` installs outside BQuant. |
| `blpapi` Desktop API, `localhost:8194`, `//blp/refdata` | BDP/BDH-style enrichment of survivors (short interest, earnings dates) | confirmed. Version 3.26.9.1, Python >= 3.10, installed from Bloomberg's own index (not on pypi.org). |
| `//blp/refdata` `BeqsRequest` | Running a saved EQS screen | corrected: elements are `screenName`, `screenType` (`PRIVATE`/`GLOBAL`), `Group`, and optional `asOfDate`. There is no `languageId`. |
| `//blp/tasvc` `studyRequest` | Per-security technical study (one security per request, so it cannot screen) | corrected (element paths below) |
| `//blp/bqlsvc` `sendQuery` | **Do not use.** Undocumented. Community libraries reach it by impersonating Excel with `appName='EXCEL'`. | prohibited |
| Data License REST `https://api.bloomberg.com/eap` (JWT) | Bulk/enterprise route, priced separately | confirmed [M] |
| Enterprise MCP on DL+ (2026-09-29) | Per-name drill-down after gate G4. Capped per call. | confirmed launch [M]. Tool names, auth and caps are unknown. |
| ASKB (Terminal) | Analyst-side reader; emits BQL | Product confirmed [M]. **No public API found.** |

```python
# Install the Desktop API SDK from Bloomberg's own index
#   python -m pip install --index-url=https://blpapi.bloomberg.com/repository/releases/python/simple/ blpapi

# --- BQuant session (zone T only) ---
import bql
bq = bql.Service()
d, f, u = bq.data, bq.func, bq.univ
res = bq.execute(bql.Request(u.members('INDU Index'),
                             {'Upside': d.best_target_price() / d.px_last() - 1},
                             with_params={'currency': 'USD'}))
df = bql.combined_df(res)          # or res[0].df(); normalise header case (MAXLINE etc.)
```

```python
# --- Desktop API BDP equivalent, failing fast on bad mnemonics (zone T only) ---
import blpapi, pandas as pd

def open_session(host='localhost', port=8194):
    o = blpapi.SessionOptions(); o.setServerHost(host); o.setServerPort(port)
    s = blpapi.Session(o)
    if not (s.start() and s.openService('//blp/refdata')):
        raise RuntimeError('blpapi session/service failed')
    return s

def bdp(s, secs, flds, overrides=None, timeout_ms=30000):
    req = s.getService('//blp/refdata').createRequest('ReferenceDataRequest')
    req.fromPy({'securities': list(secs), 'fields': list(flds),
                'overrides': [{'fieldId': k, 'value': v} for k, v in (overrides or {}).items()]})
    s.sendRequest(req); out, errs = {}, {}
    while True:
        ev = s.nextEvent(timeout_ms)
        if ev.eventType() == blpapi.Event.TIMEOUT:
            raise TimeoutError('BDP timeout')
        for msg in ev:
            for sd in msg.toPy().get('securityData', []):
                out[sd['security']] = sd.get('fieldData', {})
                if sd.get('fieldExceptions'):
                    errs[sd['security']] = sd['fieldExceptions']
        if ev.eventType() == blpapi.Event.RESPONSE:
            return pd.DataFrame.from_dict(out, orient='index'), errs
```

```python
# --- BDH equivalent ---
req = s.getService('//blp/refdata').createRequest('HistoricalDataRequest')
req.fromPy({'securities': ['AAPL US Equity'], 'fields': ['PX_LAST', 'PX_VOLUME'],
            'periodicitySelection': 'DAILY', 'startDate': '20240901', 'endDate': '20261001'})

# --- //blp/tasvc study (one security per request) ---
# priceSource.securityName; priceSource.dataRange.historical.{startDate,endDate,periodicitySelection}
# studyAttributes.rsiStudyAttributes.period=14 | smavgStudyAttributes.period | macdStudyAttributes.{maPeriod1,maPeriod2,sigPeriod}

# --- BEst forward 12M via override (confirmed) ---
# bdp(s, survivors, ['BEST_SALES', 'BEST_EPS'], overrides={'BEST_FPERIOD_OVERRIDE': '1BF'})
```

### 1.2 Push-down grammar

These rules are confirmed unless marked otherwise.
- **Clauses.** A request is `let(...) get(...) for(...) with(...) preferences(...)`. `get` and `for` are mandatory. `let` variables can be used inside `filter()`.
- **Universe.** `equitiesuniv(['ACTIVE','PRIMARY'])` must always be wrapped in `filter()`.
- **Filter order.** Nest `filter()` so that cheap static predicates (country, size) run first and time-series studies run only on what survives.
- **Combining predicates.** In Python, use `f.and_(a, b)`; it takes two arguments, so nest it. In a string, infix `and` and `==` also work.
- **Dates.**
  - `dates=` gives point-in-time values.
  - `calc_interval=range(start, end)` is for return and risk items.
  - Absolute dates work: `range('2018-12-31','2019-12-31')` and `dates='2017-05-05'`.
- **Literals.** Use plain numbers. Do not use `'2B'`.
- **Cached mode.** Pass `mode='cached'` only through `with_params`. The string `with(mode=cached)` form is UNVERIFIED.
- **Units.**
  - `total_return` and the computed drawdown are fractions.
  - `free_cash_flow_yield` / `sales_growth` (percent) are UNVERIFIED.
  - `pct_chg` units are UNVERIFIED.
- **Ranking.** `groupzscore`, `grouprank` and `groupsort` run server-side. **Never filter after `grouprank`.**
- **Silent drops.** A wrong item name returns an error or NaN, and NaN inside `filter()` silently drops every security.
- **Backtests.** Use `members('RAY Index', dates=...)` (subject to index entitlement). `equitiesuniv(dates=)` is UNVERIFIED.

**Full representative screen.** It runs in zone T: push-down, then residual predicates over all survivors, then rank, then top N.

```python
import bql, pandas as pd, numpy as np
bq = bql.Service(); d, f, u = bq.data, bq.func, bq.univ
AS_OF = pd.Timestamp(spec['as_of'])
D = lambda days: (AS_OF - pd.Timedelta(days=days)).strftime('%Y-%m-%d')
A = AS_OF.strftime('%Y-%m-%d')

# --- static pre-filter (inner filter) ---
mcap = d.cur_mkt_cap(currency='USD')
base = u.filter(u.equitiesuniv(['ACTIVE', 'PRIMARY']),
                f.and_(d.cntry_of_risk() == 'US',            # economic exposure; country_iso() semantics UNVERIFIED
                       f.and_(mcap >= spec['mcap_min'], mcap <= spec['mcap_max'])))

# --- technical push-down (outer filter): confirmed primitives only ---
pxs    = d.px_last(dates=f.range(D(730), A), fill='prev', ca_adj='full')    # 2Y: a 200d SMA needs ~290 calendar days
sma50  = d.smavg(pxs, period=spec['ma_fast']).last('1')
sma200 = d.smavg(pxs, period=spec['ma_slow']).last('1')
rsi14  = d.rsi(close=d.px_last(), period=14)          # pass period explicitly; anchored-series PIT form RSI(pxs, period=14) -> VERIFY shape
hi52   = d.maxmin(d.px_high(dates=f.range(D(365), A), fill='prev'),
                  d.px_low(dates=f.range(D(365), A), fill='prev')) \
           .with_additional_parameters(period=260)['maxline'].last('1')
dd     = d.px_last(fill='prev') / hi52 - 1                                    # fraction
v5     = d.px_volume(dates=f.range(D(14), A)).dropna().last(5).avg()
v60    = d.px_volume(dates=f.range(D(95), A)).dropna().last(60).avg()
crit = f.and_(sma50 > sma200, f.and_(dd <= spec['dd_max'], f.and_(dd >= spec['dd_min'],
        f.and_(rsi14 < spec['rsi_max'], v5 > spec['vol_mult'] * v60))))
screen = u.filter(base, crit)

# --- survivor items; residual predicates are fetched, NOT pushed ---
items = {
  'Name': d.name(), 'Sector': d.gics_sector_name(), 'MktCap': mcap, 'DD52w': dd, 'RSI14': rsi14,
  'SMA50': sma50, 'SMA200': sma200, 'Vol5_60': v5 / v60,
  'R6M': d.total_return(calc_interval=f.range(D(182), A)),
  'RevLTM': d.sales_rev_turn(fpt='LTM', dates=A, fill='prev'),
  'RevLTM_1Y': d.sales_rev_turn(fpt='LTM', dates=D(365), fill='prev'),      # trailing growth -> admit in BQLX
  'FCFY': d.free_cash_flow_yield(),                                          # UNVERIFIED name+units -> z-score only until admitted
  'NetUp12w': d.contributor_revisions(d.is_eps(fpt='A', fpo='1'), revision_type='NETUP', revision_window='12w'),
  'RevNTM_3mChg': d.sales_rev_turn(fpt='BT', fpo='1', dates=f.range(D(91), A), fill='prev').pct_chg(),  # units UNVERIFIED
  'EV_EBITDA': d.curr_entp_val() / f.avail(d.ebitda(fpt='LTM'), d.is_oper_inc(fpt='LTM')),
  'ND_EBITDA': d.net_debt() / d.ebitda(fpt='LTM'),
  'IV30': d.implied_volatility(expiry='30d', delta='50'),
  'RV252': d.volatility(calc_interval='252d'),
  'Upside': d.best_target_price() / d.px_last() - 1,
}
df = bql.combined_df(bq.execute(bql.Request(screen, items, with_params={'fill': 'prev', 'mode': 'cached'})))
# Normalise header case before lookups: some BQL headers come back upper-case (e.g. MAXLINE/MINLINE).

# --- short interest for ALL survivors (Desktop API, same workstation; all mnemonics UNVERIFIED -> FLDS first) ---
s = open_session()
si, si_err = bdp(s, list(df.index), ['SHORT_INT', 'SHORT_INT_DT', 'EQY_FLOAT', 'SHORT_INT_RATIO'])
df = df.join(si.apply(pd.to_numeric, errors='coerce'))
df['SI_pct_float'] = df['SHORT_INT'] / df['EQY_FLOAT']                       # fraction

# --- residual evaluation with the vendored canonical evaluator (NaN never passes; attrition per predicate) ---
df['RevG_LTM'] = df['RevLTM'] / df['RevLTM_1Y'] - 1
# evaluator.apply(spec, df, admitted=FIELD_LOG) -> survivors, attrition, suspended_legs
# then rank_v1 (z-scores, versioned weights) -> top spec['top_n']; groupzscore/grouprank as server-side cross-check
```

ASKB "show the BQL" is a source of **candidate** items, for example a short-interest item. An item it suggests enters `items` or `crit` only after a human tests it in BQLX and records it in the admission log.

### 1.3 Bloomberg field map

| Concept | Field code | Status | Notes |
|---|---|---|---|
| Universe | `equitiesuniv(['ACTIVE','PRIMARY'])` in `filter()`; `members(idx, dates=)` | confirmed | `equitiesuniv(dates=)` is UNVERIFIED |
| Market cap | BQL `cur_mkt_cap(currency='USD')`; BDP `CUR_MKT_CAP` | confirmed | Raw currency units. Use plain-number thresholds. |
| Last price | BQL `px_last(fill='prev')`; series `px_last(dates=range(..), fill='prev', ca_adj='full')`; BDP `PX_LAST` | confirmed | |
| Daily volume | BQL `px_volume(dates=range(..)).dropna()`; BDP `PX_VOLUME` | confirmed | |
| Avg volume N-day | BQL `px_volume(dates=range(..)).dropna().last(N).avg()` | corrected | Built from confirmed primitives |
| Avg volume 20d (BDP) | `VOLUME_AVG_20D` | unverifiable | Community only |
| SMA 50/200 | `smavg(<px series, 2Y>, period=N).last('1')` | corrected | `.last()` in string form is UNVERIFIED |
| EMA (scalar) | `emavg(period=N)` | confirmed | This is an EMA, not an SMA |
| SMA bare form | `smavg(period=N)` | unverifiable | Do not use |
| SMA (BDP) | `MOV_AVG_50D`, `MOV_AVG_200D` | unverifiable | |
| RSI 14 | `rsi(close=px_last(), period=14)`; `RSI(<series>, period=14)` | confirmed | Default period unconfirmed, so always pass it |
| RSI (BDP) | `RSI_14D` | unverifiable | |
| MACD | `macd(<series>)['MACD2']` | confirmed | Only the MACD2 column is confirmed. Fallback: `emavg(12)-emavg(26)`. |
| MACD (tasvc) | `macdStudyAttributes.maPeriod1/maPeriod2/sigPeriod` | corrected | One security per request |
| 52w high | `maxmin(px_high, px_low).with_additional_parameters(period=260)['maxline']`; `max(px_high(dates=range(-52W,0D), fill=prev))` | confirmed | Header may come back as `MAXLINE` |
| 52w high (BDP) | `HIGH_52WEEK` | unverifiable | |
| Drawdown from high | `px_last()/hi52 - 1` | confirmed | Fraction |
| Total return 3/6/12m | `total_return(calc_interval=range(start,end))` | confirmed | Fraction |
| Price change (BDP) | `CHG_PCT_3M/6M/1YR` | unverifiable | Price only |
| Relative strength | `total_return(R) - value(total_return(R), list(['SPX Index']))` | confirmed | Broadcasting `value()` inside `filter()` is untested: keep it as a column |
| Alpha | `alpha(calc_interval=range('-252d','0d'), benchmark_ticker='SPX Index')` | confirmed | Regression alpha, not raw relative strength |
| FCF TTM | BQL `cf_free_cash_flow(fpt=LTM)`; BDP `CF_FREE_CASH_FLOW` | unverifiable | |
| FCF yield | BQL `free_cash_flow_yield()`; BDP `FREE_CASH_FLOW_YIELD` | unverifiable | Not in the cited notebooks. Units unknown. |
| FCF growth | `free_cash_flow_1_year_growth()` | unverifiable | |
| Enterprise value | `curr_entp_val()`; custom `cur_mkt_cap()+net_debt().znav()+minority_noncontrolling_interest().znav()+bs_pfd_eqy().znav()`; BDP `CURR_ENTP_VAL` | confirmed | |
| EV/EBITDA | `curr_entp_val()/avail(ebitda(fpt='LTM'), is_oper_inc(fpt='LTM'))` | corrected | BDP `EV_TO_T12M_EBITDA` and `BEST_CUR_EV_TO_EBITDA` are unverifiable |
| EV/Sales | `curr_entp_val()/sales_rev_turn(fpt='LTM')` | confirmed | BDP `EV_TO_T12M_SALES` unverifiable |
| Revenue growth, trailing | `sales_rev_turn(fpt='LTM', dates=as_of)/sales_rev_turn(fpt='LTM', dates=as_of-1Y, fill='prev') - 1` | corrected | Components confirmed. Admit the combination in BQLX. |
| Revenue growth, FY1 forward | `sales_growth(fpt='A', fpo='1')` | corrected | Forward and estimate-driven. **Not trailing.** |
| Revenue growth (no-arg / BDP) | `sales_growth()`; `SALES_GROWTH` | unverifiable | |
| Revenue NTM | `sales_rev_turn(fpt='BT', fpo='1')`; BDP `BEST_SALES` + `BEST_FPERIOD_OVERRIDE='1BF'` | confirmed | `est_source='BST'` is the default. `fpts` parameter also exists. |
| Estimate revisions | `contributor_revisions(is_eps(fpt='A',fpo='1'), revision_type='NETUP'\|'numup'\|'numdn', revision_window='12w')`; `<dated estimate>.pct_chg()` | confirmed | `pct_chg` units UNVERIFIED. BDP `BEST_EPS_4WK_CHG` unverifiable. |
| Gross margin | `gross_margin(fpt=..., fpo=...)`; BDP `GROSS_MARGIN` | confirmed | |
| Operating margin | `oper_margin()`; BDP `OPER_MARGIN` | unverifiable | Alternative from confirmed items: `is_oper_inc(fpt='LTM')/sales_rev_turn(fpt='LTM')` |
| Net debt | `net_debt()`; BDP `NET_DEBT` | confirmed | |
| Net debt / EBITDA | `net_debt()/ebitda(fpt='LTM')` | corrected | `net_debt_to_ebitda()` and BDP `NET_DEBT_TO_EBITDA` are unverifiable |
| Short interest % float | compute `SHORT_INT/EQY_FLOAT`; BDP `SI_PERCENT_EQUITY_FLOAT` | unverifiable | No BQL item known. May need a securities-finance entitlement. |
| Short interest shares / DTC / date | BDP `SHORT_INT`, `SHORT_INT_RATIO`, `SHORT_INT_DT`, `EQY_FLOAT` | unverifiable | Community code only. FLDS check before use. US data is bi-monthly. |
| Implied vol 30d ATM | `implied_volatility(expiry='30d', delta='50')` | corrected | The `'<N>d'` format is confirmed. `pct_moneyness` unverifiable. |
| Implied vol (BDP) | `30DAY_IMPVOL_100.0%MNY_DF` | unverifiable | |
| Realized vol | `volatility(calc_interval='252d')` | confirmed | BDP `VOLATILITY_30D` unverifiable |
| Put/call | BDP `PUT_CALL_VOLUME_RATIO_CUR_DAY`, `PUT_CALL_OPEN_INTEREST_RATIO`; BQL `options()` + `put_call()`, `open_int()` | unverifiable | |
| GICS / BICS | `gics_sector_name()`; `classification_name('BICS', level)` | confirmed | `classification_name('GICS',n)` and `gics_industry_name()` unverifiable |
| Country | `cntry_of_risk()`, `country_iso()` | corrected | Both items exist. `country_iso()` meaning for equities is UNVERIFIED. |
| Country/exchange (other) | BQL `cntry_of_domicile()`, `exch_code()`; BDP `SECURITY_TYP` | unverifiable | ADR/REIT exclusion not yet possible in BQL |
| Earnings dates | BDP `EXPECTED_REPORT_DT`, `LATEST_ANNOUNCEMENT_DT` | unverifiable | |
| Analyst target | `best_target_price()`, `target_price()` | confirmed | BDP `BEST_TARGET_PRICE`/`EQY_REC_CONS` are community-sourced |
| Transcripts | No BQL/BDP item. Terminal DS/CF/EVTS, ASKB, Document Insights. BQuant Enterprise Textual Analytics (back to 2007). | confirmed (products) | API undocumented. LLM export rights unconfirmed (licence class L4/L3). |
| News | EDF Textual News | confirmed | **Black-box use only** (L5) |
| Broker research | Terminal BRC/DS/ASKB only | confirmed | L6 |
| Filings | SEC EDGAR (public) | confirmed | L0 |

---

## 2. LSEG (Data Library for Python, `lseg-data` 2.1.1)

### 2.1 Entry points

| Entry point | Use | Status |
|---|---|---|
| `pip install lseg-data==2.1.1`; `import lseg.data as ld` (Python >= 3.9) | All pulls | confirmed |
| `ld.open_session('platform.ldp')` (unattended; v1 username/password or v2 `client_id`/`client_secret` service account) | Server runs | corrected. `desktop.workspace` is the library default but is licensed for individual use and **must not run on a server**. |
| Config lookup order: `$LD_LIB_CONFIG_PATH/lseg-data.config.json`, then cwd, then `~` | Config | corrected. The library does not look next to the script. |
| `ld.get_data(universe, fields, parameters=None, header_type=ld.HeaderType.NAME)` | Snapshots | confirmed. **Silently drops** bad or unentitled fields. |
| `ld.get_history(universe, fields, interval='daily', start, end, adjustments, count)` | Price history | corrected. Default columns depend on the venue, so request fields explicitly. A single RIC returns flat columns. |
| `lseg.data.discovery.Screener(expr)` / `'SCREEN(...)'` string | Push-down | confirmed (wraps as `screen(expr)` and resolves `TR.RIC`) |
| `ld.news.get_headlines` / `get_story` | Headlines | confirmed. Content is licence class L3. |
| `lseg.data.content.filings.search/retrieval.Definition` | Filings | confirmed. Needs a Filings entitlement. |
| StreetEvents XML over SFTP `ftp-setranscripts.lseg.com:/TRANSCRIPT/XML_Add_IDs/Current` | Transcripts | confirmed [M]. Separate licence, L3. **Pin the host key.** |
| LSEG MCP `https://api.analytics.lseg.com/lfa/mcp` (OAuth2 authorization code + PKCE, `login.ciam.refinitiv.com`) | Analyst drill-down only | confirmed. No screening tool. Not ZDR-eligible via the Anthropic connector. |

```python
import os, time, pandas as pd, lseg.data as ld
AS_OF = pd.Timestamp(spec['as_of'])
D = lambda days: (AS_OF - pd.Timedelta(days=days)).strftime('%Y-%m-%d')
os.environ.setdefault('LD_LIB_CONFIG_PATH', './config')
ld.get_config().set_param('http.request-timeout', 300)   # default 20 s; v1.1+ ~300 s server timeout, no datapoint cap
ld.open_session('platform.ldp')

def preflight(codes, ric='IBM.N'):
    """A field passes only if its column comes back non-empty on a known RIC."""
    ok = {}
    for c in codes:
        try:
            df = ld.get_data(ric, [c], header_type=ld.HeaderType.NAME)
            ok[c] = df.shape[1] == 2 and df.iloc[:, 1].notna().any()
        except Exception:
            ok[c] = False
        time.sleep(0.25)
    return ok

def get_tr(universe, fields: dict, params=None, max_points=8000):
    keys, codes = list(fields), list(fields.values())
    step, parts = max(1, max_points // len(codes)), []
    for i in range(0, len(universe), step):
        df = ld.get_data(universe[i:i + step], codes, params, header_type=ld.HeaderType.NAME)
        if df.shape[1] != len(codes) + 1:
            raise RuntimeError(f'fields dropped: {list(df.columns)}')     # LSEG drops fields silently
        df.columns = ['RIC'] + keys; parts.append(df); time.sleep(0.3)
    return pd.concat(parts).drop_duplicates('RIC').set_index('RIC')
```

### 2.2 Push-down grammar

Grammar:
- **Form.** `SCREEN(U(IN(<universe>)), cond, cond, ..., CURN=USD)`. Commas between conditions mean AND. Parenthesised OR groups are allowed.
- **Operators.** `> >= < <= ==`, `IN`, `NOT_IN`, `BETWEEN`, `CONTAINS`, `TOP/BOTTOM(f, N, nnumber|centile)`, `RELATIVEDATE`.
- **Analytics.** `AVG`, `MAX`, `PERCENT_CHG`, `VALUE`, `REL`, `IF`, `AVAIL`, `CGR`, `STD`, `MEDIAN`, plus `MAVG(series, period)` and `RSI(series)` from LSEG's grammar helper. Whether `MAVG`/`RSI` are accepted inside SCREEN across the universe is UNVERIFIED.
- **Relative dates.** `0D-49D` counts trading days and `364C` counts calendar days. Excel exports double the quotes; convert them back.

Push only predicates confirmed inside official SCREENs: universe/listing, size, `TR.TotalReturn3Mo` (percent), `TR.Volatility10D`, and StarMine ranks. Any other predicate is pushed only if it passes `preflight` on that day. Everything else is evaluated locally.

```python
from lseg.data.discovery import Screener
CAND = {'TR.PricePctChg52WkHigh': f"TR.PricePctChg52WkHigh<={spec['dd_max']*100:g}"}   # sign/scale UNVERIFIED
pf = preflight(list(CAND))
parts = ['U(IN(Equity(active,public,primary))/*UNV:Public*/)',
         'IN(TR.ExchangeCountryCode,"US")', 'IN(TR.InstrumentTypeCode,"ORD")',
         'NOT_IN(TR.ExchangeMarketIdCode,"OTCM")',                                  # official VolatilityScreening sample
         f"TR.CompanyMarketCap(Scale=6)>={spec['mcap_min']/1e6:g}",
         f"TR.CompanyMarketCap(Scale=6)<={spec['mcap_max']/1e6:g}"]                 # >= form is official
parts += [e for k, e in CAND.items() if pf[k]]
rics = list(Screener(', '.join(parts + ['CURN=USD'])))

CORE = {'name': 'TR.CommonName', 'sector': 'TR.GICSSector', 'mcap_musd': 'TR.CompanyMarketCap(Scale=6)',
        'px': 'TR.PriceClose', 'hi52': 'TR.Price52WeekHigh', 'tr_3m_pct': 'TR.TotalReturn3Mo',
        'ev_musd': 'TR.EV(Scale=6)', 'ev_ebitda': 'TR.EVToEBITDA', 'nd_ebitda': 'TR.NetDebtToEBITDA',
        'rev_fq0': 'TR.RevenueActValue(Period=FQ0)', 'rev_fq4': 'TR.RevenueActValue(Period=FQ-4)',   # FQ-4 combo UNVERIFIED
        'n_up': 'TR.NumEstRevisingUp(WP=30d)', 'n_dn': 'TR.NumEstRevisingDown(WP=30d)'}
OPTIONAL = {'fcf_ltm_musd': 'TR.FreeCashFlow(Period=LTM,Scale=6)',
            'si_shares': 'TR.ShortInterest(SDate=0D)', 'float_shares': 'TR.SharesFreeFloat(SDate=0D)'}
snap = get_tr(rics, CORE, {'Curn': 'USD'})
ok = {k: v for k, v in OPTIONAL.items() if preflight([v])[v]}         # failures -> leg NOT EVALUATED
if ok: snap = snap.join(get_tr(rics, ok, {'Curn': 'USD'}))
snap['dd'] = snap.px / snap.hi52 - 1                                   # local, unit-safe
snap['rev_g'] = snap.rev_fq0 / snap.rev_fq4 - 1                        # single-quarter YoY, not LTM: label it
snap['si_pct_float'] = snap.get('si_shares') / snap.get('float_shares') # fraction

h = ld.get_history(list(snap.index)[:10], ['TRDPRC_1', 'ACVOL_UNS'], interval='daily', start=D(420))
# normalise single-RIC flat columns to MultiIndex; SMA/Wilder RSI/MACD/vol ratio computed locally.
# Corporate-action adjustment of TRDPRC_1 is UNVERIFIED: cross-check drawdown vs TR.Price52WeekHigh around splits.
```

### 2.3 LSEG field map

| Concept | Field code | Status | Notes |
|---|---|---|---|
| Market cap | `TR.CompanyMarketCap` / `TR.CompanyMarketCap(Scale=6)` + `CURN=USD` | confirmed | `>=` form used in official SCREENs |
| Last price | `TR.PriceClose` (EOD); `TRDPRC_1` (history); `CF_LAST` (real-time) | confirmed | `CF_LAST` needs a real-time entitlement |
| Daily volume | `TR.Volume`; `ACVOL_UNS` (history) | confirmed | |
| Avg volume 20d | compute `ACVOL_UNS.tail(20).mean()` | corrected | `TR.AvgDailyVolume6M` is third-party and 6-month only |
| SMA 50/200 | compute from `TRDPRC_1`; `MAVG(...)` analytic | corrected | `MAVG` inside SCREEN UNVERIFIED |
| RSI 14 | compute Wilder locally; `TR.RSISimple14D`; `RSI()` analytic | corrected | `TR.RSISimple14D` appears only in a sample comment. `RSI()` smoothing undocumented. |
| MACD | compute locally (EMA12-EMA26, signal EMA9) | confirmed | No field exists |
| 52w high / % from high | `TR.Price52WeekHigh`, `TR.PricePctChg52WkHigh`, `TR.Price52WeekLow`, `TR.PricePctChg52WkLow` | confirmed | Codes confirmed. Sign/scale and use in SCREEN UNVERIFIED. Compute `px/hi52-1` locally. |
| Total return | `TR.TotalReturn3Mo` (in SCREEN, **percent**), `TR.TotalReturn52Wk`, `TR.TotalReturnYTD`, `TR.TotalReturn1Mo` | corrected | `TR.TotalReturn6Mo` unverifiable |
| Price change | `TR.PricePctChgYTD` | confirmed | `TR.PricePctChg1M/3M/6M/1Y` unverifiable |
| Relative strength | `TR.RelPricePctChgYTD`; `REL()`/`VALUE()` analytics; local ratio vs `.SPX` | corrected | Benchmark undocumented. `.SPX` history may be unentitled (fall back to SPY). |
| StarMine momentum | `TR.PriceMoMidTermComponent` | confirmed | In SCREEN. `TR.PriceMoRegionRank`/`CountryRank` unverifiable. |
| FCF TTM | `TR.FreeCashFlow(Period=LTM)`; `TR.F.LeveredFOCF` | unverifiable | Third-party live-validated only |
| FCF yield | compute `TR.FreeCashFlow(LTM,Scale=6)/TR.CompanyMarketCap(Scale=6)` | unverifiable | Depends on FCF |
| Enterprise value | `TR.EV` | confirmed | |
| EV/EBITDA | `TR.EVToEBITDA` | corrected | `TR.FwdEVToEBITDA` unverifiable |
| EV/Sales | `TR.EVToSales` | confirmed | |
| Revenue growth YoY | `TR.RevenueActValue(Period=FQ0)` vs `(Period=FQ-4)`, computed locally | corrected | FQ-4 combination UNVERIFIED. Single-quarter, not LTM. |
| Revenue estimate FY1 | `TR.RevenueMeanEstimate(Period=FY1,Scale=6,Curn=USD)`, `TR.RevenueSmartEst(Period=FY1)` | corrected | `TR.RevenueMean` and `Period=NTM` are third-party |
| Estimate revisions | `TR.NumEstRevisingUp(WP=30d)`, `TR.NumEstRevisingDown(WP=30d)`, `TR.EpsPreSurprisePct` | corrected | `EstimateMeasure=`/`Period=FY1` unverifiable |
| StarMine models | `TR.ValMoRegionRank`, `TR.EQCountryListRank(Period=FY0)`, `TR.IVPriceToIntrinsicValue`, `TR.PriceMoMidTermComponent` | corrected | `TR.CombinedAlphaCountryRank` unverifiable. Needs StarMine entitlement. |
| Gross margin | `TR.GrossMargin(Period=LTM)` | unverifiable | Preflight |
| Operating margin | `TR.OperProfitMarginPct` / `TR.OperatingMargin` | unverifiable | Official samples show only `TR.OperatingProfitMarginPct5YrAvg` and `TR.PretaxMarginPercent` |
| Net debt | `TR.NetDebt`, `TR.F.NetDebt` | unverifiable | Definitions differ |
| Net debt/EBITDA | `TR.NetDebtToEBITDA` | confirmed | |
| Short interest % float | `100*TR.ShortInterest(SDate=0D)/TR.SharesFreeFloat(SDate=0D)` computed locally | corrected | Third-party evidence only. The exported expression returns a fraction. `TR.SIShortInterest` unverifiable. |
| Days to cover | compute `TR.ShortInterest / avg20(ACVOL_UNS)` | corrected | `TR.ShortInterestDTC` was found nowhere |
| Implied vol 30d ATM | RIC `<ticker>ATMIV.U` + `TR.30DAYATTHEMONEYIMPLIEDVOLATILITYINDEXFORCALLOPTIONS` / `...FORPUTOPTIONS`; `IMP_VOLT`; realized `TR.Volatility10D/20D/30D/60D/90D` | confirmed | Pattern verified for WMT only, so check each RIC resolves. Fetch for survivors only. |
| Put/call | compute from chain (`PUTCALLIND`, `ACVOL_1`, `OPINT_1`) | confirmed (no field) | US chain RIC `0#<root>*.U` and `ACVOL_1`/`OPINT_1` unverified |
| GICS / TRBC | `TR.GICSSector`, `TR.GICSIndustryCode`; `TR.TRBCEconSectorCode`, `TR.TRBCBusinessSectorCode`, `TR.TRBCBusinessSector`, `TR.TRBCActivity` | corrected | `TR.GICSIndustry`/`SubIndustry`/`SectorCode` unverifiable |
| US listing / share type | `IN(TR.ExchangeCountryCode,"US")`, `IN(TR.InstrumentTypeCode,"ORD")`, `NOT_IN(TR.ExchangeMarketIdCode,"OTCM")`, `TR.HQCountryCode`, `TR.CoRTradingCountryCode` | confirmed | |
| Free float / shares | `TR.FreeFloatPct`, `TR.FreeFloat`, `TR.SharesOutstanding` | corrected | `TR.SharesFreeFloat`/`TR.CompanySharesOutstanding` are third-party |
| Earnings events | `TR.EventType`, `TR.EventTitle`, `TR.EventStartDate`, `TR.EventLastUpdate` with `{'EventType':'ALL'}`; `TR.EPSActSurprise`, `TR.RevenueActSurprise`, `TR.RevenueActReportDate` | corrected | `TR.ExpectedReportDate` unverifiable |
| Transcripts | StreetEvents XML (`EventStory_Body`, `companyTicker`, `eventTypeName`, `lastUpdate`); MarketPsych MTA REST | confirmed | MCP transcripts tool unverifiable. L3. |
| News | `ld.news.get_headlines`/`get_story` | confirmed | L3: Reuters copyright |
| Filings | `filings.search.Definition(feed=Feed.EDGAR, ...)`, `filings.retrieval.Definition(...)` | confirmed | Entitlement needed |
| Broker research | none | confirmed | |
| Ownership | `TR.FundPortfolioName`, `TR.FdAdjPctOfShrsOutHeld`, `TR.FundAdjShrsHeld`, `TR.FundHoldingsDate` | corrected | `TR.SIInstitutionalOwn` unverifiable |

Limits: Workspace allows 5 req/s, 10k requests/day, 50 MB/min and 5 GB/day [M]. Platform limits are UNVERIFIED, so measure them before production. LSEG keeps short interest only for listed names, which biases any backtest through survivorship.

---

## 3. S&P Global Market Intelligence

### 3.1 Entry points

| Entry point | Use | Status |
|---|---|---|
| Capital IQ GDS API `POST https://api-ciq.marketintelligence.spglobal.com/gdsapi/rest/v3/clientservice.json`, body `{"inputRequests":[{function, identifier, mnemonic, properties}]}`, response `GDSSDKResponse` | Survivor cross-check values | confirmed [M] |
| Auth `POST .../gdsapi/rest/authenticate/api/v1/token` (form-urlencoded username/password, gives `access_token`) | | corrected: `/tokenRefresh` and the 60-min lifetime are UNVERIFIED. Re-authenticate on 401. |
| Functions `GDSP`, `GDSPV`, `GDSHE`, `GDSHV`, `GDST`, `GDSG` | | confirmed. **No screen function.** |
| Usage `POST .../gdsapi/rest/v3/usageservice.json` `{"inputRequests":[{"mnemonic":"USAGE_METRICS"}]}` | Read real limits | [M] community, citing the May 2026 guide |
| Identifiers `TICKER:EXCH`, `TICKER:`, `IQ<id>`, `I_<ISIN>`, `CSP_<CUSIP>`, `GV<gvkey>`, `IQT<tradingItemId>`, `^<index>` | | confirmed (`^` seen only as `^ftse`) |
| Document search `v1/documents/search` | | unverifiable: do not hard-code |
| Kensho LLM-ready API `pip install kensho-kfinance` 8.1.0 (Python >= 3.10), REST `https://kfinance.kensho.com/api/v1/` | Transcripts, line items, capitalisation (L1 pending G2) | confirmed |
| Kensho MCP `https://kfinance.kensho.com/integrations/mcp`; `python -m kfinance.mcp`; `python -m kfinance.proxy_mcp` | Analyst-interactive only | confirmed. OAuth DCR/PKCE details unverifiable. |
| Xpressfeed / Snowflake CIQ tables | The only S&P push-down | [M] (table and constant names come from community SQL) |

```python
import requests
U, P = os.environ['CIQ_USER'], os.environ['CIQ_PASSWORD']   # from the S&P API welcome letter
BASE = 'https://api-ciq.marketintelligence.spglobal.com/gdsapi/rest'
tok = requests.post(f'{BASE}/authenticate/api/v1/token', data={'username': U, 'password': P},
                    headers={'Content-Type': 'application/x-www-form-urlencoded'}).json()['access_token']
H = {'Authorization': f'Bearer {tok}'}
limits = requests.post(f'{BASE}/v3/usageservice.json', json={'inputRequests': [{'mnemonic': 'USAGE_METRICS'}]}, headers=H).json()
req = {'inputRequests': [
    {'function': 'GDSP', 'identifier': 'IBM:NYSE', 'mnemonic': 'IQ_MARKETCAP', 'properties': {'currencyId': 'USD'}},
    {'function': 'GDSP', 'identifier': 'IBM:NYSE', 'mnemonic': 'IQ_TOTAL_REV_1YR_ANN_GROWTH', 'properties': {'periodType': 'IQ_LTM'}},
    {'function': 'GDSHE', 'identifier': 'IBM:NYSE', 'mnemonic': 'IQ_CLOSEPRICE_ADJ',
     'properties': {'startDate': '09/01/2025', 'endDate': '10/01/2026'}},
    {'function': 'GDSHV', 'identifier': '^SPX', 'mnemonic': 'IQ_CONSTITUENTS',            # GDSHV (not GDSHE); ^SPX untested
     'properties': {'StartRank': 1, 'EndRank': 600}}]}
r = requests.post(f'{BASE}/v3/clientservice.json', json=req, headers=H).json()['GDSSDKResponse']
if len(r) == 1 and set(r[0]) == {'ErrMsg'}:
    raise RuntimeError(r[0]['ErrMsg'])                 # e.g. 'Daily Request Limit of 10000 Exceeded'
```

```python
# Kensho (zone E; licence class L1 pending gate G2)
import os
from kfinance.client.kfinance import Client
# PeriodType enum (annual | quarterly | ltm | ytd) ships with kfinance; import it from the installed package.
kf = Client(client_id=os.environ['KENSHO_CLIENT_ID'], private_key=os.environ['KENSHO_PRIVATE_KEY'])  # key pair in prod
t = kf.ticker('SPGI')
mcap, tev = t.market_cap(), t.tev()
cfo = t.line_item(line_item='cash_from_operations', period_type=PeriodType.ltm)       # dataItemId 2006
capex = t.line_item(line_item='capital_expenditure', period_type=PeriodType.ltm)      # 2021; sign UNVERIFIED
e = t.company.latest_earnings          # .name gives the fiscal quarter; .key_dev_id
text = e.transcript.raw                # 'Speaker: text'; requires TranscriptsPermission
universe = kf.tickers(country_iso_code='US', gics='451030') & kf.tickers(exchange_code='NYSE')  # country = HQ country
```

### 3.2 Push-down grammar

No S&P REST API screens the universe:
- GDS is per identifier x mnemonic. About 4,000 names x 20 fields is roughly 80,000 requests, against an observed limit of 10,000 a day.
- Kensho offers only categorical ticker groups.
- Capital IQ Pro `=SPGScreen` exists only in Excel.

Push-down is possible only as SQL over Xpressfeed or Snowflake. **The `ID` constants are community-sourced**, so validate them against `ciqCountryGeo`, `ciqCurrency` and `ciqSecuritySubType` when the data loads.

```sql
-- Snowflake dialect. Prices MUST be dividend-adjusted; fundamentals join on financialPeriodId.
WITH us_primary AS (
  SELECT c.companyId, c.companyName, ti.tradingItemId, ti.tickerSymbol
  FROM ciqCompany c
  JOIN ciqSecurity s     ON s.companyId = c.companyId AND s.primaryFlag = 1 AND s.securitySubTypeId = 1
  JOIN ciqTradingItem ti ON ti.securityId = s.securityId AND ti.primaryFlag = 1 AND ti.currencyId = 160
  JOIN ciqExchange e     ON e.exchangeId = ti.exchangeId AND e.countryId = 213
  WHERE c.countryId = 213),
adj AS (
  SELECT pe.tradingItemId, pe.pricingDate,
         pe.priceClose * COALESCE(daf.divAdjFactor, 1) AS adjClose,
         pe.priceHigh  * COALESCE(daf.divAdjFactor, 1) AS adjHigh, pe.volume
  FROM ciqPriceEquity pe JOIN us_primary u ON u.tradingItemId = pe.tradingItemId
  LEFT JOIN ciqPriceEquityDivAdjFactor daf ON daf.tradingItemId = pe.tradingItemId
        AND daf.fromDate <= pe.pricingDate AND (daf.toDate IS NULL OR daf.toDate >= pe.pricingDate)
  WHERE pe.pricingDate BETWEEN DATEADD(day, -420, :as_of) AND :as_of),
px AS (
  SELECT tradingItemId, adjClose,
    AVG(adjClose) OVER (PARTITION BY tradingItemId ORDER BY pricingDate ROWS BETWEEN 49 PRECEDING AND CURRENT ROW)  AS sma50,
    AVG(adjClose) OVER (PARTITION BY tradingItemId ORDER BY pricingDate ROWS BETWEEN 199 PRECEDING AND CURRENT ROW) AS sma200,
    MAX(adjHigh)  OVER (PARTITION BY tradingItemId ORDER BY pricingDate ROWS BETWEEN 251 PRECEDING AND CURRENT ROW) AS high52w,
    AVG(volume)   OVER (PARTITION BY tradingItemId ORDER BY pricingDate ROWS BETWEEN 4 PRECEDING AND CURRENT ROW)   AS vol5,
    AVG(volume)   OVER (PARTITION BY tradingItemId ORDER BY pricingDate ROWS BETWEEN 59 PRECEDING AND CURRENT ROW)  AS vol60,
    COUNT(*)      OVER (PARTITION BY tradingItemId) AS n_obs
  FROM adj QUALIFY ROW_NUMBER() OVER (PARTITION BY tradingItemId ORDER BY pricingDate DESC) = 1),
mc AS (SELECT companyId, marketCap, TEV FROM ciqMarketCap WHERE pricingDate <= :as_of
       QUALIFY ROW_NUMBER() OVER (PARTITION BY companyId ORDER BY pricingDate DESC) = 1),
fin AS (
  SELECT fp.companyId,
    MAX(CASE WHEN fd.dataItemId = 2006 THEN fd.dataItemValue END) AS cfo_ltm,
    MAX(CASE WHEN fd.dataItemId = 2021 THEN fd.dataItemValue END) AS capex_ltm,   -- sign UNVERIFIED
    MAX(CASE WHEN fd.dataItemId = 4422 THEN fd.dataItemValue END) AS lfcf_ltm      -- community
  FROM ciqLatestInstanceFinPeriod fp JOIN ciqFinancialData fd ON fd.financialPeriodId = fp.financialPeriodId
  WHERE fp.periodTypeId = 4 AND fp.latestPeriodFlag = 1 AND fd.dataItemId IN (2006, 2021, 4422)
  GROUP BY fp.companyId)
SELECT u.tickerSymbol, u.companyId, mc.marketCap, px.*, px.adjClose/px.high52w - 1 AS dd,
       COALESCE(f.lfcf_ltm, f.cfo_ltm + f.capex_ltm) / NULLIF(mc.marketCap, 0) AS fcf_yield
FROM px JOIN us_primary u ON u.tradingItemId = px.tradingItemId
JOIN mc ON mc.companyId = u.companyId LEFT JOIN fin f ON f.companyId = u.companyId
WHERE px.n_obs >= 252
  AND mc.marketCap BETWEEN 2000 AND 20000          -- assumes millions: VERIFY units
  AND px.sma50 > px.sma200
  AND px.adjClose/px.high52w - 1 BETWEEN -0.45 AND -0.20
  AND px.vol5 > 1.5 * px.vol60;
-- Residual (local): RSI/MACD on the adjusted series, FCF-yield and growth thresholds (after admission),
-- trailing revenue growth (two periods or IQ_TOTAL_REV_1YR_ANN_GROWTH via GDS), short interest (no S&P field).
-- On a SQL Server Xpressfeed loader, replace QUALIFY with ROW_NUMBER subqueries.
```

```sql
-- Transcript for one company: earnings calls are keyDevEventTypeId 48; keep one version per keyDevId
SELECT tc.componentOrder, tc.transcriptComponentTypeId, tc.transcriptPersonId, tc.componentText
FROM ciqTranscript t
JOIN ciqEventToObjectToEventType ete ON ete.keyDevId = t.keyDevId AND ete.keyDevEventTypeId = 48
JOIN ciqTranscriptComponent tc ON tc.transcriptId = t.transcriptId
WHERE ete.objectId = :companyId
QUALIFY DENSE_RANK() OVER (PARTITION BY t.keyDevId ORDER BY t.transcriptCreationDateUTC DESC) = 1
ORDER BY t.keyDevId DESC, tc.componentOrder;
```

### 3.3 S&P field map

| Concept | Field code | Status | Notes |
|---|---|---|---|
| Market cap | `IQ_MARKETCAP`; Kensho `capitalization='market_cap'`; `ciqMarketCap.marketCap` | confirmed | Units (millions?) UNVERIFIED. Pass `currencyId=USD`. |
| Last price | `IQ_CLOSEPRICE`, `IQ_CLOSEPRICE_ADJ`; `IQ_LASTSALEPRICE` | confirmed | `IQ_LASTSALEPRICE` from the Excel glossary only |
| Daily volume | `IQ_VOLUME`; `ciqPriceEquity.volume`; Kensho price bars | confirmed | |
| Avg volume 20d | compute from `IQ_VOLUME` | confirmed | No mnemonic |
| SMA 50/200 | compute from `IQ_CLOSEPRICE_ADJ` or `priceClose*divAdjFactor` | corrected | Never compute from raw `ciqPriceEquity` |
| RSI 14 | compute (Wilder) | confirmed | No field |
| MACD | compute | confirmed | No field |
| 52w high | `IQ_YEARHIGH` (`IQ_YEARHIGH_DATE`) or compute MAX(adjusted high) | unverifiable | GDS parity untested |
| Total return | compute from `IQ_CLOSEPRICE_ADJ` | unverifiable | Return semantics of `PERIODTYPE '-3M'` are undocumented |
| Relative strength | compute vs `^SPX` + `IQ_CLOSEPRICE` | unverifiable | `^SPX` untested |
| FCF TTM | `IQ_LEVERED_FCF` (IQ_LTM); `IQ_CASH_OPER + IQ_CAPEX`; Xpressfeed 4422 or 2006 + 2021; Kensho `cash_from_operations` − `capital_expenditure` | corrected | Capex sign UNVERIFIED. kFinance has no FCF line item. |
| FCF yield | compute; `1/IQ_MARKET_CAP_LFCF` | unverifiable | |
| Enterprise value | `IQ_TEV`; Kensho `tev`; `ciqMarketCap.TEV` | confirmed | |
| EV/EBITDA | `IQ_TEV_EBITDA`; `IQ_TEV_EBITDA_FWD` | confirmed | GDS parity [M] |
| EV/Sales | `IQ_TEV_TOTAL_REV`; `IQ_TEV_TOTAL_REV_FWD` | confirmed | |
| Revenue growth YoY | `IQ_TOTAL_REV_1YR_ANN_GROWTH` | confirmed | Fallback: Kensho `total_revenue` (28) |
| Revenue estimate NTM | `IQ_REVENUE_EST` + `periodType=IQ_NTM`; Kensho `consensus_estimates(quarterly, 4)` | unverifiable | `_CIQ` suffix variants unverified |
| Estimate revisions | `IQ_EPS_EST`/`IQ_REVENUE_EST` with `asOfDate`; `ciqEstimate*` history | unverifiable | Kensho has no as-of date |
| Gross margin | `IQ_GROSS_MARGIN` (IQ_LTM); Kensho `gross_profit`(10)/`total_revenue`(28) | confirmed | |
| Operating margin | `IQ_EBIT_MARGIN`; `IQ_OPER_INC`; Kensho `operating_income`(21) | confirmed | |
| Net debt | `IQ_NET_DEBT`; Kensho `net_debt`(4364); `IQ_TOTAL_DEBT`(4173); `IQ_CASH_EQUIV`(1096) | confirmed | |
| Net debt/EBITDA | `IQ_NET_DEBT_EBITDA`; `IQ_TOTAL_DEBT_EBITDA` | confirmed | GDS parity untested |
| Short interest % float | none verified (`IQ_SHORT_INTEREST_PERCENT` = % of shares outstanding, community) | unverifiable | Do not ship. Source it from Bloomberg or LSEG. |
| Days to cover | compute | unverifiable | Needs shares short |
| Implied vol | **not available from S&P** | confirmed | Use Bloomberg or LSEG |
| Put/call | **not available from S&P** | confirmed | |
| GICS / industry | `IQ_PRIMARY_INDUSTRY`; `IQ_INDUSTRY`, `IQ_INDUSTRY_SECTOR`; Kensho `tickers(gics=...)` | unverifiable | The Kensho route is the verified one |
| US listing | `IQ_EXCHANGE`; Xpressfeed `countryId=213`, `currencyId=160`, `securitySubTypeId=1`; Kensho `tickers(exchange_code=...)` | confirmed | Constants are community-sourced. Kensho country = HQ country. |
| Transcripts | Kensho `get_latest_earnings_from_identifiers` → `get_transcript_from_key_dev_id`; Xpressfeed `ciqTranscript` + `ciqTranscriptComponent` (type 48) | corrected | De-duplicate versions per `keyDevId` |
| News | `IQ_NEWS` (CIQRANGE); Kensho `get_key_devs_from_identifier` | unverifiable | Key developments are curated events, not a newswire |
| SEC filings | CIQ document service | unverifiable | Use EDGAR directly (L0) |
| Broker research | not available via API | confirmed | |
| Index constituents | `IQ_CONSTITUENTS` via **GDSHV** + `StartRank/EndRank` | corrected | `^SPX`/Russell coverage untested |
| Shares outstanding | `IQ_SHARESOUTSTANDING`; Kensho `shares_outstanding` | confirmed | |
| Next earnings / surprise / target | `IQ_NEXT_EARNINGS_DATE`, `IQ_EST_EPS_SURPRISE_PERCENT`, `IQ_PRICE_TARGET`; Kensho `get_next_earnings_from_identifiers`, `get_consensus_target_price_from_identifiers` | confirmed | GDS parity untested |

---

## 4. Cross-vendor feature catalogue: where each ScreenSpec feature can be pushed down

| Feature | Bloomberg (Phase 1) | LSEG | S&P |
|---|---|---|---|
| US universe + size band | push (BQL) | push (SCREEN) | push (SQL) |
| SMA50 > SMA200 | push (`smavg(...).last('1')`) | local | push (SQL, adjusted) |
| Drawdown band | push (`maxmin`) | local (preflight `TR.PricePctChg52WkHigh`) | push (SQL) |
| RSI14 | push (`rsi`) | local | local |
| Volume ratio | push | local | push (SQL) |
| FCF yield | residual (admit `free_cash_flow_yield` or build from admitted items) | residual (`TR.FreeCashFlow` unverifiable) | push (SQL; capex sign) |
| Trailing revenue growth | residual (`sales_rev_turn` LTM vs as_of-1Y) | residual (FQ0/FQ-4) | residual (`IQ_TOTAL_REV_1YR_ANN_GROWTH`) |
| Short interest % float | residual (BDP, unverifiable) | residual (third-party fields) | **none** |
| Implied vol / IV−RV | column (`implied_volatility`, `volatility`) | survivors (`ATMIV.U`) | **none** |
| Put/call | none verified | compute from chain (unverified) | **none** |

## 5. Field admission procedure (gate G5)

1. Pick a liquid test security (`IBM US Equity`, `IBM.N`, `IBM:NYSE`) and an absolute date.
2. Run the item alone: BQLX/BQL Editor, `get_data` with `HeaderType.NAME`, or one `GDSP` call.
3. Fail on any of: an error, `fieldExceptions`, a dropped column, an all-NaN result, or `ErrMsg`.
4. Record `(vendor, item, params, test_ticker, as_of, value, units, sign, reviewer, date)` in `field_validation_log` (git).
5. Compare the value with a second source, a known filing, or a hand computation. Record whether the unit is a fraction or a percent.
6. Only then set `admitted=true`. The compiler refuses to use any non-admitted item as a threshold. It may use it as a z-score column only.

## Sources

- Bloomberg-authored BQuant notebooks (mirror): https://github.com/dmitchell28/Prop-Dev-Environments
- polars-bloomberg: https://marekozana.github.io/polars-bloomberg/usage/bql/
- blpapi changelog: https://github.com/msitt/blpapi-python/blob/master/changelog.txt
- BLPAPI Core Developer Guide: https://data.bloomberglp.com/professional/sites/10/2017/03/BLPAPI-Core-Developer-Guide.pdf
- Bloomberg Enterprise MCP (syndicated release): https://www.morningstar.com/news/pr-newswire/20260929ny57897/bloomberg-launches-enterprise-mcp-to-seamlessly-connect-bloomberg-data-with-clients-enterprise-ai-applications
- LSEG samples: https://github.com/LSEG-API-Samples/Example.DataLibrary.Python and https://github.com/LSEG-API-Samples/Example.EikonAPI.Python.VolatilityScreening
- lseg-data on PyPI: https://pypi.org/project/lseg-data/
- LSEG MCP registry entry: https://github.com/Azure/MCP/blob/main/partners/servers/lseg-mcp-server.json
- LSEG StreetEvents GenAI sample: https://github.com/LSEG-API-Samples/Article.AI.Transcripts.Python.GenAITranscriptsParseCountrySuppliers
- Capital IQ community clients: https://github.com/faaez/capiq-python and https://github.com/QuanTemplate/api-integrations
- S&P SDK: https://pypi.org/project/SPGMICIQ/
- Kensho kFinance: https://github.com/kensho-technologies/kfinance
- Xpressfeed community SQL: https://github.com/ZhengGong-hub/IBES_selected/blob/HEAD/capitaliq/databaseManager.py
- Anthropic MCP connector (not ZDR-eligible): https://platform.claude.com/docs/en/agents-and-tools/mcp-connector
