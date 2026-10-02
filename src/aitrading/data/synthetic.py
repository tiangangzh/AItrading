"""Deterministic, fully offline synthetic market-data provider.

Makes the whole pipeline demonstrable without vendor entitlements. It is a realistic simulator,
not random noise:

* Securities: fictional tickers and company names across all 11 GICS sectors, lognormal market
  caps (~0.3-300B USD), NYSE/NASDAQ listings, a few non-US ADRs, REITs, ETFs and preferreds, a
  handful of penny stocks, illiquid names and recent IPOs (so universe filters and short-history
  handling matter).
* Prices: market (GARCH-like) + sector + industry + idiosyncratic factor model with
  size-dependent volatility, quarterly earnings gaps, idiosyncratic shock days, persistent
  lognormal volume with event spikes, and consistent OHLC (low <= open, close <= high).
* Fundamentals / estimates: quarterly history (calendar or retail-style fiscal quarters) with
  report dates 25-45 days after period end. Snapshots are point-in-time on the report date;
  TTM = last four reported quarters; ``fcf_ttm == cfo_ttm - capex_ttm`` exactly. The balance
  sheet is consistent: total assets (optional columns ``total_assets`` / ``total_assets_prior_year``)
  = equity + debt + other liabilities, always above cash.
* Long windows (e.g. 2012-2026 for factor research; ~1-4 s to build 300-500 names): before the
  last ~3 years, each stock's drift is redrawn every 3 years (no 15-year persistence), returns are
  calibrated so the cap-weighted universe earns the simulated market factor, and fundamentals
  follow market-relative prices with a lag, so relative valuations stay realistic over decades
  instead of random-walking. The last ~3 years (the planted stories, and the whole of a
  default-length window) are unaffected.
* Short interest: semi-monthly settlement dates (15th and month end). Options: 30d ATM IV tied
  to realised volatility plus event premia, put/call volume and open interest.
* Documents: earnings-call transcripts, news, SEC-filing excerpts and research notes generated
  from the same simulated numbers, so every figure management quotes matches
  ``get_fundamentals`` and every price move a headline quotes matches ``get_price_history``.

Planted archetypes (synthetic-only GROUND TRUTH for evaluation, exposed by ``archetype()``,
``archetypes()`` and ``story()``; never revealed in any document):

1. ``transitory_shock`` (~3%): quality compounder in a long uptrend that sold off after an earnings
   report on heavy volume because of a one-off (destocking, FX, weather, supply-chain timing, ERP
   cut-over); estimates trimmed modestly, short interest rising, IV elevated; management quantifies
   the one-off and orders/backlog are intact.
2. ``value_trap`` (~3%): same price pattern and passes trailing screens, but the latest quarter
   decelerates sharply, margins compress, NTM estimates are cut hard (EPS -15..-30%, revenue
   -10..-18%) and the transcript reveals a structural problem with evasive answers.
3. ``guidance_reset`` (~2%): beat the quarter, guided conservatively (explicitly), net cash, buyback.
4. ``sector_contagion`` (~2%): the whole industry sold off on a macro / peer headline; fundamentals
   intact; management explains limited exposure.
5. ``momentum_leader`` (~5%): steady uptrend near highs, low short interest.
6. ``normal``: everyone else.

With the default arguments the canonical demo screen (US mid-caps in an established uptrend that
pulled back 15-40% on heavy volume, RSI < 40, FCF yield > 4%, revenue growth > 8%, short interest
> 6% of float), evaluated as of 2026-09-30, keeps roughly 10-20 names mixing archetypes 1-4. Some
planted names deliberately miss one or two thresholds ("near misses") so a screen has to work.

Planted paths are designed in total-return space and rejection-sampled per (seed, ticker, round)
until their technical profile matches the design, so the stories survive whatever the simulated
market does. Story timing is anchored to ``end`` (shock 30-46 sessions before it, follow-through
selling in the last 20 sessions); with a window shorter than 300 sessions nothing is planted.

Conventions: date columns are ``datetime64[ns]``; short-interest settlements are visible from the
settlement date; documents are generated lazily per ticker and cached; document timestamps are
naive datetimes in US/Eastern. Everything is a pure function of the constructor arguments (no wall
clock, no global RNG, no reliance on ``hash()``): the same arguments give byte-identical outputs.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from typing import TYPE_CHECKING, Any, Iterable

import numpy as np
import pandas as pd

from aitrading.core import fields as F
from aitrading.core.models import Document, DocumentKind, TranscriptSegment
from aitrading.core.policy import DataBoundary
from aitrading.data.base import Capability, PricePanel, ProviderError

if TYPE_CHECKING:  # pragma: no cover
    from aitrading.screen.spec import UniverseSpec

__all__ = ["SyntheticProvider", "ARCHETYPES", "ARCHETYPE_TO_DISLOCATION", "BENCHMARK_SYMBOL"]

ARCHETYPES: tuple[str, ...] = (
    "transitory_shock", "value_trap", "guidance_reset", "sector_contagion", "momentum_leader", "normal",
)
# Expected ``DislocationThesis.dislocation_type`` for the planted dislocation archetypes (evaluation aid).
ARCHETYPE_TO_DISLOCATION: dict[str, str] = {
    "transitory_shock": "transitory_fundamental_shock",
    "value_trap": "structural_decline_value_trap",
    "guidance_reset": "guidance_reset_overreaction",
    "sector_contagion": "sector_or_macro_contagion",
}
BENCHMARK_SYMBOL = "SYNTH-US"

_SOURCE = "Synthetic"
_NEWSWIRE = "Synthetic Newswire"
_ARCH_FRACTIONS = {
    "transitory_shock": 0.03, "value_trap": 0.03, "guidance_reset": 0.02, "sector_contagion": 0.02,
    "momentum_leader": 0.05,
}
_MIN_SESSIONS_FOR_PLANTING = 300
_MAX_PLANT_ROUNDS = 30
# Long-window valuation anchor (see SyntheticProvider._valuation_anchor): the latest 12 reported
# quarters keep their designed numbers; earlier fundamentals follow prices with a 4-quarter
# half-life - fully for each name's gap relative to the market, half for the market-wide gap.
_ANCHOR_FREE_QUARTERS = 12
_ANCHOR_HALF_LIFE_Q = 4.0
_ANCHOR_MARKET_SHARE = 0.5
_MIN_PRICE = 1e-4  # adjusted prices are stored with 4 decimals; never round a price down to 0
# Long windows: before the last 756 sessions (phased in over 252) each stock's persistent drift is
# replaced by drifts redrawn every 756 sessions (same distribution), see _regime_drifts.
_FREE_SESSIONS = 756
_DRIFT_REGIME_SESSIONS = 756

# ---------------------------------------------------------------------------------------------
# Reference data: sectors, industries, economics
# ---------------------------------------------------------------------------------------------

# w: share of names; beta: mean market beta; vol: idio-vol multiplier; ps: median price/sales;
# gm/om: gross / operating margin; da/capex/sbc: % of revenue; g/gsd: revenue growth mean / dispersion;
# lev: median total debt / TTM revenue.
_SECTORS: dict[str, dict[str, float]] = {
    "Information Technology": dict(w=0.14, beta=1.20, vol=1.15, ps=5.0, gm=0.58, om=0.18, da=0.04, capex=0.04, sbc=0.04, g=0.11, gsd=0.09, lev=0.35),
    "Health Care": dict(w=0.13, beta=0.85, vol=1.15, ps=3.6, gm=0.60, om=0.14, da=0.04, capex=0.04, sbc=0.025, g=0.08, gsd=0.08, lev=0.45),
    "Financials": dict(w=0.13, beta=1.05, vol=0.90, ps=2.8, gm=0.90, om=0.30, da=0.02, capex=0.02, sbc=0.015, g=0.06, gsd=0.05, lev=1.20),
    "Consumer Discretionary": dict(w=0.11, beta=1.15, vol=1.10, ps=1.4, gm=0.40, om=0.10, da=0.04, capex=0.045, sbc=0.01, g=0.06, gsd=0.06, lev=0.40),
    "Consumer Staples": dict(w=0.06, beta=0.65, vol=0.75, ps=1.6, gm=0.36, om=0.13, da=0.03, capex=0.035, sbc=0.005, g=0.04, gsd=0.03, lev=0.45),
    "Industrials": dict(w=0.15, beta=1.05, vol=0.95, ps=2.0, gm=0.33, om=0.14, da=0.035, capex=0.03, sbc=0.01, g=0.07, gsd=0.05, lev=0.40),
    "Energy": dict(w=0.05, beta=1.00, vol=1.25, ps=1.3, gm=0.35, om=0.16, da=0.09, capex=0.10, sbc=0.005, g=0.03, gsd=0.10, lev=0.45),
    "Materials": dict(w=0.06, beta=1.05, vol=1.00, ps=1.5, gm=0.28, om=0.13, da=0.06, capex=0.06, sbc=0.005, g=0.04, gsd=0.05, lev=0.50),
    "Utilities": dict(w=0.05, beta=0.55, vol=0.70, ps=2.6, gm=0.45, om=0.22, da=0.13, capex=0.25, sbc=0.003, g=0.05, gsd=0.02, lev=1.60),
    "Real Estate": dict(w=0.06, beta=0.90, vol=0.85, ps=7.0, gm=0.65, om=0.35, da=0.25, capex=0.10, sbc=0.01, g=0.05, gsd=0.04, lev=3.00),
    "Communication Services": dict(w=0.05, beta=1.00, vol=1.10, ps=2.5, gm=0.52, om=0.17, da=0.07, capex=0.07, sbc=0.02, g=0.06, gsd=0.06, lev=0.60),
}
_SECTOR_ORDER = list(_SECTORS)

# industry -> (weight within sector, name suffixes)
_INDUSTRIES: dict[str, list[tuple[str, float, tuple[str, ...]]]] = {
    "Information Technology": [
        ("Software", 0.30, ("Software", "Systems", "Analytics", "Cloud", "Labs")),
        ("Semiconductors & Semiconductor Equipment", 0.20, ("Semiconductor", "Microdevices", "Silicon", "Photonics")),
        ("IT Services", 0.12, ("Consulting", "Digital", "Technologies")),
        ("Electronic Equipment, Instruments & Components", 0.16, ("Instruments", "Electronics", "Sensors")),
        ("Technology Hardware, Storage & Peripherals", 0.11, ("Storage", "Computing", "Devices")),
        ("Communications Equipment", 0.11, ("Networks", "Communications", "Optical")),
    ],
    "Health Care": [
        ("Biotechnology", 0.24, ("Therapeutics", "Biosciences", "Bio", "Genomics")),
        ("Health Care Equipment & Supplies", 0.25, ("Medical", "Surgical", "Orthopedics")),
        ("Health Care Providers & Services", 0.18, ("Health", "Care Partners", "Healthcare")),
        ("Pharmaceuticals", 0.12, ("Pharmaceuticals", "Pharma")),
        ("Life Sciences Tools & Services", 0.13, ("Scientific", "Life Sciences", "Diagnostics")),
        ("Health Care Technology", 0.08, ("Health Technologies", "Clinical Systems")),
    ],
    "Financials": [
        ("Banks", 0.35, ("Bancorp", "Bancshares", "Bank")),
        ("Capital Markets", 0.20, ("Capital", "Asset Management", "Partners")),
        ("Insurance", 0.25, ("Insurance Group", "Assurance", "Specialty Insurance")),
        ("Financial Services", 0.12, ("Financial", "Payments")),
        ("Consumer Finance", 0.08, ("Credit", "Lending")),
    ],
    "Consumer Discretionary": [
        ("Specialty Retail", 0.22, ("Outfitters", "Stores", "Supply Co.")),
        ("Hotels, Restaurants & Leisure", 0.22, ("Hospitality", "Restaurants", "Resorts")),
        ("Automobile Components", 0.12, ("Automotive", "Drivetrain", "Motor Parts")),
        ("Household Durables", 0.14, ("Home Furnishings", "Appliances", "Homes")),
        ("Textiles, Apparel & Luxury Goods", 0.12, ("Apparel", "Brands", "Footwear")),
        ("Leisure Products", 0.08, ("Outdoor", "Recreation")),
        ("Broadline Retail", 0.05, ("Marketplace", "Commerce")),
        ("Diversified Consumer Services", 0.05, ("Education", "Home Services")),
    ],
    "Consumer Staples": [
        ("Food Products", 0.35, ("Foods", "Farms", "Provisions")),
        ("Beverages", 0.15, ("Beverage", "Brewing")),
        ("Household Products", 0.12, ("Household", "Home Care")),
        ("Personal Care Products", 0.13, ("Personal Care", "Beauty")),
        ("Consumer Staples Distribution & Retail", 0.17, ("Grocers", "Markets")),
        ("Tobacco", 0.08, ("Tobacco",)),
    ],
    "Industrials": [
        ("Machinery", 0.22, ("Machinery", "Industries", "Hydraulics")),
        ("Aerospace & Defense", 0.12, ("Aerospace", "Defense Systems", "Avionics")),
        ("Building Products", 0.10, ("Building Products", "Windows & Doors", "Roofing")),
        ("Electrical Equipment", 0.12, ("Electric", "Power Systems", "Electrical")),
        ("Professional Services", 0.10, ("Advisory", "Staffing")),
        ("Ground Transportation", 0.08, ("Freight", "Logistics")),
        ("Commercial Services & Supplies", 0.10, ("Environmental Services", "Facility Services")),
        ("Construction & Engineering", 0.08, ("Engineering", "Constructors")),
        ("Trading Companies & Distributors", 0.08, ("Industrial Supply", "Distribution")),
    ],
    "Energy": [
        ("Oil, Gas & Consumable Fuels", 0.70, ("Energy", "Petroleum", "Resources", "Midstream")),
        ("Energy Equipment & Services", 0.30, ("Oilfield Services", "Drilling")),
    ],
    "Materials": [
        ("Chemicals", 0.45, ("Chemicals", "Polymers", "Coatings")),
        ("Metals & Mining", 0.25, ("Mining", "Metals", "Steel")),
        ("Containers & Packaging", 0.18, ("Packaging", "Containers")),
        ("Construction Materials", 0.12, ("Aggregates", "Cement")),
    ],
    "Utilities": [
        ("Electric Utilities", 0.45, ("Power", "Electric")),
        ("Multi-Utilities", 0.25, ("Utilities", "Gas & Electric")),
        ("Gas Utilities", 0.15, ("Gas", "Natural Gas")),
        ("Water Utilities", 0.15, ("Water", "Water Works")),
    ],
    "Real Estate": [
        ("Industrial REITs", 0.15, ("Industrial Realty", "Logistics Properties")),
        ("Residential REITs", 0.20, ("Residential", "Apartment Communities")),
        ("Retail REITs", 0.15, ("Retail Properties", "Shopping Centers")),
        ("Specialized REITs", 0.25, ("Towers", "Data Centers", "Storage Trust")),
        ("Office REITs", 0.10, ("Office Properties",)),
        ("Real Estate Management & Development", 0.15, ("Land", "Realty Services")),
    ],
    "Communication Services": [
        ("Media", 0.30, ("Media", "Broadcasting", "Publishing")),
        ("Entertainment", 0.25, ("Entertainment", "Studios", "Games")),
        ("Interactive Media & Services", 0.25, ("Interactive", "Online")),
        ("Diversified Telecommunication Services", 0.12, ("Telecom", "Fiber")),
        ("Wireless Telecommunication Services", 0.08, ("Wireless", "Mobile")),
    ],
}
_INDUSTRY_ORDER = [ind for s in _SECTOR_ORDER for ind, _, _ in _INDUSTRIES[s]]
_INDUSTRY_SUFFIX = {ind: suf for s in _SECTOR_ORDER for ind, _, suf in _INDUSTRIES[s]}

# Planted-archetype settings: (sector, industry, theme). Themes drive the narrative.
_TRANSITORY_SETTINGS = [
    ("Industrials", "Machinery", "destocking"),
    ("Information Technology", "Electronic Equipment, Instruments & Components", "destocking"),
    ("Health Care", "Health Care Equipment & Supplies", "supply_chain"),
    ("Materials", "Chemicals", "weather"),
    ("Consumer Staples", "Food Products", "destocking"),
    ("Health Care", "Life Sciences Tools & Services", "fx"),
    ("Industrials", "Commercial Services & Supplies", "erp"),
    ("Consumer Discretionary", "Leisure Products", "weather"),
    ("Information Technology", "Technology Hardware, Storage & Peripherals", "erp"),
    ("Consumer Staples", "Personal Care Products", "fx"),
    ("Industrials", "Electrical Equipment", "supply_chain"),
    ("Industrials", "Building Products", "weather"),
]
_VALUE_TRAP_SETTINGS = [
    ("Information Technology", "Software", "churn"),
    ("Health Care", "Health Care Equipment & Supplies", "share_loss"),
    ("Consumer Discretionary", "Household Durables", "pricing"),
    ("Information Technology", "Communications Equipment", "transition"),
    ("Industrials", "Machinery", "share_loss"),
    ("Information Technology", "IT Services", "pricing"),
    ("Consumer Staples", "Food Products", "share_loss"),
    ("Communication Services", "Media", "churn"),
    ("Information Technology", "Technology Hardware, Storage & Peripherals", "transition"),
]
_GUIDANCE_SETTINGS = [
    ("Information Technology", "Software", "deal_timing"),
    ("Health Care", "Life Sciences Tools & Services", "macro"),
    ("Industrials", "Aerospace & Defense", "new_cfo"),
    ("Health Care", "Health Care Equipment & Supplies", "new_cfo"),
    ("Industrials", "Machinery", "macro"),
    ("Consumer Discretionary", "Hotels, Restaurants & Leisure", "macro"),
    ("Information Technology", "Electronic Equipment, Instruments & Components", "deal_timing"),
]
_CONTAGION_SCENARIOS: list[dict[str, Any]] = [
    dict(key="export_controls", sector="Information Technology", industry="Semiconductors & Semiconductor Equipment",
         group="chip stocks", peer="Arcturus Microsystems",
         event="a proposed expansion of export controls on advanced semiconductor equipment and chips",
         headline="Chip stocks slide after Washington floats broader export controls",
         followup="Chip stocks extend losses as draft export rule text is published",
         topic="China-based customers", topic_short="China", affected="the proposed rules",
         markets=("automotive", "industrial automation", "medical electronics")),
    dict(key="hyperscaler_pause", sector="Industrials", industry="Electrical Equipment",
         group="electrical-equipment stocks", peer="Veltrane Power",
         event="a large hyperscale cloud operator saying it would pause several data-center construction projects",
         headline="Electrical-equipment stocks tumble after cloud giant pauses data-center projects",
         followup="Electrical-equipment shares fall again as a second cloud operator trims its capex plan",
         topic="hyperscale data-center projects", topic_short="hyperscale data centers", affected="the announced pauses",
         markets=("utility grid", "commercial buildings", "industrial facilities")),
    dict(key="ma_rate_notice", sector="Health Care", industry="Health Care Providers & Services",
         group="health-care services stocks", peer="Corvessa Health",
         event="a proposed Medicare Advantage rate notice that came in well below expectations",
         headline="Health-care services stocks drop after Medicare Advantage rate proposal disappoints",
         followup="Health-care services stocks slide again as final rate notice confirms cut",
         topic="Medicare Advantage", topic_short="Medicare Advantage", affected="the rate notice",
         markets=("commercial insurance", "Medicaid", "employer-sponsored plans")),
    dict(key="housing_warning", sector="Industrials", industry="Building Products",
         group="building-products stocks", peer="Halcyon Building Systems",
         event="a large peer's profit warning that blamed a sharp slowdown in new housing starts",
         headline="Building-products stocks sink after peer warns on housing slowdown",
         followup="Building-products stocks fall further as housing starts hit a multi-year low",
         topic="new residential construction", topic_short="new housing", affected="the housing slowdown",
         markets=("repair and remodel", "non-residential", "institutional")),
]

_PLANT_MISS_WEIGHTS: dict[str, dict[str, float]] = {
    "transitory_shock": dict(cap=0.20, drawdown=0.15, rsi=0.14, volume=0.10, trend=0.10, fcf=0.10, growth=0.08, si=0.13),
    "value_trap": dict(cap=0.20, drawdown=0.18, rsi=0.14, volume=0.10, trend=0.10, fcf=0.10, growth=0.04, si=0.14),
    "guidance_reset": dict(cap=0.25, drawdown=0.25, rsi=0.15, volume=0.10, trend=0.05, si=0.20),
    "sector_contagion": dict(cap=0.25, drawdown=0.20, rsi=0.15, volume=0.10, fcf=0.10, si=0.20),
}

# ---------------------------------------------------------------------------------------------
# Names
# ---------------------------------------------------------------------------------------------

_PREFIXES = (
    "Northwind", "Bluepeak", "Cobaltine", "Halvard", "Tessaline", "Marrowgate", "Quillon", "Ardentis", "Brightmoor",
    "Calderon", "Driftwood", "Emberly", "Fennimore", "Granitefall", "Harrowby", "Isolane", "Junipero", "Kestrelline",
    "Merriwick", "Nimbuscore", "Oakhollow", "Pellucid", "Quarrystone", "Ravenmoor", "Saltmarsh", "Thistledown",
    "Umberfield", "Valemont", "Wyncrest", "Yarrowfield", "Zephyrine", "Alderbrook", "Birchline", "Coppervale",
    "Dunmore", "Elmstead", "Foxglove", "Glenhaven", "Hollowell", "Ironbark", "Jasperline", "Kingsferry",
    "Lanternfield", "Marlowe", "Norcrest", "Orchardine", "Pinecrest", "Quaywell", "Rosebay", "Silverfen",
    "Tamberlin", "Uplander", "Verdance", "Ashgrove", "Beaconridge", "Cindervale", "Dovetail", "Eastmere",
    "Fairhollow", "Greystoke", "Hearthstone", "Indigo Bay", "Jadeport", "Kilnworth", "Lumenvale", "Moonstone",
    "Nettlefield", "Opaline", "Peregrine", "Quintessa", "Rimefield", "Stonebridge", "Tidewell", "Vantor",
    "Wildmere", "Yellowpine", "Zirconia", "Amberlake", "Bramblecote", "Corvane", "Dalesford", "Everglow",
    "Ferncliff", "Gildencroft", "Hazelridge", "Ivorytown", "Kelpwater", "Millbrook", "Northgate", "Oxbow",
    "Palisade", "Quickwater", "Redfern", "Sablecrest", "Tarnhold", "Underhill", "Vireo", "Windrose", "Yewcroft",
    "Zenara", "Axton", "Brackwater", "Crestmoor", "Duskfield", "Embervale", "Fallowmere", "Goldcrest",
    "Highmarsh", "Ironvale", "Kettlewell", "Lindenmoor", "Mistral", "Nightjar", "Oriel", "Pemberly", "Riverine",
    "Starling", "Thornbury", "Vellum", "Wexmoor", "Arlenford", "Belcourt", "Castellan", "Dorrance", "Elverson",
    "Farrowdale", "Gallant", "Hartwell", "Ingleby", "Jessamine", "Kenmare", "Lockridge", "Mortlake", "Newhaven",
    "Ostrander", "Prescott", "Quenby", "Rookwood", "Sheffold", "Trevanion", "Ulmstead", "Varrick", "Westcliffe",
    "Ainsley", "Blackthorn", "Carrow", "Delacroix", "Eldermoor", "Fenwick", "Grantham", "Holloway", "Inverness",
    "Larkhill", "Merrimont", "Norwood", "Pellham", "Quarrington", "Ravensworth", "Selwyn", "Tolliver", "Wendover",
)
_ETF_THEMES = (
    ("US Total Market", "TM"), ("US Mid Cap", "MC"), ("US Dividend Growth", "DG"), ("US Quality Factor", "QF"),
    ("US Small Cap Value", "SV"), ("Clean Energy", "CE"), ("Semiconductor", "SC"), ("Regional Banks", "RB"),
    ("Aggregate Bond", "AB"), ("Short Treasury", "ST"), ("Momentum Factor", "MF"), ("Low Volatility", "LV"),
)
_ETF_SPONSORS = ("Synthwave", "Meridale", "Tessera")
_ADR_COUNTRIES = ("GB", "DE", "NL", "CH", "JP", "IE", "FR", "DK", "SE", "IL")
_FIRST_NAMES = (
    "Alicia", "Marcus", "Priya", "Daniel", "Helena", "Tomas", "Grace", "Rafael", "Naomi", "Owen", "Ingrid", "Victor",
    "Leah", "Samuel", "Mei", "Jonah", "Clara", "Andre", "Fiona", "Hiroshi", "Yasmin", "Patrick", "Elena", "Kofi",
    "Rebecca", "Mateo", "Hannah", "Desmond", "Sofia", "Gregory", "Amara", "Lukas", "Julia", "Nathaniel", "Rosa",
    "Elliot", "Tara", "Benedict", "Camille", "Dmitri", "Imogen", "Caleb", "Lorena", "Arjun", "Margot", "Wesley",
)
_LAST_NAMES = (
    "Whitfield", "Okafor", "Lindqvist", "Ramanathan", "Castellano", "Brennan", "Haverford", "Nakamura", "Delacorte",
    "Ashworth", "Petrova", "Mbeki", "Sorensen", "Galloway", "Achebe", "Fairbanks", "Holloway", "Iverson", "Kowalski",
    "Laurent", "Montague", "Novak", "Osei", "Pemberton", "Quintero", "Rasmussen", "Sterling", "Takahashi", "Underwood",
    "Vasquez", "Wexler", "Yamamoto", "Zielinski", "Abernathy", "Blackwood", "Calloway", "Draper", "Ellsworth",
    "Fitzgerald", "Granger", "Hartley", "Ingram", "Jansen", "Kingsley", "Lockhart", "Marchetti", "Nordstrom",
    "Oyelaran", "Prescott", "Radcliffe", "Sandoval", "Thorne", "Vance", "Winslow",
)
_BROKERS = (
    "Halvorsen & Pike", "Granite Ridge Securities", "Copperline Capital Markets", "Whitlock Brothers",
    "Ardent Bay Partners", "Tidewater Research Group", "Larkspur & Co.", "Meridale Securities", "Bramwell Fairchild",
    "Orison Capital", "Kestrel Point Advisors", "Northcote Equities", "Saltire Lane", "Vandermeer Hollis",
    "Cresset Hill Securities", "Pellham Grant",
)
# Well-known real symbols the generator must never emit (best-effort guard; the rest are random).
_TICKER_BLOCKLIST = frozenset(
    "AAPL MSFT AMZN GOOG GOOGL META NVDA TSLA BRK JPM UNH XOM JNJ WMT PG HD KO PEP IBM AMD INTC CSCO ORCL CRM "
    "ADBE NFLX DIS NKE MCD CAT GS BAC WFC CVX PFE MRK ABBV LLY AVGO QCOM TXN COST SBUX UPS FDX SPY QQQ IWM DIA "
    "VTI VOO GLD TLT HYG LQD XLF XLK XLE XLV XLI XLP XLU XLB XLY XLC XLRE BABA TSM ASML SAP SONY TM HSBC SHEL "
    "BP RIO BHP UBS NVO AZN GSK DEO ACN LIN TMO DHR ABT MDT ISRG SYK BSX AMGN GILD VRTX REGN BMY CVS CI HUM ELV "
    "MMM HON GE BA RTX LMT NOC GD DE EMR ETN ITW PH ROK CMI PCAR UNP CSX NSC DAL UAL AAL LUV MAR HLT BKNG ABNB "
    "UBER LYFT SNAP PINS SHOP SQ PYPL INTU NOW SNOW PLTR NET DDOG CRWD ZS PANW FTNT OKTA TEAM MDB WDAY ADSK ANSS "
    "CDNS SNPS KLAC LRCX AMAT MU WDC STX HPQ HPE DELL ANET JNPR MSI NOK ERIC VZ CMCSA CHTR TMUS EA TTWO RBLX "
    "WBD PARA FOX NWSA LOW TGT TJX ROST DG DLTR KR SYY GIS KHC MDLZ HSY CL KMB EL CLX MO PM STZ TAP KDP MNST "
    "NEE DUK SO D AEP EXC SRE PCG ED XEL PLD AMT CCI EQIX PSA SPG O WELL AVB EQR DLR SBAC VICI COP EOG OXY PSX "
    "MPC VLO SLB HAL BKR KMI WMB OKE FCX NEM NUE DOW DD PPG SHW ECL APD MLM VMC IP PKG BALL V MA AXP COF SCHW "
    "MS BLK SPGI MCO ICE CME CB PGR TRV AIG MET PRU AFL ALL USB PNC TFC FITB KEY RF HBAN MTB CFG ZION CMA "
    "A ALL ARE BEN CAR CAKE FUN GOOD HAS JOY LOVE MAIN PLAY RACE ROCK SAFE SAVE TRUE WELL WORK YOU ZEN".split()
)
_ORDINAL = {1: "first", 2: "second", 3: "third", 4: "fourth"}
_MONTHS = ("January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
           "November", "December")

# ---------------------------------------------------------------------------------------------
# Narrative vocabulary
# ---------------------------------------------------------------------------------------------

_SECTOR_VOCAB: dict[str, dict[str, Any]] = {
    "Information Technology": dict(products=("core platform", "next-generation product line", "services portfolio"), customers="enterprise customers", markets=("enterprise IT", "industrial", "public sector"), kpi="backlog"),
    "Health Care": dict(products=("core franchise", "pipeline programs", "international business"), customers="providers and payers", markets=("U.S. hospitals", "outpatient settings", "international markets"), kpi="order book"),
    "Financials": dict(products=("commercial lending book", "fee-based businesses", "deposit franchise"), customers="commercial and retail clients", markets=("commercial real estate", "middle-market lending", "wealth management"), kpi="pipeline"),
    "Consumer Discretionary": dict(products=("core assortment", "direct-to-consumer channel", "loyalty program"), customers="consumers", markets=("U.S. stores", "e-commerce", "international"), kpi="order book"),
    "Consumer Staples": dict(products=("core brands", "premium tier", "innovation pipeline"), customers="retail partners", markets=("grocery", "club", "convenience"), kpi="distribution gains"),
    "Industrials": dict(products=("original-equipment business", "aftermarket and services", "engineered components"), customers="industrial customers", markets=("non-residential construction", "infrastructure", "general industrial"), kpi="backlog"),
    "Energy": dict(products=("upstream portfolio", "midstream assets", "services fleet"), customers="customers", markets=("Permian Basin", "Gulf Coast", "international"), kpi="contracted volumes"),
    "Materials": dict(products=("specialty products", "commodity grades", "engineered materials"), customers="industrial customers", markets=("packaging", "construction", "automotive"), kpi="order book"),
    "Utilities": dict(products=("regulated utility", "transmission investments", "renewables portfolio"), customers="customers", markets=("residential", "commercial", "industrial load"), kpi="rate base"),
    "Real Estate": dict(products=("core portfolio", "development pipeline", "redevelopment projects"), customers="tenants", markets=("Sun Belt", "coastal", "secondary"), kpi="leasing pipeline"),
    "Communication Services": dict(products=("subscription offering", "advertising business", "content slate"), customers="subscribers and advertisers", markets=("U.S.", "international", "connected TV"), kpi="subscriber base"),
}
_INDUSTRY_VOCAB: dict[str, dict[str, Any]] = {
    "Machinery": dict(products=("compact construction equipment", "precision gear drives", "aftermarket parts and service"), customers="dealers", markets=("construction", "agriculture", "mining"), kpi="backlog"),
    "Electrical Equipment": dict(products=("medium-voltage switchgear", "power distribution units", "grid automation controls"), customers="utilities and data-center developers", markets=("data centers", "utility grid", "commercial buildings"), kpi="backlog"),
    "Electronic Equipment, Instruments & Components": dict(products=("test and measurement instruments", "precision sensors", "embedded connectivity modules"), customers="distribution partners and OEMs", markets=("industrial automation", "aerospace", "automotive electronics"), kpi="backlog"),
    "Semiconductors & Semiconductor Equipment": dict(products=("analog power-management chips", "wafer inspection systems", "RF front-end modules"), customers="OEM and foundry customers", markets=("data center", "industrial", "automotive"), kpi="backlog"),
    "Health Care Equipment & Supplies": dict(products=("minimally invasive surgical instruments", "infusion systems", "single-use consumables"), customers="hospital systems and group purchasing organizations", markets=("acute care", "ambulatory surgery centers", "international"), kpi="order book"),
    "Life Sciences Tools & Services": dict(products=("bioprocessing consumables", "mass spectrometry systems", "lab automation"), customers="biopharma and academic labs", markets=("biopharma", "academic and government", "applied markets"), kpi="backlog"),
    "Health Care Providers & Services": dict(products=("home health services", "outpatient surgery centers", "specialty pharmacy"), customers="payers and patients", markets=("Medicare Advantage", "commercial", "Medicaid"), kpi="patient volumes"),
    "Chemicals": dict(products=("crop-protection additives", "performance coatings", "engineered polymers"), customers="formulators and distributors", markets=("agriculture", "construction", "packaging"), kpi="order book"),
    "Food Products": dict(products=("premium snack brands", "frozen prepared meals", "plant-based protein lines"), customers="grocery and club retailers", markets=("grocery", "club", "foodservice"), kpi="distribution points"),
    "Personal Care Products": dict(products=("prestige skin care", "hair care", "fragrance"), customers="retail and travel-retail partners", markets=("North America", "Europe", "Latin America"), kpi="order book"),
    "Leisure Products": dict(products=("outdoor recreation equipment", "premium coolers", "e-bikes"), customers="sporting-goods retailers", markets=("specialty retail", "direct-to-consumer", "international"), kpi="dealer orders"),
    "Building Products": dict(products=("energy-efficient windows", "commercial roofing systems", "HVAC controls"), customers="distributors and contractors", markets=("repair and remodel", "new residential construction", "non-residential"), kpi="backlog"),
    "Technology Hardware, Storage & Peripherals": dict(products=("all-flash storage arrays", "ruggedized computing systems", "edge servers"), customers="enterprise and channel partners", markets=("enterprise IT", "public sector", "industrial edge"), kpi="backlog"),
    "Commercial Services & Supplies": dict(products=("facility maintenance contracts", "document security services", "uniform programs"), customers="commercial clients", markets=("healthcare facilities", "education", "industrial sites"), kpi="contracted backlog"),
    "Software": dict(products=("workflow automation platform", "security analytics suite", "data integration tools"), customers="enterprise customers", markets=("financial services", "healthcare", "public sector"), kpi="remaining performance obligations"),
    "Communications Equipment": dict(products=("optical transport systems", "enterprise Wi-Fi access points", "cable access equipment"), customers="service providers and enterprises", markets=("service provider", "enterprise", "cable"), kpi="backlog"),
    "IT Services": dict(products=("managed cloud services", "application modernization", "digital engineering"), customers="enterprise clients", markets=("banking", "healthcare", "retail"), kpi="bookings"),
    "Household Durables": dict(products=("kitchen appliances", "mattresses", "outdoor furniture"), customers="retail partners", markets=("big-box retail", "e-commerce", "specialty dealers"), kpi="order book"),
    "Media": dict(products=("subscription news and data products", "local broadcast stations", "digital advertising"), customers="subscribers and advertisers", markets=("national", "local", "digital"), kpi="subscriber base"),
    "Aerospace & Defense": dict(products=("flight-control actuators", "avionics displays", "aftermarket repair services"), customers="airframers and defense primes", markets=("commercial aerospace", "defense", "business jets"), kpi="backlog"),
    "Hotels, Restaurants & Leisure": dict(products=("company-operated restaurants", "franchise system", "loyalty program"), customers="guests", markets=("urban", "suburban", "travel centers"), kpi="development pipeline"),
}

_TRANSITORY_ISSUES: dict[str, dict[str, str]] = {
    "destocking": dict(
        short="customer inventory destocking",
        side="the issue was inventory on their side of the channel, and sell-through to end users kept growing",
        what="inventory destocking at {n_cust} of our largest {customers}, who reduced channel inventory after a period of elevated ordering",
        evidence="sell-through at those same partners was up {sellthrough}% year over year, and channel inventory ended the quarter at roughly {weeks} weeks against a normal range of {norm_lo} to {norm_hi} weeks",
        normal="we expect ordering to realign with sell-through by the end of the {next_q} quarter",
    ),
    "fx": dict(
        short="one-time currency translation and hedge-settlement impact",
        side="the impact was translational and one-time; local-currency demand was a record",
        what="a one-time currency impact tied to the abrupt devaluation of the {currency}, which hit translated revenue and forced an early settlement of our hedge book",
        evidence="on a constant-currency basis our {region} business grew {sellthrough}% and order intake there was a record",
        normal="the hedge book has been reset at current rates, so we do not expect a repeat",
    ),
    "weather": dict(
        short="unusual weather in our core markets",
        side="the demand was deferred by the weather, not lost, and their seasonal plans are unchanged",
        what="unusually wet and cold weather across our core {region} markets during the first {weeks} weeks of the season, which pushed out {customers} orders",
        evidence="as conditions normalized, weekly orders in the final {norm_lo} weeks of the quarter were up {sellthrough}% year over year",
        normal="the demand did not go away; it shifted into the {next_q} quarter",
    ),
    "supply_chain": dict(
        short="temporary supply-chain disruption",
        side="the delay came from a single component on our side; none of them cancelled or re-sourced the orders",
        what="a {weeks}-week disruption at a single-source supplier of a {component}, which left finished orders waiting on one part at quarter end",
        evidence="we exited the quarter with roughly {oneoff} of fully built units awaiting that component, and the supplier has been running at full output since mid-{month}",
        normal="we expect to ship substantially all of those units in the {next_q} quarter",
    ),
    "erp": dict(
        short="one-time ERP cut-over",
        side="the delay was in our shipping systems, not in their demand, and their order books with us are intact",
        what="the cut-over to our new enterprise resource planning system at our {region} distribution center, which delayed shipments for roughly {weeks} weeks",
        evidence="order intake was unaffected and actually grew {sellthrough}% in the quarter; the issue was entirely on the shipping side",
        normal="the system has been stable since the {norm_lo}th week of the quarter and the shipment backlog is being worked down",
    ),
}
_VALUE_TRAP_PROBLEMS: dict[str, dict[str, str]] = {
    "share_loss": dict(
        short="competition from a lower-priced entrant",
        hint="We are seeing a new lower-priced competitor in parts of the {segment} market, but our installed base and service model remain a strong differentiator.",
        ceo="Competitive intensity increased in the quarter. A lower-priced competitor that entered the {segment} market last year has become more aggressive, and in several competitive evaluations we chose not to match price.",
        mdna="lower unit volumes in our {segment} product lines, reflecting increased competition from lower-priced alternatives, and lower average selling prices",
    ),
    "churn": dict(
        short="elevated customer churn",
        hint="Gross retention dipped slightly among smaller customers, which we attribute to normal budget scrutiny.",
        ceo="Retention in our small and mid-sized customer base was weaker than we planned. Some customers consolidated vendors or moved to lower-cost bundled alternatives at renewal.",
        mdna="higher customer attrition in our small and mid-sized customer cohorts and lower expansion within existing accounts",
    ),
    "pricing": dict(
        short="pricing pressure",
        hint="Customers are scrutinizing pricing more closely, and we have been more targeted with promotions.",
        ceo="The pricing environment became more difficult. Large customers are consolidating purchasing and running more competitive tenders, and we made price concessions to protect key relationships.",
        mdna="lower average selling prices across our {segment} portfolio as customers consolidated purchasing, partially offset by modest volume growth",
    ),
    "transition": dict(
        short="a delayed product transition",
        hint="The ramp of our next-generation platform is slightly behind the plan we laid out, but customer feedback on the early units is encouraging.",
        ceo="The transition to our next-generation platform is taking longer than we expected. Customers are delaying purchases of the current generation while qualification of the new platform continues.",
        mdna="lower sales of our current-generation {segment} products ahead of the transition to our next-generation platform, which is taking longer than anticipated",
    ),
}
_GUIDANCE_REASONS: dict[str, dict[str, str]] = {
    "deal_timing": dict(
        short="large-deal timing",
        assume="we have assumed that none of the large transactions currently in late-stage negotiation close in the {next_q} quarter, and we have haircut our historical close rates on the rest of the pipeline by roughly {haircut} points",
    ),
    "macro": dict(
        short="the macro backdrop",
        assume="we have assumed no improvement in order rates from current levels for the rest of the year, a {fx_bp}-basis-point currency headwind at today's spot rates, and a slower start to the budget season for our {customers}",
    ),
    "new_cfo": dict(
        short="a more conservative guidance philosophy",
        assume="we have moved to a guidance framework that we expect to beat in the large majority of scenarios; it assumes order rates stay flat, no contribution from new products we have not yet shipped, and roughly {haircut} points of extra cushion on top of that",
    ),
}


# ---------------------------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------------------------


def _as_date(x: object) -> date:
    if isinstance(x, datetime):  # includes pd.Timestamp
        return x.date()
    if isinstance(x, date):
        return x
    if isinstance(x, np.datetime64):
        return pd.Timestamp(x).date()
    if isinstance(x, str):
        return date.fromisoformat(x[:10])
    raise TypeError(f"expected a date, got {type(x).__name__}")


def _stable_hash(*keys: object) -> int:
    """64-bit hash that, unlike ``hash()``, is stable across processes."""
    raw = "\x1f".join(str(k) for k in keys).encode()
    return int.from_bytes(hashlib.blake2b(raw, digest_size=8).digest(), "little")


def _month_end(year: int, month: int) -> np.datetime64:
    y, m = year + (month - 1) // 12, (month - 1) % 12 + 1
    ny, nm = (y + 1, 1) if m == 12 else (y, m + 1)
    return np.datetime64(f"{ny:04d}-{nm:02d}-01") - np.timedelta64(1, "D")


def _d64(x: date) -> np.datetime64:
    return np.datetime64(x.isoformat(), "D")


def _py(d: np.datetime64) -> date:
    return date.fromisoformat(str(np.datetime64(d, "D")))


def _money(x: float) -> str:
    """USD amount the way management says it: "$512.3 million" / "$1.24 billion" (absolute value)."""
    a = abs(float(x))
    return f"${a / 1e9:,.2f} billion" if a >= 1e9 else f"${a / 1e6:,.1f} million"


def _pct(x: float, dp: int = 1) -> str:
    return f"{abs(float(x)) * 100:.{dp}f}%"


def _bps(x: float) -> str:
    return f"{abs(round(float(x) * 1e4)):.0f} basis points"


def _updown(x: float, up: str = "up", down: str = "down") -> str:
    return up if x >= 0 else down


def _long_date(d: date) -> str:
    return f"{_MONTHS[d.month - 1]} {d.day}, {d.year}"


def _wilder_rsi_last(close: np.ndarray, n: int = 14) -> np.ndarray:
    """Wilder RSI at the last row for each column of a (T, m) price matrix."""
    d = np.diff(close, axis=0)
    up = pd.DataFrame(np.clip(d, 0, None)).ewm(alpha=1 / n, adjust=False).mean().to_numpy()[-1]
    dn = pd.DataFrame(np.clip(-d, 0, None)).ewm(alpha=1 / n, adjust=False).mean().to_numpy()[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(dn > 0, 100 - 100 / (1 + up / np.where(dn > 0, dn, 1)), 100.0)


def _ohlc_log(ret: np.ndarray, ev: np.ndarray, ovn: np.ndarray, hu: np.ndarray, hd: np.ndarray):
    """Log OHLC from log close-to-close returns; event days gap mostly overnight and range wider."""
    lc = np.cumsum(ret, axis=0)
    phi = np.where(ev, 0.85, 0.30)
    lo = np.empty_like(lc)
    lo[0] = lc[0] + ovn[0]
    lo[1:] = lc[:-1] + phi[1:] * ret[1:] + ovn[1:]
    wide = np.where(ev, 2.0, 1.0)
    lh = np.maximum(lo, lc) + hu * wide
    ll = np.minimum(lo, lc) - hd * wide
    return lc, lo, lh, ll


@dataclass
class _Story:
    archetype: str
    fit: bool = False  # designed to pass the canonical demo screen as of the provider's end date
    misses: tuple[str, ...] = ()  # canonical conditions deliberately missed (near misses)
    theme: str = ""  # transitory issue / structural problem / guidance reason / contagion scenario key
    scenario: int = -1  # index into provider._scenarios (sector_contagion only)
    story_k: int = -1  # quarter index of the story quarter (the latest one reported by ``end``)
    report_session: int = -1
    shock_session: int = -1  # report session, or the industry headline session for contagion
    cap_session: int = -1  # follow-through / capitulation session (-1: none)
    targets: dict[str, float] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------------------------


class SyntheticProvider:
    """Deterministic offline ``MarketDataProvider`` with planted, internally consistent stories.

    ``get_universe`` applies only the cheap reference filters ``UniverseSpec.country`` and
    ``UniverseSpec.security_types`` (plus "listed on or before as_of"); price, liquidity and sector
    filters are left to the screen engine. Unknown tickers yield all-NaN rows / columns rather than
    errors. Dates after ``end`` are clamped to ``end`` (the provider knows nothing later).
    """

    name: str = "synthetic"

    def __init__(
        self,
        n_tickers: int = 500,
        seed: int = 7,
        start: date = date(2024, 1, 2),
        end: date = date(2026, 9, 30),
    ) -> None:
        start, end = _as_date(start), _as_date(end)
        if int(n_tickers) < 1:
            raise ValueError("n_tickers must be >= 1")
        if end < start:
            raise ValueError("end must be on or after start")
        dates = pd.bdate_range(start, end)
        if len(dates) < 2:
            raise ValueError("the [start, end] window must contain at least two business days")
        self.n_tickers = int(n_tickers)
        self.seed = int(seed)
        self.start = start
        self.end = end
        self.capabilities: set[Capability] = {c for c in Capability if c is not Capability.SCREEN_PUSHDOWN}
        self.boundary = DataBoundary(provider=self.name)
        self._dates = dates
        self._dd = dates.values.astype("datetime64[D]")
        self._N = len(dates)
        self._end_d = _d64(end)

        self._build_reference()
        self._assign_archetypes()
        self._build_identities()
        self._build_calendar()
        self._simulate_prices()
        self._build_fundamentals()
        self._build_estimates()
        self._build_short_interest()
        self._build_benchmark()
        self._doc_cache: dict[str, list[Document]] = {}
        self._options: dict[str, np.ndarray] | None = None

    # ------------------------------------------------------------------ RNG / lookups

    def _rng(self, *keys: object) -> np.random.Generator:
        h = _stable_hash(*keys)
        return np.random.default_rng([self.seed % (1 << 63), h & 0xFFFFFFFF, h >> 32])

    def _session_le(self, d: date) -> int:
        """Index of the last session on or before ``d`` (-1 if none)."""
        return int(np.searchsorted(self._dd, _d64(d), side="right")) - 1

    def _rows(self, tickers: Iterable[str]) -> tuple[list[str], np.ndarray]:
        """De-duplicated tickers (order kept) and their row indices (-1 for unknown)."""
        seen: dict[str, None] = {}
        for t in tickers:
            seen.setdefault(str(t), None)
        out = list(seen)
        return out, np.array([self._idx.get(t, -1) for t in out], dtype=np.int64)

    # ------------------------------------------------------------------ reference data

    def _build_reference(self) -> None:
        n, N = self.n_tickers, self._N
        rng = self._rng("reference")
        w = np.array([_SECTORS[s]["w"] for s in _SECTOR_ORDER])
        sec = rng.choice(len(_SECTOR_ORDER), size=n, p=w / w.sum())
        u = rng.random(n)
        sector = np.array([_SECTOR_ORDER[j] for j in sec], dtype=object)
        industry = np.empty(n, dtype=object)
        for i in range(n):
            inds = _INDUSTRIES[sector[i]]
            cw = np.cumsum([x[1] for x in inds])
            industry[i] = inds[min(int(np.searchsorted(cw / cw[-1], u[i])), len(inds) - 1)][0]
        stype = np.array(["reit" if "REIT" in x else "common_stock" for x in industry], dtype=object)
        country = np.full(n, "US", dtype=object)
        role = np.full(n, "", dtype=object)
        order = rng.permutation(n)
        for r, frac, min_n in (("etf", 0.010, 50), ("preferred", 0.012, 50), ("adr", 0.030, 30),
                               ("penny", 0.016, 50), ("illiquid", 0.025, 40), ("ipo", 0.010, 100)):
            quota, cnt = (int(round(frac * n)) if n >= min_n else 0), 0
            for i in order:
                if cnt >= quota:
                    break
                if role[i] or (r not in ("etf", "preferred") and stype[i] != "common_stock"):
                    continue
                role[i] = r
                cnt += 1
        for i in range(n):
            if role[i] == "etf":
                stype[i], sector[i], industry[i] = "etf", None, None
            elif role[i] == "preferred":
                stype[i] = "preferred"
                if sector[i] not in ("Financials", "Utilities"):
                    sector[i], industry[i] = "Financials", ("Banks" if rng.random() < 0.6 else "Insurance")
            elif role[i] == "adr":
                stype[i], country[i] = "adr", _ADR_COUNTRIES[int(rng.integers(len(_ADR_COUNTRIES)))]

        rc = self._rng("caps")
        lc = rc.normal(9.72, 0.62, size=8 * n + 16)
        lc = lc[(lc >= math.log10(0.3e9)) & (lc <= math.log10(300e9))][:n]
        price = np.exp(rc.normal(math.log(48.0), 0.7, n)).clip(6.0, 900.0)
        for i in range(n):
            if role[i] == "etf":
                lc[i], price[i] = rc.uniform(8.7, 10.7), rc.uniform(20, 400)
            elif role[i] == "preferred":
                lc[i], price[i] = rc.uniform(8.3, 9.2), 25.0 * rc.uniform(0.85, 1.04)
            elif role[i] == "penny":
                lc[i], price[i] = rc.uniform(8.0, 8.6), rc.uniform(0.6, 4.6)
            elif role[i] == "illiquid":
                lc[i] = rc.uniform(8.45, 9.1)
            elif role[i] == "ipo":
                lc[i] = rc.uniform(9.0, 10.3)

        rt = self._rng("trading")
        sp = lambda key: np.array([_SECTORS[s][key] if s else np.nan for s in sector], float)  # noqa: E731
        beta = np.clip(np.nan_to_num(sp("beta"), nan=1.0) + rt.normal(0, 0.2, n), 0.25, 2.0)
        vol_mult = np.nan_to_num(sp("vol"), nan=1.0)
        idio = np.clip((0.42 - 0.07 * (lc - 9.0)) * vol_mult * np.exp(rt.normal(0, 0.15, n)), 0.12, 0.9)
        drift = rt.normal(0.03, 0.11, n)
        turnover = 0.008 * 10 ** (-0.334 * (lc - 9.0)) * np.exp(rt.normal(0, 0.35, n))
        nasdaq_p = np.array([0.65 if s in ("Information Technology", "Health Care", "Communication Services") else 0.25 for s in sector])
        exchange = np.where(rt.random(n) < nasdaq_p, "NASDAQ", "NYSE").astype(object)
        uo = rt.random(n)
        retail = np.array([s == "Consumer Discretionary" for s in sector])
        offset = np.where(uo < np.where(retail, 0.60, 0.85), 0, np.where(uo < np.where(retail, 0.90, 0.95), 1, 2))
        base_lag = rt.integers(27, 43, n)
        float_frac = rt.uniform(0.85, 0.98, n)
        opt_activity = rt.uniform(0.05, 0.35, n)
        iv_ratio = rt.uniform(1.02, 1.25, n)
        pcr_base = np.exp(rt.normal(math.log(0.75), 0.25, n))
        listing = np.zeros(n, dtype=np.int64)
        for i in range(n):
            r = role[i]
            if r == "etf":
                beta[i], idio[i], drift[i], turnover[i] = rt.uniform(0.9, 1.1), 0.04, 0.0, rt.uniform(0.005, 0.03)
            elif r == "preferred":
                beta[i], idio[i], drift[i], turnover[i] = 0.3, 0.08, 0.0, rt.uniform(0.0005, 0.002)
                opt_activity[i] = 0.0
            elif r == "penny":
                idio[i], turnover[i], opt_activity[i] = rt.uniform(0.8, 1.1), rt.uniform(0.004, 0.02), 0.0
            elif r == "illiquid":
                turnover[i], opt_activity[i] = rt.uniform(0.0002, 0.0008), 0.0
            elif r == "ipo" and N > 300:
                listing[i] = int(rt.integers(N // 3, N - 130))
        self._sector, self._industry, self._stype, self._country, self._role = sector, industry, stype, country, role
        self._lcap, self._price_end = lc, price
        self._beta, self._idio, self._drift, self._turnover = beta, idio, drift, turnover
        self._exchange, self._offset, self._base_lag = exchange, offset, base_lag
        self._float_frac, self._opt_activity, self._iv_ratio, self._pcr_base = float_frac, opt_activity, iv_ratio, pcr_base
        self._listing = listing
        self._gap_sd = np.clip(0.035 + 0.02 * (9.7 - lc), 0.025, 0.09)

    # ------------------------------------------------------------------ archetypes

    def _assign_archetypes(self) -> None:
        n = self.n_tickers
        self._arch = np.full(n, "normal", dtype=object)
        self._stories: dict[int, _Story] = {}
        self._scenarios: list[dict[str, Any]] = []
        if self._N < _MIN_SESSIONS_FOR_PLANTING:
            return
        ra = self._rng("archetypes")
        eligible = [i for i in range(n) if not self._role[i] and self._stype[i] == "common_stock" and self._country[i] == "US"]
        perm = [int(i) for i in ra.permutation(np.array(eligible, dtype=np.int64))]
        counts = {a: int(round(f * n)) for a, f in _ARCH_FRACTIONS.items()}
        n_cont = counts["sector_contagion"]
        n_scen = 0 if n_cont == 0 else (1 if n_cont < 6 else 2)
        for j in sorted(int(x) for x in ra.choice(len(_CONTAGION_SCENARIOS), size=n_scen, replace=False)):
            self._scenarios.append(dict(_CONTAGION_SCENARIOS[j]))
        busy = {s["industry"] for s in self._scenarios}
        settings = {
            "transitory_shock": [s for s in _TRANSITORY_SETTINGS if s[1] not in busy],
            "value_trap": [s for s in _VALUE_TRAP_SETTINGS if s[1] not in busy],
            "guidance_reset": [s for s in _GUIDANCE_SETTINGS if s[1] not in busy],
        }
        ptr = 0
        for arch in ("sector_contagion", "transitory_shock", "value_trap", "guidance_reset", "momentum_leader"):
            take = perm[ptr: ptr + counts[arch]]
            ptr += len(take)
            c = len(take)
            if arch in ("transitory_shock", "value_trap"):
                fit_n = c if c <= 2 else max(2, int(round(c * 0.34)))
            elif arch in ("guidance_reset", "sector_contagion"):
                fit_n = c if c <= 1 else max(1, int(round(c * 0.25)))
            else:
                fit_n = 0
            sett = settings.get(arch, [])
            sorder = [int(x) for x in ra.permutation(len(sett))] if sett else []
            for j, i in enumerate(take):
                st = _Story(archetype=arch, fit=j < fit_n)
                if arch == "sector_contagion":
                    st.scenario = j % n_scen
                    scen = self._scenarios[st.scenario]
                    self._sector[i], self._industry[i], st.theme = scen["sector"], scen["industry"], scen["key"]
                elif sett:
                    sec_name, ind_name, st.theme = sett[sorder[j % len(sett)]]
                    self._sector[i], self._industry[i] = sec_name, ind_name
                if arch != "momentum_leader" and not st.fit:
                    wts = _PLANT_MISS_WEIGHTS[arch]
                    keys = list(wts)
                    p = np.array([wts[k] for k in keys])
                    k_n = 1 if ra.random() < 0.7 else 2
                    st.misses = tuple(sorted(keys[int(x)] for x in ra.choice(len(keys), size=k_n, replace=False, p=p / p.sum())))
                self._draw_targets(i, st, ra)
                self._stories[i] = st
                self._arch[i] = arch

    def _draw_targets(self, i: int, st: _Story, ra: np.random.Generator) -> None:
        a, miss, T = st.archetype, set(st.misses), st.targets
        U = ra.uniform
        if a == "momentum_leader":
            self._lcap[i] = float(np.clip(ra.normal(10.0, 0.5), 9.0, 11.4))
            self._price_end[i] = math.exp(U(math.log(30), math.log(400)))
            T.update(growth=U(0.15, 0.30), si=U(0.008, 0.025), rev_rev=U(0.02, 0.06), eps_rev=U(0.03, 0.08),
                     surprise=U(0.05, 0.12))
        else:
            if "cap" in miss:
                self._lcap[i] = U(8.85, 9.25) if ra.random() < 0.5 else U(10.38, 10.9)
            else:
                self._lcap[i] = U(9.42, 10.2)
            self._price_end[i] = math.exp(U(math.log(22), math.log(160)))
            dd_rng = {"transitory_shock": (0.20, 0.32), "value_trap": (0.21, 0.34), "guidance_reset": (0.185, 0.24),
                      "sector_contagion": (0.19, 0.28)}[a]
            if "drawdown" in miss:
                T["drawdown"] = U(0.43, 0.52) if (a != "guidance_reset" and ra.random() < 0.5) else U(0.09, 0.125)
            else:
                T["drawdown"] = U(*dd_rng)
            fcf_rng = {"transitory_shock": (0.05, 0.09), "value_trap": (0.048, 0.08), "guidance_reset": (0.045, 0.07),
                       "sector_contagion": (0.046, 0.075)}[a]
            T["fcf_yield"] = U(0.015, 0.035) if "fcf" in miss else U(*fcf_rng)
            g_rng = {"transitory_shock": (0.11, 0.19), "value_trap": (0.092, 0.125), "guidance_reset": (0.11, 0.18),
                     "sector_contagion": (0.10, 0.17)}[a]
            T["growth"] = U(0.02, 0.065) if "growth" in miss else U(*g_rng)
            si_rng = {"transitory_shock": (0.08, 0.20), "value_trap": (0.09, 0.20), "guidance_reset": (0.065, 0.11),
                      "sector_contagion": (0.065, 0.12)}[a]
            T["si"] = U(0.015, 0.05) if "si" in miss else U(*si_rng)
            T["si_pre"] = U(0.012, 0.03) if "si" in miss else U(0.03, 0.055)
            rev_rng, eps_rng, sur_rng = {
                "transitory_shock": ((-0.08, -0.03), (-0.08, -0.035), (-0.035, 0.01)),
                "value_trap": ((-0.18, -0.10), (-0.30, -0.16), (-0.11, -0.04)),
                "guidance_reset": ((-0.09, -0.04), (-0.12, -0.05), (0.04, 0.10)),
                "sector_contagion": ((-0.02, 0.01), (-0.03, 0.01), (0.01, 0.05)),
            }[a]
            T.update(rev_rev=U(*rev_rng), eps_rev=U(*eps_rng), surprise=U(*sur_rng), iv_bump=U(0.10, 0.18))
            if a == "transitory_shock":
                T["oneoff_pts"] = U(0.06, 0.10) if "growth" not in miss else U(0.04, 0.06)
            elif a == "value_trap":
                T["last_q_growth"] = U(-0.06, 0.01)
            elif a == "guidance_reset":
                T["guide_cut"] = U(0.06, 0.11)
                T["buyback_pct"] = U(0.05, 0.10)
            elif a == "sector_contagion":
                T["exposure_pct"] = float(int(U(1, 5)))
        lc = self._lcap[i]
        self._turnover[i] = max(0.005, 0.008 * 10 ** (-0.334 * (lc - 9.0)))
        self._idio[i] = 0.2
        self._offset[i] = 0
        self._opt_activity[i] = max(self._opt_activity[i], 0.12)

    # ------------------------------------------------------------------ names / tickers

    def _build_identities(self) -> None:
        n = self.n_tickers
        rng = self._rng("identities")
        used_names: set[str] = set()
        used_tickers: set[str] = set(_TICKER_BLOCKLIST)
        names: list[str] = []
        tickers: list[str] = []

        def ticker_for(words: list[str], extra: str = "") -> str:
            w = "".join(ch for ch in words[0].upper() if ch.isalpha())
            s = "".join(ch for ch in (words[1] if len(words) > 1 else "X").upper() if ch.isalpha()) or "X"
            cons = w[0] + "".join(ch for ch in w[1:] if ch not in "AEIOU")
            cands = [w[:4], cons[:4], w[:3] + s[:1], cons[:3] + s[:1], w[:2] + s[:2], cons[:3], w[:3],
                     w[0] + s[:3], cons[:2] + s[:2], w[:5], cons[:4] + s[:1], w[:4] + s[:1]]
            for c in cands:
                c = (c + extra)[:5]
                if 3 <= len(c) <= 5 and c not in used_tickers:
                    used_tickers.add(c)
                    return c
            letters = np.array(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
            while True:
                k = int(rng.integers(3, 6))
                c = (w[0] + "".join(rng.choice(letters, size=k - 1)))[:5]
                if c not in used_tickers:
                    used_tickers.add(c)
                    return c

        for i in range(n):
            st = self._stype[i]
            if st == "etf":
                theme, code = _ETF_THEMES[int(rng.integers(len(_ETF_THEMES)))]
                sponsor = _ETF_SPONSORS[int(rng.integers(len(_ETF_SPONSORS)))]
                name = f"{sponsor} {theme} ETF"
                k = 2
                while name in used_names:
                    name, k = f"{sponsor} {theme} ETF {k}", k + 1
                tick = ticker_for([sponsor[0] + code, "ETF"], "")
            else:
                suffixes = _INDUSTRY_SUFFIX.get(self._industry[i], ("Holdings",))
                name = ""
                for _ in range(60):
                    cand = f"{_PREFIXES[int(rng.integers(len(_PREFIXES)))]} {suffixes[int(rng.integers(len(suffixes)))]}"
                    if cand not in used_names:
                        name = cand
                        break
                k = 2
                while not name or name in used_names:
                    base = f"{_PREFIXES[int(rng.integers(len(_PREFIXES)))]} {suffixes[0]}"
                    name, k = f"{base} Group {k}" if base in used_names else base, k + 1
                words = name.split()
                if st == "preferred":  # issuer's root symbol + "P"; the root itself stays reserved
                    cand = ticker_for(words)[:4] + "P"
                    if cand in used_tickers:
                        tick = ticker_for(words, "P")
                    else:
                        tick = cand
                        used_tickers.add(cand)
                    name = f"{name} {rng.uniform(5.0, 7.5):.2f}% Series {'ABCD'[int(rng.integers(4))]} Preferred"
                else:
                    tick = ticker_for(words)
                    if st == "adr":
                        form = {"GB": "plc", "IE": "plc", "NL": "N.V.", "DE": "AG", "CH": "AG", "JP": "K.K.", "FR": "S.A.",
                                "DK": "A/S", "SE": "AB", "IL": "Ltd."}.get(self._country[i], "Ltd.")
                        name = f"{name} {form} ADR"
            used_names.add(name)
            names.append(name)
            tickers.append(tick)
        self._tickers = tickers
        self._names = names
        self._idx = {t: i for i, t in enumerate(tickers)}
        # People: management and covering analysts.
        rp = self._rng("people")
        fn, ln = len(_FIRST_NAMES), len(_LAST_NAMES)
        person = lambda: f"{_FIRST_NAMES[int(rp.integers(fn))]} {_LAST_NAMES[int(rp.integers(ln))]}"  # noqa: E731
        self._mgmt = [(person(), person(), person()) for _ in range(n)]  # CEO, CFO, IR
        self._coverage = []
        for _ in range(n):
            firms = rp.choice(len(_BROKERS), size=8, replace=False)
            self._coverage.append([(person(), _BROKERS[int(f)]) for f in firms])

    # ------------------------------------------------------------------ earnings calendar

    def _build_calendar(self) -> None:
        n, N = self.n_tickers, self._N
        y0, y1 = self.start.year - 3, self.end.year + 1
        base = [(y, m) for y in range(y0, y1 + 1) for m in (3, 6, 9, 12)]
        K = len(base)
        pe = np.empty((n, K), dtype="datetime64[D]")
        months = np.empty((n, K), dtype=np.int64)
        for o in (0, 1, 2):
            rows = self._offset == o
            if rows.any():
                pe[rows] = np.array([_month_end(y, m + o) for y, m in base])
                months[rows] = np.array([(m + o - 1) % 12 + 1 for _, m in base])
        fq = ((months - self._offset[:, None] - 1) % 12) // 3 + 1
        rng = self._rng("calendar")
        lag = np.clip(self._base_lag[:, None] + rng.integers(-3, 4, size=(n, K)) + 3 * (fq == 4), 25, 45)
        rd = pe + lag.astype("timedelta64[D]")
        fwd = np.busday_offset(rd, 0, roll="forward")
        rd = np.where((fwd - pe).astype(np.int64) <= 45, fwd, np.busday_offset(rd, 0, roll="backward"))
        self._K, self._pe, self._fq = K, pe, fq
        self._surprise = np.clip(rng.normal(0.025, 0.05, size=(n, K)), -0.4, 0.4)
        t_end = N - 1
        for sc in self._scenarios:
            sc["headline_session"] = t_end - int(rng.integers(48, 57))
            sc["followup_session"] = t_end - int(rng.integers(9, 16))
        for i in sorted(self._stories):
            st = self._stories[i]
            if st.archetype == "momentum_leader":
                continue
            placed = None
            hi_lo = [(t_end - 44, t_end - 26), (t_end - 60, t_end - 22)] if st.archetype == "sector_contagion" else \
                [(t_end - 46, t_end - 30), (t_end - 70, t_end - 20)]
            for lo, hi in hi_lo:
                lo = max(lo, 1)
                for k in range(K - 1, -1, -1):
                    if pe[i, k] + 45 < self._dd[lo]:
                        break
                    win = (self._dd >= pe[i, k] + 25) & (self._dd <= pe[i, k] + 45)
                    win[:lo] = False
                    win[hi + 1:] = False
                    cand = np.flatnonzero(win)
                    if cand.size:
                        pick = int(cand[int(rng.integers(cand.size))])
                        rd[i, k] = self._dd[pick]
                        placed = (k, pick)
                        break
                if placed:
                    break
            if placed is None:  # degenerate calendars: fall back to the natural latest report
                ks = np.flatnonzero((rd[i] <= self._dd[max(0, t_end - 20)]) & (rd[i] >= self._dd[1]))
                if ks.size == 0:
                    self._arch[i] = "normal"
                    del self._stories[i]
                    continue
                k = int(ks[-1])
                placed = (k, int(np.searchsorted(self._dd, rd[i, k])))
            st.story_k, st.report_session = placed
            if st.archetype == "sector_contagion":
                st.shock_session = min(self._scenarios[st.scenario]["headline_session"], st.report_session - 3)
                st.cap_session = self._scenarios[st.scenario]["followup_session"]
            else:
                st.shock_session = st.report_session
                if "volume" in st.misses:
                    st.cap_session = min(st.report_session + int(rng.integers(2, 6)), t_end - 21)
                elif "rsi" in st.misses:
                    st.cap_session = t_end - int(rng.integers(13, 18))
                else:
                    st.cap_session = t_end - int(rng.integers(3, 16))
                if st.cap_session <= st.report_session:
                    st.cap_session = -1
            st.params["broker_idx"] = int(rng.integers(8))
        listing_d = self._dd[np.clip(self._listing, 0, N - 1)]
        avail = np.where(self._listing[:, None] > 0, np.maximum(rd, (listing_d - 14)[:, None]), rd)
        no_reports = np.isin(self._stype, ["etf", "preferred"])
        far = np.datetime64("2262-01-01")
        avail[no_reports] = far
        self._rd, self._avail = rd, avail
        pos = np.searchsorted(self._dd, rd)
        inside = (pos < N) & (self._dd[np.minimum(pos, N - 1)] == rd) & ~no_reports[:, None]
        self._rs = np.where(inside, pos, -1)
        self._k_end = (avail <= self._end_d).sum(axis=1) - 1
        for i, st in self._stories.items():
            if st.archetype == "momentum_leader":
                st.story_k = int(self._k_end[i])
                st.report_session = int(self._rs[i, st.story_k]) if st.story_k >= 0 else -1
                st.shock_session = st.report_session
            elif st.story_k != self._k_end[i]:  # pragma: no cover - placement guarantees this
                raise AssertionError("story quarter must be the latest reported quarter")

    # ------------------------------------------------------------------ prices

    def _simulate_prices(self) -> None:
        n, N, K = self.n_tickers, self._N, self._K
        rm = self._rng("market")
        target = 0.15 / math.sqrt(252)
        a_, b_ = 0.08, 0.90
        omega = target**2 * (1 - a_ - b_)
        z = rm.standard_t(5, N) * math.sqrt(3 / 5)
        h = np.empty(N)
        mret = np.empty(N)
        mu = 0.075 / 252
        h[0] = target**2
        for t in range(N):
            if t:
                h[t] = omega + a_ * (mret[t - 1] - mu) ** 2 + b_ * h[t - 1]
            mret[t] = mu + math.sqrt(h[t]) * z[t]
        if N > 200:  # one risk-off episode, well before the planted stories
            e0 = int(rm.integers(30, max(31, N - 160)))
            ln_ = int(rm.integers(6, 12))
            mret[e0: e0 + ln_] -= rm.uniform(0.07, 0.12) / ln_
        vr = np.sqrt(h) / target
        self._mret = mret

        rs_ = self._rng("sector")
        sig_s = np.array([0.012 if s == "Energy" else 0.006 if s == "Utilities" else 0.008 for s in _SECTOR_ORDER])
        S = np.hstack([rs_.standard_normal((N, len(_SECTOR_ORDER))) * sig_s * vr[:, None], np.zeros((N, 1))])
        ri = self._rng("industry")
        I = np.hstack([ri.standard_normal((N, len(_INDUSTRY_ORDER))) * 0.0045 * vr[:, None], np.zeros((N, 1))])
        for sc in self._scenarios:
            col = _INDUSTRY_ORDER.index(sc["industry"])
            hs, fs = sc["headline_session"], sc["followup_session"]
            I[hs, col] += math.log(1 - ri.uniform(0.06, 0.09))
            I[hs + 1: hs + 11, col] -= ri.uniform(0.002, 0.004)
            I[fs, col] += math.log(1 - ri.uniform(0.025, 0.04))
        sec_idx = np.array([_SECTOR_ORDER.index(s) if s else len(_SECTOR_ORDER) for s in self._sector])
        ind_idx = np.array([_INDUSTRY_ORDER.index(x) if x else len(_INDUSTRY_ORDER) for x in self._industry])
        factor = self._beta[None, :] * mret[:, None] + S[:, sec_idx] + I[:, ind_idx]
        sd_idio = self._idio / math.sqrt(252)
        sdd = np.sqrt((self._beta * target) ** 2 + 0.008**2 + 0.0045**2 + sd_idio**2)

        re = self._rng("idio")
        R = factor + re.standard_t(4, (N, n)) * math.sqrt(0.5) * sd_idio + self._drift / 252
        ev = np.zeros((N, n), dtype=bool)
        vb = np.zeros((N, n))
        rg = self._rng("events")
        cols = np.arange(n)
        for k in range(K):
            s = self._rs[:, k]
            ok = s >= 1
            if not ok.any():
                continue
            gap = 0.45 * self._surprise[:, k] + rg.normal(0, 1, n) * self._gap_sd
            mult = rg.uniform(2.0, 3.5, n)
            R[s[ok], cols[ok]] += gap[ok]
            ev[s[ok], cols[ok]] = True
            vb[s[ok], cols[ok]] += np.log(mult[ok])
            for d, m in ((1, 1.5), (2, 1.2)):
                ok2 = ok & (s + d < N)
                vb[s[ok2] + d, cols[ok2]] += math.log(m)
        n_shocks = rg.poisson(0.7 * N / 717, n)
        for i in range(n):
            if self._stype[i] in ("etf", "preferred"):
                continue
            for _ in range(int(n_shocks[i])):
                t = int(rg.integers(1, N))
                R[t, i] += rg.normal(0, 0.07) * self._idio[i] / 0.3
                ev[t, i] = True
                vb[t, i] += math.log(rg.uniform(2.5, 5.0))
        for sc in self._scenarios:  # the whole industry trades heavier on the headlines
            members = np.flatnonzero(ind_idx == _INDUSTRY_ORDER.index(sc["industry"]))
            vb[sc["headline_session"], members] += np.log(rg.uniform(1.8, 2.8, members.size))
            vb[sc["followup_session"], members] += np.log(rg.uniform(1.5, 2.2, members.size))
        R[0] = 0.0

        rv = self._rng("volume")
        eps = rv.normal(0, 0.15, (N, n))
        ar = np.empty((N, n))
        ar[0] = eps[0] * 1.9
        for t in range(1, N):
            ar[t] = 0.85 * ar[t - 1] + eps[t]
        shares0 = 10**self._lcap / self._price_end
        base_lv = np.log(np.maximum(self._turnover * shares0, 100.0))
        abn = np.minimum(np.abs(R - factor) / sd_idio[None, :], 8.0)
        LV = base_lv[None, :] + ar + rv.normal(0, 0.22, (N, n)) + vb + 0.08 * abn

        ro = self._rng("ohlc")
        ovn = ro.normal(0, 1, (N, n)) * 0.25 * sdd
        hu = np.abs(ro.normal(0, 1, (N, n))) * 0.5 * sdd
        hd = np.abs(ro.normal(0, 1, (N, n))) * 0.5 * sdd

        if self._stories:
            self._plant_prices(factor, R, LV, ev, base_lv, ovn, hu, hd)
        self._long_window_returns(R)

        lc, lo, lh, ll = _ohlc_log(R, ev, ovn, hu, hd)
        shift = np.log(self._price_end)[None, :] - lc[-1:, :]
        px = {k: np.maximum(np.round(np.exp(v + shift), 4), _MIN_PRICE)
              for k, v in (("open", lo), ("high", lh), ("low", ll), ("close", lc))}
        vol = np.maximum(np.round(np.exp(LV)), 100.0)
        pre = np.arange(N)[:, None] < self._listing[None, :]
        for k in px:
            px[k][pre] = np.nan
        vol[pre] = np.nan
        self._price_end = px["close"][-1].copy()
        idx = self._dates.copy()
        idx.name = "date"
        cols_ix = pd.Index(self._tickers, name="ticker")
        self._px = {k: pd.DataFrame(v, index=idx, columns=cols_ix) for k, v in px.items()}
        self._px["volume"] = pd.DataFrame(vol, index=idx, columns=cols_ix)
        self._close = px["close"]
        self._logret = np.vstack([np.full((1, n), np.nan), np.diff(np.log(px["close"]), axis=0)])

    def _long_window_returns(self, R: np.ndarray) -> None:
        """Long windows only: realistic multi-year return structure before the last three years.

        1. Every stock carries a persistent drift (``N(3%, 11%)`` a year). Over the default ~3-year
           window that is part of the design (cross-sectional trends, momentum), but kept for 15
           years it drives names apart by +-1.6 in log terms. Before the last ``_FREE_SESSIONS``
           sessions each stock's drift is therefore redrawn every ``_DRIFT_REGIME_SESSIONS`` sessions
           from the same distribution (same short-horizon dispersion, no 15-year persistence).
        2. The cross-section is anchored at the end of the window, so earlier market caps depend on
           later returns and the cap-weighted market trails its constituents (by roughly the
           cross-sectional return variance). A uniform daily shift of every stock's return - which
           leaves those cap weights unchanged - is calibrated so that over this period the
           cap-weighted universe (the ``SYNTH-US`` construction) earns exactly the simulated market
           factor; individual stocks keep their betas, factor exposures and idiosyncratic paths.

        Both are phased in over a year before the last ``_FREE_SESSIONS`` sessions; the recent
        sessions (where the planted stories live, and all of a default-length window) are untouched.
        """
        N = self._N
        t0 = N - _FREE_SESSIONS
        if t0 <= 1:
            return
        stocks = np.flatnonzero(~np.isin(self._stype, ["etf", "preferred"]))
        if stocks.size == 0:
            return
        t = np.arange(N)
        v = np.clip((t0 - t) / 252.0, 0.0, 1.0)  # phase-in weight (0 from t0 on)
        v[0] = 0.0
        rng = self._rng("drift-regimes")
        n_reg = (t0 - 1) // _DRIFT_REGIME_SESSIONS + 1
        draws = rng.normal(0.03, 0.11, (n_reg, self.n_tickers))  # same law as the reference drift
        reg = np.clip((t0 - 1 - t) // _DRIFT_REGIME_SESSIONS, 0, n_reg - 1)  # regime 0 ends at t0
        R[:, stocks] += (draws[reg][:, stocks] - self._drift[stocks][None, :]) * (v / 252)[:, None]

        # cap weights at t-1 ~ end cap x exp(log price(t-1) - log price(end)); unlisted names excluded
        Rs = R[:, stocks]
        lc = np.cumsum(Rs, axis=0)
        logw = (self._lcap[stocks] * math.log(10.0))[None, :] + lc - lc[-1:]
        listed = t[:, None] >= self._listing[stocks][None, :]
        zone = np.flatnonzero(v > 0)
        live = listed[zone - 1] & listed[zone]
        zone, live = zone[live.any(axis=1)], live[live.any(axis=1)]
        if zone.size == 0:
            return
        a = np.where(live, logw[zone - 1], -np.inf)
        wts = np.exp(a - a.max(axis=1, keepdims=True))
        idx_r = np.log((wts * np.exp(Rs[zone])).sum(axis=1) / wts.sum(axis=1))
        delta = float((v[zone] * (self._mret[zone] - idx_r)).sum() / v[zone].sum())
        R[:, stocks] += (delta * v)[:, None]

    def _plant_prices(self, factor, R, LV, ev, base_lv, ovn, hu, hd) -> None:
        """Rejection-sample the idiosyncratic path of every planted name until its technical
        profile matches the design (fit names inside the canonical thresholds, near misses outside
        exactly the conditions they are meant to miss). Deterministic per (seed, ticker, round)."""
        N = self._N
        pending = sorted(self._stories)
        best: dict[int, tuple[int, np.ndarray, np.ndarray, np.ndarray]] = {}
        for rnd in range(_MAX_PLANT_ROUNDS):
            if not pending:
                break
            m = len(pending)
            Rm, VBm, EPS, IID = (np.empty((N, m)) for _ in range(4))
            EVm = np.zeros((N, m), dtype=bool)
            shocks: list[tuple[int, float, int, float] | None] = []
            for j, i in enumerate(pending):
                rng = self._rng("plant", i, rnd)
                idio, vbj, evj, meta = self._plant_attempt(i, self._stories[i], rng, factor[:, i])
                Rm[:, j] = factor[:, i] + idio
                VBm[:, j], EVm[:, j] = vbj, evj
                EPS[:, j] = rng.normal(0, 0.15, N)
                IID[:, j] = rng.normal(0, 0.18, N)
                shocks.append(meta)
            Rm[0] = 0.0
            cols = np.array(pending)
            _, _, lh, _ = _ohlc_log(Rm, EVm, ovn[:, cols], hu[:, cols], hd[:, cols])
            lc = np.cumsum(Rm, axis=0)
            for j, meta in enumerate(shocks):  # hit the drawdown target by adjusting the drift between
                if meta is None:  # the shock and the final leg (the final leg itself is left intact)
                    continue
                s, D, fix_end, _ = meta
                w0 = max(0, N - 252)
                if s - 1 < w0 or s >= N - 1:
                    continue
                peak = lh[w0:s, j].max()
                Rm[s + 1: fix_end, j] += (peak + math.log(1 - D) - lc[-1, j]) / (fix_end - s - 1)
            lc, _, lh, _ = _ohlc_log(Rm, EVm, ovn[:, cols], hu[:, cols], hd[:, cols])
            ar = np.empty((N, m))
            ar[0] = EPS[0] * 1.9
            for t in range(1, N):
                ar[t] = 0.85 * ar[t - 1] + EPS[t]
            LVm = base_lv[cols][None, :] + ar + IID + VBm
            for j, meta in enumerate(shocks):  # the shock session always trades a multiple of normal volume
                if meta is not None and meta[0] >= 1:
                    s, spike = meta[0], meta[3]
                    LVm[s, j] = max(LVm[s, j], math.log(spike * np.exp(LVm[max(0, s - 120): s, j]).mean()))
            C = np.exp(lc - lc[-1])
            H = np.exp(lh - lc[-1])
            V = np.exp(LVm)
            snap = dict(
                trend=C[-50:].mean(0) / C[-200:].mean(0) - 1,
                mom=C[-22] / C[-253] - 1,
                dd=C[-1] / H[-252:].max(0) - 1,
                mvr=V[-20:].max(0) / V[-120:].mean(0),
                rsi=_wilder_rsi_last(C),
            )
            still = []
            for j, i in enumerate(pending):
                ok, score = self._plant_check(self._stories[i], {k: float(v[j]) for k, v in snap.items()})
                if i not in best or score > best[i][0]:
                    best[i] = (score, Rm[:, j].copy(), LVm[:, j].copy(), EVm[:, j].copy())
                if ok:
                    best[i] = (10**6, Rm[:, j].copy(), LVm[:, j].copy(), EVm[:, j].copy())
                else:
                    still.append(i)
            pending = still
        for i, (score, r, lv, e) in best.items():
            R[:, i], LV[:, i], ev[:, i] = r, lv, e
            self._stories[i].params["calibrated"] = score == 10**6

    def _plant_attempt(self, i: int, st: _Story, rng: np.random.Generator, fac: np.ndarray):
        """Idiosyncratic log returns, volume bumps and event mask for one planted path.

        The design is in total-return space: within each phase (uptrend, post-shock drift, final leg)
        the factor's average drift is hedged out on non-event days, so the stock's realised trend in
        that phase is the designed one while it keeps its day-to-day co-movement with the market.
        """
        N = self._N
        T, miss, a = st.targets, set(st.misses), st.archetype
        inc = np.zeros(N)
        vb = np.zeros(N)
        ev = np.zeros(N, dtype=bool)
        noise = rng.standard_normal(N) * rng.uniform(0.15, 0.22) / math.sqrt(252)
        for k in range(self._K):  # earlier quarters: compounders beat and gap up
            s_k = int(self._rs[i, k])
            if s_k < 1 or k > st.story_k or (k == st.story_k and a != "momentum_leader"):
                continue
            inc[s_k] += rng.uniform(0.01, 0.05)
            vb[s_k] += math.log(rng.uniform(1.8, 3.0))
            ev[s_k] = True

        def hedge(lo: int, hi: int) -> None:
            w = np.arange(max(lo, 1), min(hi, N))
            w = w[~ev[w]]
            if w.size > 1:
                inc[w] -= fac[w].mean()

        if a == "momentum_leader":
            inc += math.log1p(rng.uniform(0.35, 0.7)) / 252
            inc[-25:] += 0.001
            hedge(1, N - 25)
            hedge(N - 25, N)
            return inc + 0.85 * noise, vb, ev, None
        s, c, D = st.shock_session, st.cap_session, T["drawdown"]
        up0 = max(0, s - int(rng.integers(260, 381)))
        up = rng.uniform(0.55, 0.95) if "trend" not in miss else rng.uniform(0.05, 0.25)
        inc[:up0] += rng.uniform(-0.03, 0.08) / 252
        inc[up0:s] += math.log1p(up) / max(1, s - up0)
        if a == "sector_contagion":
            inc[s] += math.log(1 - rng.uniform(0.04, 0.07))
            vb[s] += math.log(rng.uniform(3.0, 5.0))
            ev[s] = True
            r_s = st.report_session
            if s < r_s < N:
                inc[r_s] += rng.uniform(0.0, 0.025)
                vb[r_s] += math.log(rng.uniform(2.2, 3.2))
                ev[r_s] = True
        else:
            frac = rng.uniform(0.55, 0.70) if a == "guidance_reset" else rng.uniform(0.45, 0.62)
            inc[s] += math.log(1 - D * frac) - fac[s]  # company-specific news dominates the session
            vb[s] += math.log(rng.uniform(4.0, 8.0))
            ev[s] = True
        tt = np.arange(s + 1, N)
        vb[s + 1:] += np.log1p(rng.uniform(0.7, 1.2) * np.exp(-(tt - s) / rng.uniform(6, 11)))
        if "rsi" in miss:  # sold off, then rebounded: oversold reading has already unwound
            lr = int(rng.integers(9, 14))
            inc[N - lr:] += math.log1p(rng.uniform(0.08, 0.12)) / lr
            pre_lo = max(s + 1, N - lr - 8)
            if pre_lo < N - lr:
                inc[pre_lo: N - lr] += math.log(1 - rng.uniform(0.03, 0.05)) / (N - lr - pre_lo)
            fix_end = pre_lo
        else:  # second leg lower into the as-of date keeps RSI depressed
            l2 = int(rng.integers(11, 17))
            inc[N - l2:] += math.log(1 - rng.uniform(0.06, 0.10)) / l2
            noise[N - l2:] *= 0.6
            fix_end = N - l2
        if s < c < N:
            inc[c] += math.log(1 - rng.uniform(0.02, 0.04))
            ev[c] = True
            if "volume" in miss:
                vb[c] += math.log(rng.uniform(1.2, 1.5))
            else:
                vb[c] += math.log(rng.uniform(3.4, 4.8))
                if c + 1 < N:
                    vb[c + 1] += math.log(1.5)
            noise[c] *= 0.3
        noise[s] *= 0.3
        fix_end = max(fix_end, s + 2)
        hedge(up0, s)
        hedge(s + 1, fix_end)
        if "rsi" in miss:
            hedge(fix_end, N - lr)
            hedge(N - lr, N)
        else:
            hedge(fix_end, N)
        spike = rng.uniform(2.6, 4.0) if a == "sector_contagion" else rng.uniform(3.3, 6.0)
        return inc + noise, vb, ev, (s, D, fix_end, spike)

    @staticmethod
    def _plant_check(st: _Story, f: dict[str, float]) -> tuple[bool, int]:
        miss = set(st.misses)
        if st.archetype == "momentum_leader":
            conds = [f["dd"] > -0.08, f["rsi"] > 52, f["trend"] > 0.02, f["mom"] > 0.10]
        else:
            conds = [f["trend"] < -0.003] if "trend" in miss else [f["trend"] > 0.012, f["mom"] > 0.04]
            if "drawdown" in miss:
                conds.append(f["dd"] < -0.42 if st.targets["drawdown"] > 0.3 else f["dd"] > -0.135)
            else:
                conds.append(-0.37 <= f["dd"] <= -0.175)
            if "rsi" in miss:
                conds.append(f["rsi"] >= 45)
            elif "drawdown" not in miss:  # a shallow / extreme drawdown need not also be oversold
                conds.append(f["rsi"] <= 36.5)
            conds.append(f["mvr"] < 1.8 if "volume" in miss else f["mvr"] >= 2.35)
        return all(conds), int(sum(conds))

    # ------------------------------------------------------------------ fundamentals

    def _build_fundamentals(self) -> None:
        n, K = self.n_tickers, self._K
        rng = self._rng("fundamentals")
        sp = lambda key, default: np.array([_SECTORS[s][key] if s else default for s in self._sector], float)  # noqa: E731
        G = np.clip(sp("g", 0.05) + rng.normal(0, 1, n) * sp("gsd", 0.05), -0.15, 0.6)
        om = sp("om", 0.15) + rng.normal(0, 0.05, n)
        loss = np.isin(self._sector, ["Information Technology", "Health Care", "Communication Services", "Consumer Discretionary"]) \
            & (rng.random(n) < 0.12)
        om = np.where(loss, -rng.uniform(0.05, 0.45, n), om)
        gm = np.clip(sp("gm", 0.4) + rng.normal(0, 0.07, n), np.maximum(om + 0.08, 0.10), 0.95)
        da = sp("da", 0.04) * rng.uniform(0.7, 1.3, n)
        capex = sp("capex", 0.04) * rng.uniform(0.6, 1.4, n)
        sbc = sp("sbc", 0.01) * rng.uniform(0.5, 1.5, n)
        lev = sp("lev", 0.4) * np.exp(rng.normal(0, 0.5, n))
        lev[rng.random(n) < 0.15] = 0.0
        cash_r = rng.uniform(0.05, 0.40, n)
        eq_r = rng.uniform(0.3, 1.6, n)
        neg = rng.random(n) < 0.03
        eq_r[neg] = -rng.uniform(0.1, 0.4, int(neg.sum()))
        rate = rng.uniform(0.04, 0.075, n)
        chg = rng.normal(-0.004, 0.006, n)
        wc_sd = np.full(n, 0.015)
        amp = np.where(np.isin(self._sector, ["Consumer Discretionary", "Consumer Staples"]),
                       rng.uniform(0.05, 0.15, n), rng.uniform(0.0, 0.05, n))
        season = 1 + amp[:, None] * np.array([-0.6, -0.1, 0.0, 0.7])[None, :]
        innov = rng.normal(0, 1, (n, K)) * (0.02 + 0.02 * (self._lcap < 9.3))[:, None]
        e = np.zeros((n, K))
        for k in range(1, K):
            e[:, k] = 0.6 * e[:, k - 1] + innov[:, k]
        g = np.clip(G[:, None] + e, -0.4, 1.0)
        gm_q = gm[:, None] + rng.normal(0, 0.004, (n, K))
        om_q = om[:, None] + rng.normal(0, 0.006, (n, K)) + 0.3 * (g - G[:, None])
        da_q = da[:, None] * rng.uniform(0.95, 1.05, (n, K))
        capex_q = capex[:, None] * rng.uniform(0.85, 1.15, (n, K))
        wc_z = rng.normal(0, 1, (n, K))
        debt_z = rng.normal(0, 0.03, (n, K))
        cash_z = rng.normal(0, 0.08, (n, K))
        ps = np.clip(sp("ps", 2.0) * np.exp(rng.normal(0, 0.35, n) + 1.5 * (G - sp("g", 0.05))) * np.where(loss, 1.8, 1.0), 0.2, 40.0)
        ka_all = self._k_end
        solve: dict[int, tuple[float, float, np.ndarray]] = {}
        for i in sorted(self._stories):
            st, ka = self._stories[i], int(ka_all[i])
            r = self._rng("fund-plant", i)
            a, T = st.archetype, st.targets
            capex[i], da[i], sbc[i], wc_sd[i] = r.uniform(0.025, 0.045), r.uniform(0.03, 0.045), r.uniform(0.01, 0.02), 0.005
            lev[i], cash_r[i], eq_r[i], chg[i] = r.uniform(0.1, 0.6), r.uniform(0.1, 0.3), r.uniform(0.5, 1.2), r.uniform(-0.006, 0.0)
            da_q[i] = da[i] * r.uniform(0.97, 1.03, K)
            capex_q[i] = capex[i] * r.uniform(0.92, 1.08, K)
            if a == "momentum_leader":
                G[i] = T["growth"]
                g[i] = G[i] + r.normal(0, 0.01, K)
                if ka >= 3:
                    g[i, ka - 3: ka + 1] += np.array([0.0, 0.01, 0.02, 0.03])
                om[i], gm[i] = r.uniform(0.18, 0.30), r.uniform(0.50, 0.70)
                drift_m = 0.002 * (np.arange(K) - ka)
                gm_q[i] = gm[i] + r.normal(0, 0.003, K) + drift_m
                om_q[i] = om[i] + r.normal(0, 0.003, K) + drift_m
                ps[i] = r.uniform(6.0, 12.0)
                continue
            under = T["growth"] + r.uniform(0.01, 0.02)
            G[i] = under
            g[i] = under + r.normal(0, 0.008, K)
            gm[i], om[i] = r.uniform(0.40, 0.60), r.uniform(0.17, 0.26)
            gm_q[i] = gm[i] + r.normal(0, 0.003, K)
            om_q[i] = om[i] + r.normal(0, 0.004, K)
            shape = np.zeros(3)
            if a == "transitory_shock":
                last = under - T["oneoff_pts"]
                gm_q[i, ka] -= 0.008
                om_q[i, ka] -= 0.015
            elif a == "value_trap":
                last = T["last_q_growth"]
                shape = np.array([0.05, 0.025, 0.0])
                gm_q[i, ka - 1] -= 0.008
                om_q[i, ka - 1] -= 0.012
                gm_q[i, ka] -= r.uniform(0.025, 0.04)
                om_q[i, ka] -= r.uniform(0.025, 0.045)
                lev[i] = r.uniform(0.4, 0.8)
            elif a == "guidance_reset":
                last = under + 0.015
                cash_r[i], lev[i] = r.uniform(0.45, 0.80), r.uniform(0.0, 0.12)
                gm_q[i, ka] += 0.004
            else:
                last = under + r.normal(0, 0.005)
            solve[i] = (T["growth"], last, shape)

        rows = np.arange(n)
        rev = np.empty((n, K))
        rev[:, :4] = season[rows[:, None], self._fq[:, :4] - 1] * (1 + G[:, None]) ** (np.arange(4)[None, :] / 4)
        for k in range(4, K):
            rev[:, k] = rev[:, k - 4] * (1 + g[:, k])
        for i, (target, last, shape) in solve.items():
            ka = int(ka_all[i])
            if ka < 7:
                continue
            w = rev[i, ka - 7: ka - 3]
            c = ((1 + target) * w.sum() - w[3] * (1 + last) - (w[:3] * (1 + shape)).sum()) / w[:3].sum()
            g[i, ka - 3: ka] = shape + c
            g[i, ka] = last
            for k in range(ka - 3, K):
                rev[i, k] = rev[i, k - 4] * (1 + g[i, k])

        def ttm_rel(x: np.ndarray) -> np.ndarray:
            out = np.empty_like(x)
            out[:, :3] = 4 * x[:, :3]
            out[:, 3:] = x[:, 3:] + x[:, 2:-1] + x[:, 1:-2] + x[:, :-3]
            return out

        adj = self._valuation_anchor(np.log(np.maximum(ttm_rel(rev), 1e-12)))
        if adj is not None:  # long windows only (see _valuation_anchor); the recent quarters keep their numbers
            rev = rev * np.exp(adj)

        gp = gm_q * rev
        oi = om_q * rev
        da_v = da_q * rev
        capex_v = capex_q * rev
        rev_ttm = ttm_rel(rev)
        debt = np.maximum(lev[:, None] * rev_ttm * (1 + debt_z), 0.0)
        cash = np.maximum(cash_r[:, None] * rev_ttm * (1 + cash_z), 0.01 * rev)
        interest = debt * rate[:, None] / 4
        pretax = oi - interest
        ni = pretax - 0.21 * np.maximum(pretax, 0.0)
        cfo = ni + da_v + sbc[:, None] * rev + wc_sd[:, None] * wc_z * rev
        equity = eq_r[:, None] * rev_ttm
        # Total assets = equity + debt + other liabilities (payables, deferred revenue, deposits for
        # banks / insurers), floored at cash + a minimum operating-asset base of 0.6x TTM revenue
        # (negative-equity names carry large liabilities, not a tiny balance sheet). Separate RNG
        # stream: adding it leaves every other simulated number unchanged.
        r_as = self._rng("assets")
        fin = np.array([s == "Financials" for s in self._sector])
        ol_r = np.where(fin, r_as.uniform(3.0, 8.0, n), r_as.uniform(0.15, 0.60, n))
        other_liab = ol_r[:, None] * rev_ttm * (1 + r_as.normal(0, 0.04, (n, K)))
        assets = np.maximum(equity + debt + other_liab, cash + 0.6 * rev_ttm)

        shares_a = np.maximum(np.round(10**self._lcap / self._price_end), 1000.0)
        cap_end = shares_a * self._price_end
        ka_c = np.clip(ka_all, 3, K - 1)
        rev_ttm_a = rev_ttm[rows, ka_c]
        fcf_ttm_a = ttm_rel(cfo - capex_v)[rows, ka_c]
        scale = cap_end / ps / rev_ttm_a
        for i, st in self._stories.items():
            if st.archetype != "momentum_leader" and fcf_ttm_a[i] > 0:
                scale[i] = st.targets["fcf_yield"] * cap_end[i] / fcf_ttm_a[i]
        sc = scale[:, None]
        self._q = {
            "rev": rev * sc, "gp": gp * sc, "oi": oi * sc, "da": da_v * sc, "ni": ni * sc, "cfo": cfo * sc,
            "capex": capex_v * sc, "debt": debt * sc, "cash": cash * sc, "interest": interest * sc, "equity": equity * sc,
            "assets": assets * sc,
        }
        self._q["ebitda"] = self._q["oi"] + self._q["da"]
        self._q["shares"] = np.round(shares_a[:, None] * (1 + chg[:, None]) ** (np.arange(K)[None, :] - ka_all[:, None]))
        self._g, self._G = g, G

        # Shares outstanding / market cap per session (as of the latest public report).
        N = self._N
        kk = np.full((N, n), -1, dtype=np.int64)
        sess = np.arange(N)[:, None]
        for k in range(K):
            first = np.searchsorted(self._dd, self._avail[:, k], side="left")
            kk = np.where(sess >= first[None, :], k, kk)
        self._kk_t = kk
        shares_t = np.take_along_axis(self._q["shares"].T, np.clip(kk, 0, None), axis=0)
        self._cap_t = self._close * shares_t

    def _valuation_anchor(self, log_rev_ttm: np.ndarray) -> np.ndarray | None:
        """Log-level revenue adjustment (n, K) that keeps valuations sane over long windows (None: no-op).

        Prices are simulated independently of the revenue path, so over many years price / sales
        random-walks without bound (a 15-year window would show FCF yields of thousands of %).
        Up to the start of the free zone (the latest ``_ANCHOR_FREE_QUARTERS`` reported quarters),
        fundamentals follow prices: each name's log price / TTM-sales gap relative to its value at the
        start of the free zone is absorbed into revenue with a causal EMA (half-life
        ``_ANCHOR_HALF_LIFE_Q`` quarters) - fully for the part relative to the cross-sectional median
        gap, and ``_ANCHOR_MARKET_SHARE`` of the market-wide median gap (the rest stays in prices, so
        aggregate multiples can still re-rate, but not without bound). Relative multiples
        therefore mean-revert through fundamentals catching up with prices: revenue growth follows
        past returns, returns stay unpredictable from valuations. From the free zone on the
        adjustment is held constant, and since the end-of-window scaling normalises any constant
        factor, the free zone (the planted stories, and every quarter of a default-length window,
        where this is a no-op) keeps its designed numbers and revenue growth stays continuous.
        Every income / cash-flow / balance-sheet line scales with revenue, so the whole statement
        moves together. Quarters before the price window keep their relative path.
        """
        n, K = log_rev_ttm.shape
        kf = self._k_end - _ANCHOR_FREE_QUARTERS
        if not (kf >= 1).any():
            return None
        # mean log close over each fiscal quarter's sessions (pe[k-1], pe[k]] inside the window
        rows = np.arange(n)[:, None]
        hi = np.searchsorted(self._dd, self._pe.ravel(), side="right").reshape(n, K)
        lo = np.concatenate([np.zeros((n, 1), dtype=hi.dtype), hi[:, :-1]], axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            logc = np.log(self._close)
        ok = np.isfinite(logc)
        cs = np.vstack([np.zeros((1, n)), np.cumsum(np.where(ok, logc, 0.0), axis=0)])
        cn = np.vstack([np.zeros((1, n)), np.cumsum(ok, axis=0)])
        cnt = cn[hi, rows] - cn[lo, rows]
        with np.errstate(divide="ignore", invalid="ignore"):
            p = np.where(cnt >= 20, (cs[hi, rows] - cs[lo, rows]) / np.maximum(cnt, 1), np.nan)
        y = p - log_rev_ttm  # log price / sales (up to a per-name constant)
        y_ref = y[np.arange(n), np.clip(kf, 0, K - 1)]
        names = (kf >= 1) & np.isfinite(y_ref) & ~np.isin(self._stype, ["etf", "preferred"])
        live = names[:, None] & (np.arange(K)[None, :] <= kf[:, None]) & np.isfinite(y)
        if not live.sum(axis=1).max(initial=0) > 1:
            return None
        z = np.where(live, y - y_ref[:, None], np.nan)
        common = np.zeros(K)  # cross-sectional median gap per quarter (left in prices)
        cols = live.sum(axis=0) >= 5
        if cols.any():
            common[cols] = np.nanmedian(z[:, cols], axis=0)
        z = z - (1.0 - _ANCHOR_MARKET_SHARE) * common[None, :]
        alpha = 1.0 - 2.0 ** (-1.0 / _ANCHOR_HALF_LIFE_Q)
        ema = np.full(n, np.nan)
        out = np.zeros((n, K))
        seen = np.zeros(n, dtype=bool)
        for k in range(K):
            use = live[:, k]
            ema = np.where(use, np.where(np.isnan(ema), z[:, k], ema + alpha * (z[:, k] - ema)), ema)
            seen |= use
            out[:, k] = np.where(seen, ema, 0.0)  # after the last live quarter: held constant
        first = np.where(live.any(axis=1), live.argmax(axis=1), 0)
        before = np.arange(K)[None, :] < first[:, None]
        out = np.where(before, out[np.arange(n), first][:, None], out)  # before the price window: relative path kept
        out[~names] = 0.0
        return out if out.any() else None

    def _kidx(self, as_of: date) -> np.ndarray:
        """Per-ticker index of the latest quarter public on or before ``as_of`` (-1: none)."""
        d = _d64(min(as_of, self.end))
        return (self._avail <= d).sum(axis=1) - 1

    def _ttm(self, key: str, rows: np.ndarray, kk: np.ndarray, back: int = 0) -> np.ndarray:
        """Sum of the four quarters ending ``kk - back`` (NaN when history is insufficient)."""
        x = self._q[key]
        k = kk - back
        ok = (rows >= 0) & (k >= 3)
        out = np.full(len(rows), np.nan)
        r, kc = rows[ok], k[ok]
        out[ok] = x[r, kc - 3] + x[r, kc - 2] + x[r, kc - 1] + x[r, kc]
        return out

    def _point(self, key: str | np.ndarray, rows: np.ndarray, kk: np.ndarray, back: int = 0) -> np.ndarray:
        """Value of quarter ``kk - back`` of a quarterly series (a ``self._q`` key or an (n, K) array)."""
        x = self._q[key] if isinstance(key, str) else key
        k = kk - back
        ok = (rows >= 0) & (k >= 0)
        out = np.full(len(rows), np.nan)
        out[ok] = x[rows[ok], k[ok]]
        return out

    # ------------------------------------------------------------------ estimates

    def _build_estimates(self) -> None:
        n, K = self.n_tickers, self._K
        rng = self._rng("estimates")
        rows = np.arange(n)
        rev_ttm = np.full((n, K), np.nan)
        ni_ttm = np.full((n, K), np.nan)
        for k in range(3, K):
            kk = np.full(n, k)
            rev_ttm[:, k] = self._ttm("rev", rows, kk)
            ni_ttm[:, k] = self._ttm("ni", rows, kk)
        g4 = np.full((n, K), np.nan)
        g4[:, 3:] = (self._g[:, 3:] + self._g[:, 2:-1] + self._g[:, 1:-2] + self._g[:, :-3]) / 4
        gexp = 0.6 * g4 + 0.4 * self._G[:, None] + rng.normal(0, 0.012, (n, K))
        est_rev = rev_ttm * (1 + gexp)
        eps_ttm = ni_ttm / self._q["shares"]
        eexp = gexp + 0.03 + rng.normal(0, 0.03, (n, K))
        est_eps = np.where(eps_ttm > 0, eps_ttm * (1 + eexp), eps_ttm + 0.35 * np.abs(eps_ttm))
        for i, st in self._stories.items():
            ka, T = int(self._k_end[i]), st.targets
            if ka < 4:
                continue
            r = self._rng("est-plant", i)
            self._surprise[i, :ka] = r.uniform(0.02, 0.07, ka)
            self._surprise[i, ka] = T["surprise"]
            if st.archetype == "momentum_leader":
                for k in range(max(4, ka - 4), ka + 1):
                    est_rev[i, k] = est_rev[i, k - 1] * (1 + r.uniform(0.02, 0.05))
                    est_eps[i, k] = est_eps[i, k - 1] * (1 + r.uniform(0.03, 0.07))
            est_rev[i, ka] = est_rev[i, ka - 1] * (1 + T["rev_rev"])
            est_eps[i, ka] = est_eps[i, ka - 1] * (1 + T["eps_rev"])
        self._est_rev, self._est_eps, self._eps_ttm = est_rev, est_eps, eps_ttm
        lc = self._lcap
        n_an = np.clip(np.round(3 + 9 * (lc - 9.0) + rng.normal(0, 2.5, n)), 1, 42)
        for i in self._stories:
            n_an[i] = max(n_an[i], 8)
        n_an[np.isin(self._stype, ["etf", "preferred"])] = np.nan
        self._n_analysts = n_an
        self._tp_prem = rng.normal(0.12, 0.06, n)

    # ------------------------------------------------------------------ short interest

    def _build_short_interest(self) -> None:
        n = self.n_tickers
        first = date(self.start.year, self.start.month, 1) - timedelta(days=125)
        settle: list[np.datetime64] = []
        y, m = first.year, first.month
        while True:
            mid = np.busday_offset(np.datetime64(f"{y:04d}-{m:02d}-15"), 0, roll="backward")
            eom = np.busday_offset(_month_end(y, m), 0, roll="backward")
            if mid > self._end_d:
                break
            settle.append(mid)
            if eom <= self._end_d:
                settle.append(eom)
            y, m = (y + 1, 1) if m == 12 else (y, m + 1)
        sd = np.array(settle, dtype="datetime64[D]")
        M = len(sd)
        rng = self._rng("short-interest")
        base = np.clip(np.exp(rng.normal(math.log(0.03), 0.75, n)), 0.003, 0.30)
        base[np.array([r == "penny" for r in self._role])] *= 2.0
        z = rng.normal(0, 0.08, (n, M))
        ar = np.zeros((n, M))
        for j in range(1, M):
            ar[:, j] = 0.9 * ar[:, j - 1] + z[:, j]
        si = np.clip(base[:, None] * np.exp(ar), 0.001, 0.45)
        sdays = sd.astype(np.int64)
        for i, st in self._stories.items():
            T = st.targets
            jit = np.exp(self._rng("si-plant", i).normal(0, 0.04, M))
            if st.archetype == "momentum_leader":
                si[i] = T["si"] * jit
            else:
                d0 = self._dd[st.shock_session].astype(np.int64)
                dl = sdays[-1]
                prog = np.clip((sdays - d0) / max(1, dl - d0), 0, 1) ** 0.8
                si[i] = (T["si_pre"] + (T["si"] - T["si_pre"]) * prog) * jit
            si[i, -1] = T["si"]
        si[np.array([s == "preferred" for s in self._stype])] = np.nan
        kk = (self._avail[:, :, None] <= sd[None, None, :]).sum(axis=1) - 1  # (n, M)
        shares = np.take_along_axis(self._q["shares"], np.clip(kk, 0, None), axis=1)
        flt = np.round(self._float_frac[:, None] * shares)
        self._si_dates, self._si_float, self._si_shares = sd, flt, np.round(si * flt)

    # ------------------------------------------------------------------ benchmark

    def _build_benchmark(self) -> None:
        close = self._close
        cap = self._cap_t
        with np.errstate(invalid="ignore", divide="ignore"):
            r = close[1:] / close[:-1] - 1
        w = cap[:-1].copy()
        inc = ~np.isin(self._stype, ["etf", "preferred"])
        ok = np.isfinite(r) & np.isfinite(w) & inc[None, :]
        w = np.where(ok, w, 0.0)
        num = (w * np.where(ok, r, 0.0)).sum(axis=1)
        den = w.sum(axis=1)
        ret = np.where(den > 0, num / np.where(den > 0, den, 1.0), 0.0)
        level = 1000.0 * np.concatenate([[1.0], np.cumprod(1 + ret)])
        self._bench = pd.Series(np.round(level, 4), index=self._px["close"].index, name=BENCHMARK_SYMBOL)

    # ------------------------------------------------------------------ options (lazy)

    def _options_arrays(self) -> dict[str, np.ndarray]:
        if self._options is not None:
            return self._options
        n, N = self.n_tickers, self._N
        rng = self._rng("options")
        lr = pd.DataFrame(self._logret)
        rv = lr.rolling(20, min_periods=5).std().to_numpy() * math.sqrt(252)
        fallback = np.sqrt(self._idio**2 + (self._beta * 0.15) ** 2)
        rv = np.where(np.isfinite(rv), rv, fallback[None, :])
        base = pd.DataFrame(rv).ewm(span=10, adjust=False).mean().to_numpy() * self._iv_ratio[None, :]
        bump = np.zeros((N, n))
        t = np.arange(N)
        for k in range(self._K):
            s = self._rs[:, k]
            for d in range(1, 11):
                ok = s - d >= 0
                bump[s[ok] - d, np.flatnonzero(ok)] += 0.035 * (11 - d) / 10
        pc_mult = np.ones((N, n))
        for i, st in self._stories.items():
            if st.archetype == "momentum_leader":
                pc_mult[:, i] = 0.7
                continue
            s = st.shock_session
            decay = np.where(t >= s, np.exp(-(t - s) / 70.0), 0.0)
            bump[:, i] += st.targets["iv_bump"] * decay
            pc_mult[:, i] = 1 + 0.9 * decay
        for sc in self._scenarios:
            h = sc["headline_session"]
            members = np.flatnonzero(self._industry == sc["industry"])
            bump[:, members] += (0.04 * np.where(t >= h, np.exp(-(t - h) / 25.0), 0.0))[:, None]
        iv = np.clip(base + bump, 0.08, 3.0)
        hi = pd.DataFrame(iv).rolling(252, min_periods=1).max().to_numpy()
        lo = pd.DataFrame(iv).rolling(252, min_periods=1).min().to_numpy()
        vol = np.nan_to_num(self._px["volume"].to_numpy(), nan=0.0)
        call = np.round(vol * self._opt_activity[None, :] / 100 * np.exp(rng.normal(0, 0.3, (N, n))))
        put = np.round(call * self._pcr_base[None, :] * pc_mult * np.exp(rng.normal(0, 0.25, (N, n))))
        call_oi = np.round(pd.DataFrame(call).rolling(20, min_periods=1).mean().to_numpy() * 14)
        put_oi = np.round(pd.DataFrame(put).rolling(20, min_periods=1).mean().to_numpy() * 16)
        none = (self._opt_activity <= 0)[None, :] | ~np.isfinite(self._close)
        out = {}
        for key, arr in ((F.IV_30D_ATM, iv), (F.IV_30D_ATM_1Y_HIGH, hi), (F.IV_30D_ATM_1Y_LOW, lo), (F.PUT_VOLUME, put),
                         (F.CALL_VOLUME, call), (F.PUT_OPEN_INTEREST, put_oi), (F.CALL_OPEN_INTEREST, call_oi)):
            a = np.where(none, np.nan, arr)
            out[key] = np.round(a, 4) if key.startswith("iv") else a
        self._options = out
        return out

    # ================================================================== public API

    def archetype(self, ticker: str) -> str:
        """Ground-truth archetype label of ``ticker`` (synthetic-only; for evaluation, never shown to the LLM)."""
        if ticker not in self._idx:
            raise KeyError(ticker)
        return str(self._arch[self._idx[ticker]])

    def archetypes(self) -> dict[str, str]:
        """Ground-truth archetype of every ticker (synthetic-only evaluation aid)."""
        return {t: str(self._arch[i]) for i, t in enumerate(self._tickers)}

    def story(self, ticker: str) -> dict[str, Any]:
        """Planted story parameters for ``ticker`` (synthetic-only evaluation aid; ``{}`` for normal names).

        Keys: archetype, expected_dislocation_type, canonical_fit (designed to pass the canonical demo
        screen as of ``end``), near_misses (canonical conditions deliberately missed), theme,
        shock_date, report_date, followthrough_date, targets.
        """
        i = self._idx.get(ticker)
        if i is None:
            raise KeyError(ticker)
        st = self._stories.get(i)
        if st is None:
            return {}
        d = lambda s: _py(self._dd[s]) if 0 <= s < self._N else None  # noqa: E731
        return {
            "archetype": st.archetype,
            "expected_dislocation_type": ARCHETYPE_TO_DISLOCATION.get(st.archetype),
            "canonical_fit": bool(st.fit),
            "near_misses": list(st.misses),
            "theme": st.theme,
            "shock_date": d(st.shock_session),
            "report_date": d(st.report_session),
            "followthrough_date": d(st.cap_session),
            "calibrated": bool(st.params.get("calibrated", True)),
            "targets": {k: float(v) for k, v in st.targets.items()},
        }

    @property
    def tickers(self) -> list[str]:
        return list(self._tickers)

    def get_universe(self, spec: "UniverseSpec | None", as_of: date) -> pd.DataFrame:
        """All securities listed on ``as_of`` with ``fields.UNIVERSE_COLUMNS``.

        Applies only ``spec.country`` and ``spec.security_types`` (cheap reference filters); price,
        liquidity and sector filters are the screen engine's job. ``market_cap`` = latest close on or
        before ``as_of`` x shares outstanding from the latest public report (NaN before ``start``).
        """
        d = _as_date(as_of)
        t = self._session_le(min(d, self.end))
        keep = np.ones(self.n_tickers, dtype=bool)
        if t >= 0:
            keep &= self._listing <= t
        else:
            keep &= self._listing == 0
        if spec is not None:
            if getattr(spec, "country", None):
                keep &= self._country == spec.country
            types = getattr(spec, "security_types", None)
            if types:
                keep &= np.isin(self._stype, list(types))
        rows = np.flatnonzero(keep)
        cap = self._cap_t[t, rows] if t >= 0 else np.full(rows.size, np.nan)
        df = pd.DataFrame(
            {
                F.NAME: [self._names[i] for i in rows],
                F.GICS_SECTOR: [self._sector[i] for i in rows],
                F.GICS_INDUSTRY: [self._industry[i] for i in rows],
                F.EXCHANGE: [self._exchange[i] for i in rows],
                F.COUNTRY: [self._country[i] for i in rows],
                F.CURRENCY: ["USD"] * rows.size,
                F.SECURITY_TYPE: [self._stype[i] for i in rows],
                F.MARKET_CAP: cap.astype(float),
                F.VENDOR_ID: [f"{self._tickers[i]} SYN" for i in rows],
            },
            index=pd.Index([self._tickers[i] for i in rows], name="ticker"),
        )
        return df[F.UNIVERSE_COLUMNS]

    def get_price_history(self, tickers: list[str], start: date, end: date) -> PricePanel:
        """OHLCV for ``tickers`` on sessions in [start, end] (clamped to the provider window)."""
        tick, _ = self._rows(tickers)
        s, e = _as_date(start), min(_as_date(end), self.end)
        idx = self._px["close"].index
        mask = (idx >= pd.Timestamp(s)) & (idx <= pd.Timestamp(e))
        frames = {k: self._px[k].loc[mask].reindex(columns=pd.Index(tick, name="ticker")) for k in F.PRICE_FIELDS}
        return PricePanel(frames["open"], frames["high"], frames["low"], frames["close"], frames["volume"])

    def get_benchmark_history(self, start: date, end: date, symbol: str | None = None) -> pd.Series:
        """Cap-weighted total-universe index "SYNTH-US" (ETFs and preferreds excluded), base 1000.

        ``symbol`` may also be one of the provider's tickers (its adjusted close is returned).
        """
        s, e = _as_date(start), min(_as_date(end), self.end)
        if symbol is None or symbol.upper() in (BENCHMARK_SYMBOL, "SYNTH", "SPX-SYN"):
            ser = self._bench
        elif symbol in self._idx:
            ser = self._px["close"][symbol].rename(symbol)
        else:
            raise ProviderError(f"unknown benchmark symbol '{symbol}' (synthetic provider offers '{BENCHMARK_SYMBOL}')")
        idx = ser.index
        return ser.loc[(idx >= pd.Timestamp(s)) & (idx <= pd.Timestamp(e))].copy()

    def get_fundamentals(self, tickers: list[str], as_of: date) -> pd.DataFrame:
        """Point-in-time fundamentals snapshot keyed on report date: ``fields.FUNDAMENTAL_COLUMNS`` followed by
        the optional ``fields.FUNDAMENTAL_OPTIONAL_COLUMNS`` (total assets at the latest quarter end and
        four quarters earlier)."""
        tick, rows = self._rows(tickers)
        kk_all = self._kidx(_as_date(as_of))
        kk = np.where(rows >= 0, kk_all[np.clip(rows, 0, None)], -1)
        q = self._point
        cfo_ttm, capex_ttm = self._ttm("cfo", rows, kk), self._ttm("capex", rows, kk)
        data = {
            F.PERIOD_END: self._date_col(self._pe, rows, kk),
            F.REPORT_DATE: self._date_col(self._rd, rows, kk),
            F.REVENUE_TTM: self._ttm("rev", rows, kk),
            F.REVENUE_TTM_PRIOR_YEAR: self._ttm("rev", rows, kk, 4),
            F.REVENUE_LAST_Q: q("rev", rows, kk),
            F.REVENUE_LAST_Q_PRIOR_YEAR: q("rev", rows, kk, 4),
            F.GROSS_PROFIT_TTM: self._ttm("gp", rows, kk),
            F.GROSS_PROFIT_TTM_PRIOR_YEAR: self._ttm("gp", rows, kk, 4),
            F.OPERATING_INCOME_TTM: self._ttm("oi", rows, kk),
            F.OPERATING_INCOME_TTM_PRIOR_YEAR: self._ttm("oi", rows, kk, 4),
            F.EBITDA_TTM: self._ttm("ebitda", rows, kk),
            F.NET_INCOME_TTM: self._ttm("ni", rows, kk),
            F.CFO_TTM: cfo_ttm,
            F.CAPEX_TTM: capex_ttm,
            F.FCF_TTM: cfo_ttm - capex_ttm,
            F.TOTAL_DEBT: q("debt", rows, kk),
            F.CASH: q("cash", rows, kk),
            F.INTEREST_EXPENSE_TTM: self._ttm("interest", rows, kk),
            F.TOTAL_EQUITY: q("equity", rows, kk),
            F.SHARES_OUTSTANDING: q("shares", rows, kk),
            F.TOTAL_ASSETS: q("assets", rows, kk),
            F.TOTAL_ASSETS_PRIOR_YEAR: q("assets", rows, kk, 4),
        }
        return pd.DataFrame(data, index=pd.Index(tick, name="ticker"))[F.FUNDAMENTAL_COLUMNS + F.FUNDAMENTAL_OPTIONAL_COLUMNS]

    def get_estimates(self, tickers: list[str], as_of: date) -> pd.DataFrame:
        """Consensus snapshot (``fields.ESTIMATE_COLUMNS``); "3m ago" values are the consensus 91 days earlier."""
        d = _as_date(as_of)
        tick, rows = self._rows(tickers)
        safe = np.clip(rows, 0, None)
        kk = np.where(rows >= 0, self._kidx(d)[safe], -1)
        k3 = np.where(rows >= 0, self._kidx(min(d, self.end) - timedelta(days=91))[safe], -1)
        q = self._point
        eps_ntm, eps_3m = q(self._est_eps, rows, kk), q(self._est_eps, rows, k3)
        t = self._session_le(min(d, self.end))
        tp = np.full(len(rows), np.nan)
        if t >= 0:
            mean_px = pd.DataFrame(self._close[max(0, t - 62): t + 1]).mean(axis=0).to_numpy()  # skips NaN, no warnings
            with np.errstate(invalid="ignore", divide="ignore"):
                rev3 = np.where(np.abs(eps_3m) > 0, (eps_ntm - eps_3m) / np.abs(eps_3m), 0.0)
            ok = rows >= 0
            tp[ok] = mean_px[rows[ok]] * (1 + self._tp_prem[rows[ok]]) * (1 + 0.6 * np.clip(np.nan_to_num(rev3[ok]), -0.5, 0.5))
            tp = np.round(tp, 2)
        has = (rows >= 0) & (kk >= 0)
        nxt = np.full(len(rows), np.datetime64("NaT"), dtype="datetime64[D]")
        okn = has & (kk + 1 < self._K)
        nxt[okn] = self._rd[rows[okn], kk[okn] + 1]
        data = {
            F.REVENUE_NTM_EST: q(self._est_rev, rows, kk),
            F.REVENUE_NTM_EST_3M_AGO: q(self._est_rev, rows, k3),
            F.EPS_NTM_EST: eps_ntm,
            F.EPS_NTM_EST_3M_AGO: eps_3m,
            F.EPS_TTM: q(self._eps_ttm, rows, kk),
            F.NUM_ANALYSTS: np.where(has, self._n_analysts[safe], np.nan),
            F.TARGET_PRICE_MEAN: np.where(has, tp, np.nan),
            F.LAST_EPS_SURPRISE: q(self._surprise, rows, kk),
            F.LAST_EARNINGS_DATE: self._date_col(self._rd, rows, kk),
            F.NEXT_EARNINGS_DATE: pd.to_datetime(nxt).astype("datetime64[ns]"),
        }
        return pd.DataFrame(data, index=pd.Index(tick, name="ticker"))[F.ESTIMATE_COLUMNS]

    def get_short_interest(self, tickers: list[str], as_of: date) -> pd.DataFrame:
        """Latest semi-monthly short-interest settlement on or before ``as_of`` and the one ~1 month earlier."""
        tick, rows = self._rows(tickers)
        j = int(np.searchsorted(self._si_dates, _d64(min(_as_date(as_of), self.end)), side="right")) - 1
        n = len(rows)
        si, si1, flt = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
        sdate = np.full(n, np.datetime64("NaT"), dtype="datetime64[D]")
        ok = rows >= 0
        if j >= 0:
            si[ok] = self._si_shares[rows[ok], j]
            flt[ok] = self._si_float[rows[ok], j]
            sdate[ok & np.isfinite(si)] = self._si_dates[j]
            if j >= 2:
                si1[ok] = self._si_shares[rows[ok], j - 2]
        data = {
            F.SHORT_INTEREST_SHARES: si,
            F.SHORT_INTEREST_SHARES_1M_AGO: si1,
            F.FLOAT_SHARES: np.where(np.isfinite(si), flt, np.nan),
            F.SI_SETTLEMENT_DATE: pd.to_datetime(sdate).astype("datetime64[ns]"),
        }
        return pd.DataFrame(data, index=pd.Index(tick, name="ticker"))[F.SHORT_INTEREST_COLUMNS]

    def get_options_summary(self, tickers: list[str], as_of: date) -> pd.DataFrame:
        """Listed-options summary for the latest session on or before ``as_of`` (NaN where no options trade)."""
        tick, rows = self._rows(tickers)
        t = self._session_le(min(_as_date(as_of), self.end))
        ok = rows >= 0
        data = {}
        arrs = self._options_arrays() if t >= 0 else {}
        for col in F.OPTIONS_COLUMNS:
            v = np.full(len(rows), np.nan)
            if t >= 0:
                v[ok] = arrs[col][t, rows[ok]]
            data[col] = v
        return pd.DataFrame(data, index=pd.Index(tick, name="ticker"))[F.OPTIONS_COLUMNS]

    def get_documents(
        self,
        ticker: str,
        kinds: set[DocumentKind] | None,
        start: date,
        end: date,
        limit: int = 10,
    ) -> list[Document]:
        """Documents published in [start, end] (clamped to the provider's end), newest first.

        ``kinds=None`` means every kind. Unknown tickers return an empty list.
        """
        if ticker not in self._idx:
            return []
        s, e = _as_date(start), min(_as_date(end), self.end)
        wanted = None if kinds is None else {DocumentKind(k) for k in kinds}
        out: list[Document] = []
        for doc in self._documents_for(ticker):
            day = doc.published_at.date()
            if day > e:
                continue
            if day < s:
                break
            if wanted is None or doc.kind in wanted:
                out.append(doc.model_copy(deep=True))
                if limit is not None and len(out) >= limit:
                    break
        return out

    def _date_col(self, arr: np.ndarray, rows: np.ndarray, kk: np.ndarray) -> pd.DatetimeIndex:
        """datetime64[ns] values of a per-quarter date array (NaT where unknown); positional, not aligned."""
        out = np.full(len(rows), np.datetime64("NaT"), dtype="datetime64[D]")
        ok = (rows >= 0) & (kk >= 0)
        out[ok] = arr[rows[ok], kk[ok]]
        return pd.to_datetime(out).astype("datetime64[ns]")

    # ================================================================== documents

    def _documents_for(self, ticker: str) -> list[Document]:
        cached = self._doc_cache.get(ticker)
        if cached is not None:
            return cached
        i = self._idx[ticker]
        docs: list[Document] = []
        if self._stype[i] not in ("etf", "preferred"):
            for k in range(self._K):
                s = int(self._rs[i, k])
                if s < 1 or s < self._listing[i] or k < 8:
                    continue
                docs.extend(self._quarter_documents(i, k, s))
            docs.extend(self._industry_news(i))
        docs = [d for d in docs if d.published_at.date() <= self.end]
        docs.sort(key=lambda d: (d.published_at, d.doc_id), reverse=True)
        self._doc_cache[ticker] = docs
        return docs

    def _vocab(self, i: int) -> dict[str, Any]:
        v = _INDUSTRY_VOCAB.get(self._industry[i]) or _SECTOR_VOCAB.get(self._sector[i]) or _SECTOR_VOCAB["Industrials"]
        st = self._stories.get(i)
        if st is not None and st.archetype == "sector_contagion":  # its revenue base sits outside the headline's reach
            v = dict(v, markets=self._scenarios[st.scenario]["markets"])
        return v

    def _fiscal(self, i: int, k: int) -> tuple[int, int]:
        pe = _py(self._pe[i, k])
        o = int(self._offset[i])
        return int(self._fq[i, k]), pe.year if pe.month - o >= 1 else pe.year - 1

    def _guide_mid(self, i: int, k: int) -> float:
        """Next-quarter revenue outlook: NTM consensus shaped by last year's seasonality."""
        rows, kk = np.array([i]), np.array([k])
        ttm = float(self._ttm("rev", rows, kk)[0])
        return float(self._est_rev[i, k] * self._q["rev"][i, k - 3] / ttm)

    def _call_facts(self, i: int, k: int, rng: np.random.Generator) -> dict[str, Any]:
        q = self._q
        rows, kk = np.array([i]), np.array([k])
        st = self._stories.get(i)
        fq, fy = self._fiscal(i, k)
        nfq, nfy = (1, fy + 1) if fq == 4 else (fq + 1, fy)
        pe, rd = _py(self._pe[i, k]), _py(self._rd[i, k])
        rev, rev_py = float(q["rev"][i, k]), float(q["rev"][i, k - 4])
        sh, sh_py = float(q["shares"][i, k]), float(q["shares"][i, k - 4])
        ceo, cfo, ir = self._mgmt[i]
        v = self._vocab(i)
        mid = self._guide_mid(i, k)
        role = "other"
        if st is not None:
            if k == st.story_k:
                role = "story"
            elif k == st.story_k - 1:
                role = "pre"
            elif k < st.story_k:
                role = "earlier"
        f: dict[str, Any] = dict(
            i=i, k=k, ticker=self._tickers[i], company=self._names[i], arch=(st.archetype if st else "normal"), role=role,
            st=st, fq=fq, fy=fy, nfq=nfq, nfy=nfy, q_word=_ORDINAL[fq], nq_word=_ORDINAL[nfq], pe=pe, rd=rd,
            pe_py=_py(self._pe[i, k - 4]), rev=rev, rev_py=rev_py, growth=rev / rev_py - 1,
            gm=float(q["gp"][i, k]) / rev, gm_py=float(q["gp"][i, k - 4]) / rev_py,
            om=float(q["oi"][i, k]) / rev, om_py=float(q["oi"][i, k - 4]) / rev_py, oi=float(q["oi"][i, k]),
            oi_py=float(q["oi"][i, k - 4]), eps=float(q["ni"][i, k]) / sh, eps_py=float(q["ni"][i, k - 4]) / sh_py,
            cfo=float(q["cfo"][i, k]), capex=float(q["capex"][i, k]),
            fcf_ttm=float(self._ttm("cfo", rows, kk)[0] - self._ttm("capex", rows, kk)[0]),
            rev_ttm=float(self._ttm("rev", rows, kk)[0]), ebitda_ttm=float(self._ttm("ebitda", rows, kk)[0]),
            cash=float(q["cash"][i, k]), debt=float(q["debt"][i, k]), guide_lo=mid * 0.98, guide_hi=mid * 1.02, guide_mid=mid,
            next_py=float(q["rev"][i, k - 3]), surprise=float(self._surprise[i, k]),
            ceo=ceo, cfo_name=cfo, ir=ir, ceo_first=ceo.split()[0], cfo_first=cfo.split()[0], ir_first=ir.split()[0],
            products=v["products"], customers=v["customers"], markets=v["markets"], kpi=v["kpi"],
            industry=self._industry[i] or "", sector=self._sector[i] or "",
            exch=self._exchange[i],
        )
        f["fcf_q"] = f["cfo"] - f["capex"]
        f["net_cash"] = f["cash"] - f["debt"]
        f["lev"] = (f["debt"] - f["cash"]) / f["ebitda_ttm"] if f["ebitda_ttm"] > 0 else float("nan")
        f["prior_mid"] = self._guide_mid(i, k - 1)
        # Narrative constants (deterministic per seed / ticker / report date).
        f.update(
            backlog=rev * rng.uniform(1.4, 2.4), bl_g=rng.uniform(0.06, 0.15), btb=rng.uniform(1.04, 1.15),
            qtd=rng.uniform(0.08, 0.16), weeks_in=max(3, (rd - pe).days // 7), price_real=rng.uniform(0.015, 0.04),
            sellthrough=int(rng.integers(6, 15)), wk=int(rng.integers(5, 8)), n_cust=int(rng.integers(2, 4)),
            win=int(rng.integers(38, 56)), awards=int(rng.integers(4, 12)), top_n=int(rng.choice([10, 15, 20, 25])),
            pipe=rng.uniform(0.08, 0.2), seg_g=rng.normal(0, 0.03, 3), lt_margin=int(round(f["om_py"] * 100 + rng.uniform(3, 6))),
            currency=["Brazilian real", "Turkish lira", "Argentine peso", "Egyptian pound", "Nigerian naira"][int(rng.integers(5))],
            region=["Midwest", "Northern European", "Pacific Northwest", "Southeast", "Great Plains", "Central European"][int(rng.integers(6))],
            component=["power module", "precision bearing", "sterile barrier film", "custom controller chip", "specialty resin"][int(rng.integers(5))],
            haircut=int(rng.integers(5, 11)), fx_bp=int(rng.choice([100, 150, 200])), svc=rng.uniform(0.07, 0.14),
            exit_rate=int(rng.integers(88, 97)),
        )
        # Segment split (sums exactly to reported revenue), opex split, share count and buybacks.
        share_a = rng.uniform(0.5, 0.7)
        seg_a_py = rev_py * share_a
        seg_a = seg_a_py * (1 + f["growth"] + rng.normal(0, 0.02))
        opex = float(q["gp"][i, k] - q["oi"][i, k])
        rnd_share = {"Information Technology": 0.45, "Health Care": 0.40, "Communication Services": 0.30}.get(
            f["sector"], 0.22) * rng.uniform(0.8, 1.2)
        f.update(seg_a=seg_a, seg_b=rev - seg_a, seg_a_g=seg_a / seg_a_py - 1, seg_b_g=(rev - seg_a) / (rev_py - seg_a_py) - 1,
                 opex=opex, rnd=opex * rnd_share, sga=opex * (1 - rnd_share), interest_q=float(q["interest"][i, k]),
                 shares=sh, shares_py=sh_py, pretax=float(q["oi"][i, k] - q["interest"][i, k]),
                 ni_ttm=float(self._ttm("ni", rows, kk)[0]), n_launch=int(rng.integers(3, 9)), vit=int(rng.integers(22, 38)),
                 savings=rev * rng.uniform(0.004, 0.01), otd=rng.uniform(91.0, 97.5), emp_k=rng.uniform(250e3, 450e3))
        f["vit_py"] = f["vit"] - int(rng.integers(1, 4))
        f["conv"] = f["fcf_ttm"] / f["ni_ttm"] if f["ni_ttm"] > 0 else float("nan")
        bought = float(q["shares"][i, k - 1]) - sh
        lo_s, hi_s = self._session_le(_py(self._pe[i, k - 1])) + 1, self._session_le(pe)
        f["bought"], f["buyback_q"] = 0.0, 0.0
        if bought > 0 and 0 <= lo_s <= hi_s:
            px = self._close[lo_s: hi_s + 1, i]
            px = px[np.isfinite(px)]
            if px.size:
                f["bought"], f["buyback_q"] = bought, bought * float(px.mean())
        if st is not None and st.archetype != "momentum_leader":
            T = st.targets
            if st.archetype == "transitory_shock" and role == "story":
                f["oneoff"] = T["oneoff_pts"] * rev_py
                f["underlying"] = f["growth"] + T["oneoff_pts"]
            if st.archetype == "guidance_reset" and role == "story":
                cap_now = float(self._cap_t[max(0, self._session_le(rd) - 1), i])
                f["buyback"] = max(25e6, round(T["buyback_pct"] * cap_now / 25e6) * 25e6)
                f["buyback_pct"] = f["buyback"] / cap_now
                f["guide_cut"] = T["guide_cut"]
                f["consensus_gap"] = mid * T["guide_cut"] / (1 - T["guide_cut"])
            if st.archetype == "sector_contagion":
                f["scen"] = self._scenarios[st.scenario]
                f["exposure"] = int(T["exposure_pct"])
                f["peer_exp"] = int(rng.integers(18, 35))
        return f

    def _quarter_documents(self, i: int, k: int, s: int) -> list[Document]:
        tick = self._tickers[i]
        rd = _py(self._rd[i, k])
        rng = self._rng("docs", tick, rd.isoformat())
        f = self._call_facts(i, k, rng)
        ymd = rd.strftime("%Y%m%d")
        docs: list[Document] = []
        segs = self._transcript_segments(f, rng)
        text = "\n\n".join(f"{sg.speaker} ({sg.role}): {sg.text}" for sg in segs)
        docs.append(Document(
            doc_id=f"SYN-TR-{tick}-{ymd}", ticker=tick, kind=DocumentKind.TRANSCRIPT,
            title=f"{f['company']} ({tick}) Q{f['fq']} FY{f['fy']} Earnings Call Transcript",
            published_at=datetime.combine(rd, time(12, 30)), source=f"{_SOURCE} Transcripts",
            url=f"synthetic://transcripts/{tick}/{ymd}", text=text, segments=segs,
            metadata={"event": "earnings_call", "fiscal_period": f"Q{f['fq']} FY{f['fy']}", "period_end": f["pe"].isoformat()},
        ))
        docs.append(self._press_release(f))
        docs.append(self._reaction_news(f, s))
        st = f["st"]
        N = self._N
        next_s = int(self._rs[i, k + 1]) if k + 1 < self._K and self._rs[i, k + 1] > 0 else N
        if st is not None and f["role"] == "story" and st.archetype != "sector_contagion" and st.cap_session > s:
            docs.extend(self._analyst_action(f, st.cap_session, rng, story=True))
        elif rng.random() < (0.8 if st is not None else 0.5):
            t = s + int(rng.integers(3, 40))
            if t < min(next_s, N):
                docs.extend(self._analyst_action(f, t, rng, story=False))
        if rng.random() < 0.35:
            t = s + int(rng.integers(8, 50))
            if t < min(next_s, N):
                docs.append(self._corporate_news(f, t, rng))
        if st is not None or _stable_hash(self.seed, tick, "files") % 100 < 35:
            docs.append(self._periodic_filing(f, rng))
        if st is not None and f["role"] == "story":
            docs.append(self._form_8k(f))
        return docs

    # ------------------------------------------------------------------ transcript

    def _transcript_segments(self, f: dict[str, Any], rng: np.random.Generator) -> list[TranscriptSegment]:
        P, Q = "prepared_remarks", "qa"
        tod = "morning" if rng.random() < 0.7 else "afternoon"
        f["tod"] = tod
        segs = [
            TranscriptSegment(speaker="Operator", role="Operator", section=P, text=(
                f"Good {tod}, and welcome to the {f['company']} {f['q_word']} quarter fiscal {f['fy']} earnings conference call. "
                "At this time, all participants are in a listen-only mode. A question-and-answer session will follow the "
                "prepared remarks. As a reminder, this call is being recorded. I would now like to turn the call over to "
                f"{f['ir']}, Vice President of Investor Relations. Please go ahead.")),
            TranscriptSegment(speaker=f["ir"], role="IR", section=P, text=(
                f"Thank you, operator, and good {tod}, everyone. With me today are {f['ceo']}, our Chief Executive Officer, "
                f"and {f['cfo_name']}, our Chief Financial Officer. Before we begin, please note that today's discussion "
                "contains forward-looking statements, including statements about our outlook, which are subject to risks and "
                "uncertainties that could cause actual results to differ materially. Please refer to our SEC filings for a "
                "discussion of those risks. We will also refer to certain non-GAAP measures; reconciliations are included in "
                f"the earnings release posted on our investor relations website. With that, I'll turn the call over to {f['ceo_first']}.")),
            TranscriptSegment(speaker=f["ceo"], role="CEO", section=P, text=" ".join(self._ceo_remarks(f, rng))),
            TranscriptSegment(speaker=f["cfo_name"], role="CFO", section=P, text=" ".join(self._cfo_remarks(f, rng))),
        ]
        planted = f["st"] is not None
        if not planted:
            n_q = int(rng.integers(2, 4))
        else:
            n_q = int(rng.integers(7, 9)) if f["role"] == "story" else int(rng.integers(6, 8))
        topics = self._qa_topics(f, rng)[:n_q]
        fu = self._followups(f, rng)
        followups = [fu[int(j)] for j in rng.permutation(len(fu))]
        analysts = [self._coverage[f["i"]][int(j)] for j in rng.permutation(8)[: len(topics)]]
        greets = ["Hi, thanks for taking my question.", f"Good {tod}, and thanks for the time.", "Thanks, operator.",
                  f"Hey, good {tod}, everyone.", "Thank you for taking the question.", "Hi, thanks."]
        for n_, (topic, (who, firm)) in enumerate(zip(topics, analysts)):
            q_text, answerer, a_text = topic
            lead = "Our first question" if n_ == 0 else "Our next question"
            segs.append(TranscriptSegment(speaker="Operator", role="Operator", section=Q,
                                          text=f"{lead} comes from {who} with {firm.rstrip('.')}. Please go ahead."))
            segs.append(TranscriptSegment(speaker=who, role="Analyst", section=Q,
                                          text=f"{greets[int(rng.integers(len(greets)))]} {q_text}"))
            name = f["ceo"] if answerer == "CEO" else f["cfo_name"]
            segs.append(TranscriptSegment(speaker=name, role=answerer, section=Q, text=a_text))
            if planted and followups and rng.random() < 0.6:
                fq_text, f_role, fa_text = followups.pop(0)
                segs.append(TranscriptSegment(speaker=who, role="Analyst", section=Q, text=fq_text))
                segs.append(TranscriptSegment(speaker=f["ceo"] if f_role == "CEO" else f["cfo_name"], role=f_role,
                                              section=Q, text=fa_text))
        segs.append(TranscriptSegment(speaker="Operator", role="Operator", section=Q, text=(
            f"There are no further questions at this time. I'll turn the call back to {f['ceo']} for closing remarks.")))
        segs.append(TranscriptSegment(speaker=f["ceo"], role="CEO", section=Q, text=self._closing(f)))
        segs.append(TranscriptSegment(speaker="Operator", role="Operator", section=Q, text=(
            "This concludes today's conference call. Thank you for participating. You may now disconnect.")))
        return segs

    def _ceo_remarks(self, f: dict[str, Any], rng: np.random.Generator) -> list[str]:
        a, role, st = f["arch"], f["role"], f["st"]
        p1, p2, p3 = f["products"]
        m1, m2, m3 = f["markets"]
        g = f["growth"]
        gw = "grew" if g >= 0 else "declined"
        out = [f"Thank you, {f['ir_first']}, and good {f.get('tod', 'morning')}, everyone."]
        story = role == "story"
        if story and a == "transitory_shock":
            iss = _TRANSITORY_ISSUES[st.theme]
            out += [
                f"I want to start by addressing our {f['q_word']} quarter revenue directly, because it came in below what we "
                f"expected when we last spoke with you. Revenue of {_money(f['rev'])} grew {_pct(g)} year over year, but that "
                f"figure absorbs a headwind of approximately {_money(f['oneoff'])} from the {iss['short']}. Excluding that item, "
                f"revenue would have grown approximately {_pct(f['underlying'])}, consistent with the trajectory we have "
                "delivered over the past several years. I want to be precise about what this is and what it is not.",
                "What happened is specific and identifiable: " + iss["what"].format(**self._issue_args(f)) + ". "
                "We saw it building late in the quarter, and in hindsight we should have flagged it sooner. That is on us.",
                "What it is not is a change in end demand, a loss of share, or a pricing problem. "
                + iss["evidence"].format(**self._issue_args(f))[0].upper() + iss["evidence"].format(**self._issue_args(f))[1:]
                + f". Price realization was positive {_pct(f['price_real'])} in the quarter, and we did not lose a single "
                f"top-{f['top_n']} customer.",
                f"The leading indicators support that view. Our {f['kpi']} ended the quarter at {_money(f['backlog'])}, up "
                f"{_pct(f['bl_g'])} year over year, and book-to-bill was {f['btb']:.2f}. In the first {f['weeks_in']} weeks of "
                f"the {f['nq_word']} quarter, orders are running up {_pct(f['qtd'], 0)} year over year. "
                + iss["normal"].format(**self._issue_args(f))[0].upper() + iss["normal"].format(**self._issue_args(f))[1:] + ".",
                f"Our strategy has not changed. We continue to invest in our {p1} and {p2} franchises, our win rate on new "
                f"programs was {f['win']}% in the quarter, and we were awarded {f['awards']} new programs, several of them "
                f"in {m1}. Free cash flow over the trailing twelve months was {_money(f['fcf_ttm'])}, which gives us "
                "considerable flexibility.",
            ]
        elif story and a == "value_trap":
            pr = _VALUE_TRAP_PROBLEMS[st.theme]
            seg = p1
            out += [
                f"Our {f['q_word']} quarter results fell short of our expectations, and I want to acknowledge that up front. "
                f"Revenue of {_money(f['rev'])} {gw} {_pct(g)} year over year, below the outlook we provided in "
                f"{_MONTHS[(f['rd'].month - 4) % 12]}, and operating margin of {_pct(f['om'])} was down "
                f"{_bps(f['om'] - f['om_py'])} from a year ago.",
                pr["ceo"].format(segment=seg) + " We have been here before in prior cycles, and we believe the "
                "steps we are taking will position us well, but I want to be realistic that the environment is likely "
                "to remain challenging for several quarters.",
                f"In response, we are realigning sales coverage toward our largest accounts, accelerating cost actions that we "
                f"expect to deliver about {_money(f['rev'] * 0.03 * 4)} of annualized savings, and revisiting our pricing "
                f"architecture in {seg}. We are also sharpening the value proposition of our {p2} offering, where customer "
                "feedback remains positive.",
                "Given the uncertainty, we are no longer reaffirming the medium-term financial framework we outlined at our "
                "last investor day. We will provide an update when we have better visibility. We continue to believe our "
                f"long-term opportunity in {m1} and {m2} is significant, and our balance sheet gives us time to execute.",
            ]
        elif story and a == "guidance_reset":
            out += [
                f"{f['company']} delivered a strong {f['q_word']} quarter. Revenue of {_money(f['rev'])} grew {_pct(g)} year "
                f"over year and came in above the high end of our guidance range, and earnings per share of ${f['eps']:.2f} "
                f"were ahead of consensus. Operating margin was {_pct(f['om'])}, and we generated {_money(f['fcf_q'])} of free "
                "cash flow in the quarter.",
                f"Demand indicators remain healthy. Our pipeline is up {_pct(f['pipe'], 0)} year over year, our {f['kpi']} "
                f"grew {_pct(f['bl_g'])}, and win rates in {m1} and {m2} were stable to slightly higher. We are not seeing "
                "cancellations or pushouts beyond normal levels.",
                f"At the same time, {f['cfo_first']} will walk you through an outlook for the {f['nq_word']} quarter that is "
                "deliberately conservative. We would rather set expectations we are confident we can beat than stretch for "
                "a number. To be clear, the outlook reflects how we chose to plan, not a change in what we are seeing "
                "from customers.",
                f"Reflecting that confidence, our board has approved a new {_money(f['buyback'])} share repurchase "
                f"authorization. With {_money(f['cash'])} of cash against {_money(f['debt'])} of debt, we can fund it entirely "
                "from the balance sheet and free cash flow while continuing to invest in "
                f"our {p1} and {p2} roadmaps.",
            ]
        elif story and a == "sector_contagion":
            sc = f["scen"]
            out += [
                "Before I discuss the quarter, I want to address the news that has weighed on our industry over the past "
                f"several weeks: {sc['event']}. We understand why investors reacted, and I want to give you the facts about "
                f"our exposure. Sales tied to {sc['topic']} represented less than {f['exposure']}% of our revenue over the "
                "last twelve months, and the portion directly affected is smaller still.",
                f"Our revenue base is concentrated in {m1} and {m2}, which are not affected by {sc['affected']}, and our "
                f"{f['kpi']} of {_money(f['backlog'])} is up {_pct(f['bl_g'])} year over year. In the weeks since the "
                f"headline, orders have been up {_pct(f['qtd'], 0)} year over year and we have not seen a single "
                "cancellation linked to it.",
                f"Turning to the quarter itself, revenue of {_money(f['rev'])} grew {_pct(g)} year over year, operating "
                f"margin was {_pct(f['om'])}, and free cash flow over the trailing twelve months was "
                f"{_money(f['fcf_ttm'])}. Execution across {p1} and {p2} was strong.",
                f"We will keep monitoring developments closely. Our business model and our exposure are different from those "
                f"of {sc['peer']} and others that have cut their outlooks, and we will continue to be transparent with you "
                "about what we are seeing.",
            ]
        elif (a == "momentum_leader") or (role in ("pre", "earlier") and a != "normal"):
            out += [
                f"{f['company']} delivered another excellent quarter. Revenue grew {_pct(g)} to {_money(f['rev'])}, operating "
                f"margin was {_pct(f['om'])}, {self._margin_phrase(f['om'] - f['om_py'])}, and demand across {m1} and {m2} "
                "remained strong.",
                f"Our {p1} business continues to lead, with revenue up {_pct(g + f['seg_g'][0] + 0.02)} year over year as "
                f"{f['customers']} standardize on our platform. {p2[0].upper() + p2[1:]} grew {_pct(abs(g + f['seg_g'][1]))}, "
                f"and {p3} grew {_pct(abs(g + f['seg_g'][2]))}. Our {f['kpi']} ended the quarter at {_money(f['backlog'])}, "
                f"up {_pct(f['bl_g'])} year over year.",
                f"We continue to invest ahead of demand: we are expanding capacity for {p1}, adding sales coverage in "
                f"{m3}, and increasing R&D on the next generation of {p2}. These investments are fully funded by free cash "
                f"flow, which was {_money(f['fcf_ttm'])} over the trailing twelve months.",
            ]
            if role == "pre" and a == "value_trap":
                out.append(_VALUE_TRAP_PROBLEMS[st.theme]["hint"].format(segment=p1))
            if a == "momentum_leader" and story:
                out.append(f"Given the strength we are seeing, we are raising our outlook, and {f['cfo_first']} will take you "
                           "through the details.")
        else:
            tone = "solid" if g > 0.04 else ("steady" if g > 0 else "challenging")
            out += [
                f"{f['company']} delivered a {tone} {f['q_word']} quarter. Revenue of {_money(f['rev'])} {gw} {_pct(g)} year "
                f"over year, and operating margin was {_pct(f['om'])}. We continued to execute on our priorities: growing "
                f"our {p1} business, improving productivity, and disciplined capital allocation.",
                f"By end market, {m1} was {'our strongest' if g > 0 else 'softer than we expected'}, {m2} was "
                f"{'stable' if g > -0.02 else 'weaker'}, and {m3} was mixed. We are watching customer inventories and "
                "order patterns closely, and we are managing costs accordingly.",
            ]
        if st is not None:
            out += self._ceo_extras(f)
        out.append(f"With that, I'll turn it over to {f['cfo_first']} to walk through the financials in more detail.")
        return out

    def _ceo_extras(self, f: dict[str, Any]) -> list[str]:
        """Longer-form prepared remarks used for planted names (end markets, innovation, operations, people)."""
        a, role, st = f["arch"], f["role"], f["st"]
        p1, p2, _ = f["products"]
        m1, m2, m3 = f["markets"]
        mk = [f["growth"] + d for d in f["seg_g"]]
        word = lambda x: "grew" if x >= 0 else "declined"  # noqa: E731
        out = [f"Looking at our end markets, {m1} {word(mk[0])} {_pct(mk[0], 0)} year over year, {m2} {word(mk[1])} "
               f"{_pct(mk[1], 0)}, and {m3} {word(mk[2])} {_pct(mk[2], 0)}. Our {f['customers']} continue to consolidate "
               "spending with suppliers that can support them across multiple sites, which plays to our scale."]
        if role == "story" and a == "value_trap":
            out.append(f"On innovation, we continue to invest in our {p2} roadmap, although the timing of some launches has "
                       "moved to the right as we incorporate customer feedback. We launched "
                       f"{f['n_launch']} new products in the quarter, and products introduced over the last three years were "
                       f"{f['vit']}% of revenue.")
        else:
            out.append(f"On innovation, we launched {f['n_launch']} new products in the quarter, including an expanded {p2} "
                       f"line. Products introduced over the last three years represented {f['vit']}% of revenue, up from "
                       f"{f['vit_py']}% a year ago, and our engineering pipeline has never been fuller.")
        if not (role == "story" and a == "transitory_shock" and st.theme in ("supply_chain", "erp")):
            out.append(f"Operationally, our productivity program delivered about {_money(f['savings'])} of savings in the "
                       f"quarter, on-time delivery was {f['otd']:.1f}%, and we continued to shorten lead times in {p1}.")
        if not (role == "story" and a == "value_trap"):
            out.append(f"Our strategy rests on three priorities. First, extend our leadership in {p1} through engineering and "
                       f"service intensity. Second, scale {p2} into a larger, more recurring revenue stream. Third, keep "
                       "compounding free cash flow and deploy it with discipline. Over the last three years these priorities "
                       "have driven above-market growth and margin expansion, and they remain the right ones.")
        else:
            out.append("We are conducting a thorough review of our portfolio, pricing and cost structure. We want to come out "
                       "of this period with a business that is more focused and more resilient, and we will share the "
                       "conclusions when the work is complete.")
        out.append(f"Finally, I want to thank our roughly {int(round(f['rev_ttm'] / f['emp_k'], -2)):,} employees for their "
                   "focus and execution, and our customers and partners for their continued trust.")
        return out

    @staticmethod
    def _margin_phrase(d: float) -> str:
        if d >= 0.0005:
            return f"up {_bps(d)} from a year ago"
        if d > -0.006:
            return "roughly in line with a year ago"
        return f"down {_bps(d)} from a year ago as we invested ahead of demand"

    def _issue_args(self, f: dict[str, Any]) -> dict[str, Any]:
        return dict(n_cust=f["n_cust"], customers=f["customers"], sellthrough=f["sellthrough"], weeks=f["wk"],
                    norm_lo=f["wk"] + 2, norm_hi=f["wk"] + 4, next_q=f["nq_word"], currency=f["currency"],
                    region=f["region"], component=f["component"], oneoff=_money(f.get("oneoff", 0.0)),
                    month=_MONTHS[f["pe"].month - 1])

    def _cfo_remarks(self, f: dict[str, Any], rng: np.random.Generator) -> list[str]:
        a, role, st = f["arch"], f["role"], f["st"]
        story = role == "story"
        g = f["growth"]
        out = [
            f"Thank you, {f['ceo_first']}. Revenue for the {f['q_word']} quarter was {_money(f['rev'])}, "
            f"{_updown(g)} {_pct(g)} compared with {_money(f['rev_py'])} in the {f['q_word']} quarter of last year."
        ]
        if story and a == "transitory_shock":
            pts = st.targets["oneoff_pts"]
            out.append(f"The {_TRANSITORY_ISSUES[st.theme]['short']} reduced revenue by approximately {_money(f['oneoff'])}, "
                       f"or about {pts * 100:.1f} points of growth. Excluding it, revenue grew approximately "
                       f"{_pct(f['underlying'])}, and price contributed {_pct(f['price_real'])}.")
        elif story and a == "value_trap":
            out.append("The year-over-year change reflected lower volumes in our core product lines and lower average "
                       "selling prices, partially offset by growth in services. We also saw longer decision cycles with "
                       "several large customers.")
        elif story and a == "guidance_reset":
            beat = f["rev"] - f["prior_mid"]
            out.append(f"That was {_money(abs(beat))} {'above' if beat >= 0 else 'below'} the midpoint of the outlook we gave "
                       f"last quarter, with upside across {f['markets'][0]} and {f['markets'][1]}.")
        elif story and a == "sector_contagion":
            out.append(f"We saw no measurable change in order patterns after the industry headline; revenue from "
                       f"{f['scen']['topic']} was again less than {f['exposure']}% of the total.")
        long_form = st is not None
        if long_form:
            p1, p2, p3 = f["products"]
            out.append(f"By segment, {p1} revenue was {_money(f['seg_a'])}, {_updown(f['seg_a_g'])} {_pct(f['seg_a_g'])} year "
                       f"over year, and {p2} and {p3} revenue was {_money(f['seg_b'])}, {_updown(f['seg_b_g'])} "
                       f"{_pct(f['seg_b_g'])}.")
        dgm = f["gm"] - f["gm_py"]
        if story and a == "transitory_shock":
            why = "reflecting lower absorption on the reduced volume, which we expect to reverse as volumes normalize"
        elif story and a == "value_trap":
            why = "reflecting pricing actions, unfavorable mix and lower absorption"
        else:
            why = "driven by favorable mix and productivity" if dgm >= 0 else "reflecting mix and input-cost inflation"
        out.append(f"Gross margin was {_pct(f['gm'])}, {_updown(dgm)} {_bps(dgm)} from a year ago, {why}. Operating income "
                   f"was {_money(f['oi'])}, or {_pct(f['om'])} of revenue, compared with {_pct(f['om_py'])} in the prior-year "
                   "quarter.")
        if long_form and f["opex"] > 0:
            tail = (" We have begun cost actions and expect operating expenses to decline sequentially." if story and a == "value_trap"
                    else " We continue to fund our roadmap while holding the growth of operating expenses below revenue growth.")
            out.append(f"Operating expenses were {_money(f['opex'])}, or {_pct(f['opex'] / f['rev'])} of revenue. Research and "
                       f"development was {_money(f['rnd'])} and selling, general and administrative expense was "
                       f"{_money(f['sga'])}.{tail}")
        eps, eps_py = f["eps"], f["eps_py"]
        e1 = f"${eps:.2f}" if eps >= 0 else f"a loss of ${abs(eps):.2f}"
        e0 = f"${eps_py:.2f}" if eps_py >= 0 else f"a loss of ${abs(eps_py):.2f}"
        out.append(f"Diluted earnings per share were {e1}, compared with {e0} a year ago.")
        if long_form:
            dsh = f["shares"] / f["shares_py"] - 1
            tax = " Our effective tax rate was 21.0%." if f["pretax"] > 0 else ""
            out.append(f"Net interest expense was {_money(f['interest_q'])}.{tax} The diluted share count was "
                       f"{f['shares'] / 1e6:.1f} million, {_updown(dsh)} {_pct(dsh)} from a year ago.")
        fq_ = f["fcf_q"]
        fcf_txt = f"free cash flow of {_money(fq_)}" if fq_ >= 0 else f"a free cash outflow of {_money(fq_)}"
        ttm_txt = (f"we generated {_money(f['fcf_ttm'])} of free cash flow" if f["fcf_ttm"] >= 0
                   else f"free cash flow was an outflow of {_money(f['fcf_ttm'])}")
        out.append(f"Cash flow from operations was {_money(f['cfo'])} in the quarter and capital expenditures were "
                   f"{_money(f['capex'])}, resulting in {fcf_txt}. Over the trailing twelve months, {ttm_txt}.")
        if long_form and f["bought"] > 0:
            out.append(f"During the quarter we repurchased approximately {f['bought'] / 1e6:.1f} million shares for about "
                       f"{_money(f['buyback_q'])}.")
        if long_form and math.isfinite(f["conv"]):
            out.append(f"Free cash flow conversion over the trailing twelve months was {f['conv'] * 100:.0f}% of net income.")
        if f["net_cash"] >= 0:
            bs = f", a net cash position of {_money(f['net_cash'])}"
        elif math.isfinite(f["lev"]):
            bs = f", or net leverage of {f['lev']:.1f} times trailing EBITDA"
        else:
            bs = ""
        out.append(f"We ended the quarter with {_money(f['cash'])} of cash and equivalents and {_money(f['debt'])} of total "
                   f"debt{bs}.")
        if story and a == "guidance_reset":
            out.append(f"Our board has authorized a new {_money(f['buyback'])} share repurchase program, roughly "
                       f"{_pct(f['buyback_pct'], 0)} of our market capitalization.")
        out.append(f"For the {f['nq_word']} quarter, we expect revenue in the range of {_money(f['guide_lo'])} to "
                   f"{_money(f['guide_hi'])}.")
        if long_form:
            out.append("The outlook assumes foreign exchange rates at current levels, an effective tax rate of approximately "
                       f"21%, net interest expense of roughly {_money(f['interest_q'])} and a diluted share count of about "
                       f"{f['shares'] / 1e6:.1f} million.")
        if story and a == "guidance_reset":
            assume = _GUIDANCE_REASONS[st.theme]["assume"].format(next_q=f["nq_word"], haircut=f["haircut"], fx_bp=f["fx_bp"],
                                                                  customers=f["customers"])
            out.append("Let me be explicit about how we built this outlook, because it is deliberately conservative. We "
                       f"have embedded a meaningful degree of conservatism: {assume}. At the midpoint, the outlook is about "
                       f"{_pct(f['guide_cut'], 0)} below where consensus stood before today, and we are comfortable with that. "
                       "If order rates simply hold at current levels, we would expect to finish above the high end of the range.")
        elif story and a == "value_trap":
            imp = f["guide_mid"] / f["next_py"] - 1
            out.append(f"At the midpoint, that implies revenue {_updown(imp, 'growth', 'decline')} of about {_pct(imp)} year over "
                       "year. The outlook reflects the competitive and pricing dynamics we have discussed, and we have taken "
                       "what we believe is a prudent view of the second half.")
        elif story and a == "transitory_shock":
            out.append(f"Our exit rate improved through the quarter: in the final {f['wk'] - 2} weeks, shipments were running at "
                       f"about {f['exit_rate']}% of normal levels, and they have continued to improve since quarter end.")
            out.append(f"The outlook assumes the {_TRANSITORY_ISSUES[st.theme]['short']} is substantially behind us by the end "
                       f"of the {f['nq_word']} quarter, and it does not assume any recovery of the {_money(f['oneoff'])} we "
                       "lost this quarter, even though we expect much of it to come back over time.")
        elif a == "momentum_leader" and story:
            out.append("Given the strength in orders, we are also raising our full-year revenue and operating margin outlook.")
        out.append("With that, operator, please open the line for questions.")
        return out

    def _qa_topics(self, f: dict[str, Any], rng: np.random.Generator) -> list[tuple[str, str, str]]:
        """(question, answering role, answer) tuples, most relevant first, lightly shuffled."""
        a, role, st = f["arch"], f["role"], f["st"]
        p1, p2, _ = f["products"]
        m1, m2, _ = f["markets"]
        story = role == "story"
        nq = f["nq_word"]
        topics: list[tuple[str, str, str]] = []
        if story and a == "transitory_shock":
            iss = _TRANSITORY_ISSUES[st.theme]
            ev = iss["evidence"].format(**self._issue_args(f))
            gap = f["prior_mid"] - f["rev"]
            if gap > 0:
                bridge = (f"We came in about {_money(gap)} below the midpoint of our prior range. The {iss['short']} "
                          f"accounts for approximately {_money(f['oneoff'])}, so {'more than all' if f['oneoff'] >= gap else 'the large majority'} "
                          "of the gap. Everything else, in aggregate, was close to plan")
            else:
                bridge = (f"Even with a headwind of approximately {_money(f['oneoff'])} from the {iss['short']}, revenue was "
                          f"close to the midpoint of our prior range, so the rest of the business ran ahead of plan")
            topics += [
                (f"{f['ceo_first']}, I appreciate the detail, but I want to push back a little. Every company that misses "
                 f"calls the problem temporary. What specifically gives you confidence that the {iss['short']} is a one-time "
                 f"event and not the leading edge of a broader slowdown in {m1}?", "CEO",
                 f"That's a fair challenge, and it's exactly the right question. Three things. First, {ev}. Second, our "
                 f"{f['kpi']} ended the quarter at {_money(f['backlog'])}, up {_pct(f['bl_g'])} year over year, with a "
                 f"book-to-bill of {f['btb']:.2f}. If end demand were rolling over, you would see it there first, and you "
                 f"don't. Third, in the first {f['weeks_in']} weeks of the {nq} quarter, orders are up {_pct(f['qtd'], 0)} "
                 "year over year. So the data we have points to a timing issue, not a demand issue. We will let the next "
                 "couple of quarters prove it."),
                ("Can you help us bridge the shortfall? How much of the gap versus your prior outlook was this item versus "
                 "everything else in the business?", "CFO",
                 f"Sure. {bridge}: price realization was positive {_pct(f['price_real'])} and our service revenue grew "
                 f"{_pct(f['svc'])}. We have also not changed any assumptions about underlying demand in the outlook."),
                (f"On gross margin, you were {_updown(f['gm'] - f['gm_py'])} {_bps(f['gm'] - f['gm_py'])}. Is any of that "
                 "price, or is it all volume?", "CFO",
                 "It's volume. Price-cost was positive in the quarter. The decline is under-absorption from running our "
                 "plants below the rate we had planned, and it reverses as volumes normalize. We have not seen any change "
                 "in the pricing environment, and our quoting activity is consistent with prior periods."),
                ("Are you seeing any competitive change? Is there any sign that customers are using this pause to "
                 "dual-source or shift share?", "CEO",
                 f"No. We have spoken directly with all of our top {f['top_n']} customers since the quarter ended. Our win rate "
                 f"on new programs was {f['win']}% in the quarter, consistent with the last two years, and we were awarded "
                 f"{f['awards']} new {p1} programs. Customers tell us {iss['side']}, and our share of their purchases was stable."),
                ("Given where the stock is trading, how are you thinking about capital deployment?", "CFO",
                 f"We have plenty of flexibility. We generated {_money(f['fcf_ttm'])} of free cash flow over the last twelve "
                 f"months and ended the quarter with {_money(f['cash'])} of cash. Our priorities are unchanged: invest in the "
                 "business, pursue bolt-on acquisitions, and return excess cash. We will be opportunistic with repurchases."),
            ]
        elif story and a == "value_trap":
            pr = _VALUE_TRAP_PROBLEMS[st.theme]
            metric = {"share_loss": "competitive win rate", "churn": "gross retention rate",
                      "pricing": "price realization", "transition": "number of customer qualifications on the new platform"}[st.theme]
            topics += [
                (f"Can you quantify how much of the shortfall came from {pr['short']} versus broader market softness?", "CEO",
                 "I don't think it's productive to parse it at that level of detail. There are a number of factors at "
                 "play, including the macro environment, customer budget cycles and, yes, competitive dynamics in certain "
                 "pockets of the market. What I would say is that we remain confident in the value we deliver, and we are "
                 "taking actions to sharpen our go-to-market."),
                (f"Gross margin was down {_bps(f['gm'] - f['gm_py'])} year over year. How much of that is price?", "CFO",
                 "There are a lot of moving pieces in gross margin this quarter: mix, absorption, some targeted commercial "
                 "actions, freight. I'd be careful about isolating any one of them. We will give you more color as the year "
                 "progresses."),
                (f"Can you give us the {metric} for the quarter? You have disclosed it in the past.", "CFO",
                 "We have decided to move to disclosing that metric on an annual basis. We think the annual view is a better "
                 "reflection of the health of the business, and we will update it at our next investor day."),
                ("Last quarter you described the competitive environment as stable and reiterated the full-year outlook. "
                 "What changed in ninety days, and why should we have confidence in the new outlook?", "CEO",
                 "Look, the environment evolved faster than we anticipated, and we have taken what we believe is a prudent "
                 "approach to the outlook. I'm not going to speculate on individual competitors. We are focused on the "
                 "things we can control."),
                (f"Do you still stand behind the {f['lt_margin']}% operating margin target from the investor day?", "CFO",
                 "It's premature to revisit long-term targets in the middle of the year. We are evaluating all elements of "
                 "the plan and we will provide an update when we have more clarity."),
                ("Have you had to match competitors on price at renewal, and if so, is that fully reflected in the "
                 "outlook?", "CEO",
                 "We're being thoughtful and disciplined. In some cases we have chosen to walk away, and in others we have "
                 "made targeted investments to protect strategic relationships. I don't want to get into the specifics of "
                 "individual negotiations."),
            ]
        elif story and a == "guidance_reset":
            assume = _GUIDANCE_REASONS[st.theme]["assume"].format(next_q=nq, haircut=f["haircut"], fx_bp=f["fx_bp"],
                                                                  customers=f["customers"])
            topics += [
                ("You beat on revenue and earnings, yet the outlook is well below where the Street was. Help us understand "
                 "the disconnect.", "CFO",
                 "Sure. I want to emphasize that nothing in the outlook reflects deterioration in the business. We have "
                 f"embedded a meaningful degree of conservatism: {assume}. Put simply, we built an outlook that we expect to "
                 "beat. If order rates just hold where they are, we would land above the high end of the range."),
                ("Has anything changed in customer behavior or in the pipeline over the last few months?", "CEO",
                 f"No. The pipeline is up {_pct(f['pipe'], 0)} year over year, win rates are stable, and our {f['kpi']} grew "
                 f"{_pct(f['bl_g'])}. Customers in {m1} and {m2} are telling us the same things they told us six months ago."),
                ("Can you talk about the size and pace of the buyback?", "CFO",
                 f"The new {_money(f['buyback'])} authorization is roughly {_pct(f['buyback_pct'], 0)} of our current market "
                 f"capitalization. With {_money(f['cash'])} of cash against {_money(f['debt'])} of debt, we can fund it "
                 "entirely from the balance sheet and free cash flow, and we intend to be active, particularly at current "
                 "levels."),
                ("What would it take to get back to where consensus was before today?", "CFO",
                 f"Mostly normal execution. The difference between our midpoint and prior consensus is roughly "
                 f"{_money(f['consensus_gap'])} of revenue for the quarter. A return to normal close rates, or simply a "
                 f"stabilization in {m1}, would close most of that gap."),
                ("How should we think about margins inside the outlook?", "CFO",
                 f"We are holding operating margin roughly flat with the {f['q_word']} quarter at the midpoint, which again "
                 "is conservative. We are not cutting the investments in our roadmap to protect a quarterly number."),
            ]
        elif story and a == "sector_contagion":
            sc = f["scen"]
            topics += [
                (f"Obviously the group has been under pressure since the news about {sc['topic']}. Can you size your "
                 "exposure precisely?", "CEO",
                 f"Yes. Sales tied to {sc['topic']} were less than {f['exposure']}% of our revenue over the last twelve "
                 f"months, and the business directly affected by {sc['affected']} is a subset of that. Our largest end markets are {m1} "
                 f"and {m2}, which are not part of the discussion at all."),
                ("What about second-order effects through your customers?", "CFO",
                 f"We have mapped our top {f['top_n']} customers against {sc['affected']}. Even under conservative assumptions "
                 f"the indirect exposure is small, and none of those customers has changed its forecast to us."),
                ("Have you seen any change in order patterns since the headline?", "CEO",
                 f"No. Orders in the weeks since have been up {_pct(f['qtd'], 0)} year over year, and we have not had a "
                 "single cancellation or push-out that customers attributed to it."),
                (f"{sc['peer']} cut its outlook. Why is your business different?", "CEO",
                 f"Their mix is very different from ours. By our estimate roughly {f['peer_exp']}% of their revenue is tied to "
                 f"{sc['topic']}; for us it is less than {f['exposure']}%. We understand why the group traded together, but "
                 "the fundamentals are not the same."),
                ("Would you consider stepping up the buyback given the share price?", "CFO",
                 f"We have the flexibility to. With {_money(f['fcf_ttm'])} of trailing free cash flow and "
                 f"{_money(f['cash'])} of cash, we can be opportunistic, and we were active in the market after the headline."),
            ]
        elif a == "momentum_leader" or (role in ("pre", "earlier") and a != "normal"):
            topics += [
                ("Congratulations on the quarter. How durable is this growth rate as the comparisons get harder?", "CEO",
                 f"We feel good about it. Our {f['kpi']} is up {_pct(f['bl_g'])}, the pipeline is up {_pct(f['pipe'], 0)}, "
                 f"and we are winning new {p1} programs at a higher rate than a year ago. Comparisons will get tougher, but "
                 "the underlying demand is broad-based."),
                ("Can you talk about capacity and the investment required to keep up with demand?", "CFO",
                 f"We are investing roughly {_pct(f['capex'] / f['rev'])} of revenue in capital expenditures this year, mostly "
                 f"for {p1} capacity, and it is fully funded from free cash flow."),
                ("How is pricing holding up?", "CEO",
                 f"Very well. Price realization was positive {_pct(f['price_real'])} and we have not seen any change in the "
                 "competitive environment."),
                ("What is embedded in the outlook for the next quarter?", "CFO",
                 f"The outlook of {_money(f['guide_lo'])} to {_money(f['guide_hi'])} assumes order rates consistent with what "
                 "we saw in the quarter, normal seasonality, and no change in pricing."),
            ]
            if role == "pre" and a == "value_trap":
                topics.insert(1, ("Are you seeing any change in the competitive landscape, particularly at the lower end of "
                                  "the market?", "CEO",
                                  "We see a new entrant in some deals at the low end. We don't think it's a significant "
                                  "factor, and our installed base and service model remain strong differentiators."))
        else:
            topics += [
                (f"How would you characterize the demand environment across {m1} and {m2}?", "CEO",
                 f"Broadly {'healthy' if f['growth'] > 0.03 else 'mixed'}. {m1[0].upper() + m1[1:]} was "
                 f"{'solid' if f['growth'] > 0 else 'softer'}, and {m2} was roughly in line with our expectations. We are "
                 "watching order patterns closely."),
                ("Can you unpack the margin performance in the quarter?", "CFO",
                 f"Gross margin was {_pct(f['gm'])}. The main drivers were mix and productivity on one side and input costs "
                 "on the other. We expect a similar range for the next quarter."),
                ("How are you thinking about capital allocation from here?", "CFO",
                 "Our priorities are unchanged: invest in the business, keep the balance sheet strong, and return excess cash "
                 "to shareholders over time."),
                ("Any change to how you are thinking about the full year?", "CEO",
                 "Not materially. We are executing on our plan and we will update you as the year progresses."),
            ]
        if len(topics) > 2:  # keep the lead question, lightly shuffle the rest
            rest = [topics[int(j) + 1] for j in rng.permutation(len(topics) - 1)]
            topics = [topics[0]] + rest
        if st is not None:
            gen = self._generic_topics(f)
            topics += [gen[int(j)] for j in rng.permutation(len(gen))]
        return topics

    def _followups(self, f: dict[str, Any], rng: np.random.Generator) -> list[tuple[str, str, str]]:
        a, role = f["arch"], f["role"]
        intl = int(rng.integers(18, 42))
        conv12 = int(rng.integers(55, 80))
        out = [
            (f"And a quick follow-up on cadence: how should we think about linearity through the {f['nq_word']} quarter?", "CFO",
             "We expect a fairly normal cadence, with the final month of the quarter the largest, consistent with prior "
             "years. Nothing unusual is embedded in the range."),
            (f"A follow-up on the {f['kpi']}: how much of it converts to revenue over the next twelve months?", "CFO",
             f"Roughly {conv12}% of it is scheduled to convert within twelve months, which is in line with history."),
            ("And can you remind us how much of the business is outside the U.S.?", "CFO",
             f"International markets were about {intl}% of revenue in the quarter, split roughly evenly between Europe and "
             "the rest of the world."),
            ("One more on the balance sheet: any change to how you think about leverage?", "CFO",
             "No change. We are comfortable operating below two times net leverage through the cycle and we would go above "
             "that only temporarily, for the right acquisition."),
        ]
        if role == "story" and a == "value_trap":
            out[0] = (f"And just to follow up: is the {f['nq_word']} quarter the bottom?", "CEO",
                      "I'm not going to call a bottom. We are focused on executing the plan and we will update you as we go.")
        return out

    def _generic_topics(self, f: dict[str, Any]) -> list[tuple[str, str, str]]:
        p1, p2, p3 = f["products"]
        conv = (f"or {f['conv'] * 100:.0f}% of net income" if math.isfinite(f["conv"]) else "despite the swing in net income")
        return [
            (f"Could you give a bit more color on the {p2} and {p3} business?", "CEO",
             f"Sure. Revenue there was {_money(f['seg_b'])}, {_updown(f['seg_b_g'])} {_pct(f['seg_b_g'])} year over year. "
             f"The mix is shifting toward recurring consumables and service, which carry above-average margins, and we "
             f"continue to see cross-selling opportunities with our {p1} customers."),
            ("Can you talk about the operating expense trajectory from here?", "CFO",
             f"Research and development was {_pct(f['rnd'] / f['rev'])} of revenue in the quarter and SG&A was "
             f"{_pct(f['sga'] / f['rev'])}. We manage the two together: we protect engineering investment and look for "
             "efficiency in the go-to-market and back-office functions, so we expect operating expenses to grow more slowly "
             "than revenue over time."),
            ("How sustainable is the free cash flow conversion?", "CFO",
             f"Trailing twelve-month free cash flow was {_money(f['fcf_ttm'])}, {conv}. Capital expenditures are running at "
             f"about {_pct(f['capex'] / f['rev'])} of revenue, and working capital has been a modest use of cash. We think "
             "the model supports consistent conversion through the cycle."),
            ("How active is the acquisition pipeline, and what return thresholds do you use?", "CEO",
             f"The pipeline is active, mostly bolt-on technologies in {p2} and channel access in new geographies. We are "
             "disciplined: we look for returns above our cost of capital by year three, and we walk away when the price is "
             "not right. Organic investment remains the first call on capital."),
        ]

    def _closing(self, f: dict[str, Any]) -> str:
        a, role = f["arch"], f["role"]
        if role == "story" and a == "transitory_shock":
            return ("Thank you all for your questions. We understand that this quarter raised questions, and we intend to "
                    "answer them with results. The underlying business is healthy, and we look forward to updating you next "
                    "quarter.")
        if role == "story" and a == "value_trap":
            return ("Thank you for joining us. We are taking the necessary actions, and we look forward to updating you on "
                    "our progress.")
        return "Thank you for joining us today and for your continued interest in the company. We look forward to speaking with you next quarter."

    # ------------------------------------------------------------------ news / research / filings

    def _move(self, i: int, t: int) -> tuple[float, float, float]:
        """(day return, close, volume multiple vs the prior 60 sessions) at session t."""
        c = self._close[:, i]
        vol = self._px["volume"].iloc[:, i].to_numpy()
        prev = vol[max(0, t - 60): t]
        prev = prev[np.isfinite(prev)]
        vm = float(vol[t] / prev.mean()) if prev.size and prev.mean() > 0 else float("nan")
        return float(c[t] / c[t - 1] - 1), float(c[t]), vm

    def _news(self, f: dict[str, Any], when: datetime, slug: str, title: str, text: str, category: str,
              kind: DocumentKind = DocumentKind.NEWS, source: str = _NEWSWIRE, extra: dict[str, str] | None = None) -> Document:
        tick = f["ticker"]
        prefix = {DocumentKind.NEWS: "NW", DocumentKind.RESEARCH: "RS", DocumentKind.FILING: "FL"}[kind]
        doc_id = f"SYN-{prefix}-{tick}-{when.strftime('%Y%m%d')}-{slug}"
        meta = {"category": category}
        meta.update(extra or {})
        return Document(doc_id=doc_id, ticker=tick, kind=kind, title=title, published_at=when, source=source,
                        url=f"synthetic://{kind.value}/{doc_id}", text=text, metadata=meta)

    def _press_release(self, f: dict[str, Any]) -> Document:
        a, role, st = f["arch"], f["role"], f["st"]
        g = f["growth"]
        eps = f"${f['eps']:.2f}" if f["eps"] >= 0 else f"a loss of ${abs(f['eps']):.2f}"
        extra = ""
        if role == "story" and a == "transitory_shock":
            extra = (f" Results were affected by the {_TRANSITORY_ISSUES[st.theme]['short']}, which reduced revenue by "
                     f"approximately {_money(f['oneoff'])}; orders and {f['kpi']} grew year over year.")
        elif role == "story" and a == "value_trap":
            extra = " Revenue was below the company's prior outlook, reflecting a more competitive environment and lower pricing."
        elif role == "story" and a == "guidance_reset":
            extra = f" The company also announced a new {_money(f['buyback'])} share repurchase authorization."
        elif role == "story" and a == "sector_contagion":
            extra = f" The company said sales tied to {f['scen']['topic']} were less than {f['exposure']}% of revenue."
        elif role == "story" and a == "momentum_leader":
            extra = " The company raised its full-year outlook."
        title = f"{f['company']} Reports {_ORDINAL[f['fq']].capitalize()} Quarter Fiscal {f['fy']} Results"
        text = (f"{f['company']} ({f['exch']}: {f['ticker']}) today reported results for its {f['q_word']} quarter of fiscal "
                f"{f['fy']}, ended {_long_date(f['pe'])}. Revenue was {_money(f['rev'])}, {_updown(g)} {_pct(g)} year over "
                f"year. Operating margin was {_pct(f['om'])} and diluted earnings per share were {eps}. Free cash flow for the "
                f"trailing twelve months was {_money(f['fcf_ttm'])}.{extra} For the {f['nq_word']} quarter, the company expects "
                f"revenue of {_money(f['guide_lo'])} to {_money(f['guide_hi'])}.")
        return self._news(f, datetime.combine(f["rd"], time(7, 0)), "PR", title, text, "earnings")

    def _reaction_news(self, f: dict[str, Any], s: int) -> Document:
        r, c, vm = self._move(f["i"], s)
        verb = ("plunge" if r <= -0.10 else "fall" if r <= -0.03 else "slip" if r < 0 else
                "edge up" if r < 0.03 else "rise" if r < 0.10 else "jump")
        a, role, st = f["arch"], f["role"], f["st"]
        why = ""
        if role == "story" and a == "transitory_shock":
            why = (f" The company blamed the {_TRANSITORY_ISSUES[st.theme]['short']} and said orders remained healthy; several "
                   "analysts questioned how quickly revenue would recover.")
        elif role == "story" and a == "value_trap":
            why = " Analysts on the call pressed management on competitive pressure and pricing."
        elif role == "story" and a == "guidance_reset":
            why = " Results beat estimates, but the company's outlook for the next quarter came in below consensus."
        vtxt = f", on volume about {vm:.1f} times the 60-day average" if math.isfinite(vm) else ""
        title = f"{f['company']} shares {verb} {_pct(r)} after {f['q_word']}-quarter results"
        text = (f"Shares of {f['company']} ({f['ticker']}) closed {_updown(r)} {_pct(r)} at ${c:.2f} on "
                f"{_long_date(f['rd'])}{vtxt}, after the company reported {f['q_word']}-quarter revenue of {_money(f['rev'])}, "
                f"{_updown(f['growth'])} {_pct(f['growth'])} from a year earlier.{why}")
        return self._news(f, datetime.combine(f["rd"], time(16, 30)), "MOVE", title, text, "market_move")

    def _analyst_action(self, f: dict[str, Any], t: int, rng: np.random.Generator, story: bool) -> list[Document]:
        i = f["i"]
        a, st = f["arch"], f["st"]
        who, firm = self._coverage[i][int(st.params.get("broker_idx", 0)) if (st is not None and story) else int(rng.integers(8))]
        day = _py(self._dd[t])
        r, c, vm = self._move(i, t)
        est = self.get_estimates([f["ticker"]], day)
        tpm = float(est[F.TARGET_PRICE_MEAN].iloc[0])
        tp = round((tpm if math.isfinite(tpm) else c * 1.1) * rng.uniform(0.88, 1.06))
        if story and a == "transitory_shock":
            action, rating = "downgrades", "Neutral"
            body = (f"We move to Neutral from Buy. Management attributes the shortfall to the "
                    f"{_TRANSITORY_ISSUES[st.theme]['short']} and points to healthy orders, but we think visibility on the "
                    f"timing of the recovery is limited and estimates for the {f['nq_word']} quarter remain at risk. We would "
                    "revisit the stock on evidence that shipments have normalized.")
        elif story and a == "value_trap":
            action, rating = "downgrades", "Underperform"
            body = (f"We downgrade to Underperform. Our channel checks point to {_VALUE_TRAP_PROBLEMS[st.theme]['short']} "
                    "that we believe is structural rather than cyclical, and management offered little detail on the call. "
                    "We see further downside to consensus estimates and expect margins to remain under pressure.")
        elif story and a == "guidance_reset":
            action, rating = "cuts price target on", "Outperform"
            body = ("We lower our target to reflect the more conservative outlook but reiterate Outperform. The quarter was "
                    "strong, the balance sheet carries net cash, and in our view the outlook embeds unusually large cushions. "
                    "We see the buyback as a signal of management's confidence.")
        else:
            up = (a == "momentum_leader") or f["growth"] > 0.08
            action, rating = ("raises price target on", "Buy") if up else ("reiterates", ["Hold", "Neutral", "Buy"][int(rng.integers(3))])
            body = (f"We {'raise' if up else 'maintain'} our estimates after a {'strong' if up else 'mixed'} quarter. "
                    f"Revenue grew {_pct(f['growth'])} year over year and operating margin was {_pct(f['om'])}.")
        title = f"{firm} {action} {f['company']} ({rating}); price target ${tp:,.0f}"
        move = f" The shares {'fell' if r < 0 else 'rose'} {_pct(r)} to ${c:.2f}" + (f" on volume {vm:.1f} times the 60-day average." if math.isfinite(vm) else ".")
        news = self._news(f, datetime.combine(day, time(6, 45)), "RATING", title,
                          f"{firm} analyst {who} {action} {f['company']} with a {rating} rating and a ${tp:,.0f} price target. "
                          f"{body}{move}", "analyst_rating", extra={"broker": firm, "rating": rating})
        out = [news]
        if story or rng.random() < 0.5:
            note = (f"{firm} Equity Research | {f['company']} ({f['ticker']}) | Rating: {rating} | Price target: ${tp:,.0f} | "
                    f"Analyst: {who}\n\n{body} Key metrics from the latest report: revenue {_money(f['rev'])} "
                    f"({_updown(f['growth'], '+', '-')}{_pct(f['growth'])} y/y), gross margin {_pct(f['gm'])}, operating margin "
                    f"{_pct(f['om'])}, trailing free cash flow {_money(f['fcf_ttm'])}.")
            out.append(self._news(f, datetime.combine(day, time(6, 30)), "NOTE", f"{f['company']}: {rating}, PT ${tp:,.0f}",
                                  note, "research_note", kind=DocumentKind.RESEARCH, source=firm,
                                  extra={"broker": firm, "rating": rating}))
        return out

    def _corporate_news(self, f: dict[str, Any], t: int, rng: np.random.Generator) -> Document:
        day = _py(self._dd[t])
        kind = int(rng.integers(3))
        if kind == 0:
            dps = max(0.05, round(float(f["eps"]) * rng.uniform(0.25, 0.45), 2))
            title = f"{f['company']} declares quarterly dividend of ${dps:.2f} per share"
            text = (f"The board of directors of {f['company']} declared a quarterly cash dividend of ${dps:.2f} per share, "
                    "payable to shareholders of record at the end of the month.")
            cat = "dividend"
        elif kind == 1:
            conf = _BROKERS[int(rng.integers(len(_BROKERS)))]
            title = f"{f['company']} to present at the {conf} {f['sector'] or 'Growth'} Conference"
            text = (f"{f['company']} said {f['ceo']}, Chief Executive Officer, and {f['cfo_name']}, Chief Financial Officer, "
                    f"will participate in a fireside chat at the {conf} conference. A webcast will be available on the company's "
                    "investor relations website.")
            cat = "conference"
        else:
            p = f["products"][int(rng.integers(3))]
            title = f"{f['company']} expands {p} portfolio"
            text = (f"{f['company']} announced an expansion of its {p} portfolio aimed at {f['customers']} in "
                    f"{f['markets'][int(rng.integers(3))]}. The company said the new offerings are available immediately.")
            cat = "product"
        return self._news(f, datetime.combine(day, time(8, 0)), "CORP", title, text, cat)

    def _periodic_filing(self, f: dict[str, Any], rng: np.random.Generator) -> Document:
        form = "10-K" if f["fq"] == 4 else "10-Q"
        filed = _py(np.busday_offset(_d64(f["rd"]), int(rng.integers(3, 9)) if form == "10-Q" else int(rng.integers(8, 15)),
                                     roll="forward"))
        a, role, st = f["arch"], f["role"], f["st"]
        g = f["growth"]
        if role == "story" and a == "transitory_shock":
            driver = (f"The change primarily reflected the {_TRANSITORY_ISSUES[st.theme]['short']}, which reduced net sales by "
                      f"approximately {_money(f['oneoff'])}. We believe this item is not indicative of underlying demand.")
        elif role == "story" and a == "value_trap":
            driver = "The change primarily reflected " + _VALUE_TRAP_PROBLEMS[st.theme]["mdna"].format(segment=f["products"][0]) + "."
        elif role == "story" and a == "sector_contagion":
            driver = (f"Sales tied to {f['scen']['topic']} were less than {f['exposure']}% of net sales. Proposed regulatory or "
                      "industry developments affecting that area could adversely affect demand from certain customers.")
        else:
            driver = (f"The change primarily reflected {'higher' if g >= 0 else 'lower'} volumes in our {f['products'][0]} "
                      f"business and {'favorable' if g >= 0 else 'unfavorable'} pricing.")
        text = (f"Item 2. Management's Discussion and Analysis of Financial Condition and Results of Operations (excerpt)\n\n"
                f"Net sales for the three months ended {_long_date(f['pe'])} were {_money(f['rev'])}, compared with "
                f"{_money(f['rev_py'])} for the three months ended {_long_date(f['pe_py'])}, "
                f"{'an increase' if g >= 0 else 'a decrease'} of {_pct(g)}. {driver} Gross profit as a percentage of net sales "
                f"was {_pct(f['gm'])}, compared with {_pct(f['gm_py'])} in the prior-year period. Operating income was "
                f"{_money(f['oi'])}, compared with {_money(f['oi_py'])}.\n\nLiquidity and Capital Resources. As of "
                f"{_long_date(f['pe'])}, we had cash and cash equivalents of {_money(f['cash'])} and total debt of "
                f"{_money(f['debt'])}. Net cash provided by operating activities for the quarter was {_money(f['cfo'])} and "
                f"capital expenditures were {_money(f['capex'])}.")
        when = datetime.combine(filed, time(16, 10))
        return self._news(f, when, form.replace("-", ""), f"{f['company']} Form {form} for the period ended {_long_date(f['pe'])}",
                          text, "periodic_report", kind=DocumentKind.FILING, source="SEC EDGAR (synthetic)",
                          extra={"form": form, "period_end": f["pe"].isoformat()})

    def _form_8k(self, f: dict[str, Any]) -> Document:
        a = f["arch"]
        text = (f"Item 2.02 Results of Operations and Financial Condition. On {_long_date(f['rd'])}, {f['company']} issued a "
                f"press release announcing its financial results for the quarter ended {_long_date(f['pe'])}. A copy of the "
                "press release is furnished as Exhibit 99.1.")
        if a == "guidance_reset":
            text += (f"\n\nItem 8.01 Other Events. On {_long_date(f['rd'])}, the Board of Directors authorized the repurchase of "
                     f"up to {_money(f['buyback'])} of the company's common stock. Repurchases may be made from time to time in "
                     "open-market transactions or privately negotiated transactions.")
        return self._news(f, datetime.combine(f["rd"], time(16, 5)), "8K", f"{f['company']} Form 8-K", text,
                          "current_report", kind=DocumentKind.FILING, source="SEC EDGAR (synthetic)", extra={"form": "8-K"})

    def _industry_news(self, i: int) -> list[Document]:
        out: list[Document] = []
        for sc in self._scenarios:
            if self._industry[i] != sc["industry"]:
                continue
            f = {"ticker": self._tickers[i]}
            for s, title, slug in ((sc["headline_session"], sc["headline"], "SECTOR"),
                                   (sc["followup_session"], sc["followup"], "SECTOR2")):
                if s < max(1, int(self._listing[i])):
                    continue
                r, c, vm = self._move(i, s)
                day = _py(self._dd[s])
                text = (f"{sc['group'][0].upper() + sc['group'][1:]} fell sharply on {_long_date(day)} after {sc['event']}. {self._names[i]} "
                        f"({self._tickers[i]}) closed {_updown(r)} {_pct(r)} at ${c:.2f}. Analysts said investors were "
                        f"selling the group broadly on concern about exposure to {sc['topic']}, and several companies are "
                        "expected to address the issue on upcoming earnings calls.")
                if slug == "SECTOR2":
                    text = (f"{sc['group'][0].upper() + sc['group'][1:]} extended their slide on {_long_date(day)}. "
                            f"{self._names[i]} ({self._tickers[i]}) closed {_updown(r)} {_pct(r)} at "
                            f"${c:.2f}. Traders said the move extended a sell-off that began on "
                            f"{_long_date(_py(self._dd[sc['headline_session']]))}.")
                out.append(self._news(f, datetime.combine(day, time(15, 45)), slug, title, text, "industry"))
        return out
