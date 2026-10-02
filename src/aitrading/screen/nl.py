"""Natural-language investment observation -> validated ``ScreenSpec``.

Two translators share one result type (:class:`TranslationResult`):

* :class:`NLScreenTranslator` - an LLM (any :class:`~aitrading.llm.base.StructuredLLM`, Claude in
  production) emits a ``ScreenSpec`` via structured output. The spec is validated against the
  feature catalog; failures are fed back for up to ``max_repair_rounds`` repair calls. The system
  prompt is byte-stable (role, DSL semantics, unit rules, defaults, catalog - no per-request
  content), so it is served from the prompt cache across observations.
* :class:`HeuristicScreenTranslator` - deterministic, offline regex/keyword parser for common
  screening language, so the pipeline runs without an API key. Every interpretive default it
  applies is recorded in ``spec.assumptions``; text it cannot map goes to
  ``spec.unsupported_requests``. Its output always passes ``validate_against``.

:func:`translate_observation` tries the LLM first and falls back to the heuristic.

The LLM only chooses features, operators and thresholds; it never computes the numbers that drive
selection (those come from the deterministic feature engine).
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import Callable

from pydantic import ValidationError

from aitrading.llm.base import LLMError, LLMOutputError, StructuredLLM
from aitrading.screen.catalog import FeatureCatalog, default_catalog
from aitrading.screen.spec import Condition, RankFactor, ScreenSpec, UniverseSpec

__all__ = [
    "SYSTEM_PROMPT_TEMPLATE",
    "TranslationResult",
    "ScreenTranslationError",
    "NLScreenTranslator",
    "HeuristicScreenTranslator",
    "build_system_prompt",
    "translate_observation",
]

TOP_N_MIN, TOP_N_MAX = 1, 100


# ------------------------------------------------------------------------------------------------
# Result / error types
# ------------------------------------------------------------------------------------------------


@dataclass
class TranslationResult:
    spec: ScreenSpec
    attempts: int  # LLM calls made (1 for the heuristic)
    errors_by_round: list[list[str]]  # validation errors per attempt; the last entry is [] on success
    translator: str  # LLM name (e.g. "claude-opus-5-5") or "heuristic"


class ScreenTranslationError(ValueError):
    """The observation could not be turned into an executable screen; ``errors`` lists why."""

    def __init__(
        self,
        errors: list[str],
        message: str | None = None,
        *,
        spec: ScreenSpec | None = None,
        errors_by_round: list[list[str]] | None = None,
    ):
        self.errors: list[str] = list(errors)
        self.spec = spec  # last (invalid) spec, if any
        self.errors_by_round: list[list[str]] = [list(e) for e in (errors_by_round or [])]
        super().__init__(message or "could not produce a valid screen: " + "; ".join(self.errors))


def _clamp_top_n(n: object) -> int:
    try:
        v = int(n)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 10
    return min(max(v, TOP_N_MIN), TOP_N_MAX)


# ------------------------------------------------------------------------------------------------
# LLM translator
# ------------------------------------------------------------------------------------------------

SYSTEM_PROMPT_TEMPLATE = """\
You are a senior buy-side quantitative analyst. A portfolio manager (PM) describes an investment observation in \
plain English; you translate it into a precise, executable stock screen - a ScreenSpec - that a deterministic \
engine evaluates on point-in-time data for the US equity universe. You never see the data and you never compute \
or guess numbers about companies: you only encode the PM's criteria, using the feature catalog below. Faithfulness \
is the goal: the screen must select exactly what the PM described - no more, no less.

# ScreenSpec semantics
- `conditions` are ANDed.
- `any_of` is a list of groups; each group is ORed internally and the groups are ANDed with `conditions`. Each \
group needs at least two alternatives. Use it only for genuine alternatives in the observation ("RSI under 30 or \
more than 30% below the 52-week high").
- A Condition compares `feature` with exactly one of:
  - a constant `value` - ops > >= < <= == !=;
  - a range - op "between" with `value` (low) and `value_high` (high), inclusive on both ends, low <= high;
  - category labels - op "in" / "not_in" with `values` (category features only, e.g. gics_sector);
  - another feature - `other_feature` scaled by `multiplier`, meaning feature <op> multiplier x other_feature \
(leave `value` null; both features must have the same unit).
- Boolean features are 1/0: use op "==" with value 1 (or 0).
- A missing (NaN) feature value never satisfies a condition: missing data excludes the stock.
- `ranking` orders the survivors: RankFactor(feature, direction higher_is_better | lower_is_better, weight > 0; \
weights are normalised). Only numeric features can be ranked. `top_n` (1-100, default 10) is how many ranked names \
are kept.
- `universe`: country (ISO alpha-2, default "US"), security_types (default ["common_stock"]), min_price (USD, \
default 5), min_avg_dollar_volume_usd_mn (20-day average dollar volume floor in USD millions, default 5), \
exclude_sectors (GICS sector names, default none).

# Feature catalog - the ONLY feature names you may use
Each line: name (dtype) [unit]: exact definition.
{catalog}

# Unit rules
- market_cap_usd_bn and enterprise_value_usd_bn are USD billions: "$2-20B" -> market_cap_usd_bn between 2 and 20; \
"above $500M" -> market_cap_usd_bn > 0.5. avg_dollar_volume_20d_usd_mn is USD millions.
- Features ending in _pct are percentages, not fractions: "FCF yield above 5%" -> fcf_yield_pct > 5 (never 0.05); \
"revenue growth above 8%" -> revenue_growth_yoy_pct > 8.
- Features ending in _pp are percentage points (pp): "beat the market by 10 points over 6 months" -> \
rel_strength_6m_pp > 10; "gross margin up 2pp year on year" -> gross_margin_change_yoy_pp > 2 (200 bps = 2 pp).
- drawdown_from_52w_high_pct is NEGATIVE (always <= 0): "down 15-40% from the highs" -> between -40 and -15; "at \
least 20% below the 52-week high" -> <= -20; "within 5% of the 52-week high" -> >= -5. above_52w_low_pct is >= 0.
- price_vs_sma_50_pct, price_vs_sma_200_pct and sma_50_vs_sma_200_pct are signed percentages: "above the 200-day" \
-> price_vs_sma_200_pct > 0; "more than 10% below the 50-day" -> price_vs_sma_50_pct < -10; "50-day above \
200-day" -> sma_50_vs_sma_200_pct > 0.
- rsi_14, iv_rank_1y and return_6m_percentile are on a 0-100 scale. Features with unit [x] are plain multiples: \
"EV/EBITDA under 10x" -> ev_to_ebitda < 10; "volume twice the average" -> 2.
- Negative values carry meaning: net_debt_usd_bn < 0 and net_debt_to_ebitda < 0 mean net cash; \
eps_revision_3m_pct < 0 means estimates were cut.

# Interpretation defaults
When the observation is vague, apply these defaults and record each one you use in `assumptions`. An explicit \
number in the observation always overrides a default ("oversold (RSI under 40)" -> rsi_14 < 40, not 30; "mid-caps \
($2-20B)" -> between 2 and 20, not 2-10).
- Size: micro cap -> market_cap_usd_bn < 0.3; small cap -> between 0.3 and 2; mid cap -> between 2 and 10 unless a \
range is given; large cap -> > 10; mega cap -> > 200.
- "uptrend" / "established uptrend" -> sma_50_vs_sma_200_pct > 0 and/or price_vs_sma_200_pct > 0 (when the PM \
spells the trend out, encode exactly what they say and nothing else); "downtrend" -> sma_50_vs_sma_200_pct < 0. \
"golden cross" -> sma_50_vs_sma_200_pct > 0, or golden_cross_20d == 1 when it must be recent.
- "momentum" -> return_12m_ex_1m_pct (12-1 momentum); relative to the market -> rel_strength_3m_pp / \
rel_strength_6m_pp / rel_strength_12m_pp ("outperforming the market over 6 months" -> rel_strength_6m_pp > 0).
- "pulled back" / "sold off" without a size -> drawdown_from_52w_high_pct <= -10; "near highs" -> >= -5.
- "oversold" -> rsi_14 < 30; "overbought" -> rsi_14 > 70 (or the level stated).
- "heavy volume" / "on volume" / "capitulation" -> max_volume_ratio_20d >= 2; a sustained pick-up in activity -> \
rel_volume_20d >= 1.3; a "volume surge" in the last few days -> rel_volume_5d >= 1.5.
- "cheap" / "undervalued" -> fcf_yield_pct (higher) and/or ev_to_ebitda (lower); "strong free cash flow" without a \
number -> fcf_yield_pct > 5.
- "growth" without a qualifier means revenue growth (revenue_growth_yoy_pct); "high growth" without a number -> \
revenue_growth_yoy_pct > 10.
- "estimates rising" -> eps_revision_3m_pct > 0; "estimates being cut" -> eps_revision_3m_pct < 0.
- "net cash" -> net_debt_usd_bn < 0; "low leverage" -> net_debt_to_ebitda < 2.
- "elevated / high short interest" without a number -> short_interest_pct_float > 10.
- Options / implied-volatility language -> iv_rank_1y, iv_30d_pct, iv_to_realized_vol_ratio, \
put_call_volume_ratio, put_call_oi_ratio ("IV rank above 50" -> iv_rank_1y > 50; "options pricing more risk than \
realised" -> iv_to_realized_vol_ratio > 1).
- Sectors use exact GICS sector names: Information Technology, Health Care, Financials, Consumer Discretionary, \
Consumer Staples, Industrials, Energy, Materials, Utilities, Real Estate, Communication Services ("tech" -> \
Information Technology). Inclusion -> gics_sector in [...]; exclusion -> universe.exclude_sectors.

# Rules
1. Copy the observation verbatim into `observation`.
2. Use only feature names from the catalog, spelled exactly. Never invent a feature, a unit or a data source.
3. Anything the catalog cannot express (dividends, insider buying, buybacks, ESG, management changes, news or \
sentiment, M&A, a lookback window the catalog lacks, non-US markets, ...) goes into `unsupported_requests` as a \
short phrase quoting the observation. If you approximate it with a close feature, say so in `assumptions` AND \
still list it in `unsupported_requests`. Never drop a request silently.
4. Keep conditions minimal and faithful: one condition per criterion the PM stated and no extra filters they did \
not ask for (the universe price/liquidity defaults are the only exception). Do not add conditions for features \
that only appear in the ranking.
5. Instructions about what happens after the screen - reading earnings calls, transcripts, news or filings, \
explaining the dislocation, writing up the names - are handled by later pipeline stages: do not encode them as \
conditions and do not list them as unsupported.
6. Ranking: follow the observation's "rank by" / "sort by" wording in its order, with equal weights unless weights \
are given. "the size of the drawdown" -> drawdown_from_52w_high_pct lower_is_better (more negative = bigger \
drawdown); "growth" -> revenue_growth_yoy_pct higher_is_better. If no ranking is stated, rank by the condition \
features that have a natural direction (e.g. higher fcf_yield_pct, higher revenue_growth_yoy_pct, lower \
ev_to_ebitda) and say so in `assumptions`. Always give at least one rank factor.
7. Universe: keep the defaults unless the observation says otherwise (a higher price or liquidity floor, excluded \
sectors).
8. top_n: the number of names the PM asks for, else 10.
9. `name`: a short snake_case slug, e.g. "midcap_uptrend_pullback".
10. Give every condition and rank factor a short `rationale` naming the words of the observation it encodes.

# Worked examples
Observation: "Find US mid-caps ($2-20B) that were in established uptrends (50-day above 200-day, positive 12-1 \
momentum) but have pulled back 15-40% from their 52-week highs over the past few months on heavy volume, now \
oversold (RSI under 40), still generating strong free cash flow (FCF yield above 4%) with revenue growth above 8%, \
and where short interest is elevated (above 6% of float). Rank by FCF yield, growth and the size of the drawdown, \
then read the latest earnings calls and explain the dislocation."
conditions:
  market_cap_usd_bn between 2 and 20
  sma_50_vs_sma_200_pct > 0
  return_12m_ex_1m_pct > 0
  drawdown_from_52w_high_pct between -40 and -15
  max_volume_ratio_20d >= 2
  rsi_14 < 40
  fcf_yield_pct > 4
  revenue_growth_yoy_pct > 8
  short_interest_pct_float > 6
ranking: fcf_yield_pct higher_is_better (1), revenue_growth_yoy_pct higher_is_better (1), \
drawdown_from_52w_high_pct lower_is_better (1)
universe: defaults; unsupported_requests: none (reading the earnings calls is a later pipeline stage).

Observation: "Small caps under $1.5bn below their 200-day that are either oversold or more than 30% off their \
highs, ex-financials, with insider buying. Rank by EV/EBITDA, top 15."
conditions: market_cap_usd_bn between 0.3 and 1.5; price_vs_sma_200_pct < 0
any_of: [[rsi_14 < 30, drawdown_from_52w_high_pct < -30]]
universe.exclude_sectors: ["Financials"]
ranking: ev_to_ebitda lower_is_better (1); top_n: 15
assumptions: small cap lower bound 0.3bn; "oversold" read as rsi_14 < 30
unsupported_requests: ["insider buying"]
"""


def build_system_prompt(catalog: FeatureCatalog | None = None) -> str:
    """The translator's system prompt (byte-stable for a given catalog: cacheable across observations)."""
    cat = catalog or default_catalog()
    return SYSTEM_PROMPT_TEMPLATE.replace("{catalog}", cat.to_prompt())


def _validation_messages(err: ValidationError | LLMOutputError) -> list[str]:
    out = []
    for e in err.errors():
        loc = ".".join(str(p) for p in e.get("loc", ()))
        out.append(f"schema error at '{loc or '<root>'}': {e.get('msg', 'invalid')}")
    return out or [str(err)]


class NLScreenTranslator:
    """LLM-backed observation -> ``ScreenSpec`` translation with validate-and-repair rounds.

    Round 0 uses purpose ``"nl_screen"``; each repair round uses ``"nl_screen:repair"`` and sends
    the observation, the previous spec as JSON and the validation errors. After normalisation
    (``observation`` overwritten with the input, ``top_n`` clamped to [1, 100]) the spec must pass
    ``validate_against(catalog)``; otherwise :class:`ScreenTranslationError` is raised once
    ``max_repair_rounds`` repairs are spent. A reply that does not validate as a ``ScreenSpec``
    (``LLMOutputError`` / pydantic ``ValidationError``) uses a repair round too; other ``LLMError`` /
    ``LLMRefusalError`` propagate.
    """

    def __init__(
        self,
        llm: StructuredLLM,
        catalog: FeatureCatalog | None = None,
        *,
        max_repair_rounds: int = 2,
        effort: str = "high",
    ):
        if max_repair_rounds < 0:
            raise ValueError("max_repair_rounds must be >= 0")
        self.llm = llm
        self.catalog = catalog or default_catalog()
        self.max_repair_rounds = max_repair_rounds
        self.effort = effort
        self._system = build_system_prompt(self.catalog)

    @property
    def system_prompt(self) -> str:
        return self._system

    @staticmethod
    def user_prompt(observation: str) -> str:
        return (
            f"<observation>\n{observation}\n</observation>\n\n"
            "Translate the observation above into a ScreenSpec. Return the complete ScreenSpec."
        )

    @staticmethod
    def repair_prompt(observation: str, previous: ScreenSpec | None, errors: list[str]) -> str:
        bullets = "\n".join(f"- {e}" for e in errors)
        if previous is not None:
            prev = json.dumps(previous.model_dump(mode="json"), separators=(",", ":"))
            head = f"Your previous ScreenSpec failed validation:\n<previous_spec>\n{prev}\n</previous_spec>\n"
        else:
            head = "Your previous response was not a valid ScreenSpec.\n"
        return (
            f"<observation>\n{observation}\n</observation>\n\n"
            f"{head}<errors>\n{bullets}\n</errors>\n\n"
            "Return the complete corrected ScreenSpec. Fix exactly what the errors require and keep every other "
            "condition, the ranking and the assumptions faithful to the observation. Use only catalog feature names; "
            "anything that cannot be expressed goes into unsupported_requests - do not invent a feature."
        )

    def translate(self, observation: str) -> TranslationResult:
        if not isinstance(observation, str) or not observation.strip():
            raise ScreenTranslationError(["observation is empty"])
        errors_by_round: list[list[str]] = []
        spec: ScreenSpec | None = None
        errors: list[str] = []
        for rnd in range(self.max_repair_rounds + 1):
            if rnd == 0:
                purpose, user = "nl_screen", self.user_prompt(observation)
            else:
                purpose, user = "nl_screen:repair", self.repair_prompt(observation, spec, errors)
            try:
                out = self.llm.structured(
                    purpose=purpose, system=self._system, user=user, output_model=ScreenSpec, effort=self.effort
                )
                if not isinstance(out, ScreenSpec):
                    out = ScreenSpec.model_validate(out)
            except (ValidationError, LLMOutputError) as e:  # schema-level failure (e.g. a bound the API cannot enforce)
                errors = _validation_messages(e)
                errors_by_round.append(errors)
                continue
            spec = out.model_copy(update={"observation": observation, "top_n": _clamp_top_n(out.top_n)})
            errors = spec.validate_against(self.catalog)
            errors_by_round.append(errors)
            if not errors:
                return TranslationResult(spec=spec, attempts=rnd + 1, errors_by_round=errors_by_round, translator=self.llm.name)
        raise ScreenTranslationError(
            errors,
            f"could not produce a valid screen after {len(errors_by_round)} attempt(s): " + "; ".join(errors),
            spec=spec,
            errors_by_round=errors_by_round,
        )


# ------------------------------------------------------------------------------------------------
# Heuristic translator: lexical building blocks (all patterns run on lower-cased text)
# ------------------------------------------------------------------------------------------------

_UNUM = r"\d+(?:,\d{3})*(?:\.\d+)?"
_NUM = rf"[-+]?{_UNUM}"
_NUMSCAN = re.compile(r"(?<![\d.,])[-+]?\d+(?:,\d{3})*(?:\.\d+)?")
_PCT = r"(?:\s*(?:%|percent\b|per\s*cent\b|pct\b))"
_PP = r"(?:\s*(?:%|pp\b|ppts?\b|percentage\s+points?\b|points?\b|pts\b))"
_X = r"(?:\s*(?:x\b|times\b))"
_DAYS = r"(?:\s*(?:days?\b|sessions?\b))"

_GE = (
    r"(?:>=|=>|≥|\bat\s+least\b|\bno\s+less\s+than\b|\bnot\s+less\s+than\b|\bno\s+lower\s+than\b|\bnot\s+below\b"
    r"|\b(?:a\s+)?minimum\s+of\b|\bmin\.?(?:\s+of)?(?=[\s\d$]))"
)
_LE = (
    r"(?:<=|=<|≤|\bat\s+most\b|\bno\s+more\s+than\b|\bnot\s+more\s+than\b|\bno\s+higher\s+than\b|\bno\s+greater\s+than\b"
    r"|\bnot\s+above\b|\bnot\s+over\b|\b(?:a\s+)?maximum\s+of\b|\bmax\.?(?:\s+of)?(?=[\s\d$])|\bup\s+to\b)"
)
_GT = (
    r"(?:>|\babove\b|\bover\b|\bgreater\s+than\b|\bmore\s+than\b|\bhigher\s+than\b|\bbigger\s+than\b|\blarger\s+than\b"
    r"|\bbetter\s+than\b|\bfaster\s+than\b|\bstronger\s+than\b|\bexceed(?:s|ing)?\b|\bin\s+excess\s+of\b|\bnorth\s+of\b"
    r"|\bupwards\s+of\b)"
)
_LT = (
    r"(?:<|\bbelow\b|\bunder\b|\bless\s+than\b|\blower\s+than\b|\bsmaller\s+than\b|\bbeneath\b|\bsouth\s+of\b"
    r"|\bcheaper\s+than\b|\bslower\s+than\b|\bweaker\s+than\b|\bsub\b)"
)
_COMP = rf"(?:{_GE}|{_LE}|{_GT}|{_LT})"
_COMP_OPS = [(">=", re.compile(_GE)), ("<=", re.compile(_LE)), (">", re.compile(_GT)), ("<", re.compile(_LT))]
_SUF = r"(?:\s*\+|\s+or\s+(?:more|higher|greater|better|above|over)\b|\s+or\s+(?:less|lower|below|under|fewer)\b|\s+plus\b)"
_SUF_GE = re.compile(r"(?:\+|\bor\s+(?:more|higher|greater|better|above|over)|\bplus)\s*$")
_SUF_LE = re.compile(r"\bor\s+(?:less|lower|below|under|fewer)\s*$")

# Connective words allowed between a feature phrase and its threshold ("short interest is elevated (above 6%").
_GLUE = (
    r"(?:\s|[(:=]|\b(?:is|are|was|were|be|been|being|of|at|currently|now|still|already|running|sitting|stands?|"
    r"standing|remains?|remaining|stays?|elevated|high|heavy|strong|solid|healthy|robust|ideally|preferably|that|"
    r"which|ttm|trailing)\b)*?"
)


def _valre(unit: str = "") -> str:
    """A threshold: range, 'between a and b', comparator + number, or a bare number (unit optional)."""
    u = f"(?:{unit})?" if unit else ""
    return (
        rf"(?:(?:between\s+|from\s+)?\$?{_NUM}{u}\s*(?:-|to|and)\s*\$?{_NUM}{u}"
        rf"|{_COMP}\s*\$?{_NUM}{u}{_SUF}?"
        rf"|\$?{_NUM}{u}{_SUF}?)"
    )


@dataclass(frozen=True)
class _Val:
    op: str | None  # ">", ">=", "<", "<=", "between" or None (bare number)
    lo: float
    hi: float | None = None


def _comp_op(s: str) -> str | None:
    for op, rx in _COMP_OPS:
        if rx.match(s):
            return op
    return None


def _parse_val(s: str | None) -> _Val | None:
    if not s:
        return None
    s = s.strip()
    nums = [float(x.replace(",", "")) for x in _NUMSCAN.findall(s)]
    if not nums:
        return None
    op = _comp_op(s)
    if op:
        return _Val(op, nums[0])
    if len(nums) >= 2:
        lo, hi = sorted(nums[:2])
        return _Val("between", lo, hi)
    if _SUF_GE.search(s):
        return _Val(">=", nums[0])
    if _SUF_LE.search(s):
        return _Val("<=", nums[0])
    return _Val(None, nums[0])


_FLIP = {">": "<", ">=": "<=", "<": ">", "<=": ">="}


def _below(v: _Val, bare: str = "<=") -> tuple[str, float, float | None]:
    """A magnitude *below* a reference ("15-40% off highs", "more than 20% below") on a signed feature."""
    if v.op == "between":
        lo, hi = sorted((abs(v.lo), abs(v.hi if v.hi is not None else v.lo)))
        return "between", -hi, -lo
    a = abs(v.lo)
    if v.op is None:
        return bare, -a, None
    return _FLIP[v.op], -a, None


_WORDNUM = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "six": 6, "nine": 9, "twelve": 12, "few": 3}

# Months requested -> available catalog windows.
_RET_WINDOWS = {1: "return_1m_pct", 3: "return_3m_pct", 6: "return_6m_pct", 12: "return_12m_pct"}
_RS_WINDOWS = {3: "rel_strength_3m_pp", 6: "rel_strength_6m_pp", 12: "rel_strength_12m_pp"}

_WINDOW = (
    r"(?:\s+(?:over|in|during|for|across|on)\s+(?:the\s+|a\s+)?(?:(?:past|last|prior|trailing|previous)\s+)?"
    r"(?P<wn>\d+|one|two|three|four|six|nine|twelve|a)?[\s-]*(?P<wu>months?|mos?|years?|yrs?|quarters?|weeks?|wks?)\b)"
)


def _months(n: str | None, unit: str | None) -> float | None:
    if not unit:
        return None
    k = 1.0 if not n else float(_WORDNUM.get(n, n) if not n.isdigit() else n)
    u = unit.rstrip("s")
    if u in ("month", "mo"):
        return k
    if u in ("year", "yr"):
        return 12 * k
    if u == "quarter":
        return 3 * k
    if u in ("week", "wk"):
        return k / 4.33
    return None


def _nearest(months: float, table: dict[int, str]) -> tuple[str, bool]:
    best = min(table, key=lambda w: (abs(w - months), w))
    return table[best], abs(best - months) > 1e-9


_MONEY_UNIT = r"(?:\s*(?:bn|b|billion|bil|mn|mm|m|million|mil|tn|t|trillion|k|thousand)\b)"
_BIG_UNIT = r"(?:\s*(?:bn|b|billion|bil|tn|t|trillion)\b)"
_MONEY_SCAN = re.compile(rf"({_UNUM})\s*(bn|b|billion|bil|mn|mm|m|million|mil|tn|t|trillion|k|thousand)?\b")
_TO_BN = {"b": 1.0, "bn": 1.0, "billion": 1.0, "bil": 1.0, "m": 1e-3, "mn": 1e-3, "mm": 1e-3, "million": 1e-3,
          "mil": 1e-3, "t": 1e3, "tn": 1e3, "trillion": 1e3, "k": 1e-6, "thousand": 1e-6}


def _money(s: str, default_unit: str = "b") -> tuple[_Val, bool] | None:
    """Parse a money threshold to USD billions; returns (value, unit_was_missing)."""
    pairs = [(float(n.replace(",", "")), u) for n, u in _MONEY_SCAN.findall(s)]
    if not pairs:
        return None
    units = [u for _, u in pairs]
    missing = not any(units)
    fill = next((u for u in reversed(units) if u), default_unit)
    vals = [round(n * _TO_BN[u or fill], 9) for n, u in pairs]
    op = _comp_op(s.strip())
    if op:
        return _Val(op, vals[0]), missing
    if len(vals) >= 2:
        lo, hi = sorted(vals[:2])
        return _Val("between", lo, hi), missing
    if _SUF_GE.search(s):
        return _Val(">=", vals[0]), missing
    if _SUF_LE.search(s):
        return _Val("<=", vals[0]), missing
    return _Val(None, vals[0]), missing


def _money_valre(unit: str = _MONEY_UNIT) -> str:
    m = rf"\$?\s*{_UNUM}{unit}?"
    return rf"(?:(?:between\s+|from\s+)?{m}\s*(?:-|to|and)\s*{m}|{_COMP}\s*{m}{_SUF}?|{m}{_SUF}?)"


# GICS sectors and the words that name them.
_SECTOR_ALIASES: dict[str, str] = {
    "Information Technology": r"information\s+technology|info(?:rmation)?[\s-]tech|technology|tech",
    "Health Care": r"health[\s-]?care",
    "Financials": r"financials?|financial\s+services",
    "Consumer Discretionary": r"consumer\s+discretionary|discretionary",
    "Consumer Staples": r"consumer\s+staples|staples",
    "Industrials": r"industrials?",
    "Energy": r"energy",
    "Materials": r"(?:basic\s+)?materials",
    "Utilities": r"utilities",
    "Real Estate": r"real\s+estate|reits?",
    "Communication Services": r"communication\s+services|communications?|telecom(?:munication)?s?|media",
}
_SECTOR_RES = [(label, re.compile(rf"\b(?:{rx})\b")) for label, rx in _SECTOR_ALIASES.items()]
_SECTOR = r"\b(?:" + "|".join(f"(?:{rx})" for rx in _SECTOR_ALIASES.values()) + r")\b"
_SECTOR_LIST = rf"{_SECTOR}(?:\s*(?:,|/|&|\band\b|\bor\b|\bnor\b)\s*(?:(?:and|or)\s+)?(?:the\s+)?{_SECTOR})*"


def _sector_labels(text: str) -> list[str]:
    found: list[tuple[int, str]] = []
    for label, rx in _SECTOR_RES:
        for m in rx.finditer(text):
            found.append((m.start(), label))
    out: list[str] = []
    for _, label in sorted(found):
        if label not in out:
            out.append(label)
    return out


# Moving averages: "200-day", "200dma", "the 50 day moving average", "sma50".
_MA_SUFFIX = (
    r"(?:[\s-]?(?:day|d|dma|sma|ma|session|period)s?\b"
    r"(?:[\s-]+(?:simple\s+|exponential\s+)?(?:moving[\s-]averages?|mas?|smas?|emas?|averages?|lines?)\b)?)"
)
_SLOPE_UP = r"rising|upward[\s-]sloping|up[\s-]?sloping|climbing|increasing"
_SLOPE_DN = r"falling|declining|downward[\s-]sloping|down[\s-]?sloping|decreasing"


def _ma(n: int) -> str:
    return (
        rf"(?:(?:the|its|their|a|an)\s+)?(?:(?:{_SLOPE_UP}|{_SLOPE_DN})\s+)?"
        rf"(?:(?:sma|ma|ema)[\s-]?{n}\b|\b{n}{_MA_SUFFIX})"
    )


_HIGH = (
    r"(?:(?:their|its|the)\s+)?(?:(?:recent|prior|previous|trailing)\s+)?"
    r"(?:(?:52|fifty[\s-]two)[\s-]?(?:week|wk|w)|1[\s-]?year|one[\s-]year|12[\s-]?month|twelve[\s-]month|yearly|annual|all[\s-]time)?"
    r"[\s-]*(?:highs?|peaks?|tops?)\b"
)
_LOW = (
    r"(?:(?:their|its|the)\s+)?(?:(?:recent|prior|previous|trailing)\s+)?"
    r"(?:(?:52|fifty[\s-]two)[\s-]?(?:week|wk|w)|1[\s-]?year|one[\s-]year|12[\s-]?month|yearly|annual)?[\s-]*(?:lows?|bottoms?)\b"
)

_NEGATION = re.compile(
    r"\b(?:not|no|never|without|nor|isn't|aren't|wasn't|weren't|don't|doesn't|non)\b[\s-]+"
    r"(?:(?:a|an|the|in|be|been|being|currently|yet|very|too|so|showing|having|have|has|had|on|at|trading|seeing|any)\s+){0,2}$"
)


# ------------------------------------------------------------------------------------------------
# Heuristic translator: matching engine
# ------------------------------------------------------------------------------------------------


@dataclass
class _Hit:
    start: int
    end: int
    conditions: list[Condition]
    covered_by: frozenset[str] | None = None  # set => a vague-word default, dropped if any of these is explicit
    notes: list[str] = field(default_factory=list)


Handler = Callable[[re.Match, "_Ctx"], "list[_Hit] | None"]


@dataclass(frozen=True)
class _Rule:
    pattern: re.Pattern
    handler: Handler
    negatable: bool = True


def _r(pattern: str, handler: Handler, negatable: bool = True) -> _Rule:
    return _Rule(re.compile(pattern), handler, negatable)


_CHAR_MAP = {"–": "-", "—": "-", "−": "-", "‐": "-", "‑": "-", "’": "'", "‘": "'",
             "“": '"', "”": '"', " ": " ", "\t": " ", "\n": " ", "\r": " "}


def _normalise(text: str) -> str:
    """Lower-case, unify dashes/quotes/whitespace; length-preserving so spans map back to the input."""
    out = []
    for ch in text:
        ch = _CHAR_MAP.get(ch, ch)
        low = ch.lower()
        out.append(low if len(low) == 1 else ch)
    return "".join(out)


def _num(x: float) -> float:
    return round(float(x), 9) + 0.0


class _Ctx:
    def __init__(self, original: str):
        self.original = original
        self.text = _normalise(original)
        self.masked = self.text
        self.hits: list[_Hit] = []
        self.assumptions: list[str] = []
        self.unsupported: list[str] = []
        self.rank: list[RankFactor] | None = None
        self.rank_start: int | None = None
        self.top_n: int | None = None
        self.min_price: float | None = None
        self.min_adv: float | None = None
        self.exclude_sectors: list[str] = []

    def quote(self, m: re.Match | tuple[int, int]) -> str:
        s, e = (m.start(), m.end()) if isinstance(m, re.Match) else m
        q = " ".join(self.original[s:e].split()).strip(" ,;:(")
        return q + ")" * max(0, q.count("(") - q.count(")"))

    def consume(self, s: int, e: int) -> None:
        self.masked = self.masked[:s] + "\x00" * (e - s) + self.masked[e:]

    def note(self, text: str) -> None:
        if text not in self.assumptions:
            self.assumptions.append(text)

    def unsupported_add(self, text: str) -> None:
        text = text.strip()
        if text and text not in self.unsupported:
            self.unsupported.append(text)

    def cond(self, m: re.Match, feature: str, op: str, value: float | None = None, high: float | None = None, *,
             values: list[str] | None = None, other: str | None = None, mult: float = 1.0, default: str = "") -> Condition:
        why = f"'{self.quote(m)}'" + (f" ({default})" if default else "")
        if op == "between":
            lo, hi = sorted((_num(value), _num(high)))  # type: ignore[arg-type]
            return Condition(feature=feature, op="between", value=lo, value_high=hi, rationale=why)
        if op in ("in", "not_in"):
            return Condition(feature=feature, op=op, values=list(values or []), rationale=why)
        if other is not None:
            return Condition(feature=feature, op=op, other_feature=other, multiplier=_num(mult), rationale=why)
        return Condition(feature=feature, op=op, value=_num(value), rationale=why)  # type: ignore[arg-type]

    def hit(self, m: re.Match, conds: list[Condition], *, covered_by: set[str] | None = None,
            notes: list[str] | None = None) -> _Hit:
        return _Hit(m.start(), m.end(), conds, frozenset(covered_by) if covered_by is not None else None, list(notes or []))

    def negated(self, start: int) -> bool:
        return bool(_NEGATION.search(self.text[max(0, start - 40):start]))


# ---- handler factories -------------------------------------------------------------------------


def _numeric(feature: str, default_op: str | None, *, note: str | None = None, scale: float = 1.0) -> Handler:
    """Feature <op> threshold, from the match's ``val`` group; a bare number takes ``default_op``."""

    def h(m: re.Match, ctx: _Ctx) -> list[_Hit] | None:
        v = _parse_val(m.group("val"))
        if v is None:
            return None
        op = v.op or default_op
        if op is None:
            return None
        hi = v.hi * scale if v.hi is not None else None
        dflt = f"bare number read as {op}" if v.op is None else ""
        return [ctx.hit(m, [ctx.cond(m, feature, op, v.lo * scale, hi, default=dflt)], notes=[note] if note else None)]

    return h


def _fixed(feature: str, op: str, value: float, *, covered_by: set[str] | None = None, default: str = "",
           note: str | None = None) -> Handler:
    """A phrase that maps to one fixed condition (a vague-word default when ``covered_by`` is given)."""

    def h(m: re.Match, ctx: _Ctx) -> list[_Hit]:
        c = ctx.cond(m, feature, op, value, default=default)
        n = [note] if note else [f"'{ctx.quote(m)}' read as {c.describe()} (default)."] if covered_by is not None else []
        return [ctx.hit(m, [c], covered_by=covered_by, notes=n)]

    return h


def _subject(subject: str, feature: str, unit: str, default_op: str | None, *, prefix: bool = True,
             note: str | None = None) -> list[_Rule]:
    """'<subject> <glue> <threshold>' and, optionally, '<threshold> <subject>' ("8%+ revenue growth")."""
    h = _numeric(feature, default_op, note=note)
    rules = [_r(rf"\b(?:{subject})\b{_GLUE}(?P<val>{_valre(unit)})", h)]
    if prefix and unit:
        rules.append(_r(rf"(?<![\w.$])(?P<val>(?:{_COMP}\s*)?{_NUM}(?:{unit}){_SUF}?)\s+(?:of\s+)?(?:{subject})\b", h))
    return rules


# ---- specific handlers ---------------------------------------------------------------------------


def _h_literal(catalog: FeatureCatalog) -> Handler:
    def h(m: re.Match, ctx: _Ctx) -> list[_Hit] | None:
        f = m.group("f")
        raw = m.group("val").strip()
        eq = re.match(r"(==|!=|=)\s*", raw)
        if eq:
            v = _parse_val(raw[eq.end():])
            if v is None:
                return None
            op = "!=" if eq.group(1) == "!=" else "=="
            return [ctx.hit(m, [ctx.cond(m, f, op, v.lo)])]
        v = _parse_val(raw)
        if v is None:
            return None
        op = v.op
        if op is None:
            hib = catalog[f].higher_is_better if f in catalog else None
            if hib is None:
                return None
            op = ">=" if hib else "<="
        return [ctx.hit(m, [ctx.cond(m, f, op, v.lo, v.hi)])]

    return h


def _h_mcap(default_unit_note: bool = True) -> Handler:
    def h(m: re.Match, ctx: _Ctx) -> list[_Hit] | None:
        parsed = _money(m.group("val"))
        if parsed is None:
            return None
        v, missing = parsed
        notes = ["Market-cap figures without a unit read as USD billions."] if missing and default_unit_note else []
        op = v.op
        if op is None:
            op = ">="
            notes.append(f"'{ctx.quote(m)}' read as a market-cap floor.")
        return [ctx.hit(m, [ctx.cond(m, "market_cap_usd_bn", op, v.lo, v.hi)], notes=notes)]

    return h


_NON_MCAP_CONTEXT = re.compile(
    r"\b(?:revenues?|sales|volume|adv|ebitda|debt|cash|income|earnings|turnover|traded|buybacks?|dividends?|backlog|"
    r"deals?|acquisitions?|fcf|flows?|liquidity|capex|assets|price|priced)\b"
)


def _h_mcap_free(m: re.Match, ctx: _Ctx) -> list[_Hit] | None:
    before = ctx.text[max(0, m.start() - 30):m.start()]
    after = ctx.text[m.end():m.end() + 25]
    if _NON_MCAP_CONTEXT.search(before) or re.match(r"\s*(?:(?:in|of)\s+)?(?:revenues?|sales|ebitda|debt|cash|volume)\b", after):
        return None
    return _h_mcap(False)(m, ctx)


def _h_adv(m: re.Match, ctx: _Ctx) -> list[_Hit] | None:
    parsed = _money(m.group("val"), default_unit="m")
    if parsed is None:
        return None
    v, _ = parsed
    mn = v.lo * 1e3
    if v.op in (">", ">=") or (v.op is None and _SUF_GE.search(m.group("val"))):
        ctx.min_adv = _num(max(mn, ctx.min_adv or 0.0))
        ctx.note(f"Liquidity floor {mn:g} USD mn from '{ctx.quote(m)}' (universe min_avg_dollar_volume_usd_mn keeps the strictest floor).")
        return []
    if v.op in ("<", "<="):
        return [ctx.hit(m, [ctx.cond(m, "avg_dollar_volume_20d_usd_mn", v.op, mn)])]
    if v.op == "between":
        return [ctx.hit(m, [ctx.cond(m, "avg_dollar_volume_20d_usd_mn", "between", v.lo * 1e3, (v.hi or 0) * 1e3)])]
    ctx.min_adv = _num(max(mn, ctx.min_adv or 0.0))
    ctx.note(f"Liquidity floor {mn:g} USD mn from '{ctx.quote(m)}' (universe min_avg_dollar_volume_usd_mn keeps the strictest floor).")
    return []


def _h_price(m: re.Match, ctx: _Ctx) -> list[_Hit] | None:
    v = _parse_val(m.group("val"))
    if v is None:
        return None
    op = v.op or ">="
    if op in (">", ">="):
        ctx.min_price = _num(max(v.lo, ctx.min_price or 0.0))
        ctx.note(f"Price floor {v.lo:g} from '{ctx.quote(m)}' (universe min_price keeps the strictest floor).")
        return []
    if op == "between":
        return [ctx.hit(m, [ctx.cond(m, "price", "between", v.lo, v.hi)])]
    return [ctx.hit(m, [ctx.cond(m, "price", op, v.lo)])]


def _h_ma_cross(m: re.Match, ctx: _Ctx) -> list[_Hit]:
    rel = m.group("rel")
    up = bool(re.search(r"above|over|>", rel))
    if "cross" in rel or "moved" in rel or "moving" in rel:
        if up:
            c = ctx.cond(m, "golden_cross_20d", "==", 1)
            return [ctx.hit(m, [c], notes=[f"'{ctx.quote(m)}' read as a golden cross within the last 20 sessions."])]
        c = ctx.cond(m, "sma_50_vs_sma_200_pct", "<", 0)
        return [ctx.hit(m, [c], notes=[f"'{ctx.quote(m)}': the catalog has no recent death-cross flag; screened on 50-day below 200-day."])]
    return [ctx.hit(m, [ctx.cond(m, "sma_50_vs_sma_200_pct", ">" if up else "<", 0)])]


def _h_cross_word(m: re.Match, ctx: _Ctx) -> list[_Hit]:
    golden = m.group("kind") == "golden"
    recent = bool(m.group("recent") or m.group("when"))
    if golden and recent:
        notes = [f"'{ctx.quote(m)}' read as golden_cross_20d == 1 (cross within the last 20 sessions)."]
        return [ctx.hit(m, [ctx.cond(m, "golden_cross_20d", "==", 1)], notes=notes)]
    c = ctx.cond(m, "sma_50_vs_sma_200_pct", ">" if golden else "<", 0)
    notes = [f"'{ctx.quote(m)}' read as the moving-average structure ({c.describe()})."]
    if not golden and recent:
        ctx.unsupported_add(f"recency of the death cross ('{ctx.quote(m)}')")
    return [ctx.hit(m, [c], notes=notes)]


def _slope_conds(m: re.Match, ctx: _Ctx, n: int) -> list[Condition]:
    """'above a rising 200-day' also constrains the slope (only the 200-day slope is in the catalog)."""
    txt = m.group(0)
    up, dn = re.search(rf"\b(?:{_SLOPE_UP})\b", txt), re.search(rf"\b(?:{_SLOPE_DN})\b", txt)
    if not (up or dn):
        return []
    if n != 200:
        ctx.unsupported_add(f"slope of the {n}-day average ('{ctx.quote(m)}')")
        return []
    return [ctx.cond(m, "sma_200_slope_1m_pct", ">" if up else "<", 0)]


def _h_price_vs(n: int) -> Handler:
    feature = {200: "price_vs_sma_200_pct", 50: "price_vs_sma_50_pct"}.get(n)

    def h(m: re.Match, ctx: _Ctx) -> list[_Hit]:
        above = m.group("rel") in ("above", "over")
        v = _parse_val(m.group("val")) if m.group("val") else None
        notes: list[str] = []
        if feature is None:  # 20-day: price vs sma_20 (both USD)
            mult = 1.0
            if v is not None and v.op in (None, ">", ">=") and v.lo:
                mult = 1 + abs(v.lo) / 100 if above else 1 - abs(v.lo) / 100
            elif v is not None:
                notes.append(f"'{ctx.quote(m)}': only the direction versus the 20-day average is screened.")
            conds = [ctx.cond(m, "price", ">=" if (above and mult != 1) else ">" if above else "<=" if mult != 1 else "<",
                              other="sma_20", mult=mult)]
        elif v is None:
            conds = [ctx.cond(m, feature, ">" if above else "<", 0)]
        else:
            a = abs(v.lo)
            if v.op == "between":
                lo, hi = sorted((abs(v.lo), abs(v.hi or 0)))
                conds = [ctx.cond(m, feature, "between", lo if above else -hi, hi if above else -lo)]
            elif v.op in ("<", "<="):  # "less than 5% above" -> between 0 and 5
                conds = [ctx.cond(m, feature, "between", 0 if above else -a, a if above else 0)]
            else:
                op = v.op or ">="
                if v.op is None:
                    notes.append(f"'{ctx.quote(m)}' read as at least {a:g}% {'above' if above else 'below'}.")
                conds = [ctx.cond(m, feature, op if above else _FLIP[op], a if above else -a)]
        return [ctx.hit(m, conds + _slope_conds(m, ctx, n), notes=notes)]

    return h


def _h_within_ma(n: int) -> Handler:
    feature = {200: "price_vs_sma_200_pct", 50: "price_vs_sma_50_pct"}.get(n)

    def h(m: re.Match, ctx: _Ctx) -> list[_Hit] | None:
        v = _parse_val(m.group("val"))
        if v is None:
            return None
        a = abs(v.lo)
        if feature is None:
            return [ctx.hit(m, [ctx.cond(m, "price", ">=", other="sma_20", mult=1 - a / 100),
                                ctx.cond(m, "price", "<=", other="sma_20", mult=1 + a / 100)])]
        return [ctx.hit(m, [ctx.cond(m, feature, "between", -a, a)])]

    return h


def _h_slope(m: re.Match, ctx: _Ctx) -> list[_Hit]:
    up = bool(re.search(rf"\b(?:{_SLOPE_UP}|sloping\s+up(?:ward)?|turn(?:ing|ed)\s+up)\b", m.group(0)))
    return [ctx.hit(m, [ctx.cond(m, "sma_200_slope_1m_pct", ">" if up else "<", 0)])]


def _sign_word(word: str | None) -> int:
    if not word:
        return 0
    if re.match(r"(?:negative|weak|poor|low|falling|fading|deteriorating|declining)", word):
        return -1
    return 1


def _h_mom121(m: re.Match, ctx: _Ctx) -> list[_Hit]:
    val = m.groupdict().get("val")
    v = _parse_val(val) if val and re.search(r"\d", val) else None
    if v is not None and v.op is not None:
        return [ctx.hit(m, [ctx.cond(m, "return_12m_ex_1m_pct", v.op, v.lo, v.hi)])]
    sign = _sign_word(val) or _sign_word(m.groupdict().get("adj"))
    notes = [] if sign else [f"'{ctx.quote(m)}' read as positive 12-1 momentum."]
    return [ctx.hit(m, [ctx.cond(m, "return_12m_ex_1m_pct", "<" if sign < 0 else ">", 0)], notes=notes)]


_BROAD_BENCH = re.compile(r"market|s&p|spx|spy|index|benchmark")


def _rs_feature(m: re.Match, ctx: _Ctx) -> tuple[str, list[str]]:
    gd = m.groupdict()
    months = _months(gd.get("wn"), gd.get("wu"))
    notes: list[str] = []
    if months is None:
        notes.append(f"'{ctx.quote(m)}': no window given; used 6-month relative strength.")
        months = 6
    feat, approx = _nearest(months, _RS_WINDOWS)
    if approx:
        notes.append(f"'{ctx.quote(m)}': relative strength is available over 3/6/12 months; used {feat}.")
        ctx.unsupported_add(f"relative-strength window of {months:g} months ('{ctx.quote(m)}')")
    bench = gd.get("bench")
    if bench and not _BROAD_BENCH.match(bench):
        ctx.unsupported_add(f"relative strength versus '{bench}' (catalog measures versus the platform benchmark)")
        notes.append(f"Relative strength is measured versus the platform benchmark, not '{bench}'.")
    return feat, notes


def _h_outperform(m: re.Match, ctx: _Ctx) -> list[_Hit]:
    feat, notes = _rs_feature(m, ctx)
    under = bool(re.match(r"under|lag|trail|behind", m.group("verb")))
    v = _parse_val(m.groupdict().get("val") or m.groupdict().get("val2"))
    if v is None:
        c = ctx.cond(m, feat, "<" if under else ">", 0)
    elif under:
        op, lo, hi = _below(v, bare="<=")
        c = ctx.cond(m, feat, op, lo, hi)
    else:
        c = ctx.cond(m, feat, v.op or ">=", v.lo, v.hi)
    return [ctx.hit(m, [c], notes=notes)]


def _h_rs(m: re.Match, ctx: _Ctx) -> list[_Hit]:
    feat, notes = _rs_feature(m, ctx)
    val = m.groupdict().get("val")
    v = _parse_val(val) if val and re.search(r"\d", val) else None
    if v is not None:
        c = ctx.cond(m, feat, v.op or ">=", v.lo, v.hi)
    else:
        c = ctx.cond(m, feat, "<" if _sign_word(val) < 0 else ">", 0)
    return [ctx.hit(m, [c], notes=notes)]


def _h_drawdown(bare: str = "<=") -> Handler:
    def h(m: re.Match, ctx: _Ctx) -> list[_Hit] | None:
        v = _parse_val(m.group("val"))
        if v is None:
            return None
        op, lo, hi = _below(v, bare=bare)
        notes = []
        if v.op is None and v.hi is None:
            notes.append(f"'{ctx.quote(m)}' read as at least {abs(v.lo):g}% below the 52-week high.")
        if re.search(r"all[\s-]time", m.group(0)):
            notes.append("The catalog measures drawdowns from the 52-week high; used it in place of the all-time high.")
            ctx.unsupported_add(f"drawdown from the all-time high ('{ctx.quote(m)}')")
        return [ctx.hit(m, [ctx.cond(m, "drawdown_from_52w_high_pct", op, lo, hi)], notes=notes)]

    return h


def _h_within_high(m: re.Match, ctx: _Ctx) -> list[_Hit] | None:
    v = _parse_val(m.group("val"))
    if v is None:
        return None
    return [ctx.hit(m, [ctx.cond(m, "drawdown_from_52w_high_pct", ">=", -abs(v.lo))])]


def _h_within_low(m: re.Match, ctx: _Ctx) -> list[_Hit] | None:
    v = _parse_val(m.group("val"))
    if v is None:
        return None
    return [ctx.hit(m, [ctx.cond(m, "above_52w_low_pct", "<=", abs(v.lo))])]


def _h_off_low(m: re.Match, ctx: _Ctx) -> list[_Hit] | None:
    v = _parse_val(m.group("val"))
    if v is None:
        return None
    op = v.op or ">="
    return [ctx.hit(m, [ctx.cond(m, "above_52w_low_pct", op, abs(v.lo), abs(v.hi) if v.hi is not None else None)])]


_DOWN_WORDS = re.compile(r"down|fallen|fell|falling|dropped|declined|declining|lost|off|under")


def _h_return(m: re.Match, ctx: _Ctx) -> list[_Hit] | None:
    v = _parse_val(m.group("val"))
    months = _months(m.groupdict().get("wn"), m.groupdict().get("wu"))
    if v is None or months is None:
        return None
    feat, approx = _nearest(months, _RET_WINDOWS)
    notes = []
    if approx:
        notes.append(f"'{ctx.quote(m)}': returns are available over 1/3/6/12 months; used {feat}.")
        ctx.unsupported_add(f"return window of {months:.3g} months ('{ctx.quote(m)}')")
    d = m.groupdict().get("dir")
    if d and _DOWN_WORDS.match(d):
        op, lo, hi = _below(v, bare="<=")
        c = ctx.cond(m, feat, op, lo, hi)
    else:
        c = ctx.cond(m, feat, v.op or ">=", v.lo, v.hi)
    return [ctx.hit(m, [c], notes=notes)]


def _h_rsi(m: re.Match, ctx: _Ctx) -> list[_Hit] | None:
    v = _parse_val(m.group("val"))
    if v is None:
        return None
    op = v.op
    if op is None:
        around = ctx.text[max(0, m.start() - 40):m.end() + 15]
        if "oversold" in around:
            op = "<="
        elif "overbought" in around:
            op = ">="
        else:
            return None
    return [ctx.hit(m, [ctx.cond(m, "rsi_14", op, v.lo, v.hi)])]


def _h_vol_ratio(m: re.Match, ctx: _Ctx) -> list[_Hit] | None:
    v = _parse_val(m.group("val"))
    if v is None:
        return None
    return [ctx.hit(m, [ctx.cond(m, "max_volume_ratio_20d", v.op or ">=", v.lo, v.hi)],
                    notes=[f"'{ctx.quote(m)}' read as max_volume_ratio_20d (peak session volume in the last 20 vs the 120-session average)."])]


def _h_relvol(m: re.Match, ctx: _Ctx) -> list[_Hit] | None:
    v = _parse_val(m.group("val"))
    if v is None:
        return None
    feat = "rel_volume_5d" if (m.groupdict().get("w") == "5" or m.groupdict().get("w2") == "5") else "rel_volume_20d"
    return [ctx.hit(m, [ctx.cond(m, feat, v.op or ">=", v.lo, v.hi)])]


def _h_iv_rv(m: re.Match, ctx: _Ctx) -> list[_Hit]:
    rich = not re.match(r"below|under|lower|cheap", m.group("rel"))
    return [ctx.hit(m, [ctx.cond(m, "iv_to_realized_vol_ratio", ">" if rich else "<", 1)])]


def _h_putcall(m: re.Match, ctx: _Ctx) -> list[_Hit] | None:
    v = _parse_val(m.group("val"))
    if v is None:
        return None
    feat = "put_call_oi_ratio" if m.group("oi") else "put_call_volume_ratio"
    return [ctx.hit(m, [ctx.cond(m, feat, v.op or ">=", v.lo, v.hi)])]


_UP_WORDS = re.compile(r"ris|rais|increas|climb|grow|up|build|higher|expand|improv|widen|positive|upward|beat|top|exceed|gain")


def _h_si_change(m: re.Match, ctx: _Ctx) -> list[_Hit]:
    up = bool(_UP_WORDS.match(m.group("d")))
    v = _parse_val(m.groupdict().get("val"))
    if v is None:
        c = ctx.cond(m, "short_interest_change_1m_pct", ">" if up else "<", 0)
    elif up:
        c = ctx.cond(m, "short_interest_change_1m_pct", v.op or ">=", abs(v.lo))
    else:
        op, lo, hi = _below(v)
        c = ctx.cond(m, "short_interest_change_1m_pct", op, lo, hi)
    return [ctx.hit(m, [c], notes=[f"'{ctx.quote(m)}' read as the 1-month change in short interest."])]


def _h_multiple_prefix(m: re.Match, ctx: _Ctx) -> list[_Hit] | None:
    v = _parse_val(m.group("val"))
    if v is None:
        return None
    what = m.group("what")
    feat = "ev_to_ebitda" if what == "ebitda" else "ev_to_sales" if what.startswith(("sales", "revenue")) else "pe_ntm"
    notes = [f"'{ctx.quote(m)}' read as {feat}."]
    return [ctx.hit(m, [ctx.cond(m, feat, v.op or "<=", v.lo, v.hi)], notes=notes)]


def _h_netdebt_x(m: re.Match, ctx: _Ctx) -> list[_Hit] | None:
    v = _parse_val(m.group("val"))
    if v is None:
        return None
    return [ctx.hit(m, [ctx.cond(m, "net_debt_to_ebitda", v.op or "<=", v.lo, v.hi)])]


def _h_digit_growth(m: re.Match, ctx: _Ctx) -> list[_Hit]:
    floor = 100 if m.group("k") == "triple" else 10
    what = m.groupdict().get("what") or ""
    feat = "eps_growth_ntm_est_pct" if re.search(r"eps|earnings", what) else "revenue_growth_yoy_pct"
    notes = [f"'{ctx.quote(m)}' read as {feat} >= {floor}."]
    if feat == "eps_growth_ntm_est_pct":
        notes.append("EPS growth uses next-12m consensus EPS vs TTM EPS (eps_growth_ntm_est_pct).")
    return [ctx.hit(m, [ctx.cond(m, feat, ">=", floor)], notes=notes)]


def _h_accel(m: re.Match, ctx: _Ctx) -> list[_Hit]:
    c = ctx.cond(m, "revenue_growth_last_q_yoy_pct", ">", other="revenue_growth_yoy_pct")
    return [ctx.hit(m, [c], notes=["Accelerating growth read as latest-quarter YoY revenue growth above TTM revenue growth."])]


def _h_margin_change(m: re.Match, ctx: _Ctx) -> list[_Hit]:
    kind = m.group("k") or "operating"
    feat = "gross_margin_change_yoy_pp" if kind == "gross" else "operating_margin_change_yoy_pp"
    up = bool(_UP_WORDS.match(m.group("d")))
    raw = m.groupdict().get("val")
    v = _parse_val(raw)
    if v is None:
        c = ctx.cond(m, feat, ">" if up else "<", 0)
    else:
        scale = 0.01 if raw and re.search(r"bps|basis", raw) else 1.0
        mag = abs(v.lo) * scale
        c = ctx.cond(m, feat, (v.op or ">=") if up else _FLIP[v.op or ">="], mag if up else -mag)
    notes = [] if m.group("k") else [f"'{ctx.quote(m)}' read as operating-margin change."]
    return [ctx.hit(m, [c], notes=notes)]


def _h_surprise(m: re.Match, ctx: _Ctx) -> list[_Hit]:
    beat = not re.search(r"miss|negative|disappoint|short\s+of", m.group(0))
    return [ctx.hit(m, [ctx.cond(m, "last_eps_surprise_pct", ">" if beat else "<", 0)])]


def _h_revision_dir(m: re.Match, ctx: _Ctx) -> list[_Hit]:
    d = m.group("d")
    down = re.match(r"cut|lower|fall|coming\s+down|going\s+down|revised\s+(?:down|lower)|reduc|declin|down|slash|negative|trim", d)
    up = not down
    what = m.groupdict().get("what") or ""
    feat = "revenue_revision_3m_pct" if re.search(r"revenue|sales", what) else "eps_revision_3m_pct"
    return [ctx.hit(m, [ctx.cond(m, feat, ">" if up else "<", 0)],
                    notes=[f"'{ctx.quote(m)}' read as the 3-month change in consensus ({feat})."])]


def _h_revision_num(m: re.Match, ctx: _Ctx) -> list[_Hit] | None:
    v = _parse_val(m.group("val"))
    if v is None or v.op is None and v.hi is None:
        return None
    what = m.groupdict().get("what") or ""
    feat = "revenue_revision_3m_pct" if re.search(r"revenue|sales", what) else "eps_revision_3m_pct"
    return [ctx.hit(m, [ctx.cond(m, feat, v.op or ">=", v.lo, v.hi)])]


def _h_analysts(m: re.Match, ctx: _Ctx) -> list[_Hit] | None:
    v = _parse_val(m.group("val"))
    if v is None:
        return None
    return [ctx.hit(m, [ctx.cond(m, "num_analysts", v.op or ">=", v.lo, v.hi)])]


def _days(n: str | None, unit: str | None, default: float = 30) -> float:
    if not unit:
        return default
    k = float(n) if n and n.isdigit() else float(_WORDNUM.get(n or "a", 1))
    u = unit.rstrip("s")
    return k * {"day": 1, "week": 7, "month": 30}.get(u, 1)


def _h_since_earnings(m: re.Match, ctx: _Ctx) -> list[_Hit]:
    d = _days(m.groupdict().get("n"), m.groupdict().get("u"))
    notes = [] if m.groupdict().get("u") else [f"'{ctx.quote(m)}' read as reported within the last 30 days."]
    return [ctx.hit(m, [ctx.cond(m, "days_since_last_earnings", "<=", d)], notes=notes)]


def _h_to_earnings(m: re.Match, ctx: _Ctx) -> list[_Hit]:
    d = _days(m.groupdict().get("n"), m.groupdict().get("u"))
    notes = [] if m.groupdict().get("u") else [f"'{ctx.quote(m)}' read as reporting within the next 30 days."]
    return [ctx.hit(m, [ctx.cond(m, "days_to_next_earnings", "<=", d)], notes=notes)]


def _h_vol_level(m: re.Match, ctx: _Ctx) -> list[_Hit] | None:
    v = _parse_val(m.group("val"))
    if v is None:
        return None
    feat = "volatility_20d_pct" if m.groupdict().get("w") in ("20", "1") else "volatility_60d_pct"
    return [ctx.hit(m, [ctx.cond(m, feat, v.op or "<=", v.lo, v.hi)])]


def _h_sector_exclude(m: re.Match, ctx: _Ctx) -> list[_Hit]:
    for label in _sector_labels(m.group("list")):
        if label not in ctx.exclude_sectors:
            ctx.exclude_sectors.append(label)
    return []


def _h_sector_include(m: re.Match, ctx: _Ctx) -> list[_Hit] | None:
    labels = _sector_labels(m.group("list"))
    if not labels:
        return None
    return [ctx.hit(m, [ctx.cond(m, "gics_sector", "in", values=labels)])]


def _h_exchange(m: re.Match, ctx: _Ctx) -> list[_Hit]:
    x = (m.groupdict().get("x") or m.groupdict().get("y") or "").upper()
    return [ctx.hit(m, [ctx.cond(m, "exchange", "in", values=[x])])]


def _h_geo(m: re.Match, ctx: _Ctx) -> list[_Hit]:
    ctx.unsupported_add(f"non-US universe ('{ctx.quote(m)}')")
    ctx.note("The screen covers US-listed equities only; the non-US request is listed as unsupported.")
    return []


def _h_consume(note: str | None = None) -> Handler:
    def h(m: re.Match, ctx: _Ctx) -> list[_Hit]:
        if note:
            ctx.note(note.format(q=ctx.quote(m)))
        return []

    return h


_CAP_BANDS: dict[str, tuple[float | None, float | None]] = {
    "micro": (None, 0.3), "nano": (None, 0.3), "small": (0.3, 2.0), "smid": (0.3, 10.0), "mid": (2.0, 10.0),
    "large": (10.0, None), "big": (10.0, None), "mega": (200.0, None),
}


def _h_cap_band(m: re.Match, ctx: _Ctx) -> list[_Hit]:
    bands = [b for b in (m.group("b"), m.groupdict().get("b2")) if b]
    los = [_CAP_BANDS[b][0] for b in bands]
    his = [_CAP_BANDS[b][1] for b in bands]
    lo = None if any(x is None for x in los) else min(x for x in los if x is not None)
    hi = None if any(x is None for x in his) else max(x for x in his if x is not None)
    if lo is not None and hi is not None:
        c = ctx.cond(m, "market_cap_usd_bn", "between", lo, hi, default="default band")
    elif lo is not None:
        c = ctx.cond(m, "market_cap_usd_bn", ">", lo, default="default band")
    else:
        c = ctx.cond(m, "market_cap_usd_bn", "<", hi, default="default band")
    return [ctx.hit(m, [c], covered_by={"market_cap_usd_bn"},
                    notes=[f"'{ctx.quote(m)}' read as {c.describe()} (default size band, USD bn)."])]


_TREND_FEATURES = {"sma_50_vs_sma_200_pct", "price_vs_sma_200_pct", "price_vs_sma_50_pct", "golden_cross_20d",
                   "sma_200_slope_1m_pct"}
_MOM_FEATURES = {"return_12m_ex_1m_pct", "return_3m_pct", "return_6m_pct", "return_12m_pct", "rel_strength_3m_pp",
                 "rel_strength_6m_pp", "rel_strength_12m_pp", "return_6m_percentile"}
_VOLUME_FEATURES = {"max_volume_ratio_20d", "rel_volume_5d", "rel_volume_20d"}
_VALUE_FEATURES = {"fcf_yield_pct", "ev_to_ebitda", "ev_to_sales", "pe_ntm", "earnings_yield_ntm_pct"}
_GROWTH_FEATURES = {"revenue_growth_yoy_pct", "revenue_growth_last_q_yoy_pct", "revenue_growth_ntm_est_pct",
                    "eps_growth_ntm_est_pct"}
_LEVERAGE_FEATURES = {"net_debt_to_ebitda", "net_debt_usd_bn", "interest_coverage"}
_IV_FEATURES = {"iv_rank_1y", "iv_30d_pct", "iv_to_realized_vol_ratio"}


def _h_trend_word(m: re.Match, ctx: _Ctx) -> list[_Hit]:
    d = m.group("d") or m.group("d2") or m.group("d3") or "up"
    up = d.startswith(("up", "higher"))
    return _fixed("sma_50_vs_sma_200_pct", ">" if up else "<", 0, covered_by=_TREND_FEATURES,
                  default="default for '" + ("uptrend" if up else "downtrend") + "'")(m, ctx)


def _h_momentum_word(m: re.Match, ctx: _Ctx) -> list[_Hit]:
    neg = _sign_word(m.group("adj")) < 0
    return _fixed("return_12m_ex_1m_pct", "<" if neg else ">", 0, covered_by=_MOM_FEATURES,
                  default="default for 'momentum'")(m, ctx)


# ------------------------------------------------------------------------------------------------
# Rule table (order matters: specific phrases first; consumed text is masked for later rules)
# ------------------------------------------------------------------------------------------------


def _build_rules() -> list[_Rule]:
    P, X, D = _PCT, _X, _DAYS
    rules: list[_Rule] = []
    add = rules.extend

    # ---- universe floors -------------------------------------------------------------------------
    adv_subject = (
        r"(?:(?:20[\s-]?day\s+)?(?:average\s+|avg\.?\s+|mean\s+)?(?:daily\s+)?(?:dollar|\$)\s+(?:trading\s+)?volume"
        r"|adv|liquidity|daily\s+(?:value\s+)?traded|average\s+daily\s+value\s+traded)"
    )
    add([_r(rf"\b{adv_subject}{_GLUE}(?P<val>{_money_valre()})", _h_adv)])
    add([
        _r(rf"\b(?:share\s+)?prices?{_GLUE}(?P<val>{_COMP}\s*\$\s*{_UNUM}|\$\s*{_UNUM}\s*\+)(?![\d.]*\s*(?:b|bn|m|mn|k|billion|million)\b)", _h_price),
        _r(rf"\bno\s+(?:stocks?\s+|names\s+)?(?:priced\s+)?(?:under|below)\s+(?P<val>\$\s*{_UNUM})(?![\d.]*\s*(?:b|bn|m|mn|k|billion|million)\b)",
           _h_price_floor_words, negatable=False),
        _r(rf"\b(?:stocks?|shares|names)\s+(?:trading\s+|priced\s+)?(?P<val>(?:above|over|at\s+least|under|below|less\s+than)\s+\$\s*{_UNUM})(?![\d.]*\s*(?:b|bn|m|mn|k|billion|million)\b)", _h_price),
        _r(r"\b(?:no|excluding|exclude|ex|avoid(?:ing)?|without)[\s-]+penny[\s-]stocks?\b",
           _h_consume("'{q}': kept the default $5 minimum price."), negatable=False),
        _r(r"\b(?:highly\s+|very\s+|sufficiently\s+)?liquid\b(?!ity)", _h_consume("'{q}': kept the default liquidity floor ($5mn 20-day ADV).")),
    ])

    # ---- size ------------------------------------------------------------------------------------
    mc_subject = r"(?:market[\s-]+cap(?:itali[sz]ations?)?s?|mkt\.?[\s-]*caps?|market\s+values?|valued\s+at|worth|valuations?\s+of)"
    add([
        _r(rf"\b{mc_subject}{_GLUE}(?P<val>{_money_valre()})", _h_mcap()),
        _r(rf"(?<![\w.])(?P<val>(?:{_COMP}\s*)?\$?\s*{_UNUM}{_MONEY_UNIT}?(?:\s*(?:-|to)\s*\$?\s*{_UNUM}{_MONEY_UNIT})?{_SUF}?)\s+(?:in\s+)?(?:market[\s-]+cap(?:itali[sz]ation)?|mkt\.?\s*cap|market\s+value)\b", _h_mcap()),
        _r(rf"(?<![\w$])(?P<val>\$\s*{_UNUM}{_MONEY_UNIT}?\s*(?:-|to)\s*\$?\s*{_UNUM}{_BIG_UNIT})", _h_mcap_free),
        _r(rf"(?P<val>{_COMP}\s*\$\s*{_UNUM}{_BIG_UNIT}{_SUF}?)", _h_mcap_free),
        _r(rf"(?<![\w$])(?P<val>\$\s*{_UNUM}{_BIG_UNIT}\s*\+)", _h_mcap_free),
    ])

    # ---- trend -----------------------------------------------------------------------------------
    add([
        _r(rf"\b{_ma(50)}\s+(?:(?:is|are|has|have|was|were|now|still|trading|sitting|holding|remains?|stays?|already|just)\s+)*"
           rf"(?P<rel>above|over|>|below|under|<|(?:crossed|crossing|cross(?:es)?|moved|moving)\s+(?:above|over|below|under))\s+{_ma(200)}",
           _h_ma_cross),
        _r(r"\b(?:(?P<recent>recent(?:ly)?|fresh|new|just)\s+(?:(?:had|formed|printed|made|seen|saw|triggered)\s+)?(?:an?\s+)?)?"
           r"(?:an?\s+)?(?P<kind>golden|death)[\s-]cross(?:es|ed|over)?"
           r"(?P<when>\s+(?:in|within|over|during)\s+the\s+(?:last|past|previous)\s+(?:\d+|one|two|three|four|few|couple\s+of)?\s*(?:days?|weeks?|sessions?|months?))?",
           _h_cross_word),
    ])
    for n in (200, 50, 20):
        add([
            _r(rf"\bwithin\s+(?P<val>{_UNUM}{P}?)\s+of\s+{_ma(n)}", _h_within_ma(n)),
            _r(rf"(?:(?P<val>{_valre(P)})\s+)?\b(?P<rel>above|over|below|under)\s+{_ma(n)}", _h_price_vs(n)),
        ])
    add([
        _r(rf"\b(?:{_SLOPE_UP}|{_SLOPE_DN}|turning\s+(?:up|down))\s+{_ma(200)}", _h_slope),
        _r(rf"\b{_ma(200)}\s+(?:is\s+|has\s+been\s+|that\s+is\s+)?(?:{_SLOPE_UP}|{_SLOPE_DN}|sloping\s+(?:up|down)(?:ward)?|turn(?:ing|ed)\s+(?:up|down))\b", _h_slope),
    ])

    # ---- momentum / relative strength ------------------------------------------------------------
    mom121 = r"12\s*(?:-|minus|ex|to|/)\s*1(?:[\s-]*months?)?\s+(?:price\s+)?(?:momentum|mom|returns?|performance)\b"
    adj = r"(?:(?P<adj>positive|negative|strong|weak|good|poor|high|low|rising|falling)\s+)?"
    tail = rf"(?:{_GLUE}(?P<val>{_COMP}\s*{_NUM}{P}?|positive|negative))?"
    add([
        _r(rf"\b{adj}{mom121}{tail}", _h_mom121),
        _r(rf"\b{adj}12[\s-]?month\s+(?:price\s+)?(?:momentum|returns?)\s*\(?\s*(?:excluding|ex\.?|skipping|less|minus)\s+(?:the\s+)?(?:last|latest|most\s+recent|recent)\s+month\s*\)?{tail}", _h_mom121),
    ])
    bench = r"(?:the\s+)?(?:broad(?:er)?\s+)?(?P<bench>market|s&p(?:\s*500)?|spx|spy|index|benchmark|russell(?:\s*\d+)?|nasdaq(?:\s*100)?|qqq|iwm)"
    by = rf"(?:\s+by\s+(?P<val>(?:{_COMP}\s*)?{_UNUM}{_PP}?))?"
    add([
        _r(rf"\b(?P<verb>out-?perform(?:ing|ed|s|ers?)?|beat(?:ing|en|s)?|ahead\s+of|under-?perform(?:ing|ed|s|ers?)?|lag(?:ging|ged|s|gards?)?|behind)\s+{bench}\b{by}{_WINDOW}?(?:\s+by\s+(?P<val2>(?:{_COMP}\s*)?{_UNUM}{_PP}?))?", _h_outperform),
        _r(rf"\b(?P<wn>\d+|three|six|twelve|one)[\s-]?(?P<wu>months?|m|mo|years?|yr)\s+relative\s+strength(?!\s+index)(?:\s+(?:vs\.?|versus|against|relative\s+to)\s+{bench})?(?:{_GLUE}(?P<val>{_valre(_PP)}|positive|negative))?", _h_rs),
        _r(rf"\brelative\s+strength(?!\s+index)(?:\s+(?:vs\.?|versus|against|relative\s+to)\s+{bench})?{_WINDOW}?(?:{_GLUE}(?P<val>{_valre(_PP)}|positive|negative))?", _h_rs),
    ])

    # ---- range / drawdown ------------------------------------------------------------------------
    dd_verb = (
        r"(?:pulled[\s-]back|pull[\s-]?backs?|pulling[\s-]back|down|declined|declining|dropped|dropping|fallen|fell|falling|off|"
        r"corrected|correcting|sold[\s-]off|selling[\s-]off|retreated|retraced|slid|slumped|lower|trading|trade|traded|sitting|"
        r"sit|are|is|been|be|now|still)"
    )
    add([
        _r(rf"\bwithin\s+(?P<val>{_UNUM}{P}?)\s+of\s+{_HIGH}", _h_within_high),
        _r(rf"(?:\b{dd_verb}\s+)*(?:by\s+)?(?:of\s+)?(?P<val>{_valre(P)})\s+(?:from|off|below|under|beneath|short\s+of)\s+{_HIGH}", _h_drawdown()),
        _r(rf"\b(?:drawdowns?|pull[\s-]?backs?|declines?|corrections?)(?:\s+from\s+{_HIGH})?{_GLUE}(?P<val>{_valre(P)})", _h_drawdown()),
        _r(rf"(?<![\w.])(?P<val>{_valre(P)})\s+(?:drawdowns?|pull[\s-]?backs?|declines?|corrections?)(?:\s+from\s+{_HIGH})?", _h_drawdown()),
        _r(rf"\bwithin\s+(?P<val>{_UNUM}{P}?)\s+of\s+{_LOW}", _h_within_low),
        _r(rf"\b(?:up|bounced|rallied|rebounded|risen|off)\s+(?P<val>{_valre(P)})\s+(?:from|off)\s+{_LOW}", _h_off_low),
    ])

    # ---- returns over a window -------------------------------------------------------------------
    add([
        _r(rf"\b(?P<dir>up|gained|gaining|rallied|rallying|risen|rose|rising|returned|climbed|advanced|down|fallen|fell|falling|dropped|declined|declining|lost|off)\s+(?:by\s+)?(?P<val>{_valre(P)}){_WINDOW}", _h_return),
        _r(rf"\b(?P<wn>\d+|one|three|six|twelve)[\s-]?(?P<wu>months?|m|mo|years?|yr)\s+(?:total\s+)?(?:returns?|performance|price\s+change){_GLUE}(?P<val>{_valre(P)})", _h_return),
    ])

    # ---- oscillators -----------------------------------------------------------------------------
    rsi = r"(?:(?:14[\s-]?(?:day|d|session|period)\s+)?(?:wilder(?:'s)?\s+)?rsi(?:\s*\(\s*14\s*\)|[\s-]?14\b)?|relative\s+strength\s+index(?:\s*\(\s*14\s*\))?)"
    add([
        _r(rf"\b{rsi}{_GLUE}(?P<val>{_valre(P)})", _h_rsi),
        _r(r"\bmacd\s+(?:line\s+)?(?:bullish\s+)?(?:cross(?:over|ed|es)?|crossing)(?:\s+(?:above|over)\s+(?:the\s+)?signal(?:\s+line)?)?|\bbullish\s+macd\s+cross(?:over)?",
           _fixed("macd_bullish_cross_10d", "==", 1, note="MACD cross read as a bullish cross within the last 10 sessions.")),
        _r(r"\b(?:positive\s+macd(?:\s+histogram)?|macd\s+(?:histogram\s+)?(?:is\s+)?(?:positive|above\s+(?:the\s+)?signal(?:\s+line)?))\b",
           _fixed("macd_histogram", ">", 0)),
        _r(r"\b(?:negative\s+macd(?:\s+histogram)?|macd\s+(?:histogram\s+)?(?:is\s+)?(?:negative|below\s+(?:the\s+)?signal(?:\s+line)?))\b",
           _fixed("macd_histogram", "<", 0)),
    ])

    # ---- volume ----------------------------------------------------------------------------------
    add([
        _r(rf"\b(?:(?P<w>5|20)[\s-]?(?:day|d)\s+)?(?:relative|rel\.?)\s+volume(?:\s*\(\s*(?P<w2>5|20)[\s-]?d(?:ay)?\s*\))?{_GLUE}(?P<val>{_valre(X)})", _h_relvol),
        _r(rf"\b(?:(?:on\s+)?volume\s+(?:of\s+|at\s+|was\s+|is\s+|running\s+)?)?(?P<val>(?:{_COMP}\s*)?{_UNUM})\s*(?:x|times)\s+(?:the\s+|its\s+|their\s+)?(?:(?:20|50|60|120)[\s-]?day\s+)?(?:average|avg\.?|normal|usual|typical|mean)(?:\s+daily)?\s+volume\b", _h_vol_ratio),
        _r(rf"\bvolume\s+(?:of\s+|at\s+)?(?P<val>(?:{_COMP}\s*)?{_UNUM})\s*(?:x|times)\s+(?:the\s+)?(?:average|avg\.?|normal|usual|typical|mean)\b", _h_vol_ratio),
        *_subject(r"up[\s/-]*(?:to[\s-])?down\s+volume(?:\s+ratio)?", "up_down_volume_ratio_50d", X, ">=", prefix=False),
    ])

    # ---- options ---------------------------------------------------------------------------------
    iv = r"(?:iv|implied\s+vol(?:atility)?)"
    add([
        _r(rf"\b{iv}\s+(?:is\s+|trading\s+)?(?P<rel>above|over|higher\s+than|greater\s+than|exceeding|rich\s+(?:to|vs\.?|versus|relative\s+to)|below|under|lower\s+than|cheap\s+(?:to|vs\.?|versus|relative\s+to))\s+(?:the\s+)?(?:realized|realised|historical|actual)(?:\s+vol(?:atility)?)?\b", _h_iv_rv),
        _r(r"\boptions?\s+(?:are\s+)?pric(?:ing|e)\s+(?:in\s+)?more\s+risk\s+than\s+(?:realized|realised|historical)(?:\s+vol(?:atility)?)?",
           _fixed("iv_to_realized_vol_ratio", ">", 1)),
        *_subject(r"iv\s*/\s*rv|iv[\s-]to[\s-](?:realized|realised|rv)(?:\s+vol(?:atility)?)?(?:\s+ratio)?|implied[\s-]to[\s-]reali[sz]ed(?:\s+vol(?:atility)?)?(?:\s+ratio)?",
                  "iv_to_realized_vol_ratio", X, ">=", prefix=False),
        *_subject(rf"{iv}\s+(?:rank|percentile)(?:\s*\(\s*1y\s*\))?", "iv_rank_1y", P, ">=", prefix=False),
        *_subject(rf"(?:30[\s-]?day\s+)?(?:atm\s+)?{iv}(?!\s+(?:rank|percentile))", "iv_30d_pct", P, ">=", prefix=False),
    ])
    add([_r(rf"\bput[\s/-]*(?:to[\s-])?call(?P<oi>\s+(?:open\s+interest|oi))?(?:\s+(?:volume\s+)?ratio)?{_GLUE}(?P<val>{_valre(X)})", _h_putcall)])

    # ---- short interest --------------------------------------------------------------------------
    si_change = r"rising|increasing|climbing|growing|up|building|higher|falling|declining|dropping|down|shrinking|lower"
    si = r"(?:short\s+interest(?:\s+(?:as\s+a\s+)?(?:%|percent(?:age)?|share|pct)\s+of\s+(?:the\s+)?(?:free\s+)?float)?|short\s+float|short\s*%\s*(?:of\s+)?float)"
    si_unit = rf"(?:{P}(?:\s+of\s+(?:the\s+)?(?:free\s+)?(?:float|shares(?:\s+outstanding)?))?)"
    add([
        _r(rf"\bshort\s+interest\s+(?:has\s+been\s+|is\s+|was\s+|that\s+is\s+)?(?P<d>{si_change})\b(?:\s+(?:by\s+)?(?P<val>(?:{_COMP}\s*)?{_UNUM}{P}))?(?:\s+(?:over|in)\s+the\s+(?:last|past)\s+month)?", _h_si_change),
        _r(r"\b(?P<d>rising|increasing|climbing|growing|building|falling|declining|dropping|shrinking)\s+short\s+interest\b", _h_si_change),
        _r(rf"\b{si}\b{_GLUE}(?P<val>{_valre(si_unit)})", _numeric("short_interest_pct_float", ">=")),
        _r(rf"(?<![\w.$])(?P<val>(?:{_COMP}\s*)?{_UNUM}{P}(?:\s*\+)?)\s+(?:of\s+(?:the\s+)?(?:free\s+)?float\s+)?(?:(?:sold|held)\s+)?short(?:ed)?(?:\s+interest)?\b",
           _numeric("short_interest_pct_float", ">=")),
        *_subject(r"days[\s-]to[\s-]cover|short\s+ratio|dtc", "days_to_cover", D, ">=", prefix=False),
    ])

    # ---- balance sheet ---------------------------------------------------------------------------
    add([
        _r(rf"\bnet\s+debt{_GLUE}(?P<val>(?:{_COMP}\s*)?{_UNUM})\s*(?:x|times)\s+(?:ttm\s+)?ebitda\b", _h_netdebt_x),
        *_subject(r"net\s+debt\s*(?:/|to|-to-)\s*ebitda|nd\s*/\s*ebitda|net\s+leverage|leverage(?:\s+ratio)?", "net_debt_to_ebitda", X, "<=", prefix=False),
        *_subject(r"interest\s+coverage(?:\s+ratio)?", "interest_coverage", X, ">=", prefix=False),
        _r(r"\bnet[\s-]cash(?:\s+(?:balance\s+sheets?|positions?|on\s+(?:the|their|its)\s+balance\s+sheets?))?\b|\bmore\s+cash\s+than\s+debt\b|\bno\s+net\s+debt\b",
           _fixed("net_debt_usd_bn", "<", 0, note="Net cash read as net_debt_usd_bn < 0."), negatable=False),
        _r(r"\b(?:debt[\s-]free|no\s+debt|zero\s+debt)\b",
           _fixed("net_debt_usd_bn", "<", 0, note="'Debt-free' approximated as net cash (net_debt_usd_bn < 0); the catalog has no gross-debt feature."),
           negatable=False),
    ])

    # ---- valuation -------------------------------------------------------------------------------
    add([
        *_subject(r"(?:fcf|free[\s-]cash[\s-]flows?|free\s+cashflow)\s+yields?", "fcf_yield_pct", P, ">="),
        _r(rf"(?<![\w.])(?P<val>(?:{_COMP}\s*)?{_UNUM}{_SUF}?)\s*(?:x|times)\s+(?:ev\s*/\s*|ev\s+to\s+)?(?:(?:forward|fwd|ntm|trailing|ttm)\s+)?(?P<what>ebitda|sales|revenues?|earnings|eps)\b", _h_multiple_prefix),
        *_subject(r"ev\s*/\s*ebitda|ev[\s-]+to[\s-]+ebitda|enterprise\s+value\s*(?:/|to)\s*ebitda|ebitda\s+multiples?", "ev_to_ebitda", X, "<=", prefix=False),
        *_subject(r"ev\s*/\s*(?:sales|revenues?)|ev[\s-]+to[\s-]+(?:sales|revenues?)|enterprise\s+value\s*(?:/|to)\s*(?:sales|revenues?)", "ev_to_sales", X, "<=", prefix=False),
        *_subject(r"(?:forward|fwd|ntm|next[\s-]12[\s-]months?)\s+(?:p\s*/\s*e|pe|price[\s-]to[\s-]earnings)(?:\s+(?:ratio|multiple))?", "pe_ntm", X, "<=", prefix=False),
        *_subject(r"(?:p\s*/\s*e|pe|price[\s-]to[\s-]earnings)(?:\s+(?:ratio|multiple))?", "pe_ntm", X, "<=", prefix=False,
                  note="P/E read as price / next-12m consensus EPS (pe_ntm)."),
        *_subject(r"(?:(?:forward|ntm)\s+)?earnings\s+yields?", "earnings_yield_ntm_pct", P, ">="),
        *_subject(r"(?:(?:analyst|consensus|street)\s+)?(?:target(?:\s+price)?|price\s+target)\s+upside|upside\s+to\s+(?:the\s+)?(?:analyst\s+|consensus\s+|street\s+|mean\s+)?(?:price\s+)?targets?|(?:analyst\s+)?upside",
                  "target_price_upside_pct", P, ">="),
    ])

    # ---- growth ----------------------------------------------------------------------------------
    rev = r"(?:revenues?|sales|top[\s-]line)"
    add([
        _r(rf"\b(?P<k>double|triple)[\s-]digits?\s+(?P<what>(?:{rev}|eps|earnings)\s+)?growth\b", _h_digit_growth),
        _r(rf"\bgrowing\s+(?:{rev}\s+)?(?:at\s+)?(?:a\s+)?(?P<k>double|triple)[\s-]digits?(?:\s+(?:rate|pace))?\b", _h_digit_growth),
        _r(rf"\baccelerating\s+{rev}\s+growth\b|\b{rev}\s+growth\s+(?:is\s+)?accelerating\b", _h_accel),
        *_subject(rf"(?:quarterly|last[\s-]quarter(?:'s)?|latest[\s-]quarter(?:'s)?|most\s+recent\s+quarter(?:'s)?)\s+{rev}\s+growth", "revenue_growth_last_q_yoy_pct", P, ">="),
        *_subject(rf"(?:forward|fwd|ntm|expected|estimated|consensus|projected|next[\s-](?:12[\s-]months?|year)(?:'s)?)\s+{rev}\s+growth", "revenue_growth_ntm_est_pct", P, ">="),
        *_subject(r"(?:(?:forward|fwd|ntm|expected|estimated|consensus|next[\s-]year(?:'s)?)\s+)?(?:eps|earnings)\s+growth(?:\s+rate)?", "eps_growth_ntm_est_pct", P, ">=",
                  note="EPS growth uses next-12m consensus EPS vs TTM EPS (eps_growth_ntm_est_pct)."),
        *_subject(rf"(?:(?:ttm|trailing|yoy|y/y|annual|year[\s-]over[\s-]year)\s+)?{rev}\s+(?:growth(?:\s+rate)?|(?:is\s+|are\s+)?(?:growing|grew|increasing|rising|up))(?:\s+(?:yoy|y/y|year[\s-]over[\s-]year))?|growing\s+(?:{rev}|the\s+top[\s-]line)(?:\s+(?:by|at))?",
                  "revenue_growth_yoy_pct", P, ">="),
    ])

    # ---- profitability ---------------------------------------------------------------------------
    up_dn = r"expanding|expanded|rising|improving|improved|increasing|widening|up|contracting|contracted|shrinking|falling|declining|down|compressing|compressed"
    chg = rf"(?:\s+(?:by\s+)?(?P<val>(?:{_COMP}\s*)?{_UNUM}\s*(?:pp\b|ppts?\b|percentage\s+points?\b|points?\b|bps\b|basis\s+points\b)))?"
    add([
        _r(r"\b(?P<d>expanding|rising|improving|increasing|widening|contracting|shrinking|falling|declining|compressing)\s+(?P<k>gross|operating|ebit)\s+margins?\b", _h_margin_change),
        _r(rf"\b(?P<k>gross|operating|ebit)\s+margins?\s+(?:are\s+|is\s+|have\s+been\s+|has\s+been\s+)?(?P<d>{up_dn})\b{chg}", _h_margin_change),
        *_subject(r"gross\s+(?:profit\s+)?margins?", "gross_margin_pct", P, ">="),
        *_subject(r"(?:operating|op\.?|ebit)\s+margins?", "operating_margin_pct", P, ">="),
        *_subject(r"ebitda\s+margins?", "ebitda_margin_pct", P, ">="),
        *_subject(r"(?:fcf|free[\s-]cash[\s-]flow)\s+margins?", "fcf_margin_pct", P, ">="),
        *_subject(r"(?:net(?:\s+profit)?|profit)\s+margins?", "net_margin_pct", P, ">="),
        *_subject(r"roe|return\s+on\s+equity", "roe_pct", P, ">="),
        *_subject(r"(?:fcf|free[\s-]cash[\s-]flow|cash)\s+conversion", "fcf_conversion_pct", P, ">="),
    ])

    # ---- estimates -------------------------------------------------------------------------------
    add([
        _r(r"\b(?:beat|beats|topped|exceeded|missed|miss(?:es)?|fell\s+short\s+of)\s+(?:on\s+)?(?:last\s+quarter'?s?\s+)?(?:earnings|eps|consensus|estimates|expectations|the\s+street)(?:\s+(?:last|this)\s+quarter)?"
           r"|\b(?:positive|negative)\s+(?:earnings|eps)\s+surprises?|\b(?:earnings|eps)\s+(?:beats?|miss(?:es)?|disappointments?)\b", _h_surprise),
        *_subject(r"(?:eps|earnings)\s+surprise", "last_eps_surprise_pct", P, ">=", prefix=False),
        _r(rf"\b(?P<what>(?:eps|earnings|revenue|sales)\s+)?(?:estimate\s+|consensus\s+)?revisions?(?:\s*\(3m\))?{_GLUE}(?P<val>{_valre(P)})", _h_revision_num),
        _r(r"\b(?P<what>(?:eps|earnings|revenue|sales|consensus|analyst|street|profit)\s+)?(?:estimates?|forecasts?|numbers|expectations)\s+"
           r"(?:(?:are|is|have|has|been|being|getting|keep|kept|still|now)\s+)*"
           r"(?P<d>rising|raised|going\s+up|moving\s+up|increasing|revised\s+(?:up|higher)(?:wards?)?|up|higher|cut|lowered|falling|coming\s+down|going\s+down|revised\s+(?:down|lower)(?:wards?)?|reduced|declining|down|lower|slashed|trimmed)\b",
           _h_revision_dir),
        _r(r"\b(?P<d>upward|positive|rising|negative|downward|falling)\s+(?P<what>(?:eps|earnings|revenue|sales|estimate|analyst)\s+)?(?:estimate\s+)?revisions?\b", _h_revision_dir),
        _r(r"\b(?P<what>eps|earnings|estimate|revenue)\s+(?P<d>upgrades|downgrades|cuts|raises|increases)\b", _h_revision_dir),
        _r(r"\banalysts?\s+(?:are\s+|have\s+been\s+)?(?P<d>raising|cutting|lowering|upgrading|downgrading|trimming)\s+(?:their\s+)?(?P<what>(?:eps|earnings|revenue|sales)\s+)?(?:estimates|numbers|forecasts|targets)\b", _h_revision_dir),
        _r(rf"\b(?:covered\s+by\s+|with\s+)?(?P<val>(?:{_COMP}\s*)?{_UNUM}{_SUF}?)\s+(?:or\s+more\s+)?analysts?\b", _h_analysts),
        _r(rf"\banalyst\s+coverage{_GLUE}(?:of\s+)?(?P<val>{_valre()})", _h_analysts),
    ])

    # ---- events ----------------------------------------------------------------------------------
    wn = r"(?P<n>\d+|one|two|three|four|a)"
    add([
        _r(rf"\b(?:reported|reporting|released)\s+(?:earnings\s+|results\s+)?(?:in|within|over)\s+the\s+(?:last|past)\s+{wn}\s+(?P<u>days?|weeks?|months?)", _h_since_earnings),
        _r(r"\b(?:just|recently)\s+reported(?:\s+(?:earnings|results))?\b|\bpost[\s-]earnings\b", _h_since_earnings),
        _r(rf"\b(?:earnings|reporting|reports?|results)\s+(?:due\s+)?(?:in|within)\s+the\s+next\s+{wn}\s+(?P<u>days?|weeks?|months?)", _h_to_earnings),
        _r(r"\b(?:ahead\s+of|before|into)\s+(?:the\s+next\s+|upcoming\s+|their\s+next\s+)?(?:earnings|report)\b|\bupcoming\s+earnings\b", _h_to_earnings),
    ])

    # ---- risk ------------------------------------------------------------------------------------
    add([
        *_subject(r"(?:1y\s+|one[\s-]year\s+)?beta", "beta_1y", X, None, prefix=False),
        _r(rf"\b(?:(?P<w>20|60|1|3)[\s-]?(?:day|d|month)\s+)?(?:realized\s+|realised\s+|historical\s+|annuali[sz]ed\s+|price\s+)?volatility{_GLUE}(?P<val>{_valre(P)})", _h_vol_level),
    ])

    # ---- sectors / exchange / geography ----------------------------------------------------------
    add([
        _r(rf"\b(?:ex(?:cluding|clude|cl\.?)?|except(?:\s+for)?|other\s+than|outside(?:\s+of)?|not\s+in|avoid(?:ing)?|without|but\s+not|leaving\s+out|omitting|non|no)[\s-]+(?:the\s+)?(?:any\s+)?(?P<list>{_SECTOR_LIST})(?:\s+(?:sectors?|names|stocks|companies|space))?",
           _h_sector_exclude, negatable=False),
        _r(rf"\b(?:in|within|from|across|among|only)\s+(?:the\s+)?(?P<list>{_SECTOR_LIST})(?:\s+(?:sectors?|space|names|stocks|companies|equities|industr(?:y|ies)))?\b", _h_sector_include),
        _r(rf"(?P<list>{_SECTOR_LIST})\s+(?:sectors?|stocks|names|companies|equities|shares|space|plays)\b", _h_sector_include),
        _r(rf"(?:(?<=cap )|(?<=caps )|(?<=cap-)|(?<=caps-))(?P<list>{_SECTOR_LIST})", _h_sector_include),
        _r(r"\blisted\s+on\s+(?:the\s+)?(?P<x>nyse|nasdaq)\b|\b(?P<y>nyse|nasdaq)[\s-]listed\b", _h_exchange),
        _r(r"\b(?:european|europe|uk|u\.k\.|british|japan(?:ese)?|china|chinese|canad(?:a|ian)|asian?|emerging[\s-]markets?|international|"
           r"non[\s-]u\.?s\.?|ex[\s-]u\.?s\.?|german(?:y)?|french|france|india(?:n)?|australian?|korean?|brazil(?:ian)?|latam|latin\s+america)\b"
           r"(?:\s+(?:stocks|equities|names|companies|markets?))?", _h_geo),
    ])

    # ---- vague words -> documented defaults (dropped when an explicit threshold covers them) -------
    add([
        _r(r"\b(?P<b>micro|nano|small|smid|mid|large|big|mega)(?:[\s-]*(?:and|/|&|or|to)[\s-]*(?P<b2>micro|small|mid|large|mega))?[\s-]?(?:cap(?:s|itali[sz]ations?)?)\b", _h_cap_band),
        _r(r"\b(?:established\s+|clear\s+|strong\s+|long[\s-]term\s+|primary\s+|secular\s+|persistent\s+)?(?P<d>up|down)[\s-]?trends?\b|\btrending\s+(?P<d2>up|higher|down|lower)\b|\b(?P<d3>up|down)trending\b", _h_trend_word),
        _r(rf"\b(?:near|close\s+to|approaching|just\s+below|testing)\s+(?:new\s+)?{_HIGH}",
           _fixed("drawdown_from_52w_high_pct", ">=", -5, covered_by={"drawdown_from_52w_high_pct"}, default="default for 'near highs'")),
        _r(rf"\b(?:at|hitting|making|setting|printing|new)\s+(?:new\s+|fresh\s+)?{_HIGH}",
           _fixed("drawdown_from_52w_high_pct", ">=", -2, covered_by={"drawdown_from_52w_high_pct"}, default="default for 'at/new highs'")),
        _r(rf"\b(?:near|close\s+to|at|hitting|testing|new)\s+(?:new\s+)?{_LOW}",
           _fixed("above_52w_low_pct", "<=", 10, covered_by={"above_52w_low_pct"}, default="default for 'near lows'")),
        _r(r"\b(?:pulled[\s-]back|pull[\s-]?backs?|pulling\s+back|sold[\s-]off|sell[\s-]?offs?|selling\s+off|corrected|correction|beaten[\s-](?:down|up)|dips?|dipped|drawdowns?)\b",
           _fixed("drawdown_from_52w_high_pct", "<=", -10, covered_by={"drawdown_from_52w_high_pct"}, default="default for 'pullback'")),
        _r(r"\b(?P<adj>positive|strong|good|solid|high|rising|improving|negative|weak|poor|low|fading|deteriorating)?\s*(?:price\s+)?momentum\b", _h_momentum_word),
        _r(r"\b(?:deeply\s+|very\s+|technically\s+)?oversold\b", _fixed("rsi_14", "<", 30, covered_by={"rsi_14"}, default="default for 'oversold'")),
        _r(r"\b(?:very\s+|technically\s+)?overbought\b", _fixed("rsi_14", ">", 70, covered_by={"rsi_14"}, default="default for 'overbought'")),
        _r(r"\bvolume\s+(?:surges?|surging|surged|explosions?|spurts?)\b|\b(?:surging|exploding|spiking|a\s+surge\s+in|a\s+spike\s+in)\s+volume\b",
           _fixed("rel_volume_5d", ">=", 1.5, covered_by=_VOLUME_FEATURES, default="default for 'volume surge'")),
        _r(r"\b(?:heavy|high|big|huge|massive|elevated|large|heavier|higher|above[\s-]average|unusual|abnormal|outsized|strong|rising)\s+(?:trading\s+)?volumes?\b|\bvolume\s+spikes?\b|\bcapitulation(?:\s+volume)?\b|\bon\s+volume\b",
           _fixed("max_volume_ratio_20d", ">=", 2, covered_by=_VOLUME_FEATURES, default="default for 'heavy volume'")),
        _r(r"\b(?:accumulation|being\s+accumulated|institutional\s+buying)\b",
           _fixed("up_down_volume_ratio_50d", ">", 1, covered_by={"up_down_volume_ratio_50d"}, default="default for 'accumulation'")),
        _r(r"\bdistribution\b", _fixed("up_down_volume_ratio_50d", "<", 1, covered_by={"up_down_volume_ratio_50d"}, default="default for 'distribution'")),
        _r(r"\b(?:elevated|high|rich|expensive|inflated)\s+(?:implied\s+vol(?:atility)?|iv|option(?:s)?\s+(?:premiums?|prices|vol(?:atility)?)|vol(?:atility)?\s+premiums?)\b|\boptions?\s+(?:are\s+)?(?:expensive|rich)\b",
           _fixed("iv_rank_1y", ">", 50, covered_by=_IV_FEATURES, default="default for 'elevated implied volatility'")),
        _r(r"\b(?:low|cheap|depressed|subdued)\s+(?:implied\s+vol(?:atility)?|iv|option(?:s)?\s+(?:premiums?|prices))\b|\boptions?\s+(?:are\s+)?cheap\b",
           _fixed("iv_rank_1y", "<", 30, covered_by=_IV_FEATURES, default="default for 'cheap implied volatility'")),
        _r(r"\b(?:elevated|high|heavy|large|significant|big|substantial|meaningful|crowded|very\s+high)\s+short\s+interest\b|\bheavily\s+shorted\b|\bcrowded\s+shorts?\b"
           r"|\bshort\s+interest\s+(?:is\s+|remains\s+|that\s+is\s+)?(?:elevated|high|heavy)\b|\bshort[\s-]squeeze(?:\s+candidates?)?\b|\bmost[\s-]shorted\b",
           _fixed("short_interest_pct_float", ">", 10, covered_by={"short_interest_pct_float", "days_to_cover"}, default="default for 'elevated short interest'")),
        _r(r"\b(?:low|little|minimal)\s+short\s+interest\b",
           _fixed("short_interest_pct_float", "<", 3, covered_by={"short_interest_pct_float", "days_to_cover"}, default="default for 'low short interest'")),
        _r(r"\b(?:fcf|free[\s-]cash[\s-]flow)[\s-]positive\b|\bpositive\s+(?:free[\s-]cash[\s-]flows?|fcf)\b",
           _fixed("fcf_yield_pct", ">", 0, covered_by={"fcf_yield_pct", "fcf_margin_pct"}, default="default for 'FCF positive'")),
        _r(r"\b(?:strong|robust|healthy|solid|high|good|significant|ample|abundant|plenty\s+of|lots\s+of|consistent|great)\s+(?:free[\s-]cash[\s-]flows?|fcf|cash\s+flows?|cash\s+generation)(?:\s+generation)?\b|\bcash[\s-]generative\b",
           _fixed("fcf_yield_pct", ">", 5, covered_by={"fcf_yield_pct", "fcf_margin_pct"}, default="default for 'strong free cash flow'")),
        _r(r"\b(?:cheap(?:ly\s+valued)?|inexpensive|undervalued|under-valued|attractively\s+valued|low\s+valuations?|trading\s+at\s+a\s+discount|value\s+(?:stocks|names))\b",
           _fixed("fcf_yield_pct", ">", 5, covered_by=_VALUE_FEATURES, default="default for 'cheap'")),
        _r(r"\b(?:high|strong|fast|rapid|solid|robust|healthy|above[\s-]average)[\s-]+(?:revenue\s+|sales\s+|top[\s-]line\s+)?growth\b|\b(?:high|fast)[\s-]+growers?\b|\bgrowth\s+(?:stocks|names|companies)\b|\bgrowing\s+(?:revenues?|sales|the\s+top[\s-]line)\b",
           _fixed("revenue_growth_yoy_pct", ">", 10, covered_by=_GROWTH_FEATURES, default="default for 'high growth'")),
        _r(r"\b(?:low|little|minimal|modest|conservative|manageable)\s+(?:leverage|debt|net\s+debt)\b|\b(?:strong|clean|solid|pristine|fortress|healthy|robust|good)\s+balance[\s-]sheets?\b|\b(?:un|under)-?levered\b",
           _fixed("net_debt_to_ebitda", "<", 2, covered_by=_LEVERAGE_FEATURES, default="default for 'low leverage'")),
        _r(r"\b(?:high|heavy|elevated|significant)\s+(?:leverage|debt)\b|\b(?:highly|over)[\s-]?levered\b",
           _fixed("net_debt_to_ebitda", ">", 3, covered_by=_LEVERAGE_FEATURES, default="default for 'high leverage'")),
        _r(r"\blow[\s-]beta\b", _fixed("beta_1y", "<", 1, covered_by={"beta_1y"}, default="default for 'low beta'")),
        _r(r"\bhigh[\s-]beta\b", _fixed("beta_1y", ">", 1.2, covered_by={"beta_1y"}, default="default for 'high beta'")),
        _r(r"\bmargin\s+expansion\b|\bexpanding\s+margins?\b|\bmargins?\s+(?:are\s+)?expanding\b",
           _fixed("operating_margin_change_yoy_pp", ">", 0, covered_by={"operating_margin_change_yoy_pp", "gross_margin_change_yoy_pp"}, default="default for 'margin expansion'")),
        _r(r"\bhigh[\s-]gross[\s-]margins?\b",
           _fixed("gross_margin_pct", ">", 50, covered_by={"gross_margin_pct"}, default="default for 'high gross margin'")),
        _r(r"\bhigh[\s-](?:operating[\s-])?margins?\b",
           _fixed("operating_margin_pct", ">", 20, covered_by={"operating_margin_pct"}, default="default for 'high margin'")),
        _r(r"\bunprofitable\b|\bloss[\s-]making\b", _fixed("net_margin_pct", "<", 0, covered_by={"net_margin_pct"}, default="default for 'unprofitable'")),
        _r(r"\bprofitable\b", _fixed("net_margin_pct", ">", 0, covered_by={"net_margin_pct"}, default="default for 'profitable'")),
        _r(r"\b(?:u\.?s\.?|united\s+states|american|domestic)\b(?:[\s-]+(?:listed|based|domiciled))?", _h_consume(), negatable=False),
    ])
    return rules


def _h_price_floor_words(m: re.Match, ctx: _Ctx) -> list[_Hit] | None:
    v = _parse_val(m.group("val"))
    if v is None:
        return None
    ctx.min_price = _num(max(v.lo, ctx.min_price or 0.0))
    ctx.note(f"Price floor {v.lo:g} from '{ctx.quote(m)}' (universe min_price keeps the strictest floor).")
    return []


_RULES = _build_rules()


# ---- pre-passes: top N, ranking clause, downstream (narrative) instructions ---------------------

_TOP_N = [
    re.compile(r"\b(?:top|best|first)\s+(?P<n>\d{1,3})\b(?!\s*(?:%|percent|pct))(?:\s+(?:names|stocks|ideas|candidates|picks|companies|tickers|results))?"),
    re.compile(r"(?<![\w$.,])(?P<n>\d{1,3})\s+(?:names|stocks|ideas|candidates|picks|tickers|results)\b"),
]
_DOWNSTREAM_VERBS = (
    r"read|review|pull|look|check|explain|summari[sz]e|analy[sz]e|dig|listen|go|write|tell|describe|study|scan|show|give|find|list|figure|understand|diagnose"
)
_RANK_CLAUSE = re.compile(
    r"\b(?:rank(?:ed|ing)?|sort(?:ed|ing)?|order(?:ed|ing)?|prioriti[sz](?:e|ed|ing)|scor(?:e|ed|ing))\s+"
    r"(?:them\s+|it\s+|the\s+(?:results|survivors|names|list|candidates|screen)\s+|results\s+|names\s+|survivors\s+|candidates\s+)?"
    r"(?:by|on|using|according\s+to|based\s+on)\s+(?P<body>(?:[^.;!?\x00]|\.(?=\d))+?)"
    rf"(?=\s*(?:,\s*)?(?:and\s+)?then\s+(?:{_DOWNSTREAM_VERBS})\b|\s*[.;!?\x00]|\s*$)"
)
_NARRATIVE = [
    re.compile(
        r"(?:,\s*)?(?:\band\s+)?(?:\bthen\s+)?\b(?:read|review|pull|go\s+through|look\s+(?:at|through)|check|summari[sz]e|analy[sz]e|listen\s+to|dig\s+into|study|scan)\b"
        r"(?:[^.;!?\x00]|\.(?=\d))*?\b(?:earnings[\s-]calls?|calls?|transcripts?|filings?|10-?[kq]s?|8-?ks?|news(?:flow)?|press\s+releases?|"
        r"commentary|conference\s+calls?|guidance|headlines)\b(?:[^.;!?\x00]|\.(?=\d))*"
    ),
    re.compile(
        r"(?:,\s*)?(?:\band\s+)?(?:\bthen\s+)?\b(?:explain|tell\s+me\s+why|describe\s+why|figure\s+out\s+why|diagnose|write\s+(?:up|a\s+note))\b(?:[^.;!?\x00]|\.(?=\d))*"
    ),
]

_H, _L = "higher_is_better", "lower_is_better"
_RANK_VOCAB: list[tuple[re.Pattern, str, str, str]] = [
    (re.compile(rf"\b(?:{p})\b"), f, d, big)
    for p, f, d, big in [
        (r"(?:fcf|free[\s-]cash[\s-]flow)\s+margins?", "fcf_margin_pct", _H, _H),
        (r"(?:fcf|free[\s-]cash[\s-]flows?|cash[\s-]flow)(?:\s+yields?)?", "fcf_yield_pct", _H, _H),
        (r"ev\s*/\s*ebitda|ev[\s-]to[\s-]ebitda|ebitda\s+multiples?", "ev_to_ebitda", _L, _H),
        (r"ev\s*/\s*(?:sales|revenues?)|ev[\s-]to[\s-](?:sales|revenues?)", "ev_to_sales", _L, _H),
        (r"(?:forward\s+)?(?:p\s*/\s*e|pe|price[\s-]to[\s-]earnings)(?:\s+ratio)?", "pe_ntm", _L, _H),
        (r"earnings\s+yields?", "earnings_yield_ntm_pct", _H, _H),
        (r"valuations?|cheapness|value", "ev_to_ebitda", _L, _H),
        (r"(?:analyst\s+|target(?:\s+price)?\s+|price\s+target\s+)?upside", "target_price_upside_pct", _H, _H),
        (r"(?:quarterly|last[\s-]quarter)\s+(?:revenue|sales)\s+growth", "revenue_growth_last_q_yoy_pct", _H, _H),
        (r"(?:forward|ntm|expected|estimated)\s+(?:revenue|sales)\s+growth", "revenue_growth_ntm_est_pct", _H, _H),
        (r"(?:eps|earnings)\s+growth", "eps_growth_ntm_est_pct", _H, _H),
        (r"(?:(?:revenue|sales|top[\s-]line)\s+)?growth|revenues?|sales", "revenue_growth_yoy_pct", _H, _H),
        (r"gross\s+margins?", "gross_margin_pct", _H, _H),
        (r"(?:operating|ebit)\s+margins?", "operating_margin_pct", _H, _H),
        (r"ebitda\s+margins?", "ebitda_margin_pct", _H, _H),
        (r"net\s+margins?|profit\s+margins?", "net_margin_pct", _H, _H),
        (r"margin\s+expansion", "operating_margin_change_yoy_pp", _H, _H),
        (r"margins?|profitability", "operating_margin_pct", _H, _H),
        (r"roe|return\s+on\s+equity", "roe_pct", _H, _H),
        (r"drawdowns?|pull[\s-]?backs?|declines?|sell[\s-]?offs?|corrections?|discount\s+to\s+(?:the\s+)?(?:52[\s-]week\s+)?highs?"
         r"|distance\s+(?:from|below)\s+(?:the\s+)?(?:52[\s-]week\s+)?highs?|(?:how\s+)?far\s+(?:they(?:'ve|\s+have)?\s+|it\s+has\s+|it's\s+)?(?:fallen|dropped)", "drawdown_from_52w_high_pct", _L, _L),
        (r"rsi|oversold|overbought|relative\s+strength\s+index", "rsi_14", _L, _H),
        (r"12[\s-]*(?:-|minus)\s*1(?:\s+month)?\s+momentum|momentum", "return_12m_ex_1m_pct", _H, _H),
        (r"relative\s+strength", "rel_strength_6m_pp", _H, _H),
        (r"days[\s-]to[\s-]cover|short\s+ratio", "days_to_cover", _H, _H),
        (r"short\s+interest|short\s+float", "short_interest_pct_float", _H, _H),
        (r"liquidity|dollar\s+volume|adv", "avg_dollar_volume_20d_usd_mn", _H, _H),
        (r"relative\s+volume", "rel_volume_20d", _H, _H),
        (r"volume(?:\s+(?:spikes?|surges?|ratio))?|capitulation", "max_volume_ratio_20d", _H, _H),
        (r"iv\s+rank", "iv_rank_1y", _H, _H),
        (r"implied\s+vol(?:atility)?|iv", "iv_30d_pct", _H, _H),
        (r"(?:estimate\s+|eps\s+)?revisions?|estimate\s+momentum", "eps_revision_3m_pct", _H, _H),
        (r"(?:earnings|eps)\s+surprises?", "last_eps_surprise_pct", _H, _H),
        (r"net\s+debt\s*(?:/|to)\s*ebitda|leverage|balance\s+sheet(?:\s+strength)?", "net_debt_to_ebitda", _L, _H),
        (r"net\s+cash", "net_debt_usd_bn", _L, _L),
        (r"interest\s+coverage", "interest_coverage", _H, _H),
        (r"market\s+cap(?:itali[sz]ation)?|size", "market_cap_usd_bn", _H, _H),
        (r"beta", "beta_1y", _L, _H),
        (r"volatility", "volatility_60d_pct", _L, _H),
    ]
]
_RANK_BIG = re.compile(r"\b(?:highest|largest|biggest|greatest|most|deepest|strongest|best|steepest|fastest|higher|larger|bigger|deeper|stronger|high|large|big|deep|size\s+of|magnitude\s+of|amount\s+of|depth\s+of|descending)\b")
_RANK_SMALL = re.compile(r"\b(?:lowest|smallest|least|cheapest|weakest|shallowest|slowest|fewest|lower|smaller|shallower|low|small|cheaper|ascending)\b")
_RANK_WEIGHT = re.compile(
    r"\(\s*(?:weight|w|wt)\s*[:=]?\s*(?P<a>\d+(?:\.\d+)?)\s*\)|\bweight(?:ed)?\s*(?:of\s*)?[:=]?\s*(?P<b>\d+(?:\.\d+)?)"
    r"|\(\s*(?P<c>\d+(?:\.\d+)?)\s*%\s*\)|^(?P<d>\d+(?:\.\d+)?)\s*%\s*(?:weight\s+)?(?:on\s+)?|\b(?P<e>\d+(?:\.\d+)?)\s*x\s+weight(?:ed)?\b|\b(?P<f>double)[\s-]weight(?:ed)?\b"
)
_RANK_SPLIT = re.compile(r"\s*(?:,\s*(?:and\s+|then\s+|plus\s+)?|\band\b|&|\bplus\b|\bthen\b|;|\s\+\s)\s*")
_RANK_STRIP = re.compile(r"^(?:(?:the|a|an|their|its|by|on|of|with)\s+)+|\s+(?:first|second|third|last|too|also)$")


def _parse_rank_item(item: str, catalog: FeatureCatalog) -> tuple[RankFactor | None, str]:
    raw = item.strip()
    weight = 1.0
    wm = _RANK_WEIGHT.search(raw)
    if wm:
        g = wm.groupdict()
        weight = 2.0 if g["f"] else float(next(v for k, v in g.items() if k != "f" and v))
        raw = (raw[:wm.start()] + raw[wm.end():]).strip()
    text = _RANK_STRIP.sub("", raw).strip()
    if not text or weight <= 0:
        return None, raw
    for name in sorted(catalog.names(), key=len, reverse=True):
        if re.search(rf"\b{re.escape(name)}\b", text):
            fdef = catalog[name]
            direction = _H if fdef.higher_is_better is not False else _L
            if _RANK_SMALL.search(text) and not _RANK_BIG.search(text):
                direction = _L
            elif _RANK_BIG.search(text) and not _RANK_SMALL.search(text):
                direction = _H
            return RankFactor(feature=name, direction=direction, weight=weight, rationale=f"rank by '{item.strip()}'"), raw
    for rx, feat, default_dir, big_dir in _RANK_VOCAB:
        if not rx.search(text):
            continue
        big, small = bool(_RANK_BIG.search(text)), bool(_RANK_SMALL.search(text))
        direction = default_dir
        if big and not small:
            direction = big_dir
        elif small and not big:
            direction = _L if big_dir == _H else _H
        if feat == "rsi_14":
            direction = _L if "oversold" in text else _H if "overbought" in text else direction
        return RankFactor(feature=feat, direction=direction, weight=weight, rationale=f"rank by '{item.strip()}'"), raw
    return None, raw


# ---- leftover analysis ---------------------------------------------------------------------------

_FILLER = frozenset("""
a an the and or but nor so yet then also plus as such either both each all any some
i me my we us our you your they them their it its this that these those which who whom whose what where when while whilst whether
is are was were be been being am has have had having do does did done get gets got getting
find finding found show showing shows screen screening screens search searching look looking list give identify want need needs
like would could should please can will may might must let lets help
for of in on at to from by with into onto within over across through via per vs versus
stocks stock names name companies company equities equity shares share tickers ticker securities security businesses business
firms firm ideas idea candidates candidate opportunities opportunity situations situation setups setup universe
now currently recently recent still already today lately past last few several couple months month weeks week days day year years
ago since during period time
established clear clearly strong solid robust healthy good nice decent very really quite fairly relatively somewhat generally
typically meaningfully significantly
generating generate generates producing produce produces showing having trading trade trades traded remain remains remained
remaining stay stays stayed holding hold holds seeing seen see
elevated not there here just only even s u
after before following amid despite given along alongside together combined because due
""".split())
_TOKEN = re.compile(r"[a-z][a-z'&-]*|\$?\d+(?:\.\d+)?%?[a-z]*")
_FRAGMENT = re.compile(r"(?:[^\x00,;:!?()\[\]{}\".]|\.(?=\d))+")
_CONJ = re.compile(r"\b(?:and|or|but|with|while|where|whereas|that|which|plus|then)\b")


def _content(tok: str) -> bool:
    parts = [p for p in re.split(r"[-']", tok) if p]
    return any(p not in _FILLER for p in parts)


# ------------------------------------------------------------------------------------------------
# Heuristic translator
# ------------------------------------------------------------------------------------------------


class HeuristicScreenTranslator:
    """Offline, deterministic observation -> ``ScreenSpec`` parser for common screening language.

    Each phrase it recognises maps to catalog features with the threshold the observation states.
    Vague words map to documented defaults - dropped whenever an explicit threshold on the same
    features is present ("oversold (RSI under 40)" -> rsi_14 < 40):

    ==============================  =============================================================
    phrase                          default
    ==============================  =============================================================
    micro / small / mid / large /   market_cap_usd_bn < 0.3 / 0.3-2 / 2-10 / > 10 / > 200 (bands
    mega cap                        combine: "small and mid caps" -> 0.3-10)
    uptrend / downtrend             sma_50_vs_sma_200_pct > 0 / < 0
    golden / death cross            sma_50_vs_sma_200_pct > 0 / < 0; "recent golden cross" ->
                                    golden_cross_20d == 1
    momentum (no window)            return_12m_ex_1m_pct > 0 (negative words: < 0)
    relative strength (no window)   rel_strength_6m_pp > 0
    pulled back / sold off          drawdown_from_52w_high_pct <= -10
    near highs / at new highs       drawdown_from_52w_high_pct >= -5 / >= -2
    near lows                       above_52w_low_pct <= 10
    oversold / overbought           rsi_14 < 30 / > 70
    heavy volume, capitulation      max_volume_ratio_20d >= 2
    volume surge                    rel_volume_5d >= 1.5
    accumulation / distribution     up_down_volume_ratio_50d > 1 / < 1
    strong free cash flow, cheap    fcf_yield_pct > 5 ("FCF positive" -> > 0)
    high growth                     revenue_growth_yoy_pct > 10 ("double-digit" -> >= 10)
    low leverage / strong balance   net_debt_to_ebitda < 2 (high leverage -> > 3)
    sheet
    net cash, debt-free             net_debt_usd_bn < 0
    elevated short interest         short_interest_pct_float > 10 (low -> < 3)
    elevated / cheap implied vol    iv_rank_1y > 50 / < 30
    low / high beta                 beta_1y < 1 / > 1.2
    margin expansion                operating_margin_change_yoy_pp > 0
    profitable / unprofitable       net_margin_pct > 0 / < 0
    estimates rising / being cut    eps_revision_3m_pct > 0 / < 0 (revenue estimates ->
                                    revenue_revision_3m_pct)
    ==============================  =============================================================

    A bare number takes the feature's natural comparison ("FCF yield of 6%" -> >= 6; "EV/EBITDA
    8x" -> <= 8). Market-cap amounts without a unit are USD billions. Drawdowns are stored as
    negative numbers ("down 15-40% from highs" -> between -40 and -15; "down 20%" -> <= -20).
    Two criteria joined only by "or" become an ``any_of`` group. Sector inclusion ("tech stocks",
    "in Energy or Materials") -> gics_sector in [...]; exclusion ("ex-financials") ->
    ``universe.exclude_sectors``. Explicit price / dollar-volume floors replace the universe
    defaults.

    Ranking comes from a "rank / sort by ..." clause. Without one, the default ranking uses every
    condition feature that has a natural catalog direction (``higher_is_better``), equal weights,
    in condition order; failing that, the direction implied by each one-sided condition (">" ->
    higher is better, "<" -> lower is better); failing that, 20-day dollar volume (most liquid
    first). Downstream instructions ("then read the latest earnings calls and explain ...") are
    left to the narrative / explanation stages. Anything not understood is listed verbatim in
    ``unsupported_requests``; negated criteria ("not oversold") are listed there too.
    """

    name = "heuristic"

    def __init__(self, catalog: FeatureCatalog | None = None):
        self.catalog = catalog or default_catalog()
        # Identifier-style names only ("rsi_14 < 35"); plain words such as "price" go through the phrase rules.
        names = sorted((n for n in self.catalog.names() if "_" in n), key=len, reverse=True)
        lit = "|".join(re.escape(n) for n in names) or r"(?!x)x"
        self._rules: list[_Rule] = [
            _r(rf"\b(?P<f>{lit})\b{_GLUE}(?P<val>(?:==|!=|=)\s*{_NUM}|{_valre(_PCT)})", _h_literal(self.catalog)),
            *_RULES,
        ]

    # -- public ------------------------------------------------------------------------------------

    def translate(self, observation: str) -> TranslationResult:
        if not isinstance(observation, str) or not observation.strip():
            raise ScreenTranslationError(["observation is empty"])
        ctx = _Ctx(observation)
        self._pre_passes(ctx)
        for rule in self._rules:
            for m in list(rule.pattern.finditer(ctx.masked)):
                if not m.group(0).strip() or "\x00" in ctx.masked[m.start():m.end()]:
                    continue
                if rule.negatable and ctx.negated(m.start()):
                    neg = _NEGATION.search(ctx.text[max(0, m.start() - 40):m.start()])
                    s = max(0, m.start() - 40) + (neg.start() if neg else 0)
                    ctx.unsupported_add(f"negated criterion: '{ctx.quote((s, m.end()))}'")
                    ctx.consume(s, m.end())
                    continue
                hits = rule.handler(m, ctx)
                if hits is None:
                    continue
                ctx.hits.extend(hits)
                ctx.consume(m.start(), m.end())
        self._leftovers(ctx)
        spec = self._assemble(ctx)
        return TranslationResult(spec=spec, attempts=1, errors_by_round=[[]], translator=self.name)

    # -- passes ------------------------------------------------------------------------------------

    def _pre_passes(self, ctx: _Ctx) -> None:
        for rx in _TOP_N:
            for m in rx.finditer(ctx.masked):
                if "\x00" in ctx.masked[m.start():m.end()]:
                    continue
                n = _clamp_top_n(m.group("n"))
                if ctx.top_n is None:
                    ctx.top_n = n
                    if n != int(m.group("n")):
                        ctx.note(f"top_n clamped to {n} (allowed range {TOP_N_MIN}-{TOP_N_MAX}).")
                ctx.consume(m.start(), m.end())
        for m in list(_RANK_CLAUSE.finditer(ctx.masked)):
            if "\x00" in ctx.masked[m.start():m.end()]:
                continue
            self._parse_ranking(ctx, m)
            ctx.consume(m.start(), m.end())
        for rx in _NARRATIVE:
            for m in list(rx.finditer(ctx.masked)):
                if "\x00" in ctx.masked[m.start():m.end()] or not m.group(0).strip(" ,"):
                    continue
                ctx.note(f"Left to the narrative / explanation stages (not a screen condition): '{ctx.quote(m).strip(' ,')}'.")
                ctx.consume(m.start(), m.end())

    def _parse_ranking(self, ctx: _Ctx, m: re.Match) -> None:
        factors = list(ctx.rank or [])
        for item in _RANK_SPLIT.split(m.group("body")):
            if not item.strip():
                continue
            rf, raw = _parse_rank_item(item, self.catalog)
            if rf is None or rf.feature not in self.catalog or self.catalog[rf.feature].dtype == "category":
                ctx.unsupported_add(f"rank by '{raw.strip()}'")
                ctx.note(f"Ranking term '{raw.strip()}' not understood; left out of the ranking.")
                continue
            if any(f.feature == rf.feature for f in factors):
                continue
            factors.append(rf)
        ctx.rank = factors
        if ctx.rank_start is None:
            ctx.rank_start = m.start()

    def _leftovers(self, ctx: _Ctx) -> None:
        missed: list[str] = []
        spans: list[tuple[int, int]] = []
        for frag in _FRAGMENT.finditer(ctx.masked):
            cut = frag.start()
            for c in _CONJ.finditer(frag.group(0)):
                spans.append((cut, frag.start() + c.start()))
                cut = frag.start() + c.end()
            spans.append((cut, frag.end()))
        for a, b in spans:
            toks = [t for t in _TOKEN.finditer(ctx.masked[a:b]) if _content(t.group(0))]
            if not toks:
                continue
            q = ctx.quote((a + toks[0].start(), a + toks[-1].end()))
            if q:
                missed.append(q)
                ctx.unsupported_add(q)
        if missed:
            ctx.note("Not understood by the offline translator (listed in unsupported_requests): "
                     + "; ".join(f"'{q}'" for q in missed) + ".")

    # -- assembly ----------------------------------------------------------------------------------

    def _assemble(self, ctx: _Ctx) -> ScreenSpec:
        cat = self.catalog
        explicit = {c.feature for h in ctx.hits if h.covered_by is None for c in h.conditions}
        kept: list[_Hit] = []
        for h in sorted(ctx.hits, key=lambda h: (h.start, h.end)):
            if h.covered_by is not None and h.covered_by & explicit:
                continue
            kept.append(h)
            for n in h.notes:
                ctx.note(n)

        # Drop conditions the catalog cannot execute (custom catalogs), keep the rest.
        valid_hits: list[_Hit] = []
        for h in kept:
            good = []
            for c in h.conditions:
                errs = _condition_errors(c, cat)
                if errs:
                    ctx.unsupported_add(f"{c.describe()} ({'; '.join(errs)})")
                else:
                    good.append(c)
            if good:
                valid_hits.append(_Hit(h.start, h.end, good, h.covered_by, h.notes))

        conditions, any_of = self._group(ctx, valid_hits)
        conditions = _merge_sector_inclusion(_dedupe(conditions))

        ranking = [f for f in (ctx.rank or []) if f.feature in cat and cat[f.feature].dtype != "category"]
        if not ranking:
            ranking = self._default_ranking(ctx, conditions + [c for g in any_of for c in g])

        universe = UniverseSpec()
        if ctx.min_price is not None:
            universe.min_price = ctx.min_price
        if ctx.min_adv is not None:
            universe.min_avg_dollar_volume_usd_mn = ctx.min_adv
        if ctx.exclude_sectors:
            universe.exclude_sectors = list(ctx.exclude_sectors)
        if ctx.min_price is None and ctx.min_adv is None:
            ctx.note("Universe: platform defaults (US common stock, price >= $5, 20-day average dollar volume >= $5mn).")

        spec = ScreenSpec(
            name=_slug(conditions + [c for g in any_of for c in g]),
            observation=ctx.original,
            universe=universe,
            conditions=conditions,
            any_of=any_of,
            ranking=ranking,
            top_n=ctx.top_n if ctx.top_n is not None else 10,
            assumptions=list(ctx.assumptions),
            unsupported_requests=list(ctx.unsupported),
        )
        errors = spec.validate_against(cat)
        if errors:  # defensive: never hand back an unexecutable spec
            raise ScreenTranslationError(errors, spec=spec, errors_by_round=[errors])
        return spec

    def _group(self, ctx: _Ctx, hits: list[_Hit]) -> tuple[list[Condition], list[list[Condition]]]:
        """Consecutive single-condition hits joined only by 'or' become an any_of group."""
        groups: list[list[_Hit]] = []
        for h in hits:
            if groups:
                prev = groups[-1][-1]
                between = ctx.text[prev.end:h.start] if prev.end <= h.start else ""
                words = re.findall(r"[a-z]+", between)
                if ("or" in words and all(w in _FILLER or w in ("or", "either") for w in words)
                        and len(prev.conditions) == 1 and len(h.conditions) == 1):
                    groups[-1].append(h)
                    continue
            groups.append([h])
        conditions: list[Condition] = []
        any_of: list[list[Condition]] = []
        for g in groups:
            if len(g) == 1:
                conditions.extend(g[0].conditions)
            else:
                alts = _dedupe([h.conditions[0] for h in g])
                if len(alts) >= 2:
                    any_of.append(alts)
                    ctx.note("Alternatives joined by 'or' screened as one any_of group: "
                             + " OR ".join(c.describe() for c in alts) + ".")
                else:
                    conditions.extend(alts)
        return conditions, any_of

    def _default_ranking(self, ctx: _Ctx, conds: list[Condition]) -> list[RankFactor]:
        cat = self.catalog
        out: list[RankFactor] = []
        for c in conds:
            fdef = cat[c.feature] if c.feature in cat else None
            if fdef is None or fdef.dtype != "number" or fdef.higher_is_better is None:
                continue
            if any(f.feature == c.feature for f in out):
                continue
            out.append(RankFactor(feature=c.feature, direction=_H if fdef.higher_is_better else _L, weight=1.0,
                                  rationale="default: catalog direction of a screened feature"))
        if out:
            ctx.note("No ranking stated; ranked by the screened features with a natural direction (equal weights): "
                     + ", ".join(f"{f.feature} {f.direction}" for f in out) + ".")
            return out
        for c in conds:
            fdef = cat[c.feature] if c.feature in cat else None
            if fdef is None or fdef.dtype != "number" or c.other_feature or c.op not in (">", ">=", "<", "<="):
                continue
            if any(f.feature == c.feature for f in out):
                continue
            out.append(RankFactor(feature=c.feature, direction=_H if c.op in (">", ">=") else _L, weight=1.0,
                                  rationale=f"default: direction implied by '{c.describe()}'"))
        if out:
            ctx.note("No ranking stated; ranked by the direction each screened condition implies (equal weights): "
                     + ", ".join(f"{f.feature} {f.direction}" for f in out) + ".")
            return out
        fallback = "avg_dollar_volume_20d_usd_mn"
        if fallback not in cat or cat[fallback].dtype == "category":
            fallback = next((f.name for f in cat if f.dtype == "number"), "")
        if not fallback:
            raise ScreenTranslationError(["the catalog has no numeric feature to rank on"])
        ctx.note(f"No ranking stated or implied; ranked by {fallback} (higher is better).")
        return [RankFactor(feature=fallback, direction=_H, weight=1.0, rationale="default fallback ranking")]


def _condition_errors(c: Condition, catalog: FeatureCatalog) -> list[str]:
    probe = ScreenSpec(name="probe", observation="", conditions=[c], ranking=[])
    return [e for e in probe.validate_against(catalog) if not e.startswith("ranking needs")]


def _key(c: Condition) -> tuple:
    return (c.feature, c.op, c.value, c.value_high, tuple(c.values or ()), c.other_feature, c.multiplier)


def _dedupe(conds: list[Condition]) -> list[Condition]:
    seen: set[tuple] = set()
    out = []
    for c in conds:
        k = _key(c)
        if k not in seen:
            seen.add(k)
            out.append(c)
    return out


def _merge_sector_inclusion(conds: list[Condition]) -> list[Condition]:
    first: Condition | None = None
    out: list[Condition] = []
    for c in conds:
        if c.feature == "gics_sector" and c.op == "in":
            if first is None:
                first = c
                out.append(c)
            else:
                first.values = list(dict.fromkeys([*(first.values or []), *(c.values or [])]))
                first.rationale = f"{first.rationale}; {c.rationale}"
            continue
        out.append(c)
    return out


def _cap_tag(c: Condition) -> str:
    if c.value is None:
        return "cap"
    if c.op == "between" and c.value_high is not None:
        mid = math.sqrt(max(c.value, 1e-6) * max(c.value_high, 1e-6))
        return "micro_cap" if mid < 0.3 else "small_cap" if mid < 2 else "mid_cap" if mid < 25 else "large_cap"
    if c.op in (">", ">="):
        return "large_cap" if c.value >= 10 else "mid_cap" if c.value >= 2 else "small_cap" if c.value >= 0.3 else "cap"
    return "micro_cap" if c.value <= 0.3 else "small_cap" if c.value <= 2 else "mid_cap" if c.value <= 10 else "cap"


def _tag(c: Condition) -> str:
    f = c.feature
    if f == "market_cap_usd_bn":
        return _cap_tag(c)
    if f in ("sma_50_vs_sma_200_pct", "price_vs_sma_200_pct", "price_vs_sma_50_pct", "golden_cross_20d", "sma_200_slope_1m_pct"):
        return "downtrend" if c.op in ("<", "<=") and (c.value or 0) <= 0 else "uptrend"
    if f in _MOM_FEATURES:
        return "momentum"
    if f == "drawdown_from_52w_high_pct":
        upper = c.value_high if c.op == "between" else c.value if c.op in ("<", "<=") else None
        return "pullback" if upper is not None and upper < -2 else "near_highs"
    if f in ("eps_revision_3m_pct", "revenue_revision_3m_pct"):
        return "revisions"
    if f in _LEVERAGE_FEATURES:
        return "net_cash" if f == "net_debt_usd_bn" else "balance_sheet"
    if f.endswith(("margin_pct", "margin_change_yoy_pp")):
        return "margins"
    if f == "rsi_14":
        return "oversold" if c.op in ("<", "<=") else "overbought" if c.op in (">", ">=") else "rsi"
    if f in _VOLUME_FEATURES:
        return "volume"
    if f in ("short_interest_pct_float", "days_to_cover", "short_interest_change_1m_pct"):
        return "short_interest"
    if f in ("fcf_yield_pct", "fcf_margin_pct"):
        return "fcf"
    if f in _GROWTH_FEATURES:
        return "growth"
    if f in _VALUE_FEATURES:
        return "value"
    if f == "gics_sector":
        return "_".join(re.sub(r"[^a-z]+", "_", v.lower()).strip("_") for v in (c.values or [])[:2]) or "sector"
    return re.sub(r"_(?:pct|pp|usd_bn|usd_mn|1y|14|ntm|yoy|3m|1m|20d|50d|5d|10d)$", "", f)


def _slug(conds: list[Condition]) -> str:
    tags: list[str] = []
    for c in conds:
        t = _tag(c)
        if t and t not in tags:
            tags.append(t)
    return ("_".join(tags[:4]) or "screen")[:60]


# ------------------------------------------------------------------------------------------------
# Convenience: LLM first, heuristic fallback
# ------------------------------------------------------------------------------------------------


def translate_observation(
    observation: str,
    llm: StructuredLLM | None = None,
    *,
    catalog: FeatureCatalog | None = None,
    max_repair_rounds: int = 2,
    effort: str = "high",
) -> TranslationResult:
    """Translate with the LLM when given, falling back to the offline heuristic.

    The fallback runs when the LLM call fails (``LLMError``, including refusals) or no valid spec
    is produced; the reason is appended to the spec's assumptions. An empty observation raises.
    """
    heuristic = HeuristicScreenTranslator(catalog)
    if llm is None:
        return heuristic.translate(observation)
    try:
        return NLScreenTranslator(llm, catalog, max_repair_rounds=max_repair_rounds, effort=effort).translate(observation)
    except (LLMError, ScreenTranslationError) as e:
        if not isinstance(observation, str) or not observation.strip():
            raise
        out = heuristic.translate(observation)
        out.spec.assumptions.append(
            f"LLM translation failed ({type(e).__name__}: {str(e)[:200]}); used the offline heuristic translator."
        )
        return out
