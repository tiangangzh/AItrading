"""Research idea (natural language) -> ``StrategySpec``.

Two translators share one result type (:class:`StrategyTranslation`):

* :class:`StrategyTranslator` - Claude (any :class:`~aitrading.llm.base.StructuredLLM`) emits a
  ``StrategySpec`` via structured output. The spec is validated against the feature catalog and
  failures are fed back for up to ``max_repair_rounds`` repair calls. The system prompt is
  byte-stable (catalog + template library + rules, no per-request content) so it is served from
  the prompt cache across ideas.
* :class:`HeuristicStrategyTranslator` - deterministic, offline: matches keywords to the curated
  template library (:mod:`aitrading.strategy.library`) and parses parameter overrides (quantiles,
  top-N, long-only, rebalance frequency, dates, costs, weighting, ...) from the text. Every
  interpretation is recorded in ``spec.assumptions``; anything it cannot express goes into
  ``spec.unsupported_requests``.

:func:`translate_idea` tries Claude first and falls back to the heuristic.
"""

from __future__ import annotations

import calendar
import difflib
import json
import re
from dataclasses import dataclass, field
from datetime import date

from pydantic import ValidationError

from aitrading.llm.base import LLMError, LLMOutputError, StructuredLLM
from aitrading.screen.catalog import FeatureCatalog, default_catalog
from aitrading.screen.spec import Condition
from aitrading.strategy.library import TEMPLATES, suggest_templates, templates_prompt
from aitrading.strategy.spec import PortfolioConstruction, StrategySpec

__all__ = [
    "StrategyTranslation",
    "StrategyTranslationError",
    "StrategyTranslator",
    "HeuristicStrategyTranslator",
    "build_system_prompt",
    "spec_errors",
    "translate_idea",
]


# ------------------------------------------------------------------------------------------------
# Result / error types
# ------------------------------------------------------------------------------------------------


@dataclass
class StrategyTranslation:
    spec: StrategySpec
    attempts: int
    errors_by_round: list[list[str]]
    translator: str
    template: str | None


class StrategyTranslationError(ValueError):
    """The idea could not be turned into an executable spec."""

    def __init__(self, message: str, errors: list[str] | None = None, suggestions: list[str] | None = None):
        super().__init__(message)
        self.errors: list[str] = list(errors or [])
        self.suggestions: list[str] = list(suggestions or [])


# ------------------------------------------------------------------------------------------------
# Validation shared by both translators
# ------------------------------------------------------------------------------------------------


def spec_errors(spec: StrategySpec, catalog: FeatureCatalog) -> list[str]:
    """``spec.validate_against(catalog)`` plus operand/type checks on conditions, with suggestions.

    Unknown features get "did you mean" hints so a repair round can fix typos; condition checks
    mirror the screen DSL (category features only with in/not_in/==/!=, numeric comparisons
    between features need matching units).
    """
    errs: list[str] = []
    names = catalog.names()
    for e in spec.validate_against(catalog):
        m = re.fullmatch(r"unknown feature '(.+)'", e)
        if m:
            close = difflib.get_close_matches(m.group(1), names, n=3, cutoff=0.5)
            if close:
                e += f" (did you mean: {', '.join(close)}?)"
        errs.append(e)
    conds = list(spec.filters)
    if spec.time_series:
        conds += [*spec.time_series.entry, *spec.time_series.exit]
    for c in conds:
        if c.feature not in catalog:
            continue
        fdef = catalog[c.feature]
        if fdef.dtype == "category" and c.op not in ("in", "not_in", "==", "!="):
            errs.append(f"{c.describe()}: category feature '{c.feature}' only supports in/not_in")
        if fdef.dtype != "category" and c.op in ("in", "not_in"):
            errs.append(f"{c.describe()}: 'in'/'not_in' only apply to category features")
        if c.other_feature and c.other_feature in catalog:
            odef = catalog[c.other_feature]
            if "category" in (fdef.dtype, odef.dtype):
                errs.append(f"{c.describe()}: cannot compare category features numerically")
            elif odef.unit != fdef.unit:
                errs.append(f"{c.describe()}: unit mismatch ({fdef.unit} vs {odef.unit})")
    return errs


# ------------------------------------------------------------------------------------------------
# Claude translator
# ------------------------------------------------------------------------------------------------

_SYSTEM_TEMPLATE = """\
You are a quantitative researcher on an equity research platform. A portfolio manager types a research idea \
("Fama-French 3-factor model", "12-1 momentum top decile", "quality at a reasonable price", "long SPY above its \
200-day") and you turn it into a precise, testable strategy definition: a StrategySpec that a deterministic engine \
will backtest on point-in-time data. You never see the backtest results, so you cannot and must not tune anything \
to them.

# Strategy kinds and their required fields
- cross_sectional: rank stocks on a composite `signal` (one or more SignalComponent: feature, direction \
higher_is_better|lower_is_better, weight > 0, transform rank|zscore, sector_neutral) at each rebalance and hold \
quantile or top-N portfolios described by `portfolio` (style long_only|long_short, selection quantile|top_n, \
n_quantiles 2-20, top_n required when selection=top_n, weighting equal|value|signal|inverse_vol, optional \
max_weight). Optional `filters` are eligibility conditions applied before ranking.
- screen: hold every name passing `filters` (one or more Conditions, ANDed) at each rebalance; \
portfolio.style long_only, weighting equal|value. Leave `signal` empty.
- factor_model: build an academic factor model from the universe and compare it with the official Kenneth French \
factors. Set `factor_model` to capm|ff3|carhart4|ff5 and attribution_model to the same model. Leave signal and \
filters empty.
- time_series: per-asset timing rule. Set `time_series` = {assets: [tickers], entry: [Conditions on the asset's \
own technical features, ANDed], exit: [] (empty = exit as soon as entry stops holding), when_flat: cash|short}. \
Set benchmark to the traded ticker when there is one asset.

A Condition is {feature, op, value, value_high, values, other_feature, multiplier, rationale}: ops > >= < <= == != \
compare the feature with exactly one of `value` (a constant) or `other_feature` (times `multiplier`); `between` \
uses value..value_high inclusive; `in` / `not_in` take `values` and apply only to category features. Booleans are \
1/0 (use == 1).
Every kind also takes: universe {country, security_types, min_price, min_avg_dollar_volume_usd_mn, \
exclude_sectors}, start / end (ISO dates or null), rebalance daily|weekly|monthly|quarterly|annual, \
attribution_model capm|ff3|carhart4|ff5, costs_bps, benchmark (ticker or null = default broad US index), \
assumptions, unsupported_requests.

# Feature catalog - the ONLY feature names you may use
{catalog}

# Units
- Features ending in _pct are percentages: 5 means 5 %, not 0.05. Features ending in _pp are percentage points.
- drawdown_from_52w_high_pct is <= 0: "at least 20% below the 52-week high" is drawdown_from_52w_high_pct <= -20.
- market_cap_usd_bn and enterprise_value_usd_bn are USD billions ("under $2bn" is < 2); \
avg_dollar_volume_20d_usd_mn is USD millions.
- rsi_14, iv_rank_1y and *_percentile features are on a 0-100 scale; ratio features marked [x] are plain multiples.
- costs_bps is the one-way cost in basis points (0.10% = 10 bps; halve a round-trip figure).

# Template library
Curated, published ideas already expressed as specs. Start from the closest template when one fits. If the idea \
is essentially that template with different parameters (quantiles, dates, costs, rebalance, weighting, long-only, \
asset), keep the template key as `name` and change only those parameters. If the idea is materially different \
(e.g. it combines two templates or uses another signal), use a new snake_case name. Fields omitted from a template \
spec take their schema defaults (rebalance monthly, attribution_model ff3, costs_bps 10, style long_short, \
selection quantile, n_quantiles 5, weighting equal, start/end null, default universe).

{templates}

# Defaults - apply when the idea is silent, and record every default you apply in `assumptions`
- rebalance: monthly.
- Factor tests (cross_sectional): quintiles (n_quantiles 5), long_short (long the best quintile, short the worst), \
equal weighted.
- When the idea says "buy", "long only", "own" or "hold the top ...": style long_only (only the best quantile or \
the top_n names are held) - unless it also names a short or sell leg ("buy last month's losers and sell last month's \
winners", "buy the cheapest quintile and short the most expensive"): that is style long_short.
- costs_bps: 10 one-way.
- start: null (as much history as the data allows, about 10 years); end: null (latest available).
- attribution_model: ff3 (capm for timing rules on a single index ETF; the model itself for factor_model).
- universe: platform default (US common stock, price >= 5 USD, 20-day average dollar volume >= 5 USD mn).

# Honesty rules
- Copy the idea verbatim into `idea`.
- Use only catalog features. Never invent a feature, a field or a data source.
- Anything the catalog or the spec cannot express (stop-losses, leverage, options, intraday data, news or \
sentiment, a lookback window the catalog does not have, an asset class other than US equities / ETFs, ...) goes \
into `unsupported_requests` as a short description. Never approximate silently: if you do substitute something \
close (e.g. the 200-day average for a 10-month average), say so in `assumptions` AND list the original request in \
`unsupported_requests`.
- Never choose parameters (lookbacks, thresholds, quantiles, dates, universe, costs) to make results look good: use \
the idea's own parameters, else the published / standard definition, else the defaults above. Do not shorten the \
sample unless the idea asks for it.
- Record every interpretive choice in `assumptions`, one short sentence each.
"""


def build_system_prompt(catalog: FeatureCatalog | None = None) -> str:
    """The translator's system prompt (byte-stable for a given catalog: cacheable across ideas)."""
    cat = catalog or default_catalog()
    return _SYSTEM_TEMPLATE.replace("{catalog}", cat.to_prompt()).replace("{templates}", templates_prompt())


def _validation_messages(err: ValidationError | LLMOutputError) -> list[str]:
    out = []
    for e in err.errors():
        loc = ".".join(str(p) for p in e.get("loc", ()))
        out.append(f"schema error at '{loc or '<root>'}': {e.get('msg', 'invalid')}")
    return out or [str(err)]


class StrategyTranslator:
    """Claude-backed idea -> ``StrategySpec`` translation with validate-and-repair rounds.

    A spec that fails :func:`spec_errors` is sent back with its errors for up to
    ``max_repair_rounds`` repair calls (purpose ``"strategy_spec:repair"``). A reply that does not
    validate as a ``StrategySpec`` at all (``LLMOutputError`` / pydantic ``ValidationError``, e.g. a
    bound the API's schema cannot enforce) uses a repair round too, as in the screen translator;
    other ``LLMError`` / ``LLMRefusalError`` propagate.
    """

    def __init__(
        self,
        llm: StructuredLLM,
        catalog: FeatureCatalog | None = None,
        *,
        max_repair_rounds: int = 2,
        effort: str = "high",
        today: date | None = None,
    ):
        if max_repair_rounds < 0:
            raise ValueError("max_repair_rounds must be >= 0")
        self.llm = llm
        self.catalog = catalog or default_catalog()
        self.max_repair_rounds = max_repair_rounds
        self.effort = effort
        self.today = today
        self._system: str | None = None

    @property
    def system_prompt(self) -> str:
        if self._system is None:
            self._system = build_system_prompt(self.catalog)
        return self._system

    def _today(self) -> date:
        return self.today or date.today()

    def user_prompt(self, idea: str) -> str:
        return (
            f"Today's date: {self._today().isoformat()} (resolve relative periods such as \"the last 5 years\" against it).\n\n"
            f"<idea>\n{idea}\n</idea>\n\n"
            "Translate the idea above into a StrategySpec."
        )

    def repair_prompt(self, idea: str, previous: StrategySpec | None, errors: list[str]) -> str:
        bullet = "\n".join(f"- {e}" for e in errors)
        if previous is not None:
            prev = json.dumps(previous.model_dump(mode="json"), separators=(",", ":"))
            head = f"Your previous StrategySpec cannot be executed:\n<previous_spec>\n{prev}\n</previous_spec>\n"
        else:
            head = "Your previous response was not a valid StrategySpec.\n"
        return (
            f"Today's date: {self._today().isoformat()}.\n\n"
            f"<idea>\n{idea}\n</idea>\n\n"
            f"{head}"
            f"<errors>\n{bullet}\n</errors>\n\n"
            "Return the complete corrected StrategySpec. Fix exactly what the errors require and keep the idea's intent. "
            "Anything that cannot be expressed with catalog features goes into unsupported_requests - do not invent a feature."
        )

    def translate(self, idea: str) -> StrategyTranslation:
        if not idea or not idea.strip():
            raise StrategyTranslationError("the idea is empty", errors=["the idea is empty"], suggestions=suggest_templates("", 5))
        errors_by_round: list[list[str]] = []
        spec: StrategySpec | None = None
        errors: list[str] = []
        for rnd in range(self.max_repair_rounds + 1):
            if rnd == 0:
                purpose, user = "strategy_spec", self.user_prompt(idea)
            else:
                purpose, user = "strategy_spec:repair", self.repair_prompt(idea, spec, errors)
            try:
                out = self.llm.structured(
                    purpose=purpose, system=self.system_prompt, user=user, output_model=StrategySpec, effort=self.effort
                )
                if not isinstance(out, StrategySpec):
                    out = StrategySpec.model_validate(out)
            except (ValidationError, LLMOutputError) as e:  # schema-level failure: repair it like a spec error
                errors = _validation_messages(e)
                errors_by_round.append(errors)
                continue
            spec = out.model_copy(update={"idea": idea})
            errors = spec_errors(spec, self.catalog)
            errors_by_round.append(errors)
            if not errors:
                return StrategyTranslation(
                    spec=spec,
                    attempts=rnd + 1,
                    errors_by_round=errors_by_round,
                    translator=self.llm.name,
                    template=spec.name if spec.name in TEMPLATES else None,
                )
        raise StrategyTranslationError(
            f"could not produce a valid strategy spec after {len(errors_by_round)} attempt(s): " + "; ".join(errors),
            errors=errors,
            suggestions=suggest_templates(idea, 5),
        )


# ------------------------------------------------------------------------------------------------
# Heuristic (offline) translator
# ------------------------------------------------------------------------------------------------

_NUMBER_WORDS = [
    (r"\btwo[\s-]+hundred\b", "200"),
    (r"\btwelve\b", "12"),
    (r"\btwenty\b", "20"),
    (r"\bfifty\b", "50"),
    (r"\bten\b", "10"),
    (r"\bsix\b", "6"),
    (r"\bfive\b", "5"),
    (r"\bfour\b", "4"),
    (r"\bthree\b", "3"),
    (r"\btwo\b", "2"),
    (r"\bone\b", "1"),
]


def _normalise(text: str) -> str:
    t = text.lower()
    for a, b in (("–", "-"), ("—", "-"), ("’", "'"), ("‘", "'"), ("_", " ")):
        t = t.replace(a, b)
    for pat, rep in _NUMBER_WORDS:
        t = re.sub(pat, rep, t)
    return re.sub(r"\s+", " ", t).strip()


# Weighting phrases are parsed first and removed before template matching ("value weighted" is
# not the value factor).
_WEIGHTING = [
    ("inverse_vol", r"inverse[\s-]*vol(?:atility)?(?:[\s-]*weight\w*)?|risk[\s-]*parity"),
    ("equal", r"equal(?:ly)?[\s-]*weight\w*"),
    ("value", r"(?:value|cap|market[\s-]*cap|capitali[sz]ation)[\s-]*weight\w*"),
    ("signal", r"(?:signal|score)[\s-]*weight\w*"),
]

# Attribution: "alpha against the 5-factor model" sets attribution_model instead of picking the
# factor-model template (the phrase is removed before template matching).
_MODEL_PHRASES = [
    ("ff5", r"(?:fama[\s-]*french[\s-]*)?5[\s-]*factors?(?:[\s-]*model)?|\bff5\b"),
    ("carhart4", r"carhart(?:[\s-]*4[\s-]*factors?)?(?:[\s-]*model)?|4[\s-]*factors?(?:[\s-]*model)?|\bff4\b"),
    ("ff3", r"fama[\s-]*french(?:[\s-]*3[\s-]*factors?)?(?:[\s-]*model)?|3[\s-]*factors?(?:[\s-]*model)?|\bff3\b"),
    ("capm", r"\bcapm\b|capital asset pricing model|market model"),
]
_ATTRIBUTION_CUE = (
    r"(?:\balpha\b|attribut\w*|regress\w*|control\w*\s+for|adjust\w*\s+for|explained\s+by|against|relative\s+to|versus|\bvs\.?)"
)

# Ordered rules: the first rule with a matching pattern is the primary template. A key may appear
# more than once; a rule given as (key, patterns, requires) only applies when ``requires`` also matches.
_RULES: list[tuple] = [
    ("dislocation_screen", [r"dislocat", r"capitulat"]),
    ("golden_cross_spy", [r"golden[\s-]*cross", r"death[\s-]*cross", r"\b50[\s-]*(?:day|d|dma|sma)\b.{0,40}\b200\b",
                          r"moving[\s-]*average[\s-]*cross", r"dual[\s-]*moving[\s-]*average"]),
    ("trend_200dma_spy", [r"\b200[\s-]*(?:day|d|dma|sma|ma)\b", r"\b10[\s-]*month[\s-]*(?:sma|ma|moving)", r"faber",
                          r"market[\s-]*timing", r"trend[\s-]*filter", r"\b\d{2,3}[\s-]*(?:day|d)[\s-]*(?:moving[\s-]*average|sma|dma|ma)\b",
                          r"\b\d{2,3}[\s-]*(?:dma|sma)\b", r"(?:above|below)\s+(?:its|their|the)\s+\d{2,3}[\s-]*(?:day|d|dma|sma)\b"]),
    ("trend_200dma_spy", [r"trend[\s-]*following", r"\btrend\b"], "ASSET_CONTEXT"),
    ("ff5", [r"\b5[\s-]*factor", r"\bff5\b", r"\brmw\b", r"\bcma\b"]),
    ("carhart4", [r"carhart", r"\b4[\s-]*factor", r"\bff4\b", r"fama[\s-]*french.{0,30}(?:\+|plus|and|with)\s*momentum",
                  r"\bff3\b.{0,10}(?:\+|plus|and|with)\s*momentum"]),
    ("ff3", [r"\b3[\s-]*factor", r"\bff3\b", r"fama[\s-]*french", r"\bsmb\b(?:.{0,12}\bhml\b)?"]),
    ("capm", [r"\bcapm\b", r"capital asset pricing", r"\b1[\s-]*factor[\s-]*model", r"single[\s-]*factor", r"market[\s-]*model",
              r"market[\s-]*factor"]),
    ("qarp", [r"\bqarp\b", r"quality.{0,40}(?:\bvalue\b|reasonable[\s-]*price|cheap|valuation)",
              r"(?:\bvalue\b|cheap).{0,40}quality", r"reasonable[\s-]*price"]),
    ("estimate_revisions", [r"revision", r"estimate[\s-]*momentum", r"earnings[\s-]*momentum", r"analyst.{0,20}(?:upgrade|raise)"]),
    ("short_interest", [r"short[\s-]*interest", r"heavily[\s-]*shorted", r"most[\s-]*shorted", r"short[\s-]*squeeze",
                        r"days[\s-]*to[\s-]*cover", r"short[\s-]*sellers"]),
    ("low_beta", [r"low[\s-]*beta", r"betting[\s-]*against[\s-]*beta", r"\bbab\b"]),
    ("low_volatility", [r"low[\s-]*vol", r"min(?:imum)?[\s-]*vol", r"min(?:imum)?[\s-]*variance", r"low[\s-]*risk",
                        r"volatility[\s-]*anomaly", r"least[\s-]*volatile"]),
    ("rsi_reversal", [r"\brsi\b", r"relative[\s-]*strength[\s-]*index", r"oversold"]),
    ("short_term_reversal", [r"revers", r"mean[\s-]*revert", r"mean[\s-]*reversion", r"last[\s-]*month'?s[\s-]*losers",
                             r"\bcontrarian\b"]),
    ("momentum_12_1", [r"momentum", r"\b12[\s-]*1\b", r"jegadeesh", r"\bwinners\b", r"relative[\s-]*strength",
                       r"trend[\s-]*following"]),
    ("quality", [r"quality", r"\bqmj\b", r"high[\s-]*roe", r"gross[\s-]*margin"]),
    ("gross_profitability", [r"gross[\s-]*profitab", r"novy[\s-]*marx",
                             r"gross[\s-]*profits?[\s-]*(?:to|/|over|scaled[\s-]*by)[\s-]*(?:total[\s-]*)?assets"]),
    ("profitability", [r"operating[\s-]*profitab", r"robust[\s-]*minus[\s-]*weak", r"profitab"]),
    ("investment", [r"asset[\s-]*growth", r"conservative[\s-]*minus[\s-]*aggressive",
                    r"\binvestment[\s-]*(?:factor|anomaly|effect|premium)\b", r"\blow[\s-]*investment\b",
                    r"balance[\s-]*sheet[\s-]*(?:growth|expansion)", r"aggressive(?:ly)?[\s-]*invest"]),
    ("value_composite", [r"value[\s-]*composite", r"composite[\s-]*value", r"multi[\s-]*(?:factor|metric)[\s-]*value",
                         r"ev[\s/-]*(?:to[\s-]*)?ebitda", r"earnings[\s-]*yield", r"blend\w*.{0,20}valu"]),
    ("value_book_to_market", [r"book[\s-]*to[\s-]*(?:market|price)", r"\bb\s*/\s*m\b", r"price[\s-]*to[\s-]*book",
                              r"\bp\s*/\s*b\b", r"\bhml\b", r"high[\s-]*minus[\s-]*low", r"book[\s-]*(?:value|equity)"]),
    ("value_fcf", [r"\bvalue\b", r"fcf[\s-]*yield", r"free[\s-]*cash[\s-]*flow[\s-]*yield", r"\bcheap", r"undervalued",
                   r"\bp\s*/?\s*e\b", r"price[\s-]*to[\s-]*earnings"]),
    ("size", [r"\bsize\b", r"small[\s-]*caps?\b.{0,30}(?:outperform|premium|beat|\bvs\b|versus|minus|effect)",
              r"small[\s-]*minus[\s-]*big", r"small[\s-]*(?:cap|firm)[\s-]*(?:premium|effect|factor|anomaly)"]),
]

# Subsumption: template -> {other template: "strong" phrases of the other idea, or None}.
# A match of the other template is read as part of this one (not as a second idea) when
#   * the value is None - this template's definition already contains that concept (Carhart's
#     momentum factor, the value leg of QARP, the 200-day inside a golden cross, ...), or
#   * none of the strong phrases occurs outside the text claimed by the matched templates' own
#     patterns ("earnings momentum" is the revisions idea; "winners" describes a reversal).
# Otherwise the other idea was genuinely requested ("12-1 momentum combined with 1-month
# reversal") and goes to ``_combine`` (merged into a composite, or reported as unsupported).
# An absorbed match whose phrase lies outside the claimed text is reported in the assumptions.
_ST_REVERSAL_STRONG = (
    r"short[\s-]*term[\s-]*revers|\b1[\s-]*(?:month|mo|m)\b[\s-]*(?:return[\s-]*)?revers|last[\s-]*month'?s[\s-]*losers|"
    r"monthly[\s-]*revers"
)
_MOMENTUM_STRONG = r"momentum|\b12[\s-]*1\b|jegadeesh|relative[\s-]*strength|trend[\s-]*following"
_XS_MOMENTUM_STRONG = (
    r"\b12[\s-]*1\b|jegadeesh|cross[\s-]*sectional[\s-]*momentum|momentum\s+(?:stocks|names|portfolios?|factor|deciles?|quintiles?)"
)
_LOW_VOL_STRONG = r"low[\s-]*vol|min(?:imum)?[\s-]*vol|min(?:imum)?[\s-]*variance|volatility[\s-]*anomaly|least[\s-]*volatile"
_OP_PROFIT_STRONG = r"operating[\s-]*profitab|robust[\s-]*minus[\s-]*weak|profitability[\s-]*(?:factor|premium|anomaly)"
_GROSS_PROFIT_STRONG = r"gross[\s-]*profitab|novy[\s-]*marx|gross[\s-]*profits?\b"
_FCF_STRONG = r"fcf|free[\s-]*cash[\s-]*flow"

_SUBSUMES: dict[str, dict[str, str | None]] = {
    "dislocation_screen": dict.fromkeys(["momentum_12_1", "rsi_reversal", "short_interest", "value_fcf", "golden_cross_spy",
                                         "trend_200dma_spy", "short_term_reversal", "quality", "size"]),
    "golden_cross_spy": {"trend_200dma_spy": None, "momentum_12_1": _XS_MOMENTUM_STRONG},
    "trend_200dma_spy": {"momentum_12_1": _XS_MOMENTUM_STRONG, "golden_cross_spy": None},
    "ff5": {"ff3": None, "capm": None, "quality": None, "size": None, "value_fcf": None, "value_book_to_market": None,
            "profitability": None, "investment": None, "carhart4": r"carhart|\b4[\s-]*factors?|\bff4\b"},
    "carhart4": dict.fromkeys(["ff3", "capm", "momentum_12_1", "size", "value_fcf", "value_book_to_market"]),
    "ff3": dict.fromkeys(["capm", "size", "value_fcf", "value_book_to_market"]),
    "capm": {},
    "qarp": dict.fromkeys(["quality", "value_fcf", "value_composite", "value_book_to_market"]),
    "value_composite": {"value_fcf": None, "value_book_to_market": None},
    "value_book_to_market": {"value_fcf": _FCF_STRONG},
    "quality": {"profitability": _OP_PROFIT_STRONG, "gross_profitability": _GROSS_PROFIT_STRONG},
    "gross_profitability": {"profitability": _OP_PROFIT_STRONG},
    "low_beta": {"low_volatility": _LOW_VOL_STRONG, "capm": None},
    "estimate_revisions": {"momentum_12_1": _MOMENTUM_STRONG},
    "short_interest": {"short_term_reversal": _ST_REVERSAL_STRONG},
    "rsi_reversal": {"short_term_reversal": _ST_REVERSAL_STRONG, "momentum_12_1": _MOMENTUM_STRONG},
    "short_term_reversal": {"momentum_12_1": _MOMENTUM_STRONG},
}

_FACTOR_MODEL_KEYS = {"capm", "ff3", "carhart4", "ff5"}

Span = tuple[int, int]


def _overlaps(span: Span, region: list[Span]) -> bool:
    return any(span[0] < e and s < span[1] for s, e in region)


def _word_span(t: str, start: int, end: int) -> Span:
    """Expand a regex match to whole words ('revers' -> 'reversal') for display and overlap tests."""
    while start > 0 and t[start - 1].isalnum():
        start -= 1
    while end < len(t) and t[end].isalnum():
        end += 1
    return start, end

_STOCK_CONTEXT = r"\b(?:stocks|names|equities|companies|shares|universe|cross[\s-]*section\w*)\b"
_ASSET_CONTEXT = (
    r"\bspy\b|s&p|\bspx\b|\bmarket\b|\bindex\b|\betfs?\b|\bqqq\b|nasdaq|russell|\biwm\b|\bdia\b|\bdow\b|timing|\bcash\b|"
    r"faber|tactical|\btlt\b|\bgld\b|\bvti\b|\befa\b|\beem\b"
)

_QUANTILE_WORDS = {"decile": 10, "quintile": 5, "quartile": 4, "tercile": 3, "tertile": 3, "vigintile": 20}

_GICS_SECTORS = {
    r"financials?|banks": "Financials",
    r"utilit(?:y|ies)": "Utilities",
    r"real[\s-]*estate|reits?": "Real Estate",
    r"energy": "Energy",
    r"materials": "Materials",
    r"industrials": "Industrials",
    r"health[\s-]*care|healthcare": "Health Care",
    r"(?:information[\s-]*)?tech(?:nology)?": "Information Technology",
    r"communication[\s-]*services|telecoms?": "Communication Services",
    r"consumer[\s-]*staples|staples": "Consumer Staples",
    r"consumer[\s-]*discretionary|discretionary": "Consumer Discretionary",
}

_UNSUPPORTED = [
    (r"stop[\s-]*loss|trailing[\s-]*stop|take[\s-]*profit|profit[\s-]*target",
     "stop-loss / take-profit exits (positions only change at rebalance dates)"),
    (r"(?:\d+(?:\.\d+)?\s*x\s+|\buse\s+|\bwith\s+|\bapply\w*\s+|\badd\w*\s+)leverage|"
     r"\b(?:levered|leveraged)\s+(?:portfolio|strategy|position|book|etf|long|version)|\bon\s+margin\b",
     "leverage (gross exposure is fixed by the portfolio construction)"),
    (r"\boptions?\b|\b(?:call|put)\s+options?|covered[\s-]*calls?|straddle|\bputs\b|\bcalls\b",
     "options positions (only stocks / ETFs are traded)"),
    (r"\bfutures\b|\bcrypto\w*|\bbitcoin\b|\bforex\b|\bfx\b|\bcurrenc(?:y|ies)\b", "asset classes other than US equities / ETFs"),
    (r"intraday|\bminute\b|\bhourly\b|opening[\s-]*range|\btick[\s-]*data", "intraday data (daily closes only)"),
    (r"sentiment|\bnews\b|twitter|reddit|social[\s-]*media|\btone\b", "news / sentiment signals (not in the feature catalog)"),
    (r"\binsiders?\b", "insider-trading data (not in the feature catalog)"),
    (r"dividend", "dividend-based signals (not in the feature catalog)"),
]

# Transaction-cost amounts: (pattern with the number in group 1, multiplier to bps).
_COST_WORD = r"(?:(?:transaction|trading|execution)\s+)?(?:costs?|commissions?|fees?|slippage)\b"
_COST_NUM = r"(\d+(?:\.\d+)?)"
_COST_PCT = r"\s*(?:%|percent\b|per\s*cent\b)"
_COST_AMOUNTS: list[tuple[str, float]] = [
    (_COST_NUM + r"\s*(?:bps|bp|basis\s+points?)\b", 1.0),                                   # '25 bps'
    (_COST_NUM + _COST_PCT + r"\s*(?:(?:1|one)[\s-]*way\s+|round[\s-]*(?:trip|turn)\s+)?" + _COST_WORD, 100.0),  # '0.15% costs'
    (_COST_WORD + r"\s*(?:of|at|=|:|is|are|around|about|approximately|approx\.?)?\s*(?:of\s+)?" + _COST_NUM + _COST_PCT, 100.0),  # 'costs of 0.2%'
    (_COST_NUM + _COST_PCT + r"\s*(?:per|a|each|every)\s+(?:trade|side|transaction|way|leg|round[\s-]*(?:trip|turn))\b", 100.0),  # '0.1% per trade'
]
# A percentage right after one of these is a position size / bucket, not a cost ('max 5% per side').
_NOT_A_COST_BEFORE = (
    r"(?:max(?:imum)?|cap(?:ped)?(?:\s+at)?|weight\w*|no\s+more\s+than|at\s+most|limit\w*(?:\s+to)?|positions?(?:\s+size)?|"
    r"top|bottom|best|worst)\s*(?:of\s+|at\s+)?$"
)
_ANNUAL_AFTER = r"\s*(?:a|per|each|every)\s+(?:year|annum)\b|\s*p\.?\s*a\b\.?|\s*annual(?:ly)?\b|\s*yearly\b"
_FEE_KIND = r"\b(?:management|performance|advisory|incentive|platform|custody)\s+fees?\b|\bexpense\s+ratios?\b"
# Cost-like amounts that are not bps / % of the amount traded ('commissions of 2 cents per share', '$5 commission').
_COST_ODD_AFTER = (
    _COST_WORD + r"\s*(?:of|at|=|:|is|are|around|about)?\s*(?:of\s+)?\$?\s*(?!(?:19|20)\d{2}\b)\d+(?:\.\d+)?"
    r"(?:\s*(?:cents?|c|dollars?|usd)\b)?(?:\s+(?:per|a|each)\s+\w+)?"
)
_COST_ODD_BEFORE = (
    r"(?<![\d.])\$?\s*(?!(?:19|20)\d{2}\b)\d+(?:\.\d+)?\s*(?:cents?|c|dollars?|usd)?\s*(?:per\s+share\s+)?" + _COST_WORD
)

_KNOWN_ETFS = ["spy", "qqq", "iwm", "dia", "tlt", "gld", "efa", "eem", "vti", "agg", "ief", "shy", "vnq", "xlk", "xlf", "xle"]
_INDEX_ALIASES = [
    (r"s&p[\s-]*500|s&p\b|\bspx\b|\bsp500\b", "SPY"),
    (r"nasdaq(?:[\s-]*100)?|\bndx\b", "QQQ"),
    (r"russell[\s-]*2000", "IWM"),
    (r"\bdow(?:[\s-]*jones)?\b", "DIA"),
]
_TICKER_STOPWORDS = {
    "SMA", "DMA", "EMA", "MA", "RSI", "MACD", "CAPM", "FF", "FF3", "FF4", "FF5", "ETF", "ETFS", "US", "USA", "USD", "AND",
    "OR", "THE", "QARP", "GARP", "BPS", "BP", "PE", "EV", "EBITDA", "FCF", "ROE", "EPS", "NTM", "TTM", "IV", "ATR", "NYSE",
    "AMEX", "AI", "CEO", "CFO", "GICS", "BAB", "QMJ", "SMB", "HML", "UMD", "RMW", "CMA", "WML", "MOM", "YOY", "QOQ",
    "IPO", "NAV", "OK", "IT", "A", "I", "WHEN", "ITS", "BUY", "SELL", "LONG", "SHORT", "ABOVE", "BELOW", "DAY", "CASH",
    "IF", "ELSE", "TOP", "ONLY", "SP", "NDX", "SPX", "DOW", "VS", "TO", "OF", "ON", "IN", "AT", "MKT", "RF",
}


@dataclass
class _Parse:
    """Mutable working state of one heuristic translation."""

    notes: list[str] = field(default_factory=list)
    unsupported: list[str] = field(default_factory=list)
    overridden: set[str] = field(default_factory=set)


def _year_start(s: str) -> date:
    parts = [int(p) for p in s.split("-")]
    return date(parts[0], parts[1] if len(parts) > 1 else 1, parts[2] if len(parts) > 2 else 1)


def _year_end(s: str) -> date:
    parts = [int(p) for p in s.split("-")]
    y = parts[0]
    if len(parts) == 1:
        return date(y, 12, 31)
    m = parts[1]
    return date(y, m, parts[2] if len(parts) > 2 else calendar.monthrange(y, m)[1])


def _years_ago(today: date, n: int) -> date:
    y = today.year - n
    return date(y, today.month, min(today.day, calendar.monthrange(y, today.month)[1]))


class HeuristicStrategyTranslator:
    """Offline idea -> ``StrategySpec``: template keyword matching + parameter overrides.

    Matching is ordered (``_RULES``); the first matching template wins and any other recognised
    idea is either merged (two cross-sectional ideas -> one composite signal with equal total
    weight per idea) or reported in ``unsupported_requests``. A match that merely describes the
    primary idea (``_SUBSUMES``: "earnings momentum" for revisions, "winners" in a reversal idea)
    is not a second idea; when its words lie outside the primary's own phrase, an assumption says
    how it was read. A moving-average condition that is
    phrased about *stocks* ("momentum stocks above their 200-day") becomes an eligibility filter
    instead of the SPY timing rule.
    """

    name = "heuristic"

    def __init__(self, catalog: FeatureCatalog | None = None, *, today: date | None = None):
        self.catalog = catalog or default_catalog()
        self.today = today

    # -- matching ---------------------------------------------------------------------------

    @staticmethod
    def _strip_weighting(t: str) -> str:
        for _, pat in _WEIGHTING:
            t = re.sub(pat, " ", t)
        return t

    @staticmethod
    def _rule_patterns(key: str, t: str) -> list[str]:
        """Every pattern of every rule for ``key`` that applies to text ``t``."""
        pats: list[str] = []
        for rule in _RULES:
            if rule[0] != key:
                continue
            if len(rule) > 2 and rule[2] == "ASSET_CONTEXT":
                if not re.search(_ASSET_CONTEXT, t) or re.search(_STOCK_CONTEXT, t):
                    continue
            pats.extend(rule[1])
        return pats

    @staticmethod
    def _spans(pats: list[str], t: str) -> list[Span]:
        return [_word_span(t, m.start(), m.end()) for pat in pats for m in re.finditer(pat, t) if m.end() > m.start()]

    def _claimed(self, key: str, t: str) -> list[Span]:
        """The text claimed by template ``key``: every match of any of its patterns."""
        return self._spans(self._rule_patterns(key, t), t)

    @staticmethod
    def _matches(t: str) -> list[tuple[str, str, Span]]:
        """Ordered (template key, matched phrase, span in ``t``), one per template."""
        out: list[tuple[str, str, Span]] = []
        seen: set[str] = set()
        for rule in _RULES:
            key, pats = rule[0], rule[1]
            if key in seen:
                continue
            if len(rule) > 2 and rule[2] == "ASSET_CONTEXT":
                if not re.search(_ASSET_CONTEXT, t) or re.search(_STOCK_CONTEXT, t):
                    continue
            for pat in pats:
                m = re.search(pat, t)
                if m:
                    s, e = _word_span(t, m.start(), m.end())
                    out.append((key, t[s:e].strip(), (s, e)))
                    seen.add(key)
                    break
        return out

    def _match_all(self, idea: str) -> tuple[list[tuple[str, str, Span]], tuple[str, str] | None, bool, str]:
        """(ordered matches, attribution (model, phrase) or None, ma-filter-on-stocks flag, matched text)."""
        t = self._strip_weighting(_normalise(idea))
        attribution: tuple[str, str] | None = None
        reduced = t
        for model, pat in _MODEL_PHRASES:
            m = re.search(_ATTRIBUTION_CUE + r"[^.;]{0,30}?(" + pat + r")", t)
            if m:
                attribution = (model, m.group(1).strip())
                reduced = t[: m.start(1)] + " " + t[m.end(1):]
                break
        matches = self._matches(reduced)
        if attribution and not matches:
            matches, attribution, reduced = self._matches(t), None, t
        stock_ma = False
        if matches and matches[0][0] in ("trend_200dma_spy", "golden_cross_spy"):
            if re.search(_STOCK_CONTEXT, reduced) and not re.search(_ASSET_CONTEXT, reduced) and not self._explicit_tickers(idea):
                stock_ma = True
        return matches, attribution, stock_ma, reduced

    def match(self, idea: str) -> tuple[str, str] | None:
        """(template key, matched phrase) of the primary template, or None."""
        matches, _, stock_ma, _ = self._match_all(idea)
        if stock_ma:
            matches = [m for m in matches if m[0] not in ("trend_200dma_spy", "golden_cross_spy")]
        return (matches[0][0], matches[0][1]) if matches else None

    def _secondary_ideas(self, matches: list[tuple[str, str, Span]], t: str, p: _Parse) -> list[tuple[str, str]]:
        """The matches after the primary that are genuinely separate ideas (see ``_SUBSUMES``).

        A match absorbed by an already accepted template is dropped; if its phrase lies outside
        the text claimed by the accepted templates, a note says how it was read, so nothing the
        user wrote disappears silently.
        """
        accepted = [matches[0][0]]
        region = self._claimed(matches[0][0], t)
        out: list[tuple[str, str]] = []
        for key, phrase, span in matches[1:]:
            absorbed_by: str | None = None
            for a in accepted:
                rel = _SUBSUMES.get(a, {})
                if key in rel:
                    strong = rel[key]
                    if strong is None or all(_overlaps(s, region) for s in self._spans([strong], t)):
                        absorbed_by = a
                        break
                elif a in _SUBSUMES.get(key, {}):
                    absorbed_by = a
                    break
            if absorbed_by is None:
                accepted.append(key)
                region = region + self._claimed(key, t)
                out.append((key, phrase))
            elif not _overlaps(span, region) and not {absorbed_by, key} <= _FACTOR_MODEL_KEYS:
                # (one factor model inside another - 'Fama-French' in a 5-factor request - is no second idea)
                p.notes.append(
                    f"'{phrase}' is read as part of the '{absorbed_by}' idea ({TEMPLATES[absorbed_by].title}), "
                    f"not tested as a separate '{key}' signal ({TEMPLATES[key].title})."
                )
        return out

    # -- translation ------------------------------------------------------------------------

    def translate(self, idea: str) -> StrategyTranslation:
        if not idea or not idea.strip():
            raise StrategyTranslationError("the idea is empty", errors=["the idea is empty"], suggestions=suggest_templates("", 5))
        today = self.today or date.today()
        t = _normalise(idea)
        matches, attribution, stock_ma, matched_text = self._match_all(idea)
        p = _Parse()

        ma_filter: Condition | None = None
        if stock_ma:
            ma_key, ma_phrase, _ = matches[0]
            ma_filter = self._ma_filter(ma_key, t)
            matches = [m for m in matches if m[0] not in ("trend_200dma_spy", "golden_cross_spy")]
            p.notes.append(f"'{ma_phrase}' refers to individual stocks, so it is applied as the eligibility filter {ma_filter.describe()}.")
            odd = [w for w in re.findall(r"\b(\d{2,3})[\s-]*(?:day|d|dma|sma|ma)\b", t) if int(w) not in (50, 200)]
            if odd and ma_key == "trend_200dma_spy":
                p.unsupported.append(f"{odd[0]}-day moving average (catalog has 50 and 200-day price-vs-average features; used the {ma_filter.feature.split('_')[3]}-day)")

        if not matches and ma_filter is None:
            raise StrategyTranslationError(
                f"no idea template matches {idea!r}; try rephrasing with a known idea or use the Claude translator",
                errors=["no matching template"],
                suggestions=suggest_templates(idea, 5),
            )

        if matches:
            key, phrase, _ = matches[0]
            tmpl = TEMPLATES[key]
            spec = tmpl.spec()
            p.notes.append(f"Heuristic translation: matched the '{key}' template ({tmpl.title}) on '{phrase}'.")
            spec = self._combine(spec, self._secondary_ideas(matches, matched_text, p), p)
        else:
            assert ma_filter is not None
            spec = StrategySpec(name="moving_average_screen", idea=idea, kind="screen", filters=[], rebalance="monthly")
            spec.portfolio.style = "long_only"
            p.notes.append("No signal idea recognised: holding every stock that passes the moving-average filter, equal weighted.")
        if ma_filter is not None:
            spec.filters.append(ma_filter)

        spec.idea = idea
        if attribution and spec.kind != "factor_model":
            spec.attribution_model = attribution[0]  # type: ignore[assignment]
            p.overridden.add("attribution_model")
            p.notes.append(f"'{attribution[1]}' (attribution request) -> attribution_model={attribution[0]}.")

        self._signal_variants(spec, t, p)
        self._portfolio(spec, t, p)
        self._rebalance(spec, t, p)
        self._dates(spec, t, today, p)
        self._costs(spec, t, p)
        self._universe(spec, t, p)
        if spec.kind == "time_series":
            self._timing(spec, idea, t, p)
        self._unsupported(spec, t, p)
        self._defaults_note(spec, p)

        spec.assumptions = [*spec.assumptions, *p.notes]
        spec.unsupported_requests = [*spec.unsupported_requests, *p.unsupported]
        errors = spec_errors(spec, self.catalog)
        if errors:
            raise StrategyTranslationError(
                "the heuristic translation is not executable: " + "; ".join(errors),
                errors=errors,
                suggestions=suggest_templates(idea, 5),
            )
        return StrategyTranslation(
            spec=spec,
            attempts=1,
            errors_by_round=[[]],
            translator=self.name,
            template=spec.name if spec.name in TEMPLATES else None,
        )

    # -- pieces -----------------------------------------------------------------------------

    @staticmethod
    def _explicit_tickers(idea: str) -> list[str]:
        out: list[str] = []
        for m in re.finditer(r"\$([A-Za-z]{1,5})\b|\b([A-Z]{2,5})\b", idea):
            tok = (m.group(1) or m.group(2)).upper()
            if tok not in _TICKER_STOPWORDS and tok not in out:
                out.append(tok)
        low = idea.lower()
        for etf in _KNOWN_ETFS:
            if re.search(rf"\b{etf}\b", low) and etf.upper() not in out:
                out.append(etf.upper())
        return out

    @staticmethod
    def _ma_filter(key: str, t: str) -> Condition:
        below = re.search(r"below\s+(?:its|their|the)?\s*(?:50|200|20)", t) is not None
        if key == "golden_cross_spy":
            op = "<" if re.search(r"death[\s-]*cross", t) else ">"
            return Condition(feature="sma_50_vs_sma_200_pct", op=op, value=0.0, rationale="50-day vs 200-day moving average")
        window = 50 if re.search(r"\b50[\s-]*(?:day|d|dma|sma)\b", t) and not re.search(r"\b200[\s-]*(?:day|d|dma|sma|ma)\b", t) else 200
        return Condition(
            feature=f"price_vs_sma_{window}_pct", op="<" if below else ">", value=0.0,
            rationale=f"price {'below' if below else 'above'} its {window}-day moving average",
        )

    def _combine(self, spec: StrategySpec, others: list[tuple[str, str]], p: _Parse) -> StrategySpec:
        if not others:
            return spec
        merged: list[str] = []
        for key, phrase in others:
            other = TEMPLATES[key].spec()
            if spec.kind == "cross_sectional" and other.kind == "cross_sectional" and len(merged) < 2:
                merged.append(key)
                # equal total weight per idea: rescale each idea's components to sum to 1
                if len(merged) == 1:
                    total = sum(s.weight for s in spec.signal)
                    spec.signal = [s.model_copy(update={"weight": s.weight / total}) for s in spec.signal]
                total = sum(s.weight for s in other.signal)
                have = {s.feature for s in spec.signal}
                for s in other.signal:
                    if s.feature in have:
                        continue
                    spec.signal.append(s.model_copy(update={"weight": s.weight / total}))
                spec.assumptions.extend(a for a in other.assumptions if a not in spec.assumptions)
                p.notes.append(f"Also matched '{key}' on '{phrase}': its signal is added to the composite with equal total weight per idea.")
            else:
                p.unsupported.append(
                    f"combining with '{TEMPLATES[key].title}' ('{phrase}') - the offline translator tests one idea of this kind at a time"
                )
        if merged:
            ports = {TEMPLATES[k].spec().portfolio.model_dump_json() for k in [spec.name, *merged]}
            if len(ports) > 1:
                spec.portfolio = PortfolioConstruction()
                p.notes.append("The combined ideas use different portfolio constructions -> platform default (quintiles, long-short, "
                               "equal weighted); the templates' own portfolio descriptions above no longer apply.")
            spec.name = "_plus_".join([spec.name, *merged])
        return spec

    def _signal_variants(self, spec: StrategySpec, t: str, p: _Parse) -> None:
        if spec.kind != "cross_sectional":
            return
        feats = [s.feature for s in spec.signal]
        if "return_12m_ex_1m_pct" in feats:
            m = re.search(
                r"\b(\d{1,2})[\s-]*(?:months?|mo|m)\b[\s-]*(?:price\s+|total\s+)?(?:momentum|returns?|winners|performance|lookback)", t
            ) or re.search(r"(?:momentum|returns?|lookback)\s+(?:over|of)\s+(?:the\s+)?(?:last|past|prior)?\s*(\d{1,2})\s+months?\b", t)
            if m and not re.search(r"\b12[\s-]*1\b|skip", t):
                n = int(m.group(1))
                repl = {3: "return_3m_pct", 6: "return_6m_pct"}.get(n)
                if repl:
                    self._swap_feature(spec, "return_12m_ex_1m_pct", repl)
                    p.notes.append(f"'{m.group(0)}' -> momentum signal {repl} ({n}-month return, no skip month).")
                elif n == 12:
                    p.notes.append("12-month momentum interpreted as the standard 12-1 definition (skipping the latest month).")
                elif n != 1:
                    p.unsupported.append(f"{n}-month momentum lookback (catalog has 3, 6 and 12-1 month returns; used 12-1)")
        if "volatility_60d_pct" in feats and re.search(r"\b20[\s-]*(?:day|d)\b", t):
            self._swap_feature(spec, "volatility_60d_pct", "volatility_20d_pct")
            p.notes.append("'20-day' -> volatility signal volatility_20d_pct.")
        if spec.name == "value_fcf" and re.search(r"\bp\s*/?\s*e\b|price[\s-]*to[\s-]*earnings|earnings[\s-]*yield", t):
            if re.search(r"forward|\bntm\b|next[\s-]*(?:12|twelve)|consensus|estimate", t):
                self._swap_feature(spec, "fcf_yield_pct", "earnings_yield_ntm_pct")
                p.notes.append("Forward P/E value -> signal earnings_yield_ntm_pct (consensus NTM earnings yield; no point-in-time "
                               "history in the free edition).")
            else:
                self._swap_feature(spec, "fcf_yield_pct", "earnings_yield_ttm_pct")
                p.notes.append("P/E value -> signal earnings_yield_ttm_pct (trailing reported earnings / market cap, point-in-time "
                               "from filings; say 'forward P/E' for the consensus NTM earnings yield).")
        if re.search(r"sector[\s-]*neutral|industry[\s-]*neutral|within[\s-]*sectors?", t):
            spec.signal = [s.model_copy(update={"sector_neutral": True}) for s in spec.signal]
            p.notes.append("'sector neutral' -> every signal component is ranked within its GICS sector.")

    @staticmethod
    def _swap_feature(spec: StrategySpec, old: str, new: str) -> None:
        spec.signal = [s.model_copy(update={"feature": new}) if s.feature == old else s for s in spec.signal]

    def _portfolio(self, spec: StrategySpec, t: str, p: _Parse) -> None:
        port = spec.portfolio
        if spec.kind != "cross_sectional":
            if spec.kind == "screen":
                for kind, pat in _WEIGHTING:
                    m = re.search(pat, t)
                    if m and kind in ("equal", "value"):
                        port.weighting = kind  # type: ignore[assignment]
                        p.notes.append(f"'{m.group(0)}' -> weighting={kind}.")
                        break
            q = re.search(r"\b(decile|quintile|quartile|tercile|tertile)s?\b", t)
            if q:
                p.notes.append(f"'{q.group(0)}' ignored: a {spec.kind.replace('_', ' ')} strategy has no quantile portfolios.")
            return

        # number of buckets
        q = re.search(r"\b(decile|quintile|quartile|tercile|tertile|vigintile)s?\b", t)
        n_q: int | None = None
        if q:
            n_q = _QUANTILE_WORDS[q.group(1)]
            p.notes.append(f"'{q.group(0)}' -> n_quantiles={n_q}.")
        else:
            m = re.search(r"\b(\d{1,2})\s*(?:quantiles|buckets|portfolios|groups|bins|fractiles)\b", t)
            pct = re.search(r"\b(?:top|bottom|best|worst)\s+(\d{1,2}(?:\.\d+)?)\s*(?:%|percent)", t)
            half = re.search(r"\btop\s+(half|third)\b", t)
            if m and 2 <= int(m.group(1)) <= 20:
                n_q = int(m.group(1))
                p.notes.append(f"'{m.group(0)}' -> n_quantiles={n_q}.")
            elif pct:
                x = float(pct.group(1))
                k = round(100.0 / x) if x > 0 else 0
                if 2 <= k <= 20 and abs(100.0 / k - x) < 0.5:
                    n_q = k
                    p.notes.append(f"'{pct.group(0)}' -> n_quantiles={k}.")
                else:
                    p.unsupported.append(f"'{pct.group(0)}' (not a whole number of equal quantiles; kept {port.n_quantiles} quantiles)")
            elif half:
                n_q = 2 if half.group(1) == "half" else 3
                p.notes.append(f"'{half.group(0)}' -> n_quantiles={n_q}.")
        if n_q is not None:
            port.n_quantiles = n_q
            port.selection = "quantile"
            p.overridden.add("n_quantiles")

        # top N names
        if n_q is None:
            m = re.search(
                r"\b(?:top|best|highest[\s-]*ranked)\s+(\d{1,3})\b(?!\s*(?:%|percent|deciles?|quintiles?|quartiles?|terciles?|days?|"
                r"weeks?|months?|years?|bps|bp|basis))",
                t,
            ) or re.search(r"\b(?:hold|buy|own)\s+(?:the\s+)?(\d{1,3})\s+(?:stocks|names|companies)\b", t) or re.search(
                r"\b(\d{1,3})[\s-]*(?:stock|name)s?\s+portfolio\b", t
            )
            if m and 1 <= int(m.group(1)) <= 500:
                port.selection = "top_n"
                port.top_n = int(m.group(1))
                p.overridden.add("top_n")
                p.notes.append(f"'{m.group(0)}' -> selection=top_n, top_n={port.top_n} names per side.")

        # long-only vs long-short: an explicit short / sell leg ("buy X and sell Y") wins over the
        # generic 'buy' long-only cue
        ls = re.search(
            r"long[\s/-]*short|market[\s-]*neutral|dollar[\s-]*neutral|\bhedged\b|zero[\s-]*(?:cost|investment)|"
            r"\b(?:short|sell)(?:ing)?\s+(?:the\s+)?(?:\d{1,3}\s+)?(?:last\s+(?:week|month|quarter|year)'?s\s+)?"
            r"(?:bottom|worst|lowest|highest|losers|losing|winners|winning|most|least|expensive|priciest|richest|"
            r"overvalued|overbought|junk|top|weakest|strongest|laggards|leaders|(?:low|high)(?=[\s-]+(?!and\b|then\b|or\b)[a-z]))\b|"
            r"(?:top|high|winners|small|best)\s+minus\s+(?:bottom|low|losers|big|worst)|\bspread\s+portfolio",
            t,
        )
        lo = re.search(
            r"long[\s-]*only|\bonly\s+long|no\s+shorts?(?:ing)?\b|without\s+short\w*|\bbuy(?:ing)?\b|\bown\b|"
            r"\bhold(?:ing)?\s+(?:the\s+)?(?:top|best)|\blong\s+(?:the\s+)?(?:top|best|highest|lowest|cheapest|winners)",
            t,
        )
        if ls:
            if port.style != "long_short":
                port.style = "long_short"
                p.notes.append(f"'{ls.group(0)}' -> style=long_short.")
            else:
                p.notes.append(f"'{ls.group(0)}' -> style=long_short (template default).")
            p.overridden.add("style")
        elif lo:
            if port.style != "long_only":
                port.style = "long_only"
                p.notes.append(f"'{lo.group(0)}' -> style=long_only (hold the best {'names' if port.selection == 'top_n' else 'quantile'} only).")
            else:
                p.notes.append(f"'{lo.group(0)}' -> style=long_only (template default).")
            p.overridden.add("style")

        topq = re.search(r"\btop\s+(?:decile|quintile|quartile|tercile|tertile)\b", t)
        if topq and not ls and port.style == "long_short":
            p.notes.append(
                f"'{topq.group(0)}' without 'long only' -> long-short kept as the factor test; the top-quantile leg is "
                "reported on its own as the 'long' series (say 'long only' to hold just that leg)."
            )

        for kind, pat in _WEIGHTING:
            m = re.search(pat, t)
            if m:
                port.weighting = kind  # type: ignore[assignment]
                p.overridden.add("weighting")
                p.notes.append(f"'{m.group(0)}' -> weighting={kind}.")
                break

        m = re.search(
            r"(?:max(?:imum)?|capped\s+at|cap\s+of|no\s+more\s+than|at\s+most)\s+(\d{1,2}(?:\.\d+)?)\s*%\s*"
            r"(?:per|in\s+(?:any|a|each|one))\s+(?:name|stock|position|holding)",
            t,
        )
        if m and 0 < float(m.group(1)) <= 100:
            port.max_weight = float(m.group(1)) / 100.0
            p.notes.append(f"'{m.group(0)}' -> max_weight={port.max_weight:g}.")

    def _rebalance(self, spec: StrategySpec, t: str, p: _Parse) -> None:
        words = {"daily": "daily", "weekly": "weekly", "monthly": "monthly", "quarterly": "quarterly", "annually": "annual",
                 "annual": "annual", "yearly": "annual"}
        m = re.search(r"rebalanc\w*\s+(?:\w+\s+){0,2}?(daily|weekly|monthly|quarterly|annually|annual|yearly)\b", t) or re.search(
            r"\b(daily|weekly|monthly|quarterly|annual|yearly)\s+rebalanc", t
        )
        freq: str | None = words[m.group(1)] if m else None
        if freq is None:
            cands = [
                ("daily", r"\bdaily\b|every\s+(?:1\s+)?day|each\s+day"),
                ("weekly", r"\bweekly\b|every\s+(?:1\s+)?week|each\s+week|once\s+a\s+week"),
                ("monthly", r"\bmonthly\b|every\s+(?:1\s+)?month\b|each\s+month|once\s+a\s+month"),
                ("quarterly", r"\bquarterly\b|every\s+quarter|each\s+quarter|every\s+3\s+months"),
                ("annual", r"\bannual(?:ly)?\b|\byearly\b|every\s+year|once\s+a\s+year|every\s+12\s+months"),
            ]
            found = [(mm.start(), f, mm.group(0)) for f, pat in cands if (mm := re.search(pat, t))]
            if found:
                _, freq, word = min(found)
                m = re.search(re.escape(word), t)
        if freq and m:
            spec.rebalance = freq  # type: ignore[assignment]
            p.overridden.add("rebalance")
            p.notes.append(f"'{m.group(0)}' -> rebalance={freq}.")

    def _dates(self, spec: StrategySpec, t: str, today: date, p: _Parse) -> None:
        ymd = r"((?:19|20)\d{2}(?:-\d{2}(?:-\d{2})?)?)"
        start: date | None = None
        end: date | None = None
        rng = re.search(r"\bbetween\s+" + ymd + r"\s+and\s+" + ymd, t) or re.search(
            r"(?<!russell )\b(?:from\s+)?" + ymd + r"\s*(?:-|to|through|until|thru)\s*" + ymd + r"\b", t
        )
        try:
            if rng:
                start, end = _year_start(rng.group(1)), _year_end(rng.group(2))
                p.notes.append(f"'{rng.group(0).strip()}' -> start={start}, end={end}.")
            else:
                m = re.search(r"\b(?:since|from|starting(?:\s+in)?|beginning(?:\s+in)?|as\s+of)\s+" + ymd + r"\b", t)
                if m:
                    start = _year_start(m.group(1))
                    p.notes.append(f"'{m.group(0)}' -> start={start}.")
                m = re.search(r"\bafter\s+((?:19|20)\d{2})\b", t)
                if m and start is None:
                    start = date(int(m.group(1)) + 1, 1, 1)
                    p.notes.append(f"'{m.group(0)}' -> start={start} (strictly after that year).")
                m = re.search(r"\b(?:until|till|through|thru|ending(?:\s+in)?|up\s+to)\s+" + ymd + r"\b", t)
                if m:
                    end = _year_end(m.group(1))
                    p.notes.append(f"'{m.group(0)}' -> end={end}.")
                m = re.search(r"\bbefore\s+((?:19|20)\d{2})\b", t)
                if m and end is None:
                    end = date(int(m.group(1)) - 1, 12, 31)
                    p.notes.append(f"'{m.group(0)}' -> end={end} (strictly before that year).")
                m = re.search(r"\b(?:last|past|previous|trailing)\s+(\d{1,2})\s+years?\b", t)
                if m and start is None:
                    start = _years_ago(today, int(m.group(1)))
                    p.notes.append(f"'{m.group(0)}' -> start={start} (relative to {today}).")
        except ValueError:  # e.g. month 13
            p.unsupported.append("an unparseable date in the idea (kept the full history)")
            start = end = None
        if end is not None and end > today:
            end = None
            p.notes.append("End date is in the future -> end = latest available.")
        if start is not None:
            spec.start = start
            p.overridden.add("start")
        if end is not None:
            spec.end = end
            p.overridden.add("end")

    def _costs(self, spec: StrategySpec, t: str, p: _Parse) -> None:
        """Transaction costs: '25 bps', '0.15% transaction costs', 'costs of 0.2%', '0.1% per trade',
        'no costs'. Cost-like amounts that are not per-trade costs (management / performance fees,
        annual figures, cents per share, ...) go to ``unsupported_requests`` - the default is never
        kept silently while the idea states a different cost."""
        default = spec.costs_bps
        reported = False
        fee = re.search(_FEE_KIND, t)
        if fee:
            p.unsupported.append(f"'{fee.group(0)}' (management / performance fees are not modelled; only per-trade transaction costs)")
            reported = True
        zero = re.search(
            r"(?:\bno|\bzero|without|ignor\w*|exclud\w*)\s+(?:transaction\s+|trading\s+)?(?:costs?|commissions?|fees?|slippage)|"
            r"frictionless|gross\s+of\s+costs",
            t,
        )
        cands: list[tuple[int, float, str]] = []  # (position, bps, source phrase)
        for pat, scale in _COST_AMOUNTS:
            for m in re.finditer(pat, t):
                cands.append((m.start(), float(m.group(1)) * scale, m.group(0).strip()))
        usable: list[tuple[int, float, str]] = []
        for pos, v, src in sorted(cands):
            end = pos + len(src)
            if re.search(_NOT_A_COST_BEFORE, t[max(0, pos - 25):pos]):
                continue  # 'max 5% per side' is a position limit
            if re.search(_FEE_KIND, t[max(0, pos - 30):end]):
                continue  # the amount of a management / performance fee (reported above)
            if re.match(_ANNUAL_AFTER, t[end:end + 25]):
                p.unsupported.append(f"'{src}' a year (an annual cost figure; costs are charged per trade - kept {default:g} bps one-way)")
                reported = True
                continue
            usable.append((pos, v, src))
        val: float | None = None
        src = ""
        if zero:
            val, src = 0.0, zero.group(0)
        elif usable:
            _, val, src = usable[0]
        if val is None:
            odd = None if reported else (re.search(_COST_ODD_AFTER, t) or re.search(_COST_ODD_BEFORE, t))
            if odd:
                p.unsupported.append(
                    f"transaction cost '{odd.group(0).strip()}' (not readable as bps or % of the amount traded; kept {default:g} bps one-way)"
                )
            return
        if val and re.search(r"round[\s-]*(?:trip|turn)", t):
            val /= 2.0
            p.notes.append(f"'{src}' is a round-trip figure -> {val:g} bps one-way.")
        spec.costs_bps = val
        p.overridden.add("costs_bps")
        p.notes.append(f"'{src}' -> costs_bps={val:g} (one-way).")

    def _universe(self, spec: StrategySpec, t: str, p: _Parse) -> None:
        if spec.kind == "cross_sectional" and spec.name != "size":
            caps = [
                (r"\bsmall[\s-]*caps?\b", Condition(feature="market_cap_usd_bn", op="<", value=2.0, rationale="small caps")),
                (r"\bmid[\s-]*caps?\b", Condition(feature="market_cap_usd_bn", op="between", value=2.0, value_high=10.0, rationale="mid caps")),
                (r"\bmega[\s-]*caps?\b", Condition(feature="market_cap_usd_bn", op=">", value=200.0, rationale="mega caps")),
                (r"\blarge[\s-]*caps?\b", Condition(feature="market_cap_usd_bn", op=">", value=10.0, rationale="large caps")),
            ]
            for pat, cond in caps:
                m = re.search(pat, t)
                if m:
                    spec.filters.append(cond)
                    p.notes.append(f"'{m.group(0)}' -> filter {cond.describe()}.")
                    break
        if spec.kind in ("cross_sectional", "screen"):
            idx = re.search(r"s&p[\s-]*(?:500|400|600)|russell[\s-]*(?:1000|2000|3000)|nasdaq[\s-]*100|\bdow\b|\bdjia\b", t)
            if idx:
                p.unsupported.append(
                    f"'{idx.group(0)}' constituent universe (point-in-time index membership is not available; the platform universe is used)"
                )
        excl: list[str] = []
        for m in re.finditer(r"(?:\bex(?:cluding|cl\.?|cept)?|without|\bno)[\s-]+((?:[a-z&\s-]+?))(?=$|[,.;)]|\s+(?:and\s+)?(?:with|since|from|rebalanc|long|short|monthly|weekly|quarterly|daily|\d))", t):
            chunk = m.group(1)
            for pat, sector in _GICS_SECTORS.items():
                if re.search(rf"\b(?:{pat})\b", chunk) and sector not in excl:
                    excl.append(sector)
        if excl:
            spec.universe.exclude_sectors = [*spec.universe.exclude_sectors, *[s for s in excl if s not in spec.universe.exclude_sectors]]
            p.notes.append(f"Excluded sectors: {', '.join(excl)}.")

    def _timing(self, spec: StrategySpec, idea: str, t: str, p: _Parse) -> None:
        ts = spec.time_series
        assert ts is not None
        tickers = self._explicit_tickers(idea)
        for pat, etf in _INDEX_ALIASES:
            if re.search(pat, t) and etf not in tickers:
                tickers.append(etf)
                p.notes.append(f"'{re.search(pat, t).group(0)}' -> traded via the {etf} ETF.")  # type: ignore[union-attr]
        if tickers:
            ts.assets = tickers
            spec.benchmark = tickers[0] if len(tickers) == 1 else None
            p.notes.append(f"Traded assets: {', '.join(tickers)}" + (f"; benchmark = buy-and-hold {tickers[0]}." if len(tickers) == 1 else "; benchmark = default broad index."))
        else:
            p.notes.append(f"No asset named -> trades {', '.join(ts.assets)} (template default).")

        if spec.name == "trend_200dma_spy":
            windows = [int(w) for w in re.findall(r"\b(\d{2,3})[\s-]*(?:day|d|dma|sma|ma)\b", t)]
            other = [w for w in windows if w not in (20, 50, 200)]
            if "faber" in t or re.search(r"\b10[\s-]*month", t):
                p.notes.append("Faber's 10-month moving average is approximated by the 200-day moving average.")
                p.unsupported.append("10-month moving average (approximated by the 200-day)")
            if other:
                p.unsupported.append(f"{other[0]}-day moving average (catalog has 20, 50 and 200-day averages; used the 200-day)")
            elif windows and windows[0] in (20, 50):
                w = windows[0]
                if w == 50:
                    ts.entry = [Condition(feature="price_vs_sma_50_pct", op=">", value=0.0, rationale="price above its 50-day moving average")]
                else:
                    ts.entry = [Condition(feature="price", op=">", other_feature="sma_20", rationale="price above its 20-day moving average")]
                p.notes.append(f"'{w}-day' -> entry: {ts.entry[0].describe()}.")
        if re.search(
            r"(?:otherwise|else|or)\s+(?:go\s+)?short|short\s+(?:it\s+|spy\s+|the\s+market\s+)?(?:when|if|while)\s+(?:it\s+is\s+|it's\s+)?below|"
            r"short\s+below|long[\s/-]*short",
            t,
        ):
            ts.when_flat = "short"
            p.notes.append("Short (not cash) when the entry condition fails.")
        else:
            p.notes.append("In cash (earning T-bills) when the entry condition fails.")

    def _unsupported(self, spec: StrategySpec, t: str, p: _Parse) -> None:
        for pat, what in _UNSUPPORTED:
            if re.search(pat, t):
                p.unsupported.append(what)

    @staticmethod
    def _defaults_note(spec: StrategySpec, p: _Parse) -> None:
        kept: list[str] = []
        if "rebalance" not in p.overridden:
            kept.append(f"rebalance {spec.rebalance}")
        if spec.kind == "cross_sectional":
            port = spec.portfolio
            if not ({"n_quantiles", "top_n"} & p.overridden):
                kept.append(f"{port.n_quantiles} quantiles" if port.selection == "quantile" else f"top {port.top_n}")
            if "style" not in p.overridden:
                kept.append(port.style.replace("_", "-"))
            if "weighting" not in p.overridden:
                kept.append(f"{port.weighting} weighted")
        if "costs_bps" not in p.overridden:
            kept.append(f"costs {spec.costs_bps:g} bps one-way")
        if "start" not in p.overridden:
            kept.append("start = maximum available history")
        if "end" not in p.overridden:
            kept.append("end = latest available")
        if "attribution_model" not in p.overridden and spec.attribution_model:
            kept.append(f"attribution {spec.attribution_model}")
        if kept:
            p.notes.append("Defaults kept: " + ", ".join(kept) + ".")


# ------------------------------------------------------------------------------------------------
# Convenience
# ------------------------------------------------------------------------------------------------


def translate_idea(
    idea: str,
    llm: StructuredLLM | None = None,
    *,
    catalog: FeatureCatalog | None = None,
    today: date | None = None,
    max_repair_rounds: int = 2,
    effort: str = "high",
) -> StrategyTranslation:
    """Translate with Claude when ``llm`` is given, falling back to the offline heuristic.

    The fallback is used when the LLM call fails (``LLMError``, including refusals) or the model
    cannot produce a valid spec; the reason is recorded in the spec's assumptions. If the heuristic
    cannot match the idea either, the original error is raised (with template suggestions).
    """
    heuristic = HeuristicStrategyTranslator(catalog, today=today)
    if llm is None:
        return heuristic.translate(idea)
    try:
        return StrategyTranslator(llm, catalog, max_repair_rounds=max_repair_rounds, effort=effort, today=today).translate(idea)
    except (LLMError, StrategyTranslationError) as e:
        try:
            out = heuristic.translate(idea)
        except StrategyTranslationError as h:
            if isinstance(e, StrategyTranslationError):
                raise StrategyTranslationError(str(e), errors=e.errors, suggestions=e.suggestions or h.suggestions) from h
            raise h from e
        out.spec.assumptions.append(f"Claude translation failed ({type(e).__name__}: {str(e)[:200]}); used the offline heuristic translator.")
        return out
