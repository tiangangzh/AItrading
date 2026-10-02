"""Deterministic narrative tagging: lexicon / regex rules over management and news sentences.

Each rule maps phrasing to a tag with a polarity relative to the dislocation question:
``+1`` supports a transitory / mispricing view, ``-1`` points to a structural or bearish reading,
``0`` is informative but neutral. The output is a compact, reproducible summary of what the
documents say (e.g. "3x transitory_language, 2x no_share_loss, 1x evasive_answer") that the
pipeline can show next to the LLM's explanation; it never drives selection.

Scope rules
-----------
* Transcripts: management turns only (operator and analyst turns are skipped), with the speaker
  recorded. Questions (sentences ending in "?") are skipped everywhere.
* Negation: a cue ("not", "no", "never", "n't", "without", "none", ...) before a match within the
  same clause (at most 15 words back, stopping at ";", ":", dashes, "but", "however", ", and" ...),
  inside a gap of the match ("orders did not grow"), or a negated predicate right after it
  ("share loss was not a factor") negates the match. Pseudo-negations ("not only", "no doubt")
  are ignored, and verbs such as "offset", "anticipate" or "immune" break the scope ("we could
  not offset the pricing pressure" is still pricing pressure).
* A negated bearish concern becomes ``no_<tag>`` with polarity +1 ("not seeing any share loss"
  -> ``no_share_loss``); some rules map negation to another tag (negated ``pricing_power`` ->
  ``pricing_pressure``); otherwise a negated match is dropped. Each tag is reported at most once
  per sentence, and a non-negated match wins over a negated one.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Union

from aitrading.core.models import Document, DocumentKind, TranscriptSegment
from aitrading.narrative.excerpts import paragraph_spans, sentence_spans

__all__ = [
    "NarrativeSignal",
    "TAG_POLARITY",
    "extract_signals",
    "extract_signals_from_documents",
    "summarize_signals",
    "net_polarity",
    "is_negated",
]


@dataclass(frozen=True)
class NarrativeSignal:
    tag: str
    polarity: int  # +1 supportive of a transitory / dislocation view, -1 structural / bearish, 0 neutral
    sentence: str  # verbatim sentence from the document
    doc_id: str
    speaker: str | None = None


# --------------------------------------------------------------------------------------------
# Rule table
# --------------------------------------------------------------------------------------------

REFUTE = "refute"  # on_negated marker: emit ``no_<tag>`` with polarity +1
_FLAGS = re.IGNORECASE


def _gap(n: int) -> str:
    """Gap of up to ``n`` words (lazy, never crossing ; : ! ?) between two phrases."""
    return rf"(?:\s+[^\s;:!?]+){{0,{n}}}?\s+"


_PatternSpec = Union[str, tuple[str, bool]]  # regex, or (regex, negatable)


@dataclass(frozen=True)
class _Rule:
    tag: str
    polarity: int
    patterns: tuple[tuple[re.Pattern[str], bool], ...]
    on_negated: str | tuple[str, int] | None = None  # REFUTE, (tag, polarity) or None (drop)
    requires: re.Pattern[str] | None = None  # sentence must also match this
    dampener: re.Pattern[str] | None = None  # nearby word that negates the match (e.g. "low churn")
    suppresses: tuple[str, ...] = ()  # tags dropped from a sentence where this rule fires


def _rule(tag: str, polarity: int, patterns: Iterable[_PatternSpec],
          on_negated: str | tuple[str, int] | None = None, requires: str | None = None,
          dampener: str | None = None, suppresses: tuple[str, ...] = ()) -> _Rule:
    compiled = []
    for p in patterns:
        rx, neg = (p, True) if isinstance(p, str) else p
        compiled.append((re.compile(rx, _FLAGS), neg))
    return _Rule(tag, polarity, tuple(compiled), on_negated,
                 re.compile(requires, _FLAGS) if requires else None,
                 re.compile(dampener, _FLAGS) if dampener else None, suppresses)


_V_UP = r"(?:rais(?:e|es|ed|ing)|increas(?:e|es|ed|ing)|lift(?:s|ed|ing)?|boost(?:s|ed|ing)?|upgrad(?:e|es|ed|ing)|hik(?:e|es|ed|ing))"
_V_DOWN = (r"(?:lower(?:s|ed|ing)?|reduc(?:e|es|ed|ing)|cut(?:s|ting)?|trim(?:s|med|ming)?|downgrad(?:e|es|ed|ing)|"
           r"tak(?:e|es|ing)\s+down|took\s+down|bring(?:s|ing)?\s+down|brought\s+down|revis(?:e|es|ed|ing)\s+(?:down|lower))")
_GUIDE = (r"(?:guidance|outlook|forecasts?|guide|full[- ]year (?:targets?|range|view|expectations)|"
          r"(?:revenue|sales|earnings|eps|ebitda|margin) (?:guidance|outlook|range))")
_DEMAND = (r"(?:end[- ])?(?:demand|orders|order (?:intake|rates?|patterns?|activity)|bookings|traffic|spending|spend|"
           r"budgets?|consumption|sell[- ]through)")
_BACKLOG = r"(?:backlog|book[- ]to[- ]bill|order book|pipeline|remaining performance obligations?|rpo|bookings)"
_EXEC = r"\b(?:ceo|cfo|coo|cto|president|chair(?:man|woman|person)?|chief \w+ officer|founder|executive|head of|board)\b"

_RULES: tuple[_Rule, ...] = (
    _rule("guidance_raised", +1, [
        rf"\b{_V_UP}\b{_gap(4)}{_GUIDE}\b",
        rf"\b{_GUIDE}\b{_gap(3)}(?:raised|increased|lifted|boosted|moved (?:up|higher))\b",
        rf"\braise[sd]?\b{_gap(2)}(?:low end|midpoint|high end|bottom end|top end)\b",
    ]),
    _rule("guidance_lowered", -1, [
        rf"\b{_V_DOWN}\b{_gap(4)}{_GUIDE}\b",
        rf"\b{_GUIDE}\b{_gap(3)}(?:lowered|reduced|cut|trimmed|revised (?:down|lower))\b",
        rf"\b{_GUIDE}\b[^.;]{{0,60}}?\bbelow\b[^.;]{{0,25}}?\b(?:consensus|the street|street (?:expectations|estimates)|"
        r"expectations|prior (?:guidance|outlook|range))\b",
    ]),
    _rule("guidance_withdrawn", -1, [
        rf"\b(?:withdr[ae]w(?:s|n|ing)?|suspend(?:s|ed|ing)?|pull(?:s|ed|ing)?|rescind(?:s|ed|ing)?)\b{_gap(3)}"
        r"(?:guidance|outlook|forecasts?|(?:financial )?framework)\b",
        (rf"\bno longer (?:provid|giv|reaffirm|reiterat|stand(?:ing)? behind|confident in)\w*\b{_gap(4)}"
         r"(?:guidance|outlook|forecasts?|(?:financial )?framework|targets?)\b", False),
    ]),
    _rule("guidance_reaffirmed", +1, [
        rf"\b(?:reaffirm(?:s|ed|ing)?|reiterat(?:e|es|ed|ing)|maintain(?:s|ed|ing)?|confirm(?:s|ed|ing)?|"
        rf"stand(?:s|ing)? behind)\b{_gap(4)}(?:guidance|outlook|forecasts?|(?:financial )?framework|"
        r"full[- ]year (?:targets?|range|view)|(?:financial|margin|long-term|medium-term) targets?)\b",
    ], on_negated=("guidance_withdrawn", -1)),
    _rule("guidance_conservative", +1, [
        (rf"\b(?:conservative|prudent|cautious|de-?risked|realistic|achievable|beatable)\b{_gap(3)}"
         r"(?:guidance|outlook|forecasts?|guide|view|approach|assumptions?|planning|plan|range)\b", False),
        (r"\b(?:guidance|outlook|forecasts?|guide|range)\b[^.;]{0,60}?\b(?:conservative|conservatism|prudent|"
         r"cushions?|buffers?|de-?risk(?:ed)?|cautious)\b", False),
        (rf"\b(?:embed(?:s|ded)?|buil(?:d|ds|t) in|bak(?:e|es|ed) in|includ(?:e|es|ed))\b{_gap(4)}"
         r"(?:cushions?|buffers?|contingency|conservatism)\b", False),
        (rf"\b(?:assum(?:e|es|ed|ing)|contemplat(?:e|es|ed|ing)|bak(?:e|es|ed) in)\b{_gap(2)}(?:no|any) "
         r"(?:improvement|recovery|rebound|pick-?up|benefit)\b", False),
        (r"\b(?:expect|confident (?:that )?we can|intend|plan|aim)(?:s|ed)? to beat\b|\bwe can beat\b", False),
        (rf"\b(?:finish|land|come in|end up)\b{_gap(2)}above the (?:high|top) end\b", False),
    ]),
    _rule("transitory_language", +1, [
        r"\b(?:one[- ]time|one[- ]off|non[- ]recurring|transitory|temporary|(?<!only )temporarily|short[- ]lived|"
        r"isolated (?:event|incident|issue)|discrete (?:event|item|issue)|timing (?:issue|matter|difference|shift)|"
        r"behind us|stable since|catch[- ]up|shifted into|realign with|"
        r"(?:has|have) (?:been )?(?:resolved|fixed|remediated|normali[sz]ed)|(?:fully|largely) (?:resolved|recovered))\b",
        rf"\b(?:expect|anticipate)(?:s|d|ed)?\b{_gap(5)}to (?:reverse|recover|normali[sz]e|come back|recapture|"
        r"catch up|unwind)\b",
        r"\b(?:reverses?|recovers?|normali[sz]es?|comes? back)\b[^.;]{0,20}\bas (?:volumes?|demand|shipments|production|"
        r"supply|conditions|inventor(?:y|ies)|ordering)\b",
        (r"\bnot indicative of\b|\b(?:deferred|delayed|shifted),? not lost\b|\b(?:did not|didn't|has not|hasn't) "
         r"(?:go|gone) away\b|\b(?:do not|don't|not) (?:expect|anticipate|see) (?:a |any )?(?:repeat|recurrence|"
         r"further impact|lasting impact)\b", False),
    ], on_negated=("structural_concern", -1)),
    _rule("structural_concern", -1, [
        r"\b(?:structural(?:ly)?|secular(?:ly)?|permanent(?:ly)?|new normal|disintermediat\w*|obsolete|obsolescence|"
        r"commoditi[sz]\w*|cannibali[sz]\w*|(?:goodwill )?impairment)\b",
        rf"\b(?:competitive intensity|competition|competitive (?:pressure|environment|landscape))\b{_gap(3)}"
        r"(?:increased|intensified|heated up|became more|has become more|got tougher)\b",
        rf"\b(?:new|lower[- ]priced|low[- ]cost|disruptive)\b{_gap(1)}(?:entrants?|competitors?|alternatives?|challengers?)\b",
        rf"\b(?:remain|stay)s?\b{_gap(1)}(?:challenging|difficult|pressured|under pressure|weak|soft|tough)\b[^.;]{{0,30}}"
        r"\b(?:for (?:several|multiple|a number of|the next (?:few|several)|some) (?:quarters|years)|for (?:some time|an "
        r"extended (?:period|time)|longer)|through(?:out)? (?:next|the) year)\b",
        rf"\b(?:transition|ramp|launch|product cycle|platform)\b{_gap(4)}(?:is|was|are|were|has been|remains)\b{_gap(1)}"
        r"(?:behind|delayed|slipping|late)\b",
    ], on_negated=REFUTE),
    _rule("demand_weakness", -1, [
        rf"\b(?:soft(?:er|ening|ness)?|weak(?:er|ening|ness)?|slow(?:er|ing|down|-down|ed)?|decelerat\w*|"
        rf"declin(?:e|es|ed|ing)|lower|reduced|deteriorat\w*|sluggish|muted|subdued|tepid|cautious|pull-?back|"
        rf"pulled back)\b{_gap(2)}(?:{_DEMAND}|(?:unit )?volumes)\b",
        rf"\b(?:{_DEMAND}|volumes)\b{_gap(3)}(?:softened|weakened|slowed|declined|decelerated|deteriorated|fell|dropped|"
        r"contracted|(?:was|were|is|are|remained|remains|has been|have been) (?:soft|weak|down|muted|sluggish|subdued|"
        r"lower|pressured))\b",
        r"\blonger (?:decision|sales|buying|purchasing|approval) cycles\b",
        rf"\b(?:customers?|clients?|buyers?)\b{_gap(2)}(?:delay(?:ing|ed)?|defer(?:ring|red)?|push(?:ing|ed) out|"
        r"paus(?:ing|ed)|postpon\w*)\b",
        r"\bmacro(?:economic)? (?:headwinds?|uncertainty|softness|weakness)\b",
        r"\b(?:cancell?ations?|cancell?(?:ed|ing)|push-?outs?|order deferrals?)\b",
        rf"\b(?:backlog|book[- ]to[- ]bill|order book|bookings)\b{_gap(4)}(?:declin\w*|fell|down \d|decreas\w*|shrank|"
        r"contracted|below (?:one|1(?:\.0)?)\b)",
        rf"\bdeteriorat\w*\b{_gap(2)}(?:the |our )?(?:business|fundamentals|end markets?|trends?)\b",
    ], on_negated=REFUTE),
    _rule("demand_recovery", +1, [
        rf"\b(?:{_DEMAND}|end markets?)\b{_gap(3)}(?:recover(?:ed|ing|y)?|rebound(?:ed|ing)?|improv(?:e|ed|es|ing)|"
        r"re-?accelerat\w*|inflect\w*|picked up|picking up|pick up|stabili[sz]\w*|bottom(?:ed|ing) out|"
        r"turn(?:ed|ing) (?:up|the corner)|came back|coming back)\b",
        rf"\b(?:recover(?:y|ing|ed)|rebound(?:ed|ing)?|improv(?:ed|ing|ement)|re-?accelerat\w*|stabili[sz]\w*|"
        rf"inflection|pick-?up|uptick)\b{_gap(2)}(?:in )?{_DEMAND}\b",
    ]),
    _rule("demand_resilience", +1, [
        rf"\b(?:{_DEMAND}|end markets?|underlying demand|demand indicators)\b{_gap(6)}(?:grew|growing|grow|increased|rose|"
        r"(?:is|are|was|were|remain(?:s|ed)?|ha(?:s|ve) been|stayed|continue(?:s|d)? to be) (?:healthy|strong|robust|"
        r"solid|resilient|intact|stable|unaffected|unchanged|a record|at a record)\b|(?:running )?up \d|unaffected\b|"
        r"intact\b)",
        rf"\b(?:healthy|strong|robust|solid|resilient|record|intact|steady|growing)\b{_gap(1)}{_DEMAND}\b",
    ], on_negated=("demand_weakness", -1)),
    _rule("backlog_strength", +1, [
        rf"\b{_BACKLOG}\b{_gap(8)}(?:record|up \d|grew|increased|rose|expanded|(?:at|of|was|is) 1\.(?:0[1-9]|[1-9])\d*|"
        r"above (?:one|1(?:\.0)?)\b|healthy|strong|robust|intact|all-time high|never been fuller)",
        rf"\b(?:record|strong|robust|healthy|growing|solid)\b{_gap(1)}{_BACKLOG}\b",
    ]),
    _rule("margin_pressure", -1, [
        rf"\bmargins?\b{_gap(8)}(?:down \d|declin(?:e|ed|ing)|contract(?:ed|ing)|compress(?:ed|ing)|fell|"
        r"decreas(?:e|ed|ing)|erod(?:ed|ing)|pressured|under pressure)",
        r"\bmargin (?:compression|pressure|erosion|headwinds?|contraction|decline|squeeze)\b",
        rf"\b(?:compress(?:ed|ing)?|pressur(?:e|ed|ing)|squeez(?:e|ed|ing)|erod(?:e|ed|ing)|weigh(?:ed|ing)? on|"
        rf"dilut(?:e|ed|ive))\b{_gap(2)}(?:gross |operating )?margins?\b",
        r"\b(?:cost|input[- ]cost|wage|labou?r|freight|raw[- ]material|commodity)[- ]inflation\b",
        r"\b(?:under-?absorption|lower absorption|unabsorbed)\b",
    ], on_negated=REFUTE),
    _rule("pricing_power", +1, [
        rf"\bprice realization\b{_gap(2)}(?:positive|up \d|\+\s?\d|\d)",
        r"\bprice(?:[- ]cost| versus cost)(?:\s+(?:spread|was|is|remained|remains|has been))*\s+(?:positive|favou?rable)\b",
        rf"\b(?:price|pricing)\b{_gap(1)}(?:contributed|added)\b",
        rf"\b(?:pass(?:ed|ing)?|push(?:ed|ing)?)\b{_gap(1)}(?:through|on)\b{_gap(2)}(?:price|pricing|cost)\b",
        r"\bpricing (?:power|discipline)\b|\b(?:favou?rable|positive|firm|strong) (?:net )?pricing\b",
        rf"\bprice (?:increases?|actions?|hikes?)\b{_gap(3)}(?:stick(?:ing)?|stuck|held|holding|took hold|realized)\b",
    ], on_negated=("pricing_pressure", -1)),
    _rule("pricing_pressure", -1, [
        rf"\b(?:pricing|price)\b{_gap(2)}(?:pressures?|concessions?|competition|wars?|erosion|headwinds?|declines?|cuts?|"
        r"compression|problems?|issues?|degradation)\b",
        rf"\b(?:lower|declining|falling|reduced|weaker)\b{_gap(1)}(?:average )?(?:selling )?prices?\b",
        r"\b(?:discounting|aggressive(?:ly)? pric\w*|competitive (?:tenders?|bids?|bidding|pricing)|"
        r"promotional (?:activity|intensity|environment)|price[- ]matching)\b",
        rf"\bpricing environment\b{_gap(2)}(?:became|has become|is|was|remains?|got|grew)\b{_gap(1)}(?:more )?"
        r"(?:difficult|challenging|competitive|tough(?:er)?|aggressive|harder)\b",
        rf"\bmatch(?:ed|ing)?\b{_gap(2)}(?:competitors?|competition|rivals?)\b{_gap(1)}on price\b",
        r"\b(?:negative|unfavou?rable) (?:price|pricing)\b|\bprice[- ]cost (?:was |is )?(?:negative|unfavou?rable)\b",
    ], on_negated=REFUTE),
    _rule("share_loss", -1, [
        rf"\b(?:los(?:e|es|t|ing)|ced(?:e|ed|ing)|g[ai]ve up|giving up)\b{_gap(2)}(?:(?:market|wallet|unit) )?"
        r"(?<!per )share\b(?! (?:count|price|repurchase|buyback))",
        r"\bloss(?:es)? of (?:(?:market|wallet) )?share\b|\bshare loss(?:es)?\b",
        rf"\b(?:competitors?|rivals?|peers?|new entrants?|challengers?)\b{_gap(3)}(?:gain(?:ed|ing|s)?|took|taking|"
        rf"tak(?:e|es)|winning|won|win)\b{_gap(2)}(?:(?:market )?share|business|customers|accounts|contracts|deals)\b",
        rf"\blost\b{_gap(2)}(?:contracts?|programs?|platforms?|bids?|tenders?|deals?|competitive evaluations?|business)"
        rf"\b{_gap(2)}to\b",
    ], on_negated=REFUTE),
    _rule("customer_churn", -1, [r"\b(?:churn(?:ed|ing)?|attrition|non-renewals?|de-?bookings?)\b"], on_negated=REFUTE,
          dampener=r"\b(?:low|lower|reduced|improv\w*|stable|minimal|record[- ]low|declin\w*|decreas\w*|fell)\b"),
    _rule("customer_churn", -1, [
        (r"\b(?:did(?: not|n't) renew|not renewing)\b", False),
        rf"\blos(?:e|es|t|ing)\b{_gap(4)}(?:customers?|clients?|accounts?|logos?)\b",
        rf"\b(?:customers?|clients?)\b{_gap(2)}(?:left|leaving|moved to|moving to|switched|switching|defected|insourc\w*|"
        r"in-sourc\w*|consolidat(?:ed|ing|e) (?:vendors|suppliers|purchasing|spend))\b",
        rf"\b(?:net revenue retention|net retention|gross retention|retention|renewal rates?)\b{_gap(8)}(?:declin\w*|"
        r"fell|dropped|dipped|down|lower|below|weaker|deteriorat\w*|softened)\b",
    ], on_negated=REFUTE),
    _rule("inventory_destocking", 0, [
        r"\bde-?stock\w*\b",
        r"\binventory (?:destocking|reduction|correction|digestion|normali[sz]ation|rebalancing|drawdown|burn|adjustments?)\b",
        rf"\b(?:customers?|distributors?|channel partners?|partners|retailers?|dealers?|oems?)\b{_gap(3)}(?:reduc\w*|"
        rf"work(?:ed|ing)? down|draw(?:ing|n)? down|drew down|lower(?:ed|ing)?|right-?siz\w*|trim\w*)\b{_gap(1)}"
        r"(?:their |its |channel )?(?:inventor(?:y|ies)|stock(?:s|ing)? levels)\b",
        r"\b(?:channel|excess|elevated|bloated) inventor(?:y|ies)\b|\binventory overhang\b",
    ]),
    _rule("fx_headwind", 0, [
        rf"\b(?:fx|foreign (?:currency|exchange)|currenc(?:y|ies)|(?:stronger|strong) (?:u\.s\. )?dollar|translation)"
        rf"\b{_gap(6)}(?:headwinds?|hurt|reduced|weigh(?:ed|s|ing)?|pressure|drag|unfavou?rable|negative(?:ly)?|"
        r"impact(?:ed)?)\b",
        rf"\b(?:unfavou?rable|adverse|negative)\b{_gap(1)}(?:fx|foreign (?:currency|exchange)|currency|translation)\b",
        r"\bdevaluation\b",
    ]),
    _rule("capital_return_cut", -1, [
        rf"\b(?:suspend(?:s|ed|ing)?|cut(?:s|ting)?|eliminat(?:e|es|ed|ing)|paus(?:e|es|ed|ing))\b{_gap(2)}"
        r"(?:dividends?|buy-?backs?|(?:share )?repurchases?)\b",
    ], suppresses=("capital_return",)),
    _rule("capital_return", +1, [
        r"\b(?:buy-?backs?|(?:share|stock) repurchases?|repurchas(?:e|es|ed|ing)|dividends?|"
        r"return(?:ed|ing)? (?:\S+ ){0,2}?(?:cash|capital) to (?:shareholders|stockholders|investors)|"
        r"return excess cash)\b",
    ]),
    _rule("balance_sheet_strength", +1, [
        r"\bnet cash(?! (?:provided|used|flows?|from|generated|inflows?|outflows?))(?: position)?\b",
        (r"\b(?:no (?:net )?debt|debt[- ]free)\b", False),
        r"\b(?:(?:strong|fortress|solid|healthy|pristine|flexible|under-?levered|underleveraged) balance sheet|"
        r"balance sheet (?:is |remains )?(?:strong|healthy|solid|flexible|pristine|gives us)|ample liquidity|"
        r"(?:considerable|plenty of|significant|ample|financial) flexibility|investment[- ]grade)\b",
        rf"\bnet leverage\b{_gap(2)}(?:of |at |was |is )?(?:0\.\d+|1\.[0-5]\d*)(?:x| times)",
        r"\b(?:below|under) (?:one|two|1|2)(?:\.\d)?(?:x| times) (?:net )?leverage\b",
    ]),
    _rule("management_change", 0, [
        r"\b(?:resign(?:s|ed|ing|ation)?|retir(?:e|es|ed|ing|ement)|step(?:s|ped|ping)? down|depart(?:s|ed|ing|ure)|"
        r"appoint(?:s|ed|ing|ment)|succe(?:ed|eds|eded|eding|ssion|ssor)|interim|leadership (?:change|transition)|"
        r"management (?:change|transition)|named)\b",
    ], requires=_EXEC),
    _rule("evasive_answer", -1, [(rx, False) for rx in (
        r"\bwe (?:do not|don't) (?:break (?:that|it|this|those) out|break out|disclose|split out|parse|"
        r"give (?:that|specific)|provide (?:that|specific)|guide (?:to|on))\b",
        r"\btoo (?:early|soon) to (?:tell|say|comment|call|quantify|know|predict)\b",
        r"(?:\bnot|n't) (?:going|gonna|want|willing|able|prepared|in a position) to (?:speculate|comment|get into|"
        r"go into|parse|quantify|provide|break|give|discuss|disclose|talk about|call)\b",
        r"\bwon't (?:speculate|comment|get into|go into|parse|quantify|provide|break|give)\b",
        r"(?:\bdo not|n't) think (?:it's|it is|that's|that is) (?:productive|useful|helpful|appropriate|constructive)\b",
        r"\b(?:i|we)(?: would|'d) (?:rather|prefer) not\b",
        r"\bnot (?:something|a number|a metric|a figure|a level of detail) (?:that )?we (?:disclose|provide|"
        r"break out|guide to|comment on|share)\b",
        r"\b(?:move|moving|moved|shift|shifting) to disclosing\b",
        r"\bno longer (?:disclose|disclosing|provide|providing|report|reporting|break(?:ing)? out)\b",
        r"\b(?:stop|stopped|discontinu\w*) (?:disclosing|providing|reporting|breaking out)\b",
        r"\b(?:when|once|until) we have (?:better|more|greater|clearer) (?:visibility|clarity)\b",
        r"\bdecline[sd]? to (?:comment|quantify|provide|specify|elaborate|disclose)\b",
        r"\bthe things we can control\b",
    )]),
    _rule("peer_contagion", +1, [
        rf"\b(?:peer'?s?|peers|industry|sector|group)\b{_gap(3)}(?:profit warning|warning|warned|headline|sell-?off|"
        r"selloff|sold off|cut (?:its|their) (?:outlook|guidance))\b",
        r"\b(?:traded|trade|sold|selling|moved) (?:\S+ ){0,2}?(?:together|in sympathy|with the group|as a group)\b|"
        r"\b(?:selling|sold) the (?:group|sector) broadly\b|\bguilt by association\b",
    ]),
    _rule("limited_exposure", +1, [
        rf"\b(?:limited|minimal|small|modest|negligible|immaterial|little|no direct)\b{_gap(1)}"
        r"(?:direct |indirect )?exposure\b",
        rf"\bexposure\b{_gap(4)}(?:is|was|are|were|remains)\b{_gap(1)}(?:limited|minimal|small|modest|negligible|"
        r"immaterial|less than)\b",
        (r"\bnot (?:directly )?(?:affected|exposed|impacted)\b", False),
    ]),
    _rule("limited_exposure", +1, [rf"\bless than \d+(?:\.\d+)?%{_gap(4)}(?:revenue|sales|net sales)\b"],
          requires=r"\b(?:exposure|exposed|tied to|affected|linked to|related to)\b"),
    _rule("results_beat", +1, [
        rf"\b(?:beat|beats|exceeded|topped|ahead of)\b{_gap(2)}(?:consensus|expectations|estimates|the street|"
        r"street estimates|our (?:outlook|guidance))\b",
        r"\b(?:came in|was|were|finished|landed|ended|exceeded) (?:\S+ ){0,3}?above the (?:high end|top end|midpoint) "
        r"of (?:our|the) (?:prior )?(?:guidance|outlook|range)\b",
    ], on_negated=("results_miss", -1)),
    _rule("results_miss", -1, [
        r"\b(?:fell|came in|was|were|landed) (?:\S+ ){0,3}?(?:short of|below) (?:our |the )?(?:expectations|consensus|"
        r"estimates|outlook|guidance|what we expected)\b",
        rf"\bmiss(?:ed|es)?\b{_gap(2)}(?:consensus|expectations|estimates|guidance)\b",
        r"\bshortfall\b|\bbelow the (?:low end|midpoint) of (?:our|the) (?:prior )?(?:outlook|guidance|range)\b|"
        r"\bbelow the (?:outlook|guidance|range) we (?:provided|gave|issued)\b",
    ]),
)

TAG_POLARITY: dict[str, int] = {}
for _r in _RULES:
    TAG_POLARITY.setdefault(_r.tag, _r.polarity)
    if _r.on_negated == REFUTE:
        TAG_POLARITY.setdefault(f"no_{_r.tag}", +1)
    elif isinstance(_r.on_negated, tuple):
        TAG_POLARITY.setdefault(*_r.on_negated)

# --------------------------------------------------------------------------------------------
# Negation
# --------------------------------------------------------------------------------------------

_NEG_CUE = re.compile(
    r"\b(?:not|no|never|nor|neither|without|nothing|none|cannot|absence of|lack of|free of|rather than|"
    r"instead of|ruled? out|den(?:y|ies|ied))\b|n't\b", _FLAGS)
_PSEUDO_NEG = re.compile(
    r"\b(?:not only|not just|no doubt|without a doubt|not to mention|nothing but|no less than|not least|"
    r"no matter|not necessarily)\b|"
    # "we did not anticipate the pricing pressure": the concern happened, it was just a surprise.
    r"\b(?:did|had)(?:\s+not|n't)\s+(?:fully\s+)?(?:anticipat|foresee|foresaw|predict|expect)\w*", _FLAGS)
_SCOPE_BREAK = re.compile(
    r"[;:!?()—–]|\s-\s|,\s+(?:and|but|so|which|while|as|because|since|whereas|although|though|yet)\b|"
    r"\b(?:but|however|although|though|whereas|yet|except|nevertheless|nonetheless)\b", _FLAGS)
_NEG_BREAKER = re.compile(
    r"\b(?:offset\w*|overc[oa]me|mitigat\w*|counter\w*|escap\w*|immune|insulated|enough to|sufficient(?:ly)? to)\b",
    _FLAGS)
_POST_NEG = re.compile(
    r"^[^;:!?]{0,40}?\b(?:was|were|is|are|has been|have been|did|does|do|has|have)(?:\s+not|n't|\s+never)\b\s*"
    r"(?:been\s+)?(?:a\s+|an\s+|any\s+)?(?:big\s+|real\s+|major\s+|meaningful\s+|significant\s+)?"
    r"(?:factor|issue|problem|concern|material|meaningful|significant|something|the case|materiali[sz]\w*|evident|"
    r"visible|seen|observed|present|happening|occurring|driver|driving|affect\w*|impact\w*)\b", _FLAGS)
_WINDOW_WORDS = 15


def _has_cue(text: str) -> bool:
    """True if ``text`` holds a negation cue not followed by a scope breaker."""
    text = _PSEUDO_NEG.sub(" ", text)
    last = None
    for m in _NEG_CUE.finditer(text):
        last = m
    return last is not None and not _NEG_BREAKER.search(text, last.end())


def is_negated(sentence: str, start: int, end: int) -> bool:
    """Whether the phrase ``sentence[start:end]`` is negated (see module docstring)."""
    clause_start = 0
    for m in _SCOPE_BREAK.finditer(sentence, 0, start):
        clause_start = m.end()
    words = sentence[clause_start:start].split()
    if _has_cue(" ".join(words[-_WINDOW_WORDS:])):
        return True
    if _has_cue(sentence[start:end]):
        return True
    m = _SCOPE_BREAK.search(sentence, end)
    after = sentence[end : m.start() if m else len(sentence)]
    return bool(_POST_NEG.search(after)) and not _NEG_BREAKER.search(after)


def _dampened(rule: _Rule, sentence: str, start: int, end: int) -> bool:
    if rule.dampener is None:
        return False
    return bool(rule.dampener.search(sentence[max(0, start - 30) : start]) or rule.dampener.search(sentence[end : end + 40]))


# --------------------------------------------------------------------------------------------
# Extraction
# --------------------------------------------------------------------------------------------


def _tag_sentence(sentence: str) -> list[tuple[str, int]]:
    hits: dict[str, int] = {}
    negated: dict[str, int] = {}
    suppressed: set[str] = set()
    for rule in _RULES:
        if rule.requires is not None and not rule.requires.search(sentence):
            continue
        fired = False
        for rx, negatable in rule.patterns:
            for m in rx.finditer(sentence):
                if negatable and (is_negated(sentence, m.start(), m.end()) or _dampened(rule, sentence, m.start(), m.end())):
                    if rule.on_negated == REFUTE:
                        negated.setdefault(f"no_{rule.tag}", +1)
                    elif isinstance(rule.on_negated, tuple):
                        negated.setdefault(*rule.on_negated)
                    continue
                fired = True
                break
            if fired:
                break
        if fired:
            hits.setdefault(rule.tag, rule.polarity)
            suppressed.update(rule.suppresses)
    for tag, pol in negated.items():
        positive = tag[3:] if tag.startswith("no_") else None
        if positive is not None and positive in hits:
            continue  # a non-negated match of the same concern wins
        hits.setdefault(tag, pol)
    return [(t, p) for t, p in hits.items() if t not in suppressed]


def _is_skipped_turn(seg: TranscriptSegment) -> bool:
    role = (seg.role or "").casefold()
    return "analyst" in role or role in {"operator", "moderator"} or (seg.speaker or "").strip().casefold() == "operator"


def _sentences(text: str) -> Iterable[str]:
    for ps, pe in paragraph_spans(text):
        for s, e in sentence_spans(text, ps, pe):
            yield text[s:e]


def extract_signals(doc: Document) -> list[NarrativeSignal]:
    """Tag every management / news sentence of ``doc``; signals are in document order."""
    out: list[NarrativeSignal] = []
    if doc.kind == DocumentKind.TRANSCRIPT and doc.segments:
        sources = [(seg.text, seg.speaker or None) for seg in doc.segments if not _is_skipped_turn(seg)]
    else:
        sources = [(doc.text, None)]
    for text, speaker in sources:
        for sentence in _sentences(text):
            if sentence.rstrip("\"'”’) ").endswith("?"):
                continue
            for tag, pol in _tag_sentence(sentence):
                out.append(NarrativeSignal(tag=tag, polarity=pol, sentence=sentence, doc_id=doc.doc_id, speaker=speaker))
    return out


def extract_signals_from_documents(documents: Iterable[Document]) -> list[NarrativeSignal]:
    """``extract_signals`` over several documents, concatenated in input order."""
    return [s for d in documents for s in extract_signals(d)]


def summarize_signals(signals: Iterable[NarrativeSignal]) -> dict[str, int]:
    """Tag -> count, most frequent first (ties alphabetical)."""
    counts = Counter(s.tag for s in signals)
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def net_polarity(signals: Iterable[NarrativeSignal]) -> float:
    """(supportive - bearish) / (supportive + bearish) in [-1, 1]; 0.0 when no polar signals."""
    pos = neg = 0
    for s in signals:
        if s.polarity > 0:
            pos += 1
        elif s.polarity < 0:
            neg += 1
    return (pos - neg) / (pos + neg) if pos + neg else 0.0
