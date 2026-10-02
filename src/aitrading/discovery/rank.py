"""Ranking and novelty of idea candidates.

Score (``score_candidate``) - a weighted sum of five components, each in [0, 1]:

==============  ======  =====================================================================
component       weight  definition
==============  ======  =====================================================================
testability     0.35    testable_now 1.0, partially_testable 0.6, needs_institutional_data 0.3,
                        not_testable 0.0 - the desk can only act on what it can backtest.
evidence        0.20    fraction of evidence quotes verified verbatim against the source.
credibility     0.20    mean of three signals: reported |t-stat| >= 3 (1.0; >= 2 -> 0.5),
                        sample >= 20 years (1.0; >= 10 -> 0.5), peer-reviewed publication (1.0,
                        see ``is_peer_reviewed``).
novelty         0.15    new 1.0, variant_of_library 0.6, duplicate 0.0.
recency         0.10    published within 3 years 1.0, within 10 years 0.5, older 0.0,
                        unknown date 0.25.
==============  ======  =====================================================================

A document that is not a trading idea scores 0. A candidate carrying a security flag (a note that
starts with ``SECURITY_NOTE_PREFIX``: the source contains instructions addressed to an AI, or the
extractor's output repeated them) is capped at ``SECURITY_SCORE_CAP`` and must not be translated or
backtested without the trader explicitly confirming it (``is_security_flagged``). Scores are rounded
to 4 decimals.

Peer review (``is_peer_reviewed``) is decided from source metadata first (a journal-article URL or
DOI, or a source name naming a journal), and otherwise only from an *affirmative* statement in the
credibility notes ("Published in the Journal of Finance") in a clause with no negation or
uncertainty ("not yet published in ...", "unclear whether ...", "working paper", "preprint").
Notes of a security-flagged candidate are not trusted for this.

Novelty (``assess_novelty``): ``duplicate`` when an existing inbox item has the same canonical URL
(arXiv abs/pdf/version links collapse to one) or a source title at least 90% similar; otherwise
``variant_of_library`` when the extractor named a closest library template or the idea title is
at least 60% similar to a template title / alias / key; otherwise ``new``.
"""

from __future__ import annotations

import difflib
import re
from datetime import date, datetime, timezone
from typing import Any, Iterable, Literal, Mapping
from urllib.parse import urlsplit

from aitrading.discovery.models import IdeaCandidate

__all__ = [
    "DUPLICATE_TITLE_SIMILARITY",
    "JOURNAL_VENUE_PATTERN",
    "NOVELTY_SCORE",
    "SECURITY_NOTE_PREFIX",
    "SECURITY_SCORE_CAP",
    "TESTABILITY_SCORE",
    "VARIANT_TITLE_SIMILARITY",
    "WEIGHTS",
    "affirms_peer_review",
    "assess_novelty",
    "canonical_url",
    "is_peer_reviewed",
    "is_security_flagged",
    "journal_venue_from_url",
    "rank_candidates",
    "sample_years",
    "score_breakdown",
    "score_candidate",
    "security_flags",
    "title_similarity",
    "titles_match",
]

Novelty = Literal["new", "variant_of_library", "duplicate"]

WEIGHTS: dict[str, float] = {"testability": 0.35, "evidence": 0.20, "credibility": 0.20, "novelty": 0.15, "recency": 0.10}
TESTABILITY_SCORE: dict[str, float] = {
    "testable_now": 1.0,
    "partially_testable": 0.6,
    "needs_institutional_data": 0.3,
    "not_testable": 0.0,
}
NOVELTY_SCORE: dict[str, float] = {"new": 1.0, "variant_of_library": 0.6, "duplicate": 0.0}
DUPLICATE_TITLE_SIMILARITY = 0.9
VARIANT_TITLE_SIMILARITY = 0.6
RECENT_YEARS = 3
UNKNOWN_DATE_RECENCY = 0.25

SECURITY_NOTE_PREFIX = "SECURITY: "
SECURITY_SCORE_CAP = 0.25

try:  # one canonicaliser for the whole discovery package (arXiv abs/pdf/version links collapse)
    from aitrading.discovery.textutil import canonical_url as _canonical_url_impl
except Exception:  # pragma: no cover - fallback if textutil is unavailable

    def _canonical_url_impl(url: str) -> str:
        u = (url or "").strip().lower()
        u = re.sub(r"^https?://", "", u)
        u = re.sub(r"^www\.", "", u)
        u = u.split("#", 1)[0]
        return u.rstrip("/")


def canonical_url(url: str) -> str:
    """Canonical form of a URL for de-duplication; never raises on a malformed (untrusted) URL."""
    try:
        return _canonical_url_impl(url)
    except Exception:  # e.g. ValueError for an out-of-range port in a search-result URL
        return (url or "").strip().lower()


# ---------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------


def _norm_title(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (s or "").casefold()).strip()


def title_similarity(a: str, b: str) -> float:
    """Similarity of two titles in [0, 1]: max(character sequence ratio, token Jaccard), case/punctuation-insensitive."""
    na, nb = _norm_title(a), _norm_title(b)
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0
    seq = difflib.SequenceMatcher(None, na, nb).ratio()
    ta, tb = set(na.split()), set(nb.split())
    jac = len(ta & tb) / len(ta | tb) if ta | tb else 0.0
    return max(seq, jac)


def titles_match(a: str, b: str, threshold: float) -> bool:
    """``title_similarity(a, b) >= threshold``, with cheap upper bounds first (used in O(n*m) scans)."""
    na, nb = _norm_title(a), _norm_title(b)
    if not na or not nb:
        return False
    if na == nb:
        return True
    ta, tb = set(na.split()), set(nb.split())
    if ta | tb and len(ta & tb) / len(ta | tb) >= threshold:
        return True
    sm = difflib.SequenceMatcher(None, na, nb)
    return sm.real_quick_ratio() >= threshold and sm.quick_ratio() >= threshold and sm.ratio() >= threshold


JOURNAL_VENUE_PATTERN = (
    r"(?:journal of (?:finance|financial economics|accounting research|accounting and economics|portfolio management|"
    r"financial and quantitative analysis|empirical finance|banking and finance|financial markets|political economy)\b|"
    r"review of financial studies\b|review of (?:accounting studies|finance|asset pricing studies|corporate finance studies)\b|"
    r"accounting review\b|management science\b|financial analysts journal\b|quarterly journal of economics\b|"
    r"econometrica\b|american economic review\b|financial management\b|journal of business\b)"
)
_VENUE_RE = re.compile(JOURNAL_VENUE_PATTERN, re.I)

# An affirmative statement that the work itself is published in a refereed venue.
_AFFIRM_PEER = re.compile(
    r"\b(?:published|forthcoming|accepted(?:\s+for\s+publication)?|appear(?:s|ed)?|to appear)\s+in\s+(?:the\s+|a\s+|an\s+)?"
    r"(?:" + JOURNAL_VENUE_PATTERN + r"|peer[- ]reviewed\b|refereed\b|(?:academic\s+|scholarly\s+)?(?:journal|review)\b)"
    r"|\bpeer[- ]reviewed\s+(?:journal\s+)?(?:article|paper|publication|study)\b"
    r"|^\W*(?:peer[- ]reviewed|refereed)\b"
    r"|\(\s*(?:peer[- ]reviewed|refereed)\s*\)",
    re.I,
)
# Any of these in the same clause turns a peer-review mention into "not / not known to be" peer-reviewed.
_NEG_OR_UNSURE = re.compile(
    r"\b(?:not|no(?!\.?\s*\d)|never|non|nor|neither|cannot|can't|isn't|wasn't|hasn't|haven't|doesn't|unrefereed|unpublished|"
    r"unreviewed|unclear|uncertain|unknown|unverified|unconfirmed|yet to be|whether|(?-i:if|may)|might|maybe|possibly|perhaps|apparently|"
    r"reportedly|supposedly|allegedly|purportedly|claims?|claimed|claiming|submitted|under review|pending|preprints?|"
    r"working papers?|drafts?|blogs?|cites?|cited|citing|references?|referenced|mentions?|mentioned)\b",
    re.I,
)
_CLAUSE_SPLIT = re.compile(r"[.;!]\s+|[.;!]$|\n|\b(?:but|however|although|though|whereas|while)\b", re.I)

# Journal-article landing pages (host -> required path prefix) and journal DOI prefixes.
_VENUE_HOSTS: tuple[tuple[str, str], ...] = (
    ("onlinelibrary.wiley.com", "/doi/"),
    ("sciencedirect.com", "/science/article/"),
    ("academic.oup.com", "/"),
    ("pubsonline.informs.org", "/doi/"),
    ("journals.uchicago.edu", "/doi/"),
    ("cambridge.org", "/core/journals/"),
    ("tandfonline.com", "/doi/"),
    ("link.springer.com", "/article/"),
    ("publications.aaahq.org", "/"),
    ("jstor.org", "/stable/"),
    ("aeaweb.org", "/articles"),
    ("pm-research.com", "/content/"),
)
_JOURNAL_DOI_PREFIXES: tuple[str, ...] = (
    "10.1111/jofi", "10.1016/j.jfineco", "10.1093/rfs", "10.1017/s0022109", "10.1287/mnsc", "10.2308/accr",
    "10.2308/tar", "10.1007/s11142", "10.1080/0015198x", "10.2469/faj", "10.3905/jpm", "10.1086/", "10.1093/qje",
    "10.1093/rapstu", "10.1093/rcfs", "10.3982/ecta", "10.1016/j.jacceco", "10.1111/1475-679x", "10.1016/j.finmar",
    "10.1016/j.jempfin", "10.1016/j.jbankfin", "10.1257/aer", "10.1111/fima",
)


def journal_venue_from_url(url: str) -> str | None:
    """The host (or DOI) when ``url`` is a journal-article page / journal DOI, else None (metadata, not page text)."""
    try:
        parts = urlsplit((url or "").strip())
        host = (parts.hostname or "").lower()
    except ValueError:
        return None
    host = host[4:] if host.startswith("www.") else host
    path = parts.path or "/"
    if host in ("doi.org", "dx.doi.org"):
        doi = path.lstrip("/").lower()
        return f"doi:{doi}" if doi.startswith(_JOURNAL_DOI_PREFIXES) else None
    for venue_host, prefix in _VENUE_HOSTS:
        if (host == venue_host or host.endswith("." + venue_host)) and path.startswith(prefix):
            return host
    return None


def affirms_peer_review(text: str) -> bool:
    """True when some clause of ``text`` affirmatively states peer-reviewed publication.

    Clauses containing a negation or an uncertainty / preprint cue ("not yet published in a
    peer-reviewed journal", "unclear whether it was peer-reviewed", "working paper") never count.
    """
    for clause in _CLAUSE_SPLIT.split(text or ""):
        if clause and _AFFIRM_PEER.search(clause.strip()) and not _NEG_OR_UNSURE.search(clause):
            return True
    return False


def security_flags(c: IdeaCandidate) -> list[str]:
    """Security notes on a candidate (instructions to an AI in the source, or echoed by the extractor)."""
    return [n for n in c.notes if n.startswith(SECURITY_NOTE_PREFIX)]


def is_security_flagged(c: IdeaCandidate) -> bool:
    """True when the candidate must not be auto-translated / backtested without the trader confirming it."""
    return bool(security_flags(c))


def is_peer_reviewed(c: IdeaCandidate) -> bool:
    """Peer-reviewed publication, judged from metadata first and only then from affirmative notes.

    Metadata: the source URL is a journal-article page or journal DOI, or the source name (set by the
    platform or the trader's feed config, not by the document) names a journal. Otherwise an
    affirmative, un-negated clause in the credibility notes (see ``affirms_peer_review``); the notes of
    a security-flagged candidate are ignored because the document may have steered them.
    """
    if journal_venue_from_url(c.source.url):
        return True
    name = c.source.source_name or ""
    if _VENUE_RE.search(name) and not _NEG_OR_UNSURE.search(name):
        return True
    if is_security_flagged(c):
        return False
    return any(affirms_peer_review(n) for n in c.extraction.credibility_notes)


def sample_years(period: str | None, today: date | None = None) -> int | None:
    """Length in years of a sample period string such as '1963-2019', 'Jan 1990 to Dec 2020', '1990-present'."""
    if not period:
        return None
    years = [int(y) for y in re.findall(r"(?<!\d)((?:1[89]|20)\d{2})(?!\d)", period)]
    if len(years) >= 2:
        return max(years) - min(years)
    if len(years) == 1 and re.search(r"present|today|now|current", period, re.I):
        return (today or _today()).year - years[0]
    return None


def _today() -> date:
    return datetime.now(timezone.utc).date()


def _tattr(tpl: Any, name: str, default: Any = None) -> Any:
    if isinstance(tpl, Mapping):
        return tpl.get(name, default)
    return getattr(tpl, name, default)


# ---------------------------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------------------------


def score_breakdown(c: IdeaCandidate, *, today: date | None = None) -> dict[str, float]:
    """Per-component scores in [0, 1] (keys = ``WEIGHTS``) plus ``total``."""
    today = today or _today()
    ext = c.extraction
    if not ext.is_trading_idea:
        return {**{k: 0.0 for k in WEIGHTS}, "total": 0.0}

    testability = TESTABILITY_SCORE.get(ext.testability, 0.0)
    evidence = (sum(ch.status == "verified" for ch in c.quote_checks) / len(c.quote_checks)) if c.quote_checks else 0.0

    t = abs(ext.reported_t_stat) if ext.reported_t_stat is not None else None
    t_score = 1.0 if t is not None and t >= 3.0 else 0.5 if t is not None and t >= 2.0 else 0.0
    yrs = sample_years(ext.sample_period, today)
    s_score = 1.0 if yrs is not None and yrs >= 20 else 0.5 if yrs is not None and yrs >= 10 else 0.0
    p_score = 1.0 if is_peer_reviewed(c) else 0.0
    credibility = (t_score + s_score + p_score) / 3.0

    novelty = NOVELTY_SCORE.get(c.novelty, 0.0)

    pub = c.source.published
    if pub is None:
        recency = UNKNOWN_DATE_RECENCY
    else:
        age_days = (today - pub).days
        recency = 1.0 if age_days <= RECENT_YEARS * 365.25 else 0.5 if age_days <= 10 * 365.25 else 0.0

    parts = {"testability": testability, "evidence": evidence, "credibility": credibility, "novelty": novelty, "recency": recency}
    total = sum(WEIGHTS[k] * v for k, v in parts.items())
    if is_security_flagged(c):
        total = min(total, SECURITY_SCORE_CAP)
    parts["total"] = round(min(1.0, max(0.0, total)), 4)
    return parts


def score_candidate(c: IdeaCandidate, *, today: date | None = None) -> float:
    """Ranking score in [0, 1] (see the module docstring for the weights)."""
    return score_breakdown(c, today=today)["total"]


# ---------------------------------------------------------------------------------------------
# novelty
# ---------------------------------------------------------------------------------------------


def assess_novelty(c: IdeaCandidate, existing: Iterable[IdeaCandidate], templates: Mapping[str, Any] | None = None) -> Novelty:
    """'duplicate' | 'variant_of_library' | 'new' for ``c`` against inbox items and idea templates."""
    url = canonical_url(c.source.url)
    for e in existing:
        if e is c:
            continue
        if e.idea_id == c.idea_id:
            return "duplicate"
        if url and url == canonical_url(e.source.url):
            return "duplicate"
        if titles_match(c.source.title, e.source.title, DUPLICATE_TITLE_SIMILARITY):
            return "duplicate"

    if c.extraction.closest_library_template:
        return "variant_of_library"
    idea_title = c.extraction.title
    for key, tpl in (templates or {}).items():
        names = [str(key).replace("_", " "), str(_tattr(tpl, "title") or "")]
        aliases = _tattr(tpl, "aliases") or ()
        names.extend([aliases] if isinstance(aliases, str) else [str(a) for a in aliases])
        if any(titles_match(idea_title, n, VARIANT_TITLE_SIMILARITY) for n in names if n):
            return "variant_of_library"
    return "new"


# ---------------------------------------------------------------------------------------------
# ranking
# ---------------------------------------------------------------------------------------------


def rank_candidates(cands: Iterable[IdeaCandidate], *, rescore: bool = False, today: date | None = None) -> list[IdeaCandidate]:
    """Sort by score (desc), then publication date (newest first; undated last), then idea_id.

    With ``rescore=True`` each candidate's ``score`` is recomputed first (returned as copies).
    """
    items = list(cands)
    if rescore:
        items = [c.model_copy(update={"score": score_candidate(c, today=today)}) for c in items]
    return sorted(items, key=lambda c: (-c.score, -(c.source.published or date.min).toordinal(), c.idea_id))
