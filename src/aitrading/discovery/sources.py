"""Idea sources: the arXiv q-fin API, RSS / Atom feeds, a single URL, a local PDF.

Every source returns ``list[SourceDocument]`` (``aitrading.discovery.models``), newest first and
de-duplicated by ``SourceDocument.doc_key``. Multi-item sources (``ArxivSource``, ``FeedSource``)
never raise for a failing request or a malformed item: the problem is recorded in ``.warnings``
(reset at the start of every public call) and the remaining items are still returned. Only invalid
arguments raise. Single-document helpers (``fetch_url``, ``load_pdf``) raise ``SourceError`` /
``FileNotFoundError`` / ``ImportError`` with an actionable message, because there is nothing
partial to return.

arXiv
-----
``ArxivSource`` uses the official arXiv API (``https://export.arxiv.org/api/query``), sorted by
submission date, newest first. arXiv asks API clients to make at most one request every three
seconds and to identify themselves; the source spaces its requests ``min_interval_s`` (default 3 s)
apart, sends a descriptive User-Agent (set ``AITRADING_CONTACT=you@example.com`` to include a
contact address) and retries 429 / 5xx responses with back-off. Reuse one ``ArxivSource`` per run so
the spacing applies across queries. A query is either free text (every term must appear in the title
or abstract) or raw arXiv syntax when it contains a field prefix, e.g.
``'ti:momentum ANDNOT abs:crypto'``.

Feeds - adding your own (e.g. your favourite quant blogs)
--------------------------------------------------------
Feeds are user-configurable. ``load_feed_config()`` reads, in order of precedence:

1. the ``path`` argument,
2. the ``AITRADING_SOURCES`` environment variable (a path to a JSON file, or inline JSON),
3. ``~/.aitrading/sources.json``,

and falls back to ``DEFAULT_FEEDS`` when none exists. Example ``~/.aitrading/sources.json``::

    {
      "feeds": [
        {"name": "My quant blog", "url": "https://blog.example.com/feed.xml", "kind": "auto"},
        {"name": "Another research site", "url": "https://research.example.org/atom.xml", "kind": "atom"}
      ],
      "arxiv_categories": ["q-fin.PM", "q-fin.TR", "q-fin.ST"],
      "arxiv_queries": ["momentum", "earnings announcement drift"],
      "web_allowed_domains": ["ssrn.com", "arxiv.org", "nber.org"]
    }

Most blogs publish their feed at a URL shown by the browser as "RSS" / "Atom" or linked from the
page head (``<link rel="alternate" type="application/rss+xml" href=...>``). ``kind`` may be
``"rss"`` (RSS 2.0 or RSS 1.0/RDF), ``"atom"`` or ``"auto"`` (detected from the document). Only the
``feeds`` key is used by ``FeedSource``; the other keys are optional settings for the arXiv and web
search sources (see ``SourcesConfig``).

Untrusted input: feed items and web pages are only parsed into plain text fields; nothing in them is
executed or followed, and XML with entity declarations is rejected (in any encoding).

Network safety: every request (including each redirect hop, which is followed manually) is refused
when its host is a local name or resolves to a loopback, private, link-local (e.g. the cloud metadata
address 169.254.169.254), multicast or otherwise non-public address, so a link found in a feed, a
paper or by the model cannot make the trader's PC read intranet pages and pass them on. Pass
``allow_private=True`` (``fetch_url``, ``FeedSource``) or set ``"allow_private": true`` on a feed in
sources.json to read an intranet page on purpose. Downloads are streamed with a size cap
(``MAX_FETCH_BYTES`` for pages, ``MAX_FEED_BYTES`` for feeds, ``MAX_API_BYTES`` for arXiv API pages)
and an overall time limit per request, so a huge, endless or slow-drip response cannot exhaust
memory or hang a discovery run.
"""

from __future__ import annotations

import copy
import ipaddress
import json
import os
import re
import socket
import threading
import time
import xml.etree.ElementTree as ET
import xml.parsers.expat as expat
from dataclasses import dataclass
from datetime import date, datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Sequence
from urllib.parse import unquote, urljoin, urlsplit

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from aitrading import __version__
from aitrading.discovery.models import SourceDocument
from aitrading.discovery.textutil import (
    canonical_url,
    decode_text,
    html_to_text,
    normalize_whitespace,
    parse_html,
    pdf_to_text,
    truncate_text,
)

__all__ = [
    "ARXIV_API_URL",
    "DEFAULT_ARXIV_CATEGORIES",
    "DEFAULT_FEEDS",
    "DEFAULT_QUERIES",
    "SOURCES_ENV",
    "ArxivSource",
    "FeedConfig",
    "FeedSource",
    "SourceError",
    "SourcesConfig",
    "build_arxiv_query",
    "default_sources_path",
    "fetch_url",
    "finalize_documents",
    "load_feed_config",
    "load_pdf",
    "load_sources_config",
    "normalize_doc_url",
    "non_public_reason",
    "parse_arxiv_feed",
    "parse_date",
    "parse_feed",
]

ARXIV_API_URL = "https://export.arxiv.org/api/query"

DEFAULT_ARXIV_CATEGORIES: list[str] = ["q-fin.PM", "q-fin.TR", "q-fin.ST", "q-fin.CP", "q-fin.GN"]
DEFAULT_QUERIES: list[str] = [
    "cross-section of stock returns",
    "anomaly",
    "factor",
    "momentum",
    "return predictability",
    "trading strategy",
]

SOURCES_ENV = "AITRADING_SOURCES"
CONTACT_ENV = "AITRADING_CONTACT"

DEFAULT_MAX_TEXT_CHARS = 60_000
MAX_FETCH_BYTES = 25 * 1024 * 1024  # a single page or PDF
MAX_FEED_BYTES = 10 * 1024 * 1024  # an RSS / Atom feed
MAX_API_BYTES = 20 * 1024 * 1024  # one arXiv API page (up to 2000 entries)
MAX_ERROR_BODY_BYTES = 1024 * 1024  # body kept from an error response (for its message)
MAX_PDF_PAGES = 60  # text beyond this is cut by max_chars anyway; bounds pypdf CPU on hostile PDFs
MAX_REDIRECTS = 5
RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
MAX_RETRY_DELAY_S = 60.0

# XML namespaces
ATOM = "http://www.w3.org/2005/Atom"
ARXIV = "http://arxiv.org/schemas/atom"
OPENSEARCH = "http://a9.com/-/spec/opensearch/1.1/"
DC = "http://purl.org/dc/elements/1.1/"
CONTENT = "http://purl.org/rss/1.0/modules/content/"
RSS1 = "http://purl.org/rss/1.0/"
RDF = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"

_CATEGORY_RE = re.compile(r"^[a-z][a-z\-]*(\.[A-Za-z][A-Za-z\-]*)?$")
_ARXIV_FIELD_RE = re.compile(r"\b(ti|au|abs|co|jr|cat|rn|id|all|submittedDate|lastUpdatedDate):", re.I)
_QUERY_STOPWORDS = frozenset(
    {"a", "an", "and", "are", "as", "at", "by", "for", "from", "in", "into", "is", "of", "on", "or", "the", "to", "with"}
)


class SourceError(RuntimeError):
    """A single document could not be fetched or read."""


# =============================================================================================
# Small helpers
# =============================================================================================


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def default_user_agent() -> str:
    contact = os.environ.get(CONTACT_ENV, "").strip()
    who = f"; mailto:{contact}" if contact else ""
    return f"aitrading-idea-scout/{__version__} (personal quant research tool; python-httpx/{httpx.__version__}{who})"


def parse_date(value: str | None) -> date | None:
    """Parse the date formats used by Atom, RSS (RFC 822) and HTML meta tags; None if unparseable."""
    s = (value or "").strip()
    if not s:
        return None
    iso = s[:-1] + "+00:00" if s.endswith(("Z", "z")) else s
    try:
        return datetime.fromisoformat(iso).date()
    except ValueError:
        pass
    m = re.match(r"^(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})(?!\d)", s)
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            return None
    try:
        return parsedate_to_datetime(s).date()
    except (TypeError, ValueError, IndexError, OverflowError):
        pass
    m = re.match(r"^(\d{4})[-/](\d{1,2})$", s)
    if m and 1 <= int(m.group(2)) <= 12:
        return date(int(m.group(1)), int(m.group(2)), 1)
    return None


def _as_since(since: date | datetime | None) -> date | None:
    if since is None:
        return None
    if isinstance(since, datetime):
        return since.date()
    if isinstance(since, date):
        return since
    raise ValueError(f"since must be a date or None, got {type(since).__name__}")


def finalize_documents(docs: Iterable[SourceDocument]) -> list[SourceDocument]:
    """De-duplicate by ``doc_key`` (first occurrence wins) and sort newest first (undated last)."""
    seen: set[str] = set()
    unique: list[SourceDocument] = []
    for d in docs:
        if d.doc_key in seen:
            continue
        seen.add(d.doc_key)
        unique.append(d)
    return sorted(unique, key=lambda d: d.published or date.min, reverse=True)  # stable for ties


def _validate_http_url(url: str) -> str:
    if not isinstance(url, str) or not url.strip():
        raise ValueError("url must be a non-empty string")
    url = url.strip()
    try:
        parts = urlsplit(url)
        parts.port  # noqa: B018 - raises ValueError for a malformed port
    except ValueError as e:
        raise ValueError(f"malformed URL {url!r}: {e}") from e
    if parts.scheme.lower() not in {"http", "https"} or not parts.hostname:
        raise ValueError(f"only http(s) URLs are supported, got {url!r}")
    return url


# Host names that only mean something on the local network.
_LOCAL_NAME_SUFFIXES = (".localhost", ".local", ".internal", ".home.arpa", ".lan", ".intranet")


def _resolve_host(host: str, port: int) -> list[str]:
    """IP addresses ``host`` resolves to here; [] when it cannot be resolved (e.g. only a proxy can)."""
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except (OSError, UnicodeError, ValueError):
        return []
    return [str(info[4][0]) for info in infos]


def _is_non_public_ip(addr: str) -> bool:
    try:
        ip = ipaddress.ip_address(addr.split("%", 1)[0])
    except ValueError:
        return False
    if isinstance(ip, ipaddress.IPv6Address):
        if ip.ipv4_mapped is not None:
            ip = ip.ipv4_mapped
        elif ip.sixtofour is not None and _is_non_public_ip(str(ip.sixtofour)):
            return True
    # is_global is False for loopback, private (RFC 1918, ULA), link-local (169.254/16, fe80::/10),
    # shared (100.64/10), reserved, documentation and unspecified addresses.
    return not ip.is_global or ip.is_multicast


def non_public_reason(url: str) -> str | None:
    """Why ``url`` points at the local machine or network (and is refused by default), or None.

    Refused: ``localhost`` and local-only names (``*.local``, ``*.internal``, ...) and hosts that are,
    or resolve to, loopback / private / link-local / multicast / reserved / unspecified addresses
    (IPv4 and IPv6, including IPv4-mapped IPv6). A host that cannot be resolved locally is allowed
    (behind a proxy only the proxy can resolve it); the request then fails or goes through the proxy.
    The name is resolved just before the request; the connection does not pin that address, so a
    DNS-rebinding server could still answer differently a moment later (defence in depth, not a sandbox).
    """
    try:
        parts = urlsplit(url)
        host, port = parts.hostname, parts.port
    except ValueError:
        return "the URL is malformed"
    if not host:
        return "the URL has no host"
    host = host.rstrip(".")
    if host == "localhost" or host.endswith(_LOCAL_NAME_SUFFIXES):
        return f"{host} is a local network name"
    try:
        ipaddress.ip_address(host.split("%", 1)[0])
        literal = True
    except ValueError:
        literal = False
    addrs = [host] if literal else _resolve_host(host, port or (443 if parts.scheme.lower() == "https" else 80))
    for addr in addrs:
        if _is_non_public_ip(addr):
            return f"{host} is a non-public address" if literal else f"{host} resolves to the non-public address {addr}"
    return None


def normalize_doc_url(url: str) -> str:
    """URL stored on a SourceDocument: arXiv links (abs/pdf/html, any version) become the versionless
    abs page so the same paper gets the same ``doc_key`` from every source; others are unchanged."""
    url = (url or "").strip()
    cu = canonical_url(url)
    return cu if cu.startswith("https://arxiv.org/abs/") else url


def _host(url: str) -> str:
    host = (urlsplit(url).hostname or "").lower()
    return host[4:] if host.startswith("www.") else host


class _PrologDone(Exception):
    pass


def _refuse_entities(*_args: Any) -> None:
    raise ValueError("XML document declares entities; refused")


def _check_xml_prolog(content: bytes) -> None:
    """Refuse entity declarations, whatever the document's encoding (UTF-8, UTF-16, Latin-1, ...).

    Entities can only be declared in the DOCTYPE before the root element, so the same expat engine
    ElementTree uses scans just the prolog: it decodes the bytes exactly like the real parse will
    (a raw byte search for ``<!ENTITY`` misses UTF-16 documents) and stops at the first element.
    """
    parser = expat.ParserCreate()
    parser.EntityDeclHandler = _refuse_entities
    parser.UnparsedEntityDeclHandler = _refuse_entities

    def _stop(*_args: Any) -> None:
        raise _PrologDone

    parser.StartElementHandler = _stop
    try:
        parser.Parse(content, True)
    except _PrologDone:
        return
    except expat.ExpatError as e:
        raise ValueError(f"malformed XML: {e}") from e


def _parse_xml(content: bytes) -> ET.Element:
    """Parse untrusted XML. Documents declaring entities are refused (entity-expansion attacks)."""
    _check_xml_prolog(content)
    try:
        return ET.fromstring(content)
    except ET.ParseError as e:
        raise ValueError(f"malformed XML: {e}") from e


def _text(el: ET.Element | None) -> str:
    return normalize_whitespace("".join(el.itertext())) if el is not None else ""


def _mb(n: int) -> str:
    return f"{n / (1024 * 1024):.1f} MB"


@dataclass
class _Fetched:
    """A fully read, size-capped HTTP response (the body is decompressed)."""

    status_code: int
    headers: httpx.Headers
    content: bytes
    url: str

    @property
    def content_type(self) -> str:
        return self.headers.get("content-type", "").split(";")[0].strip().lower()

    @property
    def charset(self) -> str | None:
        m = re.search(r"""charset\s*=\s*["']?([^"';\s]+)""", self.headers.get("content-type", ""), re.I)
        return m.group(1) if m else None


class _Http:
    """GET with polite request spacing, retries on 429/5xx and transport errors, redirects followed
    manually (each hop re-checked by ``non_public_reason``), streamed bodies with a byte cap and an
    overall per-request time limit."""

    def __init__(
        self,
        client: httpx.Client | None,
        *,
        user_agent: str | None,
        timeout_s: float,
        max_retries: int,
        min_interval_s: float,
        clock: Callable[[], float],
        sleep: Callable[[float], None],
        max_bytes: int = MAX_FETCH_BYTES,
        max_total_s: float | None = None,
        allow_private: bool = False,
    ) -> None:
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        if min_interval_s < 0:
            raise ValueError("min_interval_s must be >= 0")
        if max_bytes < 1:
            raise ValueError("max_bytes must be >= 1")
        self._client = client
        self._owns_client = client is None
        self.user_agent = user_agent or default_user_agent()
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.min_interval_s = min_interval_s
        self.max_bytes = max_bytes
        # httpx timeouts apply per network operation; this bounds a whole download (slow-drip servers).
        self.max_total_s = float(max_total_s) if max_total_s is not None else max(60.0, 4.0 * timeout_s)
        self.allow_private = allow_private
        self._clock = clock
        self._sleep = sleep
        self._last: float | None = None
        self._lock = threading.Lock()
        self.n_requests = 0

    @property
    def client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(timeout=self.timeout_s, follow_redirects=False)
        return self._client

    def close(self) -> None:
        if self._owns_client and self._client is not None:
            self._client.close()
            self._client = None

    def _wait_turn(self) -> None:
        with self._lock:
            now = self._clock()
            if self._last is not None and self.min_interval_s > 0:
                wait = self._last + self.min_interval_s - now
                if wait > 0:
                    self._sleep(wait)
                    now += wait
            self._last = now

    def _retry_delay(self, attempt: int, resp: _Fetched | None) -> float:
        if resp is not None:
            ra = resp.headers.get("retry-after", "").strip()
            if ra.isdigit():
                return min(float(ra), MAX_RETRY_DELAY_S)
        return min(max(self.min_interval_s, 1.0) * (2**attempt), MAX_RETRY_DELAY_S)

    def get(
        self,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        accept: str | None = None,
        max_bytes: int | None = None,
        allow_private: bool | None = None,
    ) -> _Fetched:
        """Return the final response (possibly a non-2xx one).

        Raises SourceError on network failure, a refused (non-public) target, too many redirects, a
        body larger than ``max_bytes`` or a download slower than ``max_total_s``.
        """
        cap = max_bytes or self.max_bytes
        allow = self.allow_private if allow_private is None else allow_private
        headers = {"User-Agent": self.user_agent}
        if accept:
            headers["Accept"] = accept
        current, current_params = url, params
        for _hop in range(MAX_REDIRECTS + 1):
            if not allow:
                reason = non_public_reason(current)
                if reason:
                    via = "" if current == url else f" (redirected from {url})"
                    raise SourceError(
                        f"refusing to fetch {current}{via}: {reason}; pass allow_private=True to read a page on "
                        "your own network on purpose"
                    )
            resp = self._get_with_retries(current, current_params, headers, cap)
            location = resp.headers.get("location", "").strip()
            if resp.status_code not in REDIRECT_STATUSES or not location:
                return resp
            target = urljoin(resp.url, location)
            try:
                _validate_http_url(target)
            except ValueError as e:
                raise SourceError(f"{url} redirected to an unsupported URL: {e}") from e
            current, current_params = target, None
        raise SourceError(f"too many redirects fetching {url} (more than {MAX_REDIRECTS})")

    def _get_with_retries(self, url: str, params: dict[str, Any] | None, headers: dict[str, str], cap: int) -> _Fetched:
        last_exc: Exception | None = None
        resp: _Fetched | None = None
        for attempt in range(self.max_retries + 1):
            self._wait_turn()
            self.n_requests += 1
            try:
                resp = self._get_once(url, params, headers, cap)
                last_exc = None
            except httpx.TransportError as e:  # connect / read errors and timeouts
                resp, last_exc = None, e
            else:
                if resp.status_code not in RETRY_STATUSES:
                    return resp
            if attempt < self.max_retries:
                self._sleep(self._retry_delay(attempt, resp))
        if resp is not None:
            return resp
        raise SourceError(f"network error fetching {url}: {type(last_exc).__name__}: {last_exc}") from last_exc

    def _get_once(self, url: str, params: dict[str, Any] | None, headers: dict[str, str], cap: int) -> _Fetched:
        deadline = self._clock() + self.max_total_s
        try:
            with self.client.stream("GET", url, params=params, headers=headers, follow_redirects=False) as resp:
                status = resp.status_code
                if status in REDIRECT_STATUSES:
                    body = b""  # never needed; the next hop is in the Location header
                elif 200 <= status < 300:
                    declared = resp.headers.get("content-length", "").strip()
                    if declared.isdigit() and int(declared) > cap:
                        raise SourceError(f"{url}: document too large ({_mb(int(declared))}; the limit is {_mb(cap)})")
                    body = self._read_capped(resp, url, cap, deadline, truncate=False)
                else:  # an error page: keep the start of it for the error message
                    body = self._read_capped(resp, url, min(cap, MAX_ERROR_BODY_BYTES), deadline, truncate=True)
                return _Fetched(status, resp.headers, body, str(resp.url))
        except httpx.DecodingError as e:
            raise SourceError(f"{url}: the response body could not be decoded ({e})") from e
        except httpx.InvalidURL as e:
            raise SourceError(f"invalid URL {url!r}: {e}") from e

    def _read_capped(self, resp: httpx.Response, url: str, limit: int, deadline: float, *, truncate: bool) -> bytes:
        chunks: list[bytes] = []
        total = 0
        for chunk in resp.iter_bytes():  # decompressed incrementally: a gzip bomb stops at the cap
            if total + len(chunk) > limit:
                if truncate:
                    chunks.append(chunk[: limit - total])
                    break
                raise SourceError(f"{url}: document too large (more than {_mb(limit)})")
            chunks.append(chunk)
            total += len(chunk)
            if self._clock() > deadline:
                raise SourceError(f"{url}: download did not finish within {self.max_total_s:.0f} s; giving up")
        return b"".join(chunks)


# =============================================================================================
# arXiv
# =============================================================================================


def _query_terms(query: str) -> list[str]:
    terms: list[str] = []
    for m in re.finditer(r'"([^"]+)"|(\S+)', query):
        if m.group(1):  # quoted phrase
            phrase = normalize_whitespace(re.sub(r'["()]', " ", m.group(1)))
            if phrase:
                terms.append(f'"{phrase}"')
            continue
        tok = re.sub(r"[^\w\-.']", "", m.group(2)).strip("-.'")
        if not tok or tok.lower() in _QUERY_STOPWORDS:
            continue
        terms.append(f'"{tok.replace("-", " ")}"' if "-" in tok else tok)
    return terms


def build_arxiv_query(query: str, categories: Sequence[str]) -> str:
    """Build an arXiv API ``search_query``: (cat:A OR cat:B ...) AND (ti:term OR abs:term) AND ...

    Free-text queries require every (non-stopword) term to appear in the title or abstract;
    hyphenated words and "quoted phrases" become phrase searches. A query that already uses arXiv
    field syntax (``ti:``, ``abs:``, ``au:``, ...) is used as-is.
    """
    cats = [f"cat:{c}" for c in categories]
    cat_part = cats[0] if len(cats) == 1 else (f"({' OR '.join(cats)})" if cats else "")
    q = normalize_whitespace(query)
    if not q:
        term_part = ""
    elif _ARXIV_FIELD_RE.search(q):
        term_part = f"({q})"
    else:
        terms = _query_terms(q)
        clauses = [f"(ti:{t} OR abs:{t})" for t in terms]
        term_part = " AND ".join(clauses)
        if len(clauses) > 1 and cat_part:
            term_part = f"({term_part})"
    if cat_part and term_part:
        return f"{cat_part} AND {term_part}"
    if not (cat_part or term_part):
        raise ValueError("an arXiv search needs a query or at least one category")
    return cat_part or term_part


def parse_arxiv_feed(
    content: bytes,
    *,
    warnings: list[str] | None = None,
    fetched_at: datetime | None = None,
) -> tuple[list[SourceDocument], int, int | None]:
    """Parse an arXiv API Atom response.

    Returns ``(documents, n_entries, total_results)`` where ``n_entries`` counts every ``<entry>``
    (including ones skipped as malformed) so callers can page correctly. Raises ValueError if the
    document is not an Atom feed.
    """
    warnings = warnings if warnings is not None else []
    root = _parse_xml(content)
    if root.tag != f"{{{ATOM}}}feed":
        raise ValueError(f"not an Atom feed (root element {root.tag!r})")
    total_txt = root.findtext(f"{{{OPENSEARCH}}}totalResults")
    total = int(total_txt) if total_txt and total_txt.strip().isdigit() else None
    fetched_at = fetched_at or _utcnow()
    docs: list[SourceDocument] = []
    entries = root.findall(f"{{{ATOM}}}entry")
    for entry in entries:
        entry_id = _text(entry.find(f"{{{ATOM}}}id"))
        summary = _text(entry.find(f"{{{ATOM}}}summary"))
        if "/api/errors" in entry_id:  # arXiv reports query errors as a feed entry
            warnings.append(f"arXiv API error: {summary or entry_id}")
            continue
        title = _text(entry.find(f"{{{ATOM}}}title"))
        href = None
        for link in entry.findall(f"{{{ATOM}}}link"):
            if link.get("rel", "alternate") == "alternate" and link.get("href"):
                href = link.get("href")
                break
        url = canonical_url(href or entry_id)
        if not title or not url:
            warnings.append(f"arXiv entry {entry_id or '?'} skipped: missing title or link")
            continue
        published = parse_date(_text(entry.find(f"{{{ATOM}}}published"))) or parse_date(
            _text(entry.find(f"{{{ATOM}}}updated"))
        )
        authors = [a for a in (_text(au.find(f"{{{ATOM}}}name")) for au in entry.findall(f"{{{ATOM}}}author")) if a]
        prim = entry.find(f"{{{ARXIV}}}primary_category")
        primary = prim.get("term") if prim is not None else None
        if not primary:
            cat = entry.find(f"{{{ATOM}}}category")
            primary = cat.get("term") if cat is not None else None
        if not summary:
            warnings.append(f"arXiv entry {url} has no abstract")
        try:
            docs.append(
                SourceDocument(
                    source_type="arxiv",
                    url=url,
                    title=title,
                    authors=authors,
                    published=published,
                    text=summary,
                    fetched_at=fetched_at,
                    source_name=f"arXiv {primary}" if primary else "arXiv",
                )
            )
        except ValidationError as e:  # pragma: no cover - fields are already normalised strings
            warnings.append(f"arXiv entry {url} skipped: {e}")
    return docs, len(entries), total


def _arxiv_error_detail(content: bytes) -> str:
    try:
        root = _parse_xml(content)
    except ValueError:
        return ""
    for entry in root.findall(f"{{{ATOM}}}entry"):
        s = _text(entry.find(f"{{{ATOM}}}summary"))
        if s:
            return f" ({s})"
    return ""


class ArxivSource:
    """Recent papers from arXiv quantitative-finance categories via the official API."""

    source_type = "arxiv"

    def __init__(
        self,
        categories: Sequence[str] = tuple(DEFAULT_ARXIV_CATEGORIES),
        *,
        client: httpx.Client | None = None,
        min_interval_s: float = 3.0,
        max_retries: int = 3,
        timeout_s: float = 30.0,
        page_size: int = 100,
        user_agent: str | None = None,
        api_url: str = ARXIV_API_URL,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if isinstance(categories, str):
            categories = [categories]
        cats = [c.strip() for c in categories]
        bad = [c for c in cats if not _CATEGORY_RE.match(c)]
        if bad:
            raise ValueError(f"invalid arXiv categories: {bad} (expected e.g. 'q-fin.PM')")
        if not 1 <= page_size <= 2000:
            raise ValueError("page_size must be between 1 and 2000 (arXiv API limit)")
        self.categories = cats
        self.page_size = page_size
        self.api_url = api_url
        self.warnings: list[str] = []
        self._http = _Http(
            client,
            user_agent=user_agent,
            timeout_s=timeout_s,
            max_retries=max_retries,
            min_interval_s=min_interval_s,
            clock=clock,
            sleep=sleep,
            max_bytes=MAX_API_BYTES,
        )

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "ArxivSource":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @staticmethod
    def _check(query: str, max_results: int) -> None:
        if not isinstance(query, str):
            raise ValueError("query must be a string")
        if not isinstance(max_results, int) or max_results < 1:
            raise ValueError("max_results must be a positive integer")

    def search(self, query: str, *, max_results: int = 25, since: date | None = None) -> list[SourceDocument]:
        """Newest papers in ``categories`` matching ``query`` (title/abstract), submitted on/after ``since``."""
        self._check(query, max_results)
        since_d = _as_since(since)
        self.warnings = []
        return finalize_documents(self._search(query, max_results, since_d))

    def search_many(
        self,
        queries: Iterable[str] = tuple(DEFAULT_QUERIES),
        *,
        max_results_per_query: int = 25,
        since: date | None = None,
    ) -> list[SourceDocument]:
        """Run several queries (rate-limited) and merge the results."""
        queries = [queries] if isinstance(queries, str) else list(queries)
        for q in queries:
            self._check(q, max_results_per_query)
        since_d = _as_since(since)
        self.warnings = []
        docs: list[SourceDocument] = []
        for q in queries:
            docs.extend(self._search(q, max_results_per_query, since_d))
        return finalize_documents(docs)

    def _search(self, query: str, max_results: int, since: date | None) -> list[SourceDocument]:
        search_query = build_arxiv_query(query, self.categories)
        out: list[SourceDocument] = []
        start = 0
        while len(out) < max_results:
            n = min(self.page_size, max_results - len(out))
            params = {
                "search_query": search_query,
                "start": start,
                "max_results": n,
                "sortBy": "submittedDate",
                "sortOrder": "descending",
            }
            try:
                resp = self._http.get(self.api_url, params=params, accept="application/atom+xml")
            except SourceError as e:
                self.warnings.append(f"arXiv query {query!r}: {e}")
                break
            if resp.status_code != 200:
                self.warnings.append(
                    f"arXiv query {query!r}: HTTP {resp.status_code}{_arxiv_error_detail(resp.content)}"
                )
                break
            try:
                docs, n_entries, total = parse_arxiv_feed(resp.content, warnings=self.warnings)
            except ValueError as e:
                self.warnings.append(f"arXiv query {query!r}: unreadable response: {e}")
                break
            reached_older = False
            for d in docs:
                if since is not None and d.published is not None and d.published < since:
                    reached_older = True  # results are sorted by submission date: the rest is older
                    continue
                if len(out) < max_results:
                    out.append(d)
            start += n_entries
            if reached_older or n_entries < n or (total is not None and start >= total):
                break
        return out


# =============================================================================================
# Feeds (RSS 2.0, RSS 1.0 / RDF, Atom)
# =============================================================================================


class FeedConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str = Field(min_length=1, description="Shown as SourceDocument.source_name.")
    url: str
    kind: Literal["rss", "atom", "auto"] = "auto"
    allow_private: bool = Field(
        False, description="Allow a feed on your own network (localhost / intranet); refused by default."
    )

    @field_validator("url")
    @classmethod
    def _http_url(cls, v: str) -> str:
        return _validate_http_url(v)


class SourcesConfig(BaseModel):
    """Contents of ``sources.json``."""

    model_config = ConfigDict(extra="ignore")

    feeds: list[FeedConfig] = Field(default_factory=list)
    arxiv_categories: list[str] | None = None
    arxiv_queries: list[str] | None = None
    web_allowed_domains: list[str] | None = None
    web_blocked_domains: list[str] | None = None


# arXiv publishes per-category RSS/Atom announcement feeds, but their current URL format could not be
# verified from the build environment, so no feed URL is shipped here: the q-fin categories are
# covered by ``ArxivSource`` (official API) instead. Add any feed - including arXiv's, copied from its
# "RSS feeds" help page - to ~/.aitrading/sources.json (see the module docstring).
DEFAULT_FEEDS: list[FeedConfig] = []


def default_sources_path() -> Path:
    return Path.home() / ".aitrading" / "sources.json"


def _read_config_text(p: Path) -> str:
    # utf-8-sig also accepts a byte-order mark (Windows PowerShell 5.1 "-Encoding UTF8" and older Notepad write one)
    return p.read_text(encoding="utf-8-sig")


def _env_value(name: str) -> str:
    """An environment variable, stripped of whitespace, a BOM and the surrounding quotes that Windows
    ``cmd`` keeps in the value (``set AITRADING_SOURCES="C:\\Users\\me\\sources.json"``)."""
    value = os.environ.get(name, "").strip().lstrip("\ufeff").strip()
    while len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        value = value[1:-1].strip()
    return value


def load_sources_config(path: str | os.PathLike[str] | None = None) -> SourcesConfig:
    """Read sources.json (``path`` > ``$AITRADING_SOURCES`` > ``~/.aitrading/sources.json``).

    Returns the defaults when no file exists at the default location. Raises FileNotFoundError when
    an explicitly given path (argument or env var) does not exist and ValueError for invalid content.
    """
    raw: str | None = None
    origin: str
    if path is not None:
        p = Path(path).expanduser()
        if not p.is_file():
            raise FileNotFoundError(f"sources config not found: {p}")
        raw, origin = _read_config_text(p), str(p)
    elif _env_value(SOURCES_ENV):
        env = _env_value(SOURCES_ENV)
        if env.startswith(("{", "[")):
            raw, origin = env, f"${SOURCES_ENV}"
        else:
            p = Path(env).expanduser()
            if not p.is_file():
                raise FileNotFoundError(f"{SOURCES_ENV} points to {p}, which does not exist")
            raw, origin = _read_config_text(p), str(p)
    else:
        p = default_sources_path()
        if not p.is_file():
            return SourcesConfig(feeds=[f.model_copy() for f in DEFAULT_FEEDS])
        raw, origin = _read_config_text(p), str(p)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"{origin}: invalid JSON ({e})") from e
    if isinstance(data, list):
        data = {"feeds": data}
    if not isinstance(data, dict):
        raise ValueError(f'{origin}: expected an object like {{"feeds": [...]}}')
    try:
        return SourcesConfig.model_validate(data)
    except ValidationError as e:
        raise ValueError(f"{origin}: invalid sources config:\n{e}") from e


def load_feed_config(path: str | os.PathLike[str] | None = None) -> list[FeedConfig]:
    """The feeds configured by the user (see ``load_sources_config``); DEFAULT_FEEDS if none."""
    return load_sources_config(path).feeds


def _split_creators(values: list[str]) -> list[str]:
    names: list[str] = []
    for v in values:
        v = normalize_whitespace(v)
        if not v:
            continue
        parts = [p.strip() for p in re.split(r",\s*|\s+and\s+", v) if p.strip()]
        # "Jane Doe, John Roe" -> two people; "Doe, Jane" -> one person written last-name first.
        if len(parts) > 1 and all(" " in p for p in parts):
            names.extend(parts)
        else:
            names.append(v)
    return names


def _rss_author(value: str) -> str:
    v = normalize_whitespace(value)
    m = re.match(r"^\S+@\S+\s*\((.+)\)$", v)  # RSS 2.0: "jane@example.com (Jane Doe)"
    return m.group(1).strip() if m else v


def _markup_to_text(value: str) -> str:
    return html_to_text(value) if "<" in value and ">" in value else normalize_whitespace(value)


def _atom_text(el: ET.Element | None) -> str:
    if el is None:
        return ""
    kind = (el.get("type") or "text").lower()
    if kind == "xhtml":
        parts = [el.text or ""]
        for child in el:
            child = copy.deepcopy(child)
            for node in child.iter():  # drop the XHTML namespace so <p>, <li>... are recognised
                if isinstance(node.tag, str) and node.tag.startswith("{"):
                    node.tag = node.tag.split("}", 1)[1]
            parts.append(ET.tostring(child, encoding="unicode"))
        return html_to_text("".join(parts))
    raw = "".join(el.itertext())
    return html_to_text(raw) if kind in {"html", "text/html"} else normalize_whitespace(raw)


def _title_fallback(text: str) -> str:
    return truncate_text(normalize_whitespace(text), 120) if text else ""


def parse_feed(
    content: bytes,
    *,
    name: str,
    kind: Literal["rss", "atom", "auto"] = "auto",
    base_url: str | None = None,
    warnings: list[str] | None = None,
    max_text_chars: int = DEFAULT_MAX_TEXT_CHARS,
    fetched_at: datetime | None = None,
) -> list[SourceDocument]:
    """Parse an RSS 2.0, RSS 1.0 (RDF) or Atom document into SourceDocuments (feed order).

    The format is detected from the root element; a mismatch with ``kind`` is noted in warnings.
    HTML in descriptions / content is converted to text; ``content:encoded`` (full article) is
    preferred over the ``description`` teaser. Raises ValueError if the document is not a feed.
    """
    warnings = warnings if warnings is not None else []
    fetched_at = fetched_at or _utcnow()
    root = _parse_xml(content)
    if root.tag == f"{{{ATOM}}}feed":
        detected, items = "atom", root.findall(f"{{{ATOM}}}entry")
    elif root.tag == "rss":
        channel = root.find("channel")
        detected, items = "rss", (channel.findall("item") if channel is not None else [])
    elif root.tag == f"{{{RDF}}}RDF":
        detected, items = "rss", root.findall(f"{{{RSS1}}}item")
    else:
        raise ValueError(f"not an RSS or Atom feed (root element {root.tag!r})")
    if kind != "auto" and kind != detected:
        warnings.append(f"feed {name!r} is configured as {kind!r} but is {detected!r}; parsed as {detected!r}")

    docs: list[SourceDocument] = []
    for i, item in enumerate(items):
        try:
            if detected == "atom":
                fields = _atom_item(item, base_url)
            else:
                fields = _rss_item(item, base_url)
        except Exception as e:  # one broken item must not lose the feed
            warnings.append(f"feed {name!r} item {i}: unreadable ({type(e).__name__}: {e})")
            continue
        title, url, text = fields["title"] or _title_fallback(fields["text"]), fields["url"], fields["text"]
        if not url:
            warnings.append(f"feed {name!r} item {i} ({title or 'untitled'}): no link, skipped")
            continue
        if not title:
            warnings.append(f"feed {name!r} item {i} ({url}): no title or text, skipped")
            continue
        docs.append(
            SourceDocument(
                source_type="rss",
                url=normalize_doc_url(url),
                title=title,
                authors=fields["authors"],
                published=fields["published"],
                text=truncate_text(text, max_text_chars) if text else "",
                fetched_at=fetched_at,
                source_name=name,
            )
        )
    return docs


def _resolve(url: str, base_url: str | None) -> str:
    url = normalize_whitespace(url)
    if not url:
        return ""
    if base_url and not urlsplit(url).scheme:
        url = urljoin(base_url, url)
    return url if urlsplit(url).scheme in {"http", "https"} else ""


def _rss_item(item: ET.Element, base_url: str | None) -> dict[str, Any]:
    def first(*tags: str) -> str:
        for t in tags:
            el = item.find(t)
            if el is not None and "".join(el.itertext()).strip():
                return "".join(el.itertext())
        return ""

    title = _markup_to_text(first("title", f"{{{RSS1}}}title", f"{{{DC}}}title"))
    link = first("link", f"{{{RSS1}}}link")
    if not link:
        guid = item.find("guid")  # a URL-valued guid is the best remaining link
        if guid is not None and (guid.text or "").strip().startswith(("http://", "https://")):
            link = guid.text or ""
    if not link:
        link = item.get(f"{{{RDF}}}about", "")
    body = first(f"{{{CONTENT}}}encoded") or first("description", f"{{{RSS1}}}description")
    text = html_to_text(body) if body else ""
    published = parse_date(first("pubDate", f"{{{DC}}}date"))
    creators = [("".join(el.itertext())) for el in item.findall(f"{{{DC}}}creator")]
    authors = _split_creators(creators) or [a for a in (_rss_author("".join(el.itertext())) for el in item.findall("author")) if a]
    return {"title": title, "url": _resolve(link, base_url), "text": text, "published": published, "authors": authors}


def _atom_item(entry: ET.Element, base_url: str | None) -> dict[str, Any]:
    title = _atom_text(entry.find(f"{{{ATOM}}}title"))
    href = ""
    links = entry.findall(f"{{{ATOM}}}link")
    for link in links:
        if link.get("rel", "alternate") == "alternate" and link.get("href"):
            href = link.get("href", "")
            break
    if not href and links:
        href = links[0].get("href", "")
    if not href:
        entry_id = _text(entry.find(f"{{{ATOM}}}id"))
        href = entry_id if entry_id.startswith(("http://", "https://")) else ""
    content = _atom_text(entry.find(f"{{{ATOM}}}content"))
    summary = _atom_text(entry.find(f"{{{ATOM}}}summary"))
    text = content if len(content) >= len(summary) else summary
    published = parse_date(_text(entry.find(f"{{{ATOM}}}published"))) or parse_date(
        _text(entry.find(f"{{{ATOM}}}updated"))
    )
    authors = [a for a in (_text(au.find(f"{{{ATOM}}}name")) for au in entry.findall(f"{{{ATOM}}}author")) if a]
    return {"title": title, "url": _resolve(href, base_url), "text": text, "published": published, "authors": authors}


FEED_ACCEPT = "application/rss+xml, application/atom+xml, application/xml;q=0.9, text/xml;q=0.8, */*;q=0.5"


class FeedSource:
    """Items from user-configured RSS / Atom feeds (research blogs, journals, arXiv listings).

    Feeds on the local machine / network are refused unless ``allow_private=True`` here or
    ``allow_private: true`` on that feed's config; feed documents are capped at ``MAX_FEED_BYTES``.
    """

    source_type = "rss"

    def __init__(
        self,
        feeds: Sequence[FeedConfig | dict[str, Any]] | None = None,
        *,
        client: httpx.Client | None = None,
        timeout_s: float = 20.0,
        max_retries: int = 2,
        min_interval_s: float = 1.0,
        max_text_chars: int = DEFAULT_MAX_TEXT_CHARS,
        user_agent: str | None = None,
        allow_private: bool = False,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if feeds is None:
            self.feeds = load_feed_config()
        else:
            self.feeds = [f if isinstance(f, FeedConfig) else FeedConfig.model_validate(f) for f in feeds]
        if max_text_chars < 1:
            raise ValueError("max_text_chars must be >= 1")
        self.max_text_chars = max_text_chars
        self.warnings: list[str] = []
        self._http = _Http(
            client,
            user_agent=user_agent,
            timeout_s=timeout_s,
            max_retries=max_retries,
            min_interval_s=min_interval_s,
            clock=clock,
            sleep=sleep,
            max_bytes=MAX_FEED_BYTES,
            allow_private=allow_private,
        )

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "FeedSource":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def fetch(self, max_items_per_feed: int = 20, since: date | None = None) -> list[SourceDocument]:
        """The newest ``max_items_per_feed`` items of every feed (published on/after ``since``).

        Items without a date are kept (they cannot be filtered) and sort last.
        """
        if not isinstance(max_items_per_feed, int) or max_items_per_feed < 1:
            raise ValueError("max_items_per_feed must be a positive integer")
        since_d = _as_since(since)
        self.warnings = []
        if not self.feeds:
            self.warnings.append(
                f"no feeds configured - add some to {default_sources_path()} or ${SOURCES_ENV} "
                "(see aitrading.discovery.sources)"
            )
            return []
        docs: list[SourceDocument] = []
        for feed in self.feeds:
            try:
                resp = self._http.get(feed.url, accept=FEED_ACCEPT, allow_private=feed.allow_private or None)
            except SourceError as e:
                self.warnings.append(f"feed {feed.name!r}: {e}")
                continue
            if resp.status_code != 200:
                self.warnings.append(f"feed {feed.name!r}: HTTP {resp.status_code} from {feed.url}")
                continue
            try:
                items = parse_feed(
                    resp.content,
                    name=feed.name,
                    kind=feed.kind,
                    base_url=str(resp.url),
                    warnings=self.warnings,
                    max_text_chars=self.max_text_chars,
                )
            except ValueError as e:
                self.warnings.append(f"feed {feed.name!r}: not a readable RSS/Atom feed ({e})")
                continue
            if since_d is not None:
                items = [d for d in items if d.published is None or d.published >= since_d]
            docs.extend(finalize_documents(items)[:max_items_per_feed])
        return finalize_documents(docs)


# =============================================================================================
# Single URL / local PDF
# =============================================================================================

PAGE_ACCEPT = "text/html,application/xhtml+xml,application/pdf;q=0.9,text/plain;q=0.8,*/*;q=0.5"
_DATE_META = (
    "citation_publication_date",
    "citation_online_date",
    "citation_date",
    "article:published_time",
    "dc.date",
    "dcterms.date",
    "dc.date.issued",
    "date",
    "pubdate",
)


def _title_from_text(text: str) -> str:
    for line in text.splitlines():
        line = normalize_whitespace(line)
        if len(line) < 4 or re.match(r"^(arXiv:\d|\d+$|page \d+)", line, re.I):
            continue
        return truncate_text(line, 200)
    return ""


def _name_from_url(url: str) -> str:
    stem = unquote(urlsplit(url).path.rstrip("/").rsplit("/", 1)[-1])
    return re.sub(r"\.(pdf|html?)$", "", stem, flags=re.I).replace("_", " ").replace("-", " ").strip()


_PDF_TYPES = frozenset({"application/pdf", "application/x-pdf", "application/acrobat"})


def fetch_url(
    url: str,
    *,
    client: httpx.Client | None = None,
    timeout_s: float = 30.0,
    max_retries: int = 2,
    max_chars: int = DEFAULT_MAX_TEXT_CHARS,
    user_agent: str | None = None,
    allow_private: bool = False,
) -> SourceDocument:
    """Fetch one page the trader provides (an article, a paper's abstract page, or a PDF).

    A PDF (``%PDF-`` body, or a PDF content type / ``.pdf`` link whose body is not a web page) is
    read with ``pdf_to_text`` (needs the optional ``pypdf``); HTML is decoded with the charset from
    the header or the page's own ``<meta charset>`` and converted with ``parse_html``. The title
    comes from the page's citation / Open Graph metadata, ``<title>`` or first ``<h1>``. Pages on
    the local machine / network are refused unless ``allow_private=True`` (see the module docstring).
    """
    url = _validate_http_url(url)
    if max_chars < 1:
        raise ValueError("max_chars must be >= 1")
    http = _Http(
        client,
        user_agent=user_agent,
        timeout_s=timeout_s,
        max_retries=max_retries,
        min_interval_s=0.0,
        clock=time.monotonic,
        sleep=time.sleep,
        max_bytes=MAX_FETCH_BYTES,
        allow_private=allow_private,
    )
    try:
        resp = http.get(url, accept=PAGE_ACCEPT)
    finally:
        http.close()
    if resp.status_code >= 400:
        hint = ""
        if resp.status_code in (401, 403, 429):
            hint = (
                " - the site may block automated downloads; open it in your browser, save the page or PDF "
                "and load it with load_pdf() instead"
            )
        raise SourceError(f"HTTP {resp.status_code} fetching {url}{hint}")
    if not 200 <= resp.status_code < 300:
        raise SourceError(f"HTTP {resp.status_code} fetching {url}")
    data = resp.content
    final_url = resp.url or url
    ctype = resp.content_type
    body = data.lstrip()
    is_pdf_body = body[:5] == b"%PDF-"
    looks_like_markup = body.removeprefix(b"\xef\xbb\xbf").lstrip()[:1] == b"<"
    claims_pdf = ctype in _PDF_TYPES or urlsplit(final_url).path.lower().endswith(".pdf")

    authors: list[str] = []
    published: date | None = None
    if not is_pdf_body and claims_pdf and (looks_like_markup or "html" in ctype):
        # Download links often answer with a login, cookie-consent or paywall page instead of the file.
        raise SourceError(
            f"{url}: expected a PDF but the site returned a web page (often a login, cookie-consent or paywall "
            "page); open the link in your browser, download the PDF and load it with load_pdf() instead"
        )
    if is_pdf_body or (claims_pdf and (not ctype or ctype in _PDF_TYPES or ctype == "application/octet-stream")):
        try:
            text = pdf_to_text(data, max_pages=MAX_PDF_PAGES)
        except ValueError as e:
            raise SourceError(f"{url}: {e}") from e
        title = _title_from_text(text) or _name_from_url(final_url)
        if not text.strip():
            raise SourceError(f"{url}: the PDF has no text layer (scanned?); OCR it first")
    elif not ctype or "html" in ctype or "xml" in ctype or looks_like_markup:
        parsed = parse_html(decode_text(data, resp.charset))
        text = parsed.text
        title = (
            parsed.meta_first("citation_title", "og:title", "dc.title")
            or parsed.title
            or parsed.h1
            or _name_from_url(final_url)
        )
        authors = parsed.meta_all("citation_author") or parsed.meta_all("dc.creator") or parsed.meta_all("author")
        published = next((d for d in (parse_date(parsed.meta_first(k)) for k in _DATE_META) if d), None)
    elif ctype.startswith("text/"):
        plain = decode_text(data, resp.charset, sniff_markup=False)
        text = "\n".join(normalize_whitespace(line) for line in plain.splitlines()).strip()
        title = _title_from_text(text) or _name_from_url(final_url)
    else:
        raise SourceError(f"{url}: unsupported content type {ctype!r}")
    if not text.strip():
        raise SourceError(f"{url}: no text could be extracted")
    return SourceDocument(
        source_type="url",
        url=normalize_doc_url(final_url),
        title=normalize_whitespace(title) or final_url,
        authors=[normalize_whitespace(a) for a in authors],
        published=published,
        text=truncate_text(text, max_chars),
        fetched_at=_utcnow(),
        source_name=_host(final_url),
    )


def load_pdf(path: str | os.PathLike[str], *, max_chars: int = DEFAULT_MAX_TEXT_CHARS) -> SourceDocument:
    """Read a local PDF (e.g. a paper the trader downloaded). Needs the optional ``pypdf``."""
    p = Path(path).expanduser()
    if not p.is_file():
        raise FileNotFoundError(f"no such PDF file: {p}")
    if max_chars < 1:
        raise ValueError("max_chars must be >= 1")
    try:
        text = pdf_to_text(p, max_pages=MAX_PDF_PAGES)
    except ValueError as e:
        raise SourceError(f"{p}: {e}") from e
    if not text.strip():
        raise SourceError(f"{p}: the PDF has no text layer (scanned?); OCR it first")
    title = _title_from_text(text) or _name_from_url(p.name) or p.stem
    return SourceDocument(
        source_type="pdf",
        url=p.resolve().as_uri(),
        title=title,
        text=truncate_text(text, max_chars),
        fetched_at=_utcnow(),
        source_name=f"PDF {p.name}",
    )
