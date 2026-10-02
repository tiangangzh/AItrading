"""Text helpers for idea sources: HTML -> text, PDF -> text, truncation, URL canonicalisation.

Everything here is stdlib-only except ``pdf_to_text``, which lazily imports the *optional* ``pypdf``
package (``pip install pypdf``). Web and paper content is untrusted: these helpers only ever turn
it into plain text, they never execute or follow anything inside it.
"""

from __future__ import annotations

import codecs
import io
import os
import re
from dataclasses import dataclass, field
from html.parser import HTMLParser
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

__all__ = [
    "ParsedHtml",
    "canonical_url",
    "decode_text",
    "html_to_text",
    "normalize_whitespace",
    "parse_html",
    "pdf_to_text",
    "truncate_text",
]

# Elements whose content is never article text (dropped, including everything nested inside).
_HARD_SKIP = frozenset(
    {"script", "style", "noscript", "template", "svg", "head", "iframe", "textarea", "datalist"}
)
# Page furniture: dropped when properly closed. If one is left open (malformed HTML) it would swallow
# the rest of the page, so its text is kept aside and used when the element is never closed.
_SOFT_SKIP = frozenset({"nav", "footer", "button", "select"})
# Start tags that close a still-open soft element (``<main>`` / ``<article>`` never sit inside nav/footer).
_CLOSES_SOFT = frozenset({"main", "article"})
# Tags that may appear inside <head>; any other start tag (or text) implicitly ends the head (HTML5).
_HEAD_TAGS = frozenset(
    {"base", "basefont", "bgsound", "link", "meta", "title", "noscript", "noframes", "style", "script", "template", "head"}
)
_VOID_TAGS = frozenset(
    {"area", "base", "br", "col", "embed", "hr", "img", "input", "keygen", "link", "meta", "param", "source", "track", "wbr"}
)
# Elements whose end tag may be omitted, and the start tags that implicitly close them.
_P_CLOSERS = frozenset(
    {
        "address", "article", "aside", "blockquote", "details", "div", "dl", "fieldset", "figcaption", "figure",
        "footer", "form", "h1", "h2", "h3", "h4", "h5", "h6", "header", "hgroup", "hr", "main", "menu", "nav", "ol",
        "p", "pre", "section", "table", "ul",
    }
)
_IMPLIED_END = {
    "p": _P_CLOSERS,
    "li": frozenset({"li"}),
    "dt": frozenset({"dt", "dd"}),
    "dd": frozenset({"dt", "dd"}),
    "option": frozenset({"option", "optgroup"}),
    "tr": frozenset({"tr"}),
    "td": frozenset({"td", "th", "tr"}),
    "th": frozenset({"td", "th", "tr"}),
}
# Elements that bound the "button scope" in which a start tag implicitly closes an open <p>.
_SCOPE_TAGS = frozenset({"applet", "button", "caption", "html", "marquee", "object", "table", "td", "template", "th"})
# Elements that start / end a block of text (a paragraph break in the output).
_BLOCK_TAGS = frozenset(
    {
        "p", "div", "section", "article", "main", "header", "aside", "blockquote", "pre", "figure", "figcaption",
        "h1", "h2", "h3", "h4", "h5", "h6", "ul", "ol", "li", "dl", "dt", "dd", "table", "thead", "tbody", "tr",
        "td", "th", "caption", "hr", "address", "details", "summary", "body", "html", "form", "fieldset",
    }
)
_HIDDEN_STYLE = re.compile(r"(?<![\w-])(?:display\s*:\s*none|visibility\s*:\s*(?:hidden|collapse))(?![\w-])", re.I)
_ZERO_WIDTH = re.compile("[​‌‍⁠﻿]")
_INLINE_WS = re.compile(r"[^\S\n]+")  # whitespace other than newlines (includes NBSP)


def _is_hidden(attrs: list[tuple[str, str | None]]) -> bool:
    """True for elements a reader never sees: ``hidden``, ``aria-hidden="true"``, inline display:none / visibility:hidden.

    ``hidden="until-found"`` (collapsed sections a reader can expand) counts as visible.
    """
    for key, value in attrs:
        key, value = key.lower(), (value or "").strip()
        if key == "hidden" and value.lower() != "until-found":
            return True
        if key == "aria-hidden" and value.lower() == "true":
            return True
        if key == "style" and value and _HIDDEN_STYLE.search(value):
            return True
    return False


def normalize_whitespace(text: str) -> str:
    """Collapse all whitespace (incl. newlines and NBSP) to single spaces and strip."""
    return re.sub(r"\s+", " ", _ZERO_WIDTH.sub("", text or "")).strip()


@dataclass
class ParsedHtml:
    text: str
    title: str | None = None
    h1: str | None = None
    meta: dict[str, list[str]] = field(default_factory=dict)

    def meta_first(self, *names: str) -> str | None:
        for name in names:
            for value in self.meta.get(name.lower(), []):
                if value.strip():
                    return value.strip()
        return None

    def meta_all(self, name: str) -> list[str]:
        return [v.strip() for v in self.meta.get(name.lower(), []) if v.strip()]


def _make_block(parts: list[str]) -> str:
    block = _INLINE_WS.sub(" ", _ZERO_WIDTH.sub("", "".join(parts)))
    block = "\n".join(line.strip() for line in block.split("\n"))
    return re.sub(r"\n{2,}", "\n", block).strip()


class _HtmlExtractor(HTMLParser):
    """Collects visible article text.

    ``self._stack`` holds the open elements of a skipped region as ``(tag, kind)`` frames, where kind
    is "hard" (never text: script, head, hidden elements, ...), "soft" (page furniture: nav, footer,
    ...) or "inner" (an ordinary element nested inside a skipped region, tracked so that end tags
    match the right element). The stack is empty while reading normal text. ``self._open`` tracks the
    elements open outside skipped regions, so that an end tag closing one of them also ends a skipped
    region left open inside it (``<h1>Title <span hidden>x</h1>``), as browsers do.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)  # decodes &amp; &#8217; &nbsp; ...
        self._stack: list[tuple[str, str]] = []
        self._open: list[str] = []
        self._pre = 0
        self._blocks: list[str] = []
        self._buf: list[str] = []
        # text of the currently open soft element, and of the closed ones (with their position)
        self._soft_buf: list[str] = []
        self._soft_blocks: list[str] = []
        self._closed_soft: list[tuple[int, list[str]]] = []
        self._in_title = False
        self._title: list[str] = []
        self._in_h1 = False
        self._h1: list[str] = []
        self.h1: str | None = None
        self.meta: dict[str, list[str]] = {}

    # -- helpers -------------------------------------------------------------------------------
    def _flush(self) -> None:
        if self._buf:
            block = _make_block(self._buf)
            if block:
                self._blocks.append(block)
            self._buf = []

    def _flush_soft(self) -> None:
        if self._soft_buf:
            block = _make_block(self._soft_buf)
            if block:
                self._soft_blocks.append(block)
            self._soft_buf = []

    def _mode(self) -> str:
        """"normal", "soft" (inside an open nav/footer/...) or "hard" (inside dropped content)."""
        if not self._stack:
            return "normal"
        kinds = {kind for _, kind in self._stack}
        return "hard" if "hard" in kinds else "soft"

    def _close_soft_capture(self) -> None:
        """A soft element was closed: its text is furniture; keep it only for the empty-page fallback."""
        self._flush_soft()
        if self._soft_blocks:
            self._closed_soft.append((len(self._blocks), self._soft_blocks))
        self._soft_blocks = []

    def _pop_to(self, index: int) -> None:
        had_soft = any(kind == "soft" for _, kind in self._stack)
        del self._stack[index:]
        if had_soft and not any(kind == "soft" for _, kind in self._stack):
            self._close_soft_capture()

    def _close_implied(self, tag: str) -> None:
        """Pop open (non-skipped) elements whose end tag is implied by start tag ``tag`` (HTML5)."""
        if tag in _P_CLOSERS and "p" in self._open:
            i = len(self._open) - 1 - self._open[::-1].index("p")
            if not any(t in _SCOPE_TAGS for t in self._open[i + 1 :]):
                del self._open[i:]
        while self._open and tag in _IMPLIED_END.get(self._open[-1], ()):
            self._open.pop()

    def _maybe_close_head(self) -> None:
        """HTML5 ends <head> implicitly at the first body element or text, even without </head>.

        Only when <head> itself is the innermost open element: inside a head <noscript>, <template>,
        <script> or <style> browsers treat the content as raw text, so the head stays open.
        """
        if self._stack and self._stack[-1][0] == "head":
            self._pop_to(len(self._stack) - 1)

    # -- parser callbacks ----------------------------------------------------------------------
    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag == "meta":
            a = {k.lower(): (v or "") for k, v in attrs}
            key = (a.get("name") or a.get("property") or a.get("itemprop") or "").strip().lower()
            if key and "content" in a:
                self.meta.setdefault(key, []).append(a["content"])
            return
        if tag == "title":
            # <title> lives in <head> (otherwise skipped); ignore <svg><title> and repeats.
            self._in_title = not any(t == "svg" for t, _ in self._stack) and not self._title
            return
        if tag not in _HEAD_TAGS:
            self._maybe_close_head()
        # omitted end tags: <p>a<p>b, <li>a<li>b, ... (only matters inside skipped regions)
        while self._stack and tag in _IMPLIED_END.get(self._stack[-1][0], ()):
            self._pop_to(len(self._stack) - 1)
        if tag in _CLOSES_SOFT and self._stack and self._mode() == "soft":
            self._soft_buf, self._soft_blocks = [], []  # an unclosed nav/footer: drop its furniture...
            self._pop_to(0)  # ...and read the main content
        hidden = _is_hidden(attrs)
        mode = self._mode()
        if mode == "normal":
            self._close_implied(tag)
        else:
            if tag in _VOID_TAGS:
                if mode == "soft":
                    if tag == "br":
                        self._soft_buf.append("\n")
                    elif tag in _BLOCK_TAGS:
                        self._flush_soft()
                return
            kind = "hard" if (tag in _HARD_SKIP or hidden) else ("soft" if tag in _SOFT_SKIP else "inner")
            self._stack.append((tag, kind))
            if mode == "soft" and kind != "hard" and tag in _BLOCK_TAGS:
                self._flush_soft()
            return
        if tag in _VOID_TAGS:
            if hidden:
                return
        elif tag in _HARD_SKIP or hidden or tag in _SOFT_SKIP:
            if tag in _BLOCK_TAGS or tag in ("nav", "footer"):
                self._flush()  # a skipped block still separates the text before and after it
            if tag in _SOFT_SKIP and not hidden:
                self._stack.append((tag, "soft"))
                self._soft_buf, self._soft_blocks = [], []
            else:
                self._stack.append((tag, "hard"))
            return
        else:
            self._open.append(tag)
        if tag == "h1" and self.h1 is None:
            self._in_h1 = True
        if tag == "pre":
            self._pre += 1
        if tag == "br":
            self._buf.append("\n")
        elif tag in _BLOCK_TAGS:
            self._flush()

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag == "meta":
            self.handle_starttag(tag, attrs)
            return
        if tag == "title":  # <title/>: an empty title
            return
        if tag not in _HEAD_TAGS:
            self._maybe_close_head()
        if self._stack or tag in _HARD_SKIP or tag in _SOFT_SKIP or _is_hidden(attrs):
            # an empty element (<svg/>, <div hidden/>): nothing to skip, nothing to read
            if tag == "br" and self._mode() == "soft":
                self._soft_buf.append("\n")
            return
        self.handle_starttag(tag, attrs)
        if tag not in _VOID_TAGS:  # XHTML-style <p/>, <pre/>: an element that opens and closes at once
            self.handle_endtag(tag)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag == "title":
            self._in_title = False
            return
        if self._stack:
            mode = self._mode()
            for i in range(len(self._stack) - 1, -1, -1):
                if self._stack[i][0] == tag:
                    if mode == "soft" and tag in _BLOCK_TAGS:
                        self._flush_soft()
                    self._pop_to(i)
                    return
            if tag not in self._open or tag in ("body", "html"):
                return  # a stray end tag (ignored, like browsers do); </body> is treated as end of input
            self._pop_to(0)  # it closes an element that encloses the whole skipped region
        if tag in self._open:
            while self._open.pop() != tag:
                pass
        if tag == "h1" and self._in_h1:
            self._in_h1 = False
            h1 = normalize_whitespace("".join(self._h1))
            self.h1 = h1 or None
        if tag == "pre" and self._pre:
            self._pre -= 1
        if tag in _BLOCK_TAGS:
            self._flush()

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self._title.append(data)
            return
        if self._stack and data.strip():
            self._maybe_close_head()
        mode = self._mode()
        if mode == "hard":
            return
        if mode == "soft":
            self._soft_buf.append(re.sub(r"\s+", " ", data))
            return
        if self._in_h1:
            self._h1.append(data)
        # Source newlines are just whitespace in HTML (only <br> / blocks break lines), except in <pre>.
        self._buf.append(data if self._pre else re.sub(r"\s+", " ", data))

    def result(self) -> ParsedHtml:
        self._flush()
        blocks = list(self._blocks)
        if any(kind == "soft" for _, kind in self._stack):
            # A nav / footer / button / select that was never closed swallowed the rest of the page:
            # browsers still show that text, so keep it.
            self._flush_soft()
            blocks += self._soft_blocks
        if not blocks and self._closed_soft:
            # Nothing outside page furniture (e.g. a page laid out entirely inside <nav>): use the furniture.
            for _, soft in self._closed_soft:
                blocks += soft
        title = normalize_whitespace("".join(self._title)) or None
        return ParsedHtml(text="\n\n".join(blocks), title=title, h1=self.h1, meta=self.meta)


def parse_html(html: str) -> ParsedHtml:
    """Parse an HTML page or fragment into text plus title / first <h1> / <meta> tags.

    Script, style, head, navigation, footer, form controls and hidden elements (``hidden``,
    ``aria-hidden="true"``, inline ``display:none`` / ``visibility:hidden`` - a common place to plant
    prompt-injection text) are dropped; paragraphs, headings and list items each become their own
    block (blocks are separated by a blank line); whitespace inside a block is collapsed and HTML
    entities are decoded. Malformed pages are handled the way browsers do: a missing ``</head>``
    ends at the first body element, and a nav / footer that is never closed does not swallow the
    article (``<main>`` / ``<article>`` close it, and otherwise its text is kept).
    """
    parser = _HtmlExtractor()
    try:
        parser.feed(html or "")
        parser.close()
    except Exception:  # pragma: no cover - html.parser is very lenient; keep whatever was parsed
        pass
    return parser.result()


def html_to_text(html: str) -> str:
    """Plain text of an HTML page or fragment (see ``parse_html``)."""
    return parse_html(html).text


# ---------------------------------------------------------------------------------------------
# Bytes -> str
# ---------------------------------------------------------------------------------------------

_META_CHARSET = re.compile(rb"""<meta\b[^>]*?\bcharset\s*=\s*["']?\s*([A-Za-z0-9_.:\-]+)""", re.I)
_XML_ENCODING = re.compile(rb"""^\s*<\?xml\b[^>]*?\bencoding\s*=\s*["']([A-Za-z0-9_.:\-]+)["']""", re.I)
# WHATWG Encoding Standard: these labels are decoded as windows-1252 by browsers.
_AS_CP1252 = frozenset({"iso-8859-1", "iso8859-1", "latin-1", "latin1", "l1", "us-ascii", "ascii", "cp819", "windows-1252"})


def _codec(label: str | None) -> str | None:
    label = (label or "").strip().strip("\"'").lower()
    if not label:
        return None
    if label in _AS_CP1252:
        return "cp1252"
    try:
        return codecs.lookup(label).name
    except LookupError:
        return None


def decode_text(data: bytes, declared: str | None = None, *, sniff_markup: bool = True) -> str:
    """Decode a downloaded page the way browsers do.

    Order: a byte-order mark, then the ``declared`` charset (from the Content-Type header), then -
    for HTML/XML (``sniff_markup``) - a ``<meta charset>`` / ``<meta http-equiv=Content-Type>`` or
    ``<?xml encoding=...?>`` declaration in the first 4 KB, then UTF-8, and finally windows-1252.
    Undecodable bytes become U+FFFD; this never raises.
    """
    data = bytes(data or b"")
    if data.startswith(codecs.BOM_UTF8):
        return data[len(codecs.BOM_UTF8) :].decode("utf-8", "replace")
    if data.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        return data.decode("utf-16", "replace")
    encoding = _codec(declared)
    if encoding is None and sniff_markup:
        head = data[:4096]
        m = _META_CHARSET.search(head) or _XML_ENCODING.search(head)
        encoding = _codec(m.group(1).decode("ascii", "ignore")) if m else None
        if encoding and encoding.startswith("utf-16"):  # found by an ASCII scan, so the bytes are not UTF-16
            encoding = "utf-8"
    if encoding is not None:
        return data.decode(encoding, "replace")
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("cp1252", "replace")


# ---------------------------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------------------------

PYPDF_HINT = (
    "Reading PDFs needs the optional 'pypdf' package: pip install pypdf "
    "(it is optional - everything else in aitrading works without it)."
)


def pdf_to_text(source: str | os.PathLike[str] | bytes | bytearray, *, max_pages: int | None = None) -> str:
    """Extract the text of a PDF given as a path or raw bytes.

    Uses ``pypdf`` if installed, otherwise raises ``ImportError`` explaining that it is an optional
    dependency (``pip install pypdf``). Raises ``ValueError`` if the input is not a PDF.
    Scanned (image-only) PDFs have no text layer and yield an empty string.
    """
    if max_pages is not None and max_pages < 1:
        raise ValueError("max_pages must be >= 1")
    if isinstance(source, (bytes, bytearray)):
        data = bytes(source)
    else:
        with open(source, "rb") as fh:
            data = fh.read()
    if not data.lstrip()[:5] == b"%PDF-":
        raise ValueError("not a PDF document (missing %PDF- header)")
    try:
        from pypdf import PdfReader
    except ImportError as e:
        raise ImportError(PYPDF_HINT) from e

    reader = PdfReader(io.BytesIO(data))
    pages = list(reader.pages)
    if max_pages is not None:
        pages = pages[:max_pages]
    out: list[str] = []
    for page in pages:
        try:
            raw = page.extract_text() or ""
        except Exception:  # a single broken page should not lose the whole document
            raw = ""
        lines = [_INLINE_WS.sub(" ", _ZERO_WIDTH.sub("", line)).strip() for line in raw.splitlines()]
        page_text = "\n".join(line for line in lines if line)
        if page_text:
            out.append(page_text)
    return "\n\n".join(out)


# ---------------------------------------------------------------------------------------------
# Truncation
# ---------------------------------------------------------------------------------------------

# A sentence ends with . ! or ? (optionally followed by a closing quote/bracket) and whitespace,
# or at a paragraph break.
_SENTENCE_END = re.compile(r"(?<=[.!?])[\"'”’)\]]?(?=\s)|\n\s*\n")
_ELLIPSIS = " …"


def truncate_text(text: str, max_chars: int) -> str:
    """Shorten ``text`` to at most ``max_chars`` characters, cutting at a sentence boundary.

    If no sentence boundary lies in the second half of the allowed window, the cut falls back to a
    word boundary and an ellipsis is appended (still within ``max_chars``).
    """
    if max_chars < 1:
        raise ValueError("max_chars must be >= 1")
    if text is None:
        return ""
    if len(text) <= max_chars:
        return text
    window = text[: max_chars + 1]
    best = -1
    for m in _SENTENCE_END.finditer(window):
        end = m.end() if m.group(0).strip() else m.start()
        if end <= max_chars:
            best = max(best, end)
    if best >= max_chars // 2:
        return text[:best].rstrip()
    limit = max(max_chars - len(_ELLIPSIS), 1)
    cut = text[:limit]
    space = cut.rfind(" ")
    if space >= limit // 2:
        cut = cut[:space]
    cut = cut.rstrip()
    return (cut + _ELLIPSIS)[:max_chars] if max_chars > len(_ELLIPSIS) else text[:max_chars]


# ---------------------------------------------------------------------------------------------
# URLs
# ---------------------------------------------------------------------------------------------

_TRACKING_PARAMS = re.compile(r"^(utm_[a-z]+|fbclid|gclid|mc_cid|mc_eid|ref|ref_src|cmpid)$", re.I)
_ARXIV_PATH = re.compile(r"^/(abs|pdf|html)/(?P<id>[a-z\-]+(?:\.[A-Z]{2})?/\d{7}|\d{4}\.\d{4,5})(?:v\d+)?(?:\.pdf)?/?$", re.I)


def canonical_url(url: str) -> str:
    """Canonical form of a URL for comparing / de-duplicating links.

    Lower-cases scheme and host, maps http to https, drops ``www.``, fragments, tracking parameters
    and trailing slashes, and maps every arXiv abs/pdf/html link (any version) to
    ``https://arxiv.org/abs/<id>``.
    """
    raw = (url or "").strip()
    if not raw:
        return ""
    if "://" not in raw:
        raw = "https://" + raw
    try:  # malformed URLs (bad port, bad IPv6 literal) never raise: they map to themselves, lower-cased
        parts = urlsplit(raw)
        port = parts.port
    except ValueError:
        return raw.lower()
    host = (parts.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    if ":" in host:  # IPv6 literal
        host = f"[{host}]"
    port = port if port not in (None, 80, 443) else None
    netloc = f"{host}:{port}" if port else host
    path = parts.path or "/"
    if host in {"arxiv.org", "export.arxiv.org"}:
        m = _ARXIV_PATH.match(path)
        if m:
            return f"https://arxiv.org/abs/{m.group('id')}"
        netloc = "arxiv.org"
    query = urlencode([(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True) if not _TRACKING_PARAMS.match(k)])
    path = re.sub(r"/{2,}", "/", path)
    if len(path) > 1:
        path = path.rstrip("/")
    return urlunsplit(("https", netloc, path, query, ""))
