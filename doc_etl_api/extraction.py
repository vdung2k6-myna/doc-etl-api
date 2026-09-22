"""Select the main content of a fetched web page before Docling converts it.

A fetched page arrives wrapped in navigation, sidebars, promotional banners and
footers. Docling's HTML backend does not identify those regions -- its content
layer heuristic is defeated by the first heading anywhere in the document -- so
left alone they are converted, chunked and embedded alongside the content.

The cascade here tries, in order:

1. a declared landmark container -- ``<main>``, ``<article>`` or ``role="main"``;
2. text-density selection, via trafilatura, when the page declares nothing;
3. the page unchanged, when neither yields content.

A declared landmark is an exact signal, so it is never overridden by density
scoring, which is an inference. Every step returns HTML, so Docling remains the
only producer of markdown and a URL source keeps the output shape of a file
source. The final fallback means extraction can only replace chrome with
content, never replace content with nothing.

Whichever of the three produces the selection, the page's declared title is then
added to it as a heading if the selection carries no heading of its own. The
density pass drops headings wholesale on the pages its own recovery pass handles
-- measured, the 26 section headings inside an article's content container are
all absent from the selection -- so without this a source whose content yields no
heading is stored, chunked and embedded with nothing in it that names the
document.
"""

from __future__ import annotations

from urllib.parse import urljoin

import trafilatura
from bs4 import BeautifulSoup

LANDMARK_SELECTORS = ("main", "article", '[role="main"]')

# The document's declared title, and the heading elements that make it
# redundant. An ``<svg>`` declares a ``<title>`` of its own for accessibility,
# so a page inlining icons declares several and only the first non-SVG one names
# the document.
TITLE_TAG = "title"
HEADING_TAGS = ("h1", "h2", "h3", "h4", "h5", "h6")

# Chrome removed before density scoring. trafilatura removes these itself on
# pages its main extractor succeeds on, but falls back to an uncleaned pass when
# it does not -- which would resurrect the navigation it normally discards, and
# would make a text-sparse page score as nothing but its own navigation. Note
# that this list is what makes the whole-page fallback reachable at all: a page
# whose text is entirely chrome is correctly judged as having no main content.
CHROME_SELECTORS = (
    "nav",
    "aside",
    "footer",
    "form",
    '[role="navigation"]',
    '[role="banner"]',
    '[role="complementary"]',
    '[role="contentinfo"]',
)


def select_main_content(html: str | bytes, url: str | None = None) -> str:
    """Return the main-content region of ``html`` as HTML.

    ``url`` is the page's final URL, which trafilatura uses to resolve relative
    link, image and table targets. Returns the page unchanged when no main
    content can be identified, so an unidentifiable page is converted exactly as
    it is today. The page's declared title is added to the result as a heading
    where the selection carries none of its own -- see ``_with_title``.
    """
    page = _as_text(html)
    selected = _select_landmark(page) or _select_by_density(page, url) or page
    selected = _with_title(selected, _page_title(page))
    return _resolve_targets(selected, url) if url else selected


def _as_text(html: str | bytes) -> str:
    """Return fetched bytes as text, letting the parser detect the encoding.

    ``requests`` defaults an HTML response with no declared charset to
    ISO-8859-1, which mangles a page that relies on its own ``<meta>``
    declaration. The parser reads that declaration, so decoding through it
    keeps the page's encoding rather than a header default.
    """
    if isinstance(html, str):
        return html
    return str(BeautifulSoup(html, "html.parser"))


def _select_landmark(html: str) -> str | None:
    """Return the first declared main-content container that carries text.

    Containers are tried in ``LANDMARK_SELECTORS`` order, so a page nesting an
    ``<article>`` inside its ``<main>`` selects the ``<main>``. An empty
    container is not a selection: it falls through to density scoring rather
    than yielding a blank document.
    """
    soup = BeautifulSoup(html, "html.parser")
    for selector in LANDMARK_SELECTORS:
        for element in soup.select(selector):
            if element.get_text(strip=True):
                return str(element)
    return None


def _select_by_density(html: str, url: str | None) -> str | None:
    """Return trafilatura's text-density selection, or None if it yields nothing.

    ``output_format="html"`` is deliberate -- see design Decision 1. The links,
    tables and images opt-ins are needed because trafilatura drops all three by
    default, and dropping images would lose content that today's whole-page
    conversion keeps.
    """
    selected = trafilatura.extract(
        _remove_chrome(html),
        url=url,
        output_format="html",
        include_links=True,
        include_tables=True,
        include_images=True,
    )
    if selected and selected.strip():
        return selected
    return None


def _page_title(html: str) -> str | None:
    """Return the title the page declares for itself, or None if it declares none.

    Read from ``<title>`` and never from the page's first heading element, which
    may name the site rather than the document: measured on the MediaWiki corpus,
    the page ``<h1>`` is "Truyện kiếm hiệp", the wiki's own name, on an article
    whose ``<title>`` is "Wiki:Ý Thiên Đồ Long ký". A title that is empty or only
    whitespace counts as absent, so a page declaring one yields no heading
    rather than a blank one. An ``<svg>``'s own ``<title>`` is skipped: it labels
    an icon, not the document.
    """
    soup = BeautifulSoup(html, "html.parser")
    for element in soup.find_all(TITLE_TAG):
        if element.find_parent("svg") is not None:
            continue
        return element.get_text(strip=True) or None
    return None


def _with_title(selected: str, title: str | None) -> str:
    """Prepend ``title`` to ``selected`` as a heading, unless it already has one.

    A selection that carries a heading is left untouched: the heading is the
    document's own, and adding a second would name the document twice in the
    converted markdown. The title goes into the HTML as an element rather than
    onto finished markdown, so Docling stays the only producer of markdown, and
    it arrives as content -- embedded and searchable like any other text, and
    charged nothing against the chunk-size budget, unlike node metadata.
    """
    if not title or _has_heading(selected):
        return selected
    soup = BeautifulSoup(selected, "html.parser")
    heading = soup.new_tag("h1")
    heading.string = title
    # A fragment has no ``<body>``, and a heading placed outside ``<html>`` is
    # dropped by Docling's HTML backend. Measured: it converts the serialization
    # ``<h1>Title</h1><html>...`` to markdown without the heading at all, which
    # is why this inserts into the body rather than at the head of the document.
    target = soup.body if soup.body is not None else soup
    target.insert(0, heading)
    return str(soup)


def _has_heading(html: str) -> bool:
    """Whether ``html`` already carries a heading element of its own."""
    return BeautifulSoup(html, "html.parser").find(HEADING_TAGS) is not None


def _resolve_targets(html: str, base: str) -> str:
    """Resolve relative link and image targets in ``html`` against ``base``.

    Docling is given no base of its own -- see design Decision 4 -- so a
    root-relative ``href`` would otherwise reach the markdown as a Windows
    filesystem path, and, once any backend option is supplied, fail the
    conversion outright. trafilatura resolves targets on the density path by
    itself; this covers the landmark path, which never passes through it.
    """
    soup = BeautifulSoup(html, "html.parser")
    for tag, attribute in (("a", "href"), ("img", "src")):
        for element in soup.find_all(tag):
            target = element.get(attribute)
            if target and not _is_absolute(target):
                element[attribute] = urljoin(base, target)
    return str(soup)


def _is_absolute(target: str) -> bool:
    """Whether ``target`` already names a destination rather than a relative one."""
    return target.startswith(("http://", "https://", "mailto:", "tel:", "#", "data:"))


def _remove_chrome(html: str) -> str:
    """Strip navigation, sidebars, footers and forms from ``html``.

    Removing them before scoring is what keeps them out of the result whichever
    of trafilatura's internal passes ends up producing it.
    """
    soup = BeautifulSoup(html, "html.parser")
    for selector in CHROME_SELECTORS:
        for element in soup.select(selector):
            element.decompose()
    return str(soup)
