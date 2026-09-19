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
"""

from __future__ import annotations

from urllib.parse import urljoin

import trafilatura
from bs4 import BeautifulSoup

LANDMARK_SELECTORS = ("main", "article", '[role="main"]')

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
    it is today.
    """
    page = _as_text(html)
    selected = _select_landmark(page) or _select_by_density(page, url) or page
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
