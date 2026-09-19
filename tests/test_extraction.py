"""Tests for main-content extraction.

Every assertion here is about the region that survives selection. The fixtures
carry real prose rather than a token paragraph, because trafilatura scores text
density: a page it judges too sparse is dropped wholesale, which would make a
tiny fixture pass for the wrong reason.
"""

from __future__ import annotations

import io

import pytest

from doc_etl_api.extraction import select_main_content
from doc_etl_api.pipeline import DoclingConverter

PAGE_URL = "https://example.com/page"

# Long enough that trafilatura treats the region as content rather than boilerplate.
BODY = " ".join(
    f"Sentence {index} describes the widget frame and its latch clearance."
    for index in range(1, 26)
)

NAV = '<nav><h2>Browse</h2><a href="/">Home</a><a href="/docs">Docs</a></nav>'
BANNER = '<div class="promo"><h3>Buy now</h3><p>Save 20% on every latch this week.</p></div>'
SIDEBAR = '<aside><h3>Related</h3><a href="/other">Another widget</a></aside>'
FOOTER = "<footer><p>&copy; 2026 Widget Co. All rights reserved.</p></footer>"


def page(body: str, chrome: str = "") -> str:
    """Wrap *body* in a page whose surrounding chrome is constant across tests."""
    return (
        "<html><head><title>Widget Handbook</title></head><body>"
        f"{NAV}{chrome}{body}{FOOTER}"
        "</body></html>"
    )


def assert_chrome_absent(selected: str) -> None:
    assert "Browse" not in selected, "navigation survived extraction"
    assert "Widget Co" not in selected, "footer survived extraction"


# --- 1.2 the landmark pass -------------------------------------------------


@pytest.mark.parametrize(
    ("label", "markup"),
    [
        ("main", f"<main><h1>Widget Handbook</h1><p>{BODY}</p></main>"),
        ("article", f"<article><h1>Widget Handbook</h1><p>{BODY}</p></article>"),
        (
            "role-main",
            f'<div role="main"><h1>Widget Handbook</h1><p>{BODY}</p></div>',
        ),
    ],
)
def test_landmark_container_is_selected_and_chrome_excluded(label: str, markup: str) -> None:
    selected = select_main_content(page(markup, chrome=BANNER + SIDEBAR), PAGE_URL)

    assert "Widget Handbook" in selected
    assert "Sentence 1 describes" in selected
    assert_chrome_absent(selected)
    assert "Buy now" not in selected, "promotional banner survived extraction"
    assert "Another widget" not in selected, "sidebar survived extraction"


def test_landmark_container_wins_over_density_scoring() -> None:
    """A declared landmark is exact, so it must not be second-guessed by scoring."""
    markup = f"<main><h1>Widget Handbook</h1><p>{BODY}</p></main>"
    # A far denser sibling would win a density contest but is not the main content.
    decoy = f'<div class="comments"><h2>Comments</h2><p>{BODY} {BODY}</p></div>'

    selected = select_main_content(page(markup, chrome=decoy), PAGE_URL)

    assert "Widget Handbook" in selected
    assert "Comments" not in selected


def test_empty_landmark_falls_through_to_density() -> None:
    """An empty container is not a selection -- it must not yield a blank document."""
    markup = f'<main></main><div class="content"><h1>Widget Handbook</h1><p>{BODY}</p></div>'

    selected = select_main_content(page(markup), PAGE_URL)

    assert "Widget Handbook" in selected


# --- 1.3 the density pass --------------------------------------------------


def test_density_pass_selects_main_text_without_a_landmark() -> None:
    body = f'<div class="content"><h1>Widget Handbook</h1><h2>Assembly</h2><p>{BODY}</p></div>'

    selected = select_main_content(page(body, chrome=BANNER + SIDEBAR), PAGE_URL)

    assert "Widget Handbook" in selected
    assert "Assembly" in selected
    assert "Sentence 1 describes" in selected
    assert_chrome_absent(selected)


def test_density_pass_excludes_chrome_that_carries_a_heading() -> None:
    """The measured failure: a heading in the chrome flips Docling's content layer.

    ``NAV`` carries ``<h2>Browse</h2>``, which is exactly what defeats Docling's
    own boilerplate heuristic. Selection must remove the region regardless.
    """
    body = f'<div class="content"><h1>Widget Handbook</h1><p>{BODY}</p></div>'

    selected = select_main_content(page(body), PAGE_URL)

    assert "Widget Handbook" in selected
    assert_chrome_absent(selected)


def test_density_pass_excludes_chrome_on_a_link_heavy_page() -> None:
    """trafilatura's own cleaning is skipped on pages its main extractor fails on.

    Measured: a page of links keeps its navigation unless the chrome is removed
    before scoring, so this asserts the pre-clean rather than trusting it.
    """
    links = "".join(
        f'<li><a href="/guide-{index}">Guide {index}</a></li>' for index in range(1, 40)
    )
    body = f'<div class="content"><h1>Guides</h1><ul>{links}</ul></div>'

    selected = select_main_content(page(body), PAGE_URL)

    assert "Guides" in selected
    assert "Guide 1" in selected
    assert_chrome_absent(selected)


def test_density_pass_resolves_links_in_the_selected_region() -> None:
    body = (
        '<div class="content"><h1>Widget Handbook</h1>'
        f'<p>See <a href="/pricing">our pricing page</a> before ordering. {BODY}</p></div>'
    )

    selected = select_main_content(page(body), PAGE_URL)

    assert "https://example.com/pricing" in selected, "relative link was not resolved"


def test_density_pass_preserves_inline_images() -> None:
    """Today's whole-page conversion keeps images; extraction must not drop them."""
    body = f'<div class="content"><h1>Widget Handbook</h1><img src="/latch.png"><p>{BODY}</p></div>'

    selected = select_main_content(page(body), PAGE_URL)

    assert "latch.png" in selected, "inline image was dropped from the selected region"


# --- 1.4 the whole-page fallback -------------------------------------------


def test_empty_page_is_returned_unchanged() -> None:
    html = "<html><body></body></html>"

    assert select_main_content(html, PAGE_URL) == html


def test_landmark_free_page_whose_density_pass_yields_nothing_is_unchanged() -> None:
    """A page carrying no text at all yields nothing to score, so it survives whole.

    Measured: trafilatura returns content for any page with text, however short,
    and returns nothing only when there is no text whatsoever. So this page --
    images and no prose, and no chrome for the pre-clean to strip -- is the case
    that reaches the fallback on its own.
    """
    html = (
        '<html><body><div class="gallery"><img src="/a.jpg"><img src="/b.jpg"></div></body></html>'
    )

    selected = select_main_content(html, PAGE_URL)

    assert selected.count("<img") == 2, "the fallback dropped page content"
    assert "https://example.com/a.jpg" in selected, "an image target was left relative"


def test_page_is_returned_verbatim_when_no_page_url_is_available() -> None:
    """With no page URL to resolve against, the fallback returns the page as it is."""
    html = '<html><body><div class="gallery"><img src="/a.jpg"></div></body></html>'

    assert select_main_content(html) == html


def test_text_sparse_page_is_converted_whole_rather_than_as_its_chrome() -> None:
    """An image gallery has no text to score, so its navigation is not "the content".

    Measured: without the chrome pre-clean this page selects the navigation and
    the images are lost entirely. The whole page is the safe answer, and it is
    what makes the fallback reachable rather than theoretical.
    """
    images = "".join(f'<img src="/shot-{index}.jpg" alt="Shot {index}">' for index in range(1, 20))
    html = page(f'<div class="gallery">{images}</div>')

    selected = select_main_content(html, PAGE_URL)

    # The whole page is converted -- every image kept, and the navigation still
    # present because nothing selected it -- rather than being reduced to the
    # navigation. The image targets are the only thing rewritten.
    assert selected.count("<img") == 19, "a text-sparse page lost its images"
    assert "Browse" in selected, "a text-sparse page was reduced to its chrome"
    assert "https://example.com/shot-1.jpg" in selected, "an image target was left relative"


# --- 1.5 the selected region is convertible --------------------------------


def convert(selected: str) -> str:
    """Run the selected region through the real converter, as ingestion will."""
    return DoclingConverter().convert_file(io.BytesIO(selected.encode()), "page.html")


def test_selected_region_converts_to_markdown_without_chrome() -> None:
    """The region must survive the real converter, not just the selection step."""
    body = f'<div class="content"><h1>Widget Handbook</h1><h2>Assembly</h2><p>{BODY}</p></div>'

    markdown = convert(select_main_content(page(body, chrome=BANNER + SIDEBAR), PAGE_URL))

    assert "Widget Handbook" in markdown
    assert "Assembly" in markdown
    assert "Sentence 1 describes" in markdown
    assert "Browse" not in markdown, "navigation reached the markdown"
    assert "Buy now" not in markdown, "banner reached the markdown"
    assert "Another widget" not in markdown, "sidebar reached the markdown"
    assert "Widget Co" not in markdown, "footer reached the markdown"


def test_landmark_region_converts_to_markdown_without_chrome() -> None:
    """The landmark pass returns a bare fragment, so it needs the same round trip."""
    markup = f"<main><h1>Widget Handbook</h1><p>{BODY}</p></main>"

    markdown = convert(select_main_content(page(markup, chrome=BANNER), PAGE_URL))

    assert "Widget Handbook" in markdown
    assert "Sentence 1 describes" in markdown
    assert "Browse" not in markdown
    assert "Widget Co" not in markdown
