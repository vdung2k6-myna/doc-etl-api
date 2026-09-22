import hashlib
import logging
import re
import threading
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import requests
import tiktoken
from docling.datamodel.base_models import ConversionStatus
from llama_index.core import Document as LlamaDocument
from llama_index.core import Settings as LlamaSettings
from llama_index.core.embeddings import MockEmbedding
from llama_index.core.schema import MetadataMode

from doc_etl_api.config import DEFAULT_USER_AGENT, Settings
from doc_etl_api.pipeline import (
    DoclingConverter,
    IndexPipeline,
    _content_token_ids,
    _node_id,
    _split_oversized_text,
    _split_sections,
)
from tests.stubs import (
    EMBEDDING_MAX_TOKENS,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_TOKENIZER,
    StubEmbedding,
)

# Sentence-packed document: long enough to split into several nodes at the chunk
# sizes used below, and not paragraph-shaped, so adjacent nodes share a boundary.
SENTENCE_DOC = " ".join(
    f"Sentence {index} covers its own separate subject in its own separate wording."
    for index in range(1, 61)
)

# A passage unlike anything around it, placed where the pre-change sizing left it
# in a chunk's discarded tail. Measured: 381 tokens into its chunk against the
# ~254-token window the model actually reads, so none of it reached the embedding.
LATE_MARKER = " ".join(
    [
        "The closing remark concerns xylophones rebuilt from volcanic basalt.",
        "Basalt xylophones require tuning against obsidian resonators.",
        "Obsidian resonators are quarried nowhere near the administrative cases.",
    ]
)
LONG_DOC = " ".join(
    [
        f"Paragraph {i} revisits the ordinary administrative details of case {i}."
        for i in range(1, 30)
    ]
    + [LATE_MARKER]
    + [
        f"Paragraph {i} revisits the ordinary administrative details of case {i}."
        for i in range(30, 81)
    ]
)


def _model_tokens(text: str) -> int:
    """Tokens the embedding model reads for text, special tokens included."""
    return len(EMBEDDING_TOKENIZER.encode(text, add_special_tokens=True))


def _cosine(left, right) -> float:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    return float(left @ right / (np.linalg.norm(left) * np.linalg.norm(right)))


def _boundary_overlap(earlier: str, later: str) -> int:
    """Shared content tokens between two adjacent nodes, computed independently."""
    head = EMBEDDING_TOKENIZER.encode(earlier, add_special_tokens=False)
    tail = EMBEDDING_TOKENIZER.encode(later, add_special_tokens=False)
    for size in range(min(len(head), len(tail)), 0, -1):
        if head[-size:] == tail[:size]:
            return size
    return 0


def _stub_pipeline(chunk_size=None, chunk_overlap=10, embedding_model=None, min_chunk_tokens=32):
    """A pipeline paired with a double stating its own limit and tokenizer."""
    return IndexPipeline(
        Settings(
            vector_store_backend="simple",
            embedding_model=EMBEDDING_MODEL_NAME,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            min_chunk_tokens=min_chunk_tokens,
            default_top_k=5,
        ),
        converter=MagicMock(),
        embedding_model=embedding_model or StubEmbedding(embed_dim=8),
    )


@pytest.fixture
def settings():
    # Chunk sizes are stated explicitly here: IndexPipeline derives its default
    # from the embedding model, so a fixture that leaves them unset would assert
    # nothing about the sizing these tests exercise. 128 fits within the 256-token
    # model they are paired with.
    return Settings(
        vector_store_backend="simple",
        embedding_model=EMBEDDING_MODEL_NAME,
        chunk_size=128,
        chunk_overlap=10,
        default_top_k=2,
    )


def test_docling_converter_supported_files():
    converter = DoclingConverter()
    assert converter.is_supported_file("report.pdf")
    assert converter.is_supported_file("notes.docx")
    assert converter.is_supported_file("page.html")
    assert not converter.is_supported_file("archive.zip")


def test_docling_converter_file(settings, tmp_path):
    converter = DoclingConverter()
    sample = tmp_path / "hello.txt"
    sample.write_text("Hello, world!")

    with patch.object(
        converter._converter,
        "convert",
        return_value=MagicMock(
            status=ConversionStatus.SUCCESS,
            document=MagicMock(export_to_markdown=lambda: "# Hello\n\nHello, world!"),
        ),
    ):
        with sample.open("rb") as f:
            result = converter.convert_file(f, "hello.txt")

    assert "Hello, world!" in result


def test_docling_converter_url(settings, tmp_path):
    converter = DoclingConverter()
    html = b"<html><body><h1>Title</h1><p>Body text.</p></body></html>"

    with patch("doc_etl_api.pipeline.requests.get") as mock_get:
        mock_get.return_value = MagicMock(
            url="https://example.com/page",
            content=html,
            raise_for_status=MagicMock(),
        )
        with patch.object(
            converter._converter,
            "convert",
            return_value=MagicMock(
                status=ConversionStatus.SUCCESS,
                document=MagicMock(export_to_markdown=lambda: "# Title\n\nBody text."),
            ),
        ):
            markdown, final_url = converter.convert_url("https://example.com/page")

    assert "# Title" in markdown
    assert final_url == "https://example.com/page"


# The chrome around the main content of the page used by the extraction tests
# below. `NAV` carries a heading, which is what defeats Docling's own
# boilerplate heuristic, so it reaches the markdown unless it is extracted away.
_URL_BODY = " ".join(
    f"Sentence {index} describes the widget frame and its latch clearance."
    for index in range(1, 26)
)
_URL_PAGE = (
    "<html><body>"
    '<nav><h2>Browse</h2><a href="/">Home</a><a href="/docs">Docs</a></nav>'
    '<div class="promo"><h3>Buy now</h3><p>Save 20% this week.</p></div>'
    f'<main><h1>Widget Handbook</h1><p>See <a href="/pricing">our pricing</a> first.</p>'
    f"<h2>Assembly</h2><p>{_URL_BODY}</p></main>"
    "<footer><p>&copy; 2026 Widget Co.</p></footer>"
    "</body></html>"
)


def fetch(converter, *, url: str = "https://example.com/page", html: str = _URL_PAGE):
    """Convert *html* through `convert_url`, standing in for the HTTP fetch."""
    with patch("doc_etl_api.pipeline.requests.get") as mock_get:
        mock_get.return_value = MagicMock(
            url=url, content=html.encode(), raise_for_status=MagicMock()
        )
        return converter.convert_url("https://example.com/start")


def test_convert_url_indexes_only_the_main_content() -> None:
    markdown, final_url = fetch(DoclingConverter())

    assert final_url == "https://example.com/page"
    assert "Widget Handbook" in markdown
    assert "Assembly" in markdown
    assert "Sentence 1 describes" in markdown
    assert "Browse" not in markdown, "navigation reached the markdown"
    assert "Buy now" not in markdown, "promotional banner reached the markdown"
    assert "Widget Co" not in markdown, "footer reached the markdown"


def _pipeline_for_fetch(user_agent: str) -> tuple[IndexPipeline, DoclingConverter]:
    """A pipeline with a real converter, so the outgoing request can be observed."""
    converter = DoclingConverter()
    pipeline = IndexPipeline(
        Settings(
            embedding_model=EMBEDDING_MODEL_NAME,
            chunk_overlap=10,
            user_agent=user_agent,
        ),
        converter=converter,
        embedding_model=StubEmbedding(embed_dim=8),
    )
    return pipeline, converter


def _ingest_url_observing_the_fetch(user_agent: str) -> str | None:
    """Ingest a URL with a stubbed fetch, and report the `User-Agent` it sent."""
    pipeline, converter = _pipeline_for_fetch(user_agent)
    conversion = MagicMock(
        status=ConversionStatus.SUCCESS,
        document=MagicMock(export_to_markdown=lambda: "# Widget Handbook\n\nBody."),
    )

    with patch("doc_etl_api.pipeline.requests.get") as mock_get:
        mock_get.return_value = MagicMock(
            url="https://example.com/page",
            content=_URL_PAGE.encode(),
            raise_for_status=MagicMock(),
        )
        with patch.object(converter._converter, "convert", return_value=conversion):
            pipeline.ingest_url(source_id="source-1", url="https://example.com/start")

    _, kwargs = mock_get.call_args
    return (kwargs.get("headers") or {}).get("User-Agent")


def test_url_ingestion_fetches_with_the_configured_user_agent() -> None:
    configured = "doc-etl-api/0.1.0 (contact: ops@example.com)"

    assert _ingest_url_observing_the_fetch(configured) == configured


def test_the_fetch_never_sends_the_http_library_default() -> None:
    """Regression: the library's own user agent is what such hosts refuse.

    A fetch that sent no header at all would be refused the same way, so the
    absent case is asserted explicitly rather than allowed to pass by comparing
    unequal to the library default.
    """
    library_default = requests.utils.default_user_agent()

    for configured, expected in (
        (DEFAULT_USER_AGENT, DEFAULT_USER_AGENT),
        ("   ", DEFAULT_USER_AGENT),
        (
            "doc-etl-api/0.1.0 (contact: ops@example.com)",
            "doc-etl-api/0.1.0 (contact: ops@example.com)",
        ),
    ):
        sent = _ingest_url_observing_the_fetch(configured)

        assert sent is not None, f"no user agent was sent for {configured!r}"
        assert sent != library_default, f"the library default was sent for {configured!r}"
        assert sent == expected


def test_convert_url_excludes_navigation_that_carries_a_heading() -> None:
    """Regression: a heading inside the navigation used to defeat extraction.

    Docling's boilerplate heuristic labels everything before the first heading as
    furniture, so a heading anywhere in the page chrome flips the whole document
    to body content. Measured against the pre-change `convert_url`, this page
    reached the markdown with its navigation and promotional banner intact; both
    assertions below failed.
    """
    markdown, _ = fetch(DoclingConverter())

    assert "Browse" not in markdown, "a heading in the chrome defeated extraction"
    assert "Buy now" not in markdown, "a heading in the chrome defeated extraction"


def test_convert_url_markdown_links_are_urls() -> None:
    """Asserted platform-independently: the leak is `\\pricing` on Windows and
    `/pricing` elsewhere, and only the first looks obviously wrong.
    """
    markdown, _ = fetch(DoclingConverter())

    targets = re.findall(r"\]\(([^)]+)\)", markdown)
    assert targets, "the fixture is supposed to carry a link"
    for target in targets:
        assert target.startswith(("http://", "https://")), f"link target is not a URL: {target}"


def test_convert_url_resolves_relative_links_against_the_final_url() -> None:
    """A root-relative href must reach the markdown as an absolute URL.

    The page redirects, and the link is resolved against where it landed rather
    than against the URL that was requested.
    """
    markdown, final_url = fetch(DoclingConverter(), url="https://example.com/guides/page")

    assert final_url == "https://example.com/guides/page"
    assert "https://example.com/pricing" in markdown
    assert "\\pricing" not in markdown, "a link target was rewritten as a filesystem path"


def test_index_pipeline_ingest_file(settings):
    pipeline = IndexPipeline(settings)
    source_id = "source-file-1"

    with patch.object(
        pipeline.converter,
        "convert_file",
        return_value="# Doc\n\nThis is the content.",
    ):
        with patch.object(pipeline._index, "insert_nodes") as mock_insert:
            result, timings = pipeline.ingest_file(
                source_id=source_id,
                file=BytesIO(b"content"),
                filename="doc.txt",
                mime_type="text/plain",
            )

    assert result["source_id"] == source_id
    assert result["source_type"] == "file"
    assert result["name"] == "doc.txt"
    assert result["mime_type"] == "text/plain"
    assert result["status"] == "indexed"
    assert "parse_ms" in timings
    assert "chunk_ms" in timings
    assert "embed_ms" in timings
    assert "index_ms" in timings
    mock_insert.assert_called_once()


def test_index_pipeline_ingest_url(settings):
    pipeline = IndexPipeline(settings)
    source_id = "source-url-1"

    with patch.object(
        pipeline.converter,
        "convert_url",
        return_value=("# Page\n\nContent from URL.", "https://example.com/final"),
    ):
        with patch.object(pipeline._index, "insert_nodes") as mock_insert:
            result, timings = pipeline.ingest_url(
                source_id=source_id,
                url="https://example.com/page",
            )

    assert result["source_id"] == source_id
    assert result["source_type"] == "url"
    assert result["name"] == "https://example.com/page"
    assert result["final_url"] == "https://example.com/final"
    assert result["status"] == "indexed"
    assert "parse_ms" in timings
    assert "chunk_ms" in timings
    assert "embed_ms" in timings
    assert "index_ms" in timings
    mock_insert.assert_called_once()


@pytest.fixture
def searchable_pipeline(settings):
    """A pipeline over a real vector store with a deterministic embed model."""
    return IndexPipeline(
        settings,
        converter=MagicMock(),
        embedding_model=StubEmbedding(embed_dim=8),
    )


def test_concurrent_ingestion_keeps_every_source_searchable(searchable_pipeline):
    """Concurrent indexing must not drop either source's chunks."""

    def convert(file, filename):
        label = filename.removesuffix(".txt").removeprefix("doc")
        return f"# Document {label}\n\nUnique content for source {label}."

    searchable_pipeline.converter.convert_file.side_effect = convert

    def ingest(index: int) -> None:
        searchable_pipeline.ingest_file(
            source_id=f"source-{index}",
            file=BytesIO(f"doc {index}".encode()),
            filename=f"doc{index}.txt",
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(ingest, [1, 2]))

    results = searchable_pipeline.search("Unique content", top_k=10)
    assert {r["source_id"] for r in results} == {"source-1", "source-2"}


def test_search_during_ingestion_returns_well_formed_results(searchable_pipeline):
    """A search overlapping an index write must not raise or return junk."""
    ingest_started = threading.Event()
    allow_ingest_to_finish = threading.Event()

    def convert(file, filename):
        ingest_started.set()
        assert allow_ingest_to_finish.wait(timeout=5)
        return "# Doc\n\nSome content."

    searchable_pipeline.converter.convert_file.side_effect = convert

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(
            searchable_pipeline.ingest_file,
            source_id="source-1",
            file=BytesIO(b"content"),
            filename="doc.txt",
        )
        assert ingest_started.wait(timeout=5)
        results = searchable_pipeline.search("Some content", top_k=5)
        allow_ingest_to_finish.set()
        future.result(timeout=5)

    assert isinstance(results, list)
    for result in results:
        assert set(result) == {"text", "score", "source_id", "source_type", "source_name"}


def test_index_counters_track_sources_and_chunks(searchable_pipeline):
    """The readiness counters must match what was actually indexed."""
    searchable_pipeline.converter.convert_file.side_effect = lambda file, filename: (
        "# Doc\n\nSome content."
    )

    assert searchable_pipeline.indexed_sources == 0
    assert searchable_pipeline.indexed_chunks == 0

    for index in (1, 2):
        searchable_pipeline.ingest_file(
            source_id=f"source-{index}",
            file=BytesIO(b"content"),
            filename=f"doc{index}.txt",
        )

    assert searchable_pipeline.indexed_sources == 2
    # Each source is a single short paragraph, well under the 128-token
    # chunk size, so one node per source.
    assert searchable_pipeline.indexed_chunks == 2


def test_default_chunk_size_follows_the_embedding_model_limit():
    """An unset CHUNK_SIZE must track the model, not a constant that can go stale."""
    model = StubEmbedding(embed_dim=8)

    pipeline = _stub_pipeline(chunk_size=None, embedding_model=model)

    assert pipeline.chunk_size == model.max_length


def test_chunk_size_above_the_model_limit_is_rejected():
    """Oversized chunks are refused at startup rather than truncated at ingestion."""
    with pytest.raises(ValueError) as error:
        _stub_pipeline(chunk_size=EMBEDDING_MAX_TOKENS + 1)

    message = str(error.value)
    assert str(EMBEDDING_MAX_TOKENS) in message, "the error must name the limit"
    assert EMBEDDING_MODEL_NAME in message, "the error must name the model"


def test_embedding_model_without_a_stated_limit_is_rejected():
    """ "No limit" is a startup error, not a licence to guess a chunk size."""
    with pytest.raises(ValueError, match="no maximum input length"):
        _stub_pipeline(chunk_size=None, embedding_model=MockEmbedding(embed_dim=8))


def test_embedding_model_without_a_tokenizer_is_rejected():
    """Chunk size cannot be measured in a vocabulary the pipeline cannot reach."""

    class LimitOnly(MockEmbedding):
        max_length: int = EMBEDDING_MAX_TOKENS

    with pytest.raises(ValueError, match="exposes no tokenizer"):
        _stub_pipeline(chunk_size=None, embedding_model=LimitOnly(embed_dim=8))


def test_splitter_counts_tokens_the_way_the_embedding_model_does():
    """Chunk size must be counted in the vocabulary the embedder actually reads."""
    _stub_pipeline(chunk_size=None)
    sample = "The quick brown fox jumps over the lazy dog."
    count = LlamaSettings.node_parser._tokenizer

    # The model adds its own special tokens, and its window covers them, so the
    # count the splitter budgets with has to include them.
    assert len(count(sample)) == _model_tokens(sample)
    # ...which is not the count the pipeline used before: tiktoken measured a
    # different, unrelated vocabulary against a window that belongs to this one.
    tiktoken_count = len(tiktoken.encoding_for_model("gpt-3.5-turbo").encode(sample))
    assert len(count(sample)) != tiktoken_count


def test_ingestion_reports_the_chunking_it_achieved():
    """The reported outcome must describe the nodes actually produced."""
    pipeline = _stub_pipeline(chunk_size=128, chunk_overlap=32)
    pipeline.converter.convert_file.return_value = SENTENCE_DOC

    with patch.object(pipeline._index, "insert_nodes") as mock_insert:
        result, _ = pipeline.ingest_file(
            source_id="source-1", file=BytesIO(b"content"), filename="doc.txt"
        )

    nodes = mock_insert.call_args.args[0]
    assert len(nodes) > 1, "the document must split for this test to mean anything"

    chunking = result["chunking"]
    assert chunking["nodes"] == len(nodes)
    assert chunking["max_node_tokens"] == max(_model_tokens(n.get_content()) for n in nodes)
    assert chunking["overlap_tokens"] == min(
        _boundary_overlap(earlier.get_content(), later.get_content())
        for earlier, later in zip(nodes, nodes[1:], strict=False)
    )
    assert chunking["overlap_tokens"] > 0, "packed sentences must overlap"


def test_overlap_is_reported_as_achieved_not_as_configured():
    """Overlap is quantized to whole sentences, so a configured overlap can deliver
    none at all. The report must describe what happened, not what was asked for."""
    # 8 tokens of overlap cannot hold a whole sentence, which is ~14 tokens, so the
    # splitter repeats none of the previous node. The chunk size is 128 rather than
    # 64 so that the node floor stays out of the way: at 64 the splitter emits
    # ~30-token nodes, every one of them below the floor, so the nodes this asserts
    # on would be merged blobs instead of the splitter's own output.
    pipeline = _stub_pipeline(chunk_size=128, chunk_overlap=8)
    pipeline.converter.convert_file.return_value = SENTENCE_DOC

    result, _ = pipeline.ingest_file(
        source_id="source-1", file=BytesIO(b"content"), filename="doc.txt"
    )

    assert result["chunking"]["nodes"] > 1
    assert result["chunking"]["floor_merges"] == 0, "the floor changed the nodes under test"
    assert result["chunking"]["overlap_tokens"] == 0


def test_overlap_is_reported_as_unset_when_there_is_no_boundary():
    """A single-node source has no adjacent pair, so it reports no overlap."""
    pipeline = _stub_pipeline(chunk_size=None, chunk_overlap=10)
    pipeline.converter.convert_file.return_value = "One short paragraph."

    result, _ = pipeline.ingest_file(
        source_id="source-1", file=BytesIO(b"content"), filename="doc.txt"
    )

    assert result["chunking"]["nodes"] == 1
    assert result["chunking"]["overlap_tokens"] is None


def test_per_source_log_line_carries_the_chunking_outcome(caplog):
    """A sizing mismatch must be visible in the logs without polling the API."""
    pipeline = _stub_pipeline(chunk_size=64, chunk_overlap=8)
    pipeline.converter.convert_file.return_value = SENTENCE_DOC
    pipeline.converter.convert_url.return_value = (SENTENCE_DOC, "https://example.com/final")

    with caplog.at_level(logging.INFO):
        file_result, _ = pipeline.ingest_file(
            source_id="source-1", file=BytesIO(b"content"), filename="doc.txt"
        )
        url_result, _ = pipeline.ingest_url(source_id="source-2", url="https://example.com/page")

    messages = [record.getMessage() for record in caplog.records]
    file_line = next(m for m in messages if m.startswith("Indexed file"))
    url_line = next(m for m in messages if m.startswith("Indexed URL"))

    for line, result in ((file_line, file_result), (url_line, url_result)):
        chunking = result["chunking"]
        assert "chunking=" in line
        assert f"'nodes': {chunking['nodes']}" in line
        assert f"'max_node_tokens': {chunking['max_node_tokens']}" in line


def test_no_node_exceeds_the_embedding_model_input_limit():
    """Regression: chunks were measured in tiktoken against a 512 default while the
    embedder stops at 256, so the tail of every chunk was discarded unembedded."""
    pipeline = _stub_pipeline(chunk_size=None, chunk_overlap=50)
    pipeline.converter.convert_file.return_value = LONG_DOC

    with patch.object(pipeline._index, "insert_nodes") as mock_insert:
        result, _ = pipeline.ingest_file(
            source_id="source-1", file=BytesIO(b"content"), filename="doc.txt"
        )

    nodes = mock_insert.call_args.args[0]
    assert len(nodes) > 1

    sizes = [_model_tokens(node.get_content()) for node in nodes]
    assert max(sizes) <= EMBEDDING_MAX_TOKENS
    assert result["chunking"]["max_node_tokens"] == max(sizes)


def test_content_deep_inside_a_document_reaches_its_embedding():
    """Regression: the passage sat past the end of the model's window, so it reached
    a node whose text carried it but whose embedding did not represent it -- stored,
    returned by search for its neighbours, and findable by nothing."""
    pipeline = IndexPipeline(
        Settings(
            vector_store_backend="simple",
            embedding_model=EMBEDDING_MODEL_NAME,
            chunk_size=None,
            chunk_overlap=50,
            default_top_k=5,
        ),
        converter=MagicMock(),
    )
    pipeline.converter.convert_file.return_value = LONG_DOC

    # `wraps` lets the insert run for real -- so the search below runs against what
    # was stored -- while still handing back the nodes as they were embedded.
    with patch.object(
        pipeline._index, "insert_nodes", wraps=pipeline._index.insert_nodes
    ) as mock_insert:
        result, _ = pipeline.ingest_file(
            source_id="source-1", file=BytesIO(b"content"), filename="doc.txt"
        )
    assert result["chunking"]["nodes"] > 1

    nodes = mock_insert.call_args.args[0]
    carrying = next(node for node in nodes if LATE_MARKER in node.get_content())
    content = carrying.get_content()

    # Nothing of the node was discarded: what was stored is the embedding of the
    # node's whole text, not of the first window's worth of it.
    whole = pipeline._embedding_model.get_text_embedding(content)
    assert _cosine(carrying.embedding, whole) > 0.999999

    # ...which is what makes the passage findable: it lies inside the part of the
    # node the model actually read.
    offset = _model_tokens(content[: content.index(LATE_MARKER)])
    assert offset + _model_tokens(LATE_MARKER) <= EMBEDDING_MAX_TOKENS

    results = pipeline.search(LATE_MARKER, top_k=10)
    assert any(LATE_MARKER in item["text"] for item in results)


# --- Ingestion identity: re-ingesting a source replaces it rather than duplicating ---

# A paragraph both documents carry verbatim. The sentences are distinct from one
# another on purpose: the splitter repeats whole trailing sentences across a node
# boundary, so a paragraph built from one repeated sentence would have its opening
# present in every node that follows -- there would be no way to tell a source
# that had lost its first node from one that had not.
SHARED_SENTENCES = (
    "Shared opening material carried by two different documents in this test.",
    "Second shared sentence that both of the documents keep in common here.",
    "Third shared sentence present in each of the two documents under test.",
    "Fourth shared sentence that the two documents hold identically as well.",
    "Fifth shared sentence closing the paragraph both documents carry along.",
)
SHARED_PARAGRAPH = " ".join(SHARED_SENTENCES)
SHARED_OPENING = SHARED_SENTENCES[0]
ALPHA_PARAGRAPH = "Alpha-specific material that appears in no other document here. " * 3
BETA_PARAGRAPH = "Beta-specific material that appears in no other document here. " * 3
# The splitter's own paragraph break, so the shared paragraph is never packed
# together with the source-specific paragraph that follows it.
PARAGRAPH_GAP = "\n\n\n"


def _stored_nodes(pipeline) -> list:
    """The nodes the index currently holds, read through the docstore.

    `SimpleVectorStore` keeps only embeddings -- `get_nodes` on it raises -- so
    the node text lives in the index's docstore.
    """
    return list(pipeline.index.storage_context.docstore.docs.values())


def _ingest_text(pipeline, filename: str, text: str) -> None:
    pipeline.converter.convert_file.return_value = text
    pipeline.ingest_file(
        source_id=f"source-{filename}", file=BytesIO(b"content"), filename=filename
    )


def test_re_ingesting_a_file_reuses_its_document_identity(searchable_pipeline):
    """A file's identity is derived from its name, not minted per ingestion."""
    _ingest_text(searchable_pipeline, "doc.txt", "# Doc\n\nSome content.")
    first = set(searchable_pipeline.index.ref_doc_info)

    _ingest_text(searchable_pipeline, "doc.txt", "# Doc\n\nSome content.")

    assert first == {"doc.txt"}
    assert set(searchable_pipeline.index.ref_doc_info) == first


def test_urls_landing_on_one_page_share_a_document_identity(searchable_pipeline):
    """A page's identity is where it finally landed, not what was requested.

    Two different request URLs that redirect to one page are one source.
    """
    searchable_pipeline.converter.convert_url.return_value = (
        "# Page\n\nSome content.",
        "https://example.com/final",
    )

    for submitted in ("https://example.com/a", "https://example.com/b"):
        searchable_pipeline.ingest_url(source_id=submitted, url=submitted)

    assert set(searchable_pipeline.index.ref_doc_info) == {"https://example.com/final"}


def test_every_stored_node_carries_its_sources_identity(searchable_pipeline):
    """Nothing is stored that cannot later be located and replaced."""
    _ingest_text(searchable_pipeline, "doc.txt", SENTENCE_DOC)

    identities = set(searchable_pipeline.index.ref_doc_info)
    nodes = _stored_nodes(searchable_pipeline)

    assert nodes
    assert {node.ref_doc_id for node in nodes} == identities


def test_node_ids_are_stable_across_re_ingestion(searchable_pipeline):
    """Re-parsing the same source must produce the same node identities.

    LlamaIndex mints a random id per node, so without a derived one every
    ingestion is made of nodes the index has never seen.
    """
    _ingest_text(searchable_pipeline, "doc.txt", SENTENCE_DOC)
    first = {node.node_id for node in _stored_nodes(searchable_pipeline)}
    assert len(first) > 1, "the document must split for this test to mean anything"

    _ingest_text(searchable_pipeline, "doc.txt", SENTENCE_DOC)

    assert {node.node_id for node in _stored_nodes(searchable_pipeline)} == first


def test_two_sources_sharing_text_keep_their_own_copies(searchable_pipeline):
    """Sharing a paragraph must not make two sources share a node."""
    _ingest_text(
        searchable_pipeline, "alpha.txt", SHARED_PARAGRAPH + PARAGRAPH_GAP + ALPHA_PARAGRAPH
    )
    _ingest_text(searchable_pipeline, "beta.txt", SHARED_PARAGRAPH + PARAGRAPH_GAP + BETA_PARAGRAPH)

    by_source: dict[str, list] = {}
    for node in _stored_nodes(searchable_pipeline):
        if SHARED_OPENING in node.get_content():
            by_source.setdefault(node.ref_doc_id, []).append(node)

    # Both sources kept their own nodes rather than sharing one entry between
    # them, and the nodes carrying the shared paragraph hold the same text --
    # which is what a node id derived from text would have collapsed into a
    # single entry the two of them claim.
    assert set(by_source) == {"alpha.txt", "beta.txt"}
    alpha_texts = [node.get_content() for node in by_source["alpha.txt"]]
    beta_texts = [node.get_content() for node in by_source["beta.txt"]]
    assert alpha_texts == beta_texts


def test_re_ingesting_an_edited_source_removes_the_previous_version(searchable_pipeline):
    """A corrected document must not leave the version it replaced in the index."""
    _ingest_text(searchable_pipeline, "doc.txt", ALPHA_PARAGRAPH)
    _ingest_text(searchable_pipeline, "doc.txt", BETA_PARAGRAPH)

    stored = [node.get_content() for node in _stored_nodes(searchable_pipeline)]

    assert stored
    assert any("Beta-specific" in text for text in stored)
    assert not any("Alpha-specific" in text for text in stored), "the superseded version survived"


def test_a_source_that_was_never_indexed_ingests_normally(searchable_pipeline):
    """Replacement removes what is there; on a first ingestion there is nothing
    to remove, which must be a no-op rather than an error."""
    assert searchable_pipeline.index.ref_doc_info == {}

    _ingest_text(searchable_pipeline, "doc.txt", ALPHA_PARAGRAPH)

    assert searchable_pipeline.indexed_sources == 1
    assert _stored_nodes(searchable_pipeline)


def test_editing_one_source_does_not_disturb_another_sharing_its_text(searchable_pipeline):
    """The reason node identity is scoped to its source rather than its text.

    With text-addressed node ids the shared opening is an entry both sources
    claim, so the deletion below removes it on behalf of the source that no
    longer carries it -- taking it away from the source that still does. The
    check is on the shared paragraph's opening rather than on the paragraph
    appearing somewhere: a source can lose its first node and still have the
    shared text in the node that follows, which is exactly how this failure
    hides.
    """
    _ingest_text(
        searchable_pipeline, "alpha.txt", SHARED_PARAGRAPH + PARAGRAPH_GAP + ALPHA_PARAGRAPH
    )
    _ingest_text(searchable_pipeline, "beta.txt", SHARED_PARAGRAPH + PARAGRAPH_GAP + BETA_PARAGRAPH)

    # Beta is corrected so that it no longer carries the shared paragraph.
    _ingest_text(searchable_pipeline, "beta.txt", BETA_PARAGRAPH)

    alpha_holds_opening = any(
        node.ref_doc_id == "alpha.txt" and SHARED_OPENING in node.get_content()
        for node in _stored_nodes(searchable_pipeline)
    )

    assert alpha_holds_opening, "beta's edit deleted alpha's copy of the shared text"


def test_counters_describe_stored_content_not_submitted_content(searchable_pipeline):
    """Re-ingesting unchanged content must not inflate the readiness counts."""
    _ingest_text(searchable_pipeline, "doc.txt", SENTENCE_DOC)
    sources, chunks = searchable_pipeline.indexed_sources, searchable_pipeline.indexed_chunks
    assert (sources, chunks) == (1, len(_stored_nodes(searchable_pipeline)))

    for _ in range(2):
        _ingest_text(searchable_pipeline, "doc.txt", SENTENCE_DOC)

    assert searchable_pipeline.indexed_sources == sources
    assert searchable_pipeline.indexed_chunks == chunks
    assert len(_stored_nodes(searchable_pipeline)) == chunks


def test_ingesting_one_source_repeatedly_stores_one_copy(searchable_pipeline):
    """Regression: every ingestion used to index a fresh set of chunks.

    Measured against the pre-change pipeline, ingesting this document three
    times stored three times its chunks and the assertion below failed.
    """
    _ingest_text(searchable_pipeline, "doc.txt", SENTENCE_DOC)
    once = len(_stored_nodes(searchable_pipeline))
    assert once > 1, "the document must split for this test to mean anything"

    for _ in range(2):
        _ingest_text(searchable_pipeline, "doc.txt", SENTENCE_DOC)

    assert len(_stored_nodes(searchable_pipeline)) == once


class _ContentEmbedding(StubEmbedding):
    """Embeds identical text identically, as a real model does.

    `MockEmbedding` embeds at random, so two copies of one chunk would receive
    different vectors and would not rank alike -- which is what hides a
    duplicated ingestion from a retrieval test.
    """

    def _get_text_embedding(self, text: str) -> list[float]:
        return [byte / 255.0 for byte in hashlib.sha256(text.encode()).digest()[:8]]


def test_duplicate_ingestion_does_not_consume_search_result_slots(settings):
    """Duplicates crowd out distinct results, because copies of one chunk score alike."""
    pipeline = IndexPipeline(
        settings,
        converter=MagicMock(),
        embedding_model=_ContentEmbedding(embed_dim=8),
    )
    for _ in range(3):
        _ingest_text(pipeline, "doc.txt", SENTENCE_DOC)

    results = pipeline.search("Sentence 5 covers its own separate subject", top_k=5)
    texts = [result["text"] for result in results]

    assert len(texts) > 1
    assert len(texts) == len(set(texts)), "a duplicated chunk consumed a result slot"


# --- Chunk coherence: tables as blocks, the node floor, repeated text ---

# An infobox shaped like the ones on the corpus this change was measured against:
# a short header, a separator, and rows whose values say nothing without the
# column they belong to. Measured at 64 tokens, so it fits one node at the chunk
# sizes used below.
INFOBOX_HEADER = "| | Thư tịch | Không rõ |"
INFOBOX = "\n".join(
    [
        INFOBOX_HEADER,
        "| --- | --- | --- |",
        "| Loại hình | Kiếm pháp |",
        "| Người sáng tạo | Độc Cô Cầu Bại |",
        "| Xuất hiện | Thần điêu hiệp lữ |",
    ]
)
INFOBOX_VALUE = "Độc Cô Cầu Bại"
TABLE_SEPARATOR = "| --- | --- | --- |"
# 28 tokens: below the 32-token floor, which is what makes it a fragment.
SMALL_TABLE_HEADER = "| Ghi chú | Ngắn |"
SMALL_TABLE = "\n".join([SMALL_TABLE_HEADER, "| --- | --- |", "| Xem thêm | Phụ lục |"])
# 992 tokens, so it cannot fit one node at any chunk size used here.
WIDE_TABLE_HEADER = "| Tên | Mô tả |"
WIDE_TABLE_SEPARATOR = "| --- | --- |"
WIDE_TABLE = "\n".join(
    [WIDE_TABLE_HEADER, WIDE_TABLE_SEPARATOR]
    + [
        f"| Chiêu thức {index} | Mô tả chi tiết về chiêu thức số {index} trong bộ võ học |"
        for index in range(1, 40)
    ]
)
# 266 tokens, so it splits into more than one node and none of them is a fragment.
PROSE = " ".join(
    f"Sentence {index} records the ordinary course of matter {index} here."
    for index in range(1, 25)
)
# Docling pads a table out to the width of its widest cell, and a dash run costs
# the tokenizer about a token per dash. Measured on the corpus: a fourteen-row
# table's separator row was a 1169-character dash run, 1171 tokens against a
# 256-token window. This fixture reproduces that shape.
PADDED_TABLE_HEADER = "| Tên | Mô tả |"
PADDED_SEPARATOR = f"| {'-' * 350} | {'-' * 350} |"
PADDED_TABLE = "\n".join(
    [PADDED_TABLE_HEADER, PADDED_SEPARATOR]
    + [
        f"| Chiêu thức {index} | Mô tả chi tiết về chiêu thức số {index} trong bộ võ học |"
        for index in range(1, 40)
    ]
)
# One row wider than the window on its own, so the table cannot be split at row
# boundaries alone: the row itself has to be cut, beside the repeated header.
LONG_ROW_TABLE = "\n".join(
    [
        PADDED_TABLE_HEADER,
        PADDED_SEPARATOR,
        "| Chiêu thức dài | "
        + " ".join(f"mô tả thứ {index} của chiêu thức" for index in range(1, 60))
        + " |",
    ]
)
# Short enough to be a single node at the chunk sizes below and above the floor,
# so a repeat of it is exactly two byte-identical nodes.
REPEATED_PARAGRAPH = " ".join(
    f"Clause {index} keeps its own separate wording about matter {index} alone."
    for index in range(1, 6)
)


def _ingest_markdown(pipeline, markdown: str, filename: str = "page.txt") -> dict:
    """Ingest *markdown* as *filename* and return the reported chunking outcome."""
    pipeline.converter.convert_file.return_value = markdown
    result, _ = pipeline.ingest_file(
        source_id=f"source-{filename}", file=BytesIO(b"content"), filename=filename
    )
    return result["chunking"]


def test_a_table_is_chunked_as_a_block_not_as_a_run_of_sentences():
    """A table has no sentences, so the sentence packer cut it mid-row.

    Measured against the pre-change chunking, a document shaped like this one
    reached the index as rows detached from their header and cut mid-cell -- one
    stored node began ``thức 17 | Mô tả chi tiết...``, with the row's own label
    left behind in the node before it.
    """
    pipeline = _stub_pipeline(chunk_size=128, chunk_overlap=0)
    _ingest_markdown(pipeline, PROSE + PARAGRAPH_GAP + INFOBOX + PARAGRAPH_GAP + PROSE)

    stored = [node.get_content() for node in _stored_nodes(pipeline)]

    assert any(text.strip() == INFOBOX for text in stored), "the table was not stored as a block"
    fragments = [text for text in stored if "|" in text and INFOBOX_HEADER not in text]
    assert not fragments, f"table text was stored without its header: {fragments[0][:60]!r}"


def test_a_table_that_fits_becomes_one_node_beginning_with_its_header():
    pipeline = _stub_pipeline(chunk_size=128, chunk_overlap=0)

    chunking = _ingest_markdown(pipeline, INFOBOX)

    stored = [node.get_content() for node in _stored_nodes(pipeline)]
    assert chunking["nodes"] == 1
    assert len(stored) == 1
    assert stored[0].startswith(INFOBOX_HEADER), "the node does not begin with the header row"
    assert stored[0].splitlines()[1] == TABLE_SEPARATOR, "the separator row is missing"
    assert INFOBOX_VALUE in stored[0]


def test_a_table_larger_than_the_chunk_size_keeps_its_header_in_every_part():
    pipeline = _stub_pipeline(chunk_size=128, chunk_overlap=0)

    chunking = _ingest_markdown(pipeline, WIDE_TABLE)

    stored = [node.get_content() for node in _stored_nodes(pipeline)]
    assert len(stored) > 1, "the table must split for this test to mean anything"
    assert len(stored) == chunking["nodes"]
    for text in stored:
        lines = text.splitlines()
        assert lines[0] == WIDE_TABLE_HEADER, "a part lost the header row"
        assert lines[1] == WIDE_TABLE_SEPARATOR, "a part lost the separator row"
        # Cut at row boundaries: no row is divided, so no cell is orphaned from
        # the row it belongs to.
        for line in lines:
            assert line.startswith("|") and line.endswith("|"), f"a row was cut mid-cell: {line!r}"
    assert chunking["max_node_tokens"] <= EMBEDDING_MAX_TOKENS


def test_a_table_padded_out_to_its_widest_cell_stays_within_the_model_window():
    """Regression: a repeated padded separator row pushed every part past the window.

    The header row of a table is small, but the separator row underneath it is as
    wide as the table's widest cell, and a dash run is not cheap to tokenize.
    Repeating that head at the top of every part therefore put each part past the
    window on its own, before a single row was added to it -- measured end to end
    against the corpus this change was tuned on, the pre-fix chunking stored
    nodes of up to 1277 tokens against a 256-token window, so the model never read
    the end of any of them.
    """
    pipeline = _stub_pipeline(chunk_size=128, chunk_overlap=0)
    assert _model_tokens(PADDED_SEPARATOR) > EMBEDDING_MAX_TOKENS, (
        "the fixture's separator row must breach the window on its own for this test to bite"
    )

    chunking = _ingest_markdown(pipeline, PADDED_TABLE)

    stored = [node.get_content() for node in _stored_nodes(pipeline)]
    assert len(stored) > 1, "the table must split for this test to mean anything"
    assert chunking["max_node_tokens"] <= EMBEDDING_MAX_TOKENS, "a node exceeded the model window"
    for text in stored:
        lines = text.splitlines()
        assert lines[0] == PADDED_TABLE_HEADER, "a part lost the header row"
        assert lines[1] == WIDE_TABLE_SEPARATOR, "the repeated separator was not shortened"
        assert _model_tokens(text) <= EMBEDDING_MAX_TOKENS, (
            "a stored node exceeded the model window"
        )


def test_a_row_cut_to_fit_keeps_the_characters_the_page_wrote():
    """Regression: the cut that bounded a wide row rewrote the text it kept.

    The cut decoded the token ids it kept and joined them back up, and a BERT
    tokenizer's decode does not run its encode backwards: measured on the corpus,
    a row's ``*Bích huyết kiếm (1956)*`` came back as ``* bich huyet kiem``, its
    accents stripped and its case lowered. Nothing is dropped by a slice, so the
    pieces are cut out of the row's own characters.
    """
    pipeline = _stub_pipeline(chunk_size=None, chunk_overlap=0)
    row = LONG_ROW_TABLE.splitlines()[2]

    pieces = _split_oversized_text(row, 40, pipeline._tokenizer)

    assert len(pieces) > 1, "the row must be cut for this test to mean anything"
    assert all(len(_content_token_ids(pipeline._tokenizer, piece)) <= 40 for piece in pieces), (
        "a piece exceeded the budget it was cut to"
    )
    assert "".join("".join(piece.split()) for piece in pieces) == "".join(row.split()), (
        "the cut dropped or reordered the row's characters"
    )
    assert "mô tả" in " ".join(pieces), "the cut lost the accents the page wrote"
    assert "mo ta" not in " ".join(pieces), "the cut stored text the page never had"


def test_a_cut_row_reaches_the_index_in_the_pages_own_spelling():
    """The row's own wording is what a query has to be able to find."""
    pipeline = _stub_pipeline(chunk_size=None, chunk_overlap=0)

    _ingest_markdown(pipeline, LONG_ROW_TABLE)

    stored = [node.get_content() for node in _stored_nodes(pipeline)]
    assert any("mô tả thứ" in text for text in stored), "the row's own wording is not indexed"
    assert not any("mo ta thu" in text for text in stored), "a node holds text the page never had"


def test_no_node_reaches_past_the_window_when_the_chunk_size_is_the_window():
    """The service's own configuration: CHUNK_SIZE unset, so it is the model's limit.

    The last cut a table part can need is the one that bounds a single row wider
    than the window, and it is made in content tokens while the limit counts the
    tokens the model adds to whatever it reads. Measured end to end, a part cut to
    that budget came back at 257 tokens against a 256-token window -- one token of
    it past what the model reads, which is to say truncated.
    """
    pipeline = _stub_pipeline(chunk_size=None, chunk_overlap=0)
    assert pipeline.chunk_size == EMBEDDING_MAX_TOKENS, "the fixture must chunk at the window"
    assert _model_tokens(LONG_ROW_TABLE.splitlines()[2]) > EMBEDDING_MAX_TOKENS, (
        "the fixture's row must be wider than the window for this test to bite"
    )

    chunking = _ingest_markdown(pipeline, LONG_ROW_TABLE)

    stored = [node.get_content() for node in _stored_nodes(pipeline)]
    assert len(stored) > 1, "the row must be cut for this test to mean anything"
    assert chunking["max_node_tokens"] <= EMBEDDING_MAX_TOKENS, "a node exceeded the model window"
    for text in stored:
        assert text.splitlines()[0] == PADDED_TABLE_HEADER, "a part lost the header row"
        assert _model_tokens(text) <= EMBEDDING_MAX_TOKENS, (
            "a stored node exceeded the model window"
        )


def test_the_node_floor_defaults_to_32_and_honours_a_configured_value():
    assert Settings().min_chunk_tokens == 32

    def outcome(floor: int) -> dict:
        pipeline = _stub_pipeline(chunk_size=64, chunk_overlap=0, min_chunk_tokens=floor)
        return _ingest_markdown(pipeline, SENTENCE_DOC, filename="doc.txt")

    # At this chunk size the splitter emits ~30-token nodes, so a floor of 0
    # leaves every one of them alone while a higher floor folds them together.
    # The count is the only way to see that the configured value was applied.
    unfloored = outcome(0)
    floored = outcome(96)

    assert unfloored["floor_merges"] == 0
    assert unfloored["overlap_tokens"] is not None
    assert floored["floor_merges"] > 0
    assert floored["nodes"] < unfloored["nodes"]


def test_a_fragment_below_the_floor_is_merged_into_a_neighbour():
    """A node too small to answer anything is folded into the text beside it."""
    pipeline = _stub_pipeline(chunk_size=128, chunk_overlap=0)
    alone = _ingest_markdown(pipeline, PROSE, filename="prose.txt")
    together = _ingest_markdown(pipeline, PROSE + PARAGRAPH_GAP + SMALL_TABLE, filename="both.txt")

    assert _model_tokens(SMALL_TABLE) < 32, "the fixture must be below the floor"
    assert together["floor_merges"] == 1
    # The fragment's content joined a node rather than becoming one of its own.
    assert together["nodes"] == alone["nodes"]

    stored = [node.get_content() for node in _stored_nodes(pipeline)]
    assert not any(text.strip() == SMALL_TABLE for text in stored), "the fragment stood alone"
    assert any(SMALL_TABLE in text for text in stored), "the fragment's text was dropped"


def test_a_document_below_the_floor_is_stored_with_its_content_intact():
    pipeline = _stub_pipeline(chunk_size=128, chunk_overlap=0)

    chunking = _ingest_markdown(pipeline, "Ngắn.", filename="doc.txt")

    stored = [node.get_content() for node in _stored_nodes(pipeline)]
    assert chunking["nodes"] == 1
    assert len(stored) == 1
    assert "Ngắn." in stored[0], "content was discarded to satisfy the floor"
    assert chunking["floor_merges"] == 0


def test_a_merge_that_would_breach_the_model_window_is_refused():
    """The floor must not be satisfied by producing a node the model truncates.

    Measured before this guard existed, the floor folded a whole document into a
    single 842-token node against this 256-token window, discarding the tail of
    every chunk the model was handed -- the loss `doc-etl-api-chunk-sizing`
    exists to prevent.
    """
    pipeline = _stub_pipeline(chunk_size=64, chunk_overlap=0)

    chunking = _ingest_markdown(pipeline, SENTENCE_DOC, filename="doc.txt")

    assert chunking["floor_merges"] > 0, "the fixture must exercise the floor"
    assert chunking["max_node_tokens"] <= EMBEDDING_MAX_TOKENS, "a node exceeded the model window"

    # A fragment at the head of a document whose only neighbour is already at the
    # ceiling: neither merge is legal, so it is stored as it is rather than at
    # the cost of a node whose tail the model would silently truncate.
    pipeline = _stub_pipeline(chunk_size=None, chunk_overlap=0)

    chunking = _ingest_markdown(
        pipeline, SMALL_TABLE + PARAGRAPH_GAP + WIDE_TABLE, filename="doc.txt"
    )

    assert chunking["floor_refusals"] == 1
    assert chunking["max_node_tokens"] <= EMBEDDING_MAX_TOKENS
    stored = [node.get_content() for node in _stored_nodes(pipeline)]
    assert any(text.strip() == SMALL_TABLE for text in stored), "the fragment's text was dropped"
    assert any(_model_tokens(text) < 32 for text in stored), "a refusal is what leaves it there"


def test_text_repeated_within_one_source_is_stored_once():
    """A paragraph carried twice occupies one node, not one per occurrence.

    The infobox between the two copies keeps the splitter from packing them into
    one node, so the repetition here really is two byte-identical nodes.
    """
    pipeline = _stub_pipeline(chunk_size=128, chunk_overlap=0)

    chunking = _ingest_markdown(
        pipeline,
        PARAGRAPH_GAP.join([REPEATED_PARAGRAPH, INFOBOX, REPEATED_PARAGRAPH]),
    )

    assert chunking["duplicate_nodes"] == 1, "the repeated paragraph was stored twice"
    stored = [node.get_content() for node in _stored_nodes(pipeline)]
    assert len(stored) == chunking["nodes"]
    assert sum(1 for text in stored if text.strip() == REPEATED_PARAGRAPH) == 1

    # Positions stay contiguous after the collapse, so identity is still source
    # and position rather than source and position with a gap in it.
    assert {node.node_id for node in _stored_nodes(pipeline)} == {
        _node_id("page.txt", position) for position in range(len(stored))
    }


def test_identical_text_in_two_sources_is_stored_once_per_source():
    """The boundary that keeps collapsing repeats safe.

    Both sources hold the same infobox, so their nodes for it are byte-identical
    -- exactly what a collapse applied across sources would fold into one entry
    that both of them claim.
    """
    pipeline = _stub_pipeline(chunk_size=128, chunk_overlap=0)
    pipeline.converter.convert_file.return_value = INFOBOX

    for name in ("alpha.txt", "beta.txt"):
        pipeline.ingest_file(source_id=name, file=BytesIO(b"content"), filename=name)

    holders = {
        node.ref_doc_id for node in _stored_nodes(pipeline) if INFOBOX_HEADER in node.get_content()
    }
    assert holders == {"alpha.txt", "beta.txt"}, "a source lost its own copy of the shared text"
    assert pipeline.indexed_sources == 2
    assert pipeline.indexed_chunks == 2


def test_collapsing_repeats_across_sources_would_take_a_source_s_copy(monkeypatch):
    """Evidence for the boundary above rather than an assertion about it.

    The same ingestion with the collapse hoisted above the source: the second
    source stores nothing of text the first already holds, which is the deletion
    hazard `doc-etl-api-ingestion-identity` measured.
    """
    seen: set[str] = set()

    def across_sources(texts):
        distinct, duplicates = [], 0
        for text in texts:
            if text in seen:
                duplicates += 1
                continue
            seen.add(text)
            distinct.append(text)
        return distinct, duplicates

    monkeypatch.setattr("doc_etl_api.pipeline._dedupe_texts", across_sources)

    pipeline = _stub_pipeline(chunk_size=128, chunk_overlap=0)
    pipeline.converter.convert_file.return_value = INFOBOX
    for name in ("alpha.txt", "beta.txt"):
        pipeline.ingest_file(source_id=name, file=BytesIO(b"content"), filename=name)

    holders = {
        node.ref_doc_id for node in _stored_nodes(pipeline) if INFOBOX_HEADER in node.get_content()
    }
    assert holders == {"alpha.txt"}, "the fixture does not exercise a cross-source collapse"


def test_readme_documents_the_node_floor():
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")

    assert "| `MIN_CHUNK_TOKENS` |" in readme, "the setting is missing from the configuration table"
    # The escape hatch is what the setting to zero does under the boundaries this
    # change introduced: every unit of the document stored as its own node. It no
    # longer restores sentence packing, because nothing packs sentences any more.
    assert "merging nothing" in readme, "the escape hatch is not documented"
    assert Settings().min_chunk_tokens == 32


def test_readme_documents_the_user_agent():
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")

    assert "| `USER_AGENT` |" in readme, "the setting is missing from the configuration table"
    assert "contact details" in readme, "the reason to set it is not documented"
    assert Settings().resolved_user_agent == "doc-etl-api/0.1.0"


def test_readme_documents_collections():
    """A caller cannot use a scope the README does not describe.

    Named settings, the endpoints that take a collection, and the two rules a
    caller would otherwise discover by being refused: the cap on how many one
    request may name, and the rejection of an empty filter.
    """
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")

    assert "| `KNOWLEDGE_CORPUS_COLLECTIONS` |" in readme, (
        "the setting is missing from the configuration table"
    )
    assert "[Collections](#collections)" in readme, "the collections section is missing"
    assert "localhost:8000/sources" in readme, "the source catalog endpoint is not documented"
    assert "at most 16 collections" in readme, "the cap per request is not documented"
    assert '"collections": []' in readme, "the rejection of an empty filter is not documented"
    assert Settings().knowledge_corpus_collection_list == []


def test_the_chunking_outcome_reports_what_the_floor_and_the_collapse_did():
    """An ingestion that both merges a fragment and drops a repeat."""
    pipeline = _stub_pipeline(chunk_size=128, chunk_overlap=0)

    chunking = _ingest_markdown(
        pipeline,
        PARAGRAPH_GAP.join(
            [
                REPEATED_PARAGRAPH,  # kept, and carried again below
                INFOBOX,  # kept: a table block of its own
                REPEATED_PARAGRAPH,  # a repeat of the first
                SMALL_TABLE,  # below the floor: folded into the repeat above it
                REPEATED_PARAGRAPH,  # a repeat of the first again
            ]
        ),
    )

    assert chunking["floor_merges"] == 1, "the below-floor fragment was not merged"
    assert chunking["duplicate_nodes"] == 1, "the repeated paragraph was not collapsed"
    assert chunking["nodes"] == len(_stored_nodes(pipeline))


def test_a_source_needing_no_merging_reports_none():
    pipeline = _stub_pipeline(chunk_size=128, chunk_overlap=0)

    chunking = _ingest_markdown(pipeline, SENTENCE_DOC, filename="doc.txt")

    assert chunking["nodes"] > 1, "the document must split for this test to mean anything"
    assert chunking["floor_merges"] == 0
    assert chunking["floor_refusals"] == 0
    assert chunking["duplicate_nodes"] == 0


def test_the_measured_corpus_shape_no_longer_produces_scraps():
    """Regression for the measured failure: 27% of stored chunks under 32 tokens
    and 26% table rows detached from the header naming their columns.

    The comparison at the end runs the same document through the splitter alone
    -- the pre-change chunking -- so the failure being guarded against is shown
    rather than assumed.
    """
    document = PARAGRAPH_GAP.join([PROSE, INFOBOX, WIDE_TABLE, PADDED_TABLE])
    pipeline = _stub_pipeline(chunk_size=128, chunk_overlap=0)

    _ingest_markdown(pipeline, document)

    stored = [node.get_content() for node in _stored_nodes(pipeline)]
    assert min(_model_tokens(text) for text in stored) >= 32, "a node below the floor was stored"
    assert max(_model_tokens(text) for text in stored) <= EMBEDDING_MAX_TOKENS, (
        "a node the model cannot read whole was stored"
    )

    table_nodes = [text for text in stored if text.strip().startswith("|")]
    assert table_nodes, "the fixture is supposed to carry tables"
    for text in table_nodes:
        assert text.splitlines()[0] in (INFOBOX_HEADER, WIDE_TABLE_HEADER, PADDED_TABLE_HEADER), (
            f"a table node lost its header: {text[:60]!r}"
        )

    legacy = LlamaSettings.node_parser.get_nodes_from_documents(
        [
            LlamaDocument(
                text=document,
                id_="legacy.txt",
                metadata={
                    "source_id": "source-page.txt",
                    "source_type": "file",
                    "source_name": "legacy.txt",
                    "mime_type": None,
                },
            )
        ]
    )
    legacy_texts = [node.get_content() for node in legacy]
    assert any(_model_tokens(text) < 32 for text in legacy_texts), (
        "the pre-change chunking is supposed to store fragments"
    )
    assert any(
        text.strip().startswith("|")
        and INFOBOX_HEADER not in text
        and WIDE_TABLE_HEADER not in text
        for text in legacy_texts
    ), "the pre-change chunking is supposed to leave rows without a header"


def test_a_query_for_an_infobox_value_returns_a_node_carrying_its_header():
    """A row is answerable only if the node holding it also names its column.

    Uses the real embedding model -- a stub embeds at random, so it could not
    show that the value is what brings the node back -- and a chunk size small
    enough that the infobox cannot fit beside the prose around it, which is the
    shape the pre-change chunking cut in half.
    """
    pipeline = IndexPipeline(
        Settings(
            vector_store_backend="simple",
            embedding_model=EMBEDDING_MODEL_NAME,
            chunk_size=64,
            chunk_overlap=0,
            default_top_k=5,
        ),
        converter=MagicMock(),
    )
    _ingest_markdown(pipeline, PARAGRAPH_GAP.join([PROSE, INFOBOX, PROSE]))

    results = pipeline.search(INFOBOX_VALUE, top_k=5)

    assert results
    assert any(
        INFOBOX_VALUE in item["text"] and INFOBOX_HEADER in item["text"] for item in results
    ), "the infobox value came back without the header naming its column"


# --- Structural boundaries: the document's own sections and paragraphs --------
#
# A document of heading, section and paragraph, so the boundaries a node is
# allowed to have can be told apart from the ones a token count would place.


def _paragraph(label: str, sentences: int = 5) -> str:
    """A paragraph of *sentences* distinct sentences, labelled so it is findable.

    Measured against the stub model's tokenizer: one is 55 tokens, two pack into
    a 128-token node with room to spare and a third does not, so packing and the
    paragraph boundary can each be observed on their own.
    """
    return " ".join(
        f"Under {label}, remark {index} stands entirely on its own."
        for index in range(1, sentences + 1)
    )


HEADING_ONE = "## Niên biểu"
HEADING_TWO = "## Thư tịch"
# Below the 32-token floor however it is stored, so the floor has to act on it.
FRAGMENT = "Phụ lục."
# Also below the floor, but wide enough that folding it into a table part already
# at the model's window would breach the window: measured at 17 tokens against a
# part of 248, the merge reaches 266 against a 256-token window and is refused,
# leaving the merge across the heading as the only legal one. A five-token
# fragment does not do this: it fits beside the part, so the fragment's own
# section can take it and the crossing is never reached.
FRAGMENT_WIDE = " ".join(f"Phụ lục {index}." for index in range(1, 4))

PARA_ALPHA = _paragraph("Alpha")
PARA_BRAVO = _paragraph("Bravo")
PARA_CHARLIE = _paragraph("Charlie")
PARA_DELTA = _paragraph("Delta")
PARAGRAPHS = [PARA_ALPHA, PARA_BRAVO, PARA_CHARLIE, PARA_DELTA]
# Six paragraphs, so a 128-token cap divides one section into three nodes: the
# smallest shape in which a shared heading is carried by more than two of them.
PARAGRAPHS_SIX = [*PARAGRAPHS, _paragraph("Echo"), _paragraph("Foxtrot")]

# The paragraphs as the body of one section, and those sections under their heading.
PARAGRAPHS_JOINED = "\n\n".join(PARAGRAPHS)
PARAGRAPHS_SIX_JOINED = "\n\n".join(PARAGRAPHS_SIX)
SECTION_OF_PARAGRAPHS = f"{HEADING_ONE}\n\n{PARAGRAPHS_JOINED}"
SECTION_OF_SIX = f"{HEADING_ONE}\n\n{PARAGRAPHS_SIX_JOINED}"


def _paragraph_runs(paragraphs: Sequence[str]) -> set[str]:
    """Every node body a run of whole paragraphs could produce, in document order."""
    return {
        "\n\n".join(paragraphs[start:stop])
        for start in range(len(paragraphs))
        for stop in range(start + 1, len(paragraphs) + 1)
    }


def test_a_heading_opens_a_section_that_the_next_heading_closes():
    markdown = PARAGRAPH_GAP.join(
        [HEADING_ONE, PARA_ALPHA, PARA_BRAVO, HEADING_TWO, PARA_CHARLIE]
    )

    sections = _split_sections(markdown)

    assert [section.heading for section in sections] == [HEADING_ONE, HEADING_TWO]
    # Paragraphs inside one section are units of their own rather than one run.
    assert [unit.text for unit in sections[0].units] == [PARA_ALPHA, PARA_BRAVO]
    assert PARA_CHARLIE not in [unit.text for unit in sections[0].units], (
        "a unit of one section carried text from the section after it"
    )


def test_content_before_the_first_heading_is_a_section_of_its_own():
    sections = _split_sections(PARAGRAPH_GAP.join([PARA_ALPHA, HEADING_ONE, PARA_BRAVO]))

    assert [section.heading for section in sections] == [None, HEADING_ONE]
    assert [unit.text for unit in sections[0].units] == [PARA_ALPHA]
    assert [unit.text for unit in sections[1].units] == [PARA_BRAVO]


def test_a_document_with_no_headings_is_one_headingless_section():
    sections = _split_sections(PARAGRAPH_GAP.join([PARA_ALPHA, PARA_BRAVO]))

    assert len(sections) == 1
    assert sections[0].heading is None
    assert [unit.text for unit in sections[0].units] == [PARA_ALPHA, PARA_BRAVO]


def test_a_heading_with_no_content_after_it_is_a_section_with_no_units():
    sections = _split_sections(PARAGRAPH_GAP.join([HEADING_ONE, PARA_ALPHA, HEADING_TWO]))

    assert [(section.heading, len(section.units)) for section in sections] == [
        (HEADING_ONE, 1),
        (HEADING_TWO, 0),
    ]


def test_a_table_between_two_headings_is_a_unit_of_the_section_above_it():
    markdown = PARAGRAPH_GAP.join([HEADING_ONE, PARA_ALPHA, INFOBOX, HEADING_TWO, PARA_BRAVO])

    sections = _split_sections(markdown)

    assert [unit.kind for unit in sections[0].units] == ["prose", "table"]
    assert sections[0].units[1].text == INFOBOX, "the table run was not kept whole"
    assert [section.heading for section in sections] == [HEADING_ONE, HEADING_TWO]
    assert [unit.text for unit in sections[1].units] == [PARA_BRAVO]


def test_a_comment_inside_a_fenced_code_block_does_not_open_a_section():
    """A hash at the start of a line is only a heading outside a code block.

    A fenced sample whose line begins with a hash is content, and cutting there
    would store half a code block under a heading it never had.
    """
    markdown = "\n".join(
        [HEADING_ONE, "", "```python", "# not a heading", "value = 1", "```", "", PARA_ALPHA]
    )

    sections = _split_sections(markdown)

    assert [section.heading for section in sections] == [HEADING_ONE]
    assert [unit.text.splitlines()[0] for unit in sections[0].units] == ["```python", PARA_ALPHA]


def test_each_paragraph_of_a_fitting_section_becomes_its_own_node():
    """The document's boundaries are the nodes', however much smaller than the cap.

    Two paragraphs the author separated are two thoughts, so the cap does not
    join them into one node merely because both would fit inside it.
    """
    pipeline = _stub_pipeline(chunk_size=128, chunk_overlap=0)
    section = "\n\n".join([PARA_ALPHA, PARA_BRAVO])
    assert _model_tokens(section) < 128, "the section must fit the cap for this test to mean anything"

    chunking = _ingest_markdown(pipeline, f"{HEADING_ONE}\n\n{section}")

    stored = [node.get_content() for node in _stored_nodes(pipeline)]
    assert chunking["nodes"] == 2, "a section that fits was packed rather than stored as it stands"
    assert stored == [f"{HEADING_ONE}\n\n{PARA_ALPHA}", f"{HEADING_ONE}\n\n{PARA_BRAVO}"]


def test_a_section_over_the_cap_is_divided_at_its_own_paragraph_boundaries():
    pipeline = _stub_pipeline(chunk_size=128, chunk_overlap=0)
    section = "\n\n".join(PARAGRAPHS)
    assert _model_tokens(section) > 128, "the section must exceed the cap for this test to bite"

    chunking = _ingest_markdown(pipeline, f"{HEADING_ONE}\n\n{section}")

    stored = [node.get_content() for node in _stored_nodes(pipeline)]
    assert len(stored) > 1, "the section must be divided for this test to mean anything"
    assert len(stored) == chunking["nodes"]
    for text in stored:
        assert text.removeprefix(f"{HEADING_ONE}\n\n") in _paragraph_runs(PARAGRAPHS), (
            "a node's boundaries are not the section's own paragraph boundaries"
        )
    for paragraph in PARAGRAPHS:
        holders = [text for text in stored if paragraph in text]
        assert len(holders) == 1, "a paragraph was divided between nodes or stored twice"


def test_every_node_of_a_long_section_carries_its_heading():
    pipeline = _stub_pipeline(chunk_size=128, chunk_overlap=0)

    _ingest_markdown(pipeline, f"{HEADING_ONE}\n\n{PARAGRAPHS_JOINED}")

    stored = [node.get_content() for node in _stored_nodes(pipeline)]
    assert len(stored) > 1, "the section must be divided for this test to mean anything"
    for text in stored:
        assert text.startswith(f"{HEADING_ONE}\n\n"), "a node divided out of a section lost its heading"


def test_a_section_that_is_one_node_carries_its_heading_once():
    pipeline = _stub_pipeline(chunk_size=128, chunk_overlap=0)

    _ingest_markdown(pipeline, f"{HEADING_ONE}\n\n{PARA_ALPHA}")

    stored = [node.get_content() for node in _stored_nodes(pipeline)]
    assert len(stored) == 1, "the paragraph must fit one node for this test to mean anything"
    assert stored[0].count(HEADING_ONE) == 1, "the heading was repeated inside one node"


def test_a_paragraph_wider_than_the_cap_is_divided_at_sentence_boundaries():
    """The last resort: a single unit with no paragraph break left inside it.

    It is the only place the configured overlap still applies, so its nodes are
    the ones that repeat the sentence at their boundary.
    """
    pipeline = _stub_pipeline(chunk_size=128, chunk_overlap=32)
    assert _model_tokens(SENTENCE_DOC) > 128, "the fixture must exceed the cap for this test to bite"

    chunking = _ingest_markdown(pipeline, f"{HEADING_ONE}\n\n{SENTENCE_DOC}")

    stored = [node.get_content() for node in _stored_nodes(pipeline)]
    assert len(stored) > 1, "the paragraph must be divided for this test to mean anything"
    assert chunking["max_node_tokens"] <= 128, "a node exceeded the cap it was divided to"
    assert all(text.startswith(f"{HEADING_ONE}\n\n") for text in stored)
    assert "Sentence 1 " in stored[0], "the paragraph's own opening was not stored"
    body = stored[0].removeprefix(f"{HEADING_ONE}\n\n")
    trailing = f"Sentence {body.rstrip().rsplit('Sentence ', 1)[-1]}"
    assert trailing in stored[1], "the trailing sentence of one node was not repeated into the next"
    assert chunking["overlap_tokens"] > 0, "the configured overlap was not reported between them"


def _document_shapes() -> dict[str, str]:
    """One document per way a node can be bounded: section, paragraph, neither."""
    return {
        "section-fits": f"{HEADING_ONE}\n\n{PARA_ALPHA}\n\n{PARA_BRAVO}",
        "section-paragraphs": SECTION_OF_PARAGRAPHS,
        "paragraph-divided": f"{HEADING_ONE}\n\n{SENTENCE_DOC}",
        "table-under-a-heading": f"{HEADING_ONE}\n\n{PARA_ALPHA}\n\n{INFOBOX}",
        "wide-table": f"{HEADING_ONE}\n\n{WIDE_TABLE}",
        "padded-table": f"{HEADING_ONE}\n\n{PADDED_TABLE}",
        "row-wider-than-the-window": f"{HEADING_ONE}\n\n{LONG_ROW_TABLE}",
        "no-structure": SENTENCE_DOC,
    }


def test_no_node_exceeds_the_cap_on_any_shape_of_document():
    """The floor is switched off so the cap is the only thing sizing a node.

    A merge is allowed past the cap, up to the model's window, which is the
    separate bound the test after this one covers.
    """
    pipeline = _stub_pipeline(chunk_size=128, chunk_overlap=32, min_chunk_tokens=0)

    for name, markdown in _document_shapes().items():
        chunking = _ingest_markdown(pipeline, markdown, filename=f"{name}.txt")
        assert chunking["max_node_tokens"] <= 128, f"{name} produced a node past the cap"
        assert chunking["floor_merges"] == 0, f"{name} merged a node with the floor switched off"


def test_no_node_exceeds_the_model_window_on_any_shape_of_document():
    pipeline = _stub_pipeline(chunk_size=128, chunk_overlap=32)

    for name, markdown in _document_shapes().items():
        chunking = _ingest_markdown(pipeline, markdown, filename=f"{name}.txt")
        assert chunking["max_node_tokens"] <= EMBEDDING_MAX_TOKENS, (
            f"{name} produced a node the model cannot read whole"
        )


def test_overlap_is_not_repeated_between_the_nodes_of_a_section():
    """A structural seam is where the source changed subject, not a cut to hide."""
    pipeline = _stub_pipeline(chunk_size=128, chunk_overlap=32)

    chunking = _ingest_markdown(pipeline, SECTION_OF_PARAGRAPHS)

    stored = [node.get_content() for node in _stored_nodes(pipeline)]
    assert len(stored) > 1, "the section must be divided for this test to mean anything"
    assert chunking["overlap_tokens"] == 0, "text was repeated across a paragraph boundary"
    for paragraph in PARAGRAPHS:
        assert sum(1 for text in stored if paragraph in text) == 1, "a paragraph was repeated"


def test_a_fragment_is_merged_inside_its_own_section_before_reaching_across_a_heading():
    """Both merges would be legal; the one that keeps the sections apart is taken.

    The fragment opens the second section, so the neighbour before it lies beyond
    the heading and the neighbour after it does not.
    """
    pipeline = _stub_pipeline(chunk_size=128, chunk_overlap=0)
    # The infobox keeps the fragment from being packed with the paragraph after
    # it: a table is a node of its own, so the fragment is left standing alone at
    # a size the floor has to act on.
    markdown = PARAGRAPH_GAP.join(
        [f"{HEADING_ONE}\n\n{PARA_ALPHA}", f"{HEADING_TWO}\n\n{FRAGMENT}\n\n{INFOBOX}"]
    )

    chunking = _ingest_markdown(pipeline, markdown)

    assert chunking["floor_merges"] == 1, "the fragment was not merged"
    carriers = [text for text in (n.get_content() for n in _stored_nodes(pipeline)) if FRAGMENT in text]
    assert len(carriers) == 1, "the fragment was dropped or stored twice"
    assert INFOBOX_HEADER in carriers[0], "the fragment was merged beyond its own section"
    assert "Alpha" not in carriers[0], "the merge joined two of the document's sections"


def test_a_fragment_whose_only_legal_merge_lies_across_a_heading_is_still_merged():
    """The preference is an ordering of candidates, not a prohibition.

    The fragment's own section offers a neighbour that is already at the model's
    window, so the only merge left is the one across the heading -- and a floor
    that refused it would leave the fragment stored alone, which is the state the
    floor exists to prevent. FRAGMENT_WIDE carries the width this needs: a fragment
    narrow enough to fit beside that neighbour is taken by its own section instead,
    which the neighbouring test asserts.
    """
    pipeline = _stub_pipeline(chunk_size=None, chunk_overlap=0)
    markdown = PARAGRAPH_GAP.join(
        [f"{HEADING_ONE}\n\n{PARA_ALPHA}", f"{HEADING_TWO}\n\n{FRAGMENT_WIDE}\n\n{WIDE_TABLE}"]
    )

    chunking = _ingest_markdown(pipeline, markdown)

    assert chunking["floor_merges"] == 1, "the fragment was left stored on its own"
    assert chunking["floor_refusals"] == 0, "a legal merge across the heading was refused"
    carriers = [text for text in (n.get_content() for n in _stored_nodes(pipeline)) if FRAGMENT_WIDE in text]
    assert len(carriers) == 1
    assert "Alpha" in carriers[0], "the only legal merge available was not the one taken"
    assert chunking["max_node_tokens"] <= EMBEDDING_MAX_TOKENS


def test_a_heading_carried_by_every_node_of_its_section_is_not_a_repeat():
    pipeline = _stub_pipeline(chunk_size=128, chunk_overlap=0)

    chunking = _ingest_markdown(pipeline, SECTION_OF_SIX)

    stored = [node.get_content() for node in _stored_nodes(pipeline)]
    assert len(stored) >= 3, "the fixture must produce several nodes of one section"
    assert all(text.startswith(HEADING_ONE) for text in stored)
    assert chunking["duplicate_nodes"] == 0, "the heading they share was collapsed as repeated text"
    assert len(stored) == chunking["nodes"]


def test_a_merge_stores_the_heading_its_nodes_share_only_once():
    """A merged node names its section once, however many texts were folded into it.

    Every unit of a section leads with that section's heading, so joining two of
    them verbatim stores the heading once per text: measured on a real page, one
    node carried 15 copies of it, and the copies were 15% of the source's stored
    tokens -- paid for in embedding space by text that says nothing new.
    """
    pipeline = _stub_pipeline(chunk_size=128, chunk_overlap=0)
    short_one = _paragraph("Echo", sentences=1)
    short_two = _paragraph("Foxtrot", sentences=1)

    chunking = _ingest_markdown(
        pipeline, PARAGRAPH_GAP.join([HEADING_ONE, short_one, short_two]), filename="shared.txt"
    )

    assert chunking["floor_merges"] == 1, "the fixture must exercise the floor"
    stored = [node.get_content() for node in _stored_nodes(pipeline)]
    assert len(stored) == 1, "two below-floor paragraphs did not become one node"
    assert stored[0].count(HEADING_ONE) == 1, "the shared heading was stored once per merged node"
    assert stored[0].startswith(HEADING_ONE), "the merged node lost the heading naming it"
    assert short_one in stored[0] and short_two in stored[0], "content was dropped to satisfy the floor"


def test_a_merge_across_a_heading_keeps_both_headings():
    """A heading only one of the merged texts carries still marks where it begins.

    Dropping the repeated heading must not drop a *different* one: the merged node
    holds two of the document's sections, and each has to stay named where it
    opens, or the second section's text reads as part of the first.
    """
    pipeline = _stub_pipeline(chunk_size=None, chunk_overlap=0)
    markdown = PARAGRAPH_GAP.join(
        [f"{HEADING_ONE}\n\n{PARA_ALPHA}", f"{HEADING_TWO}\n\n{FRAGMENT_WIDE}\n\n{WIDE_TABLE}"]
    )

    chunking = _ingest_markdown(pipeline, markdown, filename="crossing.txt")

    assert chunking["floor_merges"] == 1, "the fragment was not merged across the heading"
    stored = [node.get_content() for node in _stored_nodes(pipeline)]
    carrier = [text for text in stored if FRAGMENT_WIDE in text]
    assert len(carrier) == 1
    assert carrier[0].count(HEADING_ONE) == 1, "the heading of the section it opens in was lost"
    assert carrier[0].count(HEADING_TWO) == 1, "the differing heading was dropped as if it repeated"


def test_text_repeated_inside_one_section_is_still_stored_once():
    """The collapse still runs, and still runs last.

    The infobox between the two copies keeps them from being packed into one
    node, so the repetition really is two byte-identical nodes of one section.
    """
    pipeline = _stub_pipeline(chunk_size=128, chunk_overlap=0)
    markdown = PARAGRAPH_GAP.join([HEADING_ONE, PARA_ALPHA, INFOBOX, PARA_ALPHA])

    chunking = _ingest_markdown(pipeline, markdown)

    assert chunking["duplicate_nodes"] == 1, "the repeated paragraph was stored twice"
    stored = [node.get_content() for node in _stored_nodes(pipeline)]
    assert len(stored) == chunking["nodes"]
    assert sum(1 for text in stored if PARA_ALPHA in text) == 1


def test_the_chunking_outcome_reports_which_boundaries_produced_the_nodes():
    pipeline = _stub_pipeline(chunk_size=128, chunk_overlap=32)

    fits = _ingest_markdown(
        pipeline, f"{HEADING_ONE}\n\n{PARA_ALPHA}\n\n{PARA_BRAVO}", filename="fits.txt"
    )
    divided = _ingest_markdown(
        pipeline, f"{HEADING_ONE}\n\n{SENTENCE_DOC}", filename="divided.txt"
    )

    assert fits["nodes"] == 2
    assert fits["structural_nodes"] == 2, (
        "paragraphs stored under their own boundary were not reported as structural"
    )
    assert fits["divided_nodes"] == 0

    assert divided["floor_merges"] == 0, "the fixture must not need the floor for this comparison"
    assert divided["structural_nodes"] == 0, "a divided paragraph was reported as structural"
    assert divided["divided_nodes"] == divided["nodes"], (
        "the reported counts do not describe the nodes that were stored"
    )


# --- Collections: a filter key that is never content ------------------------


def _collections_pipeline(settings) -> IndexPipeline:
    """A pipeline whose embedder depends on the text it is given.

    The vectors are what this section compares, and `MockEmbedding` returns one
    vector for every text, so a comparison against it would hold whatever the
    text was. `_ContentEmbedding` embeds text as a function of itself, so equal
    vectors mean equal embedded text.
    """
    return IndexPipeline(
        settings, converter=MagicMock(), embedding_model=_ContentEmbedding(embed_dim=8)
    )


def _stored_vectors(pipeline) -> dict[str, list[float]]:
    """The vectors the store holds, keyed by node id.

    The store is the only place they exist: the docstore strips a node's
    embedding as it stores it, and `SimpleVectorStore.get_nodes` raises.
    """
    return dict(pipeline.index.storage_context.vector_store._data.embedding_dict)


def test_a_collection_is_filterable_metadata_that_the_model_never_reads(settings):
    """The prefilter must see the collections; nothing else may.

    Both halves matter: the metadata carrying the key is what the filter tests,
    and the key being excluded from the embedded rendering is what keeps tagging
    from becoming part of the content a query is matched against.
    """
    pipeline = _collections_pipeline(settings)
    pipeline.converter.convert_file.return_value = SENTENCE_DOC

    pipeline.ingest_file(
        source_id="source",
        file=BytesIO(b"x"),
        filename="doc.txt",
        collections=["csharp", "dotnet"],
    )

    nodes = _stored_nodes(pipeline)
    assert len(nodes) > 1, "the document must split for this test to mean anything"
    for node in nodes:
        assert node.metadata["collections"] == ["csharp", "dotnet"]
        embedded = node.get_content(metadata_mode=MetadataMode.EMBED)
        assert "collections" not in embedded
        assert "csharp" not in embedded, "a collection name reached the embedded text"

    # The store keeps the same metadata for the prefilter to read, so the key is
    # present exactly where filtering looks for it.
    stored = next(iter(pipeline.index.storage_context.vector_store._data.metadata_dict.values()))
    assert stored["collections"] == ["csharp", "dotnet"]


def test_tagging_a_source_does_not_move_its_vectors(settings):
    """Collections are a predicate, not content.

    Metadata is otherwise charged against the chunk-size budget before splitting
    -- the splitter renders a node for the embedder and for an LLM and budgets
    for whichever is longer -- so without the exclusion a collection name changes
    where a document splits, and therefore every vector it produces. This is the
    test that fails when only one of the two renderings is excluded.
    """
    untagged = _collections_pipeline(settings)
    tagged = _collections_pipeline(settings)
    for pipeline in (untagged, tagged):
        pipeline.converter.convert_file.return_value = SENTENCE_DOC

    untagged.ingest_file(source_id="plain", file=BytesIO(b"x"), filename="doc.txt")
    tagged.ingest_file(
        source_id="tagged",
        file=BytesIO(b"x"),
        filename="doc.txt",
        collections=["csharp", "dotnet-retrieval"],
    )

    plain_nodes = _stored_nodes(untagged)
    assert len(plain_nodes) > 1, "the document must split for this test to mean anything"
    assert [node.get_content() for node in plain_nodes] == [
        node.get_content() for node in _stored_nodes(tagged)
    ], "tagging changed how the document was chunked"
    assert _stored_vectors(untagged) == _stored_vectors(tagged)

    # Which is the point of the property: a caller who never filters cannot tell
    # the tagged source from the untagged one.
    query = "Sentence 5 covers its own separate subject"
    assert [item["score"] for item in untagged.search(query, top_k=5)] == [
        item["score"] for item in tagged.search(query, top_k=5)
    ]


def test_re_uploading_a_document_replaces_its_collections(settings):
    """A source belongs to what its most recent upload named, and nothing else."""
    pipeline = _collections_pipeline(settings)
    pipeline.converter.convert_file.return_value = SENTENCE_DOC

    pipeline.ingest_file(
        source_id="first", file=BytesIO(b"x"), filename="doc.txt", collections=["csharp"]
    )
    pipeline.ingest_file(
        source_id="second", file=BytesIO(b"x"), filename="doc.txt", collections=["dotnet"]
    )

    assert len(pipeline.source_catalog) == 1, "the re-upload was indexed as a second source"
    assert pipeline.source_catalog[0].collections == ("dotnet",)
    assert _stored_nodes(pipeline)[0].metadata["collections"] == ["dotnet"]
    assert pipeline.search("Sentence 5", top_k=5, collections=["csharp"]) == []


def test_collections_reach_a_source_whose_url_redirected(settings):
    """A page is identified by where it landed; its collections follow it there."""
    pipeline = _collections_pipeline(settings)
    pipeline.converter.convert_url.return_value = (SENTENCE_DOC, "https://example.com/final")

    result, _ = pipeline.ingest_url(
        source_id="source", url="https://example.com/start", collections=["csharp"]
    )

    record = pipeline.source_catalog[0]
    assert record.name == "https://example.com/start"
    assert record.collections == ("csharp",)
    assert result["collections"] == ["csharp"]
    assert pipeline.search("Sentence 5", top_k=5, collections=["csharp"])


# --- Search scoping ---------------------------------------------------------


# Unlike SENTENCE_DOC, so a result can be attributed to the source it came from
# by its text alone.
PLAIN_DOC = " ".join(
    f"Item {index} documents an unrelated administrative procedure in its own wording."
    for index in range(1, 61)
)


def _scoped_pipeline(settings) -> IndexPipeline:
    """One source in a collection, one in another, and one in none.

    Deliberately more content outside any single scope than inside it, so a
    search that padded its result count from outside the scope would show up.
    """
    pipeline = _collections_pipeline(settings)
    contents = {
        "csharp.txt": "# C#\n\n" + SENTENCE_DOC,
        "dotnet.txt": "# .NET\n\n" + SENTENCE_DOC.replace("Sentence", "Clause"),
        "plain.txt": "# Plain\n\n" + PLAIN_DOC,
    }
    pipeline.converter.convert_file.side_effect = lambda _file, filename: contents[filename]
    pipeline.ingest_file(
        source_id="csharp", file=BytesIO(b"x"), filename="csharp.txt", collections=["csharp"]
    )
    pipeline.ingest_file(
        source_id="dotnet", file=BytesIO(b"x"), filename="dotnet.txt", collections=["dotnet"]
    )
    pipeline.ingest_file(source_id="plain", file=BytesIO(b"x"), filename="plain.txt")
    return pipeline


def _chunk_count(pipeline: IndexPipeline, name: str) -> int:
    return next(record.chunk_count for record in pipeline.source_catalog if record.name == name)


def test_a_filtered_search_returns_only_in_scope_chunks(settings):
    pipeline = _scoped_pipeline(settings)

    results = pipeline.search("content", top_k=50, collections=["csharp"])

    assert results
    assert {item["source_id"] for item in results} == {"csharp"}


def test_several_requested_collections_match_any_of_them(settings):
    pipeline = _scoped_pipeline(settings)

    results = pipeline.search("content", top_k=50, collections=["csharp", "dotnet"])

    assert {item["source_id"] for item in results} == {"csharp", "dotnet"}


def test_the_result_count_applies_within_the_scope(settings):
    """A requested count is drawn from the scope, not padded from outside it.

    The store prefilters before the similarity scan, so a scope holding fewer
    chunks than were asked for returns fewer -- rather than spending the rest of
    the budget on sources the caller excluded.
    """
    pipeline = _scoped_pipeline(settings)
    in_scope = _chunk_count(pipeline, "csharp.txt")
    out_of_scope = sum(
        record.chunk_count for record in pipeline.source_catalog if record.name != "csharp.txt"
    )
    assert in_scope < 50, "the fixture must hold fewer chunks in scope than will be asked for"
    assert out_of_scope > in_scope, "the fixture must hold enough outside the scope to pad with"

    results = pipeline.search("content", top_k=50, collections=["csharp"])

    assert len(results) == in_scope
    assert {item["source_id"] for item in results} == {"csharp"}


def test_a_filter_naming_an_unknown_collection_is_an_empty_result(settings):
    """A mistyped collection returns nothing rather than erroring or widening."""
    pipeline = _scoped_pipeline(settings)

    assert pipeline.search("content", top_k=5, collections=["cshrap"]) == []


def test_an_unfiltered_search_still_reaches_an_untagged_source(settings):
    """The change is additive: search without a filter behaves as it always has."""
    pipeline = _scoped_pipeline(settings)

    results = pipeline.search("content", top_k=50)

    assert {item["source_id"] for item in results} == {"csharp", "dotnet", "plain"}


# --- Where a hit sits, and what neighbours it ---------------------------------


def _placed(pipeline: IndexPipeline, name: str, neighbours: int = 0) -> list[dict]:
    """Every result a broad search returns for one source, in position order."""
    results = [
        item
        for item in pipeline.search("content", top_k=50, neighbours=neighbours)
        if item["source_name"] == name
    ]
    return sorted(results, key=lambda item: item["position"])


def test_a_result_states_where_its_chunk_sits_in_its_source(settings):
    """The reported position names the chunk the result carries.

    A position that merely numbered the results would be useless for finding
    what comes next in the document, which is what it is for.
    """
    pipeline = _scoped_pipeline(settings)

    results = pipeline.search("content", top_k=50)

    assert results
    store = pipeline.index.storage_context.docstore.docs
    for item in results:
        # A file's source key is its filename, and the position is what the id
        # was hashed from, so an id rebuilt from the reported position is the id
        # of the only chunk this result can honestly be describing.
        holder = store.get(_node_id(item["source_name"], item["position"]))
        assert holder is not None, "the reported position names no stored chunk"
        assert holder.get_content() == item["text"], "the reported position names another chunk"


def test_a_result_reports_how_many_chunks_neighbour_it(settings):
    """The counts say how much context exists, whether or not it is returned."""
    pipeline = _scoped_pipeline(settings)
    name = "plain.txt"
    count = _chunk_count(pipeline, name)
    assert count > 2, "the fixture must hold a middle chunk, not only edges"

    placed = _placed(pipeline, name)

    assert [item["position"] for item in placed] == list(range(count))
    for item in placed:
        assert item["neighbours_before"] == item["position"]
        assert item["neighbours_after"] == count - item["position"] - 1


def test_a_sources_edges_report_no_neighbour_beyond_them(settings):
    """A neighbour count is bounded by its source, not by the corpus.

    The fixture holds three sources, so chunks at one source's end have other
    sources' chunks available to be counted as its neighbours by mistake.
    """
    pipeline = _scoped_pipeline(settings)
    name = "plain.txt"
    count = _chunk_count(pipeline, name)

    placed = _placed(pipeline, name)
    first, last = placed[0], placed[-1]

    assert first["position"] == 0
    assert first["neighbours_before"] == 0
    assert first["neighbours_after"] == count - 1
    assert last["position"] == count - 1
    assert last["neighbours_before"] == count - 1
    assert last["neighbours_after"] == 0


def test_a_hit_carries_the_chunks_either_side_of_it(settings):
    """Asking for neighbours returns the passage around a hit, in reading order.

    The neighbours are compared against what the same search reports for those
    positions, so what comes back is the stored chunk next to the hit rather
    than merely some other chunk of the document.
    """
    pipeline = _scoped_pipeline(settings)
    name = "plain.txt"
    count = _chunk_count(pipeline, name)
    assert count > 2, "the fixture must hold a hit with neighbours on both sides"

    placed = _placed(pipeline, name, neighbours=1)
    middle = placed[1]

    assert [item["position"] for item in middle["neighbours"]] == [0, 2]
    assert [item["text"] for item in middle["neighbours"]] == [
        placed[0]["text"],
        placed[2]["text"],
    ]


def test_the_requested_count_sets_how_many_neighbours_come_back(settings):
    """One number per side, and it is the caller's: the hit need not be central."""
    pipeline = _scoped_pipeline(settings)
    name = "plain.txt"
    count = _chunk_count(pipeline, name)
    assert count > 4, "the fixture must hold a hit with two chunks on each side"

    placed = _placed(pipeline, name, neighbours=2)
    middle = placed[3]

    assert [item["position"] for item in middle["neighbours"]] == [1, 2, 4, 5]
    assert [item["text"] for item in middle["neighbours"]] == [
        placed[1]["text"],
        placed[2]["text"],
        placed[4]["text"],
        placed[5]["text"],
    ]


def test_the_ranked_results_are_the_list_they_were_without_neighbours(settings):
    """Asking for context does not add results to the ranking or reorder it.

    `top_k` keeps meaning ranked hits: a caller that widens one result's context
    is not spending slots on unranked chunks.
    """
    pipeline = _scoped_pipeline(settings)

    ranked = pipeline.search("content", top_k=5)
    widened = pipeline.search("content", top_k=5, neighbours=2)

    assert len(widened) == len(ranked)
    assert [item["position"] for item in widened] == [item["position"] for item in ranked]
    assert [item["text"] for item in widened] == [item["text"] for item in ranked]
    assert [item["score"] for item in widened] == [item["score"] for item in ranked]


def test_neighbours_stop_at_the_source_boundary(settings):
    """A request wider than a source can fill returns what exists, and not another source's text.

    The short source sits beside a longer one, so a walk past its end would reach
    the other source's chunks rather than returning fewer.
    """
    pipeline = _collections_pipeline(settings)
    contents = {"short.txt": PROSE, "long.txt": SENTENCE_DOC}
    pipeline.converter.convert_file.side_effect = lambda _file, filename: contents[filename]
    pipeline.ingest_file(source_id="short", file=BytesIO(b"x"), filename="short.txt")
    pipeline.ingest_file(source_id="long", file=BytesIO(b"x"), filename="long.txt")

    count = _chunk_count(pipeline, "short.txt")
    widest = 5  # the widest neighbour count a request may ask for
    assert count > 1, "the fixture must hold a source with more than one chunk"
    assert count < widest, "the fixture must hold a source too short to fill the request"

    placed = _placed(pipeline, "short.txt", neighbours=widest)
    first, last = placed[0], placed[-1]

    assert len(placed) == count
    assert [item["position"] for item in first["neighbours"]] == list(range(1, count))
    assert [item["position"] for item in last["neighbours"]] == list(range(count - 1))
    assert len(last["neighbours"]) == count - 1, "fewer neighbours exist than were asked for"
    own_texts = {item["text"] for item in placed}
    for item in (first, last):
        assert {n["text"] for n in item["neighbours"]} <= own_texts, (
            "a neighbour came from outside the hit's source"
        )


def test_a_neighbouring_chunk_reports_no_score(settings):
    """A neighbour was not ranked against the query, so it has no relevance to report.

    Inheriting the hit's score would present unranked context as equally relevant
    to the query that found the hit.
    """
    pipeline = _scoped_pipeline(settings)

    placed = _placed(pipeline, name := "plain.txt", neighbours=1)
    middle = placed[1]

    assert middle["neighbours"], "the fixture returned no neighbour to inspect"
    for neighbour in middle["neighbours"]:
        assert set(neighbour) == {"text", "position"}, neighbour
    assert isinstance(middle["score"], float)
    assert middle["score"] == _placed(pipeline, name)[1]["score"]


def test_a_scoped_search_returns_neighbours_only_from_its_scope(settings):
    """Adjacency cannot leak past a collection filter, and the first chunk has nothing before it."""
    pipeline = _scoped_pipeline(settings)
    name = "csharp.txt"
    own_texts = {item["text"] for item in _placed(pipeline, name)}

    results = pipeline.search("content", top_k=50, collections=["csharp"], neighbours=5)

    assert results
    for item in results:
        assert item["source_name"] == name
        for neighbour in item["neighbours"]:
            assert neighbour["text"] in own_texts, "a neighbour came from outside the scope"

    first = min(results, key=lambda item: item["position"])
    assert first["position"] == 0
    assert first["neighbours_before"] == 0
    assert all(neighbour["position"] > 0 for neighbour in first["neighbours"])


# --- The catalog's per-source record ----------------------------------------


def test_the_source_record_describes_what_was_ingested(settings):
    pipeline = _collections_pipeline(settings)
    pipeline.converter.convert_file.return_value = SENTENCE_DOC
    pipeline.converter.convert_url.return_value = (PLAIN_DOC, "https://example.com/final")

    pipeline.ingest_file(
        source_id="file-source",
        file=BytesIO(b"x"),
        filename="report.txt",
        collections=["csharp", "dotnet"],
    )
    pipeline.ingest_url(source_id="url-source", url="https://example.com/start")

    records = {(record.name, record.source_type): record for record in pipeline.source_catalog}
    filed = records[("report.txt", "file")]
    untagged = records[("https://example.com/start", "url")]

    assert filed.collections == ("csharp", "dotnet")
    # A source ingested without collections is described, not omitted, so the
    # catalog is what makes an untagged source visible.
    assert untagged.collections == ()
    # Every stored node belongs to one of the two records, and each counts its own.
    assert len(_stored_nodes(pipeline)) == pipeline.indexed_chunks
    assert filed.chunk_count + untagged.chunk_count == pipeline.indexed_chunks
    assert pipeline.indexed_sources == 2


def test_the_source_record_is_replaced_rather_than_duplicated(settings):
    pipeline = _collections_pipeline(settings)
    pipeline.converter.convert_file.return_value = SENTENCE_DOC

    pipeline.ingest_file(
        source_id="first", file=BytesIO(b"x"), filename="doc.txt", collections=["csharp"]
    )
    first_chunks = _chunk_count(pipeline, "doc.txt")

    pipeline.ingest_file(
        source_id="second", file=BytesIO(b"x"), filename="doc.txt", collections=["dotnet"]
    )

    assert pipeline.indexed_sources == 1
    assert _chunk_count(pipeline, "doc.txt") == first_chunks
    assert pipeline.indexed_chunks == first_chunks


def test_the_catalog_a_reader_holds_cannot_change_underneath_it(settings):
    """The catalog is published as an immutable snapshot, not a live view.

    A reader iterating the mapping while an ingestion replaced a source would
    see half a replacement, or fail outright, so each write publishes a tuple
    that describes the state it produced.
    """
    pipeline = _collections_pipeline(settings)
    pipeline.converter.convert_file.return_value = SENTENCE_DOC
    pipeline.ingest_file(
        source_id="first", file=BytesIO(b"x"), filename="doc.txt", collections=["csharp"]
    )

    before = pipeline.source_catalog
    pipeline.ingest_file(
        source_id="second", file=BytesIO(b"x"), filename="doc.txt", collections=["dotnet"]
    )

    assert isinstance(before, tuple)
    assert before[0].collections == ("csharp",)
    assert pipeline.source_catalog[0].collections == ("dotnet",)


def _recorded_positions(pipeline: IndexPipeline) -> dict[str, int]:
    """Every node id the pipeline recorded a position for, and that position."""
    return {
        node_id: position
        for positions in pipeline._source_positions.values()
        for node_id, position in positions.items()
    }


def _ids_in_store(pipeline: IndexPipeline) -> set[str]:
    return {node.node_id for node in _stored_nodes(pipeline)}


def test_the_recorded_order_describes_what_the_source_holds(settings):
    """A hit's position has to be recorded when the nodes are stored.

    The id is that position hashed, so the position cannot be recovered from a
    stored node and nothing else holds it. Without this mapping a hit cannot be
    placed inside its source, so neither its position nor its neighbours can be
    reported. A re-ingestion replaces a source wholesale, and positions recorded
    for the content it replaced would place hits in nodes that no longer exist.
    """
    pipeline = _collections_pipeline(settings)
    pipeline.converter.convert_file.return_value = PARAGRAPH_GAP.join([PROSE] * 3)
    pipeline.ingest_file(source_id="first", file=BytesIO(b"x"), filename="doc.txt")

    wide = _recorded_positions(pipeline)
    assert wide, "the fixture stored nothing to place"
    assert set(wide) == _ids_in_store(pipeline), "the recorded ids are not the stored ones"
    assert sorted(wide.values()) == list(range(len(wide))), "positions do not run from zero"

    # The position has to be the one the id was derived from, not merely some
    # numbering: the id is what a hit arrives carrying, so a position that does
    # not hash back to that id would name a different node as the hit.
    for node in _stored_nodes(pipeline):
        assert _node_id("doc.txt", wide[node.node_id]) == node.node_id, (
            "a recorded position does not hash back to the id it is recorded against"
        )

    pipeline.converter.convert_file.return_value = "Ngắn."
    pipeline.ingest_file(source_id="second", file=BytesIO(b"x"), filename="doc.txt")

    narrow = _recorded_positions(pipeline)
    assert len(narrow) < len(wide), "the fixture did not shrink the source it re-ingested"
    assert set(narrow) == _ids_in_store(pipeline), (
        "positions recorded for the replaced content survived the replacement"
    )


def test_recording_a_position_does_not_reach_the_stored_node(settings):
    """The order is bookkeeping beside the store, not part of what is embedded.

    Metadata that is not listed as excluded is embedded with the node's text, so
    a position recorded on the node would enter every stored vector and move
    retrieval. This asserts the node itself is untouched.
    """
    pipeline = _collections_pipeline(settings)
    pipeline.converter.convert_file.return_value = PARAGRAPH_GAP.join([PROSE] * 3)
    pipeline.ingest_file(
        source_id="first", file=BytesIO(b"x"), filename="doc.txt", collections=["csharp"]
    )

    keys = {key for node in _stored_nodes(pipeline) for key in node.metadata}
    assert keys == {
        "source_id",
        "source_type",
        "source_name",
        "mime_type",
        "collections",
    }, keys
    assert _recorded_positions(pipeline), "the fixture recorded no positions to test against"
