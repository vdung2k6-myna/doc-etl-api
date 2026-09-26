"""Integration checks that span the HTTP routes, the pipeline, and startup."""

import threading
import time
from io import BytesIO
from unittest.mock import MagicMock, patch

import requests
from docling.datamodel.base_models import ConversionStatus
from fastapi.testclient import TestClient

from doc_etl_api.config import DEFAULT_USER_AGENT, Settings
from doc_etl_api.main import create_app
from doc_etl_api.pipeline import DoclingConverter
from tests.stubs import StubEmbedding


def _stub_converter(convert=None):
    converter = MagicMock()
    converter.is_supported_file.return_value = True
    converter.SUPPORTED_FILE_EXTENSIONS = {".txt"}
    if convert is not None:
        converter.convert_file.side_effect = convert

    # A bare MagicMock iterates to nothing, so an unstubbed `fetch_page` would
    # surface as "not enough values to unpack (expected 2, got 0)" when
    # `ingest_url` unpacks what the fetch returned -- a mock artifact that reads
    # like a parsing bug and hides its own cause. Fail with the reason instead.
    def unexpected_url(url, **kwargs):
        raise AssertionError(
            f"fetch_page({url!r}) was called, but this test does not stub URL "
            "ingestion -- a corpus URL probably reached the settings"
        )

    converter.fetch_page.side_effect = unexpected_url
    return converter


def _build_app(converter):
    with patch(
        "doc_etl_api.main._load_models",
        return_value=(converter, StubEmbedding(embed_dim=8)),
    ):
        return create_app()


def _wait_for_bootstrap(client, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        state = client.get("/health").json()["bootstrap"]
        if state in ("complete", "failed", "disabled"):
            return state
        time.sleep(0.05)
    raise AssertionError("bootstrap never finished")


def test_search_while_indexing_is_well_formed_and_excludes_the_new_source():
    """A search overlapping an indexing run must stay usable, and a source that
    has not finished indexing must not surface in its results."""
    gate = threading.Event()
    converter = _stub_converter(lambda file, filename: "# Doc\n\nExisting content.")
    app = _build_app(converter)
    pipeline = app.state.pipeline
    client = TestClient(app)

    pipeline.ingest_file(source_id="existing", file=BytesIO(b"x"), filename="existing.txt")

    # Second ingestion stalls in parsing, standing in for a background job that
    # is only part-way through. (TestClient runs background tasks to completion
    # before returning, so the thread calls the pipeline directly.)
    parsing_started = threading.Event()

    def stalling_convert(file, filename):
        parsing_started.set()
        gate.wait(timeout=10)
        return "# Doc\n\nUnique marker from the new source."

    converter.convert_file.side_effect = stalling_convert

    ingesting = threading.Thread(
        target=pipeline.ingest_file,
        kwargs={
            "source_id": "incoming",
            "file": BytesIO(b"y"),
            "filename": "incoming.txt",
        },
    )
    ingesting.start()
    assert parsing_started.wait(timeout=10), "ingestion never reached the parse stage"

    response = client.post("/search", json={"query": "content", "top_k": 10})

    assert response.status_code == 200
    results = response.json()["results"]
    assert isinstance(results, list)
    assert results, "the already-indexed source should still be searchable"
    assert "incoming" not in {r["source_id"] for r in results}
    for result in results:
        assert set(result) == {
            "text",
            "score",
            "source_id",
            "source_type",
            "source_name",
            "address",
            "collections",
            "position",
            "neighbours_before",
            "neighbours_after",
            "neighbours",
        }
        assert result["address"] == "existing.txt"
        assert result["collections"] == []
        assert result["neighbours"] == [], "the request asked for no neighbours"

    gate.set()
    ingesting.join(timeout=10)

    response = client.post("/search", json={"query": "marker", "top_k": 10})
    assert "incoming" in {r["source_id"] for r in response.json()["results"]}


def test_corpus_is_searchable_again_after_a_restart(monkeypatch, tmp_path):
    """A restart must repopulate the index from the configured corpus, with no
    caller re-uploading anything."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "guide.txt").write_text(
        "The STAR method structures behavioural answers into four parts."
    )

    monkeypatch.setattr("doc_etl_api.main.settings", Settings(knowledge_corpus_dir=str(corpus)))

    for attempt in (1, 2):
        converter = _stub_converter(lambda file, filename: (corpus / filename).read_text())
        app = _build_app(converter)
        client = TestClient(app)

        assert _wait_for_bootstrap(client) == "complete"
        health = client.get("/health").json()
        assert health["indexed_sources"] == 1
        assert health["indexed_chunks"] >= 1

        response = client.post("/search", json={"query": "STAR method", "top_k": 5})

        assert response.status_code == 200
        results = response.json()["results"]
        assert results, f"the index was empty after restart {attempt}"
        assert results[0]["source_name"] == "guide.txt"


def test_readiness_distinguishes_empty_from_unreachable(monkeypatch, tmp_path):
    """An empty index must be reported as ready-with-zero, not as an error."""
    monkeypatch.setattr("doc_etl_api.main.settings", Settings())
    client = TestClient(_build_app(_stub_converter()))

    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert body["indexed_sources"] == 0
    assert body["indexed_chunks"] == 0
    assert body["bootstrap"] == "disabled"


def test_re_uploading_a_file_replaces_it_end_to_end():
    """The whole path, through the routes: re-upload replaces, and an edit of a
    source leaves none of the version it superseded behind."""
    content = {"markdown": "# Guide\n\nThe STAR method structures behavioural answers."}
    converter = _stub_converter(lambda file, filename: content["markdown"])
    app = _build_app(converter)
    client = TestClient(app)

    def upload():
        return client.post(
            "/sources/files",
            files={"files": ("guide.txt", BytesIO(b"x"), "text/plain")},
        )

    assert upload().status_code == 202
    assert client.get("/health").json()["indexed_sources"] == 1

    # The same file uploaded again is one source, not two copies of it.
    upload()
    health = client.get("/health").json()
    assert health["indexed_sources"] == 1, "the file was indexed as a second source"
    assert health["indexed_chunks"] == 1, "a second copy of the file was stored"

    # The file is edited and re-uploaded. The superseded text must be gone.
    content["markdown"] = "# Guide\n\nThe XYZZY method replaces the earlier guidance."
    upload()

    results = client.post("/search", json={"query": "STAR method", "top_k": 10}).json()["results"]
    assert results
    assert not any("STAR method" in result["text"] for result in results), (
        "the superseded version of the file is still searchable"
    )
    assert client.get("/health").json()["indexed_sources"] == 1


def test_a_host_that_refuses_unidentified_clients_is_ingested_anyway(monkeypatch):
    """The whole path, through the route: a page that answers `403` to the HTTP
    library's default user agent is fetched and indexed when the request says
    who it is.

    The stub answers the way the measured host does -- the library default is
    refused, an identifying client is served -- so the assertion is the job's
    status rather than the arguments of a call.
    """
    monkeypatch.setattr(
        "doc_etl_api.main.settings",
        Settings(knowledge_corpus_dir="", knowledge_corpus_urls=""),
    )
    converter = DoclingConverter()
    app = _build_app(converter)
    client = TestClient(app)

    library_default = requests.utils.default_user_agent()
    sent_agents: list[str | None] = []
    page = b"<html><body><main><h1>Design Patterns</h1><p>Elements of reusable object "
    page += b"oriented software.</p></main></body></html>"

    def fetch(url, timeout=30, allow_redirects=True, headers=None):
        sent = (headers or {}).get("User-Agent")
        sent_agents.append(sent)
        if not sent or sent == library_default:
            refused = MagicMock(url=url)
            refused.raise_for_status.side_effect = requests.HTTPError(
                f"403 Client Error: Forbidden for url: {url}"
            )
            return refused
        return MagicMock(url=url, content=page, raise_for_status=MagicMock())

    conversion = MagicMock(
        status=ConversionStatus.SUCCESS,
        document=MagicMock(
            export_to_markdown=lambda: "# Design Patterns\n\nElements of reusable software."
        ),
    )

    with patch("doc_etl_api.pipeline.requests.get", side_effect=fetch):
        with patch.object(converter._converter, "convert", return_value=conversion):
            accepted = client.post(
                "/sources/urls",
                json={"urls": ["https://en.wikipedia.org/wiki/Design_Patterns"]},
            )
            job = client.get(f"/jobs/{accepted.json()[0]['job_id']}").json()

    assert job["status"] == "completed", f"the ingestion did not survive the fetch: {job}"
    assert sent_agents == [DEFAULT_USER_AGENT], "the request did not identify itself"
    assert client.get("/health").json()["indexed_sources"] == 1


def test_collections_scope_the_whole_path_from_upload_to_search():
    """The whole path, through the routes: two uploads land in different
    collections, the catalog reports both, and a filtered search returns only the
    source in the collection that was asked for.

    Run end to end because the parts are only useful joined: a collection that a
    filter cannot act on, or that the catalog does not report, would leave a
    caller with no way to find what the filter is meant to select.
    """
    contents = {
        "guide.txt": "# Guide\n\nThe STAR method structures behavioural answers.",
        "notes.txt": "# Notes\n\nThe XYZZY convention records unrelated remarks.",
    }
    converter = _stub_converter(lambda file, filename: contents[filename])
    client = TestClient(_build_app(converter))

    for name, collection in (("guide.txt", "interviews"), ("notes.txt", "conventions")):
        response = client.post(
            "/sources/files",
            files={"files": (name, BytesIO(b"x"), "text/plain")},
            data={"collections": [collection]},
        )
        assert response.status_code == 202, response.text

    catalog = client.get("/sources").json()["sources"]
    assert {source["name"]: source["collections"] for source in catalog} == {
        "guide.txt": ["interviews"],
        "notes.txt": ["conventions"],
    }
    assert all(source["chunk_count"] >= 1 for source in catalog)

    scoped = client.post(
        "/search", json={"query": "method", "top_k": 10, "collections": ["interviews"]}
    )
    assert scoped.status_code == 200
    results = scoped.json()["results"]
    assert results
    assert {result["source_name"] for result in results} == {"guide.txt"}

    # The other source is still reachable without a filter, so the scope narrowed
    # one search rather than the index.
    unscoped = client.post("/search", json={"query": "method", "top_k": 10}).json()["results"]
    assert {result["source_name"] for result in unscoped} == {"guide.txt", "notes.txt"}
