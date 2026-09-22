import asyncio
import logging
import threading
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient

from doc_etl_api.config import MAX_COLLECTIONS_PER_REQUEST, Settings
from doc_etl_api.jobs import JobRegistry, JobStatus
from doc_etl_api.main import create_app
from doc_etl_api.pipeline import IndexPipeline, SourceRecord
from doc_etl_api.schemas import SearchRequest, SearchResult
from tests.stubs import EMBEDDING_MAX_TOKENS, EMBEDDING_MODEL_NAME, StubEmbedding


@pytest.fixture
def client():
    converter = MagicMock()
    embedding = MagicMock()
    with patch("doc_etl_api.main._load_models", return_value=(converter, embedding)):
        with patch("doc_etl_api.main.create_pipeline", return_value=MagicMock()):
            app = create_app()
    app.state.pipeline = MagicMock()
    app.state.pipeline.converter.is_supported_file.return_value = True
    app.state.jobs = JobRegistry()
    return TestClient(app)


def test_healthcheck(client):
    response = client.get("/openapi.json")
    assert response.status_code == 200


def test_health_reports_ready_with_zero_counts(client):
    client.app.state.pipeline.indexed_sources = 0
    client.app.state.pipeline.indexed_chunks = 0

    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert body["indexed_sources"] == 0
    assert body["indexed_chunks"] == 0
    assert body["jobs_in_flight"] == 0
    assert body["bootstrap"] == "disabled"


def test_health_reports_index_counts(client):
    client.app.state.pipeline.indexed_sources = 4
    client.app.state.pipeline.indexed_chunks = 17

    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["indexed_sources"] == 4
    assert body["indexed_chunks"] == 17


def test_health_reports_jobs_in_flight(client):
    client.app.state.pipeline.indexed_sources = 0
    client.app.state.pipeline.indexed_chunks = 0
    client.app.state.jobs.create(source_id="source-1")
    client.app.state.jobs.create(source_id="source-2")

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["jobs_in_flight"] == 2


def test_health_performs_no_retrieval_or_embedding(client):
    pipeline = client.app.state.pipeline
    pipeline.indexed_sources = 3
    pipeline.indexed_chunks = 9
    # Any attempt to reach the index or the embedding model must explode, so a
    # passing test proves readiness did neither.
    pipeline.index.as_retriever.side_effect = AssertionError("readiness must not retrieve")
    pipeline._embedding_model.get_text_embedding.side_effect = AssertionError(
        "readiness must not embed"
    )

    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["indexed_sources"] == 3
    assert body["indexed_chunks"] == 9
    pipeline.search.assert_not_called()


async def test_search_runs_off_the_event_loop(client):
    """Retrieval embeds the query synchronously, so it must not hold the loop."""
    app = client.app
    search_threads: list[int] = []

    def record_search(query, top_k):
        search_threads.append(threading.get_ident())
        return []

    app.state.pipeline.search.side_effect = record_search

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
        response = await http.post("/search", json={"query": "hello"})

    assert response.status_code == 200
    assert search_threads, "the search endpoint never reached the pipeline"
    assert search_threads[0] != threading.get_ident(), "search ran on the event loop"


async def test_health_answers_while_a_search_is_in_flight(client):
    """A search in flight must not delay an unrelated readiness request."""
    app = client.app
    app.state.pipeline.indexed_sources = 0
    app.state.pipeline.indexed_chunks = 0

    release_search = threading.Event()
    timeline: list[str] = []

    def blocking_search(query, top_k):
        timeline.append("search_start")
        release_search.wait(timeout=2)
        timeline.append("search_end")
        return []

    app.state.pipeline.search.side_effect = blocking_search

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
        search_task = asyncio.create_task(http.post("/search", json={"query": "hello"}))
        # Give the search a chance to get underway before readiness is asked.
        await asyncio.sleep(0.05)

        health = await http.get("/health")
        timeline.append("health_end")

        release_search.set()
        await search_task

    assert health.status_code == 200
    assert "search_start" in timeline, "the search endpoint never reached the pipeline"
    # Held on the event loop, the search would have finished before readiness
    # could be answered at all.
    assert timeline.index("health_end") < timeline.index("search_end"), (
        f"readiness was served only after the search finished: {timeline}"
    )


def test_ingest_files_success(client):
    client.app.state.pipeline.ingest_file.return_value = (
        {
            "source_id": "uuid-file",
            "source_type": "file",
            "name": "report.pdf",
            "mime_type": "application/pdf",
            "status": "indexed",
        },
        {"parse_ms": 10.0, "chunk_ms": 1.0, "embed_ms": 2.0, "index_ms": 0.5},
    )

    response = client.post(
        "/sources/files",
        files={"files": ("report.pdf", BytesIO(b"pdf content"), "application/pdf")},
    )

    assert response.status_code == 202
    body = response.json()
    assert len(body) == 1
    assert body[0]["source_type"] == "file"
    assert body[0]["status"] == "pending"
    assert "job_id" in body[0]
    assert "source_id" in body[0]


def test_ingest_files_empty_file(client):
    response = client.post(
        "/sources/files",
        files={"files": ("empty.pdf", BytesIO(b""), "application/pdf")},
    )

    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_ingest_files_unsupported_type(client):
    client.app.state.pipeline.converter.is_supported_file.return_value = False

    response = client.post(
        "/sources/files",
        files={"files": ("archive.zip", BytesIO(b"zip"), "application/zip")},
    )

    assert response.status_code == 400
    assert "unsupported" in response.json()["detail"].lower()


def test_ingest_urls_success(client):
    client.app.state.pipeline.ingest_url.return_value = (
        {
            "source_id": "uuid-url",
            "source_type": "url",
            "name": "https://example.com",
            "final_url": "https://example.com/",
            "status": "indexed",
        },
        {"parse_ms": 10.0, "chunk_ms": 1.0, "embed_ms": 2.0, "index_ms": 0.5},
    )

    response = client.post(
        "/sources/urls",
        json={"urls": ["https://example.com"]},
    )

    assert response.status_code == 202
    body = response.json()
    assert len(body) == 1
    assert body[0]["source_type"] == "url"
    assert body[0]["status"] == "pending"
    assert "job_id" in body[0]
    assert "source_id" in body[0]


def test_ingest_urls_invalid_url(client):
    response = client.post(
        "/sources/urls",
        json={"urls": ["not-a-url"]},
    )

    assert response.status_code == 400
    assert "invalid" in response.json()["detail"].lower()


def test_same_url_submitted_twice_queues_one_job(client):
    """One URL submitted twice in a request is one source, not two jobs.

    Both submissions derive the same source identity, so two jobs would race
    two replacements of the same page. The request reports the source once.
    """
    client.app.state.pipeline.ingest_url.return_value = (
        {"source_id": "uuid-url", "source_type": "url", "name": "https://example.com"},
        {"parse_ms": 10.0, "chunk_ms": 1.0, "embed_ms": 2.0, "index_ms": 0.5},
    )

    response = client.post(
        "/sources/urls",
        json={"urls": ["https://example.com", "https://example.com", "https://other.example"]},
    )

    assert response.status_code == 202
    body = response.json()
    # The repeat collapsed, and the URL that was not a repeat still got its job.
    assert [item["name"] for item in body] == ["https://example.com", "https://other.example"]
    assert client.app.state.pipeline.ingest_url.call_count == 2


def test_same_filename_uploaded_twice_queues_one_job(client):
    """One file uploaded twice in a request is one source, not two jobs."""
    client.app.state.pipeline.ingest_file.return_value = (
        {"source_id": "uuid-file", "source_type": "file", "name": "report.pdf"},
        {"parse_ms": 10.0, "chunk_ms": 1.0, "embed_ms": 2.0, "index_ms": 0.5},
    )

    response = client.post(
        "/sources/files",
        files=[
            ("files", ("report.pdf", BytesIO(b"pdf content"), "application/pdf")),
            ("files", ("report.pdf", BytesIO(b"pdf content"), "application/pdf")),
            ("files", ("notes.pdf", BytesIO(b"more content"), "application/pdf")),
        ],
    )

    assert response.status_code == 202
    assert [item["name"] for item in response.json()] == ["report.pdf", "notes.pdf"]
    assert client.app.state.pipeline.ingest_file.call_count == 2


def test_a_rejected_duplicate_is_still_rejected(client):
    """Collapsing repeats must not skip validating them.

    A repeated upload is still checked, so a second copy that is empty or of an
    unsupported type is reported rather than silently dropped.
    """
    response = client.post(
        "/sources/files",
        files=[
            ("files", ("report.pdf", BytesIO(b"pdf content"), "application/pdf")),
            ("files", ("report.pdf", BytesIO(b""), "application/pdf")),
        ],
    )

    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_search_success(client):
    client.app.state.pipeline.search.return_value = [
        {
            "text": "chunk one",
            "score": 0.9,
            "source_id": "s1",
            "source_type": "file",
            "source_name": "report.pdf",
            "position": 3,
            "neighbours_before": 3,
            "neighbours_after": 2,
        },
        {
            "text": "chunk two",
            "score": 0.8,
            "source_id": "s2",
            "source_type": "url",
            "source_name": "https://example.com",
            "position": 0,
            "neighbours_before": 0,
            "neighbours_after": 0,
        },
    ]

    response = client.post("/search", json={"query": "test", "top_k": 2})

    assert response.status_code == 200
    body = response.json()
    assert len(body["results"]) == 2
    assert body["results"][0]["text"] == "chunk one"
    assert body["results"][0]["position"] == 3
    assert body["results"][0]["neighbours_before"] == 3
    assert body["results"][0]["neighbours_after"] == 2
    assert body["results"][1]["position"] == 0
    assert body["results"][1]["neighbours_before"] == 0
    assert body["results"][1]["neighbours_after"] == 0
    client.app.state.pipeline.search.assert_called_once_with("test", top_k=2)


def test_search_empty_query(client):
    response = client.post("/search", json={"query": "   "})

    assert response.status_code == 400


def test_search_uses_default_top_k(client):
    client.app.state.pipeline.search.return_value = []

    response = client.post("/search", json={"query": "hello"})

    assert response.status_code == 200
    client.app.state.pipeline.search.assert_called_once_with("hello", top_k=5)


def test_openapi_file_upload_schema_is_binary(client):
    response = client.get("/openapi.json")
    assert response.status_code == 200
    schema = response.json()
    body_schema = schema["components"]["schemas"]["Body_ingest_files_sources_files_post"]
    files_items = body_schema["properties"]["files"]["items"]
    assert files_items["type"] == "string"
    assert files_items["format"] == "binary"


def test_a_search_that_does_not_ask_for_neighbours_passes_no_count(client):
    """A caller that wants ranked hits only is given the call this route always made.

    The default is not sent to the pipeline, so "asking for no neighbours behaves
    as it always has" is structural rather than a promise about the callee.
    """
    client.app.state.pipeline.search.return_value = []

    response = client.post("/search", json={"query": "test", "top_k": 2})

    assert response.status_code == 200
    client.app.state.pipeline.search.assert_called_once_with("test", top_k=2)


def test_the_neighbour_count_reaches_the_pipeline_when_asked_for(client):
    client.app.state.pipeline.search.return_value = []

    response = client.post("/search", json={"query": "test", "top_k": 2, "neighbours": 2})

    assert response.status_code == 200
    client.app.state.pipeline.search.assert_called_once_with("test", top_k=2, neighbours=2)


def test_the_requested_neighbours_reach_the_response_inside_the_hit(client):
    """A neighbour is returned attached to its hit, not as a result of its own."""
    client.app.state.pipeline.search.return_value = [
        {
            "text": "chunk two",
            "score": 0.9,
            "source_id": "s1",
            "source_type": "file",
            "source_name": "report.pdf",
            "position": 2,
            "neighbours_before": 2,
            "neighbours_after": 1,
            "neighbours": [
                {"text": "chunk one", "position": 1},
                {"text": "chunk three", "position": 3},
            ],
        }
    ]

    response = client.post("/search", json={"query": "test", "top_k": 1, "neighbours": 1})

    assert response.status_code == 200
    body = response.json()
    assert len(body["results"]) == 1
    assert body["results"][0]["neighbours"] == [
        {"text": "chunk one", "position": 1},
        {"text": "chunk three", "position": 3},
    ]


def test_the_neighbour_count_is_bounded(client):
    """The count is small and non-negative: it widens context, it does not fetch a document."""
    for count in (-1, 6):
        response = client.post("/search", json={"query": "test", "neighbours": count})
        assert response.status_code == 422, count


def test_the_search_documentation_names_every_field_the_route_serves():
    """The documented surface is the served one.

    A field added to the request or the response without a line in the README's
    search section is one a caller cannot discover, so the two are checked
    against each other rather than trusted to stay in step.
    """
    readme = (Path(__file__).resolve().parent.parent / "README.md").read_text(encoding="utf-8")
    section = readme.split("### Search", 1)[1].split("## Configuration", 1)[0]

    for name in [*SearchRequest.model_fields, *SearchResult.model_fields]:
        assert name in section, f"{name} is served by /search but not documented"


def test_search_logs_elapsed_time(client, caplog):
    client.app.state.pipeline.search.return_value = [
        {
            "text": "chunk one",
            "score": 0.9,
            "source_id": "s1",
            "source_type": "file",
            "source_name": "report.pdf",
            "position": 0,
            "neighbours_before": 0,
            "neighbours_after": 4,
        }
    ]

    with caplog.at_level(logging.INFO, logger="doc_etl_api.routes"):
        response = client.post("/search", json={"query": "hello", "top_k": 1})

    assert response.status_code == 200
    assert "search_ms=" in caplog.text
    assert "results=1" in caplog.text


def test_search_logs_the_neighbour_count(client, caplog):
    """Widening a response is a cost, so the log says how wide it was asked to be."""
    client.app.state.pipeline.search.return_value = []

    with caplog.at_level(logging.INFO, logger="doc_etl_api.routes"):
        response = client.post("/search", json={"query": "hello", "top_k": 1, "neighbours": 3})

    assert response.status_code == 200
    assert "neighbours=3" in caplog.text
    assert "search_ms=" in caplog.text


def test_search_logs_elapsed_time_for_empty_results(client, caplog):
    client.app.state.pipeline.search.return_value = []

    with caplog.at_level(logging.INFO, logger="doc_etl_api.routes"):
        response = client.post("/search", json={"query": "nothing matches this"})

    assert response.status_code == 200
    assert response.json()["results"] == []
    assert "search_ms=" in caplog.text
    assert "results=0" in caplog.text


def test_get_job_status(client):
    # The double takes every argument the route passes to the pipeline, so a
    # signature that drifts from the real one shows up here as a failed job
    # rather than as a silently untested call.
    def _ingest_file(source_id, file, filename, mime_type=None, collections=()):
        return (
            {
                "source_id": source_id,
                "source_type": "file",
                "name": filename,
                "mime_type": mime_type,
                "status": "indexed",
            },
            {"parse_ms": 10.0, "chunk_ms": 1.0, "embed_ms": 2.0, "index_ms": 0.5},
        )

    client.app.state.pipeline.ingest_file.side_effect = _ingest_file

    response = client.post(
        "/sources/files",
        files={"files": ("report.pdf", BytesIO(b"pdf content"), "application/pdf")},
    )
    assert response.status_code == 202
    body = response.json()[0]
    job_id = body["job_id"]
    source_id = body["source_id"]

    response = client.get(f"/jobs/{job_id}")
    assert response.status_code == 200
    body = response.json()
    assert body["job_id"] == job_id
    assert body["source_id"] == source_id
    assert body["status"] in (JobStatus.COMPLETED.value, JobStatus.PENDING.value)
    if body["status"] == JobStatus.COMPLETED.value:
        assert body["result"]["source_id"] == source_id
        assert "parse_ms" in body["timings"]


# --- The chunking outcome reaching the job result ----------------------------


def _structural_pipeline() -> IndexPipeline:
    """A real pipeline over the stub embedder, so the outcome reported is real.

    The route's job result is what is under test, so the pipeline behind it is
    the real one: a double returning a hand-written outcome would prove only that
    the route copies a dict, not that the fields the chunker reports arrive.
    """
    pipeline = IndexPipeline(
        Settings(
            vector_store_backend="simple",
            embedding_model=EMBEDDING_MODEL_NAME,
            chunk_overlap=0,
            # The floor off, so each paragraph is stored as the node it is and
            # the counts below describe the boundaries rather than a merge.
            min_chunk_tokens=0,
        ),
        converter=MagicMock(),
        embedding_model=StubEmbedding(embed_dim=8),
    )
    pipeline.converter.is_supported_file.return_value = True
    pipeline.converter.convert_file.return_value = (
        "Under alpha, remark one stands entirely on its own.\n\n"
        "Under bravo, remark two stands entirely on its own."
    )
    return pipeline


def test_a_completed_job_reports_the_boundaries_that_produced_its_nodes(client):
    client.app.state.pipeline = _structural_pipeline()

    response = client.post(
        "/sources/files",
        files={"files": ("page.md", BytesIO(b"markdown"), "text/markdown")},
    )

    chunking = _job_result(client, response)["chunking"]
    assert chunking["nodes"] == 2, "the report does not describe the nodes that were stored"
    assert chunking["structural_nodes"] == 2, "the paragraphs were not reported as structural"
    assert chunking["divided_nodes"] == 0
    assert chunking["overlap_tokens"] == 0, "a structural boundary was reported as overlapped"
    assert chunking["max_node_tokens"] <= EMBEDDING_MAX_TOKENS


def test_get_job_not_found(client):
    response = client.get("/jobs/does-not-exist")
    assert response.status_code == 404


def test_get_job_failed(client):
    client.app.state.pipeline.ingest_file.side_effect = RuntimeError("conversion failed")

    response = client.post(
        "/sources/files",
        files={"files": ("report.pdf", BytesIO(b"pdf content"), "application/pdf")},
    )
    assert response.status_code == 202
    job_id = response.json()[0]["job_id"]

    response = client.get(f"/jobs/{job_id}")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == JobStatus.FAILED.value
    assert "conversion failed" in body["error"]


def test_models_preloaded_at_startup():
    converter = MagicMock()
    embedding = MagicMock()
    with patch("doc_etl_api.main._load_models", return_value=(converter, embedding)) as mock_load:
        with patch("doc_etl_api.main.create_pipeline") as mock_create_pipeline:
            app = create_app()
    mock_load.assert_called_once()
    mock_create_pipeline.assert_called_once_with(converter=converter, embedding_model=embedding)
    assert app.state.pipeline is not None
    assert app.state.jobs is not None


# --- Collections reaching the ingestion routes -------------------------------


def _echo_collections(source_id, file, filename, mime_type=None, collections=()):
    """Stand in for the pipeline, reporting back the collections it was handed."""
    return (
        {
            "source_id": source_id,
            "source_type": "file",
            "name": filename,
            "mime_type": mime_type,
            "collections": list(collections),
            "status": "indexed",
        },
        {"parse_ms": 10.0, "chunk_ms": 1.0, "embed_ms": 2.0, "index_ms": 0.5},
    )


def _echo_collections_url(source_id, url, timeout=None, collections=()):
    return (
        {
            "source_id": source_id,
            "source_type": "url",
            "name": url,
            "collections": list(collections),
            "status": "indexed",
        },
        {"parse_ms": 10.0, "chunk_ms": 1.0, "embed_ms": 2.0, "index_ms": 0.5},
    )


def _job_result(client, response) -> dict:
    """The result of the job the response reported, once it has run."""
    job_id = response.json()[0]["job_id"]
    body = client.get(f"/jobs/{job_id}").json()
    assert body["status"] == JobStatus.COMPLETED.value, body
    return body["result"]


def test_upload_naming_one_collection_reaches_the_ingestion_path(client):
    """A repeated form field is how a multipart upload names its collections."""
    client.app.state.pipeline.ingest_file.side_effect = _echo_collections

    response = client.post(
        "/sources/files",
        files={"files": ("report.pdf", BytesIO(b"pdf content"), "application/pdf")},
        data={"collections": ["csharp"]},
    )

    assert response.status_code == 202
    assert _job_result(client, response)["collections"] == ["csharp"]


def test_upload_names_several_collections_and_all_are_recorded(client):
    client.app.state.pipeline.ingest_file.side_effect = _echo_collections

    response = client.post(
        "/sources/files",
        files={"files": ("report.pdf", BytesIO(b"pdf content"), "application/pdf")},
        data={"collections": ["csharp", "dotnet", "csharp-12"]},
    )

    assert response.status_code == 202
    assert _job_result(client, response)["collections"] == ["csharp", "dotnet", "csharp-12"]


def test_upload_without_collections_still_ingests(client):
    """The field is optional: an upload that names none is ingested untagged."""
    client.app.state.pipeline.ingest_file.side_effect = _echo_collections

    response = client.post(
        "/sources/files",
        files={"files": ("report.pdf", BytesIO(b"pdf content"), "application/pdf")},
    )

    assert response.status_code == 202
    assert _job_result(client, response)["collections"] == []


@pytest.mark.parametrize(
    "collections",
    [
        ["CSharp"],
        ["my collection"],
        ["c" * 65],
        ["csharp", "dotnet", "c#"],
    ],
)
def test_an_invalid_collection_costs_the_request_and_no_job(client, collections):
    """Validation runs at the boundary, before any work is queued.

    A rejected request leaves nothing behind to ingest, so the caller can fix the
    name and resubmit rather than discovering the failure in a job's error later.
    """
    response = client.post(
        "/sources/files",
        files={"files": ("report.pdf", BytesIO(b"pdf content"), "application/pdf")},
        data={"collections": collections},
    )

    assert response.status_code == 400
    # The message names the value that was refused, not just that something was.
    assert repr(collections[-1]) in response.json()["detail"]
    assert client.app.state.jobs.in_flight == 0
    client.app.state.pipeline.ingest_file.assert_not_called()


def test_too_many_collections_costs_the_request_and_no_job(client):
    """Names that are each valid are still refused as a set that is too large."""
    collections = [f"tag-{index}" for index in range(MAX_COLLECTIONS_PER_REQUEST + 1)]

    response = client.post(
        "/sources/files",
        files={"files": ("report.pdf", BytesIO(b"pdf content"), "application/pdf")},
        data={"collections": collections},
    )

    assert response.status_code == 400
    assert str(MAX_COLLECTIONS_PER_REQUEST) in response.json()["detail"]
    assert client.app.state.jobs.in_flight == 0
    client.app.state.pipeline.ingest_file.assert_not_called()


def test_url_submission_naming_a_collection_reaches_the_ingestion_path(client):
    client.app.state.pipeline.ingest_url.side_effect = _echo_collections_url

    response = client.post(
        "/sources/urls",
        json={"urls": ["https://example.com"], "collections": ["csharp"]},
    )

    assert response.status_code == 202
    assert _job_result(client, response)["collections"] == ["csharp"]


@pytest.mark.parametrize("collections", [["CSharp"], ["my collection"], ["c" * 65]])
def test_an_invalid_url_collection_costs_the_submission_and_no_job(client, collections):
    response = client.post(
        "/sources/urls",
        json={"urls": ["https://example.com"], "collections": collections},
    )

    assert response.status_code == 400
    assert repr(collections[0]) in response.json()["detail"]
    assert client.app.state.jobs.in_flight == 0
    client.app.state.pipeline.ingest_url.assert_not_called()


# --- Search scoping at the boundary ------------------------------------------


def test_search_without_a_filter_is_unchanged(client):
    """Omitting the filter calls the pipeline exactly as it did before filtering.

    Not a filter that happens to match everything: the pipeline is handed no
    collections at all, which is what keeps "unfiltered search behaves as it
    always has" a property of the call rather than of the filter's semantics.
    """
    client.app.state.pipeline.search.return_value = []

    response = client.post("/search", json={"query": "test", "top_k": 2})

    assert response.status_code == 200
    client.app.state.pipeline.search.assert_called_once_with("test", top_k=2)


def test_search_with_a_filter_passes_the_collections(client):
    client.app.state.pipeline.search.return_value = []

    response = client.post("/search", json={"query": "test", "top_k": 2, "collections": ["csharp"]})

    assert response.status_code == 200
    client.app.state.pipeline.search.assert_called_once_with(
        "test", top_k=2, collections=["csharp"]
    )


def test_an_empty_collection_filter_is_rejected_rather_than_widening(client):
    """ "Search nothing" and "search everything" both read into an empty list.

    Answering a scoped question with every indexed source is the worse of the two
    readings, so the empty list is refused instead of being taken for no filter.
    """
    client.app.state.pipeline.search.return_value = []
    client.app.state.pipeline.search.side_effect = AssertionError("no search should be performed")

    response = client.post("/search", json={"query": "test", "collections": []})

    assert response.status_code == 400
    assert "collections" in response.json()["detail"]
    client.app.state.pipeline.search.assert_not_called()


def test_an_invalid_collection_filter_is_rejected(client):
    client.app.state.pipeline.search.side_effect = AssertionError("no search should be performed")

    response = client.post("/search", json={"query": "test", "collections": ["CSharp"]})

    assert response.status_code == 400
    assert repr("CSharp") in response.json()["detail"]
    client.app.state.pipeline.search.assert_not_called()


# --- The source catalog ------------------------------------------------------


def _catalog(*records) -> tuple:
    return tuple(records)


def test_the_catalog_reports_an_ingested_source(client):
    client.app.state.pipeline.source_catalog = _catalog(
        SourceRecord(
            name="report.pdf",
            source_type="file",
            collections=("csharp", "dotnet"),
            chunk_count=12,
        )
    )

    response = client.get("/sources")

    assert response.status_code == 200
    assert response.json() == {
        "sources": [
            {
                "name": "report.pdf",
                "source_type": "file",
                "collections": ["csharp", "dotnet"],
                "chunk_count": 12,
            }
        ]
    }


def test_the_catalog_lists_every_source(client):
    client.app.state.pipeline.source_catalog = _catalog(
        SourceRecord(name="report.pdf", source_type="file", collections=(), chunk_count=3),
        SourceRecord(
            name="https://example.com", source_type="url", collections=("csharp",), chunk_count=7
        ),
    )

    response = client.get("/sources")

    assert response.status_code == 200
    body = response.json()
    assert [source["name"] for source in body["sources"]] == [
        "report.pdf",
        "https://example.com",
    ]
    assert [source["source_type"] for source in body["sources"]] == ["file", "url"]


def test_the_catalog_is_empty_and_successful_with_nothing_indexed(client):
    client.app.state.pipeline.source_catalog = _catalog()

    response = client.get("/sources")

    assert response.status_code == 200
    assert response.json() == {"sources": []}


def test_an_untagged_source_appears_with_an_empty_collection_list(client):
    """The catalog is what makes an untagged source visible at all.

    A source in no collection matches no filter, so "which sources are in no
    collection" is only answerable by listing them and reading the empty lists.
    """
    client.app.state.pipeline.source_catalog = _catalog(
        SourceRecord(name="plain.pdf", source_type="file", collections=(), chunk_count=4)
    )

    response = client.get("/sources")

    assert response.status_code == 200
    assert response.json()["sources"][0]["collections"] == []


def test_the_catalog_performs_no_retrieval_or_embedding(client):
    pipeline = client.app.state.pipeline
    pipeline.source_catalog = _catalog(
        SourceRecord(name="report.pdf", source_type="file", collections=("csharp",), chunk_count=2)
    )
    # Any attempt to reach the index or the embedding model must explode, so a
    # passing test proves the catalog did neither.
    pipeline.index.as_retriever.side_effect = AssertionError("the catalog must not retrieve")
    pipeline._embedding_model.get_text_embedding.side_effect = AssertionError(
        "the catalog must not embed"
    )

    response = client.get("/sources")

    assert response.status_code == 200
    assert len(response.json()["sources"]) == 1
    pipeline.search.assert_not_called()
