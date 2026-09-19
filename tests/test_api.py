import asyncio
import logging
import threading
from io import BytesIO
from unittest.mock import MagicMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient

from doc_etl_api.jobs import JobRegistry, JobStatus
from doc_etl_api.main import create_app


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
        },
        {
            "text": "chunk two",
            "score": 0.8,
            "source_id": "s2",
            "source_type": "url",
            "source_name": "https://example.com",
        },
    ]

    response = client.post("/search", json={"query": "test", "top_k": 2})

    assert response.status_code == 200
    body = response.json()
    assert len(body["results"]) == 2
    assert body["results"][0]["text"] == "chunk one"
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


def test_search_logs_elapsed_time(client, caplog):
    client.app.state.pipeline.search.return_value = [
        {
            "text": "chunk one",
            "score": 0.9,
            "source_id": "s1",
            "source_type": "file",
            "source_name": "report.pdf",
        }
    ]

    with caplog.at_level(logging.INFO, logger="doc_etl_api.routes"):
        response = client.post("/search", json={"query": "hello", "top_k": 1})

    assert response.status_code == 200
    assert "search_ms=" in caplog.text
    assert "results=1" in caplog.text


def test_search_logs_elapsed_time_for_empty_results(client, caplog):
    client.app.state.pipeline.search.return_value = []

    with caplog.at_level(logging.INFO, logger="doc_etl_api.routes"):
        response = client.post("/search", json={"query": "nothing matches this"})

    assert response.status_code == 200
    assert response.json()["results"] == []
    assert "search_ms=" in caplog.text
    assert "results=0" in caplog.text


def test_get_job_status(client):
    def _ingest_file(source_id, file, filename, mime_type=None):
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
