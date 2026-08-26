from io import BytesIO
from unittest.mock import MagicMock, patch

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
