import logging
from io import BytesIO
from unittest.mock import MagicMock

import pytest

from doc_etl_api.bootstrap import BootstrapState, BootstrapStatus, corpus_files, ingest_corpus
from doc_etl_api.config import Settings
from doc_etl_api.pipeline import IndexPipeline
from tests.stubs import EMBEDDING_MODEL_NAME, StubEmbedding


@pytest.fixture
def corpus_dir(tmp_path):
    (tmp_path / "alpha.txt").write_text("alpha")
    (tmp_path / "beta.txt").write_text("beta")
    (tmp_path / "ignored.zip").write_bytes(b"zip")
    return tmp_path


def test_corpus_files_filters_by_supported_extension(corpus_dir):
    files = corpus_files(corpus_dir, {".txt"})

    assert [f.name for f in files] == ["alpha.txt", "beta.txt"]


def test_corpus_files_missing_directory_is_empty(tmp_path):
    assert corpus_files(tmp_path / "nope", {".txt"}) == []
    assert corpus_files(None, {".txt"}) == []


def test_bootstrap_disabled_when_no_corpus_configured():
    pipeline = MagicMock()
    state = BootstrapState()

    ingest_corpus(pipeline, Settings(), state)

    assert state.status is BootstrapStatus.DISABLED
    pipeline.ingest_file.assert_not_called()
    pipeline.ingest_url.assert_not_called()


def test_bootstrap_ingests_every_file_and_url(corpus_dir):
    pipeline = MagicMock()
    pipeline.converter.SUPPORTED_FILE_EXTENSIONS = {".txt"}
    state = BootstrapState()
    app_settings = Settings(
        knowledge_corpus_dir=str(corpus_dir),
        knowledge_corpus_urls="https://example.com/a,https://example.com/b",
    )

    ingest_corpus(pipeline, app_settings, state)

    assert state.status is BootstrapStatus.COMPLETE
    assert state.failures == []
    assert {c.kwargs["filename"] for c in pipeline.ingest_file.call_args_list} == {
        "alpha.txt",
        "beta.txt",
    }
    assert [c.kwargs["url"] for c in pipeline.ingest_url.call_args_list] == [
        "https://example.com/a",
        "https://example.com/b",
    ]


def test_bootstrap_isolates_a_failing_source(corpus_dir, caplog):
    pipeline = MagicMock()
    pipeline.converter.SUPPORTED_FILE_EXTENSIONS = {".txt"}

    def ingest_file(source_id, file, filename, mime_type=None):
        if filename == "beta.txt":
            raise RuntimeError("unreadable")
        return {}, {}

    pipeline.ingest_file.side_effect = ingest_file
    state = BootstrapState()

    with caplog.at_level(logging.ERROR, logger="doc_etl_api.bootstrap"):
        ingest_corpus(
            pipeline,
            Settings(
                knowledge_corpus_dir=str(corpus_dir),
                knowledge_corpus_urls="https://example.com/a",
            ),
            state,
        )

    # Every source was attempted, not just the ones before the failure.
    assert pipeline.ingest_file.call_count == 2
    assert pipeline.ingest_url.call_count == 1
    assert state.status is BootstrapStatus.FAILED
    assert len(state.failures) == 1
    assert "beta.txt" in state.failures[0]
    assert "unreadable" in state.failures[0]
    assert "beta.txt" in caplog.text


def test_bootstrap_runs_without_blocking_startup(monkeypatch, tmp_path):
    """Startup must return while the corpus is still loading."""
    import threading
    from unittest.mock import patch

    from fastapi.testclient import TestClient

    from doc_etl_api.main import create_app

    release = threading.Event()
    finished = threading.Event()

    def blocking_bootstrap(pipeline, app_settings, state):
        state.status = BootstrapStatus.IN_PROGRESS
        release.wait(timeout=5)
        state.status = BootstrapStatus.COMPLETE
        finished.set()

    monkeypatch.setattr("doc_etl_api.main.settings", Settings(knowledge_corpus_dir=str(tmp_path)))
    monkeypatch.setattr("doc_etl_api.main.ingest_corpus", blocking_bootstrap)

    with patch("doc_etl_api.main._load_models", return_value=(MagicMock(), MagicMock())):
        with patch("doc_etl_api.main.create_pipeline", return_value=MagicMock()):
            app = create_app()

    app.state.pipeline.indexed_sources = 0
    app.state.pipeline.indexed_chunks = 0
    client = TestClient(app)

    # Startup has already returned even though the corpus is still loading.
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["bootstrap"] in ("pending", "in_progress")

    release.set()
    assert finished.wait(timeout=5), "the bootstrap thread never finished"

    response = client.get("/health")
    assert response.json()["bootstrap"] == "complete"


def test_bootstrap_stays_disabled_without_a_corpus(monkeypatch):
    from unittest.mock import patch

    from fastapi.testclient import TestClient

    from doc_etl_api.main import create_app

    monkeypatch.setattr("doc_etl_api.main.settings", Settings())

    with patch("doc_etl_api.main._load_models", return_value=(MagicMock(), MagicMock())):
        with patch("doc_etl_api.main.create_pipeline", return_value=MagicMock()):
            app = create_app()

    app.state.pipeline.indexed_sources = 0
    app.state.pipeline.indexed_chunks = 0

    response = TestClient(app).get("/health")

    assert response.json()["bootstrap"] == "disabled"


# --- Corpus identity: a corpus file and a later upload of it are one source ---


def _identity_pipeline() -> IndexPipeline:
    """A real pipeline with a stubbed converter, so identity is really derived."""
    return IndexPipeline(
        Settings(
            vector_store_backend="simple",
            embedding_model=EMBEDDING_MODEL_NAME,
            chunk_size=128,
            chunk_overlap=10,
        ),
        converter=MagicMock(),
        embedding_model=StubEmbedding(embed_dim=8),
    )


def _stored_texts(pipeline: IndexPipeline, source_key: str) -> list[str]:
    return [
        node.get_content()
        for node in pipeline.index.storage_context.docstore.docs.values()
        if node.ref_doc_id == source_key
    ]


def test_corpus_file_and_a_later_upload_share_one_source(corpus_dir):
    """A corpus file and an upload of that same file are one source, not two.

    The bootstrap goes through the ordinary ingestion path, so a corpus file's
    identity is its name -- the same identity an upload of that name derives.
    Re-uploading a corpus file therefore replaces the copy the corpus put there
    rather than indexing a second copy beside it.
    """
    pipeline = _identity_pipeline()
    pipeline.converter.SUPPORTED_FILE_EXTENSIONS = {".txt"}
    pipeline.converter.convert_file.side_effect = lambda _file, name: f"corpus content of {name}"

    ingest_corpus(pipeline, Settings(knowledge_corpus_dir=str(corpus_dir)), BootstrapState())

    assert set(pipeline.index.ref_doc_info) == {"alpha.txt", "beta.txt"}
    assert pipeline.indexed_sources == 2

    # The corpus file is later uploaded again, with edited content.
    pipeline.converter.convert_file.side_effect = lambda _file, _name: "edited content"
    pipeline.ingest_file(source_id="upload", file=BytesIO(b"edited"), filename="alpha.txt")

    assert set(pipeline.index.ref_doc_info) == {"alpha.txt", "beta.txt"}
    assert pipeline.indexed_sources == 2, "the upload was indexed as a source of its own"

    # The upload replaced the corpus copy of that one file, leaving the rest of
    # the corpus alone.
    stored = _stored_texts(pipeline, "alpha.txt")
    assert any("edited content" in text for text in stored)
    assert not any("corpus content" in text for text in stored)
    assert any("corpus content of beta.txt" in text for text in _stored_texts(pipeline, "beta.txt"))
