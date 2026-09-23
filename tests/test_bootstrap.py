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

    def ingest_file(source_id, file, filename, mime_type=None, collections=()):
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


# --- Corpus tagging ----------------------------------------------------------


def _corpus_settings(
    corpus_dir,
    collections: str,
    urls: str = "",
    file_collections: dict[str, list[str]] | None = None,
) -> Settings:
    return Settings(
        vector_store_backend="simple",
        embedding_model=EMBEDDING_MODEL_NAME,
        chunk_size=128,
        chunk_overlap=10,
        knowledge_corpus_dir=str(corpus_dir),
        knowledge_corpus_urls=urls,
        knowledge_corpus_collections=collections,
        knowledge_corpus_file_collections=file_collections or {},
    )


def _tagged_pipeline(corpus_dir, **settings_kwargs) -> tuple[IndexPipeline, Settings]:
    """A real pipeline over a stubbed converter, with a text file per source."""
    settings = _corpus_settings(corpus_dir, **settings_kwargs)
    pipeline = _corpus_pipeline(settings)
    pipeline.converter.SUPPORTED_FILE_EXTENSIONS = {".txt"}
    pipeline.converter.convert_file.side_effect = lambda _file, name: f"Corpus file {name}."
    pipeline.converter.convert_url.return_value = ("Corpus page body.", "https://example.com/page")
    return pipeline, settings


def _tags_by_name(pipeline: IndexPipeline) -> dict[str, tuple[str, ...]]:
    return {record.name: record.collections for record in pipeline.source_catalog}


def _corpus_pipeline(settings: Settings) -> IndexPipeline:
    """A real pipeline over a stubbed converter, so ingestion runs for real."""
    return IndexPipeline(
        settings,
        converter=MagicMock(),
        embedding_model=StubEmbedding(embed_dim=8),
    )


def test_corpus_collections_tag_every_file_and_url(corpus_dir):
    """The corpus is tagged through the ordinary ingestion path, as uploads are.

    Applied in `ingest_corpus` rather than by a corpus-only code path, so what a
    corpus source is tagged with is exactly what an upload naming the same
    collection would be tagged with.
    """
    settings = _corpus_settings(
        corpus_dir, collections="csharp, dotnet", urls="https://example.com/page"
    )
    pipeline = _corpus_pipeline(settings)
    pipeline.converter.SUPPORTED_FILE_EXTENSIONS = {".txt"}
    pipeline.converter.convert_file.side_effect = lambda _file, name: f"Corpus file {name}."
    pipeline.converter.convert_url.return_value = ("Corpus page body.", "https://example.com/page")
    state = BootstrapState()

    ingest_corpus(pipeline, settings, state)

    assert state.status is BootstrapStatus.COMPLETE
    assert state.failures == []
    # Both kinds of corpus source reached the index, and both are tagged.
    assert {record.source_type for record in pipeline.source_catalog} == {"file", "url"}
    assert {record.collections for record in pipeline.source_catalog} == {("csharp", "dotnet")}
    # Tagged in the stored metadata too, which is what a filter reads.
    for node in pipeline.index.storage_context.docstore.docs.values():
        assert node.metadata["collections"] == ["csharp", "dotnet"]


@pytest.mark.parametrize("collections", ["", "   ", ","])
def test_an_unset_corpus_collection_leaves_the_corpus_untagged(corpus_dir, collections):
    """No configured collection is not a failure: the corpus loads untagged."""
    settings = _corpus_settings(corpus_dir, collections=collections)
    pipeline = _corpus_pipeline(settings)
    pipeline.converter.SUPPORTED_FILE_EXTENSIONS = {".txt"}
    pipeline.converter.convert_file.side_effect = lambda _file, name: f"Corpus file {name}."
    state = BootstrapState()

    ingest_corpus(pipeline, settings, state)

    assert state.status is BootstrapStatus.COMPLETE
    assert state.failures == []
    assert pipeline.indexed_sources == 2
    assert {record.collections for record in pipeline.source_catalog} == {()}


def test_corpus_content_is_reachable_by_a_filtered_search(corpus_dir):
    """Without the setting the corpus is the one thing no filter can see.

    The corpus is the content that exists to ground answers, so a filtered search
    that cannot reach it is answering a scoped question from everything except the
    material kept on hand for it. The contrast below is the point: the identical
    search reaches the corpus only when the corpus was tagged.
    """
    text = "Every corpus file records its own subject in its own wording."
    tagged_settings = _corpus_settings(corpus_dir, collections="csharp")
    tagged = _corpus_pipeline(tagged_settings)
    tagged.converter.SUPPORTED_FILE_EXTENSIONS = {".txt"}
    tagged.converter.convert_file.side_effect = lambda _file, _name: text

    untagged_settings = _corpus_settings(corpus_dir, collections="")
    untagged = _corpus_pipeline(untagged_settings)
    untagged.converter.SUPPORTED_FILE_EXTENSIONS = {".txt"}
    untagged.converter.convert_file.side_effect = lambda _file, _name: text

    for pipeline, settings in ((tagged, tagged_settings), (untagged, untagged_settings)):
        ingest_corpus(pipeline, settings, BootstrapState())
        assert pipeline.indexed_sources == 2, "the corpus did not load"

    results = tagged.search("subject", top_k=5, collections=["csharp"])
    assert results
    assert {result["source_name"] for result in results} == {"alpha.txt", "beta.txt"}
    # The same search over the same content, differing only in the setting.
    assert untagged.search("subject", top_k=5, collections=["csharp"]) == []


# --- Per-file corpus collections ---------------------------------------------


def test_a_per_file_entry_overrides_the_corpus_collections(corpus_dir):
    """One corpus directory can hold documents belonging to different collections.

    The entry replaces the corpus collections for the file it names. A merge
    would be the wrong shape: the file would then also answer a search for the
    corpus default, which is a collection it has nothing to do with.
    """
    pipeline, settings = _tagged_pipeline(
        corpus_dir, collections="csharp", file_collections={"alpha.txt": ["dotnet"]}
    )
    state = BootstrapState()

    ingest_corpus(pipeline, settings, state)

    assert state.status is BootstrapStatus.COMPLETE
    assert state.failures == []
    assert _tags_by_name(pipeline) == {"alpha.txt": ("dotnet",), "beta.txt": ("csharp",)}
    # The override reaches the stored metadata too, which is what a filter reads.
    by_name = {
        node.ref_doc_id: node.metadata["collections"]
        for node in pipeline.index.storage_context.docstore.docs.values()
    }
    assert by_name["alpha.txt"] == ["dotnet"]
    assert by_name["beta.txt"] == ["csharp"]


def test_a_per_file_entry_applies_with_no_corpus_collections(corpus_dir):
    """An entry does not need a corpus-wide default to be there."""
    pipeline, settings = _tagged_pipeline(
        corpus_dir, collections="", file_collections={"alpha.txt": ["dotnet"]}
    )

    ingest_corpus(pipeline, settings, BootstrapState())

    assert _tags_by_name(pipeline) == {"alpha.txt": ("dotnet",), "beta.txt": ()}


def test_an_entry_leaves_url_sources_on_the_corpus_collections(corpus_dir):
    """An entry is keyed by filename, so it cannot reach a URL source."""
    pipeline, settings = _tagged_pipeline(
        corpus_dir,
        collections="csharp",
        urls="https://example.com/page",
        file_collections={"alpha.txt": ["dotnet"]},
    )

    ingest_corpus(pipeline, settings, BootstrapState())

    url_tags = {r.collections for r in pipeline.source_catalog if r.source_type == "url"}
    assert url_tags == {("csharp",)}


def test_a_per_file_entry_tag_is_reachable_by_a_filtered_search(corpus_dir):
    """The point of the entry: the file is findable under its own collection.

    Both files hold the same wording, so the collection is the only difference
    between the two searches.
    """
    text = "Every corpus file records its own subject in its own wording."
    settings = _corpus_settings(
        corpus_dir, collections="csharp", file_collections={"alpha.txt": ["dotnet"]}
    )
    pipeline = _corpus_pipeline(settings)
    pipeline.converter.SUPPORTED_FILE_EXTENSIONS = {".txt"}
    pipeline.converter.convert_file.side_effect = lambda _file, _name: text

    ingest_corpus(pipeline, settings, BootstrapState())

    assert {
        r["source_name"] for r in pipeline.search("subject", top_k=5, collections=["dotnet"])
    } == {"alpha.txt"}
    assert {
        r["source_name"] for r in pipeline.search("subject", top_k=5, collections=["csharp"])
    } == {"beta.txt"}


def test_an_entry_naming_a_file_the_corpus_does_not_ingest_is_a_failure(corpus_dir, caplog):
    """A filename typo tags nothing, so it must not pass silently.

    `ignored.zip` is in the corpus directory but is not a supported document
    type, so it is not ingested either: the tag the entry asked for was never
    applied, and a failure is the only outcome that says so.
    """
    pipeline, settings = _tagged_pipeline(
        corpus_dir,
        collections="csharp",
        file_collections={"gamma.txt": ["dotnet"], "ignored.zip": ["dotnet"]},
    )
    state = BootstrapState()

    with caplog.at_level(logging.ERROR, logger="doc_etl_api.bootstrap"):
        ingest_corpus(pipeline, settings, state)

    assert state.status is BootstrapStatus.FAILED
    assert {failure.split(":")[0] for failure in state.failures} == {"gamma.txt", "ignored.zip"}
    assert "gamma.txt" in caplog.text
    # The rest of the corpus still loaded, under the collections it was given.
    assert _tags_by_name(pipeline) == {"alpha.txt": ("csharp",), "beta.txt": ("csharp",)}


def test_the_bootstrap_log_reports_both_tag_sources(corpus_dir, caplog):
    """One line shows where each file's tag came from, which a wrong tag needs."""
    pipeline, settings = _tagged_pipeline(
        corpus_dir, collections="csharp", file_collections={"alpha.txt": ["dotnet"]}
    )

    with caplog.at_level(logging.INFO, logger="doc_etl_api.bootstrap"):
        ingest_corpus(pipeline, settings, BootstrapState())

    assert "collections=['csharp']" in caplog.text
    assert "file_collections={'alpha.txt': ['dotnet']}" in caplog.text


def test_a_corpus_source_is_fetchable_by_its_filename(tmp_path):
    """The corpus and an upload agree on what an address is."""
    from unittest.mock import patch

    from fastapi.testclient import TestClient

    from doc_etl_api.jobs import JobRegistry
    from doc_etl_api.main import create_app

    (tmp_path / "handbook.txt").write_text("handbook")
    pipeline = IndexPipeline(
        Settings(knowledge_corpus_dir=str(tmp_path), embedding_model=EMBEDDING_MODEL_NAME),
        converter=MagicMock(),
        embedding_model=StubEmbedding(embed_dim=8),
    )
    pipeline.converter.SUPPORTED_FILE_EXTENSIONS = {".txt"}
    pipeline.converter.convert_file.return_value = "# Handbook\n\nGrounding content."
    state = BootstrapState()

    ingest_corpus(pipeline, Settings(knowledge_corpus_dir=str(tmp_path)), state)
    assert state.status is BootstrapStatus.COMPLETE

    with patch("doc_etl_api.main._load_models", return_value=(MagicMock(), MagicMock())):
        with patch("doc_etl_api.main.create_pipeline", return_value=MagicMock()):
            app = create_app()
    app.state.pipeline = pipeline
    app.state.jobs = JobRegistry()

    response = TestClient(app).get("/sources/content", params={"address": "handbook.txt"})

    assert response.status_code == 200
    body = response.json()
    assert body["name"] == "handbook.txt"
    assert body["chunks"], "the corpus source stored no chunks"
    assert "Grounding content" in body["chunks"][0]["text"]
