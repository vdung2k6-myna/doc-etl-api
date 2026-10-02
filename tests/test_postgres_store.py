"""The durable backend, against a real Postgres.

These are the tests that need a database. They are skipped unless one is named in
``DOC_ETL_API_TEST_POSTGRES_URL``, so the suite stays green on a machine without
one -- which is the whole of CI -- and reporting as skipped rather than passing
without having established anything. Point it at a database with the vector
extension and at one this suite may create tables in: every test builds a
collection named for itself and drops it afterwards. The compose service under
the ``postgres`` profile is such a database::

    export DOC_ETL_API_TEST_POSTGRES_URL=\\
        postgresql+psycopg2://doc_etl_api:verify@localhost:55432/doc_etl_api
    python -m pytest tests/test_postgres_store.py
"""

import itertools
import json
import os
import subprocess
import sys
import threading
import time
from collections.abc import Iterator, Sequence
from functools import partial
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest
import sqlalchemy
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.engine import make_url
from sqlalchemy.pool import NullPool

from doc_etl_api.bootstrap import BootstrapState, BootstrapStatus, ingest_corpus
from doc_etl_api.claims import claim_heartbeat
from doc_etl_api.config import Settings, VectorStoreBackend
from doc_etl_api.main import create_app
from doc_etl_api.pipeline import IndexPipeline, _node_id, content_hash, create_pipeline
from doc_etl_api.store import DurableStoreError, PostgresIndexStore
from tests.stubs import EMBEDDING_MODEL_NAME, StubEmbedding

DATABASE_URL = os.environ.get("DOC_ETL_API_TEST_POSTGRES_URL", "")

needs_database = pytest.mark.skipif(
    not DATABASE_URL,
    reason="set DOC_ETL_API_TEST_POSTGRES_URL to run the durable backend tests",
)

# Run by a second interpreter over the same database, reporting what it finds as
# one line of JSON. It reuses this module's own helpers so the collection it reads
# is built the way the collection it was written by was.
_PROBE_SOURCE = """
import json
import os

from tests.test_postgres_store import _create, _settings

pipeline = _create(_settings(table=os.environ["PROBE_TABLE"]))
record, chunks = pipeline.source_content("docs.txt")
print(
    json.dumps(
        {
            "sources": pipeline.indexed_sources,
            "address": record.address,
            "digest": record.content_hash,
            "collections": list(record.collections),
            "positions": [chunk["position"] for chunk in chunks],
        }
    )
)
pipeline._store.close()
"""

# A paragraph of prose, long enough to split at the stub model's input limit and
# short enough that its own text is one chunk: the difference is what lets a test
# tell a re-ingestion from the content it replaced.
PARAGRAPH = ("The index keeps every chunk of a source in reading order. " * 5).strip()
LONG_DOC = "\n\n".join(
    [
        PARAGRAPH,
        PARAGRAPH.replace("reading order", "source order"),
        PARAGRAPH.replace("a source", "one source"),
    ]
)

# Two documents of different length, for the tests that race two instances
# against one source. A copy of one is then distinguishable from a copy of the
# other by how many chunks it holds -- which is what makes a mixture of the two
# visible as a source recording one document's count and holding both. Each
# carries a word the other does not, so that the copy can also be shown to be one
# document's rather than a splice of both.
ALPHA_DOC = PARAGRAPH.replace("reading order", "ALPHAMARK order")
BRAVO_DOC = "\n\n".join(
    [
        PARAGRAPH.replace("reading order", "BRAVOMARK order"),
        PARAGRAPH.replace("a source", "BRAVOMARK one source"),
        PARAGRAPH.replace("every chunk", "BRAVOMARK all chunks"),
    ]
)

_COLLECTIONS = itertools.count(1)


def _settings(**overrides) -> Settings:
    """Settings for the test database, with the collection named by the test."""
    url = make_url(DATABASE_URL) if DATABASE_URL else None
    return Settings(
        embedding_model=overrides.pop("embedding_model", EMBEDDING_MODEL_NAME),
        chunk_overlap=10,
        vector_store_backend=overrides.pop("backend", VectorStoreBackend.POSTGRES),
        postgres_host=overrides.pop("host", (url and url.host) or "localhost"),
        postgres_port=overrides.pop("port", (url and url.port) or 5432),
        postgres_database=overrides.pop("database", (url and url.database) or ""),
        postgres_user=overrides.pop("user", (url and url.username) or ""),
        postgres_password=overrides.pop("password", (url and url.password) or ""),
        postgres_table_name=overrides.pop("table", "doc_etl_api_test"),
        **overrides,
    )


def _create(settings: Settings, markdown: str = "", embed_dim: int = 8) -> IndexPipeline:
    converter = MagicMock()
    converter.convert_file.return_value = markdown
    return create_pipeline(
        settings, converter=converter, embedding_model=StubEmbedding(embed_dim=embed_dim)
    )


def _ingest(
    pipeline: IndexPipeline,
    filename: str,
    markdown: str,
    collections: Sequence[str] = (),
    payload: bytes | None = None,
):
    """Ingest a source, with bytes of its own so its digest is its own.

    The bytes are what a source is recorded under, so two sources given the same
    ones would be indistinguishable in the catalog by digest alone -- and the
    assertions below that a digest survived a restart would then pass even if the
    store had written one source's digest for both.
    """
    pipeline.converter.convert_file.return_value = markdown
    return pipeline.ingest_file(
        source_id=f"id-{filename}",
        file=BytesIO(payload or f"the bytes of {filename}".encode()),
        filename=filename,
        collections=collections,
    )[0]


def _drop_collection(table: str) -> None:
    """Remove every table one collection could have created.

    Named explicitly rather than by prefix: this runs against a database someone
    may also be using, and a `DROP TABLE` that matched a pattern is one bad
    pattern away from deleting somebody else's index.
    """
    engine = sqlalchemy.create_engine(DATABASE_URL, poolclass=NullPool)
    names = [
        f"data_{table}",
        f"data_{table}_docstore",
        f"{table}_sources",
        f"{table}_positions",
        f"{table}_model",
        f"{table}_jobs",
        f"{table}_claims",
    ]
    try:
        with engine.begin() as connection:
            for name in names:
                connection.execute(sqlalchemy.text(f'DROP TABLE IF EXISTS "{name}" CASCADE'))
    finally:
        engine.dispose()


def _row_count(table: str) -> int:
    engine = sqlalchemy.create_engine(DATABASE_URL, poolclass=NullPool)
    try:
        with engine.connect() as connection:
            return connection.execute(
                sqlalchemy.text(f'SELECT count(*) FROM "{table}"')
            ).scalar_one()
    finally:
        engine.dispose()


def _wait_for_bootstrap(client: TestClient, timeout: float = 30.0) -> str:
    """The bootstrap's final state, once it reports one.

    The bootstrap runs on a worker thread, so startup returns before the corpus
    is loaded; the readiness endpoint is how a caller learns that it finished.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        state = client.get("/health").json()["bootstrap"]
        if state in ("complete", "failed", "disabled"):
            return state
        time.sleep(0.05)
    raise AssertionError("the corpus bootstrap never finished")


@pytest.fixture
def collection() -> Iterator[str]:
    """A collection name no other test uses, dropped before and after it runs.

    Dropped before as well, so a run that died between a test's ingestion and its
    teardown does not hand the next run a database it did not build.
    """
    name = f"doc_etl_api_test_{next(_COLLECTIONS)}"
    _drop_collection(name)
    yield name
    _drop_collection(name)


@pytest.fixture
def durable(collection: str) -> Iterator[IndexPipeline]:
    """A pipeline on the durable backend, closed when the test is done.

    Closed because a pipeline holds a connection pool of its own: a suite that
    builds one per test and never hands it back runs out of connections on the
    server, and then fails for a reason that has nothing to do with what it tests.
    """
    pipeline = _create(_settings(table=collection), markdown=LONG_DOC)
    yield pipeline
    pipeline._store.close()


@pytest.fixture
def restarted(collection: str) -> Iterator[IndexPipeline]:
    """A pipeline rebuilt over what an earlier one ingested and left behind.

    Two sources in two collections, ingested by a pipeline that is then closed --
    which is what a restart is, as far as the store is concerned.
    """
    first = _create(_settings(table=collection), markdown=LONG_DOC)
    _ingest(first, "docs.txt", LONG_DOC, collections=["csharp"])
    _ingest(first, "notes.txt", LONG_DOC, collections=["dotnet"])
    first._store.close()

    rebuilt = _create(_settings(table=collection), markdown=LONG_DOC)
    yield rebuilt
    rebuilt._store.close()


@needs_database
def test_the_postgres_backend_builds_the_postgres_store(durable):
    """The durable member of the dispatch builds a store in the database.

    The collection is created rather than assumed, and the docstore override is
    on: without it the index writes the vectors and leaves the docstore empty,
    so every read that goes through the docstore -- the content read and the
    neighbour lookup -- would answer nothing while search still returned hits.
    """
    assert isinstance(durable._store, PostgresIndexStore)
    assert durable._store.store_nodes_override is True
    # Built by the store's own setup, so a collection exists to be validated
    # before anything is ingested -- and empty, because setup writes nothing.
    assert _row_count(durable._store.vector_table_name) == 0


@needs_database
def test_the_collection_records_the_model_that_built_it(durable, collection):
    """A collection carries the model that built it, for the next deployment to read.

    The width in the vector column is what the database enforces and is checked
    against the model in use; this row is what lets a mismatch say *which* model,
    and it is written once, by whichever deployment built the collection.
    """
    engine = sqlalchemy.create_engine(DATABASE_URL, poolclass=NullPool)
    try:
        with engine.connect() as connection:
            rows = connection.execute(
                sqlalchemy.text(f'SELECT model_name, embed_dim FROM "{collection}_model"')
            ).all()
    finally:
        engine.dispose()

    assert rows == [(EMBEDDING_MODEL_NAME, 8)], f"unexpected model row: {rows}"


@needs_database
def test_a_model_of_another_width_is_refused_at_startup(collection):
    """A model whose vectors do not fit the collection is refused, naming both.

    The column's width is what a write is checked against, and it is created once
    and never widened: a second deployment with a wider model would be refused by
    the database at the first ingest -- after the document had been parsed and
    embedded -- so it is refused at startup instead, naming both widths.
    """
    built = _create(_settings(table=collection), embed_dim=8)
    built._store.close()

    with pytest.raises(DurableStoreError) as refused:
        _create(_settings(table=collection), embed_dim=16)

    message = str(refused.value)
    assert "16" in message, message
    assert "8" in message, message


@needs_database
def test_a_collection_built_by_another_model_is_refused_at_startup(collection):
    """Two models of the same width mix silently, so the model is checked by name.

    Every vector is comparable to every other of the same width and none of them
    means anything across models, so a collection holding both would rank hits by
    a distance that is not a distance. The refusal names both models rather than
    only the two numbers.
    """
    built = _create(_settings(table=collection))
    built._store.close()

    with pytest.raises(DurableStoreError) as refused:
        _create(_settings(table=collection, embedding_model="sentence-transformers/another-model"))

    message = str(refused.value)
    assert "another-model" in message
    assert EMBEDDING_MODEL_NAME in message


def test_an_unreachable_database_is_refused_naming_the_backend_and_the_store():
    """A durable backend that cannot be reached stops startup, naming both.

    Which database is unreachable is the part an operator acts on, and which
    backend asked for it is the part that says what to change. This test needs no
    database: the port it names is one nothing can be listening on.
    """
    with pytest.raises(DurableStoreError) as refused:
        _create(
            _settings(
                host="localhost",
                port=1,
                database="doc_etl_api",
                user="doc_etl_api",
                password="unused",
            )
        )

    message = str(refused.value)
    assert VectorStoreBackend.POSTGRES.value in message
    assert "localhost:1/doc_etl_api" in message


@needs_database
def test_a_rebuilt_pipeline_holds_what_the_last_one_ingested(restarted):
    """A restart does not discard the index, the catalog, or the text.

    The three are what a restart has to bring back together: the vectors search
    reads, the catalog the readiness endpoint and the catalog route report, and
    the docstore the content route reads and a hit's neighbours come from. A
    rebuild that brought back only the vectors would answer a search and then
    report nothing for the hit it returned.
    """
    assert restarted.indexed_sources == 2
    assert {record.name for record in restarted.source_catalog} == {"docs.txt", "notes.txt"}
    assert restarted.indexed_chunks == sum(
        record.chunk_count for record in restarted.source_catalog
    )

    record, chunks = restarted.source_content("docs.txt")
    assert record.collections == ("csharp",)
    assert [chunk["position"] for chunk in chunks] == list(range(len(chunks)))
    assert all(chunk["text"].strip() for chunk in chunks)

    hits = restarted.search("reading order", top_k=5, neighbours=1)
    assert hits
    for hit in hits:
        assert hit["position"] >= 0
        assert hit["neighbours"], "a hit's neighbours come from the docstore, and none came back"


@needs_database
def test_a_rebuilt_pipeline_holds_each_sources_digest(restarted):
    """Each source's digest comes back with its record, per source.

    The digest is what startup compares to decide whether a corpus source needs
    ingesting at all, and it is the one field of a record that describes content
    the store does not otherwise hold -- so a store that dropped it would make
    every restart re-ingest the whole corpus while reporting a catalog that looks
    complete.
    """
    digests = {record.address: record.content_hash for record in restarted.source_catalog}

    assert digests == {
        "docs.txt": content_hash(b"the bytes of docs.txt"),
        "notes.txt": content_hash(b"the bytes of notes.txt"),
    }
    # Each source is current as it stands recorded: its content and the
    # collections it was ingested under both came back from the database.
    assert all(
        restarted.is_current(record.address, record.content_hash, record.collections)
        for record in restarted.source_catalog
    )
    assert not restarted.is_current("docs.txt", digests["docs.txt"], ["dotnet"])


@needs_database
def test_a_rebuilt_pipeline_still_filters_by_collection(restarted):
    """The prefilter runs in the store, so a rebuild has to keep its metadata.

    Collections are read off the stored node rather than held beside it, which is
    what makes a scoped search a scoped *scan*: if the rebuild lost them, the
    filter would match nothing and the search would return nothing rather than
    something unscoped -- the safer failure, and still a failure.
    """
    scoped = restarted.search("reading order", top_k=5, collections=["csharp"])

    assert scoped, "the scoped search returned nothing"
    assert {tuple(hit["collections"]) for hit in scoped} == {("csharp",)}
    assert {hit["address"] for hit in scoped} == {"docs.txt"}


@needs_database
def test_re_ingesting_a_source_replaces_it_after_a_restart(collection):
    """A source re-ingested by a rebuilt pipeline replaces what it held before.

    Deleting a source's earlier content is by document identity, and the record of
    which nodes that source holds lives in the docstore -- so a rebuild that
    brought the nodes back without it would leave the replaced content stored
    beside the new: the content route would report both revisions and search would
    return chunks of a document that no longer exists.
    """
    first = _create(_settings(table=collection), markdown=LONG_DOC)
    _ingest(first, "docs.txt", LONG_DOC, collections=["csharp"])
    longest = first.source_catalog[0].chunk_count
    first._store.close()

    rebuilt = _create(_settings(table=collection), markdown=PARAGRAPH)
    _ingest(rebuilt, "docs.txt", PARAGRAPH, collections=["csharp"])

    assert rebuilt.indexed_sources == 1
    record, chunks = rebuilt.source_content("docs.txt")
    assert len(chunks) == record.chunk_count < longest, "the replaced content is still stored"
    assert _row_count(f"{rebuilt._store.vector_table_name}") == record.chunk_count
    assert _row_count(f"{collection}_positions") == record.chunk_count
    rebuilt._store.close()


# --- Both startup paths together, through the routes --------------------------


def _stubbed_models(contents: dict[str, str], parsed: list[str]) -> MagicMock:
    """A converter over named content, recording each file it is asked to parse.

    The parse is the stage a restart over a corpus it already holds must skip, so
    it is the call that has to be countable; the fetch refuses outright, since
    no URL is configured for this corpus.
    """
    converter = MagicMock()
    converter.is_supported_file.return_value = True
    converter.SUPPORTED_FILE_EXTENSIONS = {".txt"}

    def convert_file(file, filename):
        parsed.append(filename)
        return contents[filename]

    converter.convert_file.side_effect = convert_file
    converter.fetch_page.side_effect = AssertionError("no corpus URL is configured")
    return converter


def _started_app(converter: MagicMock, app_settings: Settings) -> FastAPI:
    """An application on the durable backend, over the stubbed models.

    The settings are supplied to the pipeline builder as well as to the module
    the application reads: `create_app` builds its pipeline with whatever the
    module-level settings are, and a test that patched only the module would
    otherwise get an application that reports one backend and indexes into
    another -- which is the failure this test exists to catch, arriving through
    the test's own door.
    """
    with patch(
        "doc_etl_api.main._load_models", return_value=(converter, StubEmbedding(embed_dim=8))
    ):
        with patch("doc_etl_api.main.create_pipeline", partial(create_pipeline, app_settings)):
            return create_app()


@needs_database
def test_a_restart_keeps_both_an_uploaded_document_and_a_configured_corpus(
    monkeypatch, tmp_path, collection
):
    """A restart over a real Postgres, through the routes, with both paths live.

    What a restart has to bring back is everything a running service answers
    from: the search that finds a document, the content endpoint that returns its
    chunks in reading order, the catalog that reports both sources, and the
    collections that scope a search to one of them. The corpus is configured so
    that the same startup also has to decide what it already holds -- including
    the document a caller uploaded before the restart, which is not in the corpus
    at all and survives only because the durable store kept it.

    The second startup is asserted to parse nothing: a restart that brought the
    index back by reconstructing it from the corpus would satisfy every
    assertion about what is searchable and still be the failure this whole
    change exists to remove.
    """
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    contents = {
        "guide.txt": "# Guide\n\nThe STAR method structures behavioural answers.",
        # Long enough to be stored as several chunks, so the content endpoint's
        # reading order is an order rather than a single value.
        "handbook.txt": "\n\n".join(
            [
                "# Handbook",
                "The XYZZY convention records unrelated remarks.",
                "A later section restates the convention at length. " * 60,
            ]
        ),
    }
    (corpus / "guide.txt").write_text(contents["guide.txt"])
    app_settings = _settings(table=collection, knowledge_corpus_dir=str(corpus))
    monkeypatch.setattr("doc_etl_api.main.settings", app_settings)

    first_parses: list[str] = []
    first = _started_app(_stubbed_models(contents, first_parses), app_settings)
    client = TestClient(first)
    # The application has to be on the durable backend for the restart to mean
    # anything: built on the in-process one it would answer every assertion below
    # from a store that a second application cannot see.
    assert isinstance(first.state.pipeline._store, PostgresIndexStore)

    assert _wait_for_bootstrap(client) == "complete"
    assert first_parses == ["guide.txt"], "the first startup did not load the corpus"

    accepted = client.post(
        "/sources/files",
        files={"files": ("handbook.txt", BytesIO(b"the bytes of handbook.txt"), "text/plain")},
        data={"collections": ["csharp"]},
    )
    assert accepted.status_code == 202, accepted.text

    before = client.get("/health").json()
    assert before["indexed_sources"] == 2
    assert before["indexed_chunks"] > 2, "the uploaded document was stored as one chunk"
    first.state.pipeline._store.close()

    # The restart: a second application over the same collection, with the same
    # corpus configured.
    second_parses: list[str] = []
    second = _started_app(_stubbed_models(contents, second_parses), app_settings)
    restarted = TestClient(second)
    assert isinstance(second.state.pipeline._store, PostgresIndexStore)

    assert _wait_for_bootstrap(restarted) == "complete"
    assert second_parses == [], "the restart parsed a corpus the index already held"

    health = restarted.get("/health").json()
    assert health["status"] == "ready"
    assert health["indexed_sources"] == 2
    assert health["indexed_chunks"] == before["indexed_chunks"]

    catalog = restarted.get("/sources").json()["sources"]
    assert {source["name"]: source["collections"] for source in catalog} == {
        "handbook.txt": ["csharp"],
        "guide.txt": [],
    }

    results = restarted.post("/search", json={"query": "conventions", "top_k": 10}).json()[
        "results"
    ]
    assert {result["source_name"] for result in results} == {"guide.txt", "handbook.txt"}

    body = restarted.get("/sources/content", params={"address": "handbook.txt"}).json()
    assert body["name"] == "handbook.txt"
    positions = [chunk["position"] for chunk in body["chunks"]]
    assert body["chunks"], "the content endpoint returned no chunks after the restart"
    assert positions == list(range(len(positions))), "the chunks came back out of order"
    assert "XYZZY" in " ".join(chunk["text"] for chunk in body["chunks"])

    scoped = restarted.post(
        "/search", json={"query": "conventions", "top_k": 10, "collections": ["csharp"]}
    ).json()["results"]
    assert scoped, "the collection no longer filters after the restart"
    assert {result["source_name"] for result in scoped} == {"handbook.txt"}

    second.state.pipeline._store.close()


@needs_database
def test_a_search_is_answered_while_an_ingestion_writes(monkeypatch, tmp_path, collection):
    """A search submitted during an ingestion job is answered, and the job lands.

    Through the routes, on the durable backend, with the two arriving together:
    the upload is accepted and runs on a worker of the server's own while the
    search is served, so a search that comes in at that moment has to get its
    answer rather than wait for the server to fall idle -- which is what makes the
    worker the ingestion runs on and the worker the search runs on separate things
    the service has to keep separate.

    The write is held open so the search is submitted inside it, and the search's
    own answer is stamped with that: what it has to be is a 200 carrying
    well-formed results rather than a failure or a half-written source. What the
    results are drawn from is the index as it stands when the search is served,
    which is after the write it waited on -- the search is held by the same lock
    the replacement holds, so a source is never served out of an ingestion that
    has not finished. That the uploaded document is among the results once the job
    has run is what says the two requests really did overlap and land.
    """
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    contents = {
        "guide.txt": "# Guide\n\nThe STAR method structures behavioural answers.",
        "handbook.txt": "# Handbook\n\nThe XYZZY convention records unrelated remarks.",
    }
    (corpus / "guide.txt").write_text(contents["guide.txt"])
    app_settings = _settings(table=collection, knowledge_corpus_dir=str(corpus))
    monkeypatch.setattr("doc_etl_api.main.settings", app_settings)

    app = _started_app(_stubbed_models(contents, []), app_settings)
    client = TestClient(app)
    try:
        assert _wait_for_bootstrap(client) == "complete"

        pipeline = app.state.pipeline
        # Set while the ingestion is inside its index write and cleared when it
        # leaves it, so reading it when the search is submitted says whether the
        # two really were in flight together or merely close in time.
        inside_the_write = threading.Event()
        original = type(pipeline._index).insert_nodes

        def held_open(self, nodes, **kwargs):
            inside_the_write.set()
            try:
                time.sleep(0.5)
                return original(self, nodes, **kwargs)
            finally:
                inside_the_write.clear()

        monkeypatch.setattr(type(pipeline._index), "insert_nodes", held_open)

        accepted: list[httpx.Response] = []

        def upload() -> None:
            accepted.append(
                client.post(
                    "/sources/files",
                    files={
                        "files": (
                            "handbook.txt",
                            BytesIO(b"the bytes of handbook.txt"),
                            "text/plain",
                        )
                    },
                )
            )

        uploading = threading.Thread(target=upload)
        uploading.start()
        try:
            assert inside_the_write.wait(timeout=60), "the ingestion never reached the write"
            overlapped = inside_the_write.is_set()
            answered = client.post("/search", json={"query": "STAR", "top_k": 5})
        finally:
            uploading.join(timeout=60)
            monkeypatch.setattr(type(pipeline._index), "insert_nodes", original)

        assert overlapped, "the search was submitted after the write had finished"
        assert accepted and accepted[0].status_code == 202, [r.status_code for r in accepted]
        assert answered.status_code == 200, answered.text
        results = answered.json()["results"]
        assert results, "the search returned nothing rather than the corpus it holds"
        # Every hit has to be a chunk a source holds, whole: a source served while
        # it was being written would come back as a passage with no source, a
        # splice of two documents, or a chunk of one that is not in it any more.
        held = {
            chunk["text"]
            for name in contents
            for chunk in client.get("/sources/content", params={"address": name}).json()["chunks"]
        }
        assert len(held) >= len(contents), "the sources were not stored as whole chunks"
        for result in results:
            assert result["text"] in held, "a result came back carrying text no source holds"

        # Both sources whole and searchable once the job has run, which is what
        # says the two requests overlapped rather than one happening after the
        # other: the upload's document is in the index the second search reads.
        landed = client.post("/search", json={"query": "XYZZY", "top_k": 5}).json()["results"]
        assert {result["source_name"] for result in landed} == set(contents), (
            "the ingested document did not become searchable once its job had run"
        )
        assert any("XYZZY" in result["text"] for result in landed), (
            "the ingested document came back without its own text"
        )
    finally:
        app.state.pipeline._store.close()


@needs_database
def test_a_second_process_reads_what_this_one_ingested(collection):
    """What survives a restart is the database, proved across an interpreter.

    Rebuilding the pipeline in one process is what a restart looks like from the
    store's side, but it is not the boundary the claim is about: a catalog kept
    in a module, a class attribute or a closure would survive that and not this.
    So the index is written by this process and read by another one that shares
    nothing with it but the connection string and the table name.
    """
    pipeline = _create(_settings(table=collection), markdown=LONG_DOC)
    _ingest(pipeline, "docs.txt", LONG_DOC, collections=["csharp"])
    written = pipeline.source_catalog[0]
    assert written.chunk_count > 1, "the document was stored as one chunk"
    pipeline._store.close()

    probe = subprocess.run(
        [sys.executable, "-c", _PROBE_SOURCE],
        cwd=str(Path(__file__).resolve().parent.parent),
        env={
            **os.environ,
            "DOC_ETL_API_TEST_POSTGRES_URL": DATABASE_URL,
            "PROBE_TABLE": collection,
        },
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert probe.returncode == 0, f"the second process failed:\n{probe.stderr}"

    read_back = json.loads(probe.stdout.strip().splitlines()[-1])
    assert read_back["sources"] == 1
    assert read_back["address"] == "docs.txt"
    assert read_back["digest"] == written.content_hash
    assert read_back["collections"] == ["csharp"]
    assert read_back["positions"] == list(range(written.chunk_count))


def _committed_state(engine, table: str, address: str) -> tuple[int, int, int]:
    """What one committed instant says about a source: recorded, held, positioned.

    Read on a connection of its own, which is what makes this an observation of
    what an instance has published rather than of what this process is midway
    through writing. The three numbers are the chunk count the catalog records,
    the nodes the vector table holds for the source, and the positions the
    catalog holds for it; a replacement that is one write keeps them agreeing at
    every instant, and one that is three leaves a reader arriving between two of
    them holding a source whose record counts chunks it does not have.
    """
    with engine.connect() as connection:
        recorded = connection.execute(
            sqlalchemy.text(f'SELECT chunk_count FROM "{table}_sources" WHERE address = :address'),
            {"address": address},
        ).scalar_one_or_none()
        held = connection.execute(
            sqlalchemy.text(
                f"SELECT count(*) FROM \"data_{table}\" WHERE metadata_->>'ref_doc_id' = :address"
            ),
            {"address": address},
        ).scalar_one()
        positioned = connection.execute(
            sqlalchemy.text(f'SELECT count(*) FROM "{table}_positions" WHERE address = :address'),
            {"address": address},
        ).scalar_one()
    return (recorded or 0, held, positioned)


def _observing_engine():
    """An engine for watching another instance, on connections of its own.

    `NullPool` because a pooled connection would be handed back to the pool
    between observations, and a reader is meant to arrive on the database afresh
    each time -- which is what a second instance does.
    """
    return sqlalchemy.create_engine(DATABASE_URL, poolclass=NullPool)


@needs_database
def test_two_instances_ingesting_one_filename_end_with_one_copy(collection):
    """Two instances replacing one source at once leave one copy, not two.

    Both start from a source neither has written, which is the case a row lock
    has to answer: with nothing stored there, neither instance's removal takes a
    lock on anything, and both would insert their own nodes beside the other's.
    What serializes them is the catalog row each claims first, so the second
    waits for the first to commit and then replaces the whole of what it wrote.

    The two documents are deliberately of different length, so a copy of one is
    distinguishable from a copy of the other by how many chunks it holds: a
    mixture of the two would be a source whose recorded count is one document's
    and whose content is longer than that.
    """
    alpha = _create(_settings(table=collection), markdown=ALPHA_DOC)
    bravo = _create(_settings(table=collection), markdown=BRAVO_DOC)
    measured = _create(_settings(table=collection), markdown=ALPHA_DOC)
    _ingest(measured, "alpha-probe.txt", ALPHA_DOC)
    _ingest(measured, "bravo-probe.txt", BRAVO_DOC)
    alpha_chunks = measured.source_content("alpha-probe.txt")[0].chunk_count
    bravo_chunks = measured.source_content("bravo-probe.txt")[0].chunk_count
    assert alpha_chunks != bravo_chunks, "the two documents chunk to the same length"

    start = threading.Barrier(2)
    failures: list[BaseException] = []

    def ingest(pipeline: IndexPipeline, markdown: str) -> None:
        try:
            start.wait(timeout=30)
            _ingest(pipeline, "shared.txt", markdown)
        except BaseException as exc:  # noqa: BLE001 -- reported by the assertion below
            failures.append(exc)

    threads = [
        threading.Thread(target=ingest, args=(alpha, ALPHA_DOC)),
        threading.Thread(target=ingest, args=(bravo, BRAVO_DOC)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=120)
    assert not failures, failures

    # A third instance, built over what the two left behind, because the two
    # writers each hold a catalog snapshot of their own that only their own
    # ingestion refreshed.
    reader = _create(_settings(table=collection), markdown=ALPHA_DOC)
    record, chunks = reader.source_content("shared.txt")

    assert record.chunk_count in (alpha_chunks, bravo_chunks), (
        "the source holds a chunk count belonging to neither ingestion"
    )
    assert len(chunks) == record.chunk_count, (
        "the source holds more or fewer chunks than it records"
    )
    assert _committed_state(_observing_engine(), collection, "shared.txt") == (
        record.chunk_count,
        record.chunk_count,
        record.chunk_count,
    )

    # And the copy is one document's, whole: only one of the two markers is in it.
    text = " ".join(chunk["text"] for chunk in chunks)
    assert ("ALPHAMARK" in text) != ("BRAVOMARK" in text), "the source holds both documents"

    for pipeline in (alpha, bravo, measured, reader):
        pipeline._store.close()


@needs_database
def test_a_reader_never_observes_a_half_replaced_source(collection, monkeypatch):
    """A source is never observably missing its content while it is replaced.

    The reader polls the database on connections of its own, which is what another
    instance's read is: committed state and nothing else. The writer's node insert
    is held open, and every observation is stamped with whether the writer was
    inside that window when it was taken -- so the test can point at readings
    taken while the replacement was half done rather than hope it caught one. Each
    of those readings has to be the source as it was before, whole; a replacement
    that committed its removal before its insert would be observed here as a
    source whose catalog counts chunks the vector table does not hold, which is
    the failure this test exists to be able to see.
    """
    writer = _create(_settings(table=collection), markdown=ALPHA_DOC)
    _ingest(writer, "shared.txt", ALPHA_DOC)
    before = _committed_state(_observing_engine(), collection, "shared.txt")
    assert before[0] > 0

    original = type(writer._index).insert_nodes
    inside_the_write = threading.Event()

    def held_open(self, nodes, **kwargs):
        inside_the_write.set()
        try:
            time.sleep(0.3)
            return original(self, nodes, **kwargs)
        finally:
            inside_the_write.clear()

    monkeypatch.setattr(type(writer._index), "insert_nodes", held_open)

    engine = _observing_engine()
    observations: list[tuple[tuple[int, int, int], bool]] = []
    reading = threading.Event()
    reading.set()

    def read() -> None:
        while reading.is_set():
            observations.append(
                (_committed_state(engine, collection, "shared.txt"), inside_the_write.is_set())
            )
            time.sleep(0.005)

    reader = threading.Thread(target=read)
    reader.start()
    try:
        _ingest(writer, "shared.txt", BRAVO_DOC)
    finally:
        reading.clear()
        reader.join(timeout=30)
        engine.dispose()

    after = _committed_state(_observing_engine(), collection, "shared.txt")
    assert after[0] != before[0], "the replacement did not land, so nothing was observed"

    during = [state for state, writing in observations if writing]
    assert during, "no reading was taken while the replacement was half done"
    assert set(during) == {before}, "a reader saw the source as something other than it was"

    for state, _ in observations:
        assert state[0] == state[1] == state[2], (
            f"a reader saw a source recorded as {state[0]} chunks holding {state[1]} "
            f"nodes and {state[2]} positions"
        )
        assert state in (before, after)

    writer._store.close()


@needs_database
def test_a_search_on_another_instance_never_sees_a_half_replaced_source(collection, monkeypatch):
    """A search served by a second instance answers with one whole copy, never a splice.

    The reader is an instance of its own: its own index, its own connections, its
    own lock. Nothing here is serialized by one process's lock, so what it can
    observe is committed state -- which is what makes the replacement being a single
    transaction the thing under test rather than a second, weaker guarantee that
    happens to hold here.

    The searches run while the writer's insert is held open, and each is stamped
    with whether it was served inside that window, so the test points at searches
    that really overlapped the write instead of hoping to catch one. A search that
    caught the writer between its removal and its insert would answer with a source
    that is neither document, or with a splice of the two.

    The reader is built after the earlier copy is committed, so it is an instance
    that already knows this source. One that had never seen it would answer from an
    empty map, and its answer would say nothing about the replacement either way.
    """
    writer = _create(_settings(table=collection), markdown=ALPHA_DOC)
    _ingest(writer, "shared.txt", ALPHA_DOC)
    reader = _create(_settings(table=collection), markdown=ALPHA_DOC)
    earlier = {chunk["text"] for chunk in reader.source_content("shared.txt")[1]}
    assert earlier and any("ALPHAMARK" in text for text in earlier)

    original = type(writer._index).insert_nodes
    inside_the_write = threading.Event()

    def held_open(self, nodes, **kwargs):
        inside_the_write.set()
        try:
            time.sleep(0.3)
            return original(self, nodes, **kwargs)
        finally:
            inside_the_write.clear()

    monkeypatch.setattr(type(writer._index), "insert_nodes", held_open)

    searches: list[tuple[set[str], bool]] = []
    failures: list[BaseException] = []
    searching = threading.Event()
    searching.set()

    def search() -> None:
        while searching.is_set():
            try:
                hits = reader.search("The index keeps every chunk", 10)
                searches.append(({hit["text"] for hit in hits}, inside_the_write.is_set()))
            except BaseException as exc:  # noqa: BLE001 -- reported by the assertion below
                failures.append(exc)
                return
            time.sleep(0.005)

    thread = threading.Thread(target=search)
    thread.start()
    try:
        _ingest(writer, "shared.txt", BRAVO_DOC)
    finally:
        searching.clear()
        thread.join(timeout=60)

    assert not failures, failures

    # Read back by the same instance once the replacement has committed, so the two
    # whole copies the searches below are allowed to answer with are both ones this
    # reader really serves -- and so a reader that could never see the replacement
    # would fail here rather than pass the loop by answering the earlier copy always.
    replaced = {hit["text"] for hit in reader.search("The index keeps every chunk", 10)}
    assert any("BRAVOMARK" in text for text in replaced), "the replacement never became searchable"
    assert not (replaced & earlier), "the reader returned what the replacement came for as well"

    during = [texts for texts, writing in searches if writing]
    assert during, "no search was served while the replacement was half done"
    for texts, _ in searches:
        assert texts in (earlier, replaced), (
            f"a search returned {len(texts)} chunks that are neither copy of the source"
        )
    for texts in during:
        assert texts == earlier, "a search served mid-replacement returned the replacement"

    writer._store.close()
    reader._store.close()


@needs_database
def test_ingesting_one_document_twice_leaves_the_same_node_ids(durable, collection):
    """A second ingestion of identical content lands on the nodes the first wrote.

    A source's nodes are named by the source and the position they hold in it, so
    the same content offered twice describes the same nodes both times -- and the
    second ingestion's insert reaches identities the first one's already used
    rather than names of its own. That is what a replay converges on: an insert
    that minted a fresh identity per ingestion would leave this one's nodes beside
    the earlier one's wherever a deletion had not reached them, and search would
    then return the same passage twice, with the store holding both.

    The node count is checked against the rows the vector table holds and not only
    against the positions, because those two are separate writes and a replay that
    duplicated would show up in the first and not in the second.
    """
    _ingest(durable, "docs.txt", LONG_DOC)
    first = durable._store.positions("docs.txt")
    first_chunks = durable.source_content("docs.txt")[1]
    assert len(first) > 1, "the document was stored as one chunk"

    _ingest(durable, "docs.txt", LONG_DOC)
    second = durable._store.positions("docs.txt")
    second_chunks = durable.source_content("docs.txt")[1]

    assert second == first, "a second ingestion of the same content named different nodes"
    assert second_chunks == first_chunks, "the replay stored content the first ingestion did not"
    assert set(first) == {_node_id("docs.txt", position) for position in range(len(first))}

    engine = _observing_engine()
    try:
        assert _committed_state(engine, collection, "docs.txt") == (len(first),) * 3, (
            "the replay left rows behind rather than converging on the same nodes"
        )
    finally:
        engine.dispose()


@needs_database
def test_a_replacement_that_fails_leaves_the_earlier_content_in_place(
    durable, collection, monkeypatch
):
    """An interrupted replacement leaves the source as it was, and never empty.

    The interruption is placed at the worst point there is: after the source's
    earlier content has been removed and before its replacement has been written.
    A replacement that committed its removal first would be observed here as a
    source with nothing in it -- the earlier document gone, the new one never
    written, and no later ingestion to put either back. What has to be there
    instead is the earlier document in full: the same rows, the same recorded
    order, and the same record down to the collections it was ingested under.
    """
    _ingest(durable, "docs.txt", ALPHA_DOC, collections=["csharp"])
    before_record = durable.source_content("docs.txt")[0]

    engine = _observing_engine()
    try:
        before = _committed_state(engine, collection, "docs.txt")
        assert before[0] > 0

        original = type(durable._index).insert_nodes

        def refuse(self, nodes, **kwargs):
            # After the removal: `delete_ref_doc` has already run inside the same
            # replacement, and the replacement's own nodes have not been written.
            raise RuntimeError("the replacement was interrupted")

        monkeypatch.setattr(type(durable._index), "insert_nodes", refuse)
        with pytest.raises(RuntimeError):
            _ingest(durable, "docs.txt", BRAVO_DOC, collections=["dotnet"])
        monkeypatch.setattr(type(durable._index), "insert_nodes", original)

        assert _committed_state(engine, collection, "docs.txt") == before, (
            "the interrupted replacement left the source other than it was"
        )
    finally:
        engine.dispose()

    # Read back over the store rather than from the instance that failed, so the
    # answer is what is stored and not what the interrupted ingestion cached.
    reader = _create(_settings(table=collection), markdown=ALPHA_DOC)
    try:
        record, chunks = reader.source_content("docs.txt")
        assert record == before_record, "the interrupted replacement rewrote the record"
        assert len(chunks) == record.chunk_count, "the source came back short of its record"
        text = " ".join(chunk["text"] for chunk in chunks)
        assert "ALPHAMARK" in text, "the earlier document is not in the source"
        assert "BRAVOMARK" not in text, "the replacement is in the source it failed to be"
    finally:
        reader._store.close()


# --- A corpus, and the several instances that start from it -------------------


def _corpus_pipeline(
    settings: Settings, contents: dict[str, str], parsed: list[str], delay: float = 0.0
):
    """A pipeline over *contents* as a corpus directory, counting every parse.

    The conversion is the stage the claim exists to keep to one instance, so it
    is the call that has to be countable. Each file is converted to its own text,
    so what lands in the index is the file that was read rather than one fixed
    document for all of them.

    *delay* holds each conversion open for that long, which is how a test gives a
    source that outlasts an ownership threshold: the threshold is lowered to a
    second or two rather than the source made to clear the five-minute default.
    """
    pipeline = _create(settings)
    pipeline.converter.SUPPORTED_FILE_EXTENSIONS = {".txt"}
    lock = threading.Lock()

    def convert_file(file, name):
        with lock:
            parsed.append(name)
        if delay:
            time.sleep(delay)
        return contents[name]

    pipeline.converter.convert_file.side_effect = convert_file
    return pipeline


@needs_database
def test_instances_starting_together_ingest_each_corpus_source_once(tmp_path, collection):
    """Several instances over one corpus do one instance's worth of work.

    Both halves matter and neither implies the other: a corpus loaded zero times
    is "once per instance" avoided by losing the corpus, and a corpus loaded once
    per instance is the cost the claim exists to remove. So this asserts that
    every source is in the index afterwards *and* that its parse happened once.

    The two instances run at the same time on two threads, which is the ordering
    the claim is for: one of them takes each source, and the other waits for it
    and then finds the source already current rather than loading it again. They
    are named apart by hand because they are one process, and a claim tells two
    instances apart by name -- the default name would make them one instance.
    """
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    contents = {
        "alpha.txt": "# Alpha\n\nThe STAR method structures behavioural answers.",
        "beta.txt": "# Beta\n\nThe XYZZY convention records unrelated remarks.",
    }
    for name, text in contents.items():
        # Bytes rather than text, so the digest asserted below is the digest of
        # the file: writing text would translate its newlines on the way out and
        # the file would be what the corpus holds rather than what was written.
        (corpus / name).write_bytes(text.encode())

    settings = _settings(table=collection, knowledge_corpus_dir=str(corpus))
    parsed: list[str] = []
    first = _corpus_pipeline(settings, contents, parsed)
    second = _corpus_pipeline(settings, contents, parsed)
    # One claim store, because that is what the two instances share: the same
    # database row is what tells them apart.
    claims = first.store.claim_store()
    states = [BootstrapState(), BootstrapState()]

    try:
        threads = [
            threading.Thread(
                target=ingest_corpus,
                args=(pipeline, settings, state, claims, f"instance-{index}"),
                name=f"bootstrap-{index}",
            )
            for index, (pipeline, state) in enumerate(zip((first, second), states, strict=True))
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=60)
        assert not any(thread.is_alive() for thread in threads), (
            "an instance's bootstrap did not finish"
        )

        assert [state.status for state in states] == [BootstrapStatus.COMPLETE] * 2, [
            state.status for state in states
        ]
        assert states[0].failures == [] and states[1].failures == [], [
            states[0].failures,
            states[1].failures,
        ]
        assert sorted(parsed) == sorted(contents), (
            f"the corpus was parsed {len(parsed)} times rather than once per source"
        )

        for pipeline in (first, second):
            pipeline.refresh()
            for name, text in contents.items():
                assert pipeline.is_current(name, content_hash(text.encode()), []), (
                    f"{name} is not in the index this instance reports"
                )
            hits = pipeline.search("STAR", top_k=5)
            assert any("STAR" in hit["text"] for hit in hits), "the corpus is not searchable"
        assert claims.held("alpha.txt") is None, "a claim outlived the bootstrap that took it"
        assert claims.held("beta.txt") is None, "a claim outlived the bootstrap that took it"
    finally:
        first._store.close()
        second._store.close()


@needs_database
def test_a_claim_left_by_a_stopped_instance_is_taken_up(tmp_path, collection):
    """A source claimed by an instance that stopped is loaded by a survivor.

    An instance that stops mid-bootstrap leaves its claim behind: that is what
    stopping is, as far as a row is concerned, and it is the case the claim has
    to survive rather than the case it can assume away. What tells that claim
    apart from one whose owner is still working is its age -- it is taken over
    once it is older than the threshold job ownership uses -- so the wait is
    asserted as well as the ingestion. A survivor that took the source at once
    would duplicate the work the claim exists to keep to one instance; one that
    never took it would leave the source absent from the index, which is the
    other half of what this asserts against.
    """
    contents = {"guide.txt": "# Guide\n\nThe STAR method structures behavioural answers."}
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    for name, text in contents.items():
        (corpus / name).write_bytes(text.encode())

    threshold = 1
    settings = _settings(
        table=collection,
        knowledge_corpus_dir=str(corpus),
        job_orphan_threshold_seconds=threshold,
    )
    parsed: list[str] = []
    survivor = _corpus_pipeline(settings, contents, parsed)
    claims = survivor.store.claim_store()
    state = BootstrapState()

    # The claim a stopped instance leaves: taken, and never released, because
    # the process that would have released it is the one that stopped.
    assert claims.claim("guide.txt", "instance-that-stopped", threshold)

    try:
        started = time.monotonic()
        ingest_corpus(survivor, settings, state, claims, "instance-that-survived")
        elapsed = time.monotonic() - started

        assert state.status is BootstrapStatus.COMPLETE, state.failures
        assert state.failures == []
        assert parsed == ["guide.txt"], "the survivor did not load the stranded source"
        assert elapsed >= threshold, "the survivor took the claim before it was stale"
        assert claims.held("guide.txt") is None, "the claim outlived the bootstrap"
        survivor.refresh()
        assert survivor.is_current("guide.txt", content_hash(contents["guide.txt"].encode()), [])
        hits = survivor.search("STAR", top_k=5)
        assert any("STAR" in hit["text"] for hit in hits), "the source is absent from the index"
    finally:
        survivor._store.close()


@needs_database
def test_a_source_outlasting_the_threshold_is_loaded_by_one_instance(tmp_path, collection):
    """A source that takes longer than the threshold is loaded once, all the same.

    The heartbeat is the only thing that separates an instance still loading a
    source from one that has stopped, so the source here takes longer to load
    than the ownership threshold and the beat runs well inside it. Without the
    beat the second instance would find the claim stale *while the first was
    still loading*, take it over, and load the source a second time -- the
    duplication the claim exists to remove, which the parse count catches.

    The threshold is driven down through the setting rather than the source made
    to clear the default: at one second against a two-second conversion, a claim
    that is refreshed and one that is not are unambiguously different, and the
    test costs seconds rather than minutes. The beat is driven down with it, for
    the same reason it is inside the threshold in production.

    The instance that waits is asserted to end with a failure naming the source.
    That is the bounded wait's own outcome rather than a fault here: its wait ran
    out while the holder still had the source and the index did not hold it yet,
    so the only answer left to it is that it could not report the source. The two
    halves are asserted together because they are one behaviour: waiting a live
    owner out costs a report, while taking the source from it would have cost a
    second parse.
    """
    contents = {"slow.txt": "# Slow\n\nThe STAR method structures behavioural answers."}
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    for name, text in contents.items():
        (corpus / name).write_bytes(text.encode())

    threshold = 1
    settings = _settings(
        table=collection,
        knowledge_corpus_dir=str(corpus),
        job_orphan_threshold_seconds=threshold,
    )
    parsed: list[str] = []
    # Both are slow, so whichever instance takes the source has it for longer than
    # the threshold -- and the other one, waiting, is the instance the beat keeps
    # from concluding the first has stopped.
    first = _corpus_pipeline(settings, contents, parsed, delay=threshold * 2)
    second = _corpus_pipeline(settings, contents, parsed, delay=threshold * 2)
    claims = first.store.claim_store()
    states = [BootstrapState(), BootstrapState()]

    try:
        threads = [
            threading.Thread(
                target=ingest_corpus,
                args=(pipeline, settings, state, claims, f"instance-{index}"),
                kwargs={"heartbeat_interval": threshold / 10},
                name=f"bootstrap-{index}",
            )
            for index, (pipeline, state) in enumerate(zip((first, second), states, strict=True))
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=60)
        assert not any(thread.is_alive() for thread in threads), (
            "an instance's bootstrap did not finish"
        )

        assert sorted(state.status for state in states) == sorted(
            (BootstrapStatus.COMPLETE, BootstrapStatus.FAILED)
        ), [(state.status, state.failures) for state in states]
        assert parsed == ["slow.txt"], (
            f"the slow source was parsed {len(parsed)} times rather than once"
        )
        failures = [failure for state in states for failure in state.failures]
        assert len(failures) == 1, failures
        # Named the way a corpus file is named in a report, which is its path.
        assert (
            "slow.txt: claimed by another instance that did not report it ingested" in failures[0]
        ), failures
        assert claims.held("slow.txt") is None, "the claim outlived the bootstrap"
    finally:
        first._store.close()
        second._store.close()


@needs_database
def test_a_source_present_when_the_wait_ends_is_reported_rather_than_failed(tmp_path, collection):
    """A wait that ends with the source in the index is not a failure.

    The wait can end while the holder still has the claim, and then the answer
    has to come from the index rather than from the holder -- the source may
    already have been loaded, and an instance reporting the corpus failed on top
    of a present source would misreport a startup that worked.

    The question is asked with the source's own digest, and this is what shows
    it: the index was written from the same bytes the corpus holds, so a
    presence question that read or digested the source any other way would find
    nothing and record the failure this test asserts against. The holder is only
    what makes the wait run out -- it refreshes a claim throughout, which is an
    owner that has not stopped and so cannot be waited out by age.
    """
    contents = {"guide.txt": "# Guide\n\nThe STAR method structures behavioural answers."}
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    for name, text in contents.items():
        (corpus / name).write_bytes(text.encode())

    threshold = 1
    settings = _settings(
        table=collection,
        knowledge_corpus_dir=str(corpus),
        job_orphan_threshold_seconds=threshold,
    )

    # Loaded first, on a pipeline of its own, so what is counted below is what the
    # deferring instance parsed and not what this test did.
    seed = _create(settings)
    try:
        _ingest(
            seed,
            "guide.txt",
            contents["guide.txt"],
            payload=contents["guide.txt"].encode(),
        )
    finally:
        seed._store.close()

    parsed: list[str] = []
    deferring = _corpus_pipeline(settings, contents, parsed)
    claims = deferring.store.claim_store()
    state = BootstrapState()

    try:
        assert claims.claim("guide.txt", "instance-holding", float(threshold))
        with claim_heartbeat(claims, "guide.txt", "instance-holding", interval=threshold / 10):
            ingest_corpus(
                deferring,
                settings,
                state,
                claims,
                "instance-deferring",
                heartbeat_interval=threshold / 10,
            )

        assert state.status is BootstrapStatus.COMPLETE, state.failures
        assert state.failures == []
        assert parsed == [], "the deferring instance loaded a source the index held"
        deferring.refresh()
        assert deferring.is_current("guide.txt", content_hash(contents["guide.txt"].encode()), [])
    finally:
        deferring._store.close()


@needs_database
def test_a_source_absent_when_the_wait_ends_is_recorded_as_a_failure(tmp_path, collection):
    """A wait that ends with the source still absent is reported as one.

    The other outcome of the same question, and the reason it is asked rather
    than assumed: the index does not hold the source, so there is no answer this
    instance can report except that the source is missing.

    The parse count is asserted with it because the expiry path must ask about
    the source without loading it -- loading one that another instance may be
    working on is the duplication the claim exists to remove.
    """
    contents = {"guide.txt": "# Guide\n\nThe STAR method structures behavioural answers."}
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    for name, text in contents.items():
        (corpus / name).write_bytes(text.encode())

    threshold = 1
    settings = _settings(
        table=collection,
        knowledge_corpus_dir=str(corpus),
        job_orphan_threshold_seconds=threshold,
    )
    parsed: list[str] = []
    deferring = _corpus_pipeline(settings, contents, parsed)
    claims = deferring.store.claim_store()
    state = BootstrapState()

    try:
        assert claims.claim("guide.txt", "instance-holding", float(threshold))
        with claim_heartbeat(claims, "guide.txt", "instance-holding", interval=threshold / 10):
            ingest_corpus(
                deferring,
                settings,
                state,
                claims,
                "instance-deferring",
                heartbeat_interval=threshold / 10,
            )

        assert state.status is BootstrapStatus.FAILED, state.failures
        assert len(state.failures) == 1, state.failures
        # Named the way a corpus file is named in a report, which is its path.
        assert (
            "guide.txt: claimed by another instance that did not report it ingested"
            in state.failures[0]
        ), state.failures
        assert parsed == [], "the expiry path loaded a source it was only asked about"
    finally:
        deferring._store.close()


@needs_database
def test_a_deferring_instance_reports_the_corpus_once_another_loads_it(
    monkeypatch, tmp_path, collection
):
    """An instance that loaded nothing still reports the corpus the other loaded.

    The readiness endpoint is what a caller asks whether the service is usable,
    so an instance that deferred the whole corpus has to answer with what the
    index holds rather than with what it did -- a deferring instance that
    reported an empty index would be taken for one that never got its corpus.

    The other half is asserted before that: while the loading instance is still
    loading, the deferring one must not report the corpus as present, because
    nobody has ingested it yet. Both halves are read off the endpoint, which is
    why the deferring instance is a whole application rather than a pipeline.
    """
    contents = {"guide.txt": "# Guide\n\nThe STAR method structures behavioural answers."}
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    for name, text in contents.items():
        (corpus / name).write_bytes(text.encode())

    app_settings = _settings(table=collection, knowledge_corpus_dir=str(corpus))
    monkeypatch.setattr("doc_etl_api.main.settings", app_settings)

    # The loading instance, held inside the parse: the claim is taken before the
    # source is read, so holding the parse holds the claim, and the deferring
    # instance below has something real to defer to.
    loaded = threading.Event()
    release_the_loader = threading.Event()

    def convert_file(file, filename):
        loaded.set()
        release_the_loader.wait(timeout=60)
        return contents[filename]

    loader_converter = _stubbed_models(contents, [])
    loader_converter.convert_file.side_effect = convert_file

    loader = _started_app(loader_converter, app_settings)
    loader_client = TestClient(loader)
    try:
        assert loaded.wait(timeout=60), "the loader never reached the source"
        assert loader_client.get("/health").json()["bootstrap"] in ("pending", "in_progress")

        # Named apart by hand: a claim tells instances apart by name, and these
        # two are one process.
        monkeypatch.setattr("doc_etl_api.bootstrap.instance_id", lambda: "instance-deferring")
        deferring_parsed: list[str] = []
        deferring = _started_app(_stubbed_models(contents, deferring_parsed), app_settings)
        deferring_client = TestClient(deferring)
        try:
            # Nothing has been ingested yet, and the deferring instance says so
            # by not being complete rather than by claiming a corpus it has not
            # seen anyone load.
            health = deferring_client.get("/health").json()
            assert health["bootstrap"] in ("pending", "in_progress"), (
                "the deferring instance reported a corpus nobody had loaded"
            )
            assert health["indexed_sources"] == 0, (
                "the deferring instance counted a corpus nobody had loaded"
            )

            release_the_loader.set()
            assert _wait_for_bootstrap(loader_client) == "complete"
            assert _wait_for_bootstrap(deferring_client) == "complete", (
                "the deferring instance never reported the corpus"
            )
            health = deferring_client.get("/health").json()
            assert health["indexed_sources"] == len(contents), (
                "the deferring instance reported an index without the corpus in it"
            )
            assert deferring_parsed == [], "the deferring instance loaded the corpus itself"
        finally:
            deferring.state.pipeline._store.close()
    finally:
        release_the_loader.set()
        loader.state.pipeline._store.close()
