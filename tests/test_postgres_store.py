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
import time
from collections.abc import Iterator, Sequence
from functools import partial
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import sqlalchemy
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.engine import make_url
from sqlalchemy.pool import NullPool

from doc_etl_api.config import Settings, VectorStoreBackend
from doc_etl_api.main import create_app
from doc_etl_api.pipeline import IndexPipeline, content_hash, create_pipeline
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
