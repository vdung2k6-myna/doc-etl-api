"""Where a pipeline's index lives, and what it has ingested.

The pipeline is written against one small interface -- `IndexStore` -- and not
against a backend, so the same ingestion, search and content paths run on either
of the two that implement it. The default one is the process's own memory: the
in-memory vector store the index has always used, and the catalog dicts the
pipeline has always kept beside it. The durable one puts the index in Postgres
-- the embeddings in a pgvector table, each chunk's text and metadata in a
docstore table, and what this service knows about its sources in tables of its
own -- so that a restart does not discard them. Jobs are kept there too, by the
registry that reads this store's engine, so that a job's record outlives the
instance that accepted it.

Three catalog tables for one collection, from two naming rules. The library
names its own two `data_<table>` and `data_<table>_docstore` after the configured
table name; the catalog tables are named `<table>_sources`, `<table>_positions`
and `<table>_model` beside them, so that everything one collection needs shares
one identifier and a second collection is built by naming a second table. The
jobs table is named by that same rule -- `<table>_jobs` -- and defined in
`jobs.py`, beside the registry whose rows it holds.
"""

import contextlib
import logging
import re
from collections.abc import Iterator, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Protocol

import sqlalchemy
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.data_structs import IndexDict
from llama_index.core.storage.index_store.keyval_index_store import KVIndexStore
from llama_index.core.storage.storage_context import StorageContext
from llama_index.storage.docstore.postgres import PostgresDocumentStore
from llama_index.storage.kvstore.postgres import PostgresKVStore
from llama_index.vector_stores.postgres import PGVectorStore
from sqlalchemy.dialects import postgresql
from sqlalchemy.ext.asyncio import create_async_engine

from doc_etl_api.claims import PostgresClaimStore, claims_table
from doc_etl_api.config import Settings
from doc_etl_api.jobs import PostgresJobStore, jobs_table

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SourceRecord:
    """What a source is, as the catalog reports it.

    ``name`` and ``address`` are two different things, and for a URL that
    redirected they differ. The name is what the source was submitted as -- the
    filename an upload carried, the URL a caller sent -- and it is what the
    catalog displays. The address is what the index stores the source under and
    therefore the only thing that finds one again: a filename, or the final URL a
    submitted URL landed on. Both are reported so a caller can display a source
    and fetch it without one of those needing to be derived from the other.

    ``content_hash`` is the digest of the bytes the source was ingested from,
    written as it is ingested. It is optional because a record can be built by a
    caller that ingested nothing -- the tests do -- and not because a stored
    source is ever without one: a reader comparing it has to treat None as "not
    comparable", never as a digest that matches nothing.
    """

    name: str
    address: str
    source_type: str
    collections: tuple[str, ...]
    chunk_count: int
    content_hash: str | None = None


class DurableStoreError(RuntimeError):
    """The configured durable store could not be established.

    Raised while the service is starting rather than when the first request
    reaches it. A durable backend that cannot be reached, has no vector
    extension, or holds a collection this model does not belong to is a
    deployment that is wrong in a way no request can fix, and failing once at
    startup says which wrong thing it is instead of failing every request
    against a store that was never usable.
    """


class IndexStore(Protocol):
    """What the pipeline asks of wherever its index lives.

    The catalog half -- `records`, `positions` and `replace` -- is what the
    pipeline reads and writes to describe its sources, and `document` and
    `store_document` are the text that description is about: what each source was
    converted into, kept so that a content read can answer with the document
    itself rather than with the chunks cut from it. `storage_context` is what
    the index is built on, `store_nodes_override` decides whether the docstore
    keeps the text when the vector store also keeps it, and `close` gives the
    pipeline a store-agnostic way to hand resources back.
    """

    @property
    def storage_context(self) -> StorageContext | None: ...

    @property
    def store_nodes_override(self) -> bool: ...

    def index_struct(self) -> IndexDict:
        """The map from a source to the nodes it holds, as the store remembers it.

        The index keeps its own copy of this map, and its delete path reads the
        map rather than the store: deleting a source the map does not mention is
        an error there, even though the store holds it. So a store that outlives
        the process has to answer with the map as well as the content, or the
        first re-ingestion after a restart fails on content that is right there.
        """
        ...

    def records(self) -> tuple[SourceRecord, ...]: ...

    def positions(self, address: str) -> dict[str, int]: ...

    def document(self, address: str) -> str | None:
        """The document *address* was converted into, or None where none was captured.

        The document is the markdown the source was converted into, which is the
        text its chunks were cut from. The chunks are not it: a heading repeats in
        every node of its section, a unit below the floor merges into a neighbour,
        and text repeated inside one source is stored once.

        It is kept beside the catalog rather than in it, because the catalog
        answers what a collection holds -- the listing, the readiness counts, and
        whether a source needs ingesting at all -- and none of those reads should
        carry a document along with the row it is reading.

        None and "" are two different answers, and both are returned as
        themselves. A source that converted to nothing holds an empty document. A
        source whose document was never captured holds none -- which is every
        source ingested before documents were stored, until it is ingested again.
        Collapsing the two would report a source that converted to nothing as one
        this service can say nothing about.
        """
        ...

    def store_document(self, address: str, text: str) -> None:
        """Write *address*'s document, replacing whatever it stored before.

        Called inside the same `transaction` as the record and the chunks, and
        after them: what a source is, what it holds and what it was converted from
        then commit together or not at all, so no reader can find a document
        describing an ingestion the index did not receive. The record's write
        comes first because it is what claims the source's row.
        """
        ...

    @property
    def index_struct_is_stored(self) -> bool:
        """Whether the map is the store's to answer for, rather than the index's.

        True where the store outlives the process. There the map the index holds
        is a copy taken when it was built, and an instance that did not write the
        last replacement holds a copy that is behind -- so a replacement has to
        read the map back before it deletes anything, or it fails on a source's
        node ids that are in the store and not in its own copy.
        """
        ...

    def transaction(self) -> AbstractContextManager[None]:
        """Run the writes inside this block as one unit, or none of them.

        The pipeline wraps one replacement in this, which is the reason it
        exists: removing a source's earlier nodes, writing its new ones and
        recording it in the catalog have to reach other instances together. Split
        into separate commits, a reader on another instance can observe the
        source with its earlier content removed and its replacement not yet
        written, and a failure after the removal leaves it that way for good.

        What that costs is the backend's to decide. In this process's memory the
        writes are already indivisible while the caller's lock is held, so the
        block has nothing of its own to do. In Postgres it is one transaction on
        one connection, with the library's own stores joined to it, so that a
        reader anywhere sees the source before the replacement or after it and
        never in between.

        The block is not a lock across instances: two instances replacing two
        different sources still run side by side. It is the source's own catalog
        row that serializes two replacements of one source, which is why
        `replace` writes it first.
        """
        ...

    def replace(
        self,
        address: str,
        *,
        name: str,
        source_type: str,
        collections: Sequence[str],
        chunk_count: int,
        content_hash: str | None,
        positions: dict[str, int],
    ) -> None: ...

    def close(self) -> None: ...


class InProcessIndexStore:
    """The default backend: the index and the catalog in this process's memory.

    Nothing durable is established and nothing needs closing. The index is built
    on the default storage context, which is an in-memory vector store and an
    in-memory docstore -- so the docstore holds each node's text, and the index
    is written and read as it always has been.
    """

    def __init__(self) -> None:
        self._records: dict[str, SourceRecord] = {}
        self._positions: dict[str, dict[str, int]] = {}
        self._documents: dict[str, str] = {}

    @property
    def storage_context(self) -> StorageContext | None:
        return None

    @property
    def store_nodes_override(self) -> bool:
        return False

    def index_struct(self) -> IndexDict:
        """A map for a collection that has never been written to.

        Every process starts with the whole index, so there is nothing here to
        remember: the map is built as the index is written, by the index itself.
        """
        return IndexDict()

    def records(self) -> tuple[SourceRecord, ...]:
        return tuple(self._records[key] for key in sorted(self._records))

    def positions(self, address: str) -> dict[str, int]:
        return dict(self._positions.get(address, {}))

    def document(self, address: str) -> str | None:
        """Absence and emptiness kept apart, as the protocol requires of both.

        An address never written reads as None and one written empty reads as "",
        which is the difference between a source whose document was not captured
        and one that converted to nothing.
        """
        return self._documents.get(address)

    def store_document(self, address: str, text: str) -> None:
        self._documents[address] = text

    @property
    def index_struct_is_stored(self) -> bool:
        """False: the map is the index's, and this store has nothing to add to it.

        There is one process and one index, so the map the index built is the
        only one there is and re-reading it would mean replacing it with an
        empty. See the member's documentation on `IndexStore`.
        """
        return False

    @contextlib.contextmanager
    def transaction(self) -> Iterator[None]:
        """Nothing of its own: this process's writes are already one unit.

        There is no second instance reading this store and no commit to get half
        way through, so the block only needs the caller's lock, which is held
        around it. It is here rather than left out so that the pipeline has one
        shape on both backends and the durable one's transaction is not a branch
        the writer path takes.
        """
        yield

    def replace(
        self,
        address: str,
        *,
        name: str,
        source_type: str,
        collections: Sequence[str],
        chunk_count: int,
        content_hash: str | None,
        positions: dict[str, int],
    ) -> None:
        self._records[address] = SourceRecord(
            name=name,
            address=address,
            source_type=source_type,
            collections=tuple(collections),
            chunk_count=chunk_count,
            content_hash=content_hash,
        )
        self._positions[address] = dict(positions)

    def close(self) -> None:
        """Nothing to hand back: this store's memory goes when the process does."""


class PostgresIndexStore:
    """The durable backend: everything one collection needs, in Postgres.

    The service holds one engine pair for the whole store. The vector store and
    the docstore are both built on it rather than opening their own, because the
    catalog's reads and the index's writes then reach the same database through
    the same pool, and because the library's stores require an asynchronous
    engine to be given alongside the synchronous one whether or not anything
    asynchronous is run against it. One engine is also what makes a replacement
    one transaction rather than three: `transaction` joins the stores' sessions
    to a single connection, and there has to be a single engine for there to be
    one connection to join them to.

    The docstore holds each chunk's text even though the vector table holds it
    too. `PGVectorStore` reports that it stores text, which is what keeps the
    index from writing to the docstore at all -- so without
    ``store_nodes_override`` the docstore stays empty and every read that goes
    through it, which is both the content read and the neighbour lookup, answers
    nothing. The override makes the index write the node to both places. The cost
    is a second copy of every chunk's text; the reason it is paid is that reads
    then take one path for both backends, and the docstore's lookup by node id is
    indexed where the vector table's is not.
    """

    def __init__(self, app_settings: Settings, embedding_model: BaseEmbedding) -> None:
        self._settings = app_settings
        self._embed_dim = _embedding_dimension(embedding_model)

        url = app_settings.postgres_connection_url
        # `pool_pre_ping` because this service outlives the connections it opens:
        # a database that was restarted underneath it leaves dead sockets in the
        # pool, and pre-ping retires them on the way out instead of failing the
        # request that drew one.
        self._engine = sqlalchemy.create_engine(url, pool_pre_ping=True)
        self._async_engine = create_async_engine(
            url.set(drivername="postgresql+asyncpg"), pool_pre_ping=True
        )

        self._require_reachable()
        self._require_vector_extension()
        self._define_catalog_tables()

        self._vector_store = PGVectorStore(
            engine=self._engine,
            async_engine=self._async_engine,
            table_name=app_settings.postgres_table_name,
            embed_dim=self._embed_dim,
            # Setup failures are raised rather than logged and swallowed: the
            # library's default leaves a store that answers nothing, which reads
            # as an empty collection rather than as the misconfiguration it is.
            initialization_fail_on_error=True,
        )
        kvstore = PostgresKVStore(
            table_name=self.docstore_table_name,
            engine=self._engine,
            async_engine=self._async_engine,
        )
        # Kept so that `transaction` can reach the session factory the docstore
        # and the index store both write through. Both are built on this one
        # kvstore rather than one each, which is what makes joining them to a
        # single transaction a matter of joining one factory.
        self._kvstore = kvstore
        self._docstore = PostgresDocumentStore(postgres_kvstore=kvstore)
        # The map from a source to its nodes is persisted beside the nodes
        # themselves, in the same table under its own namespace. It is what makes
        # a source re-ingested after a restart replace its earlier content
        # instead of failing to find it.
        self._index_store = KVIndexStore(kvstore)
        self._storage_context = StorageContext.from_defaults(
            vector_store=self._vector_store,
            docstore=self._docstore,
            index_store=self._index_store,
        )
        self._establish_vector_table()
        self._validate_collection()
        # The connection a replacement is in progress on, or None outside one.
        # `transaction` sets it and clears it; `replace` reads it so that a
        # record written during a replacement goes to the same transaction as the
        # nodes rather than to a second one of its own.
        self._connection: sqlalchemy.Connection | None = None

    @property
    def vector_table_name(self) -> str:
        """The name of the table the library puts the vectors in.

        Derived here rather than read back from the store: it is the library's
        own naming rule, `data_<table_name>`, and the messages that name it are
        the ones an operator acts on. A version of the library that named it
        differently would say so at the first query, naming the table it could
        not find.
        """
        return f"data_{self._settings.postgres_table_name}"

    @property
    def docstore_table_name(self) -> str:
        return f"{self._settings.postgres_table_name}_docstore"

    @property
    def storage_context(self) -> StorageContext | None:
        return self._storage_context

    @property
    def store_nodes_override(self) -> bool:
        return True

    @property
    def index_struct_is_stored(self) -> bool:
        """True: the map is in the database, and another instance may have moved it.

        The map is written to the same table as the nodes, under a namespace of
        its own, so it is part of what a replacement commits. A process is
        therefore only ever holding a copy of it, taken when the index was built.
        See the member's documentation on `IndexStore`.
        """
        return True

    def index_struct(self) -> IndexDict:
        """The map the last process to write this collection left behind.

        Empty for a collection nothing has been ingested into yet, which is the
        same answer a collection written and emptied would give: the map
        describes the stored nodes, so a collection holding none has none.
        """
        structs = self._index_store.index_structs()
        return structs[0] if structs else IndexDict()

    def _require_reachable(self) -> None:
        """Establish that the database answers before anything is built on it."""
        try:
            with self._engine.connect() as connection:
                connection.execute(sqlalchemy.text("SELECT 1"))
        except sqlalchemy.exc.SQLAlchemyError as exc:
            raise DurableStoreError(
                f"Vector store backend '{self._settings.vector_store_backend.value}' could "
                f"not reach its store at {self._settings.postgres_address}: {exc}. Start "
                "the database (see 'Vector store backends' in the README), correct the "
                "POSTGRES_* settings, or select the in-memory backend."
            ) from exc

    def _require_vector_extension(self) -> None:
        """Establish that the database can store vectors, naming the extension.

        Checked before the library is built rather than left to it: the library
        asks the database to create the extension itself, so a database that
        cannot offer it fails as a raw driver error from inside a constructor,
        which does not say which extension or which database was the problem.
        """
        with self._engine.connect() as connection:
            installed = connection.execute(
                sqlalchemy.text("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
            ).scalar()
            if installed:
                return
            offered = connection.execute(
                sqlalchemy.text("SELECT 1 FROM pg_available_extensions WHERE name = 'vector'")
            ).scalar()
        if not offered:
            raise DurableStoreError(
                "Vector store backend "
                f"'{self._settings.vector_store_backend.value}' needs the 'vector' "
                f"extension, which the database at {self._settings.postgres_address} does "
                "not offer. Use a Postgres image with pgvector -- the compose service "
                "under the 'postgres' profile is one -- or install the extension."
            )
        try:
            with self._engine.begin() as connection:
                connection.execute(sqlalchemy.text("CREATE EXTENSION IF NOT EXISTS vector"))
        except sqlalchemy.exc.SQLAlchemyError as exc:
            raise DurableStoreError(
                f"The 'vector' extension exists on {self._settings.postgres_address} but "
                f"could not be created: {exc}. Creating an extension needs a role allowed "
                "to; create it once as an administrator and start the service again."
            ) from exc

    def _define_catalog_tables(self) -> None:
        """Create this service's own tables, beside the library's.

        The source row and its position rows are separate tables rather than one
        row per chunk, because they answer different questions: the source is
        what the catalog lists and the readiness counters count, and the positions
        are what turns a hit into a chunk of a document. The foreign key is what
        keeps the second from outliving the first, and it cascades so that
        replacing a source is one delete and one insert rather than a walk.
        """
        base = self._settings.postgres_table_name
        metadata = sqlalchemy.MetaData()
        self._sources = sqlalchemy.Table(
            f"{base}_sources",
            metadata,
            sqlalchemy.Column("address", sqlalchemy.Text, primary_key=True),
            sqlalchemy.Column("name", sqlalchemy.Text, nullable=False),
            sqlalchemy.Column("source_type", sqlalchemy.Text, nullable=False),
            sqlalchemy.Column("collections", postgresql.ARRAY(sqlalchemy.Text), nullable=False),
            sqlalchemy.Column("chunk_count", sqlalchemy.Integer, nullable=False),
            # Nullable so the column and `SourceRecord` agree on what an unknown
            # hash is. Nothing has an unknown one once ingestion records them,
            # and a reader comparing hashes treats None as "not comparable",
            # which ingests the source again rather than skipping it.
            sqlalchemy.Column("content_hash", sqlalchemy.Text, nullable=True),
        )
        self._positions = sqlalchemy.Table(
            f"{base}_positions",
            metadata,
            sqlalchemy.Column(
                "address",
                sqlalchemy.Text,
                sqlalchemy.ForeignKey(f"{base}_sources.address", ondelete="CASCADE"),
                primary_key=True,
            ),
            sqlalchemy.Column("node_id", sqlalchemy.Text, primary_key=True),
            sqlalchemy.Column("position", sqlalchemy.Integer, nullable=False),
        )
        # The document each source was converted into, one row per source, keyed
        # like the catalog row it sits beside. A table of its own rather than a
        # column on the source row, because the catalog answers what the index
        # holds -- the listing, the readiness counts, and whether a source needs
        # ingesting -- and a document carried in that row would be dragged through
        # every one of those reads. The foreign key is what keeps a document from
        # outliving the source it describes, and it cascades, so removing a source
        # removes what it was converted from with it. The text is not nullable:
        # a stored document is the text that was converted, and a source that
        # converted to nothing is an empty string rather than an absent row --
        # which is what keeps "no document was captured" a different answer from
        # "the document is empty".
        self._documents = sqlalchemy.Table(
            f"{base}_documents",
            metadata,
            sqlalchemy.Column(
                "address",
                sqlalchemy.Text,
                sqlalchemy.ForeignKey(f"{base}_sources.address", ondelete="CASCADE"),
                primary_key=True,
            ),
            sqlalchemy.Column("document", sqlalchemy.Text, nullable=False),
        )
        # One row, and it is the collection's own record of which model built it.
        # The width of the vector column is what the database enforces; this row
        # is what lets a mismatch name both models instead of only two numbers.
        self._model = sqlalchemy.Table(
            f"{base}_model",
            metadata,
            sqlalchemy.Column("model_name", sqlalchemy.Text, primary_key=True),
            sqlalchemy.Column("embed_dim", sqlalchemy.Integer, nullable=False),
        )
        # The job records, created here with the rest of the schema because this
        # is where the schema is established: they belong to the same database
        # and the same base name, and a jobs table that appeared on first use
        # instead would be a second thing to establish, in a second place, for a
        # table the readiness endpoint counts rows in.
        jobs_table(base, metadata)
        # The bootstrap claims, created here for the same reason and in the same
        # way: one database, one base name, one place the schema is established.
        claims_table(base, metadata)
        metadata.create_all(self._engine)

    def job_store(self) -> "PostgresJobStore":
        """A jobs registry backing over this store's own database and engine.

        Built from the engine rather than from the settings, so it cannot be
        pointed at a different database than the index it is read beside, and so
        a deployment holds one pool rather than two.
        """
        return PostgresJobStore(self._engine, self._settings.postgres_table_name)

    def claim_store(self) -> "PostgresClaimStore":
        """A claim store over this store's own database and engine.

        Built from the engine rather than from the settings, for the same reason
        the job store is: it cannot be pointed at a different database than the
        index the claims are about, and a deployment holds one pool rather than
        two.
        """
        return PostgresClaimStore(self._engine, self._settings.postgres_table_name)

    def _establish_vector_table(self) -> None:
        """Have the vector store create its table now rather than at first use.

        The library sets its schema up lazily -- on the first add, query or
        delete -- and the collection has to be read before any of those run, to
        check that this model belongs to it. Adding nothing establishes the table
        (and, through the same setup, the extension) without writing a row; with
        ``initialization_fail_on_error`` a failure raises here instead of being
        logged and swallowed, which is the difference between a deployment that
        refuses to start and one that accepts work it cannot store.
        """
        self._vector_store.add([])

    def _validate_collection(self) -> None:
        """Refuse to write into a collection this embedding model does not belong to.

        Both checks are needed because either can fail alone. The width of the
        vector column is what a write has to fit, and a model of another width
        would be refused by the database at the first ingest -- after the source
        had been parsed and embedded. The recorded model name is what catches the
        case the width cannot: two models of the same width mix silently, ranking
        hits by a distance that means nothing.

        Called after the stores are built, because building them is what creates
        the vector table the width is read from.
        """
        width = self._embedding_column_width()
        if width != self._embed_dim:
            raise DurableStoreError(
                f"Embedding model '{self._settings.embedding_model}' produces "
                f"{self._embed_dim}-dimensional vectors, but the collection in "
                f"{self._settings.postgres_address} (table {self.vector_table_name}) "
                f"stores vector({width}). Use a model of {width} dimensions for this "
                "collection, or point POSTGRES_TABLE_NAME at a table of its own."
            )
        with self._engine.begin() as connection:
            recorded = connection.execute(
                sqlalchemy.select(self._model.c.model_name, self._model.c.embed_dim)
            ).first()
            if recorded is None:
                connection.execute(
                    self._model.insert().values(
                        model_name=self._settings.embedding_model, embed_dim=self._embed_dim
                    )
                )
                return
            if recorded.model_name != self._settings.embedding_model:
                raise DurableStoreError(
                    f"The collection in {self._settings.postgres_address} (table "
                    f"{self.vector_table_name}) was built by embedding model "
                    f"'{recorded.model_name}' ({recorded.embed_dim} dimensions), but this "
                    f"service is configured for '{self._settings.embedding_model}'. "
                    "Vectors from two models are not comparable, so one collection cannot "
                    "hold both. Point POSTGRES_TABLE_NAME at a table of its own, or "
                    "configure the model the collection was built with."
                )

    def _embedding_column_width(self) -> int:
        """The width the collection's vector column declares.

        Read from the database's own catalog rather than assumed from the
        settings, because the column is what a write is checked against: it is
        created once, by whichever model first built the collection, and does not
        change when a later deployment selects a different model.
        """
        with self._engine.connect() as connection:
            declared = connection.execute(
                sqlalchemy.text(
                    "SELECT format_type(a.atttypid, a.atttypmod) "
                    "FROM pg_attribute a JOIN pg_class c ON c.oid = a.attrelid "
                    "WHERE c.relname = :table AND a.attname = 'embedding' "
                    "AND a.attnum > 0 AND NOT a.attisdropped AND pg_table_is_visible(c.oid)"
                ),
                {"table": self.vector_table_name},
            ).scalar()
        if declared is None:
            raise DurableStoreError(
                f"The collection in {self._settings.postgres_address} (table "
                f"{self.vector_table_name}) has no 'embedding' column, so it is not a "
                "table this backend created and cannot be written to as one. Point "
                "POSTGRES_TABLE_NAME at a table of its own."
            )
        match = re.fullmatch(r"vector\((\d+)\)", declared)
        if match is None:
            raise DurableStoreError(
                f"The collection in {self._settings.postgres_address} (table "
                f"{self.vector_table_name}) declares its 'embedding' column as {declared!r}, "
                "which is not a vector of a declared width, so the backend cannot tell "
                "whether this model's vectors fit it."
            )
        return int(match.group(1))

    def records(self) -> tuple[SourceRecord, ...]:
        """Every source, ordered by address.

        The order is part of the answer: it is the order the catalog is published
        in, and a snapshot that reordered itself between restarts would make two
        runs of the same corpus read as two different catalogs.
        """
        with self._engine.connect() as connection:
            rows = connection.execute(
                sqlalchemy.select(self._sources).order_by(self._sources.c.address)
            ).all()
        return tuple(
            SourceRecord(
                name=row.name,
                address=row.address,
                source_type=row.source_type,
                collections=tuple(row.collections),
                chunk_count=row.chunk_count,
                content_hash=row.content_hash,
            )
            for row in rows
        )

    def positions(self, address: str) -> dict[str, int]:
        """Each node id this source holds, mapped to its position within it."""
        with self._engine.connect() as connection:
            rows = connection.execute(
                sqlalchemy.select(self._positions.c.node_id, self._positions.c.position).where(
                    self._positions.c.address == address
                )
            ).all()
        return {row.node_id: row.position for row in rows}

    def document(self, address: str) -> str | None:
        """The stored document, reached by primary key rather than by scan.

        One indexed lookup, so a content read costs one row however much the
        collection holds -- which matters because the documents table is the one
        table here whose rows are whole documents.
        """
        with self._engine.connect() as connection:
            return connection.execute(
                sqlalchemy.select(self._documents.c.document).where(
                    self._documents.c.address == address
                )
            ).scalar_one_or_none()

    def store_document(self, address: str, text: str) -> None:
        """Write *address*'s document, replacing what it stored before.

        An upsert, so a re-ingestion replaces the document rather than adding a
        second one beside it: what a source was converted into is the conversion
        of the content the index holds for it, and never the one before.
        """
        with self._write_connection() as connection:
            connection.execute(
                postgresql.insert(self._documents)
                .values(address=address, document=text)
                .on_conflict_do_update(
                    index_elements=[self._documents.c.address],
                    set_={"document": text},
                )
            )

    @contextlib.contextmanager
    def transaction(self) -> Iterator[None]:
        """One transaction on one connection, with the library's stores joined to it.

        The index writes a replacement through the library's own stores -- the
        vector store, the docstore and the index store -- and each of those opens
        a session per call, which is a connection of its own from the pool and a
        transaction of its own with it. So the replacement would commit in three
        pieces however this method is written, unless the sessions are pointed at
        this connection instead.

        That is what happens here, and it is the whole of the mechanism:
        `sessionmaker(bind=connection)` is SQLAlchemy's own way of putting a
        session into a transaction that is already open -- it joins as a
        savepoint, so the session's own `commit()` releases the savepoint and
        leaves this transaction to decide. It is done by assigning the two
        session factories the library builds for itself, which is the only way
        in: neither store accepts a connection, only an engine, and a session
        built from an engine takes a fresh connection every time.

        Both assignments are private state of a third-party library, so they are
        restored before this returns whatever happened, and two threads must
        never be inside this block at once -- the caller holds `_index_lock`
        around it, and the same lock keeps every reader in this process out while
        the factories are redirected. Assigning them is the cost of the
        guarantee: without it a second instance reads a source whose earlier
        content has been deleted and whose replacement has not been written yet.

        The connection is opened here rather than taken from the session, so that
        a failure inside the block rolls the whole thing back and leaves the
        source exactly as it was. The catalog row is written by the caller's
        first statement inside the block, which is what makes a second instance
        replacing the same source wait here rather than interleave.
        """
        connection = self._engine.connect()
        outer = connection.begin()
        vector_sessions = self._vector_store._session
        kv_sessions = self._kvstore._session
        self._vector_store._session = sqlalchemy.orm.sessionmaker(bind=connection)
        self._kvstore._session = sqlalchemy.orm.sessionmaker(bind=connection)
        self._connection = connection
        try:
            yield
        except BaseException:
            outer.rollback()
            raise
        else:
            outer.commit()
        finally:
            self._connection = None
            self._vector_store._session = vector_sessions
            self._kvstore._session = kv_sessions
            connection.close()

    def replace(
        self,
        address: str,
        *,
        name: str,
        source_type: str,
        collections: Sequence[str],
        chunk_count: int,
        content_hash: str | None,
        positions: dict[str, int],
    ) -> None:
        """Write *address*'s record and positions, replacing what it held.

        The pipeline calls this first inside a replacement's `transaction`, and
        that order is load-bearing rather than incidental: the upsert below takes
        the source's row lock, so a second instance replacing the same source
        waits here instead of writing its nodes alongside the first one's. Called
        that way, the record commits with the nodes or not at all.

        It also stands on its own, outside a transaction, where it is a write of
        this call's own. Nothing in the service does that; it is what the
        interface promises a store asked to write one source, and what the
        in-process backend answers by writing its dicts.

        The positions are deleted and rewritten rather than merged, so a source's
        recorded order always describes one ingestion rather than two halves of
        two -- an ingestion that produced fewer chunks than the one before it
        would otherwise leave positions past its end pointing at nothing.
        """
        values = {
            "address": address,
            "name": name,
            "source_type": source_type,
            "collections": list(collections),
            "chunk_count": chunk_count,
            "content_hash": content_hash,
        }
        with self._write_connection() as connection:
            connection.execute(
                postgresql.insert(self._sources)
                .values(**values)
                .on_conflict_do_update(
                    index_elements=[self._sources.c.address],
                    set_={key: value for key, value in values.items() if key != "address"},
                )
            )
            connection.execute(self._positions.delete().where(self._positions.c.address == address))
            if positions:
                connection.execute(
                    self._positions.insert(),
                    [
                        {"address": address, "node_id": node_id, "position": position}
                        for node_id, position in positions.items()
                    ],
                )

    @contextlib.contextmanager
    def _write_connection(self) -> Iterator[sqlalchemy.Connection]:
        """The connection a replacement is in progress on, or one of this call's own.

        Two cases, and the difference between them is the whole of what
        `transaction` buys: inside one, this is the connection the index's stores
        were joined to, so the record and the nodes commit together; outside one,
        it is a transaction that begins and ends with this write, which is what a
        caller asking this store to write a single source is entitled to.
        """
        if self._connection is not None:
            yield self._connection
            return
        with self._engine.begin() as connection:
            yield connection

    def close(self) -> None:
        """Release the connections this store holds.

        The asynchronous engine is disposed through its synchronous side, because
        `dispose` on the asynchronous engine is a coroutine and this is called
        from a synchronous shutdown path -- where awaiting it would mean either
        not closing or closing on the event loop's behalf.
        """
        self._engine.dispose()
        self._async_engine.sync_engine.dispose()


def _embedding_dimension(embedding_model: BaseEmbedding) -> int:
    """How many numbers the model puts in each vector.

    The width is written into the table's schema and enforced by the database on
    every write, so a wrong answer here is not cosmetic: too small and every
    insert is refused, too large and the vectors are padded rather than
    comparable. It is read from the model rather than from the settings, because
    a setting can name a model that is not the one loaded.

    Two sources are asked, in order. ``embed_dim`` is what the test doubles and
    the library's own mock state outright. A sentence-transformers model does not
    state it there but answers the question, and that is the real model's answer.
    A model that answers neither is a construction error, not a reason to guess a
    width: the guess would be written into a table that outlives the process.
    """
    declared = getattr(embedding_model, "embed_dim", None)
    if isinstance(declared, int) and declared > 0:
        return declared
    sentence_transformers = getattr(embedding_model, "_model", None)
    dimension = getattr(sentence_transformers, "get_embedding_dimension", None)
    if callable(dimension):
        width = dimension()
        if isinstance(width, int) and width > 0:
            return width
    raise ValueError(
        f"Embedding model {type(embedding_model).__name__} states no vector width -- "
        "neither `embed_dim` nor `get_embedding_dimension` -- so a durable store "
        "cannot be sized for it."
    )
