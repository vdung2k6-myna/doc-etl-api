"""Where a pipeline's index lives, and what it has ingested.

The pipeline is written against one small interface -- `IndexStore` -- and not
against a backend, so the same ingestion, search and content paths run on either
of the two that implement it. The default one is the process's own memory: the
in-memory vector store the index has always used, and the catalog dicts the
pipeline has always kept beside it. The durable one puts all three kinds of row
in Postgres -- the embeddings in a pgvector table, each chunk's text and metadata
in a docstore table, and what this service knows about its sources in tables of
its own -- so that a restart does not discard them.

Three tables for one collection, from two naming rules. The library names its
own two `data_<table>` and `data_<table>_docstore` after the configured table
name; the catalog tables are named `<table>_sources`, `<table>_positions` and
`<table>_model` beside them, so that everything one collection needs shares one
identifier and a second collection is built by naming a second table.
"""

import logging
import re
from collections.abc import Sequence
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

from doc_etl_api.config import Settings

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
    pipeline reads and writes to describe its sources. `storage_context` is what
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
    asynchronous is run against it.

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
        # One row, and it is the collection's own record of which model built it.
        # The width of the vector column is what the database enforces; this row
        # is what lets a mismatch name both models instead of only two numbers.
        self._model = sqlalchemy.Table(
            f"{base}_model",
            metadata,
            sqlalchemy.Column("model_name", sqlalchemy.Text, primary_key=True),
            sqlalchemy.Column("embed_dim", sqlalchemy.Integer, nullable=False),
        )
        metadata.create_all(self._engine)

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

        Both halves are one transaction, so a source's recorded order always
        describes one ingestion rather than two halves of two. That is as far as
        one transaction reaches: the nodes themselves are written by the index to
        the library's own tables, before this is called, and the two writes are
        not joined. A crash between them leaves a source whose text is stored and
        whose record is a revision behind -- which the next ingestion of that
        source repairs, and which the bootstrap's content comparison would also
        catch, since a record that does not match the content it holds is not a
        record of it.
        """
        values = {
            "address": address,
            "name": name,
            "source_type": source_type,
            "collections": list(collections),
            "chunk_count": chunk_count,
            "content_hash": content_hash,
        }
        with self._engine.begin() as connection:
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
    dimension = getattr(sentence_transformers, "get_sentence_embedding_dimension", None)
    if callable(dimension):
        width = dimension()
        if isinstance(width, int) and width > 0:
            return width
    raise ValueError(
        f"Embedding model {type(embedding_model).__name__} states no vector width -- "
        "neither `embed_dim` nor `get_sentence_embedding_dimension` -- so a durable store "
        "cannot be sized for it."
    )
