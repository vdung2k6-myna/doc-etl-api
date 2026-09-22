import logging
import os
import re
from collections.abc import Sequence
from enum import Enum
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_LOG_LEVELS = tuple(
    logging.getLevelName(level)
    for level in (logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL)
)

# Collection names are validated rather than normalized, so the name a caller
# sends is the name that is stored and can be filtered on. Rewriting whatever
# arrives would leave the caller guessing which spelling was kept -- `C# 12` and
# `c-12` would become one collection with two spellings and no way to tell which
# one to search for. Surrounding whitespace is the exception: it is unavoidable
# in a form field and does not change the name that was typed, so it is trimmed
# before the grammar is applied. Keeping the grammar this narrow also keeps
# filter values free of characters that would change how a filter reads them.
COLLECTION_NAME_PATTERN = re.compile(r"^[a-z0-9]+(?:[-_][a-z0-9]+)*$")
MAX_COLLECTION_NAME_LENGTH = 64
# How many collections one request may name. The filter stays a single entry
# however many are named, so this is not about cost: it bounds the size of a
# request and of the metadata written for every node it ingests.
MAX_COLLECTIONS_PER_REQUEST = 16


def validate_collection_name(value: str) -> str:
    """The collection name to store for *value*, or a ValueError naming it.

    Trimming happens first, so ``" csharp "`` is the same collection as
    ``"csharp"``; everything else must match the grammar exactly.
    """
    name = value.strip()
    if not COLLECTION_NAME_PATTERN.match(name) or len(name) > MAX_COLLECTION_NAME_LENGTH:
        raise ValueError(
            f"Invalid collection name: {value!r}. A collection name is at most "
            f"{MAX_COLLECTION_NAME_LENGTH} characters long and uses lowercase letters, "
            "digits and single hyphens or underscores, for example 'csharp-12'."
        )
    return name


def validate_collections(values: Sequence[str]) -> list[str]:
    """The collections to store for a request, validated and de-duplicated.

    A name given twice is one collection, so it is recorded once -- the same way
    the file and URL routes collapse a filename or URL repeated within a single
    request.
    """
    names: list[str] = []
    for value in values:
        name = validate_collection_name(value)
        if name not in names:
            names.append(name)
    if len(names) > MAX_COLLECTIONS_PER_REQUEST:
        raise ValueError(
            f"Too many collections: {len(names)} requested, at most "
            f"{MAX_COLLECTIONS_PER_REQUEST} are accepted."
        )
    return names


# Sent on every page fetch. It names the service rather than the HTTP library,
# because a host with a client-identity policy refuses the library's default.
# No contact details are invented here: a host whose policy asks for them gets
# whatever the operator sets, and a deployment that never sets it still sends
# an identity that such hosts accept.
DEFAULT_USER_AGENT = "doc-etl-api/0.1.0"


class VectorStoreBackend(str, Enum):
    SIMPLE = "simple"


# The file settings are read from. A test session sets ``DOC_ETL_API_ENV_FILE``
# to an empty value to disable it, so a ``Settings()`` built without explicit
# arguments falls back to the code defaults instead of picking up whatever the
# developer's ``.env`` happens to hold -- which is how the suite came to assert a
# real corpus as the unconfigured default. Environment variables and explicit
# arguments both outrank this file, so a test can still override any one setting.
ENV_FILE = os.environ.get("DOC_ETL_API_ENV_FILE", ".env") or None


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "Documents ETL API"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    app_reload: bool = False

    vector_store_backend: VectorStoreBackend = Field(default=VectorStoreBackend.SIMPLE)
    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    # Unset means "follow the embedding model's input limit"; see IndexPipeline,
    # which resolves the default against the model it is constructed with.
    chunk_size: int | None = Field(default=None, gt=0)
    chunk_overlap: int = Field(default=50)
    # Nodes smaller than this are merged into a neighbour rather than indexed
    # alone, in the same tokens as `chunk_size`. Zero restores indexing every
    # node the splitter produces, fragments included.
    min_chunk_tokens: int = Field(default=32, ge=0)
    default_top_k: int = Field(default=5)

    max_file_size_mb: int = Field(default=50)
    url_fetch_timeout_seconds: int = Field(default=30)
    user_agent: str = Field(default=DEFAULT_USER_AGENT)

    knowledge_corpus_dir: str = Field(default="")
    knowledge_corpus_urls: str = Field(default="")
    # The collections every corpus source is tagged with. Without this the corpus
    # -- the content that exists specifically to ground answers -- would be the
    # one set of sources invisible to every filtered search.
    knowledge_corpus_collections: str = Field(default="")

    log_level: str = Field(default="INFO")

    @property
    def knowledge_corpus_path(self) -> Path | None:
        """The configured corpus directory, or None when unset."""
        return Path(self.knowledge_corpus_dir) if self.knowledge_corpus_dir.strip() else None

    @property
    def knowledge_corpus_url_list(self) -> list[str]:
        """The configured corpus URLs, parsed from the comma-separated setting."""
        return [url.strip() for url in self.knowledge_corpus_urls.split(",") if url.strip()]

    @property
    def knowledge_corpus_collection_list(self) -> list[str]:
        """The configured corpus collections, parsed from the comma-separated setting.

        Already validated when these settings were built, so this only splits.
        """
        return [
            name.strip() for name in self.knowledge_corpus_collections.split(",") if name.strip()
        ]

    @property
    def resolved_user_agent(self) -> str:
        """The user agent to send, falling back to the default when unset.

        A blank value resolves to the default rather than to an empty header:
        an empty ``User-Agent`` is refused by the same hosts that refuse the
        library's default, so honouring it literally would be a silent way to
        reproduce the failure this setting exists to avoid.
        """
        return self.user_agent.strip() or DEFAULT_USER_AGENT

    @field_validator("log_level", mode="before")
    @classmethod
    def _normalize_log_level(cls, value: str) -> str:
        name = str(value).strip().upper()
        if name not in _LOG_LEVELS:
            supported = ", ".join(_LOG_LEVELS)
            raise ValueError(f"Unsupported log level: {value!r}. Supported levels: {supported}")
        return name

    @field_validator("vector_store_backend", mode="before")
    @classmethod
    def _normalize_backend(cls, value: str) -> VectorStoreBackend:
        try:
            return VectorStoreBackend(value.lower())
        except ValueError as exc:
            supported = ", ".join(b.value for b in VectorStoreBackend)
            raise ValueError(
                f"Unsupported vector store backend: {value!r}. Supported backends: {supported}"
            ) from exc

    @field_validator("knowledge_corpus_collections")
    @classmethod
    def _validate_corpus_collections(cls, value: str) -> str:
        """Reject a malformed corpus collection as the settings are built.

        Checked here rather than only where the corpus is ingested: a typo in a
        deployment's configuration should stop the service from starting, not
        fail every source of the bootstrap one at a time and leave the index
        empty. Each name is validated without being rewritten, so the value the
        operator reads back is the value they wrote.
        """
        for name in value.split(","):
            if name.strip():
                validate_collection_name(name)
        return value


settings = Settings()
