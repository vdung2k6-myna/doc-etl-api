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

# Vietnamese. Left unconfigured the OCR engine would pick its own default, a set
# of European languages, which reads Vietnamese with a recogniser that does not
# know its diacritics.
DEFAULT_PDF_OCR_LANGUAGES = "vi"
# The shape of an OCR language code: `vi`, `en`, `ch_sim`. Deliberately loose,
# because the codes are the OCR engine's and they change when it gains a
# language; this rejects a mistyped setting without claiming to know the list.
OCR_LANGUAGE_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*$")


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
    # The languages a scanned document is read in. Comma-separated, like the
    # corpus settings below. Vietnamese by default, because the service's content
    # is Vietnamese and a recogniser that does not know the language's diacritics
    # drops them -- and in Vietnamese the diacritics are the word, not decoration.
    pdf_ocr_languages: str = Field(default=DEFAULT_PDF_OCR_LANGUAGES)

    knowledge_corpus_dir: str = Field(default="")
    knowledge_corpus_urls: str = Field(default="")
    # The collections every corpus source is tagged with. Without this the corpus
    # -- the content that exists specifically to ground answers -- would be the
    # one set of sources invisible to every filtered search.
    knowledge_corpus_collections: str = Field(default="")
    # Per-file corpus collections, keyed by filename, for a corpus directory that
    # holds documents belonging to different collections. An entry replaces the
    # setting above for the file it names; every other file keeps it. Read as
    # JSON, so a filename needs no escaping and cannot be mistaken for part of
    # the value beside it.
    knowledge_corpus_file_collections: dict[str, list[str]] = Field(default_factory=dict)

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
    def pdf_ocr_language_list(self) -> list[str]:
        """The configured OCR languages, parsed from the comma-separated setting.

        Already validated and non-empty when these settings were built, so this
        only splits: the order is the operator's, and a language named twice is
        passed through as the operator wrote it.
        """
        return [name.strip() for name in self.pdf_ocr_languages.split(",") if name.strip()]

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

    @field_validator("pdf_ocr_languages")
    @classmethod
    def _validate_pdf_ocr_languages(cls, value: str) -> str:
        """Reject an OCR language setting that names no usable language.

        Checked here rather than where the recogniser is built, for the same
        reason as the corpus settings above: a typo should stop the service from
        starting, not fail the first document that needs OCR -- by which time the
        job has already been accepted and the operator is looking at a job log
        rather than at their configuration.

        A setting naming no language is refused rather than read as "recognise
        nothing". An empty language list does not mean no OCR: it means the
        engine's own default, a set of European languages, so a deployment that
        configured nothing would still OCR, in the wrong language, with nothing
        in the configuration to show it.
        """
        names = [name.strip() for name in value.split(",") if name.strip()]
        if not names:
            raise ValueError(
                f"Invalid OCR language setting: {value!r} names no language. Scanned "
                "documents are read in the configured languages; to read them with the "
                "engine's own default, name that default's languages here rather than "
                f"leaving the setting empty. For example: {DEFAULT_PDF_OCR_LANGUAGES!r}."
            )
        for name in names:
            if not OCR_LANGUAGE_PATTERN.match(name):
                raise ValueError(
                    f"Invalid OCR language: {name!r}. A language is named by its code, "
                    "for example 'vi' for Vietnamese or 'en' for English."
                )
        return value

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

    @field_validator("knowledge_corpus_file_collections")
    @classmethod
    def _validate_corpus_file_collections(cls, value: dict[str, list[str]]) -> dict[str, list[str]]:
        """Reject a malformed per-file corpus entry as the settings are built.

        Checked here for the same reason as the corpus collections above: a typo
        should stop the service from starting, not tag a corpus source with
        something no filter will ever ask for.

        Two shapes are refused because neither can do what it says. A key naming
        a path can never match, since the corpus directory is scanned without
        recursing and an entry matches a file by its filename alone. Both
        separators are refused on every platform, so the setting means the same
        thing wherever the service runs. An entry naming no collection would
        store a source that every filtered search is blind to, which is the
        opposite of what an entry is for -- so it is read as a mistake rather
        than as a request to tag nothing, and to tag nothing an operator writes
        no entry and configures no corpus collections.

        The keys are kept verbatim: a filename is a literal, not a value with
        padding to forgive. The collections are normalized exactly as an
        upload's are, so the tag stored here is the tag a filter reads.
        """
        normalized: dict[str, list[str]] = {}
        for filename, collections in value.items():
            if not filename.strip():
                raise ValueError(
                    "Invalid corpus file collection entry: the filename is empty. An "
                    "entry names a file in the corpus directory."
                )
            # Both separators, on every platform. `pathlib` reads a backslash as
            # a separator only on Windows, so testing the name with it would let
            # a name through here that Windows reads as a path -- the same
            # setting validating differently depending on where it runs, which
            # is how an entry that fails on a developer's machine reaches a
            # Linux deployment, or the reverse.
            if "/" in filename or "\\" in filename:
                raise ValueError(
                    f"Invalid corpus file collection entry: {filename!r} names a path. "
                    "The corpus directory is scanned without recursing, so an entry "
                    "matches a file by its filename alone and a path can never match."
                )
            if not collections:
                raise ValueError(
                    f"Invalid corpus file collection entry: {filename!r} names no "
                    "collection. A corpus source in no collection is invisible to every "
                    "filtered search; to tag it with nothing, configure no entry for it "
                    "and no corpus collections."
                )
            try:
                normalized[filename] = validate_collections(collections)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid corpus file collection entry for {filename!r}: {exc}"
                ) from exc
        return normalized


settings = Settings()
