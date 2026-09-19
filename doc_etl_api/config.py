import logging
from enum import Enum
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_LOG_LEVELS = tuple(
    logging.getLevelName(level)
    for level in (logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL)
)

# Sent on every page fetch. It names the service rather than the HTTP library,
# because a host with a client-identity policy refuses the library's default.
# No contact details are invented here: a host whose policy asks for them gets
# whatever the operator sets, and a deployment that never sets it still sends
# an identity that such hosts accept.
DEFAULT_USER_AGENT = "doc-etl-api/0.1.0"


class VectorStoreBackend(str, Enum):
    SIMPLE = "simple"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
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


settings = Settings()
