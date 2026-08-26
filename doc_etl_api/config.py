from enum import Enum

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


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
    chunk_size: int = Field(default=512)
    chunk_overlap: int = Field(default=50)
    default_top_k: int = Field(default=5)

    max_file_size_mb: int = Field(default=50)
    url_fetch_timeout_seconds: int = Field(default=30)

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
