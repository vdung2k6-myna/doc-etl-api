from typing import Any, Literal

from pydantic import BaseModel, Field


class IngestFileResponse(BaseModel):
    job_id: str = Field(..., description="Unique identifier for the ingestion job")
    source_id: str = Field(..., description="Unique identifier for the ingested source")
    source_type: Literal["file"] = "file"
    name: str = Field(..., description="Original filename")
    mime_type: str | None = Field(None, description="Detected MIME type")
    status: Literal["pending"] = "pending"


class IngestUrlResponse(BaseModel):
    job_id: str = Field(..., description="Unique identifier for the ingestion job")
    source_id: str = Field(..., description="Unique identifier for the ingested source")
    source_type: Literal["url"] = "url"
    name: str = Field(..., description="Original URL")
    final_url: str | None = Field(None, description="Final URL after redirects")
    status: Literal["pending"] = "pending"


class UrlIngestRequest(BaseModel):
    urls: list[str] = Field(..., min_length=1, description="URLs to ingest")
    collections: list[str] | None = Field(
        None, description="Collections the ingested sources belong to"
    )


class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query string")
    top_k: int | None = Field(None, ge=1, le=100, description="Number of results to return")
    collections: list[str] | None = Field(
        None,
        description="Only return chunks from sources in any of these collections. "
        "Omit to search every indexed source. An empty list is rejected.",
    )


class SearchResult(BaseModel):
    text: str = Field(..., description="Chunk text")
    score: float = Field(..., description="Relevance score")
    source_id: str = Field(..., description="Source identifier")
    source_type: str = Field(..., description="Source type: file or url")
    source_name: str = Field(..., description="Original filename or URL")


class SearchResponse(BaseModel):
    results: list[SearchResult] = Field(default_factory=list)


class SourceCatalogEntry(BaseModel):
    name: str = Field(..., description="Original filename or submitted URL")
    source_type: Literal["file", "url"] = Field(..., description="Source type")
    collections: list[str] = Field(
        default_factory=list, description="Collections the source belongs to, if any"
    )
    chunk_count: int = Field(..., description="Number of chunks indexed for this source")


class SourceCatalogResponse(BaseModel):
    sources: list[SourceCatalogEntry] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str = Field(..., description="Service readiness: 'ready'")
    indexed_sources: int = Field(..., description="Number of sources in the index")
    indexed_chunks: int = Field(..., description="Number of chunks in the index")
    jobs_in_flight: int = Field(..., description="Ingestion jobs not yet finished")
    bootstrap: str = Field(
        ...,
        description="Startup corpus bootstrap state: disabled, pending, in_progress, complete, or failed",
    )


class JobStatusResponse(BaseModel):
    job_id: str = Field(..., description="Unique identifier for the ingestion job")
    source_id: str = Field(..., description="Unique identifier for the ingested source")
    status: str = Field(..., description="Job status: pending, completed, or failed")
    result: dict[str, Any] | None = Field(None, description="Source metadata when completed")
    error: str | None = Field(None, description="Error message when failed")
    timings: dict[str, float] = Field(
        default_factory=dict, description="Per-stage timing in milliseconds"
    )
    duration_seconds: float | None = Field(None, description="Total elapsed time in seconds")
