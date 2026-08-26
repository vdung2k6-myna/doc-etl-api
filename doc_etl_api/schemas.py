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


class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query string")
    top_k: int | None = Field(None, ge=1, le=100, description="Number of results to return")


class SearchResult(BaseModel):
    text: str = Field(..., description="Chunk text")
    score: float = Field(..., description="Relevance score")
    source_id: str = Field(..., description="Source identifier")
    source_type: str = Field(..., description="Source type: file or url")
    source_name: str = Field(..., description="Original filename or URL")


class SearchResponse(BaseModel):
    results: list[SearchResult] = Field(default_factory=list)


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
