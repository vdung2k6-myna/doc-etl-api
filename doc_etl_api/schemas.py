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
    neighbours: int = Field(
        0,
        ge=0,
        le=5,
        description="How many adjacent chunks to return on each side of every result. "
        "The default, 0, returns none. Neighbours are the chunks stored before and "
        "after a result within its own source, not the paragraphs around it, and they "
        "carry no score because they were not ranked against the query.",
    )


class NeighbourChunk(BaseModel):
    text: str = Field(..., description="Chunk text")
    position: int = Field(
        ..., description="Zero-based position this chunk holds in its source, in reading order"
    )


class SearchResult(BaseModel):
    text: str = Field(..., description="Chunk text")
    score: float = Field(..., description="Relevance score")
    source_id: str = Field(..., description="Source identifier")
    source_type: str = Field(..., description="Source type: file or url")
    source_name: str = Field(..., description="Source filename or URL")
    address: str = Field(
        default="",
        description="The name the index stores this chunk's source under, and therefore the "
        "value that source's content is fetched by: pass it to `GET /sources/content`. For a "
        "file it is the filename; for a URL it is the final URL after redirects, which differs "
        "from the reported source name when the submitted URL redirected. It identifies a "
        "source to fetch, not to display: it is percent-encoded for a URL.",
    )
    collections: list[str] = Field(
        default_factory=list,
        description="The collections this chunk's source belongs to, the same set "
        "`GET /sources` reports for that source. Empty for a source ingested without any, "
        "which is a source that matches no filter rather than a missing value.",
    )
    position: int = Field(
        ...,
        description="Zero-based position this chunk holds in its source, in reading order",
    )
    neighbours_before: int = Field(
        ...,
        description="Chunks this chunk has before it in its source, whether or not they "
        "were requested. Zero for the source's first chunk.",
    )
    neighbours_after: int = Field(
        ...,
        description="Chunks this chunk has after it in its source, whether or not they "
        "were requested. Zero for the source's last chunk.",
    )
    neighbours: list[NeighbourChunk] = Field(
        default_factory=list,
        description="The chunks stored immediately before and after this one in its source, "
        "in reading order, up to the neighbour count the request asked for. Empty when it "
        "asked for none. They carry no score because they were not ranked against the query.",
    )


class SearchResponse(BaseModel):
    results: list[SearchResult] = Field(default_factory=list)


class SourceCatalogEntry(BaseModel):
    name: str = Field(..., description="Original filename or submitted URL")
    address: str = Field(
        ...,
        description="What the source is stored under and fetched by: the filename for a file, "
        "or the final URL for a URL. A URL that redirected reports the submitted URL as its "
        "name and the final URL as its address, so the two differ.",
    )
    source_type: Literal["file", "url"] = Field(..., description="Source type")
    collections: list[str] = Field(
        default_factory=list, description="Collections the source belongs to, if any"
    )
    chunk_count: int = Field(..., description="Number of chunks indexed for this source")


class SourceCatalogResponse(BaseModel):
    sources: list[SourceCatalogEntry] = Field(default_factory=list)


class SourceContentResponse(BaseModel):
    name: str = Field(..., description="Original filename or submitted URL")
    source_type: Literal["file", "url"] = Field(..., description="Source type")
    collections: list[str] = Field(
        default_factory=list, description="Collections the source belongs to, if any"
    )
    chunk_count: int = Field(..., description="Number of chunks stored for this source")
    chunks: list[NeighbourChunk] = Field(
        default_factory=list,
        description="Every chunk the index stores for this source, ordered by position. These "
        "are the chunks as they were chunked and indexed, which is not a copy of the document "
        "that was submitted: chunking removes repeated headings, splits tables and dedupes "
        "text. Join them to read the content through.",
    )


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
