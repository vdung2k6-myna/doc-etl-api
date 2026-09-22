import logging
import time
import uuid
from io import BytesIO
from typing import Annotated
from urllib.parse import urlparse

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
    status,
)
from starlette.concurrency import run_in_threadpool

from doc_etl_api.bootstrap import BootstrapState
from doc_etl_api.config import Settings, settings, validate_collections
from doc_etl_api.jobs import JobRegistry
from doc_etl_api.pipeline import IndexPipeline
from doc_etl_api.schemas import (
    HealthResponse,
    IngestFileResponse,
    IngestUrlResponse,
    JobStatusResponse,
    SearchRequest,
    SearchResponse,
    SearchResult,
    SourceCatalogEntry,
    SourceCatalogResponse,
    UrlIngestRequest,
)


def _is_valid_http_url(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _requested_collections(values: list[str] | None) -> list[str]:
    """The collections a request named, or a 400 naming the one that is malformed.

    Every route validates at the boundary and before it queues any work: a name
    outside the grammar is a caller's mistake, so it costs a 400 rather than a
    job that fails later, and the message names the value rather than leaving the
    caller to guess which of several was refused.
    """
    try:
        return validate_collections(values or [])
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


router = APIRouter()

logger = logging.getLogger(__name__)


def get_pipeline(request: Request) -> IndexPipeline:
    pipeline: IndexPipeline = request.app.state.pipeline
    return pipeline


def get_jobs(request: Request) -> JobRegistry:
    jobs: JobRegistry = request.app.state.jobs
    return jobs


PipelineDep = Annotated[IndexPipeline, Depends(get_pipeline)]
JobsDep = Annotated[JobRegistry, Depends(get_jobs)]


def _run_file_ingestion(
    pipeline: IndexPipeline,
    jobs: JobRegistry,
    job_id: str,
    source_id: str,
    file_bytes: bytes,
    filename: str,
    mime_type: str | None,
    collections: list[str],
) -> None:
    job = jobs.get(job_id)
    if job is None:
        return
    try:
        result, timings = pipeline.ingest_file(
            source_id=source_id,
            file=BytesIO(file_bytes),
            filename=filename,
            mime_type=mime_type,
            collections=collections,
        )
        job.complete(result, timings)
    except Exception as exc:
        job.fail(str(exc))


def _run_url_ingestion(
    pipeline: IndexPipeline,
    jobs: JobRegistry,
    job_id: str,
    source_id: str,
    url: str,
    timeout: int,
    collections: list[str],
) -> None:
    job = jobs.get(job_id)
    if job is None:
        return
    try:
        result, timings = pipeline.ingest_url(
            source_id=source_id, url=url, timeout=timeout, collections=collections
        )
        job.complete(result, timings)
    except Exception as exc:
        job.fail(str(exc))


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Service readiness",
    description="Report whether the service is ready and what is currently in the index. "
    "Performs no retrieval and no embedding, so it stays responsive regardless of "
    "index size or search load.",
)
async def health(
    request: Request,
    pipeline: PipelineDep,
    jobs: JobsDep,
) -> HealthResponse:
    bootstrap: BootstrapState = request.app.state.bootstrap
    return HealthResponse(
        status="ready",
        indexed_sources=pipeline.indexed_sources,
        indexed_chunks=pipeline.indexed_chunks,
        jobs_in_flight=jobs.in_flight,
        bootstrap=bootstrap.status.value,
    )


@router.post(
    "/sources/files",
    response_model=list[IngestFileResponse],
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload documents for ingestion",
    description="Upload one or more supported document files. Each file is parsed with Docling, "
    "chunked, embedded, and indexed into the vector store in the background. "
    "Repeat the `collections` form field to put the uploads in collections. "
    "Use `GET /jobs/{job_id}` to poll for completion.",
)
async def ingest_files(
    pipeline: PipelineDep,
    jobs: JobsDep,
    background_tasks: BackgroundTasks,
    files: Annotated[list[UploadFile], File(...)],
    collections: Annotated[list[str] | None, Form()] = None,
) -> list[IngestFileResponse]:
    requested_collections = _requested_collections(collections)

    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No files provided.",
        )

    results: list[IngestFileResponse] = []
    # A file's identity is its filename, so two uploads sharing one filename are
    # one source. Queuing both would race two replacements of the same content,
    # so the request reports that source once. Validation still runs for every
    # upload, so a rejected file is reported rather than silently skipped.
    queued_names: set[str] = set()
    for upload in files:
        name = upload.filename or "unnamed"
        if upload.size == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File {upload.filename!r} is empty.",
            )
        if not pipeline.converter.is_supported_file(name):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"Unsupported file type for {upload.filename!r}. "
                    f"Supported extensions: {sorted(pipeline.converter.SUPPORTED_FILE_EXTENSIONS)}"
                ),
            )
        if name in queued_names:
            continue
        queued_names.add(name)

        file_bytes = await upload.read()
        source_id = str(uuid.uuid4())
        job = jobs.create(source_id=source_id)
        background_tasks.add_task(
            _run_file_ingestion,
            pipeline,
            jobs,
            job.job_id,
            source_id,
            file_bytes,
            name,
            upload.content_type,
            requested_collections,
        )
        results.append(
            IngestFileResponse(
                job_id=job.job_id,
                source_id=source_id,
                name=name,
                mime_type=upload.content_type,
            )
        )
    return results


@router.post(
    "/sources/urls",
    response_model=list[IngestUrlResponse],
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit URLs for ingestion",
    description="Submit one or more HTTP URLs. Each page is fetched, parsed with Docling, "
    "chunked, embedded, and indexed into the vector store in the background. "
    "Use `GET /jobs/{job_id}` to poll for completion.",
)
async def ingest_urls(
    pipeline: PipelineDep,
    jobs: JobsDep,
    background_tasks: BackgroundTasks,
    request: UrlIngestRequest,
    app_settings: Annotated[Settings, Depends(lambda: settings)],
) -> list[IngestUrlResponse]:
    requested_collections = _requested_collections(request.collections)

    for url in request.urls:
        if not _is_valid_http_url(url):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid URL: {url!r}. Only HTTP and HTTPS URLs are supported.",
            )

    # The same URL submitted twice is one source; a second job would race a
    # replacement of content the first is already storing. URLs that differ but
    # redirect to one page collapse later, when their final URL is known.
    results: list[IngestUrlResponse] = []
    queued_urls: set[str] = set()
    for url in request.urls:
        if url in queued_urls:
            continue
        queued_urls.add(url)

        source_id = str(uuid.uuid4())
        job = jobs.create(source_id=source_id)
        background_tasks.add_task(
            _run_url_ingestion,
            pipeline,
            jobs,
            job.job_id,
            source_id,
            url,
            app_settings.url_fetch_timeout_seconds,
            requested_collections,
        )
        results.append(
            IngestUrlResponse(
                job_id=job.job_id,
                source_id=source_id,
                name=url,
            )
        )
    return results


@router.get(
    "/jobs/{job_id}",
    response_model=JobStatusResponse,
    summary="Get ingestion job status",
    description="Retrieve the current status, result, timing, and any error for a background ingestion job.",
)
async def get_job(
    jobs: JobsDep,
    job_id: str,
) -> JobStatusResponse:
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id!r} not found.",
        )
    return JobStatusResponse(**job.to_dict())


@router.post(
    "/search",
    response_model=SearchResponse,
    summary="Search indexed content",
    description="Query the vector index and return ranked chunks with source metadata. "
    "Supplying `collections` restricts the search to sources in any of them; "
    "omitting it searches every indexed source.",
)
async def search(
    pipeline: PipelineDep,
    request: SearchRequest,
    app_settings: Annotated[Settings, Depends(lambda: settings)],
) -> SearchResponse:
    query = request.query.strip()
    if not query:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query must contain non-whitespace characters.",
        )
    collections = request.collections
    if collections is not None and not collections:
        # "Search nothing" and "search everything" are both readings of an empty
        # list, and the second silently answers a scoped question with unrelated
        # content. Refusing it makes the caller's mistake loud instead of
        # plausible, so an empty list is never treated as no filter.
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The collections filter must name at least one collection; omit it to "
            "search every indexed source.",
        )
    if collections is not None:
        collections = _requested_collections(collections)

    top_k = request.top_k or app_settings.default_top_k
    # Retrieval embeds the query synchronously, so it must not run on the event
    # loop: doing so would block every other route for the duration.
    search_start = time.perf_counter()
    # An unfiltered search calls the pipeline exactly as it did before
    # collections existed, rather than passing a filter that happens to match
    # everything, so "no filter behaves as it always has" is structural. A search
    # that asks for no neighbours is passed the same way: the default is not sent,
    # so a caller that does not ask for adjacency reaches the ranked results and
    # nothing more.
    search_kwargs: dict = {"top_k": top_k}
    if collections is not None:
        search_kwargs["collections"] = collections
    if request.neighbours:
        search_kwargs["neighbours"] = request.neighbours
    raw_results = await run_in_threadpool(pipeline.search, query, **search_kwargs)
    search_ms = round((time.perf_counter() - search_start) * 1000, 2)
    results = [SearchResult(**r) for r in raw_results]
    logger.info(
        "Search query=%r top_k=%d collections=%s neighbours=%d results=%d search_ms=%s",
        query,
        top_k,
        collections,
        request.neighbours,
        len(results),
        search_ms,
    )
    return SearchResponse(results=results)


@router.get(
    "/sources",
    response_model=SourceCatalogResponse,
    summary="List indexed sources",
    description="Report every source currently in the index with its name, type, collections, "
    "and chunk count. Served from state maintained as sources are ingested, so it "
    "performs no retrieval and no embedding.",
)
async def list_sources(pipeline: PipelineDep) -> SourceCatalogResponse:
    return SourceCatalogResponse(
        sources=[
            SourceCatalogEntry(
                name=record.name,
                source_type=record.source_type,
                collections=list(record.collections),
                chunk_count=record.chunk_count,
            )
            for record in pipeline.source_catalog
        ]
    )
