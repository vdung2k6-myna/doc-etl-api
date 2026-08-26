import uuid
from io import BytesIO
from typing import Annotated
from urllib.parse import urlparse

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    HTTPException,
    Request,
    UploadFile,
    status,
)

from doc_etl_api.config import Settings, settings
from doc_etl_api.jobs import JobRegistry
from doc_etl_api.pipeline import IndexPipeline
from doc_etl_api.schemas import (
    IngestFileResponse,
    IngestUrlResponse,
    JobStatusResponse,
    SearchRequest,
    SearchResponse,
    SearchResult,
    UrlIngestRequest,
)


def _is_valid_http_url(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


router = APIRouter()


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
) -> None:
    job = jobs.get(job_id)
    if job is None:
        return
    try:
        result, timings = pipeline.ingest_url(source_id=source_id, url=url, timeout=timeout)
        job.complete(result, timings)
    except Exception as exc:
        job.fail(str(exc))


@router.post(
    "/sources/files",
    response_model=list[IngestFileResponse],
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload documents for ingestion",
    description="Upload one or more supported document files. Each file is parsed with Docling, "
    "chunked, embedded, and indexed into the vector store in the background. "
    "Use `GET /jobs/{job_id}` to poll for completion.",
)
async def ingest_files(
    pipeline: PipelineDep,
    jobs: JobsDep,
    background_tasks: BackgroundTasks,
    files: Annotated[list[UploadFile], File(...)],
) -> list[IngestFileResponse]:
    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No files provided.",
        )

    results: list[IngestFileResponse] = []
    for upload in files:
        if upload.size == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File {upload.filename!r} is empty.",
            )
        if not pipeline.converter.is_supported_file(upload.filename or ""):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"Unsupported file type for {upload.filename!r}. "
                    f"Supported extensions: {sorted(pipeline.converter.SUPPORTED_FILE_EXTENSIONS)}"
                ),
            )

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
            upload.filename or "unnamed",
            upload.content_type,
        )
        results.append(
            IngestFileResponse(
                job_id=job.job_id,
                source_id=source_id,
                name=upload.filename or "unnamed",
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
    for url in request.urls:
        if not _is_valid_http_url(url):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid URL: {url!r}. Only HTTP and HTTPS URLs are supported.",
            )

    results: list[IngestUrlResponse] = []
    for url in request.urls:
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
    description="Query the vector index and return ranked chunks with source metadata.",
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
    top_k = request.top_k or app_settings.default_top_k
    raw_results = pipeline.search(query, top_k=top_k)
    results = [SearchResult(**r) for r in raw_results]
    return SearchResponse(results=results)
