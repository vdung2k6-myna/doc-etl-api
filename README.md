# Documents ETL API

A FastAPI service that extracts structured content from documents and web pages,
transforms it into searchable chunks, and exposes it through a query interface.

It uses [Docling](https://docling.ai/) for layout-aware document and web-page
conversion to markdown, and [LlamaIndex](https://www.llamaindex.ai/) for
chunking, embedding, indexing, and retrieval.

## Features

- **File ingestion**: upload PDF, DOCX, TXT, HTML, and other supported formats.
- **URL ingestion**: submit one or more HTTP URLs and parse the main content of
  the fetched pages.
- **Unified ETL pipeline**: normalize all sources into LlamaIndex `Document`
  objects, chunk them, embed them, and load them into a vector index.
- **Search**: query the index and retrieve ranked chunks with source metadata.
- **Asynchronous ingestion**: file and URL ingestion run as background jobs so
  the HTTP response returns immediately. Poll `GET /jobs/{job_id}` for status,
  per-stage timings, and the final result.
- **Configurable**: vector store backend, embedding model, chunk size, and top-k
  results are all configurable via environment variables.

## Quick start

### 1. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

### 2. Run the API

```bash
cp .env.example .env
python -m doc_etl_api.main
```

Or using the installed script:

```bash
doc-etl-api
```

The API will be available at `http://localhost:8000`.

### 3. Explore the docs

FastAPI generates interactive OpenAPI documentation at:

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Usage

### Check service readiness

```bash
curl "http://localhost:8000/health"
```

Response:

```json
{
  "status": "ready",
  "indexed_sources": 2,
  "indexed_chunks": 41,
  "jobs_in_flight": 0,
  "bootstrap": "complete"
}
```

`/health` performs no retrieval and no embedding, so it stays responsive
regardless of index size or search load. `indexed_chunks: 0` means the index is
empty, which is a different condition from the service being unreachable — use
this endpoint to tell the two apart. `bootstrap` reports the startup corpus
state (`disabled`, `pending`, `in_progress`, `complete`, or `failed`).

`indexed_sources` and `indexed_chunks` count what the index currently holds, not
what has been submitted: they describe stored content, so re-ingesting a source
leaves both unchanged unless that source's content actually changed.

### Ingest a file

```bash
curl -X POST "http://localhost:8000/sources/files" \
  -F "files=@example.pdf" \
  -F "files=@notes.docx"
```

Response (HTTP 202 Accepted):

```json
[
  {
    "job_id": "550e8400-e29b-41d4-a716-446655440000",
    "source_id": "a1b2c3d4...",
    "source_type": "file",
    "name": "example.pdf",
    "mime_type": "application/pdf",
    "status": "pending"
  }
]
```

Poll for completion:

```bash
curl "http://localhost:8000/jobs/550e8400-e29b-41d4-a716-446655440000"
```

Example completed response:

`chunking` describes the nodes actually stored, in the embedding model's tokens.
`floor_merges` counts nodes folded into a neighbour for falling below
`MIN_CHUNK_TOKENS`, `floor_refusals` counts nodes kept below it because every
merge would have pushed a neighbour past the model's input limit, and
`duplicate_nodes` counts nodes dropped as text repeated inside this one source.
Repeats across different sources are never dropped.

```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "source_id": "a1b2c3d4...",
  "status": "completed",
  "result": {
    "source_id": "a1b2c3d4...",
    "source_type": "file",
    "name": "example.pdf",
    "mime_type": "application/pdf",
    "status": "indexed",
    "chunking": {
      "nodes": 4,
      "max_node_tokens": 198,
      "overlap_tokens": 39,
      "floor_merges": 1,
      "floor_refusals": 0,
      "duplicate_nodes": 0
    }
  },
  "timings": {
    "parse_ms": 1234.56,
    "chunk_ms": 12.34,
    "embed_ms": 567.89,
    "index_ms": 0.12
  },
  "error": null,
  "duration_seconds": 1.82
}
```

### Ingest URLs

```bash
curl -X POST "http://localhost:8000/sources/urls" \
  -H "Content-Type: application/json" \
  -d '{"urls": ["https://docling.ai/"]}'
```

Response (HTTP 202 Accepted):

```json
[
  {
    "job_id": "6ba7b810-9dad-11d1-80b4-00c04fd430c8",
    "source_id": "e5f6g7h8...",
    "source_type": "url",
    "name": "https://docling.ai/",
    "final_url": null,
    "status": "pending"
  }
]
```

Poll for completion:

```bash
curl "http://localhost:8000/jobs/6ba7b810-9dad-11d1-80b4-00c04fd430c8"
```

Example completed response:

```json
{
  "job_id": "6ba7b810-9dad-11d1-80b4-00c04fd430c8",
  "source_id": "e5f6g7h8...",
  "status": "completed",
  "result": {
    "source_id": "e5f6g7h8...",
    "source_type": "url",
    "name": "https://docling.ai/",
    "final_url": "https://docling.ai/",
    "status": "indexed",
    "chunking": {
      "nodes": 3,
      "max_node_tokens": 201,
      "overlap_tokens": 39,
      "floor_merges": 0,
      "floor_refusals": 0,
      "duplicate_nodes": 1
    }
  },
  "timings": {
    "parse_ms": 2345.67,
    "chunk_ms": 8.90,
    "embed_ms": 432.10,
    "index_ms": 0.05
  },
  "error": null,
  "duration_seconds": 2.79
}
```

#### What gets indexed

Only the page's main content is indexed. Navigation, sidebars, promotional
banners and footers are removed before conversion, so they are never chunked or
embedded. The main content is located by a `<main>`, `<article>` or
`role="main"` container where the page declares one, and by text density where
it does not. A page that offers neither and yields no main content is converted
whole, rather than producing an empty document.

Relative link and image targets are resolved against the page's final URL after
redirects, so ingested markdown carries absolute URLs: a root-relative
`href="/pricing"` is stored as `https://example.com/pricing`, not as a
filesystem path.

#### Re-ingesting a source

A source is identified by what it is rather than by the request that submitted
it: an uploaded file by its filename, a page by its final URL after redirects.
Submitting a source again therefore **replaces** the content its earlier
ingestion indexed instead of adding a second copy beside it. Re-upload a
corrected `example.pdf` and the superseded text stops appearing in search
results.

Three consequences worth knowing:

- **A name is an identity.** Two different documents uploaded under one filename
  are one source, the second replacing the first. Two URLs that redirect to the
  same page are likewise one source, whichever of them was submitted.
- **Repeats within one request collapse.** The same filename or the same URL
  given twice in a single request queues one job, because a second job would race
  a replacement of content the first is already storing. Every upload is still
  validated, so a rejected one is reported rather than silently dropped.
- **Sharing text is not sharing a copy.** A paragraph that two sources both
  contain is stored for each of them, so editing one leaves the other's copy
  untouched. Only a source's own earlier version is ever removed.

The counts reported by `GET /health` describe what the index holds rather than
what has been submitted, so they do not grow when unchanged content is ingested
again.

### Search

```bash
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "document conversion", "top_k": 3}'
```

Response:

```json
{
  "results": [
    {
      "text": "Docling converts documents to structured markdown...",
      "score": 0.89,
      "source_id": "a1b2c3d4...",
      "source_type": "file",
      "source_name": "example.pdf"
    }
  ]
}
```

## Configuration

All settings are loaded from environment variables or an `.env` file. Copy
`.env.example` to `.env` and adjust as needed.

| Variable | Default | Description |
|---|---|---|
| `VECTOR_STORE_BACKEND` | `simple` | Vector store backend. Only `simple` is supported in this version. |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Hugging Face embedding model name. |
| `CHUNK_SIZE` | the embedding model's input limit | Chunk size in the embedding model's tokens, special tokens included. Unset derives it from the model (256 for the default model), so it tracks the model rather than going stale against it; an explicit value larger than the model's limit is rejected at startup, because such chunks would be truncated before they were embedded. Set it only to go smaller. |
| `CHUNK_OVERLAP` | `50` | Overlap between consecutive chunks, in the same tokens as `CHUNK_SIZE`. Overlap is quantized to whole sentences, so the achieved overlap can be lower than this, and is zero when a single sentence exceeds the budget. |
| `MIN_CHUNK_TOKENS` | `32` | Minimum node size, in the same tokens as `CHUNK_SIZE` but measured on node content, where `CHUNK_SIZE` is measured on content plus metadata. A node smaller than this is merged into a neighbour rather than indexed alone; a merge that would take the neighbour past the embedding model's input limit is refused and the node kept, so the model's window is never traded away for the minimum. The default follows the measured noise floor: the corpus this was tuned on stored 9–14 token scraps that outranked real passages. Setting it to `0` restores the previous behaviour, indexing every node the splitter produces. |
| `DEFAULT_TOP_K` | `5` | Default number of search results. |
| `MAX_FILE_SIZE_MB` | `50` | Maximum uploaded file size in MB. |
| `URL_FETCH_TIMEOUT_SECONDS` | `30` | Timeout for fetching URLs. |
| `USER_AGENT` | `doc-etl-api/0.1.0` | Sent on every page fetch. It names this service rather than the HTTP library, because hosts with a client-identity policy refuse the library's default — Wikipedia answers `403` to `python-requests/*` and `200` to this. Set it to include the contact details such a policy asks for. A blank value falls back to this default rather than sending an empty header, which those hosts refuse as well. |
| `KNOWLEDGE_CORPUS_DIR` | *(empty)* | Local directory whose supported files are ingested at startup, so a restart does not leave the index empty. Empty disables startup ingestion for local files. |
| `KNOWLEDGE_CORPUS_URLS` | *(empty)* | Comma-separated URLs ingested at startup. Empty contributes nothing. |
| `LOG_LEVEL` | `INFO` | Level for application logs. Search timings and bootstrap progress are logged at `INFO`, so raising this to `WARNING` hides them. |
| `DOC_ETL_API_ENV_FILE` | `.env` | Which environment file to read — not a setting, and it cannot be set inside one. It is taken from the process environment before any setting is read, because it decides whether a file is read at all, so a value written into `.env` would never be seen. An empty value disables the file entirely: a `Settings()` built with no arguments then falls back to the code defaults instead of the developer's `.env`. That is what `tests/conftest.py` does, so a test session asserts the defaults rather than whatever corpus the developer has configured. Leave it unset to run the service. |

## Performance notes

- **Model pre-loading**: Docling and the Hugging Face embedding model are loaded
  once at application startup, so the first request does not pay cold-start
  costs.
- **Per-stage timing**: every ingestion job records `parse_ms`, `chunk_ms`,
  `embed_ms`, and `index_ms`. Use these values to identify which stage dominates
  for your workload.
- **Background ingestion**: large PDFs and slow URL fetches are processed in
  FastAPI `BackgroundTasks`. Callers receive a `202 Accepted` response with a
  `job_id` immediately and can poll `GET /jobs/{job_id}` until the job completes,
  fails, or times out.

## Running tests

```bash
pytest
```

## Linting and formatting

```bash
ruff check .
ruff format .
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
