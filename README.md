# Documents ETL API

A FastAPI service that extracts structured content from documents and web pages,
transforms it into searchable chunks, and exposes it through a query interface.

It uses [Docling](https://docling.ai/) for layout-aware document and web-page
conversion to markdown, and [LlamaIndex](https://www.llamaindex.ai/) for
chunking, embedding, indexing, and retrieval.

## Features

- **File ingestion**: upload PDF, DOCX, TXT, HTML, and other supported formats.
- **URL ingestion**: submit one or more HTTP URLs and parse the fetched pages.
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
    "status": "indexed"
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
    "status": "indexed"
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
| `CHUNK_SIZE` | `512` | Chunk size in tokens/characters. |
| `CHUNK_OVERLAP` | `50` | Overlap between consecutive chunks. |
| `DEFAULT_TOP_K` | `5` | Default number of search results. |
| `MAX_FILE_SIZE_MB` | `50` | Maximum uploaded file size in MB. |
| `URL_FETCH_TIMEOUT_SECONDS` | `30` | Timeout for fetching URLs. |

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
