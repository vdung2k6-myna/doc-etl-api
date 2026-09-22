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
  -F "files=@notes.docx" \
  -F "collections=handbook"
```

Repeat `collections` to put a request's uploads in several of them (see
[Collections](#collections)). Every upload in one request is tagged with the same
set, so a request that needs different tags per file is submitted as several
requests.

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
`structural_nodes` counts nodes stored under a boundary the document itself
provided and `divided_nodes` counts the nodes a single unit had to be divided into
because it was wider than `CHUNK_SIZE`; both are counted as the boundaries are
chosen, so they say how the nodes were produced rather than how many survived the
floor and the de-duplication below. Every node of a section leads with that
section's heading, once — a node holding several of the section's units because
the floor merged them carries the heading a single time, not once per unit.
`overlap_tokens` is the smallest overlap
between adjacent nodes, measured below the heading a node leads with — adjacent
nodes of one section carry the same heading by construction, so it is not itself
overlap — and it is `0` for a source whose nodes were all bounded by the
document's own structure, because overlap applies only where a unit had to be
divided at sentence boundaries. `floor_merges` counts nodes folded into a
neighbour for falling below `MIN_CHUNK_TOKENS`, `floor_refusals` counts nodes kept
below it because every merge would have pushed a neighbour past the model's input
limit, and `duplicate_nodes` counts nodes dropped as text repeated inside this one
source. Repeats across different sources are never dropped.

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
      "duplicate_nodes": 0,
      "structural_nodes": 3,
      "divided_nodes": 2
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
  -d '{"urls": ["https://docling.ai/"], "collections": ["handbook"]}'
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
      "overlap_tokens": 0,
      "floor_merges": 0,
      "floor_refusals": 0,
      "duplicate_nodes": 1,
      "structural_nodes": 4,
      "divided_nodes": 0
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
embedded, and a heading inside those regions is removed with them. The main
content is located by a `<main>`, `<article>` or `role="main"` container where
the page declares one, and by text density where it does not. A page that offers
neither and yields no main content is converted whole, rather than producing an
empty document.

The page's own title is kept as a heading in the converted content, so a source
whose main content yields no headings of its own is still stored under a name
that identifies it, and searching for that title retrieves the source. The title
is read from the page's `<title>`, not from its first heading element, which may
name the site rather than the document. Where the main content already carries a
heading, none is added, so a document is never named twice.

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

### Collections

A collection is a label a source is filed under, so a search can be scoped to the
material meant for it instead of the whole index:

```bash
curl -X POST "http://localhost:8000/sources/urls" \
  -H "Content-Type: application/json" \
  -d '{"urls": ["https://example.com/csharp-12"], "collections": ["csharp", "handbook"]}'
```

```bash
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "nullable reference types", "top_k": 5, "collections": ["csharp"]}'
```

A source can belong to several collections, and a search naming several matches a
source in any of them. Re-ingesting a source replaces its collections along with
its content, so a source always belongs to what its most recent ingestion named.

- **A name is a slug.** Lowercase letters, digits and single hyphens or
  underscores, at most 64 characters — `csharp-12`, `dotnet_8`. Surrounding
  whitespace is trimmed; anything else is rejected with `400`, naming the value
  that was refused. Names are never rewritten, so the name you send is the name
  stored and the name to filter on.
- **A request may name at most 16 collections.** Each name is validated and the
  list de-duplicated before any work is queued, so a request with a malformed
  name costs a `400` rather than a job that fails later.
- **An empty filter is rejected.** Supplying `"collections": []` returns `400`:
  "search nothing" and "search everything" are both readings of an empty list, and
  the second silently answers a scoped question from unrelated content. Omit the
  field to search every indexed source.
- **A filter naming an unknown collection returns no results**, not an error, so a
  mistyped collection reads as "nothing matched" rather than widening the search.
- **Tagging does not change what is retrieved.** A collection is filtering
  metadata, never text the embedding model reads, so a source's chunks are
  chunked and embedded identically whether it is tagged or not — a caller who
  never filters sees the same results either way.

The startup corpus can be tagged from configuration so it is reachable by a
filtered search; see `KNOWLEDGE_CORPUS_COLLECTIONS` below. A source in no
collection matches no filter, so it is only findable by an unfiltered search —
which is what `GET /sources` is for.

### List indexed sources

```bash
curl "http://localhost:8000/sources"
```

Response:

```json
{
  "sources": [
    {
      "name": "example.pdf",
      "source_type": "file",
      "collections": ["handbook"],
      "chunk_count": 12
    },
    {
      "name": "https://example.com/archive",
      "source_type": "url",
      "collections": [],
      "chunk_count": 4
    }
  ]
}
```

One entry per indexed source, including sources in no collection — an empty
`collections` list is reported rather than the source being left out, so this is
how you find what a filter will never reach. `GET /sources` performs no retrieval
and no embedding, like `/health`, so it stays responsive regardless of index size
or search load.

### Search

```bash
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "document conversion", "top_k": 3}'
```

Add `"collections": ["handbook"]` to scope the search to sources filed under any
of those collections; omit it to search every indexed source. See
[Collections](#collections) for the name grammar and how a filter behaves.

Add `"neighbours": 1` to return the chunks stored either side of each result; see
[Neighbours](#neighbours) for what they are and what they cost.

Response:

```json
{
  "results": [
    {
      "text": "Docling converts documents to structured markdown...",
      "score": 0.89,
      "source_id": "a1b2c3d4...",
      "source_type": "file",
      "source_name": "example.pdf",
      "position": 4,
      "neighbours_before": 4,
      "neighbours_after": 7,
      "neighbours": [
        {"text": "Headings survive as markdown headings...", "position": 3},
        {"text": "Tables become markdown tables...", "position": 5}
      ]
    }
  ]
}
```

| Field | Meaning |
|---|---|
| `text` | The chunk's text. |
| `score` | Its relevance to the query. |
| `source_id`, `source_type`, `source_name` | Where the chunk came from. |
| `position` | Where the chunk sits in its source: counted from 0, in reading order. |
| `neighbours_before`, `neighbours_after` | How many chunks the source holds on each side of this one, whether or not any were returned. Zero at the source's first or last chunk. |
| `neighbours` | The adjacent chunks themselves, in reading order, when the request asked for them; empty when it did not. |

`top_k` applies within the scope rather than being padded from outside it: a
collection holding fewer chunks than `top_k` returns fewer results.

### Neighbours

A hit is a ranked chunk, and a chunk is not always a self-contained passage — the
sentence that makes one readable can be the next chunk. `neighbours` returns the
chunks stored around each result, so a caller can receive the passage without a
second request and without a second ranking:

```bash
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "document conversion", "top_k": 3, "neighbours": 1}'
```

- **The count is per side.** `"neighbours": 2` returns up to two chunks before
  and two after every result. It defaults to `0`, which returns none, and accepts
  `0`–`5`; anything else is rejected with `422`.
- **They are the adjacent stored chunks, not the adjacent paragraphs.** A chunk is
  what the pipeline stored after merging short passages and dropping duplicates,
  so the next chunk is the next text in the document in the common case but not in
  every case.
- **They attach to the result; they do not join the results.** The ranked list is
  the list the same search returns without `neighbours`, in the same order, so
  `top_k` still counts ranked hits and an unranked chunk never spends a slot.
- **A neighbour carries no score.** It was not ranked against the query, so a
  score would present context as equally relevant to the query that found the hit.
  Its `position` says where it sits instead.
- **They stop at the source boundary.** A result on its source's first or last
  chunk returns only the neighbours that exist, never another source's text, and a
  source holding fewer chunks than were asked for returns fewer rather than
  padding. A search scoped to collections takes neighbours only from sources in
  those collections.
- **Reading order, and repeated text when hits are adjacent.** Neighbours arrive in
  the order they appear in the source. When two hits are adjacent each carries the
  other, deliberately: a result's context does not depend on which other chunks the
  same query happened to return.
- **Every result grows three numbers wider**, whether or not neighbours were
  asked for, because the position and the counts say that context exists.

## Configuration

All settings are loaded from environment variables or an `.env` file. Copy
`.env.example` to `.env` and adjust as needed.

| Variable | Default | Description |
|---|---|---|
| `VECTOR_STORE_BACKEND` | `simple` | Vector store backend. Only `simple` is supported in this version. |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Hugging Face embedding model name. |
| `CHUNK_SIZE` | the embedding model's input limit | Bounds a node rather than placing its boundary, in the embedding model's tokens, special tokens included. A node's boundaries are the document's own — a heading opens a section, a paragraph break ends a unit — and a unit that fits is stored as itself; only a unit wider than this is divided further, at sentence boundaries, or at row boundaries with the header repeated if it is a table. Unset derives it from the model (256 for the default model), so it tracks the model rather than going stale against it; an explicit value larger than the model's limit is rejected at startup, because such chunks would be truncated before they were embedded. Measured on the corpus this was tuned on, it is a bound rather than a lever: lowering it from 8,192 to 512 moved the node count only from 386 to 404, because all but 10 of the 577 units were already smaller than 512. Set it only to go smaller. |
| `CHUNK_OVERLAP` | `50` | Overlap, in the same tokens as `CHUNK_SIZE`, applied only where a unit had to be divided at sentence boundaries — the one case where no boundary of the document's own was left to divide on. A node separated from its neighbour at a heading or paragraph boundary therefore repeats nothing: the seam is where the source changed subject, and repeating text across it would store the same text twice. Overlap is quantized to whole sentences, so the achieved overlap can be lower than this, and is zero when a single sentence exceeds the budget. |
| `MIN_CHUNK_TOKENS` | `32` | Minimum node size, in the same tokens as `CHUNK_SIZE` and measured the same way — on the node's text as the model reads it, the heading it carries included. A node smaller than this is merged into a neighbour of its own section in preference to one beyond a heading, so satisfying the minimum does not join two topics into one node; a merge that would take the neighbour past the embedding model's input limit is refused and the node kept, so the model's window is never traded away for the minimum. A merged node carries the heading its texts share once rather than once per text, since it opens with it already and the copies say nothing new. The default follows the measured noise floor: the corpus this was tuned on stored 9–14 token scraps that outranked real passages. Setting it to `0` stores every unit of the document as a node of its own, merging nothing. |
| `DEFAULT_TOP_K` | `5` | Default number of search results. |
| `MAX_FILE_SIZE_MB` | `50` | Maximum uploaded file size in MB. |
| `URL_FETCH_TIMEOUT_SECONDS` | `30` | Timeout for fetching URLs. |
| `USER_AGENT` | `doc-etl-api/0.1.0` | Sent on every page fetch. It names this service rather than the HTTP library, because hosts with a client-identity policy refuse the library's default — Wikipedia answers `403` to `python-requests/*` and `200` to this. Set it to include the contact details such a policy asks for. A blank value falls back to this default rather than sending an empty header, which those hosts refuse as well. |
| `KNOWLEDGE_CORPUS_DIR` | *(empty)* | Local directory whose supported files are ingested at startup, so a restart does not leave the index empty. Empty disables startup ingestion for local files. |
| `KNOWLEDGE_CORPUS_URLS` | *(empty)* | Comma-separated URLs ingested at startup. Empty contributes nothing. |
| `KNOWLEDGE_CORPUS_COLLECTIONS` | *(empty)* | Comma-separated collections every startup corpus source is tagged with, so a corpus is reachable by a filtered search. Empty leaves corpus sources untagged; a malformed name is rejected as the settings are built, which stops the service from starting rather than failing each corpus source in turn and leaving the index empty. |
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
