"""The application module must have no effect on import.

See `startup-model-preload` (change `doc-etl-api-env-and-startup-isolation`).
`create_app` loads the Docling and embedding models and starts the corpus
bootstrap. A module-level `app = create_app()` therefore made importing this
module pay a full model load -- and, with a corpus configured, a round of network
fetches -- before any caller had asked for an application. `run()` names the
factory on the uvicorn command line instead, so uvicorn constructs it at startup.

These assertions need a fresh interpreter. `tests/test_api.py` imports
`doc_etl_api.main` at module level, so by the time this file runs the import is
cached in `sys.modules` and re-importing it executes nothing to observe. The
child also inherits `DOC_ETL_API_ENV_FILE=""` from `tests/conftest.py`, so it
reads no `.env` -- these tests state their own configuration.
"""

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Runs in a child interpreter. The heavy constructors, the bootstrap entry point
# and the HTTP seam are all patched BEFORE `doc_etl_api.main` is imported, so an
# import-time call is recorded rather than performed -- which is what lets the
# detector check run without loading a model or reaching the network.
_PROBE = """
import sys
import threading
from unittest.mock import patch

main = None
raised = None

with (
    patch("llama_index.embeddings.huggingface.HuggingFaceEmbedding") as embedding,
    patch("doc_etl_api.pipeline.DoclingConverter") as converter,
    patch("doc_etl_api.bootstrap.ingest_corpus") as ingest,
    patch("requests.Session.request") as request,
):
    try:
        import doc_etl_api.main as main
    except Exception as exc:
        # An import that constructs an application fails here rather than at the
        # assertions below: `create_app` builds a real pipeline against these
        # mocks, which is far enough to raise. Recorded as a signal so the
        # failure names the import-time construction instead of only its
        # downstream symptom.
        raised = "import raised %s: %s" % (type(exc).__name__, exc)

performed = []
if raised is not None:
    performed.append(raised)
if converter.called:
    performed.append("DoclingConverter()")
if embedding.called:
    performed.append("HuggingFaceEmbedding()")
if ingest.called:
    performed.append("ingest_corpus()")
if request.called:
    performed.append("requests.Session.request()")
if main is not None and hasattr(main, "app"):
    performed.append("module attribute app")
# Read without waiting on the thread: `_start_corpus_bootstrap` names it, and it
# exists from `start()` onward, so this needs no sleep and cannot race.
if any(thread.name == "knowledge-bootstrap" for thread in threading.enumerate()):
    performed.append("knowledge-bootstrap thread")

if performed:
    sys.stderr.write("import performed: " + ", ".join(performed) + "\\n")
    sys.exit(1)
"""


def _import_in_fresh_interpreter(**env):
    """Import the application module in a child interpreter and return its result."""
    return subprocess.run(
        [sys.executable, "-c", _PROBE],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env={**os.environ, **env},
        timeout=300,
    )


def test_importing_the_module_loads_no_model_and_starts_no_bootstrap():
    """Importing the application module must construct nothing and load nothing."""
    result = _import_in_fresh_interpreter()
    assert result.returncode == 0, result.stderr


def test_importing_the_module_performs_no_corpus_fetch():
    """A configured corpus must not make import fetch it.

    The host is `.invalid`, reserved and unresolvable, so a fetch that escapes
    the patch fails immediately rather than reaching a real host.
    """
    result = _import_in_fresh_interpreter(KNOWLEDGE_CORPUS_URLS="https://example.invalid/guide")
    assert result.returncode == 0, result.stderr
