from __future__ import annotations

import logging
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING

from doc_etl_api.pipeline import content_hash

if TYPE_CHECKING:
    from doc_etl_api.config import Settings
    from doc_etl_api.pipeline import IndexPipeline

logger = logging.getLogger(__name__)


class BootstrapStatus(str, Enum):
    """Lifecycle of the startup corpus bootstrap."""

    DISABLED = "disabled"
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass
class BootstrapState:
    """Observable state of the startup corpus bootstrap.

    Reported through the readiness endpoint so a caller can tell "no corpus was
    configured" apart from "a corpus was configured and failed" apart from "a
    corpus is still being ingested".
    """

    status: BootstrapStatus = BootstrapStatus.DISABLED
    failures: list[str] = field(default_factory=list)


def corpus_files(directory: Path | None, supported: set[str]) -> list[Path]:
    """Supported files directly inside the corpus directory, in stable order."""
    if directory is None or not directory.is_dir():
        return []
    return sorted(
        path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in supported
    )


def _record_failure(state: BootstrapState, label: str, reason: str) -> None:
    """Record one corpus source's failure and log it, without propagating it.

    Both kinds of failure go through here -- a source that could not be ingested
    and a configuration entry that names a source the corpus does not have -- so
    the reason a bootstrap reports itself failed reads the same way either way.
    """
    state.failures.append(f"{label}: {reason}")
    logger.error("Knowledge bootstrap failed for source=%s error=%s", label, reason)


def _ingest_one(state: BootstrapState, label: str, ingest: Callable[[], object]) -> None:
    """Run one corpus source, recording rather than propagating its failure."""
    try:
        ingest()
    except Exception as exc:
        _record_failure(state, label, str(exc))


def ingest_corpus(
    pipeline: IndexPipeline,
    app_settings: Settings,
    state: BootstrapState,
) -> None:
    """Ingest the configured corpus, isolating the failure of each source.

    Reuses the ordinary ingestion path so the corpus goes through the same
    parse/chunk/embed/index stages as an uploaded file, and a source that fails
    does not abort the rest of the corpus.

    Each source is compared with what the index already holds before it is
    ingested, so a restart over a durable backend spends its time on the corpus
    that changed rather than on the whole corpus again. The comparison is on a
    digest of the source's own bytes -- for a file, the bytes on disk; for a URL,
    the body that came back -- which is the one thing that can be taken before
    the parse the comparison exists to avoid.
    """
    corpus_path = app_settings.knowledge_corpus_path
    corpus_urls = app_settings.knowledge_corpus_url_list
    # The whole corpus is tagged from one setting. Without it the corpus -- the
    # content that exists specifically to ground answers -- would be the one set
    # of sources invisible to every filtered search.
    collections = app_settings.knowledge_corpus_collection_list
    # A per-file entry replaces the setting above for the file it names, so one
    # corpus directory can hold documents belonging to different collections.
    # URLs are unaffected: an entry is keyed by filename.
    file_collections = app_settings.knowledge_corpus_file_collections

    if corpus_path is None and not corpus_urls:
        state.status = BootstrapStatus.DISABLED
        return

    state.status = BootstrapStatus.IN_PROGRESS
    files = corpus_files(corpus_path, pipeline.converter.SUPPORTED_FILE_EXTENSIONS)
    logger.info(
        "Knowledge bootstrap started dir=%s files=%d urls=%d collections=%s file_collections=%s",
        corpus_path,
        len(files),
        len(corpus_urls),
        collections,
        file_collections,
    )

    ingested: set[str] = set()
    unchanged: list[str] = []

    def skip(label: str) -> None:
        """Note a source the index already holds, which needs no work from here."""
        unchanged.append(label)
        logger.info("Knowledge bootstrap skipped source=%s reason=unchanged", label)

    for path in files:
        ingested.add(path.name)
        file_tag = file_collections.get(path.name, collections)

        def ingest_file(target: Path = path, tag: list[str] = file_tag) -> None:
            # Read whole before deciding, and read only: nothing is parsed,
            # chunked or embedded for a file the index holds as it now stands.
            payload = target.read_bytes()
            if pipeline.is_current(target.name, content_hash(payload), tag):
                skip(target.name)
                return
            pipeline.ingest_file(
                source_id=str(uuid.uuid4()),
                file=BytesIO(payload),
                filename=target.name,
                collections=tag,
            )

        _ingest_one(state, str(path), ingest_file)

    # An entry naming a file the corpus does not ingest -- absent, or of an
    # unsupported type -- is a configuration that does not do what it says.
    # Checked against what was ingested rather than what exists on disk, because
    # either way the tag the operator asked for was never applied. Reported like
    # a failed source: silently tagging nothing is the same outcome as tagging
    # the wrong collection, which is what this setting exists to prevent.
    for filename in sorted(set(file_collections) - ingested):
        _record_failure(
            state,
            filename,
            "named by the corpus file collections but not ingested from the corpus",
        )

    for url in corpus_urls:

        def ingest_url(target: str = url) -> None:
            # Fetched, because a page's content cannot be known without asking
            # for it, then compared before anything parses it. The fetch is the
            # cost this cannot avoid; the conversion is the cost it saves.
            body, final_url = pipeline.fetch_page(target)
            if pipeline.is_current(final_url, content_hash(body), collections):
                skip(target)
                return
            pipeline.ingest_page(
                source_id=str(uuid.uuid4()),
                url=target,
                body=body,
                final_url=final_url,
                collections=collections,
            )

        _ingest_one(state, url, ingest_url)

    state.status = BootstrapStatus.FAILED if state.failures else BootstrapStatus.COMPLETE
    logger.info(
        "Knowledge bootstrap finished status=%s considered=%d unchanged=%d failures=%d",
        state.status.value,
        len(files) + len(corpus_urls),
        len(unchanged),
        len(state.failures),
    )
