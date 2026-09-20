from __future__ import annotations

import logging
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

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


def _ingest_one(state: BootstrapState, label: str, ingest: Callable[[], object]) -> None:
    """Run one corpus source, recording rather than propagating its failure."""
    try:
        ingest()
    except Exception as exc:
        state.failures.append(f"{label}: {exc}")
        logger.error("Knowledge bootstrap failed for source=%s error=%s", label, exc)


def ingest_corpus(
    pipeline: IndexPipeline,
    app_settings: Settings,
    state: BootstrapState,
) -> None:
    """Ingest the configured corpus, isolating the failure of each source.

    Reuses the ordinary ingestion path so the corpus goes through the same
    parse/chunk/embed/index stages as an uploaded file, and a source that fails
    does not abort the rest of the corpus.
    """
    corpus_path = app_settings.knowledge_corpus_path
    corpus_urls = app_settings.knowledge_corpus_url_list
    # The whole corpus is tagged from one setting. Without it the corpus -- the
    # content that exists specifically to ground answers -- would be the one set
    # of sources invisible to every filtered search.
    collections = app_settings.knowledge_corpus_collection_list

    if corpus_path is None and not corpus_urls:
        state.status = BootstrapStatus.DISABLED
        return

    state.status = BootstrapStatus.IN_PROGRESS
    files = corpus_files(corpus_path, pipeline.converter.SUPPORTED_FILE_EXTENSIONS)
    logger.info(
        "Knowledge bootstrap started dir=%s files=%d urls=%d collections=%s",
        corpus_path,
        len(files),
        len(corpus_urls),
        collections,
    )

    for path in files:

        def ingest_file(target: Path = path) -> None:
            with target.open("rb") as handle:
                pipeline.ingest_file(
                    source_id=str(uuid.uuid4()),
                    file=handle,
                    filename=target.name,
                    collections=collections,
                )

        _ingest_one(state, str(path), ingest_file)

    for url in corpus_urls:
        _ingest_one(
            state,
            url,
            lambda target=url: pipeline.ingest_url(
                source_id=str(uuid.uuid4()), url=target, collections=collections
            ),
        )

    state.status = BootstrapStatus.FAILED if state.failures else BootstrapStatus.COMPLETE
    logger.info(
        "Knowledge bootstrap finished status=%s failures=%d",
        state.status.value,
        len(state.failures),
    )
