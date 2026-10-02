from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING

from doc_etl_api.claims import ClaimStore, InProcessClaimStore
from doc_etl_api.jobs import instance_id
from doc_etl_api.pipeline import content_hash

if TYPE_CHECKING:
    from doc_etl_api.config import Settings
    from doc_etl_api.pipeline import IndexPipeline

logger = logging.getLogger(__name__)

# How often an instance that deferred a source to another looks again to see
# whether that instance is done with it. Short enough not to add noticeable
# delay to a deferring instance's startup, long enough that the check -- one
# indexed read -- is nothing against the parse it is waiting on.
CLAIM_POLL_SECONDS = 0.25


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


def _hold_claim(
    claims: ClaimStore,
    address: str,
    owner: str,
    *,
    stale_after: float,
    poll_seconds: float = CLAIM_POLL_SECONDS,
) -> bool:
    """Take the claim on *address*, or report that it could not be taken.

    A refusal is another instance loading the same source, and the answer is to
    wait rather than to load it a second time: that wait is the whole point of
    the claim. It ends either way within the ownership threshold, because a
    claim older than that is granted to whoever asks for it -- so an instance
    waiting on one whose owner has stopped takes it over and loads the source
    rather than waiting for an instance that is not coming back.

    An owner that is merely slow past the threshold is treated as one that
    stopped, which costs a duplicate ingestion; that is the same trade the
    ownership threshold makes for jobs, and it is affordable here for the same
    reason -- the replacement the second ingestion performs is atomic, so the
    index ends holding one of the two copies rather than a splice of both.

    The threshold is also the longest wait, which is what an instance that never
    gets the claim reports: a source this instance did not load and could not
    confirm is one the bootstrap must not call complete, and the deadline is
    what keeps that from being forever.
    """
    deadline = time.monotonic() + stale_after + poll_seconds
    while True:
        if claims.claim(address, owner, stale_after):
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(poll_seconds)


def _run_claimed(
    pipeline: IndexPipeline,
    claims: ClaimStore,
    state: BootstrapState,
    label: str,
    address: str,
    ingest: Callable[[], object],
    *,
    owner: str,
    stale_after: float,
) -> None:
    """Load one corpus source, with one instance doing the work for all of them.

    The claim is taken before the source is read, so an instance that loses the
    race contributes nothing rather than contributing the same parse and embed
    as the winner. What it does contribute is the wait, and what it must not
    contribute is a wrong answer about the source: it re-reads the store once it
    holds the claim, so the currency check it is about to make is against what
    the index holds -- including what another instance wrote while this one
    waited -- rather than against this process's own last known state.
    """
    if not _hold_claim(claims, address, owner, stale_after=stale_after):
        _record_failure(
            state,
            label,
            f"claimed by another instance that did not report it ingested within {stale_after:g}s",
        )
        return
    try:
        pipeline.refresh()
        _ingest_one(state, label, ingest)
    finally:
        # Released even when the ingestion failed, because a claim held by an
        # instance that has given up is a source no one will touch again until
        # it goes stale.
        claims.release(address, owner)


def ingest_corpus(
    pipeline: IndexPipeline,
    app_settings: Settings,
    state: BootstrapState,
    claims: ClaimStore | None = None,
    owner: str | None = None,
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

    *claims* is where instances starting together agree on which of them loads
    each source. Without one, the corpus is loaded as it was before claims
    existed -- which is right for a deployment with one instance and no shared
    index, and is what the in-memory backend always is.

    *owner* names this instance in a claim, and defaults to what this process is
    called. It is what a claim is released by and what a claim's holder is
    attributed to, so two callers sharing a name are one instance as far as the
    claims are concerned -- which is what they are, and what a caller running one
    process as several instances has to say otherwise.
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

    claims = InProcessClaimStore() if claims is None else claims
    # The threshold that decides when a claim belongs to an instance that
    # stopped. The same one job ownership uses -- see the module docstring of
    # `claims` for why it is the same number rather than a second setting.
    owner = instance_id() if owner is None else owner
    stale_after = float(app_settings.job_orphan_threshold_seconds)

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

        _run_claimed(
            pipeline,
            claims,
            state,
            str(path),
            path.name,
            ingest_file,
            owner=owner,
            stale_after=stale_after,
        )

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

        # Keyed by the URL as configured rather than by where it lands: where it
        # lands is only known once the request has been made, which is after the
        # point the claim exists to settle. An instance that defers a URL still
        # makes the request -- the landing address is what tells it whether the
        # page is already indexed -- but it is the parse, the chunk and the embed
        # that the claim keeps to one instance, and those are what it is for.
        _run_claimed(
            pipeline,
            claims,
            state,
            url,
            url,
            ingest_url,
            owner=owner,
            stale_after=stale_after,
        )

    state.status = BootstrapStatus.FAILED if state.failures else BootstrapStatus.COMPLETE
    logger.info(
        "Knowledge bootstrap finished status=%s considered=%d unchanged=%d failures=%d",
        state.status.value,
        len(files) + len(corpus_urls),
        len(unchanged),
        len(state.failures),
    )
