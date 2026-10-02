"""What became of a background ingestion, and where that is remembered.

A job is accepted by one instance and polled at an address that may reach any of
them, so its record has to be readable from a process that never saw the
submission. The record is small -- a status, a result, an error, some timings --
so where it lives follows the index: rows beside the catalog when the backend is
Postgres, this process's memory when it is not.

The record of a job is not the job. Ingestion still runs in the instance that
accepted it and dies with it; what is shared is the answer to "what became of
it", which is what a caller polling a load-balanced address needs and what a
per-process dict cannot give.

Ownership is what makes an abandoned job distinguishable from a running one. A
row names the instance that owns it and when it last advanced, and a job that
has stopped advancing for longer than a configured threshold is reported failed,
naming that instance -- rather than staying pending for as long as the row
exists, which is forever.
"""

from __future__ import annotations

import logging
import os
import socket
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol

import sqlalchemy
from sqlalchemy.dialects import postgresql
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)


def instance_id() -> str:
    """What this process is called when a job names the instance that owns it.

    The host and the process id, because an operator who reads an error has to be
    able to find the instance it names: in a container the host is the pod or
    container's name, and the process id separates two workers sharing a host.
    A name that were merely unique -- a fresh UUID per process -- would identify
    the dead instance perfectly and locate it not at all.
    """
    return f"{socket.gethostname()}:{os.getpid()}"


class JobStatus(str, Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Job:
    job_id: str
    source_id: str
    status: JobStatus = JobStatus.PENDING
    result: dict[str, Any] | None = None
    error: str | None = None
    timings: dict[str, float] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    duration_seconds: float | None = None
    # The instance that owns this job. None means no owner was recorded -- a job
    # built by a caller that is not the registry -- and a reader must treat it as
    # "unattributable" rather than as an instance that has stopped: an orphan
    # verdict is a claim about a particular instance, and there is none to make
    # it about. Only a job that *is* recorded is ever reported orphaned.
    instance_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """The job as a status response, in the shape callers already parse.

        The owning instance is deliberately not a field here. This dict is the
        response body, and its shape is a contract the endpoints do not change;
        the instance reaches a caller through the error text of an orphaned job,
        which is where it is actionable anyway.
        """
        return {
            "job_id": self.job_id,
            "source_id": self.source_id,
            "status": self.status.value,
            "result": self.result,
            "error": self.error,
            "timings": self.timings,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "duration_seconds": self.duration_seconds,
        }

    def complete(self, result: dict[str, Any], timings: dict[str, float]) -> None:
        self.status = JobStatus.COMPLETED
        self.result = result
        self.timings = timings
        self.updated_at = time.time()
        self.duration_seconds = self.updated_at - self.created_at

    def fail(self, error: str, timings: dict[str, float] | None = None) -> None:
        self.status = JobStatus.FAILED
        self.error = error
        if timings is not None:
            self.timings = timings
        self.updated_at = time.time()
        self.duration_seconds = self.updated_at - self.created_at


class JobStore(Protocol):
    """Where a job's record is kept, as the registry needs it.

    Four operations and a heartbeat, which is the whole of what a job's record
    is asked: make one, read one, write one back, count the ones still running,
    and say that a running one is still advancing. Reading a job is what decides
    whether its owner is still alive, so the count takes the same cutoff the read
    uses -- otherwise the readiness endpoint would report a job as in flight that
    the job endpoint reports failed, for the same job at the same moment.
    """

    def create(self, source_id: str, owner: str) -> Job: ...

    def get(self, job_id: str) -> Job | None: ...

    def update(self, job: Job) -> None: ...

    def touch(self, job_id: str) -> None: ...

    def in_flight(self, refreshed_after: float) -> int: ...

    def close(self) -> None: ...


class InProcessJobStore:
    """The default backing: job records in this process's memory.

    Behaviour is exactly what it was when the records were a bare dict, with a
    lock added because a job now has a heartbeat thread advancing its timestamp
    while a request thread reads it. The dict is not shared with anything, so this
    backing is the one case where a job polled at a second instance still 404s --
    there is nowhere shared to put the record, and the README says so.
    """

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def create(self, source_id: str, owner: str) -> Job:
        job = Job(job_id=str(uuid.uuid4()), source_id=source_id, instance_id=owner)
        with self._lock:
            self._jobs[job.job_id] = job
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            job = self._jobs.get(job_id)
            # A copy, so a caller reading a job cannot be handed one another
            # thread is midway through advancing. Writes go back through
            # `update`, which is what a durable backing needs and what this one
            # now matches.
            return None if job is None else _copy(job)

    def update(self, job: Job) -> None:
        with self._lock:
            self._jobs[job.job_id] = _copy(job)

    def touch(self, job_id: str) -> None:
        """Advance a running job's timestamp, and only a running one.

        Guarded on the status so that a heartbeat's last beat, landing after the
        job has finished, cannot move the timestamp of a job whose duration has
        already been computed -- the record would then claim it was updated after
        it completed.
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if job is not None and job.status is JobStatus.PENDING:
                job.updated_at = time.time()

    def in_flight(self, refreshed_after: float) -> int:
        with self._lock:
            return sum(
                1
                for job in self._jobs.values()
                if job.status is JobStatus.PENDING and job.updated_at >= refreshed_after
            )

    def close(self) -> None:
        """Nothing to hand back: this store's records go when the process does."""


JOBS_TABLE_SUFFIX = "_jobs"


def jobs_table(base: str, metadata: sqlalchemy.MetaData) -> sqlalchemy.Table:
    """The table a job's record lives in, named after the collection's own table.

    Beside the catalog tables and under the same base name, so everything the
    service keeps in one database shares one identifier and a second collection
    is built by naming a second table -- the same rule the catalog follows.

    The timings and the result are JSON rather than columns of their own: they
    are the pipeline's own shapes, which this table stores and does not read
    into, and a column per field would have to be added to whenever a stage or a
    result field is.

    Indexed on the two columns the in-flight count filters by. That count runs on
    every readiness check while this table only grows -- one row per job ever
    accepted -- so the alternative is a scan that gets slower for the life of the
    deployment.
    """
    name = f"{base}{JOBS_TABLE_SUFFIX}"
    table = sqlalchemy.Table(
        name,
        metadata,
        sqlalchemy.Column("job_id", sqlalchemy.Text, primary_key=True),
        sqlalchemy.Column("source_id", sqlalchemy.Text, nullable=False),
        sqlalchemy.Column("status", sqlalchemy.Text, nullable=False),
        sqlalchemy.Column("result", postgresql.JSONB, nullable=True),
        sqlalchemy.Column("error", sqlalchemy.Text, nullable=True),
        sqlalchemy.Column("timings", postgresql.JSONB, nullable=False),
        sqlalchemy.Column("duration_seconds", sqlalchemy.Double, nullable=True),
        sqlalchemy.Column("created_at", sqlalchemy.Double, nullable=False),
        sqlalchemy.Column("updated_at", sqlalchemy.Double, nullable=False),
        sqlalchemy.Column("instance_id", sqlalchemy.Text, nullable=False),
        sqlalchemy.Index(f"{name}_status_updated_at", "status", "updated_at"),
    )
    return table


class PostgresJobStore:
    """The durable backing: one row per job, in the index's own database.

    It shares the index store's engine rather than opening its own, so a job
    written by an ingestion and the content that ingestion produced reach the
    same database through the same pool, and a deployment closes one thing.

    The table is created by the index store at startup, with the rest of the
    schema; a store built here is a way to read and write it, not a second place
    it is defined.
    """

    def __init__(self, engine: Engine, base: str) -> None:
        self._engine = engine
        self._table = jobs_table(base, sqlalchemy.MetaData())

    def create(self, source_id: str, owner: str) -> Job:
        job = Job(job_id=str(uuid.uuid4()), source_id=source_id, instance_id=owner)
        self.update(job)
        return job

    def get(self, job_id: str) -> Job | None:
        with self._engine.connect() as connection:
            row = connection.execute(
                sqlalchemy.select(self._table).where(self._table.c.job_id == job_id)
            ).first()
        return None if row is None else _from_row(row)

    def update(self, job: Job) -> None:
        """Write the whole record, replacing whatever the row held.

        A job is written a handful of times -- on creation, on completion, on
        failure, and by its heartbeat -- and every write carries the whole record,
        so there is no partial update to merge and no ordering to get wrong. The
        upsert is what lets `create` and `update` be the same statement.
        """
        values = {
            "job_id": job.job_id,
            "source_id": job.source_id,
            "status": job.status.value,
            "result": job.result,
            "error": job.error,
            "timings": job.timings,
            "duration_seconds": job.duration_seconds,
            "created_at": job.created_at,
            "updated_at": job.updated_at,
            "instance_id": job.instance_id or "",
        }
        with self._engine.begin() as connection:
            connection.execute(
                postgresql.insert(self._table)
                .values(**values)
                .on_conflict_do_update(
                    index_elements=[self._table.c.job_id],
                    set_={key: value for key, value in values.items() if key != "job_id"},
                )
            )

    def touch(self, job_id: str) -> None:
        with self._engine.begin() as connection:
            connection.execute(
                self._table.update()
                .where(
                    sqlalchemy.and_(
                        self._table.c.job_id == job_id,
                        self._table.c.status == JobStatus.PENDING.value,
                    )
                )
                .values(updated_at=time.time())
            )

    def in_flight(self, refreshed_after: float) -> int:
        with self._engine.connect() as connection:
            return connection.execute(
                sqlalchemy.select(sqlalchemy.func.count())
                .select_from(self._table)
                .where(
                    sqlalchemy.and_(
                        self._table.c.status == JobStatus.PENDING.value,
                        self._table.c.updated_at >= refreshed_after,
                    )
                )
            ).scalar_one()

    def close(self) -> None:
        """Nothing to hand back: the engine belongs to the index store."""


def _copy(job: Job) -> Job:
    """A job detached from the one a store holds, so reads cannot be mutated."""
    return Job(
        job_id=job.job_id,
        source_id=job.source_id,
        status=job.status,
        result=None if job.result is None else dict(job.result),
        error=job.error,
        timings=dict(job.timings),
        created_at=job.created_at,
        updated_at=job.updated_at,
        duration_seconds=job.duration_seconds,
        instance_id=job.instance_id,
    )


def _from_row(row: Any) -> Job:
    return Job(
        job_id=row.job_id,
        source_id=row.source_id,
        status=JobStatus(row.status),
        result=row.result,
        error=row.error,
        timings=dict(row.timings or {}),
        created_at=row.created_at,
        updated_at=row.updated_at,
        duration_seconds=row.duration_seconds,
        instance_id=row.instance_id or None,
    )


class JobRegistry:
    """The jobs this service knows about, whichever backing they live in.

    The interface the routes use is unchanged -- create, get, update, in_flight
    -- so a route neither knows nor selects the backing; `main.py` is what builds
    one from the configured backend, beside the pipeline it sits next to.
    """

    def __init__(
        self,
        store: JobStore,
        owner: str,
        orphan_threshold_seconds: float,
    ) -> None:
        self._store = store
        self._owner = owner
        self._orphan_threshold_seconds = orphan_threshold_seconds

    @property
    def owner(self) -> str:
        """The instance this registry writes as."""
        return self._owner

    def create(self, source_id: str) -> Job:
        return self._store.create(source_id, self._owner)

    def get(self, job_id: str) -> Job | None:
        """The job, or what is known about it that the row does not say.

        A pending job whose timestamp has stopped advancing has an owner that is
        no longer reporting it, so it is answered as failed rather than as
        pending. The verdict is written back before it is returned, so the row
        stops claiming to be running: the readiness count that filters on the
        same cutoff agrees with this answer, and every later read is a plain read
        rather than a second verdict about the same job.
        """
        job = self._store.get(job_id)
        if job is None:
            return None
        if not self._is_abandoned(job):
            return job
        return self._report_abandoned(job)

    def update(self, job: Job) -> None:
        self._store.update(job)

    def touch(self, job_id: str) -> None:
        """Say that a running job is still advancing. Called by its heartbeat."""
        self._store.touch(job_id)

    @property
    def in_flight(self) -> int:
        """Jobs that have been created but not yet completed or failed.

        An abandoned job is not one of them, on the same cutoff `get` uses: it
        has stopped running whether or not its row still says pending.
        """
        return self._store.in_flight(self._cutoff())

    def close(self) -> None:
        self._store.close()

    def _cutoff(self) -> float:
        return time.time() - self._orphan_threshold_seconds

    def _is_abandoned(self, job: Job) -> bool:
        """Whether *job* is pending with an owner that has stopped reporting it.

        A job with no recorded owner is never abandoned: the verdict names an
        instance, and a job that names none gives nothing to name.
        """
        return (
            job.status is JobStatus.PENDING
            and job.instance_id is not None
            and job.updated_at < self._cutoff()
        )

    def _report_abandoned(self, job: Job) -> Job:
        threshold = int(self._orphan_threshold_seconds)
        job.fail(
            f"Instance {job.instance_id} stopped while this job was running: it has "
            f"made no progress for more than {threshold} seconds, so the job is no "
            "longer being carried out and will not finish. Resubmit the source to "
            "ingest it again."
        )
        self._store.update(job)
        logger.warning(
            "Job %s reported failed: its instance %s stopped advancing it for more "
            "than %s seconds.",
            job.job_id,
            job.instance_id,
            threshold,
        )
        return job


# How often a running job says it is still advancing. Well inside a threshold of
# the default order, because the interval is what the threshold has to tolerate:
# the margin between them is how many beats an instance may miss -- one lost
# write, one slow stage -- before a healthy job is called abandoned.
HEARTBEAT_INTERVAL_SECONDS = 30.0


@contextmanager
def job_heartbeat(
    jobs: JobRegistry,
    job_id: str,
    interval: float = HEARTBEAT_INTERVAL_SECONDS,
) -> Iterator[None]:
    """Advance a running job's timestamp until the work it names is done.

    A job's last-updated timestamp is the only thing that distinguishes a job
    whose instance is working on it from one whose instance has stopped, so a job
    that takes longer than the threshold to finish has to keep saying so while it
    runs. Without this the two are the same row, and the more slowly a job
    legitimately runs -- a large document, a slow parse -- the more certainly it
    would be reported as belonging to a dead instance.

    The beat is a thread of its own because the work it covers is one long
    blocking call: a pipeline ingestion reports no progress to beat against, and
    waiting for it to finish is exactly the wait that must not be mistaken for
    silence.
    """
    stop = threading.Event()

    def beat() -> None:
        while not stop.wait(interval):
            jobs.touch(job_id)

    thread = threading.Thread(target=beat, name="job-heartbeat", daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join()
