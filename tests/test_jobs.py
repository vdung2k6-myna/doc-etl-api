"""What became of a background ingestion, and where the answer is remembered.

The registry's own behaviour is tested here against the in-memory backing, which
is the one every machine has: what a job records, when a job is called abandoned,
and that the heartbeat keeps a slow job from being mistaken for a dead one. The
durable backing is tested at the end of the file against a real database, because
what it has to establish -- that a job written by one instance is readable by
another -- is not something an in-process dict can be asked about.
"""

import itertools
import json
import os
import subprocess
import sys
import time
from collections.abc import Iterator
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import sqlalchemy
from fastapi.testclient import TestClient
from sqlalchemy.pool import NullPool

from doc_etl_api import routes
from doc_etl_api.config import Settings
from doc_etl_api.jobs import (
    HEARTBEAT_INTERVAL_SECONDS,
    InProcessJobStore,
    Job,
    JobRegistry,
    JobStatus,
    PostgresJobStore,
    instance_id,
    job_heartbeat,
)
from doc_etl_api.store import PostgresIndexStore
from tests.test_postgres_store import (
    DATABASE_URL,
    _create,
    _drop_collection,
    _settings,
    _started_app,
    needs_database,
)

# A threshold short enough that a test can outlive it without sleeping for the
# default's five minutes, and long enough that the registry built around it does
# not call a job abandoned between two statements.
SHORT_THRESHOLD_SECONDS = 0.5


def _registry(
    store=None,
    owner: str = "instance-under-test",
    threshold: float = SHORT_THRESHOLD_SECONDS,
) -> JobRegistry:
    return JobRegistry(
        store=store or InProcessJobStore(),
        owner=owner,
        orphan_threshold_seconds=threshold,
    )


def _age(jobs: JobRegistry, job_id: str, seconds: float) -> None:
    """Move a job's last-updated timestamp back, so it reads as neglected.

    Sleeping for the threshold instead would test the clock rather than the
    verdict, and would make every one of these tests as slow as its own timeout.
    """
    job = jobs.get(job_id)
    assert job is not None
    job.updated_at -= seconds
    jobs.update(job)


# --- The record a job is -------------------------------------------------------


def test_a_job_round_trips_through_the_registry():
    """Create, read, write back, read again: the four operations, in order."""
    jobs = _registry()
    created = jobs.create(source_id="source-1")

    assert created.status is JobStatus.PENDING
    assert jobs.get(created.job_id).source_id == "source-1"

    stored = jobs.get(created.job_id)
    stored.complete({"name": "report.pdf"}, {"parse_ms": 12.5})
    jobs.update(stored)

    finished = jobs.get(created.job_id)
    assert finished.status is JobStatus.COMPLETED
    assert finished.result == {"name": "report.pdf"}
    assert finished.timings == {"parse_ms": 12.5}
    assert finished.duration_seconds is not None


def test_a_failed_job_keeps_the_error_it_failed_with():
    jobs = _registry()
    created = jobs.create(source_id="source-1")

    stored = jobs.get(created.job_id)
    stored.fail("the converter refused the file")
    jobs.update(stored)

    failed = jobs.get(created.job_id)
    assert failed.status is JobStatus.FAILED
    assert failed.error == "the converter refused the file"


def test_an_unknown_job_is_not_found_rather_than_invented():
    assert _registry().get("no-such-job") is None


def test_a_job_names_the_instance_that_owns_it():
    """The name has to locate the instance, not merely identify it."""
    jobs = _registry(owner="worker-3:4711")
    assert jobs.get(jobs.create(source_id="source-1").job_id).instance_id == "worker-3:4711"


def test_the_default_instance_name_is_the_host_and_process():
    name = instance_id()
    assert ":" in name
    assert name.rsplit(":", 1)[1].isdigit()


def test_a_read_hands_back_a_copy_rather_than_the_stored_job():
    """Mutating what a read returned must not be a write nobody recorded."""
    jobs = _registry()
    created = jobs.create(source_id="source-1")

    read = jobs.get(created.job_id)
    read.status = JobStatus.COMPLETED

    assert jobs.get(created.job_id).status is JobStatus.PENDING


# --- Which jobs count as running -----------------------------------------------


def test_in_flight_counts_the_jobs_that_have_not_finished():
    jobs = _registry()
    jobs.create(source_id="source-1")
    second = jobs.create(source_id="source-2")

    assert jobs.in_flight == 2

    finished = jobs.get(second.job_id)
    finished.complete({}, {})
    jobs.update(finished)

    assert jobs.in_flight == 1


def test_in_flight_excludes_a_job_whose_instance_stopped():
    jobs = _registry()
    created = jobs.create(source_id="source-1")
    _age(jobs, created.job_id, SHORT_THRESHOLD_SECONDS * 2)

    assert jobs.in_flight == 0


# --- A job whose instance stopped ----------------------------------------------


def test_a_job_that_stopped_advancing_is_reported_failed_naming_its_instance():
    jobs = _registry(owner="worker-1:99")
    created = jobs.create(source_id="source-1")
    _age(jobs, created.job_id, SHORT_THRESHOLD_SECONDS * 2)

    reported = jobs.get(created.job_id)

    assert reported.status is JobStatus.FAILED
    assert "worker-1:99" in reported.error
    assert reported.duration_seconds is not None


def test_the_verdict_is_written_back_so_every_later_read_agrees():
    """A row that still said pending after the verdict would disagree with it."""
    jobs = _registry()
    created = jobs.create(source_id="source-1")
    _age(jobs, created.job_id, SHORT_THRESHOLD_SECONDS * 2)

    jobs.get(created.job_id)

    assert jobs.get(created.job_id).status is JobStatus.FAILED
    assert jobs.in_flight == 0


def test_a_job_whose_instance_is_running_reports_its_own_status():
    jobs = _registry(owner="worker-1:99")
    created = jobs.create(source_id="source-1")

    reported = jobs.get(created.job_id)

    assert reported.status is JobStatus.PENDING
    assert reported.error is None


def test_a_completed_job_is_never_called_abandoned_for_being_old():
    """Age only means something about a job that claims to still be running."""
    jobs = _registry()
    created = jobs.create(source_id="source-1")
    stored = jobs.get(created.job_id)
    stored.complete({"name": "report.pdf"}, {})
    jobs.update(stored)
    _age(jobs, created.job_id, SHORT_THRESHOLD_SECONDS * 100)

    reported = jobs.get(created.job_id)

    assert reported.status is JobStatus.COMPLETED
    assert reported.error is None


def test_a_job_naming_no_instance_is_never_called_abandoned():
    """The verdict names an owner; a job with none gives nothing to name."""
    store = InProcessJobStore()
    jobs = _registry(store=store)
    orphan = Job(job_id="ownerless", source_id="source-1")
    orphan.updated_at -= SHORT_THRESHOLD_SECONDS * 2
    store.update(orphan)

    reported = jobs.get("ownerless")

    assert reported.status is JobStatus.PENDING
    assert reported.error is None


# --- The heartbeat -------------------------------------------------------------


def test_the_heartbeat_keeps_a_slow_job_from_looking_abandoned():
    """Work that outlasts the threshold still reports as running while it runs."""
    jobs = _registry()
    created = jobs.create(source_id="source-1")
    before = jobs.get(created.job_id).updated_at

    with job_heartbeat(jobs, created.job_id, interval=SHORT_THRESHOLD_SECONDS / 5):
        time.sleep(SHORT_THRESHOLD_SECONDS * 2)

    reported = jobs.get(created.job_id)
    assert reported.status is JobStatus.PENDING
    assert reported.updated_at > before


def test_the_heartbeat_leaves_a_finished_job_alone():
    """A late beat must not move a timestamp whose duration is already computed."""
    jobs = _registry()
    created = jobs.create(source_id="source-1")

    with job_heartbeat(jobs, created.job_id, interval=0.02):
        stored = jobs.get(created.job_id)
        stored.complete({}, {})
        jobs.update(stored)
        completed_at = stored.updated_at
        time.sleep(0.1)

    assert jobs.get(created.job_id).updated_at == completed_at


def test_the_heartbeat_stops_beating_when_the_work_ends():
    """The thread is joined on exit, so no beat lands after the job is written."""
    jobs = _registry()
    created = jobs.create(source_id="source-1")

    with job_heartbeat(jobs, created.job_id, interval=0.02):
        pass

    finished_at = jobs.get(created.job_id).updated_at
    time.sleep(0.1)

    assert jobs.get(created.job_id).updated_at == finished_at


def test_the_beat_is_well_inside_the_default_threshold():
    assert HEARTBEAT_INTERVAL_SECONDS * 2 < Settings().job_orphan_threshold_seconds


# --- The durable backing, against a real database ------------------------------


def _postgres_job_store(settings: Settings) -> PostgresJobStore:
    """The durable job store of a pipeline built for the test database."""
    store = _create(settings).store
    assert isinstance(store, PostgresIndexStore), "the test database is not in use"
    return store.job_store()


def _postgres_registry(settings: Settings, owner: str) -> JobRegistry:
    return JobRegistry(
        store=_postgres_job_store(settings),
        owner=owner,
        orphan_threshold_seconds=60,
    )


_JOB_TABLES = itertools.count(1)


@pytest.fixture
def collection() -> Iterator[str]:
    """A collection no other test uses, dropped before and after it runs.

    The prefix is this module's own rather than the store suite's fixture reused:
    a fixture defined in another test module is not visible here, and the two
    counters would name the same tables if they could be. Dropped before as well
    as after, so a run that died mid-test does not hand the next one a database
    it did not build.
    """
    name = f"doc_etl_api_test_jobs_{next(_JOB_TABLES)}"
    _drop_collection(name)
    yield name
    _drop_collection(name)


@needs_database
def test_a_job_round_trips_through_a_real_database(collection):
    """What one instance writes, another reads -- result and duration included."""
    settings = _settings(table=collection)
    accepting = _postgres_registry(settings, owner="worker-1:1")
    polling = _postgres_registry(settings, owner="worker-2:2")
    try:
        created = accepting.create(source_id="source-1")

        assert polling.get(created.job_id).status is JobStatus.PENDING

        stored = accepting.get(created.job_id)
        stored.complete({"name": "report.pdf"}, {"parse_ms": 12.5})
        accepting.update(stored)

        from_elsewhere = polling.get(created.job_id)
        assert from_elsewhere.status is JobStatus.COMPLETED
        assert from_elsewhere.result == {"name": "report.pdf"}
        assert from_elsewhere.timings == {"parse_ms": 12.5}
        assert from_elsewhere.duration_seconds == stored.duration_seconds
    finally:
        _drop_collection(collection)


@needs_database
def test_an_error_message_survives_a_real_database(collection):
    settings = _settings(table=collection)
    accepting = _postgres_registry(settings, owner="worker-1:1")
    polling = _postgres_registry(settings, owner="worker-2:2")
    try:
        created = accepting.create(source_id="source-1")
        stored = accepting.get(created.job_id)
        stored.fail("the converter refused the file")
        accepting.update(stored)

        reported = polling.get(created.job_id)
        assert reported.status is JobStatus.FAILED
        assert reported.error == "the converter refused the file"
    finally:
        _drop_collection(collection)


@needs_database
def test_a_job_row_names_the_instance_that_wrote_it(collection):
    settings = _settings(table=collection)
    accepting = _postgres_registry(settings, owner="worker-1:1")
    try:
        created = accepting.create(source_id="source-1")

        assert accepting.get(created.job_id).instance_id == "worker-1:1"
        assert _instance_id_in_row(collection, created.job_id) == "worker-1:1"
    finally:
        _drop_collection(collection)


@needs_database
def test_the_timestamp_a_row_holds_is_the_one_the_heartbeat_advances(collection):
    """The beat is only a liveness signal if it reaches the row, not just memory."""
    settings = _settings(table=collection)
    jobs = _postgres_registry(settings, owner="worker-1:1")
    try:
        created = jobs.create(source_id="source-1")
        before = jobs.get(created.job_id).updated_at

        with job_heartbeat(jobs, created.job_id, interval=0.05):
            time.sleep(0.2)

        assert jobs.get(created.job_id).updated_at > before
    finally:
        _drop_collection(collection)


@needs_database
def test_in_flight_is_counted_from_the_rows(collection):
    settings = _settings(table=collection)
    jobs = _postgres_registry(settings, owner="worker-1:1")
    try:
        first = jobs.create(source_id="source-1")
        jobs.create(source_id="source-2")
        assert jobs.in_flight == 2

        stored = jobs.get(first.job_id)
        stored.complete({}, {})
        jobs.update(stored)

        assert jobs.in_flight == 1
    finally:
        _drop_collection(collection)


@needs_database
def test_a_job_left_by_a_stopped_instance_reads_as_failed(collection):
    """The row outlives the instance, so the verdict has to come from the row."""
    settings = _settings(table=collection)
    accepting = _postgres_registry(settings, owner="worker-1:1")
    polling = _postgres_registry(settings, owner="worker-2:2")
    try:
        created = accepting.create(source_id="source-1")

        # The instance that accepted it is gone; the row it wrote is not, and it
        # stops advancing at the moment that instance stopped.
        _age(polling, created.job_id, 120)

        reported = polling.get(created.job_id)
        assert reported.status is JobStatus.FAILED
        assert "worker-1:1" in reported.error
    finally:
        _drop_collection(collection)


@needs_database
def test_an_ingestion_advances_its_row_while_it_is_still_running(collection, monkeypatch):
    """The beat has to reach the row from inside the ingestion, not only after it.

    Read from within the conversion, which is the one moment at which the job is
    still running and the ingestion has not yet written its own completion: after
    the route returns, `complete` has set the timestamp whatever the beat did.
    """
    settings = _settings(table=collection)
    seen: list[tuple[float, float]] = []

    def convert_file(file, filename):
        # Long enough for several beats at the interval patched in below, and over
        # in under a second rather than the thirty the real interval would cost.
        time.sleep(0.6)
        seen.append(_only_row_times(collection))
        return "A paragraph of prose."

    converter = MagicMock()
    converter.is_supported_file.return_value = True
    converter.SUPPORTED_FILE_EXTENSIONS = {".txt"}
    converter.convert_file.side_effect = convert_file

    # The interval is a default argument, so the seam is the function the route
    # calls rather than the constant behind the default.
    beating = routes.job_heartbeat
    monkeypatch.setattr(
        routes, "job_heartbeat", lambda jobs, job_id: beating(jobs, job_id, interval=0.05)
    )

    client = TestClient(_started_app(converter, settings))
    response = client.post(
        "/sources/files",
        files={"files": ("report.txt", BytesIO(b"report content"), "text/plain")},
    )
    assert response.status_code == 202
    job_id = response.json()[0]["job_id"]

    assert seen, "the conversion never ran"
    created_at, updated_at = seen[0]
    assert updated_at > created_at + 0.2, "the row did not advance while the job ran"

    # And the record it finished with is one another instance can read.
    polling = _postgres_registry(settings, owner="poller:2")
    assert polling.get(job_id).status is JobStatus.COMPLETED


def _only_row_times(table: str) -> tuple[float, float]:
    """The created and updated timestamps of the one job row a test has written."""
    engine = sqlalchemy.create_engine(DATABASE_URL, poolclass=NullPool)
    try:
        with engine.connect() as connection:
            return connection.execute(
                sqlalchemy.text(f'SELECT created_at, updated_at FROM "{table}_jobs"')
            ).one()
    finally:
        engine.dispose()


def _instance_id_in_row(table: str, job_id: str) -> str:
    engine = sqlalchemy.create_engine(DATABASE_URL, poolclass=NullPool)
    try:
        with engine.connect() as connection:
            return connection.execute(
                sqlalchemy.text(f'SELECT instance_id FROM "{table}_jobs" WHERE job_id = :id'),
                {"id": job_id},
            ).scalar_one()
    finally:
        engine.dispose()


# --- Two instances, one database -----------------------------------------------
#
# Run as a second interpreter rather than as a second registry in this one. What
# the requirement claims is that a job outlives the instance that accepted it, and
# two objects in one interpreter share memory in a way two instances do not: a
# registry that read its own process's dict would pass either way.
_ACCEPTING_INSTANCE_SOURCE = """
import json
import os
import time
from pathlib import Path

from doc_etl_api.jobs import JobRegistry
from tests.test_postgres_store import _create, _settings

pipeline = _create(_settings(table=os.environ["JOBS_TABLE"]))
jobs = JobRegistry(
    store=pipeline.store.job_store(),
    owner=os.environ["JOBS_OWNER"],
    orphan_threshold_seconds=60,
)

job = jobs.create(source_id="source-1")
print(json.dumps({"job_id": job.job_id}), flush=True)

if os.environ["JOBS_BEHAVIOUR"] == "complete":
    # Held pending until this process is told to finish, so the other instance has
    # something to poll while the job is still running.
    marker = Path(os.environ["JOBS_MARKER"])
    deadline = time.time() + 30
    while not marker.exists() and time.time() < deadline:
        time.sleep(0.05)
    stored = jobs.get(job.job_id)
    stored.complete({"name": "report.pdf"}, {"parse_ms": 12.5})
    jobs.update(stored)
    marker.with_suffix(".json").write_text(
        json.dumps(jobs.get(job.job_id).to_dict()), encoding="utf-8"
    )
"""

ACCEPTING_TIMEOUT_SECONDS = 30.0


def _accepting_instance(
    table: str, owner: str, marker, behaviour: str
) -> tuple[subprocess.Popen, str]:
    """A second instance of the service over the same database, and its job id."""
    process = subprocess.Popen(
        [sys.executable, "-c", _ACCEPTING_INSTANCE_SOURCE],
        cwd=str(Path(__file__).resolve().parent.parent),
        env={
            **os.environ,
            "DOC_ETL_API_TEST_POSTGRES_URL": DATABASE_URL,
            "JOBS_TABLE": table,
            "JOBS_OWNER": owner,
            "JOBS_MARKER": str(marker),
            "JOBS_BEHAVIOUR": behaviour,
        },
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )
    line = process.stdout.readline()
    if not line:
        process.wait(timeout=ACCEPTING_TIMEOUT_SECONDS)
        raise AssertionError(f"the accepting instance accepted nothing:\n{process.stderr.read()}")
    return process, json.loads(line)["job_id"]


def _poll_until(jobs: JobRegistry, job_id: str, status: JobStatus) -> Job:
    """The job, once it reports *status*, or a failure naming what it reported.

    Polled rather than waited for: the two instances only share the row, so
    "finished" is a thing this instance learns by asking, which is the whole of
    what a caller polling a load-balanced address does.
    """
    deadline = time.time() + ACCEPTING_TIMEOUT_SECONDS
    reported = jobs.get(job_id)
    while reported is not None and reported.status is not status and time.time() < deadline:
        time.sleep(0.05)
        reported = jobs.get(job_id)
    assert reported is not None, f"job {job_id} disappeared while it was being polled"
    assert reported.status is status, f"job {job_id} reported {reported.status.value}"
    return reported


@needs_database
def test_a_job_is_readable_from_an_instance_that_did_not_accept_it(collection, tmp_path):
    """The whole point of sharing the record: polled elsewhere, pending then done."""
    polling = _postgres_registry(_settings(table=collection), owner="poller:2")
    marker = tmp_path / "finish"
    process, job_id = _accepting_instance(
        collection, owner="acceptor:1", marker=marker, behaviour="complete"
    )
    try:
        assert polling.get(job_id).status is JobStatus.PENDING

        marker.write_text("finish", encoding="utf-8")
        completed = _poll_until(polling, job_id, JobStatus.COMPLETED)

        # What the accepting instance said it wrote, read from the other one: the
        # result and the duration have to be the same values, not merely present.
        process.wait(timeout=ACCEPTING_TIMEOUT_SECONDS)
        assert process.returncode == 0, process.stderr.read()
        written = json.loads(marker.with_suffix(".json").read_text(encoding="utf-8"))
        assert completed.to_dict() == written
    finally:
        process.kill()
        process.wait()
        _drop_collection(collection)


@needs_database
def test_a_job_left_pending_by_a_stopped_instance_is_reported_failed(collection, tmp_path):
    """An instance that really stopped, rather than one a test declared stopped."""
    polling = JobRegistry(
        store=_postgres_job_store(_settings(table=collection)),
        owner="poller:2",
        orphan_threshold_seconds=SHORT_THRESHOLD_SECONDS,
    )
    process, job_id = _accepting_instance(
        collection, owner="acceptor:1", marker=tmp_path / "unused", behaviour="exit"
    )
    try:
        # The accepting instance is gone the moment it accepted the job, and the
        # row it left is the only thing that says the job ever existed.
        process.wait(timeout=ACCEPTING_TIMEOUT_SECONDS)
        assert process.returncode == 0, process.stderr.read()

        time.sleep(SHORT_THRESHOLD_SECONDS * 2)
        reported = polling.get(job_id)

        assert reported.status is JobStatus.FAILED
        assert "acceptor:1" in reported.error
    finally:
        process.kill()
        process.wait()
        _drop_collection(collection)
