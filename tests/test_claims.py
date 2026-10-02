"""Which instance is loading a corpus source, and how that answer stays current.

The claim's own behaviour is tested here against the in-memory backing, which is
the one every machine has: that a refresh moves a claim only for the instance
holding it, and that the heartbeat keeps a slow source from being mistaken for one
whose instance has stopped. The durable backing is tested at the end of the file
against a real database, because what it has to establish -- that a refresh from
an instance which lost the claim writes nothing -- is a property of the row, and
an in-process dict cannot be asked about it.
"""

import inspect
import itertools
import time
from collections.abc import Iterator

import pytest

from doc_etl_api.claims import (
    ClaimStore,
    InProcessClaimStore,
    claim_heartbeat,
)
from doc_etl_api.config import Settings
from doc_etl_api.jobs import HEARTBEAT_INTERVAL_SECONDS
from doc_etl_api.pipeline import IndexPipeline
from doc_etl_api.store import PostgresIndexStore
from tests.test_postgres_store import (
    _create,
    _drop_collection,
    _settings,
    needs_database,
)


def _held(claims: ClaimStore, address: str) -> float:
    """When the claim on *address* was taken, which is what a refresh moves."""
    claim = claims.held(address)
    assert claim is not None, f"nothing holds {address}"
    return claim.claimed_at


def test_a_refresh_by_the_holder_moves_the_claim():
    """The holder saying it is still working is what keeps its claim live."""
    claims = InProcessClaimStore()
    assert claims.claim("alpha.txt", "instance-1", 30.0)
    taken = _held(claims, "alpha.txt")

    time.sleep(0.01)

    assert claims.refresh("alpha.txt", "instance-1")
    assert _held(claims, "alpha.txt") > taken


def test_a_refresh_by_another_instance_moves_nothing():
    """A beat from an instance that no longer holds the claim must not reclaim it."""
    claims = InProcessClaimStore()
    assert claims.claim("alpha.txt", "instance-1", 30.0)
    claims.release("alpha.txt", "instance-1")
    assert claims.claim("alpha.txt", "instance-2", 30.0)
    taken = _held(claims, "alpha.txt")

    time.sleep(0.01)

    assert not claims.refresh("alpha.txt", "instance-1")
    assert _held(claims, "alpha.txt") == taken
    assert claims.held("alpha.txt").owner == "instance-2"


def test_a_refresh_with_no_claim_at_all_moves_nothing():
    claims = InProcessClaimStore()

    assert not claims.refresh("alpha.txt", "instance-1")
    assert claims.held("alpha.txt") is None


# --- The heartbeat -------------------------------------------------------------


def test_the_heartbeat_moves_a_held_claim_while_the_work_runs():
    """Work that outlasts the threshold still says it is running while it runs."""
    claims = InProcessClaimStore()
    assert claims.claim("alpha.txt", "instance-1", 30.0)
    before = _held(claims, "alpha.txt")

    with claim_heartbeat(claims, "alpha.txt", "instance-1", interval=0.02):
        time.sleep(0.1)

    assert _held(claims, "alpha.txt") > before


def test_the_heartbeat_stops_beating_when_the_work_ends():
    """The thread is joined on exit, so no beat lands after the claim is released."""
    claims = InProcessClaimStore()
    assert claims.claim("alpha.txt", "instance-1", 30.0)

    with claim_heartbeat(claims, "alpha.txt", "instance-1", interval=0.02):
        pass

    finished_at = _held(claims, "alpha.txt")
    time.sleep(0.1)

    assert _held(claims, "alpha.txt") == finished_at


def test_the_claim_beat_reuses_the_job_interval():
    """One number, because both beats answer how long an instance may stay silent."""
    default = inspect.signature(claim_heartbeat).parameters["interval"].default

    assert default is HEARTBEAT_INTERVAL_SECONDS


# --- The durable backing, against a real database ------------------------------


def _claim_pipeline(settings: Settings) -> IndexPipeline:
    """A durable pipeline, whose store is where a claim lives."""
    pipeline = _create(settings)
    assert isinstance(pipeline.store, PostgresIndexStore), "the test database is not in use"
    return pipeline


_CLAIM_TABLES = itertools.count(1)


@pytest.fixture
def collection() -> Iterator[str]:
    """A collection no other test uses, dropped before and after it runs.

    The prefix is this module's own rather than the store suite's fixture reused:
    a fixture defined in another test module is not visible here, and the two
    counters would name the same tables if they could be.
    """
    name = f"doc_etl_api_test_claims_{next(_CLAIM_TABLES)}"
    _drop_collection(name)
    yield name
    _drop_collection(name)


@needs_database
def test_a_refresh_by_the_holder_moves_the_row(collection):
    """The beat is only a liveness signal if it reaches the row, not just memory."""
    pipeline = _claim_pipeline(_settings(table=collection))
    try:
        claims = pipeline.store.claim_store()
        assert claims.claim("alpha.txt", "instance-1", 30.0)
        taken = _held(claims, "alpha.txt")

        time.sleep(0.01)

        assert claims.refresh("alpha.txt", "instance-1")
        assert _held(claims, "alpha.txt") > taken
    finally:
        pipeline._store.close()


@needs_database
def test_a_refresh_from_a_superseded_owner_writes_no_row(collection):
    """A beat after a takeover must not take the claim back from its new holder."""
    pipeline = _claim_pipeline(_settings(table=collection))
    try:
        claims = pipeline.store.claim_store()
        # Old enough that the next instance may take it over, which is what a
        # stopped owner leaves behind.
        assert claims.claim("alpha.txt", "instance-1", 1.0)
        time.sleep(1.1)
        assert claims.claim("alpha.txt", "instance-2", 1.0)
        taken = _held(claims, "alpha.txt")

        time.sleep(0.01)

        assert not claims.refresh("alpha.txt", "instance-1")
        held = claims.held("alpha.txt")
        assert held.owner == "instance-2"
        assert held.claimed_at == taken
    finally:
        pipeline._store.close()
