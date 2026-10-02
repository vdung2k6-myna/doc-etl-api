"""Which instance is loading a corpus source, so that only one of them does.

The startup bootstrap runs on every instance, against the same configured
corpus and, when the backend is durable, against the same index. Without an
agreement between them, several instances starting together parse, chunk and
embed the same corpus once each -- the work the corpus comparison inside the
pipeline cannot save, because it is a comparison each instance makes against
what it has already seen rather than against what the others are doing.

So a claim is taken before the work: a row keyed by the source's address,
naming the instance that took it and when. The claim is the same shape as a job
record, and for the same reason -- the answer has to be readable from a process
that never saw the claim being taken, and it has to outlive a dropped connection
honestly, which an advisory lock scoped to a session does not.

Ownership is what keeps a stopped instance from stranding a source. The row
records when the claim was taken; a claim taken longer ago than the ownership
threshold `job_orphan_threshold_seconds` configures is one whose owner is gone,
and it is taken over rather than waited on forever. The same threshold, because
it answers the same question -- how long may an instance be silent before it is
treated as stopped -- and a deployment that tunes one and not the other would
have two answers to it.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Protocol

import sqlalchemy
from sqlalchemy.dialects import postgresql
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Claim:
    """One instance's hold on one corpus source, as the store remembers it."""

    address: str
    owner: str
    claimed_at: float

    def age(self, now: float) -> float:
        """How long ago this claim was taken, which is what staleness is read from."""
        return now - self.claimed_at


class ClaimStore(Protocol):
    """Where the question "is anyone loading this source" is asked and answered."""

    def claim(self, address: str, owner: str, stale_after: float) -> bool: ...

    def release(self, address: str, owner: str) -> None: ...

    def held(self, address: str) -> Claim | None: ...

    def close(self) -> None: ...


class InProcessClaimStore:
    """The claims of one process, which is all the in-memory backend can have.

    Nothing shares this store with another instance, so nothing conflicts with
    it: the first claim on an address is always granted and every source is
    ingested exactly as it was before claims existed. That is the honest answer
    rather than a missing one -- with the in-memory backend there is no second
    instance to defer to, and refusing here would strand a source in the one
    deployment where no one else can pick it up.

    A dict rather than nothing at all, because the claim is also read while it
    is held: the deferring path and the readiness report both ask who holds a
    source, and both are exercised against this backend too.
    """

    def __init__(self) -> None:
        self._held: dict[str, Claim] = {}

    def claim(self, address: str, owner: str, stale_after: float) -> bool:
        if address in self._held:
            return False
        self._held[address] = Claim(address=address, owner=owner, claimed_at=time.time())
        return True

    def release(self, address: str, owner: str) -> None:
        held = self._held.get(address)
        # Released only by the instance holding it, so a takeover that was
        # granted to a second instance is not undone by the first one finishing.
        if held is not None and held.owner == owner:
            del self._held[address]

    def held(self, address: str) -> Claim | None:
        return self._held.get(address)

    def close(self) -> None:
        """Nothing to hand back: these claims go when the process does."""


CLAIMS_TABLE_SUFFIX = "_claims"


def claims_table(base: str, metadata: sqlalchemy.MetaData) -> sqlalchemy.Table:
    """The table a claim on a corpus source lives in.

    Beside the job records and under the same base name, so everything the
    service keeps in one database shares one identifier.

    The address is the key, and there is one row per claimed source: the claim
    answers "who is loading this" for one address at a time, so two rows for one
    address would be two answers. There is no index beyond that key because
    nothing queries this table by anything else -- it is read and written one
    address at a time, and the row for an address is gone once its ingestion is.

    No foreign key to the catalog: a claim is taken on a source that is, by
    definition, not in the catalog yet.
    """
    name = f"{base}{CLAIMS_TABLE_SUFFIX}"
    return sqlalchemy.Table(
        name,
        metadata,
        sqlalchemy.Column("address", sqlalchemy.Text, primary_key=True),
        sqlalchemy.Column("owner", sqlalchemy.Text, nullable=False),
        sqlalchemy.Column("claimed_at", sqlalchemy.Double, nullable=False),
    )


class PostgresClaimStore:
    """The durable backing: one row per claimed source, in the index's database.

    It shares the index store's engine rather than opening its own, so a claim
    and the content the claimed ingestion produces reach the same database
    through the same pool, and a deployment closes one thing.

    The table is created by the index store at startup, with the rest of the
    schema; a store built here is a way to read and write it, not a second place
    it is defined.
    """

    def __init__(self, engine: Engine, base: str) -> None:
        self._engine = engine
        self._table = claims_table(base, sqlalchemy.MetaData())

    def claim(self, address: str, owner: str, stale_after: float) -> bool:
        """Take the claim, or report that someone else holds a live one.

        One statement, because the alternative is a read followed by a write
        with a gap between them that two instances starting together would both
        find empty. The insert conflicts on the address, and the conflict is
        resolved as an update only when the row it found is one this instance
        may take: its own, which makes re-claiming after a release harmless, or
        one older than the ownership threshold, which is a stopped instance's.
        When the condition on the update does not hold, nothing is written and
        nothing is returned -- so the returned row is the claim, and its absence
        is the refusal. Both halves arrive from the same row lock, which is what
        makes several instances starting together settle on one claimant rather
        than on several.
        """
        now = time.time()
        with self._engine.begin() as connection:
            row = connection.execute(
                postgresql.insert(self._table)
                .values(address=address, owner=owner, claimed_at=now)
                .on_conflict_do_update(
                    index_elements=[self._table.c.address],
                    set_={"owner": owner, "claimed_at": now},
                    where=sqlalchemy.or_(
                        self._table.c.owner == owner,
                        self._table.c.claimed_at < now - stale_after,
                    ),
                )
                .returning(self._table.c.owner)
            ).first()
        return row is not None

    def release(self, address: str, owner: str) -> None:
        """Give up a claim, but only one this instance holds.

        The owner is part of the condition rather than checked beforehand, so an
        instance that lost its claim to a takeover -- and finished its ingestion
        afterwards -- does not delete the successor's claim on the strength of
        the one it no longer has.
        """
        with self._engine.begin() as connection:
            connection.execute(
                self._table.delete().where(
                    sqlalchemy.and_(
                        self._table.c.address == address,
                        self._table.c.owner == owner,
                    )
                )
            )

    def held(self, address: str) -> Claim | None:
        with self._engine.connect() as connection:
            row = connection.execute(
                sqlalchemy.select(self._table).where(self._table.c.address == address)
            ).first()
        if row is None:
            return None
        return Claim(address=row.address, owner=row.owner, claimed_at=row.claimed_at)

    def close(self) -> None:
        """Nothing to hand back: the engine belongs to the index store."""
