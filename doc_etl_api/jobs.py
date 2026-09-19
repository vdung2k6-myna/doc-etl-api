from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


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

    def to_dict(self) -> dict[str, Any]:
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


class JobRegistry:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}

    def create(self, source_id: str) -> Job:
        job_id = str(uuid.uuid4())
        job = Job(job_id=job_id, source_id=source_id)
        self._jobs[job_id] = job
        return job

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    @property
    def in_flight(self) -> int:
        """Jobs that have been created but not yet completed or failed."""
        return sum(1 for job in self._jobs.values() if job.status is JobStatus.PENDING)

    def update(self, job: Job) -> None:
        self._jobs[job.job_id] = job
