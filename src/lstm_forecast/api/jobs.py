"""In-process async job queue for the API.

This is a deliberately small, dependency-free job runner: a single
:class:`concurrent.futures.ThreadPoolExecutor` plus an in-memory dictionary of
job records guarded by a lock. It exists so the API can accept long-running
forecast requests without blocking the request thread.

.. note::

   This runner is **in-process and single-worker by design**. Jobs live only in
   the memory of one server process, so they are lost on restart and are not
   shared across replicas. A production deployment would back this with a durable
   broker/worker system such as Redis + Celery (or RQ / Dramatiq). The public
   surface (:meth:`JobManager.submit` / :meth:`JobManager.get` /
   :meth:`JobManager.list`) is intentionally tiny so it can be swapped for such a
   backend without touching the routes.
"""

from __future__ import annotations

import threading
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

# Job lifecycle states.
STATUS_QUEUED = "queued"
STATUS_RUNNING = "running"
STATUS_DONE = "done"
STATUS_ERROR = "error"


@dataclass
class Job:
    """A single unit of background work and its outcome.

    Attributes
    ----------
    id:
        Opaque ``uuid4`` hex identifier.
    status:
        One of ``queued``, ``running``, ``done`` or ``error``.
    result:
        The function's return value once ``status == 'done'``, else ``None``.
    error:
        A string description of the failure once ``status == 'error'``, else ``None``.
    """

    id: str
    status: str = STATUS_QUEUED
    result: dict[str, Any] | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a plain-dict view suitable for serialisation."""
        return {
            "id": self.id,
            "status": self.status,
            "result": self.result,
            "error": self.error,
        }


@dataclass
class JobManager:
    """Thread-safe, in-process manager for background jobs.

    Parameters
    ----------
    max_workers:
        Size of the backing thread pool. Defaults to ``2`` which is plenty for a
        single-process service; raise it only if forecasts are I/O bound.
    """

    max_workers: int = 2
    _jobs: dict[str, Job] = field(default_factory=dict, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _executor: ThreadPoolExecutor = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._executor = ThreadPoolExecutor(max_workers=self.max_workers)

    def submit(self, fn: Callable[..., dict[str, Any]], *args: Any, **kwargs: Any) -> str:
        """Schedule ``fn(*args, **kwargs)`` to run in the background.

        Parameters
        ----------
        fn:
            Callable returning a JSON-serialisable ``dict`` (stored as the result).
        *args, **kwargs:
            Arguments forwarded to ``fn``.

        Returns
        -------
        str
            The new job id; poll :meth:`get` to observe progress.
        """
        job_id = uuid.uuid4().hex
        with self._lock:
            self._jobs[job_id] = Job(id=job_id)
        self._executor.submit(self._run, job_id, fn, args, kwargs)
        return job_id

    def _run(
        self,
        job_id: str,
        fn: Callable[..., dict[str, Any]],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        """Execute a job, recording its result or error under the lock."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is not None:
                job.status = STATUS_RUNNING
        try:
            result = fn(*args, **kwargs)
        except Exception as exc:
            with self._lock:
                job = self._jobs.get(job_id)
                if job is not None:
                    job.status = STATUS_ERROR
                    job.error = f"{type(exc).__name__}: {exc}"
            return
        with self._lock:
            job = self._jobs.get(job_id)
            if job is not None:
                job.status = STATUS_DONE
                job.result = result

    def get(self, job_id: str) -> Job | None:
        """Return the :class:`Job` for ``job_id`` or ``None`` if unknown."""
        with self._lock:
            return self._jobs.get(job_id)

    def list(self) -> list[Job]:
        """Return a snapshot list of all known jobs."""
        with self._lock:
            return list(self._jobs.values())

    def shutdown(self, *, wait: bool = False) -> None:
        """Shut down the backing thread pool (mainly for tests)."""
        self._executor.shutdown(wait=wait)


# Module-level singleton used by the API routes. In-process only; see module docstring.
_manager: JobManager | None = None
_manager_lock = threading.Lock()


def get_job_manager() -> JobManager:
    """Return the process-wide :class:`JobManager`, creating it on first use."""
    global _manager
    with _manager_lock:
        if _manager is None:
            _manager = JobManager()
        return _manager
