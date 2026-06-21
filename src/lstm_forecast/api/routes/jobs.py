"""Asynchronous (background) forecast job endpoints.

Forecast training can be slow; these endpoints let a client submit a forecast as
a background job and poll for the result instead of holding a request open. Jobs
are managed by an in-process :class:`~lstm_forecast.api.jobs.JobManager` (see that
module for the single-worker / production caveats).
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from lstm_forecast.api import service
from lstm_forecast.api.jobs import (
    STATUS_DONE,
    STATUS_ERROR,
    get_job_manager,
)
from lstm_forecast.api.schemas import (
    ForecastRequest,
    ForecastResponse,
    JobStatusResponse,
    JobSubmitResponse,
)

router = APIRouter(prefix="/jobs", tags=["jobs"])


def _forecast_job(req: ForecastRequest) -> dict[str, Any]:
    """Run a forecast and return a serialisable :class:`ForecastResponse` dict.

    Executed inside the job worker thread, so it must return plain data (not a
    pydantic model instance) for storage in the job record.
    """
    _, result = service.run_forecast(req)
    response = service.to_response(req, result, insights=None)
    return response.model_dump()


@router.post("/forecast", response_model=JobSubmitResponse)
def submit_forecast(req: ForecastRequest) -> JobSubmitResponse:
    """Submit a forecast to run in the background and return its job id."""
    manager = get_job_manager()
    job_id = manager.submit(_forecast_job, req)
    return JobSubmitResponse(job_id=job_id)


@router.get("/{job_id}", response_model=JobStatusResponse)
def job_status(job_id: str) -> JobStatusResponse:
    """Return the status of a job, plus its forecast result or error when finished."""
    job = get_job_manager().get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Unknown job id: {job_id}")
    result = (
        ForecastResponse(**job.result)
        if job.status == STATUS_DONE and job.result is not None
        else None
    )
    error = job.error if job.status == STATUS_ERROR else None
    return JobStatusResponse(job_id=job.id, status=job.status, result=result, error=error)
