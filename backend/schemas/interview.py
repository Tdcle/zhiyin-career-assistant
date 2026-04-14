"""Interview API schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field


class StartInterviewRequest(BaseModel):
    job_id: str = Field(..., min_length=1, description="Target job id")
