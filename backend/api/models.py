"""Pydantic request/response models for pipeline control API."""

from pydantic import BaseModel, field_validator


# -- Requests --

class AddChannelRequest(BaseModel):
    source: str


class RemoveChannelRequest(BaseModel):
    channel_id: int


class SetPhaseRequest(BaseModel):
    phase: str


class UpdateConfigRequest(BaseModel):
    confidence_threshold: float | None = None
    inference_interval: int | None = None


class EntryExitLine(BaseModel):
    label: str
    start: tuple[float, float]
    end: tuple[float, float]


class SiteConfigRequest(BaseModel):
    site_id: str
    roi_polygon: list[tuple[float, float]]
    entry_exit_lines: dict[str, EntryExitLine] = {}

    @field_validator("roi_polygon")
    @classmethod
    def polygon_min_vertices(cls, v):
        if len(v) < 3:
            raise ValueError("ROI polygon must have at least 3 vertices")
        return v


# -- Responses --

class StatusResponse(BaseModel):
    status: str


class ChannelAddedResponse(BaseModel):
    channel_id: int


class PhaseResponse(BaseModel):
    status: str
    phase: str
