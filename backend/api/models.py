"""Pydantic request/response models for pipeline control API."""

from pydantic import BaseModel, field_validator


# -- Requests --


class AddChannelRequest(BaseModel):
    source: str


class RemoveChannelRequest(BaseModel):
    channel_id: int


class EntryExitLine(BaseModel):
    label: str
    start: tuple[float, float]
    end: tuple[float, float]
    junction_side: str = "left"  # "left" or "right" of start→end vector


class SetPhaseRequest(BaseModel):
    phase: str
    roi_polygon: list[tuple[float, float]] | None = None
    entry_exit_lines: dict[str, EntryExitLine] | None = None


class UpdateConfigRequest(BaseModel):
    confidence_threshold: float | None = None


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


class ResolvingResponse(BaseModel):
    status: str  # "resolving"
    request_id: str


class PhaseResponse(BaseModel):
    status: str
    phase: str
