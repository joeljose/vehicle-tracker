"""Pydantic request/response models for pipeline control API."""

from pydantic import BaseModel


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


# -- Responses --

class StatusResponse(BaseModel):
    status: str


class ChannelAddedResponse(BaseModel):
    channel_id: int


class PhaseResponse(BaseModel):
    status: str
    phase: str
