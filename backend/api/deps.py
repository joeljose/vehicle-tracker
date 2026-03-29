"""FastAPI dependency injection helpers."""

from fastapi import Request

from backend.pipeline.alerts import AlertStore
from backend.pipeline.clip_extractor import ClipExtractor
from backend.pipeline.protocol import PipelineBackend


def get_backend(request: Request) -> PipelineBackend:
    return request.app.state.backend


def get_alert_store(request: Request) -> AlertStore:
    return request.app.state.alert_store


def get_clip_extractor(request: Request) -> ClipExtractor:
    return request.app.state.clip_extractor
