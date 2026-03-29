"""FastAPI dependency injection helpers."""

from fastapi import Request

from backend.pipeline.alerts import AlertStore
from backend.pipeline.protocol import PipelineBackend


def get_backend(request: Request) -> PipelineBackend:
    return request.app.state.backend


def get_alert_store(request: Request) -> AlertStore:
    return request.app.state.alert_store
