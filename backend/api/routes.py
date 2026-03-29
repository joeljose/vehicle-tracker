"""Pipeline control REST endpoints."""

from fastapi import APIRouter, Depends, HTTPException, Request

from backend.api.deps import get_alert_store, get_backend
from backend.api.models import (
    AddChannelRequest,
    ChannelAddedResponse,
    PhaseResponse,
    RemoveChannelRequest,
    SetPhaseRequest,
    StatusResponse,
    UpdateConfigRequest,
)
from backend.pipeline.alerts import AlertStore
from backend.pipeline.protocol import ChannelPhase, PipelineBackend

router = APIRouter()


@router.post("/pipeline/start", response_model=StatusResponse)
def start_pipeline(
    request: Request,
    backend: PipelineBackend = Depends(get_backend),
):
    try:
        backend.start()
    except RuntimeError:
        raise HTTPException(status_code=409, detail="Pipeline already started")
    request.app.state.pipeline_started = True
    return StatusResponse(status="started")


@router.post("/pipeline/stop", response_model=StatusResponse)
def stop_pipeline(
    request: Request,
    backend: PipelineBackend = Depends(get_backend),
    alert_store: AlertStore = Depends(get_alert_store),
):
    try:
        backend.stop()
    except RuntimeError:
        raise HTTPException(status_code=409, detail="Pipeline not started")
    request.app.state.pipeline_started = False
    request.app.state.next_channel_id = 0
    alert_store.clear()
    return StatusResponse(status="stopped")


@router.post("/channel/add", response_model=ChannelAddedResponse)
def add_channel(
    body: AddChannelRequest,
    request: Request,
    backend: PipelineBackend = Depends(get_backend),
):
    if not request.app.state.pipeline_started:
        raise HTTPException(status_code=409, detail="Pipeline not started")
    channel_id = request.app.state.next_channel_id
    request.app.state.next_channel_id += 1
    backend.add_channel(channel_id, body.source)
    return ChannelAddedResponse(channel_id=channel_id)


@router.post("/channel/remove", response_model=StatusResponse)
def remove_channel(
    body: RemoveChannelRequest,
    backend: PipelineBackend = Depends(get_backend),
    alert_store: AlertStore = Depends(get_alert_store),
):
    try:
        backend.remove_channel(body.channel_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Channel not found")
    alert_store.clear_channel(body.channel_id)
    return StatusResponse(status="removed")


@router.post("/channel/{channel_id}/phase", response_model=PhaseResponse)
def set_channel_phase(
    channel_id: int,
    body: SetPhaseRequest,
    backend: PipelineBackend = Depends(get_backend),
):
    try:
        phase = ChannelPhase(body.phase)
    except ValueError:
        valid = [p.value for p in ChannelPhase]
        raise HTTPException(
            status_code=422, detail=f"Invalid phase. Must be one of: {valid}"
        )
    try:
        backend.set_channel_phase(channel_id, phase)
    except KeyError:
        raise HTTPException(status_code=404, detail="Channel not found")
    return PhaseResponse(status="ok", phase=phase.value)


@router.patch("/config", response_model=StatusResponse)
def update_config(
    body: UpdateConfigRequest,
    backend: PipelineBackend = Depends(get_backend),
):
    if body.confidence_threshold is not None:
        backend.set_confidence_threshold(body.confidence_threshold)
    if body.inference_interval is not None:
        backend.set_inference_interval(body.inference_interval)
    return StatusResponse(status="updated")
